"""EXTERNAL SOURCE (rules: 'external source notice', section 5): a broad-area summary of the ECMWF IFS open-data forecast, the same value for every plant.

    python 17_ecmwf_national.py --start 2024-03-01 --end 2026-06-01 [--workers 12] [--cycle 12|18]      # writes <CACHE>/ecmwf_national_raw.parquet (cycle 18: ecmwf18_national_raw.parquet)

ECMWF IFS (HRES) open data on the public AWS Open Data bucket ecmwf-forecasts (CC-BY-4.0, free for commercial use, archived at fixed addresses from 2024-02-28):
  https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/YYYYMMDD/12z/ifs/0p25/oper/YYYYMMDD120000-{step}h-oper-fc.grib2   (+ .index with byte ranges)
  * surface solar radiation downwards (ssrd, accumulated since the start of the forecast), forecast steps 33, 36, ..., 48 h
  * the 12 UTC cycle of TWO days before the target day (init 21:00 KST of D-2, online about 05:00 KST of D-1: published before the 17:00 KST cut-off);
    steps 33-48 h are 06:00-21:00 KST of the target day
  * ONE box (33-39 N, 125-130 E) averaged to a single national value per hour and attached to every plant identically.
Output rows: target date, forecast step, box mean and spatial std of the accumulated ssrd (J/m2).
"""
from __future__ import annotations
import argparse, importlib, json, os, time
import concurrent.futures as cf
import datetime as dt
import numpy as np
import pandas as pd
from oibc_lib import *

g13 = importlib.import_module("13_gfs_national")
g14 = importlib.import_module("14_gefs_national")
import random, urllib.request, urllib.error


def get(u: str, rng: tuple | None = None, tries: int = 14):
    """The ECMWF bucket answers 503 SlowDown under load: retry with jittered back-off (404 = file does not exist)."""
    for k in range(tries):
        try:
            req = urllib.request.Request(u)
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(random.uniform(0.5, 2.5) * min(k + 1, 6))
        except Exception:
            time.sleep(random.uniform(0.5, 2.5) * min(k + 1, 6))
    return None


BASE = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
CYCLES = {12: dict(path="12z/ifs/0p25/oper", stream="oper", hh="12", steps=(33, 36, 39, 42, 45, 48)),       # 12 UTC of two days before: 27 h before 00:00 KST of the target day
          18: dict(path="18z/ifs/0p25/scda", stream="scda", hh="18", steps=(27, 30, 33, 36, 39, 42)),
          0: dict(path="00z/ifs/0p25/oper", stream="oper", hh="00", steps=(21, 24, 27, 30, 33, 36))}          # 00 UTC of the day BEFORE the target day (init 09:00 KST of D-1, on AWS at 07:34 UTC = 16:34 KST, measured on 2025-11, 2026-03, 2026-04): 15 h before 00:00 KST       # 18 UTC of two days before (online about 10:30 KST of the day before): 21 h before 00:00 KST
CYCLE = 12


def url(init: dt.date, step: int, ext: str, stream: str | None = None) -> str:
    c = CYCLES[CYCLE]
    stream = stream or c["stream"]
    return f"{BASE}/{init:%Y%m%d}/{c['hh']}z/ifs/0p25/{stream}/{init:%Y%m%d}{c['hh']}0000-{step}h-{stream}-fc.{ext}"


def one(task):
    day, step = task
    init = day - dt.timedelta(days=(1 if CYCLE == 0 else 2))
    row = dict(date=day, step=step, ssrd=np.nan, ssrd_sd=np.nan)
    txt, stream = None, None
    for stream in ((CYCLES[CYCLE]["stream"], "oper") if CYCLE == 18 else (CYCLES[CYCLE]["stream"],)):       # the 06/18 UTC cycles moved from the 'scda' to the 'oper' stream in May 2026
        txt = get(url(init, step, "index", stream))
        if txt is not None:
            break
    if txt is None:
        return row
    for ln in txt.decode().strip().split("\n"):
        j = json.loads(ln)
        if j.get("param") == "ssrd" and j.get("levtype") == "sfc":
            off, ln_ = j["_offset"], j["_length"]
            for _ in range(3):
                blob = get(url(init, step, "grib2", stream), (off, off + ln_ - 1))
                if blob:
                    try:
                        row["ssrd"], row["ssrd_sd"] = g14.stats(blob)
                        return row
                    except Exception:
                        pass
            break
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-03-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cycle", type=int, choices=(0, 12, 18), default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global CYCLE
    CYCLE = args.cycle
    if args.out is None:
        args.out = {12: f"{CACHE}/ecmwf_national_raw.parquet", 18: f"{CACHE}/ecmwf18_national_raw.parquet", 0: f"{CACHE}/ecmwf0_national_raw.parquet"}[CYCLE]
    days = [d.date() for d in pd.date_range(args.start, args.end, freq="D")]
    tasks = [(d, s) for d in days for s in CYCLES[CYCLE]["steps"]]
    rows = []
    if os.path.exists(args.out):
        old = pd.read_parquet(args.out)
        old = old[old.ssrd.notna()]
        done = {(pd.Timestamp(r.date).date(), r.step) for r in old.itertuples(index=False)}
        rows = [r._asdict() for r in old.itertuples(index=False)]
        tasks = [t for t in tasks if t not in done]
    print(f"{len(tasks):,} message tasks, {len(rows):,} cached", flush=True)
    t0 = time.time()

    def safe(t):
        try:
            return one(t)
        except Exception:
            return dict(date=t[0], step=t[1], ssrd=np.nan, ssrd_sd=np.nan)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(safe, t) for t in tasks]
        for i, f in enumerate(cf.as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 500 == 0:
                print(f"  {i+1:,}/{len(tasks):,} in {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_parquet(args.out, index=False)
    df = pd.DataFrame(rows).sort_values(["date", "step"]).reset_index(drop=True)
    df.to_parquet(args.out, index=False)
    print(f"saved {args.out}: {len(df):,} rows, missing {df.ssrd.isna().mean():.2%}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
