"""EXTERNAL SOURCE (rules: 'external source notice', section 5): a broad-area summary of the NOAA GEFS-Aerosols forecast, the same value for every plant.

    python 15_aerosol_national.py --start 2021-01-01 --end 2026-06-01 [--workers 32]      # writes <CACHE>/aerosol_national_raw.parquet

NOAA GEFS-Aerosols (aerosol-enabled GEFS run) on the public AWS Open Data bucket noaa-gefs-pds, 0.25-degree 2-D chemistry product
  https://noaa-gefs-pds.s3.amazonaws.com/gefs.YYYYMMDD/00/chem/pgrb2ap25/gefs.chem.t00z.a2d_0p25.fXXX.grib2   (US-government data, archived at fixed addresses since 2020)
Fields (instantaneous, box mean over 33-39 N, 125-130 E, ONE national value per day for every plant): aerosol optical depth at 550 nm (total and dust), surface PM2.5 (total and dust).
00 UTC cycle of the day BEFORE the target day only, forecast hours 27 and 30 (12:00 and 15:00 KST of the target day): published before the 17:00 KST cut-off.
Why: yellow dust and haze reduce the sunlight that reaches the panels, and none of the weather forecasts in the competition data models aerosols.
"""
from __future__ import annotations
import argparse, importlib, os, time
import concurrent.futures as cf
import datetime as dt
import numpy as np
import pandas as pd
from oibc_lib import *

g13 = importlib.import_module("13_gfs_national")
g14 = importlib.import_module("14_gefs_national")
get = g13.get
BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
FHOURS = (27, 30)
WANT = {"aod": ("AOTK", "aerosol=Total aerosol", "5.45e-07"), "aod_dust": ("AOTK", "aerosol=Dust dry", "5.45e-07"),
        "pm25": ("PMTF", "aerosol=Total aerosol", ""), "pm25_dust": ("PMTF", "aerosol=Dust dry", "")}


def parse_idx_full(txt: str):
    rows = []
    for ln in txt.strip().split("\n"):
        p = ln.split(":")
        if len(p) >= 6:
            rows.append((int(p[1]), p[3], p[4], ":".join(p[5:])))
    return rows


def one(task):
    day, fh = task
    init = day - dt.timedelta(days=1)
    u = f"{BASE}/gefs.{init:%Y%m%d}/00/chem/pgrb2ap25/gefs.chem.t00z.a2d_0p25.f{fh:03d}.grib2"
    out = dict(date=day, fh=fh, **{k: np.nan for k in WANT})
    txt = get(u + ".idx")
    if txt is None:
        return out
    idx = parse_idx_full(txt.decode())
    for k, (var, a, b) in WANT.items():
        for i, (off, v, lv, d) in enumerate(idx):
            if v == var and a.lower() in d.lower() and b in d:                  # older archive files spell 'Total Aerosol' / 'Dust Dry' with capitals
                end = idx[i + 1][0] - 1 if i + 1 < len(idx) else off + 3_000_000
                for _ in range(3):
                    blob = get(u, (off, end))
                    if blob:
                        try:
                            out[k] = g14.stats(blob)[0]
                            break
                        except Exception:
                            pass
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=f"{CACHE}/aerosol_national_raw.parquet")
    args = ap.parse_args()
    days = [d.date() for d in pd.date_range(args.start, args.end, freq="D")]
    tasks = [(d, fh) for d in days for fh in FHOURS]
    rows = []
    if os.path.exists(args.out):
        old = pd.read_parquet(args.out)
        old = old[old[list(WANT)].notna().all(axis=1)]
        done = {(pd.Timestamp(r.date).date(), r.fh) for r in old.itertuples(index=False)}
        rows = [r._asdict() for r in old.itertuples(index=False)]
        tasks = [t for t in tasks if t not in done]
    print(f"{len(tasks):,} file tasks, {len(rows):,} cached", flush=True)
    t0 = time.time()

    def safe(t):
        try:
            return one(t)
        except Exception:
            return dict(date=t[0], fh=t[1], **{k: np.nan for k in WANT})
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(safe, t) for t in tasks]
        for i, f in enumerate(cf.as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 500 == 0:
                print(f"  {i+1:,}/{len(tasks):,} in {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_parquet(args.out, index=False)
    df = pd.DataFrame(rows).sort_values(["date", "fh"]).reset_index(drop=True)
    df.to_parquet(args.out, index=False)
    print(f"saved {args.out}: {len(df):,} rows, missing " + ", ".join(f"{k} {df[k].isna().mean():.2%}" for k in WANT) + f", {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
