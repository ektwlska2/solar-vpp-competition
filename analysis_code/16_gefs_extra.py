"""EXTERNAL SOURCE (rules: 'external source notice', section 5): more broad-area summaries of the NOAA GEFS ensemble mean, the same value for every plant.

    python 16_gefs_extra.py --start 2021-01-01 --end 2026-06-01 [--workers 48]      # writes <CACHE>/gefs_extra_raw.parquet

Same bucket, cycle and box as 14_gefs_national.py (ensemble mean 'geavg', 0.5 degree, 00 UTC cycle of the day before the target day, box 33-39 N, 125-130 E, one value for every plant).
Fields at forecast hours 27 and 30 (12:00 and 15:00 KST of the target day): precipitable water, 2 m relative humidity, 2 m temperature (instantaneous) and the 3-hour precipitation of the
two windows 24-27 and 27-30 h (de-accumulated from the 6-hour buckets).
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
get, parse_idx = g13.get, g13.parse_idx
FHOURS = (27, 30)
WANT = {"pwat": ("PWAT", "entire atmosphere", lambda d, fh: d.strip() == f"{fh} hour fcst"),
        "rh2m": ("RH", "2 m above ground", lambda d, fh: d.strip() == f"{fh} hour fcst"),
        "t2m": ("TMP", "2 m above ground", lambda d, fh: d.strip() == f"{fh} hour fcst"),
        "apcp": ("APCP", "surface", lambda d, fh: "hour acc" in d)}


def one(task):
    day, fh = task
    init = day - dt.timedelta(days=1)
    out = dict(date=day, fh=fh, a=np.nan, b=np.nan, **{k: np.nan for k in WANT}, apcp_sp=np.nan)
    for kind in ("geavg", "gespr"):
        u = g14.url(init, kind, fh)
        txt = get(u + ".idx")
        if txt is None:
            continue
        idx = parse_idx(txt.decode())
        for k, (var, lv, pred) in WANT.items():
            if kind == "gespr" and k != "apcp":
                continue
            for i, (off, v, l, d) in enumerate(idx):
                if v == var and l.startswith(lv) and pred(d, fh):
                    end = idx[i + 1][0] - 1 if i + 1 < len(idx) else off + 1_000_000
                    for _ in range(3):
                        blob = get(u, (off, end))
                        if blob:
                            try:
                                val = g14.stats(blob)[0]
                                if kind == "geavg":
                                    out[k] = val
                                    if k == "apcp":
                                        a_, b_ = d.split(" hour acc")[0].split("-")
                                        out["a"], out["b"] = float(a_), float(b_)
                                else:
                                    out["apcp_sp"] = val
                                break
                            except Exception:
                                pass
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=f"{CACHE}/gefs_extra_raw.parquet")
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
            return dict(date=t[0], fh=t[1], a=np.nan, b=np.nan, **{k: np.nan for k in WANT}, apcp_sp=np.nan)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(safe, t) for t in tasks]
        for i, f in enumerate(cf.as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 1000 == 0:
                print(f"  {i+1:,}/{len(tasks):,} in {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_parquet(args.out, index=False)
    df = pd.DataFrame(rows).sort_values(["date", "fh"]).reset_index(drop=True)
    df.to_parquet(args.out, index=False)
    print(f"saved {args.out}: {len(df):,} rows, missing " + ", ".join(f"{k} {df[k].isna().mean():.2%}" for k in list(WANT) + ["apcp_sp"]) + f", {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
