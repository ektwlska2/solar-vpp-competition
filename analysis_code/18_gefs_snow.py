"""EXTERNAL SOURCE (rules: 'external source notice', section 5): national box mean of the NOAA GEFS ensemble-mean snow depth, the same value for every plant.

    python 18_gefs_snow.py --start 2021-01-01 --end 2026-06-01 [--workers 32]      # writes <CACHE>/gefs_snow_raw.parquet

Same bucket, cycle (00 UTC of the day before the target day) and box (33-39 N, 125-130 E) as 14_gefs_national.py; fields SNOD (snow depth, m) and WEASD (water equivalent of snow, kg/m2)
at forecast hour 27 (12:00 KST of the target day), 0.5 degree ensemble mean.  Snow lying on the ground (and on panels) is a winter error source of the generation forecasts.
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


def one(task):
    day = task
    init = day - dt.timedelta(days=1)
    row = dict(date=day, snod=np.nan, snod_sd=np.nan, weasd=np.nan)
    u = g14.url(init, "geavg", 27)
    txt = get(u + ".idx")
    if txt is None:
        return row
    idx = parse_idx(txt.decode())
    pos = {}
    for i, (off, v, lv, d) in enumerate(idx):
        if v in ("SNOD", "WEASD") and lv.startswith("surface") and d.strip() == "27 hour fcst":
            pos[v] = i
    if len(pos) < 2:
        return row
    for k, col in (("SNOD", "snod"), ("WEASD", "weasd")):
        i = pos[k]
        end = idx[i + 1][0] - 1 if i + 1 < len(idx) else idx[i][0] + 1_000_000
        for _ in range(3):
            b = get(u, (idx[i][0], end))
            if b:
                try:
                    m, s = g14.stats(b)
                    row[col] = m
                    if k == "SNOD":
                        row["snod_sd"] = s
                    break
                except Exception:
                    pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=f"{CACHE}/gefs_snow_raw.parquet")
    args = ap.parse_args()
    days = [d.date() for d in pd.date_range(args.start, args.end, freq="D")]
    rows = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(lambda d: (lambda: one(d))() if True else None, days)):
            rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i+1:,}/{len(days):,} in {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"saved {args.out}: {len(df):,} rows, missing {df.snod.isna().mean():.2%}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
