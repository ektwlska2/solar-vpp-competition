"""EXPERIMENT D (test plan): hourly 0.25-degree NOAA GFS surface downward shortwave radiation (DSWRF), 00 UTC cycle of the day BEFORE the target day (same eligible run as 13_gfs_national.py; online ~03:40 UTC = 12:40 KST on the
AWS bucket noaa-gfs-bdp-pds, checked on sample dates 2021-03, 2022-06, 2023-03, 2024-03, 2025-03, 2025-11).  Forecast hours 19..35 (valid 11:00 D-1 ... 24:00 KST of the target day label range 5..20 needs fh = label + 15; fh 19 is only needed to de-accumulate fh 20).
DSWRF in the hourly files is a bucket average since the last multiple of 6 h ('12-18 hour ave'); the hourly mean of one hour is recovered from two consecutive files in the loader (gfs_features-style code in rev/g25 loader).
Only the fixed box 33-39 N, 125-130 E is kept (cell values, 0.25 degree, the same cells for every plant); aggregation to national / regional means happens in the loader.
    python 13d_gfs025_dsw.py --start 2021-03-01 --end 2026-06-01 [--workers 16]      # writes <CACHE>/gfs025_dsw_raw.parquet (+ _geo.json)"""
from __future__ import annotations
import argparse, json, os, time
import concurrent.futures as cf
import datetime as dt
import numpy as np
import pandas as pd
import eccodes
from oibc_lib import *
g13 = __import__("13_gfs_national")
BASE = g13.BASE
BOX_LAT, BOX_LON = (33.0, 39.0), (125.0, 130.0)
FH = tuple(range(19, 36))
_GEO = {}


def url(init: dt.date, fh: int, ext: str = "") -> str:
    sub = "atmos/" if init >= dt.date(2021, 3, 23) else ""
    return f"{BASE}/gfs.{init:%Y%m%d}/00/{sub}gfs.t00z.pgrb2.0p25.f{fh:03d}{ext}"


def cells(msg: bytes) -> np.ndarray:
    gid = eccodes.codes_new_from_message(msg)
    try:
        vals = eccodes.codes_get_values(gid)
        if "i" not in _GEO:
            lat, lon = eccodes.codes_get_array(gid, "latitudes"), eccodes.codes_get_array(gid, "longitudes")
            m = (lat >= BOX_LAT[0]) & (lat <= BOX_LAT[1]) & (lon >= BOX_LON[0]) & (lon <= BOX_LON[1])
            idx = np.where(m)[0]
            order = np.lexsort((lon[idx], -lat[idx]))
            _GEO["i"] = idx[order]; _GEO["lat"] = lat[idx][order]; _GEO["lon"] = lon[idx][order]
        return vals[_GEO["i"]].astype("float32")
    finally:
        eccodes.codes_release(gid)


def one(task):
    day, fh = task
    init = day - dt.timedelta(days=1)
    row = dict(date=day, fh=fh, a=np.nan, b=np.nan, v=None)
    txt = g13.get(url(init, fh, ".idx"))
    if txt is None:
        return row
    idx = g13.parse_idx(txt.decode())
    m = g13.message_bytes(idx, None, "DSWRF", "surface", lambda d: "hour ave" in d)
    if not m:
        return row
    a_, b_ = m[2].split(" hour ave")[0].split("-")
    row["a"], row["b"] = float(a_), float(b_)
    for _ in range(3):
        b = g13.get(url(init, fh), (m[0], m[1]))
        if b:
            try:
                row["v"] = cells(b); break
            except Exception:
                pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-03-01"); ap.add_argument("--end", default="2026-06-01"); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=f"{CACHE}/gfs025_dsw_raw.parquet")
    args = ap.parse_args()
    days = [d.date() for d in pd.date_range(args.start, args.end, freq="D")]
    tasks = [(d, fh) for d in days for fh in FH]
    rows, done = [], set()
    if os.path.exists(args.out):
        old = pd.read_parquet(args.out); old = old[old.v.notna()]
        rows = [r._asdict() for r in old.itertuples(index=False)]; done = {(pd.Timestamp(r["date"]).date(), r["fh"]) for r in rows}
        tasks = [t for t in tasks if t not in done]
    print(f"{len(tasks):,} file tasks, {len(done):,} cached", flush=True)
    t0 = time.time()
    # warm the grid on the main thread
    first = one(tasks[0]) if tasks else None
    if first is not None and first["v"] is not None: rows.append(first); tasks = tasks[1:]
    def safe(t):
        try: return one(t)
        except Exception: return dict(date=t[0], fh=t[1], a=np.nan, b=np.nan, v=None)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(safe, t) for t in tasks]
        for i, f in enumerate(cf.as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 2000 == 0:
                print(f"  {i+1:,}/{len(tasks):,} in {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_parquet(args.out, index=False)
    df = pd.DataFrame(rows).sort_values(["date", "fh"]).reset_index(drop=True)
    df.to_parquet(args.out, index=False)
    json.dump(dict(lat=[float(x) for x in _GEO["lat"]], lon=[float(x) for x in _GEO["lon"]]), open(args.out.replace(".parquet", "_geo.json"), "w"))
    print(f"saved {args.out}: {len(df):,} rows, missing {df.v.isna().mean():.3%}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
