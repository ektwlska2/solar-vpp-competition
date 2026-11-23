"""EXTERNAL SOURCE (rules: 'external source notice', section 5): a broad-area summary of the NOAA GFS day-ahead forecast, the same value for every plant.

    python 13_gfs_national.py --start 2021-01-01 --end 2026-06-01 [--workers 24]      # writes <CACHE>/gfs_national_raw.parquet

What is used and why it is allowed
  * NOAA GFS 1-degree forecasts on the public AWS Open Data bucket noaa-gfs-bdp-pds (US-government data, free for commercial use, archived at fixed addresses since 2021-01,
    re-downloadable by anyone: https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.YYYYMMDD/00/atmos/gfs.t00z.pgrb2.1p00.fXXX).
  * Only the 00 UTC cycle of the day BEFORE the target day (09:00 KST, files online about 13:00 KST), i.e. published before the 17:00 KST cut-off.
  * Only two fields, surface downward shortwave radiation (3-hour averages) and total cloud cover, averaged over ONE box (33-39 N, 125-130 E) that covers the Korean peninsula's
    south and the surrounding sea: a single national value per hour that is attached to every plant identically.  No plant coordinates, no nearest-grid selection.
  * No statistic of the evaluation year is used to normalise anything.
Output rows: target date, forecast hour, averaging window (a, b), radiation (W/m2), cloud cover (%).
"""
from __future__ import annotations
import argparse, io, os, sys, time, urllib.request, urllib.error
import concurrent.futures as cf
import datetime as dt
import numpy as np
import pandas as pd
import eccodes
from oibc_lib import *

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
BOX_LAT, BOX_LON = (33.0, 39.0), (125.0, 130.0)
FHOURS = (18, 21, 24, 27, 30, 33, 36, 39)                # valid 03:00 ... 24:00 KST of the target day for the 00 UTC cycle of the previous day (3-hourly files)
_MASK = {}


def url(init: dt.date, fh: int, ext: str = "") -> str:
    sub = "atmos/" if init >= dt.date(2021, 3, 23) else ""      # folder layout changed with GFS v16
    return f"{BASE}/gfs.{init:%Y%m%d}/00/{sub}gfs.t00z.pgrb2.1p00.f{fh:03d}{ext}"


def get(u: str, rng: tuple | None = None, tries: int = 6) -> bytes | None:
    for k in range(tries):
        try:
            req = urllib.request.Request(u)
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(1.5 * (k + 1))
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None


def parse_idx(txt: str):
    rows = []
    for ln in txt.strip().split("\n"):
        p = ln.split(":")
        if len(p) >= 7:
            rows.append((int(p[1]), p[3], p[4], p[5]))                      # offset, variable, level, forecast descriptor
    return rows


def message_bytes(idx, offsets_next, var, level, desc_pred):
    for i, (off, v, lv, d) in enumerate(idx):
        if v == var and lv.startswith(level) and desc_pred(d):
            end = idx[i + 1][0] - 1 if i + 1 < len(idx) else off + 2_000_000
            return off, end, d
    return None


def box_mean(msg: bytes) -> float:
    gid = eccodes.codes_new_from_message(msg)
    try:
        vals = eccodes.codes_get_values(gid)
        if "m" not in _MASK:
            lat, lon = eccodes.codes_get_array(gid, "latitudes"), eccodes.codes_get_array(gid, "longitudes")
            lon = np.where(lon > 180, lon - 360, lon) if False else lon
            _MASK["m"] = (lat >= BOX_LAT[0]) & (lat <= BOX_LAT[1]) & (lon >= BOX_LON[0]) & (lon <= BOX_LON[1])
        return float(np.mean(vals[_MASK["m"]]))
    finally:
        eccodes.codes_release(gid)


def one(task):
    day, fh = task
    init = day - dt.timedelta(days=1)
    txt = get(url(init, fh, ".idx"))
    if txt is None:
        return dict(date=day, fh=fh, a=np.nan, b=np.nan, dsw=np.nan, tcc=np.nan)
    idx = parse_idx(txt.decode())
    out = dict(date=day, fh=fh, a=np.nan, b=np.nan, dsw=np.nan, tcc=np.nan)
    m1 = message_bytes(idx, None, "DSWRF", "surface", lambda d: "hour ave" in d)
    m2 = message_bytes(idx, None, "TCDC", "entire atmosphere", lambda d: d.strip() == f"{fh} hour fcst")
    def fetch_mean(m):
        for _ in range(3):                                             # truncated range responses happen occasionally: fetch again
            b = get(url(init, fh), (m[0], m[1]))
            if b:
                try:
                    return box_mean(b)
                except Exception:
                    pass
        return np.nan
    if m1:
        out["dsw"] = fetch_mean(m1)
        a_, b_ = m1[2].split(" hour ave")[0].split("-")
        out["a"], out["b"] = float(a_), float(b_)
    if m2:
        out["tcc"] = fetch_mean(m2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=f"{CACHE}/gfs_national_raw.parquet")
    args = ap.parse_args()
    days = [d.date() for d in pd.date_range(args.start, args.end, freq="D")]
    tasks = [(d, fh) for d in days for fh in FHOURS]
    done = {}
    if os.path.exists(args.out):
        old = pd.read_parquet(args.out)
        old = old[old.dsw.notna() & old.tcc.notna()]                  # rows that failed earlier are fetched again
        done = {(pd.Timestamp(r.date).date(), r.fh): r._asdict() for r in old.itertuples(index=False)}
        tasks = [t for t in tasks if t not in done]
    print(f"{len(tasks):,} file tasks ({len(days)} target days x {len(FHOURS)} forecast hours), {len(done):,} cached", flush=True)
    t0 = time.time()
    rows = list(done.values())
    # warm the grid mask on the main thread before threading
    def safe(t):
        try:
            return one(t)
        except Exception as e:                                         # a bad file must not stop the run; it stays NaN and is retried by the next call
            return dict(date=t[0], fh=t[1], a=np.nan, b=np.nan, dsw=np.nan, tcc=np.nan)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(safe, t) for t in tasks]
        for i, f in enumerate(cf.as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 1000 == 0:
                print(f"  {i+1:,}/{len(tasks):,} in {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_parquet(args.out, index=False)           # checkpoint
    df = pd.DataFrame(rows).sort_values(["date", "fh"]).reset_index(drop=True)
    df.to_parquet(args.out, index=False)
    print(f"saved {args.out}: {len(df):,} rows, missing radiation {df.dsw.isna().mean():.3%}, missing cloud {df.tcc.isna().mean():.3%}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
