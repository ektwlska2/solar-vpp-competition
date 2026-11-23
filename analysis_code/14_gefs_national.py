"""EXTERNAL SOURCE (rules: 'external source notice', section 5): a broad-area summary of the NOAA GEFS ensemble forecast, the same value for every plant.

    python 14_gefs_national.py --start 2021-01-01 --end 2026-06-01 [--workers 48]      # writes <CACHE>/gefs_national_raw.parquet

NOAA GEFS (ensemble of 31 GFS runs) on the public AWS Open Data bucket noaa-gefs-pds (US-government data, free for commercial use, archived at fixed addresses since 2020,
re-downloadable by anyone): https://noaa-gefs-pds.s3.amazonaws.com/gefs.YYYYMMDD/00/atmos/pgrb2ap5/{geavg|gespr}.t00z.pgrb2a.0p50.fXXX
  * ensemble MEAN (geavg) and ensemble SPREAD (gespr) of surface downward shortwave radiation (3-hour averages) and total cloud cover (3-hour averages)
  * 00 UTC cycle of the day BEFORE the target day only (09:00 KST, files online about 14:00 KST: before the 17:00 KST cut-off)
  * ONE box (33-39 N, 125-130 E) averaged to a single national value per hour and attached to every plant identically; no plant coordinates, no nearest-grid selection;
    also the spatial standard deviation of the ensemble mean inside the box (how patchy the forecast sky is)
Output rows: target date, forecast hour, averaging window (a, b), and for mean/spread of DSWRF and TCDC the box mean (and, for the mean, the spatial std).
"""
from __future__ import annotations
import argparse, importlib, os, time
import concurrent.futures as cf
import datetime as dt
import numpy as np
import pandas as pd
import eccodes
from oibc_lib import *

g13 = importlib.import_module("13_gfs_national")
get, parse_idx = g13.get, g13.parse_idx
BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
FHOURS = (21, 24, 27, 30, 33, 36)                    # valid 06:00 ... 21:00 KST of the target day for the 00 UTC cycle of the previous day
_MASK = {}


def url(init: dt.date, kind: str, fh: int, ext: str = "") -> str:
    return f"{BASE}/gefs.{init:%Y%m%d}/00/atmos/pgrb2ap5/{kind}.t00z.pgrb2a.0p50.f{fh:03d}{ext}"


def stats(msg: bytes) -> tuple[float, float]:
    gid = eccodes.codes_new_from_message(msg)
    try:
        v = eccodes.codes_get_values(gid)
        if "m" not in _MASK:
            lat, lon = eccodes.codes_get_array(gid, "latitudes"), eccodes.codes_get_array(gid, "longitudes")
            _MASK["m"] = (lat >= g13.BOX_LAT[0]) & (lat <= g13.BOX_LAT[1]) & (lon >= g13.BOX_LON[0]) & (lon <= g13.BOX_LON[1])
        x = v[_MASK["m"]]
        return float(np.mean(x)), float(np.std(x))
    finally:
        eccodes.codes_release(gid)


def two_fields(init, kind, fh):
    """TCDC and DSWRF are neighbouring messages of the file: one range request returns both."""
    txt = get(url(init, kind, fh, ".idx"))
    if txt is None:
        return None
    idx = parse_idx(txt.decode())
    pos = {}
    for i, (off, v, lv, d) in enumerate(idx):
        if v == "TCDC" and lv.startswith("entire atmosphere") and "hour ave" in d:
            pos["TCDC"] = i
        if v == "DSWRF" and lv.startswith("surface") and "hour ave" in d:
            pos["DSWRF"] = i
    if "TCDC" not in pos or "DSWRF" not in pos:
        return None
    lo, hi = sorted((pos["TCDC"], pos["DSWRF"]))
    if hi - lo != 1:                                       # not adjacent: fall back to two requests
        out = {}
        for k, i in pos.items():
            end = idx[i + 1][0] - 1 if i + 1 < len(idx) else idx[i][0] + 2_000_000
            for _ in range(3):
                b = get(url(init, kind, fh), (idx[i][0], end))
                try:
                    out[k] = (stats(b), idx[i][3]) if b else None
                    break
                except Exception:
                    out[k] = None
        return out
    end = idx[hi + 1][0] - 1 if hi + 1 < len(idx) else idx[hi][0] + 2_000_000
    for _ in range(3):
        blob = get(url(init, kind, fh), (idx[lo][0], end))
        if not blob:
            continue
        try:
            cut = idx[hi][0] - idx[lo][0]
            parts = {idx[lo][1]: (stats(blob[:cut]), idx[lo][3]), idx[hi][1]: (stats(blob[cut:]), idx[hi][3])}
            return parts
        except Exception:
            continue
    return None


def one(task):
    day, fh = task
    init = day - dt.timedelta(days=1)
    row = dict(date=day, fh=fh, a=np.nan, b=np.nan, dsw_m=np.nan, dsw_sd=np.nan, tcc_m=np.nan, tcc_sd=np.nan, dsw_sp=np.nan, tcc_sp=np.nan)
    m = two_fields(init, "geavg", fh)
    if m:
        if "DSWRF" in m and m["DSWRF"]:
            (row["dsw_m"], row["dsw_sd"]), d = m["DSWRF"]
            a_, b_ = d.split(" hour ave")[0].split("-")
            row["a"], row["b"] = float(a_), float(b_)
        if "TCDC" in m and m["TCDC"]:
            (row["tcc_m"], row["tcc_sd"]), _ = m["TCDC"]
    s = two_fields(init, "gespr", fh)
    if s:
        if "DSWRF" in s and s["DSWRF"]:
            row["dsw_sp"] = s["DSWRF"][0][0]
        if "TCDC" in s and s["TCDC"]:
            row["tcc_sp"] = s["TCDC"][0][0]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=f"{CACHE}/gefs_national_raw.parquet")
    args = ap.parse_args()
    days = [d.date() for d in pd.date_range(args.start, args.end, freq="D")]
    tasks = [(d, fh) for d in days for fh in FHOURS]
    rows = []
    if os.path.exists(args.out):
        old = pd.read_parquet(args.out)
        old = old[old.dsw_m.notna() & old.tcc_m.notna() & old.dsw_sp.notna()]
        done = {(pd.Timestamp(r.date).date(), r.fh) for r in old.itertuples(index=False)}
        rows = [r._asdict() for r in old.itertuples(index=False)]
        tasks = [t for t in tasks if t not in done]
    print(f"{len(tasks):,} file tasks, {len(rows):,} cached", flush=True)
    t0 = time.time()

    def safe(t):
        try:
            return one(t)
        except Exception:
            return dict(date=t[0], fh=t[1], a=np.nan, b=np.nan, dsw_m=np.nan, dsw_sd=np.nan, tcc_m=np.nan, tcc_sd=np.nan, dsw_sp=np.nan, tcc_sp=np.nan)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(safe, t) for t in tasks]
        for i, f in enumerate(cf.as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 1000 == 0:
                print(f"  {i+1:,}/{len(tasks):,} in {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_parquet(args.out, index=False)
    df = pd.DataFrame(rows).sort_values(["date", "fh"]).reset_index(drop=True)
    df.to_parquet(args.out, index=False)
    print(f"saved {args.out}: {len(df):,} rows, missing mean radiation {df.dsw_m.isna().mean():.3%}, missing spread {df.dsw_sp.isna().mean():.3%}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
