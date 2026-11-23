"""EXTERNAL SOURCE (guide section 7: fixed area summaries, the same columns for every plant): the values of a FIXED grid of cells over the Korean peninsula, not only one box mean.

    python 20_nwp_cells.py --source gfs|gefs|ecmwf --start 2021-01-01 --end 2026-06-01 [--workers 32]   # writes <CACHE>/nwp_cells_<source>.parquet

Same files, cycles and publication times as 13_gfs_national.py (GFS 1-degree, 00 UTC cycle of the day before), 14_gefs_national.py (GEFS ensemble mean, 0.5-degree, 00 UTC of the day before)
and 17_ecmwf_national.py (ECMWF IFS open data, 12 UTC of two days before); see external_data/README.md for the licences and the publication-time evidence.
Instead of the box mean, every grid point inside the fixed box 33-39 N, 125-130 E is stored (GFS 42 points, GEFS 143, ECMWF 525) in the order of the GRIB message (latitude descending, longitude ascending).
The box is a regular rectangle around the whole peninsula and the surrounding sea; it was not chosen by looking at plant locations.  Later steps reduce the points to fixed 1-degree cells
(regional_features.py), and the same cell columns are given to every plant.
Fields: GFS surface downward shortwave radiation (3-hour averages) and total cloud cover; GEFS ensemble-mean radiation and cloud; ECMWF accumulated surface solar radiation downwards.
"""
from __future__ import annotations
import argparse, importlib, json, os, time
import concurrent.futures as cf
import datetime as dt
import numpy as np
import pandas as pd
import eccodes
from oibc_lib import *

g13 = importlib.import_module("13_gfs_national")
g14 = importlib.import_module("14_gefs_national")
g17 = importlib.import_module("17_ecmwf_national")
get, parse_idx = g13.get, g13.parse_idx
BOX_LAT, BOX_LON = g13.BOX_LAT, g13.BOX_LON
_GEO = {}


def points(msg: bytes):
    """values of the grid points inside the box, in message order, plus their latitude / longitude (stored once per source)."""
    gid = eccodes.codes_new_from_message(msg)
    try:
        v = eccodes.codes_get_values(gid)
        if "m" not in _GEO:
            lat, lon = eccodes.codes_get_array(gid, "latitudes"), eccodes.codes_get_array(gid, "longitudes")
            m = (lat >= BOX_LAT[0]) & (lat <= BOX_LAT[1]) & (lon >= BOX_LON[0]) & (lon <= BOX_LON[1])
            _GEO["m"], _GEO["lat"], _GEO["lon"] = m, lat[m], lon[m]
        return v[_GEO["m"]].astype("float32")
    finally:
        eccodes.codes_release(gid)


def fetch(u, off, end):
    for _ in range(3):
        b = get(u, (off, end))
        if b:
            try:
                return points(b)
            except Exception:
                pass
    return None


# --------------------------------------------------------------------------------------------------- GFS
def one_gfs(task):
    day, fh = task
    init = day - dt.timedelta(days=1)
    row = dict(date=day, fh=fh, a=np.nan, b=np.nan, dsw=None, tcc=None)
    txt = get(g13.url(init, fh, ".idx"))
    if txt is None:
        return row
    idx = parse_idx(txt.decode())
    m1 = g13.message_bytes(idx, None, "DSWRF", "surface", lambda d: "hour ave" in d)
    m2 = g13.message_bytes(idx, None, "TCDC", "entire atmosphere", lambda d: d.strip() == f"{fh} hour fcst")
    if m1:
        row["dsw"] = fetch(g13.url(init, fh), m1[0], m1[1])
        a_, b_ = m1[2].split(" hour ave")[0].split("-")
        row["a"], row["b"] = float(a_), float(b_)
    if m2:
        row["tcc"] = fetch(g13.url(init, fh), m2[0], m2[1])
    return row


# --------------------------------------------------------------------------------------------------- GEFS ensemble mean
def one_gefs(task):
    day, fh = task
    init = day - dt.timedelta(days=1)
    row = dict(date=day, fh=fh, a=np.nan, b=np.nan, dsw=None, tcc=None)
    u = g14.url(init, "geavg", fh)
    txt = get(u + ".idx")
    if txt is None:
        return row
    idx = parse_idx(txt.decode())
    pos = {}
    for i, (off, v, lv, d) in enumerate(idx):
        if v == "TCDC" and lv.startswith("entire atmosphere") and "hour ave" in d:
            pos["tcc"] = i
        if v == "DSWRF" and lv.startswith("surface") and "hour ave" in d:
            pos["dsw"] = i
    for k, i in pos.items():
        end = idx[i + 1][0] - 1 if i + 1 < len(idx) else idx[i][0] + 1_000_000
        row[k] = fetch(u, idx[i][0], end)
        if k == "dsw":
            a_, b_ = idx[i][3].split(" hour ave")[0].split("-")
            row["a"], row["b"] = float(a_), float(b_)
    return row


# --------------------------------------------------------------------------------------------------- ECMWF IFS (12 UTC cycle of two days before)
def one_ecmwf(task):
    day, step = task
    init = day - dt.timedelta(days=2)
    row = dict(date=day, step=step, ssrd=None)
    txt = g17.get(g17.url(init, step, "index"))
    if txt is None:
        return row
    for ln in txt.decode().strip().split("\n"):
        j = json.loads(ln)
        if j.get("param") == "ssrd" and j.get("levtype") == "sfc":
            row["ssrd"] = fetch_with(g17.get, g17.url(init, step, "grib2"), j["_offset"], j["_offset"] + j["_length"] - 1)
            break
    return row


def fetch_with(getter, u, off, end):
    for _ in range(3):
        b = getter(u, (off, end))
        if b:
            try:
                return points(b)
            except Exception:
                pass
    return None


SOURCES = {"gfs": (one_gfs, g13.FHOURS, "fh"), "gefs": (one_gefs, g14.FHOURS, "fh"), "ecmwf": (one_ecmwf, g17.CYCLES[12]["steps"], "step")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(SOURCES))
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fn, steps, key = SOURCES[args.source]
    out = args.out or f"{CACHE}/nwp_cells_{args.source}.parquet"
    start = args.start if args.source != "ecmwf" else max(args.start, "2024-03-01")
    days = [d.date() for d in pd.date_range(start, args.end, freq="D")]
    tasks = [(d, s) for d in days for s in steps]
    fld = "dsw" if args.source != "ecmwf" else "ssrd"
    rows = []
    if os.path.exists(out):
        old = pd.read_parquet(out)
        old = old[old[fld].notna()]
        done = {(pd.Timestamp(r.date).date(), getattr(r, key)) for r in old.itertuples(index=False)}
        rows = [r._asdict() for r in old.itertuples(index=False)]
        tasks = [t for t in tasks if t not in done]
    print(f"{args.source}: {len(tasks):,} tasks, {len(rows):,} cached", flush=True)
    t0 = time.time()

    def safe(t):
        try:
            return fn(t)
        except Exception:
            return dict(date=t[0], **{key: t[1]})
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(safe, t) for t in tasks]
        for i, f in enumerate(cf.as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 1000 == 0:
                print(f"  {i+1:,}/{len(tasks):,} in {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_parquet(out, index=False)
    df = pd.DataFrame(rows).sort_values(["date", key]).reset_index(drop=True)
    df.to_parquet(out, index=False)
    json.dump(dict(lat=[float(x) for x in _GEO["lat"]], lon=[float(x) for x in _GEO["lon"]]), open(out.replace(".parquet", "_geo.json"), "w"))
    print(f"saved {out}: {len(df):,} rows, missing {df[fld].isna().mean():.2%}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
