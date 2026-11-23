"""EXTERNAL SOURCE, post-contest study (18 Nov 2026): more fields from the SAME NOAA GFS files as 13_gfs_national.py, as national box means.

    python 26_gfs_extra_national.py [--start 2021-01-01 --end 2026-06-01 --workers 24]     # writes <CACHE>/gfsx_national_raw.parquet

Why: our GFS features hold only surface radiation and total cloud cover. On dull days, how dark it gets depends on the cloud TYPE and THICKNESS (low stratus vs thin
cirrus, cloud water); haze and fog (visibility) may explain the winter over-forecast; convective energy relates to afternoon showers. All of these are in the GFS files we
already use, so nothing changes about the source's legality:
  * the 00 UTC cycle of the day BEFORE the target day only (public about 12:45 KST, measured margin before 17:00 >= 2.3 h on every scoring day; post_contest/pubtime_check.py);
  * one box (33-39 N, 125-130 E) averaged to a single national value, the same for every plant; no plant coordinates.
Fields (instantaneous at the file's valid time unless noted): low / middle / high / convective cloud cover (%), cloud water and precipitable water (kg/m2), visibility (m),
sunshine duration (s, accumulated since the last 6-hourly reset), surface CAPE (J/kg), relative humidity at 850 and 700 hPa (%), and the precipitation total of the\n3- or 6-hour bucket ending at the valid time (kg/m2; apcp_a = start of the bucket).
"""
from __future__ import annotations
import argparse, importlib, os, time
import concurrent.futures as cf
import datetime as dt
import numpy as np
import pandas as pd
from oibc_lib import *

g13 = importlib.import_module("13_gfs_national")
FIELDS = {  # column: (variable, level prefix, descriptor test given the forecast hour)
    "lcdc": ("LCDC", "low cloud layer", lambda d, fh: d.strip() == f"{fh} hour fcst"),
    "mcdc": ("MCDC", "middle cloud layer", lambda d, fh: d.strip() == f"{fh} hour fcst"),
    "hcdc": ("HCDC", "high cloud layer", lambda d, fh: d.strip() == f"{fh} hour fcst"),
    "ccdc": ("TCDC", "convective cloud layer", lambda d, fh: d.strip() == f"{fh} hour fcst"),
    "cwat": ("CWAT", "entire atmosphere", lambda d, fh: d.strip() == f"{fh} hour fcst"),
    "pwat": ("PWAT", "entire atmosphere", lambda d, fh: d.strip() == f"{fh} hour fcst"),
    "vis": ("VIS", "surface", lambda d, fh: d.strip() == f"{fh} hour fcst"),
    "sunsd": ("SUNSD", "surface", lambda d, fh: d.strip() == f"{fh} hour fcst"),
    "cape": ("CAPE", "surface", lambda d, fh: d.strip() == f"{fh} hour fcst"),
    "rh850": ("RH", "850 mb", lambda d, fh: d.strip() == f"{fh} hour fcst"),
    "rh700": ("RH", "700 mb", lambda d, fh: d.strip() == f"{fh} hour fcst"),
    "apcp": ("APCP", "surface", lambda d, fh: d.strip() in (f"{fh - 3}-{fh} hour acc fcst", f"{fh - 6}-{fh} hour acc fcst")),   # GFS alternates 3- and 6-hour buckets
}


def one(task):
    day, fh = task
    init = day - dt.timedelta(days=1)
    out = dict(date=day, fh=fh, apcp_a=np.nan, **{k: np.nan for k in FIELDS})
    txt = g13.get(g13.url(init, fh, ".idx"))
    if txt is None:
        return out
    idx = g13.parse_idx(txt.decode())
    for col, (var, lev, test) in FIELDS.items():
        m = g13.message_bytes(idx, None, var, lev, lambda d: test(d, fh))
        if not m:
            continue
        for _ in range(3):
            b = g13.get(g13.url(init, fh), (m[0], m[1]))
            if b:
                try:
                    out[col] = g13.box_mean(b)
                    break
                except Exception:
                    pass
        if col == "apcp":
            out["apcp_a"] = float(m[2].strip().split("-")[0])                  # start of the accumulation window (fh - 3 or fh - 6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=f"{CACHE}/gfsx_national_raw.parquet")
    args = ap.parse_args()
    days = [d.date() for d in pd.date_range(args.start, args.end, freq="D")]
    tasks = [(d, fh) for d in days for fh in g13.FHOURS]
    rows = []
    if os.path.exists(args.out):
        old = pd.read_parquet(args.out)
        old = old[old[list(FIELDS)].notna().all(axis=1)]
        rows = [r._asdict() for r in old.itertuples(index=False)]
        done = {(pd.Timestamp(r["date"]).date(), r["fh"]) for r in rows}
        tasks = [t for t in tasks if t not in done]
    print(f"{len(tasks):,} file tasks ({len(days)} target days x {len(g13.FHOURS)} forecast hours, {len(FIELDS)} fields each), {len(rows):,} cached", flush=True)
    t0 = time.time()

    def safe(t):
        try:
            return one(t)
        except Exception:
            return dict(date=t[0], fh=t[1], apcp_a=np.nan, **{k: np.nan for k in FIELDS})
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(safe, t) for t in tasks]
        for i, f in enumerate(cf.as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1:,}/{len(tasks):,} in {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_parquet(args.out, index=False)
    df = pd.DataFrame(rows).sort_values(["date", "fh"]).reset_index(drop=True)
    df.to_parquet(args.out, index=False)
    print(f"saved {args.out}: {len(df):,} rows; missing share per field: " + ", ".join(f"{k} {df[k].isna().mean():.2%}" for k in FIELDS) + f"; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
