"""EXTERNAL SOURCE (guide sections 5 and 7): KMA's own 1.5 km local weather model (UM-LDAPS), surface irradiance and cloud fields, reduced to FIXED 0.5-degree cells.

    export KMA_AUTHKEY=...            # or put the key in ~/.kma_authkey (never commit it; every user needs his or her own free key from apihub.kma.go.kr and has to press
                                      #  "API 활용신청" once for "8.1.1 UM모델 다운로드")
    python 22_kma_ldaps.py --probe
    python 22_kma_ldaps.py --download --start 2025-06-01 --end 2026-04-02 --vars tdsw [--workers 5] [--max-gb 3.5]

What is downloaded (KMA API Hub, terms: KOGL type 1, free general-member key, 20,000 calls / 5 GB per day; archive of this product 2018-01 ... 2026-03-31):
  * request       : https://apihub.kma.go.kr/api/typ06/url/nwp_vars_down.php?nwp=l015&sub=unis&vars=tdsw&tmfc=<YYYYMMDD>00&ef=<lead>&dataType=GRIB   (one GRIB2 field of ~1 MB per call)
  * model run     : the 09:00 KST (= 00 UTC) UM-LDAPS run of the day BEFORE the target day.  The official guide of the API Hub (NwpModelInfoService) lists the 09 KST run as callable from
                    15:00 KST ("09KST -> 15시 이후"), i.e. before the 17:00 KST cut-off of the contest.
  * lead hours    : 24, 26, 28, 30, 32 h = valid 09, 11, 13, 15, 17 KST of the target day; the field is the 1-hour mean flux that ends at the valid hour (GRIB stepRange 23-24 for lead 24),
                    i.e. exactly the contest's hour label.
  * variable      : tdsw = total downward surface short-wave flux (W/m2, GRIB 0/4/200); optional cloud layers lcdc / mcdc / hcdc / tcar and swdr / swdf (direct / diffuse flux)
  * area          : the 1.5 km grid points inside the fixed box 33-39 N, 125-130 E are averaged into 120 fixed cells of 0.5 degree (12 x 10); every plant later receives the same columns.
                    No plant coordinate is used anywhere in this script.
Output: <CACHE>/nwp_cells_ldaps.parquet  rows (date, lead, var, cells = list[120] of cell means, n = list[120] of the number of 1.5 km points per cell) + nwp_cells_ldaps_geo.json (cell centres).

The UM was switched off on 2026-03-31 (its last run is 2026-03-31 12 UTC); for target days after 2026-04-01 the KIM local model (nwp=l010, variable dswrsfc) has to be used instead (--model kim).
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, datetime as dt, json, os, sys, time, urllib.error, urllib.parse, urllib.request
import numpy as np
import pandas as pd
import eccodes
from oibc_lib import *

ROOT6 = "https://apihub.kma.go.kr/api/typ06"
BOX_LAT, BOX_LON = (33.0, 39.0), (125.0, 130.0)
CELL = 0.5
CLAT = np.arange(BOX_LAT[0], BOX_LAT[1], CELL) + CELL / 2          # 12 cell centres
CLON = np.arange(BOX_LON[0], BOX_LON[1], CELL) + CELL / 2          # 10 cell centres
CENTRES = [(a, b) for a in CLAT for b in CLON]                       # 120 fixed cells, latitude-major
LEADS = {"um": (24, 26, 28, 30, 32), "kim": (24, 26, 28, 30, 32)}   # valid 09, 11, 13, 15, 17 KST of the target day for the run of 00 UTC of the day before
_BYTES = {"n": 0}


def authkey() -> str:
    k = os.environ.get("KMA_AUTHKEY")
    if not k:
        p = os.path.expanduser("~/.kma_authkey")
        if os.path.exists(p):
            k = open(p).read().strip()
    if not k:
        raise RuntimeError("KMA authKey not found: set KMA_AUTHKEY or write it into ~/.kma_authkey (free key: apihub.kma.go.kr -> My page)")
    return k


def call(path: str, params: dict, tries: int = 5, timeout: int = 180) -> bytes:
    q = dict(params); q["authKey"] = authkey()
    url = f"{ROOT6}/{path}?" + urllib.parse.urlencode(q, safe=",")
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as r:
                b = r.read()
                _BYTES["n"] += len(b)
                return b
        except urllib.error.HTTPError as e:                                      # never put the URL (it contains the key) into a message
            last = f"HTTP {e.code}"
            if e.code in (400, 401, 403, 404):
                break
        except Exception as e:
            last = type(e).__name__
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{path}: {last}")


def grib_message(b: bytes):
    """the API answers with the file name on the first line followed by one GRIB2 message; '#...' lines mean 'no such file'."""
    i = b.find(b"GRIB")
    if i < 0:
        return None
    n = int.from_bytes(b[i + 8:i + 16], "big")
    return eccodes.codes_new_from_message(b[i:i + n])


# ------------------------------------------------------------------------------------------------ grid -> cells
def cell_index(cache: str, gid) -> np.ndarray:
    """cell number (0..119, -1 outside the fixed box) of every grid point of the 602 x 781 Lambert grid (taken from the GRIB message itself, cached)."""
    p = f"{cache}/ldaps_cell_index.npy"
    if os.path.exists(p):
        return np.load(p)
    lat = eccodes.codes_get_array(gid, "latitudes"); lon = eccodes.codes_get_array(gid, "longitudes")
    ia = np.floor((lat - BOX_LAT[0]) / CELL).astype(int)
    ib = np.floor((lon - BOX_LON[0]) / CELL).astype(int)
    ok = (ia >= 0) & (ia < len(CLAT)) & (ib >= 0) & (ib < len(CLON))
    idx = np.where(ok, ia * len(CLON) + ib, -1).astype("int16")
    np.save(p, idx)
    return idx


def cell_means(vals: np.ndarray, cell: np.ndarray):
    ok = (cell >= 0) & np.isfinite(vals) & (vals < 9000) & (vals > -900)                 # missing = 9999 in the GRIB
    s = np.bincount(cell[ok], weights=vals[ok], minlength=len(CENTRES))
    n = np.bincount(cell[ok], minlength=len(CENTRES))
    return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype("float32"), n.astype("int32")


# ------------------------------------------------------------------------------------------------ one field
def one_field(task, model, cell):
    day, lead, var = task
    init = day - dt.timedelta(days=1)
    row = dict(date=day, lead=lead, var=var, cells=None, n=None)
    if model == "um":
        params = dict(nwp="l015", sub="unis", vars=var, tmfc=init.strftime("%Y%m%d") + "00", ef=lead, dataType="GRIB")
    else:
        params = dict(nwp="l010", sub="unis", vars=var, tmfc=init.strftime("%Y%m%d") + "0000", ef=lead, dataType="GRIB")
    try:
        gid = grib_message(call("url/nwp_vars_down.php", params))
        if gid is None:
            return row
        try:
            v = eccodes.codes_get_values(gid)
            if len(v) == len(cell):
                row["cells"], row["n"] = cell_means(v, cell)
        finally:
            eccodes.codes_release(gid)
    except Exception:
        pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--model", default="um", choices=["um", "kim"])
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2026-04-02")
    ap.add_argument("--vars", default="tdsw")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-gb", type=float, default=3.5, help="stop after this much download volume (daily quota of a general member: 5 GB)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    if args.probe:
        b = call("url/nwp_vars_down.php", dict(nwp="l015", sub="unis", vars="tdsw", tmfc="2025060100", ef=24, dataType="GRIB"))
        gid = grib_message(b)
        cell = cell_index(CACHE, gid)
        v = eccodes.codes_get_values(gid)
        m, n = cell_means(v, cell)
        print("points", len(v), "in box", int((cell >= 0).sum()), "cells with data", int(np.isfinite(m).sum()), "mean tdsw in cells", float(np.nanmean(m)))
        return
    if args.download:
        out = args.out or f"{CACHE}/nwp_cells_{'ldaps' if args.model == 'um' else 'kimloc'}.parquet"
        b = call("url/nwp_vars_down.php", dict(nwp="l015", sub="unis", vars="tdsw", tmfc="2025060100", ef=24, dataType="GRIB")) if args.model == "um" else None
        if b is not None:
            gid = grib_message(b); cell = cell_index(CACHE, gid); eccodes.codes_release(gid)
        else:
            cell = np.load(f"{CACHE}/ldaps_cell_index.npy")
        days = [d.date() for d in pd.date_range(args.start, args.end, freq="D")]
        tasks = [(d, L, v) for d in days for L in LEADS[args.model] for v in args.vars.split(",")]
        rows = []
        if os.path.exists(out):
            old = pd.read_parquet(out)
            old = old[old.cells.notna()]
            done = {(pd.Timestamp(r.date).date(), int(r.lead), r.var) for r in old.itertuples(index=False)}
            rows = [r._asdict() for r in old.itertuples(index=False)]
            tasks = [t for t in tasks if t not in done]
        print(f"{len(tasks):,} fields to fetch, {len(rows):,} cached, budget {args.max_gb} GB", flush=True)
        t0 = time.time()
        stop = {"flag": False}

        def run(t):
            if stop["flag"]:
                return dict(date=t[0], lead=t[1], var=t[2], cells=None, n=None, skipped=True)
            r = one_field(t, args.model, cell)
            if _BYTES["n"] > args.max_gb * 1e9:
                stop["flag"] = True
            return r
        with cf.ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(run, t) for t in tasks]
            for i, f in enumerate(cf.as_completed(futs)):
                r = f.result()
                if not r.get("skipped"):
                    rows.append({k: r[k] for k in ("date", "lead", "var", "cells", "n")})
                if (i + 1) % 500 == 0:
                    print(f"  {i+1:,}/{len(tasks):,} in {time.time()-t0:.0f}s, {_BYTES['n']/1e9:.2f} GB", flush=True)
                    pd.DataFrame(rows).to_parquet(out, index=False)
        df = pd.DataFrame(rows).sort_values(["date", "var", "lead"]).reset_index(drop=True)
        df.to_parquet(out, index=False)
        json.dump(dict(lat=[float(a) for a, b in CENTRES], lon=[float(b) for a, b in CENTRES], cell_deg=CELL), open(out.replace(".parquet", "_geo.json"), "w"))
        print(f"saved {out}: {len(df):,} rows, missing {df.cells.isna().mean():.2%}, {_BYTES['n']/1e9:.2f} GB, {time.time()-t0:.0f}s" + ("  (stopped at the budget)" if stop["flag"] else ""))


if __name__ == "__main__":
    main()
