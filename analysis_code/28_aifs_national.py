"""EXTERNAL SOURCE, post-contest study (18 Nov 2026): ECMWF's machine-learned model AIFS Single, 00 UTC run of the day BEFORE the target day, as national box means.

    python -I 28_aifs_national.py [--start 2025-02-26 --end 2026-05-31 --workers 6]      # writes <CACHE>/aifs_national_raw.parquet   (run with venv_grib: needs eccodes)

Source: ECMWF open data (CC-BY-4.0), Google Cloud mirror https://storage.googleapis.com/ecmwf-open-data/<init>/00z/aifs-single/0p25/oper/<init>000000-<step>h-oper-fc.grib2
  (the same files are on AWS: https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/...),
  one GRIB message per field read with an HTTP byte range taken from the matching .index file; nothing is written to disk except the table of box means.
History: AIFS Single 1.0 became operational on 25 Feb 2025; the earlier experimental AIFS files in the same bucket carry no radiation or cloud fields (checked 18 Nov 2026).
Rules: only the 00 UTC run of D-1 (posted about 14:15 KST on AWS: 364 of 365 scoring days before the 17:00 KST cut-off, post_contest/pubtime_candidates.py); one box (33-39 N, 125-130 E,
  the same box as every other national feature) averaged to one value per field, identical for every plant; no plant coordinates.
Fields (steps 24 h and 30 h = 09 and 15 KST of the target day):
  ssrd  surface solar radiation downwards, accumulated since the start of the run (J/m2): (ssrd30 - ssrd24) / 21600 = mean flux 09-15 KST (hour labels 10-15)
  tcc   total cloud cover (%), instantaneous
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, datetime as dt, email.utils, json, os, random, time, urllib.error, urllib.request
import numpy as np
import pandas as pd

BASE = "https://storage.googleapis.com/ecmwf-open-data"          # ECMWF's Google Cloud mirror (same files and timestamps as AWS; AWS throttled with 503 on 18 Nov 2026)
BOX_LAT, BOX_LON = (33.0, 39.0), (125.0, 130.0)
STEPS, PARAMS = (24, 30), ("ssrd", "tcc")
CACHE = os.environ.get("OIBC_CACHE", os.path.expanduser("~/oibc_scratch_backup/cache2"))
_MASK = {}
_BASE = [BASE]


def http(u, rng=None, tries=8, head=False):
    for k in range(tries):
        try:
            req = urllib.request.Request(u, method="HEAD" if head else "GET", headers={"User-Agent": "python-urllib"})
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=90) as r:
                return (r.headers.get("Last-Modified") if head else r.read())
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(random.uniform(0.5, 2.5) * min(k + 1, 6))
        except Exception:
            time.sleep(random.uniform(0.5, 2.5) * min(k + 1, 6))
    return None


def box_mean(msg):
    import eccodes
    gid = eccodes.codes_new_from_message(msg)
    try:
        v = np.asarray(eccodes.codes_get_values(gid))
        if "m" not in _MASK:
            lat, lon = eccodes.codes_get_array(gid, "latitudes"), eccodes.codes_get_array(gid, "longitudes")
            _MASK["m"] = (lat >= BOX_LAT[0]) & (lat <= BOX_LAT[1]) & (lon >= BOX_LON[0]) & (lon <= BOX_LON[1])
        x = v[_MASK["m"]]
        return float(x.mean()), float(x.std()), int(x.size), int(eccodes.codes_get(gid, "endStep")), int(eccodes.codes_get(gid, "dataDate"))
    finally:
        eccodes.codes_release(gid)


def one(task):
    D, st = task
    init = D - dt.timedelta(days=1)
    base = f"{_BASE[0]}/{init:%Y%m%d}/00z/aifs-single/0p25/oper/{init:%Y%m%d}000000-{st}h-oper-fc"
    rows = []
    txt = http(base + ".index")
    posted = None   # the Google mirror was back-filled for early 2025 (its file times are up to 2 days late): publication times come from AWS (post_contest/pubtime_candidates.py)
    found = {}
    if txt is not None:
        for ln in txt.decode().strip().split("\n"):
            j = json.loads(ln)
            if j.get("param") in PARAMS and j.get("levtype") == "sfc":
                found[j["param"]] = (j["_offset"], j["_length"], j.get("expver"))
    for p in PARAMS:
        r = dict(date=pd.Timestamp(D), step=st, param=p, mean=np.nan, sd=np.nan, n=0, posted=posted, expver=None)
        if p in found:
            off, length, ev = found[p]
            r["expver"] = ev
            for _ in range(3):
                blob = http(base + ".grib2", (off, off + length - 1))
                if blob:
                    try:
                        m, s, n, es, dd = box_mean(blob)
                        if es == st and dd == int(f"{init:%Y%m%d}"):
                            r.update(mean=m, sd=s, n=n)
                        break
                    except Exception as e:
                        print("decode error", D, st, p, e, flush=True)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-02-26")
    ap.add_argument("--end", default="2026-05-31")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=f"{CACHE}/aifs_national_raw.parquet")
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()
    _BASE[0] = a.base
    days = [d.date() for d in pd.date_range(a.start, a.end, freq="D")]
    tasks = [(D, st) for D in days for st in STEPS]
    t0, rows = time.time(), []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, tasks)):
            rows.extend(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(tasks)} files ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["date", "step", "param"]).reset_index(drop=True)
    df.to_parquet(a.out, index=False)
    miss = df.groupby("param")["mean"].apply(lambda s: int(s.isna().sum()))
    print(f"wrote {a.out}: {len(df)} rows, {df.date.nunique()} target days; missing per field: {miss.to_dict()}; expver {sorted(df.expver.dropna().unique())}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
