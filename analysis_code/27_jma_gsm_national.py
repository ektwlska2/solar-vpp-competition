"""EXTERNAL SOURCE, post-contest study (18 Nov 2026): JMA's global model GSM, 00 UTC run of the day BEFORE the target day, as national box means.

    python 27_jma_gsm_national.py [--start 2021-01-01 --end 2026-05-31 --workers 3] [--overlap-days 30]      # writes <CACHE>/jma_gsm_national_raw.parquet
    (run with venv_grib: needs eccodes)

Source: the public archive of Kyoto University RISH, http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/<YYYY>/<MM>/<DD>/
  Japan Meteorological Agency numerical forecasts distributed through the Japan Meteorological Business Support Center; the archive serves education and research
  institutions ("企業活動等のためにデータを頻繁に必要とされる方は，気象業務支援センターからデータを直接購入"); free, no key.
Why this source: it is almost certainly the third source of the leading team xHigh. The organisers' audit lists it as on time on all 365 days of the scoring year with a
  smallest margin of 4.37 h before 17:00 KST of D-1; the archive's file times give exactly that: posted 12:37-12:38 KST every day (post_contest/pubtime_jma.py).
Rules: only the 00 UTC run of D-1 (posted by 12:38 KST); one box (33-39 N, 125-130 E, the same box as the GFS features) averaged to a single value per hour, identical
  for every plant; no plant coordinates.
Products (surface GPV, hourly):
  target days up to 2024-09-18: GSM_GPV_Rjp_Lsurf_FD0000-0312 (0.25 x 0.2 deg, leads 0-84 h; discontinued after the 2024-09-17 run)
  later target days:            GSM_GPV_Rjp_Gll0p1deg_Lsurf_FD0000-0100 + FD0101-0200 (0.1 deg; exists from the 2023-03-14 run)
  --overlap-days N also reads the 0.1 deg product for the last N runs of the old one, to check that the two grids give the same box means.
Fields, leads 17-38 h (lead L of the D-1 00 UTC run = hour label L-15 KST of the target day):
  dsw   mean downward short-wave flux of the hour ending at the lead (avg_sdswrf, W/m2)
  tcc, lcc, mcc, hcc   total / low / middle / high cloud cover (%), at the lead
  apcp  precipitation accumulated since the start of the run (kg/m2)
  t2, rh2   2 m temperature (K) and relative humidity (%)
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, datetime as dt, os, sys, tempfile, time, urllib.request
import numpy as np
import pandas as pd

BASE = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original"
OLD_LAST_RUN = dt.date(2024, 9, 17)
BOX = (33.0, 39.0, 125.0, 130.0)
LEADS = range(17, 39)
PARAMS = {(4, 7): "dsw", (6, 1): "tcc", (6, 3): "lcc", (6, 4): "mcc", (6, 5): "hcc", (1, 8): "apcp", (0, 0): "t2", (1, 1): "rh2"}
CACHE = os.environ.get("OIBC_CACHE", os.path.expanduser("~/oibc_scratch_backup/cache2"))


def files_for(run: dt.date, product: str):
    stamp = f"{run:%Y%m%d}000000"
    if product == "old":
        return [f"Z__C_RJTD_{stamp}_GSM_GPV_Rjp_Lsurf_FD0000-0312_grib2.bin"]
    return [f"Z__C_RJTD_{stamp}_GSM_GPV_Rjp_Gll0p1deg_Lsurf_FD0000-0100_grib2.bin", f"Z__C_RJTD_{stamp}_GSM_GPV_Rjp_Gll0p1deg_Lsurf_FD0101-0200_grib2.bin"]


def fetch(url: str, path: str):
    for k in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=180) as r, open(path, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            time.sleep(10 * (k + 1))
        except Exception:
            time.sleep(10 * (k + 1))
    return False


def box_means(path: str):
    import eccodes
    eccodes.codes_grib_multi_support_on()
    out, mask = {}, None
    with open(path, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                key = (eccodes.codes_get(h, "parameterCategory"), eccodes.codes_get(h, "parameterNumber"))
                name = PARAMS.get(key)
                if name is None or eccodes.codes_get(h, "discipline") != 0:
                    continue
                if name in ("t2", "rh2") and eccodes.codes_get(h, "typeOfLevel") != "heightAboveGround":
                    continue
                lead = int(str(eccodes.codes_get(h, "stepRange")).split("-")[-1])
                if lead not in LEADS:
                    continue
                if mask is None:
                    ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
                    lat = np.asarray(eccodes.codes_get_array(h, "latitudes")).reshape(nj, ni)
                    lon = np.asarray(eccodes.codes_get_array(h, "longitudes")).reshape(nj, ni)
                    mask = (lat >= BOX[0]) & (lat <= BOX[1]) & (lon >= BOX[2]) & (lon <= BOX[3])
                v = np.asarray(eccodes.codes_get_values(h)).reshape(mask.shape)
                miss = eccodes.codes_get(h, "missingValue")
                sel = v[mask]
                sel = sel[sel != miss]
                out[(lead, name)] = float(sel.mean()) if sel.size else np.nan
            finally:
                eccodes.codes_release(h)
    return out


def one(task):
    target, product = task
    run = target - dt.timedelta(days=1)
    vals, ok = {}, True
    with tempfile.TemporaryDirectory(prefix="jma_") as tmp:
        for fn in files_for(run, product):
            p = os.path.join(tmp, "f.bin")
            if not fetch(f"{BASE}/{run:%Y/%m/%d}/{fn}", p):
                ok = False
                break
            try:
                vals.update(box_means(p))
            except Exception:
                ok = False
                break
            finally:
                if os.path.exists(p):
                    os.remove(p)
    rows = []
    for lead in LEADS:
        r = dict(date=pd.Timestamp(target), lead=lead, label=lead - 15, product=product)
        for nm in PARAMS.values():
            r[nm] = vals.get((lead, nm), np.nan) if ok else np.nan
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01", help="first target day")
    ap.add_argument("--end", default="2026-05-31", help="last target day")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--overlap-days", type=int, default=0)
    ap.add_argument("--out", default=f"{CACHE}/jma_gsm_national_raw.parquet")
    args = ap.parse_args()
    days = [d.date() for d in pd.date_range(args.start, args.end, freq="D")]
    tasks = [(d, "old" if d - dt.timedelta(days=1) <= OLD_LAST_RUN else "new") for d in days]
    if args.overlap_days:
        tasks += [(OLD_LAST_RUN + dt.timedelta(days=1) - dt.timedelta(days=k), "new") for k in range(args.overlap_days)]
    rows = []
    if os.path.exists(args.out):
        old = pd.read_parquet(args.out)
        good = old.groupby(["date", "product"]).dsw.apply(lambda s: s.notna().sum() >= 15)
        keep = {(pd.Timestamp(d).date(), p) for (d, p), g in good.items() if g}
        rows = old[[(pd.Timestamp(d).date(), p) in keep for d, p in zip(old.date, old["product"])]].to_dict("records")
        tasks = [t for t in tasks if t not in keep]
    print(f"{len(tasks):,} runs to read, {len(rows) // len(LEADS):,} cached", flush=True)
    t0 = time.time()
    with cf.ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(one, tasks, chunksize=1)):
            rows.extend(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1:,}/{len(tasks):,} in {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_parquet(args.out, index=False)
    df = pd.DataFrame(rows).sort_values(["product", "date", "lead"]).reset_index(drop=True)
    df.to_parquet(args.out, index=False)
    miss = df.groupby("product")[list(PARAMS.values())].apply(lambda g: g.isna().mean())
    print(f"saved {args.out}: {len(df):,} rows, {df.date.nunique()} target days; missing share:\n{miss.round(4).to_string()}\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
