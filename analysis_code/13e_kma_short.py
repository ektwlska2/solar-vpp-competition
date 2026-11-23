"""EXPERIMENT C (test plan): KMA village-forecast (동네예보) SHORT-RANGE GRID, the 14:00 KST issue of the day BEFORE the target day (issued 14:00, available ~14:10 KST: before the 17:00 cut-off).

    python 13e_kma_short.py --start 2021-09-01 --end 2025-05-31 [--max-gb 4.7]      # writes <CACHE>/kma_short_raw.parquet

  * endpoint: https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd?tmfc=<D-1>14&tmef=<D>HH&vars=SKY|PTY&authKey=...   (API Hub key in KMA_AUTHKEY or ~/.kma_authkey, usage application 'KMA village forecast grid' approved;
    Public Nuri type 1 / source attribution; the URL (with the key) is never written to any message or file)
  * valid hours used: 09, 11, 13, 15, 17 KST of the target day;  variables: SKY (1 clear, 3 mostly cloudy, 4 cloudy), PTY (0 none, 1 rain, 2 rain/snow, 3 snow, 4 shower)
  * archive: valid from about 2021-08/09 (2021-06 and earlier return 0 / -99); the 5 km grid (149 x 253, Lambert) is reduced at once to fixed-area category fractions: the national box 33-39 N x 125-130 E and its four fixed quadrants
    (split at 36.0 N and 127.5 E); the same columns for every plant, no plant coordinate is used
  * ordinary member quota 20,000 calls / 5 GB per day, one grid call ~ 340 kB; parallel calls above ~2 are throttled by the server (timeouts), so 2 workers with retries are used.
"""
from __future__ import annotations
import argparse, math, os, re, sys, time, urllib.error, urllib.request
import concurrent.futures as cf
import datetime as dt
import numpy as np
import pandas as pd
from oibc_lib import *

ROOT = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd"
HOURS = (9, 11, 13, 15, 17)
VARS = {"SKY": (1, 3, 4), "PTY": (0, 1, 2, 3, 4)}
_BYTES = {"n": 0}
_AREA = {}


def authkey() -> str:
    k = os.environ.get("KMA_AUTHKEY")
    if not k and os.path.exists(os.path.expanduser("~/.kma_authkey")):
        k = open(os.path.expanduser("~/.kma_authkey")).read().strip()
    if not k:
        raise RuntimeError("KMA authKey not found")
    return k


def grid_latlon(nx=149, ny=253):
    RE, GRID, SLAT1, SLAT2, OLON, OLAT, XO, YO = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0, 43, 136
    D = math.pi / 180; re_ = RE / GRID; s1, s2, olon, olat = SLAT1 * D, SLAT2 * D, OLON * D, OLAT * D
    sn = math.log(math.cos(s1) / math.cos(s2)) / math.log(math.tan(math.pi * .25 + s2 * .5) / math.tan(math.pi * .25 + s1 * .5))
    sf = math.tan(math.pi * .25 + s1 * .5) ** sn * math.cos(s1) / sn
    ro = re_ * sf / math.tan(math.pi * .25 + olat * .5) ** sn
    lat = np.zeros((ny, nx)); lon = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            xn, yn = (i + 1) - XO, ro - ((j + 1) - YO)
            ra = math.hypot(xn, yn) * (1 if sn > 0 else -1)
            la = 2 * math.atan((re_ * sf / ra) ** (1 / sn)) - math.pi * .5
            th = math.atan2(xn, yn) if ra != 0 else 0
            lat[j, i] = la / D; lon[j, i] = (th / sn + olon) / D
    return lat, lon


def areas():
    if not _AREA:
        lat, lon = grid_latlon()
        box = (lat >= 33) & (lat <= 39) & (lon >= 125) & (lon <= 130)
        _AREA["nat"] = box
        _AREA["NW"] = box & (lat >= 36.0) & (lon < 127.5); _AREA["NE"] = box & (lat >= 36.0) & (lon >= 127.5)
        _AREA["SW"] = box & (lat < 36.0) & (lon < 127.5); _AREA["SE"] = box & (lat < 36.0) & (lon >= 127.5)
    return _AREA


def fetch(tmfc: str, tmef: str, var: str, tries: int = 10):
    url = f"{ROOT}?tmfc={tmfc}&tmef={tmef}&vars={var}&authKey={authkey()}"
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=12) as r:
                b = r.read()
            _BYTES["n"] += len(b)
            v = np.array([float(x) for x in re.findall(rb"-?\d+\.?\d*", b)])
            if v.size == 149 * 253:
                return v.reshape(253, 149)
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404):
                return None
        except Exception:
            pass
        time.sleep(0.4 * (k + 1))
    return None


def one(task):
    day, h, var = task
    issue = day - dt.timedelta(days=1)
    g = fetch(f"{issue:%Y%m%d}14", f"{day:%Y%m%d}{h:02d}", var)
    row = dict(date=day, hour=h, var=var, ok=False)
    if g is None:
        return row
    for a, m in areas().items():
        vals = g[m]; vals = vals[vals > -900]
        if vals.size == 0:
            continue
        for c in VARS[var]:
            row[f"{a}_{c}"] = float((vals == c).mean())
        row[f"{a}_n"] = int(vals.size)
    row["ok"] = True
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-09-01"); ap.add_argument("--end", default="2025-05-31"); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-gb", type=float, default=4.7); ap.add_argument("--out", default=f"{CACHE}/kma_short_raw.parquet")
    args = ap.parse_args()
    areas()
    days = [d.date() for d in pd.date_range(args.start, args.end, freq="D")]
    tasks = [(d, h, v) for d in days for h in HOURS for v in VARS]
    rows, done = [], set()
    if os.path.exists(args.out):
        old = pd.read_parquet(args.out); old = old[old.ok]
        rows = [r._asdict() for r in old.itertuples(index=False)]; done = {(pd.Timestamp(r["date"]).date(), r["hour"], r["var"]) for r in rows}
        tasks = [t for t in tasks if t not in done]
    print(f"{len(tasks):,} grid calls to fetch, {len(done):,} cached, budget {args.max_gb} GB", flush=True)
    t0 = time.time(); stop = {"f": False}

    def run(t):
        if stop["f"]:
            return None
        r = one(t)
        if _BYTES["n"] > args.max_gb * 1e9:
            stop["f"] = True
        return r
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run, t) for t in tasks]
        for i, f in enumerate(cf.as_completed(futs)):
            r = f.result()
            if r is not None:
                rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i+1:,}/{len(tasks):,} in {time.time()-t0:.0f}s, {_BYTES['n']/1e9:.2f} GB", flush=True)
                pd.DataFrame(rows).to_parquet(args.out, index=False)
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"saved {args.out}: {len(df):,} rows, ok {df.ok.mean():.3%}, {_BYTES['n']/1e9:.2f} GB, {time.time()-t0:.0f}s" + ("  (stopped at the byte budget)" if stop["f"] else ""), flush=True)


if __name__ == "__main__":
    main()
