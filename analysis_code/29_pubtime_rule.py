"""Publication-time rule for the external weather tables (post-contest, 18 Nov 2026; used by V19p_gfsx_gru).

    python 29_pubtime_rule.py scan  [--out external_data/pubtimes_scan.csv.gz --start 2021-01-01 --end 2026-05-31 --workers 24]   # needs eccodes (imports the download scripts for their addresses)
    python 29_pubtime_rule.py apply [--scan external_data/pubtimes_scan.csv.gz --src external_data --out external_data_pubrule]     # pandas / pyarrow only
    then: python 08_make_submission.py ... --nwp-dir external_data_pubrule

Rule (contest notice section 6: a file that reached its public archive after 17:00 KST of the day before the target day may not be used for that
day, even if it was computed earlier; values may be left out by a rule fixed in advance):
  * for every external table and every target day, the table's values for that day are kept only if EVERY file the table reads for that day was
    posted at the public address named in its download script before 17:00 KST of the previous day;
  * posting time = HTTP Last-Modified of the file on the AWS Open Data bucket (the address in 13_ / 14_ / 15_ / 17_*.py);
  * otherwise (posted at or after the cut-off, or the file can no longer be found) ALL values of that table for that day are left out:
    the rows are emptied exactly like a failed download, which every feature builder already treats as missing;
  * the same rule for the training years and the scoring year; nothing is filled in from other runs.

Files checked per target day D (the files the download scripts read; GRIB messages are read with byte ranges from these files):
  gfs   NOAA GFS 00 UTC of D-1, pgrb2.1p00 f018-f039 (3-hourly)        -> gfs_national_raw, gfsx_national_raw, nwp_cells_gfs
  gefsm NOAA GEFS 00 UTC of D-1, geavg pgrb2a.0p50 f021-f036            -> gefs_national_raw, nwp_cells_gefs
  gefss NOAA GEFS 00 UTC of D-1, gespr pgrb2a.0p50 f021-f036            -> gefs_national_raw
  chem  NOAA GEFS-Aerosols 00 UTC of D-1, a2d_0p25 f027, f030           -> aerosol_national_raw
  ecm12 ECMWF IFS open data 12 UTC of D-2, steps 33-48 h                -> ecmwf_national_raw, nwp_cells_ecmwf
Requests are HEAD only (nothing is downloaded), User-Agent python-urllib.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, datetime as dt, email.utils, hashlib, importlib, json, os, shutil, time, urllib.error, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KST = dt.timezone(dt.timedelta(hours=9))
GROUP_NAMES = ("gfs", "gefsm", "gefss", "chem", "ecm12")


def groups():
    """group: (days before the target day of the run's start date, list of (forecast hour, url builder)), with the addresses and hours of the download scripts."""
    g13 = importlib.import_module("13_gfs_national"); g14 = importlib.import_module("14_gefs_national")
    g15 = importlib.import_module("15_aerosol_national"); g17 = importlib.import_module("17_ecmwf_national")
    return {
        "gfs":   (1, [(fh, lambda i, fh=fh: g13.url(i, fh)) for fh in g13.FHOURS]),
        "gefsm": (1, [(fh, lambda i, fh=fh: g14.url(i, "geavg", fh)) for fh in g14.FHOURS]),
        "gefss": (1, [(fh, lambda i, fh=fh: g14.url(i, "gespr", fh)) for fh in g14.FHOURS]),
        "chem":  (1, [(fh, lambda i, fh=fh: f"{g15.BASE}/gefs.{i:%Y%m%d}/00/chem/pgrb2ap25/gefs.chem.t00z.a2d_0p25.f{fh:03d}.grib2") for fh in g15.FHOURS]),
        "ecm12": (2, [(st, lambda i, st=st: g17.url(i, st, "grib2")) for st in g17.CYCLES[12]["steps"]]),
    }
FIRST_DAY = {"ecm12": "2024-03-01"}          # ECMWF open data archive starts 2024-02-28; ecmwf_national_raw starts at target day 2024-03-01
TABLES = {   # table file: groups whose files it reads
    "gfs_national_raw.parquet": ("gfs",), "gfsx_national_raw.parquet": ("gfs",), "nwp_cells_gfs.parquet": ("gfs",),
    "gefs_national_raw.parquet": ("gefsm", "gefss"), "nwp_cells_gefs.parquet": ("gefsm",),
    "aerosol_national_raw.parquet": ("chem",),
    "ecmwf_national_raw.parquet": ("ecm12",), "nwp_cells_ecmwf.parquet": ("ecm12",),
}
KEYS = ("date", "fh", "step")


def head(url: str, tries: int = 5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return "ok", email.utils.parsedate_to_datetime(r.headers["Last-Modified"])
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return "missing", None
            time.sleep(0.5 * (k + 1))
        except Exception:
            time.sleep(0.5 * (k + 1))
    return "error", None


def scan(a):
    tasks = []
    for grp, (lag, files) in groups().items():
        for d in pd.date_range(max(pd.Timestamp(a.start), pd.Timestamp(FIRST_DAY.get(grp, a.start))), a.end, freq="D"):
            init = (d - pd.Timedelta(days=lag)).date()
            for fh, fn in files:
                tasks.append((grp, d.date(), fh, fn(init)))
    print(f"{len(tasks):,} files to check ({a.start} .. {a.end})", flush=True)

    def one(t):
        grp, day, fh, u = t
        st, lm = head(u)
        cut = dt.datetime(day.year, day.month, day.day, 17, 0, tzinfo=KST) - dt.timedelta(days=1)
        return dict(group=grp, target_day=str(day), fh=fh, url=u, status=st,
                    posted_utc=(lm.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if lm else None),
                    margin_h=(round((cut - lm).total_seconds() / 3600.0, 4) if lm else np.nan))
    rows, t0 = [], time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(one, tasks)):
            rows.append(r)
            if (k + 1) % 5000 == 0:
                print(f"  {k + 1:,}/{len(tasks):,} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    err = df.status == "error"                                   # second, slower pass for transient errors
    if err.any():
        print(f"retrying {int(err.sum())} errors", flush=True)
        for i in df.index[err]:
            r = one((df.at[i, "group"], dt.date.fromisoformat(df.at[i, "target_day"]), df.at[i, "fh"], df.at[i, "url"]))
            for k, v in r.items():
                df.at[i, k] = v
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df):,} files, scanned {time.strftime('%Y-%m-%d %H:%M %Z')}")
    summarise(df)


def bad_days(df: pd.DataFrame) -> dict:
    """group -> sorted target days on which at least one file was posted at/after the cut-off or could not be verified."""
    bad = df[(df.status != "ok") | ~(df.margin_h > 0)]
    return {g: sorted(bad.loc[bad.group == g, "target_day"].unique()) for g in GROUP_NAMES}


def summarise(df: pd.DataFrame):
    b = bad_days(df)
    for g in GROUP_NAMES:
        x = df[df.group == g]
        d = x.groupby("target_day").margin_h.min()
        print(f"  {g:6s} {x.target_day.nunique():5d} days, {len(x):6,} files (missing {int((x.status == 'missing').sum())}, errors {int((x.status == 'error').sum())}); "
              f"day margin median {d.median():+.2f} h; left out {len(b[g])}: {', '.join(b[g]) if b[g] else '-'}")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def apply(a):
    df = pd.read_csv(a.scan, dtype={"target_day": str})
    summarise(df)
    bad = bad_days(df)
    os.makedirs(a.out, exist_ok=True)
    man = dict(rule="values of a table for target day D are left out unless every file it reads for D was posted (AWS Last-Modified) before 17:00 KST of D-1",
               scan=os.path.basename(a.scan), scan_md5=md5(a.scan), tables={})
    for f in sorted(os.listdir(a.src)):
        sp, op = os.path.join(a.src, f), os.path.join(a.out, f)
        if not os.path.isfile(sp):
            continue
        if f not in TABLES:
            shutil.copy2(sp, op)
            continue
        days = sorted(set().union(*[bad[g] for g in TABLES[f]]))
        t = pd.read_parquet(sp)
        m = t["date"].astype(str).str[:10].isin(days).to_numpy()
        for c in t.columns:
            if c in KEYS:
                continue
            if t[c].dtype == object:
                t[c] = t[c].astype(object); t.loc[m, c] = None
            else:
                t.loc[m, c] = np.nan
        t.to_parquet(op, index=False)
        hit = sorted(set(t.loc[m, "date"].astype(str).str[:10]))
        man["tables"][f] = dict(groups=list(TABLES[f]), days_left_out=hit, rows_emptied=int(m.sum()), md5_in=md5(sp), md5_out=md5(op))
        print(f"  {f:30s} {len(hit):3d} days left out ({int(m.sum())} rows): {', '.join(hit) if hit else '-'}")
    json.dump(man, open(os.path.join(a.out, "pubtime_rule_manifest.json"), "w"), indent=1)
    print(f"wrote {a.out} (+ pubtime_rule_manifest.json)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--out", default=os.path.join(HERE, "external_data", "pubtimes_scan.csv.gz"))
    s.add_argument("--start", default="2021-01-01")
    s.add_argument("--end", default="2026-05-31")
    s.add_argument("--workers", type=int, default=24)
    p = sub.add_parser("apply")
    p.add_argument("--scan", default=os.path.join(HERE, "external_data", "pubtimes_scan.csv.gz"))
    p.add_argument("--src", default=os.path.join(HERE, "external_data"))
    p.add_argument("--out", default=os.path.join(HERE, "external_data_pubrule"))
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else apply(a)


if __name__ == "__main__":
    main()
