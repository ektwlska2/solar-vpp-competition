"""Quality checks for a submission folder (composition.csv + forecast.csv) beyond the organisers' format validator.

    python 09_check_submission.py --dir ../submission_v1 [--baseline-dir <folder with baseline.py output>] [--negative-tests]

Checks
  1. organisers' validator (exit code 0)
  2. no forecast while the sun is below the horizon; group forecast never above group capacity
  3. shape against the last full training year: diurnal profile and month-by-month yield for the same sites
  4. (optional) agreement with the organisers' baseline.py forecast: hourly correlation, relative difference, annual totals
  5. (optional) deliberately broken copies of the files must be REJECTED by the validator
There is no scoring-year actual, so these are plausibility checks, not an accuracy estimate.
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, tempfile
import numpy as np
import pandas as pd
from oibc_lib import *

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATOR = os.path.join(HERE_DIR, "organiser_tools", "validate_submission.py")
lines = []


def say(msg=""):
    print(msg, flush=True)
    lines.append(msg)


def run_validator(comp, fc):
    r = subprocess.run([sys.executable, VALIDATOR, "--composition", comp, "--forecast", fc, "--data", DATA], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def find(dir_, kind):
    """<Name>_composition.csv / <Name>_forecast.csv (or the plain composition.csv / forecast.csv) inside a submission folder."""
    plain = os.path.join(dir_, f"{kind}.csv")
    if os.path.exists(plain):
        return plain
    hits = sorted(f for f in os.listdir(dir_) if f.endswith(f"_{kind}.csv"))
    assert len(hits) == 1, f"expected exactly one *_{kind}.csv in {dir_}, found {hits}"
    return os.path.join(dir_, hits[0])


def load(dir_):
    comp = pd.read_csv(find(dir_, "composition"))
    fc = pd.read_csv(find(dir_, "forecast"))
    fc["ts"] = pd.to_datetime(fc.ts, utc=True).dt.tz_convert(TZ)
    return comp, fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--negative-tests", action="store_true")
    args = ap.parse_args()
    comp_p, fc_p = find(args.dir, "composition"), find(args.dir, "forecast")
    comp, fc = load(args.dir)
    sites = load_sites()
    cap = sites.set_index("site_id").capacity_kw
    gcap = comp.merge(sites[["site_id", "capacity_kw"]], on="site_id").groupby("group").capacity_kw.sum()
    say(f"== {args.dir}")
    code, out = run_validator(comp_p, fc_p)
    say(f"1. organisers' validator exit code {code}\n   " + out.replace("\n", "\n   "))
    ok = code == 0

    # 2. night hours and capacity
    h = fc.ts.dt.hour
    night = fc[(h <= 4) | (h >= 22)]
    say(f"2. forecast at hour labels 22-04: sum {night.forecast_kwh.sum():.3f} kWh (must be 0); maximum per hour label 5 / 21: "
        f"{fc[h == 5].forecast_kwh.max():.1f} / {fc[h == 21].forecast_kwh.max():.1f} kWh")
    ok &= night.forecast_kwh.sum() == 0
    peak = fc.groupby("group").forecast_kwh.max()
    for g in ("G1", "G2"):
        say(f"   {g}: capacity {gcap[g]:,.1f} kW, peak hourly forecast {peak[g]:,.1f} kWh ({peak[g]/gcap[g]:.1%} of capacity), annual {fc[fc.group==g].forecast_kwh.sum()/1e6:.2f} GWh")
        ok &= peak[g] <= gcap[g]

    # 3. shape against the last full training year (same sites)
    gen = pd.read_parquet(f"{DATA}/train/generation.parquet")
    last = gen[(gen.ts > K("2024-06-01 00:00")) & (gen.ts <= K("2025-06-01 00:00"))]
    cov = last.groupby("site_id").size() / 8760
    keep = cov[cov >= 0.9].index
    last = last[last.site_id.isin(keep)].merge(comp, on="site_id")
    # month-by-month yield ratio forecast / last year's actual, restricted to the groups' full-year sites (share of capacity differs, so compare kWh per kW)
    act_m = last.groupby(last.ts.dt.month).energy.sum() / cap.reindex(keep).sum()
    fc_m = fc.groupby(fc.ts.dt.month).forecast_kwh.sum() / cap.sum()
    say("3. month-by-month yield, forecast scoring year vs last training year (kWh per kW of the respective site set; weather differs by year):")
    say("   month  " + " ".join(f"{m:>6d}" for m in range(1, 13)))
    say("   last   " + " ".join(f"{act_m.get(m, np.nan):6.1f}" for m in range(1, 13)))
    say("   fcst   " + " ".join(f"{fc_m.get(m, np.nan):6.1f}" for m in range(1, 13)))
    prof_a = last.groupby(last.ts.dt.hour).energy.sum()
    prof_a = prof_a / prof_a.sum()
    prof_f = fc.groupby(fc.ts.dt.hour).forecast_kwh.sum()
    prof_f = prof_f / prof_f.sum()
    say(f"   diurnal profile (share of daily energy), max abs difference between forecast and last year: {np.abs(prof_a.reindex(range(24)).fillna(0) - prof_f.reindex(range(24)).fillna(0)).max():.3f}; "
        f"peak hour label forecast {int(prof_f.idxmax())} vs last year {int(prof_a.idxmax())}")

    # 4. agreement with baseline.py output
    if args.baseline_dir:
        bcomp, bfc = load(args.baseline_dir)
        a = fc.set_index(["group", "ts"]).forecast_kwh
        b = bfc.set_index(["group", "ts"]).forecast_kwh.reindex(a.index)
        tot_a, tot_b = a.groupby(level=0).sum(), b.groupby(level=0).sum()
        s_a = a.groupby(level="ts").sum()
        s_b = b.groupby(level="ts").sum()
        day = s_b > 0
        say(f"4. fleet-total hourly forecast vs organisers' baseline.py output: correlation {np.corrcoef(s_a[day], s_b[day])[0, 1]:.4f}, "
            f"annual total ratio {s_a.sum() / s_b.sum():.3f}, mean |difference| {np.abs(s_a - s_b).sum() / s_b.sum():.1%} of baseline energy")
        lagc = {k: float(np.corrcoef(s_a.shift(k).fillna(0)[day], s_b[day])[0, 1]) for k in (-2, -1, 0, 1, 2)}
        say("   hourly correlation with the baseline when our series is shifted by k hours (a time-label error would peak away from k = 0): "
            + ", ".join(f"{k:+d}: {v:.3f}" for k, v in lagc.items()))
        ok &= max(lagc, key=lagc.get) == 0
        mm = (s_a.groupby(s_a.index.month).sum() / s_b.groupby(s_b.index.month).sum()).round(3)
        say("   monthly ratio (ours / baseline): " + " ".join(f"{v:.2f}" for v in mm.values))
        diff = comp.merge(bcomp, on="site_id", suffixes=("", "_base"))
        agree = max((diff.group == diff.group_base).mean(), (diff.group != diff.group_base).mean())
        say(f"   composition: {agree:.0%} of sites sit on the same side as in the baseline's reference split (50% = unrelated)")

    # 5. negative tests
    if args.negative_tests:
        say("5. the validator must reject deliberately broken copies:")
        tmp = tempfile.mkdtemp()
        tests = {}
        c, f = comp.copy(), pd.read_csv(fc_p)
        tests["drop one site"] = (c.iloc[1:], f)
        c2 = comp.copy()
        c2.loc[:, "group"] = "G1"
        tests["all sites in G1 (G2 below 20 MW)"] = (c2, f)
        f2 = f.copy()
        f2["ts"] = f2.ts.str.replace("+09:00", "", regex=False)
        tests["timestamps without timezone"] = (comp, f2)
        f3 = f.copy()
        f3.loc[10, "forecast_kwh"] = -1.0
        tests["negative forecast"] = (comp, f3)
        f4 = f.copy()
        f4.loc[11, "forecast_kwh"] = np.nan
        tests["NaN forecast"] = (comp, f4)
        tests["missing hour"] = (comp, f.iloc[1:])
        for name, (cc, ff) in tests.items():
            cp, fp = os.path.join(tmp, "c.csv"), os.path.join(tmp, "f.csv")
            cc.to_csv(cp, index=False)
            ff.to_csv(fp, index=False)
            code, _ = run_validator(cp, fp)
            say(f"   {name:36s} -> exit code {code} ({'rejected' if code != 0 else 'NOT REJECTED'})")
            ok &= code != 0
        shutil.rmtree(tmp)
    say("")
    say("RESULT: " + ("all checks passed" if ok else "SOMETHING FAILED, see above"))
    open(os.path.join(args.dir, "QA_report.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
