"""Re-blend the member forecasts of a finished build (site_forecasts.parquet of 08_make_submission.py) with a different family share / day-shrink, without retraining.

    python 21_reblend.py --src ../submissions/v6_regional_nn --out ../submissions/v6b --name V6b --share 0.30 --beta 0.10

Uses the same functions as the builder (blend_utils.py).  Only hold-out derived settings may be used here (never a leaderboard reading): see submissions/README.md.
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys
import numpy as np
import pandas as pd
from oibc_lib import *
from blend_utils import family_blend, day_shrink, offday_correction, CF_REF, OFFDAY

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+", help="one or more finished builds; with several, all their members are pooled (tree members with tree members, network members with network members)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--share", type=float, required=True, help="weight of the network family")
    ap.add_argument("--beta", type=float, required=True, help="day-level shrink strength (0 = none)")
    ap.add_argument("--trim", action="store_true", help="robust family blend: reduce the network share of days on which network and trees disagree strongly (blend_utils.family_blend)")
    ap.add_argument("--offday", action="store_true", help="off-day correction for plants > 500 kW (blend_utils.OFFDAY): weekends / holidays in spring and autumn")
    ap.add_argument("--nn-src", default=None, help="folder whose site_forecasts.parquet provides the network members (kwh_nn_*) instead of --src (refreshed networks of a finished build)")
    ap.add_argument("--trim-side", choices=["both", "below"], default="both", help="with --trim: 'below' lowers the network share only on days when the network family is below the trees (Test C, rule T)")
    ap.add_argument("--extra-nets", nargs="*", default=[], help="post-contest (18 Nov 2026): day-sequence network forecasts (parquet: site_id, ts, kwh_gru_*; post_contest/run_gru.py) added to the network family; hours they do not cover (night) count as 0")
    ap.add_argument("--month-factor", nargs="*", default=[], metavar="MONTH:FACTOR", help="calendar level factors fixed on hold-out winters (Test C, rule W), e.g. 12:0.980 1:0.955; the month is that of the target day")
    args = ap.parse_args()
    sf = None
    for k, src in enumerate(args.src):
        x = pd.read_parquet(os.path.join(src, "site_forecasts.parquet"))
        x["ts"] = pd.DatetimeIndex(x.ts).tz_convert(TZ)
        if len(args.src) > 1:
            x = x.rename(columns={c: f"{c}_s{k}" for c in x.columns if c.startswith("kwh_")}).drop(columns=["kwh"])
        sf = x if sf is None else sf.merge(x, on=["site_id", "ts"], how="inner")
        assert sf.shape[0] == x.shape[0], "the builds do not cover the same site-hours"
    args.src0 = args.src[0]
    if args.nn_src:
        nf = pd.read_parquet(os.path.join(args.nn_src, "site_forecasts.parquet"))
        nf["ts"] = pd.DatetimeIndex(nf.ts).tz_convert(TZ)
        ncols = [c for c in nf.columns if c.startswith("kwh_nn_")]
        assert ncols, "no kwh_nn_* members in --nn-src"
        sf = sf.drop(columns=[c for c in sf.columns if c.startswith("kwh_nn_")]).merge(nf[["site_id", "ts"] + ncols], on=["site_id", "ts"], how="left")
        assert sf[ncols].notna().all().all(), "network members do not cover every site-hour"
    for p in args.extra_nets:
        g = pd.read_parquet(p); g["ts"] = pd.DatetimeIndex(g.ts).tz_convert(TZ)
        gcols = [c for c in g.columns if c.startswith("kwh_gru_")]
        assert gcols and not set(gcols) & set(sf.columns), f"{p}: no new kwh_gru_* column"
        n0 = len(sf); sf = sf.merge(g[["site_id", "ts"] + gcols], on=["site_id", "ts"], how="left"); assert len(sf) == n0
        print(f"extra network member(s) {gcols} from {p}: {int(sf[gcols[0]].notna().sum()):,} of {n0:,} site-hours covered, the rest (night) set to 0")
        sf[gcols] = sf[gcols].fillna(0.0)
    sites = load_sites().set_index("site_id")
    cap = sf.site_id.map(sites.capacity_kw).to_numpy("float64")
    members = [c for c in sf.columns if c.startswith("kwh_")]
    tree = [sf[c].to_numpy("float64") / cap for c in members if c.startswith(("kwh_lgb_", "kwh_xgb_"))]
    nnm = [sf[c].to_numpy("float64") / cap for c in members if c.startswith(("kwh_nn_", "kwh_gru_"))]          # kwh_gru_*: only with --extra-nets
    assert tree and nnm, f"need tree and network members, found {members}"
    trim = dict(tol=0.05, hard=0.15, floor=0.15) if args.trim else None
    if trim is not None and args.trim_side != "both":
        trim["side"] = args.trim_side
    u = family_blend(tree, nnm, args.share, trim, sf.ts, cap)
    if args.beta:
        u, f_row, cf_ref = day_shrink(u, sf.ts, cap, float(sites.capacity_kw.sum()), args.beta, cf_ref=CF_REF)
        print(f"day-level shrink beta {args.beta}: reference cf {cf_ref:.4f} (fixed constant), factor range {f_row.min():.3f}..{f_row.max():.3f}")
    if args.offday:
        e0 = float((u * cap).sum())
        u, f_off = offday_correction(u, sf.ts, cap)
        print(f"off-day correction: {int((f_off < 1).sum()):,} site-hours reduced, annual energy {e0/1e6:.3f} -> {float((u * cap).sum())/1e6:.3f} GWh ({100*(float((u * cap).sum())/e0-1):+.3f} %), strongest factor {f_off.min():.3f}")
    month_factor = {int(s.split(":")[0]): float(s.split(":")[1]) for s in args.month_factor}
    if month_factor:
        assert all(1 <= m <= 12 and 0.90 <= f <= 1.0 for m, f in month_factor.items()), month_factor
        mon = (pd.DatetimeIndex(sf.ts) - pd.Timedelta(hours=1)).month.to_numpy()                 # month of the target day (KST; hour labels are right-closed)
        f_cal = pd.Series(mon).map(month_factor).fillna(1.0).to_numpy()
        e0 = float((u * cap).sum()); u = u * f_cal
        print(f"calendar level factors {month_factor}: {int((f_cal != 1.0).sum()):,} site-hours scaled, annual energy {e0/1e6:.3f} -> {float((u * cap).sum())/1e6:.3f} GWh")
    sf = sf.assign(kwh=u * cap)
    comp = pd.read_csv(glob.glob(os.path.join(args.src0, "*_composition.csv"))[0])
    for src in args.src[1:]:
        assert comp.equals(pd.read_csv(glob.glob(os.path.join(src, "*_composition.csv"))[0])), "the builds have different compositions"
    fc = sf.merge(comp, on="site_id").groupby(["group", "ts"], as_index=False).kwh.sum().rename(columns={"kwh": "forecast_kwh"}).sort_values(["group", "ts"]).reset_index(drop=True)
    os.makedirs(args.out, exist_ok=True)
    comp_path, fc_path = os.path.join(args.out, f"{args.name}_composition.csv"), os.path.join(args.out, f"{args.name}_forecast.csv")
    comp.to_csv(comp_path, index=False)
    o = fc.copy(); o["ts"] = o.ts.dt.strftime("%Y-%m-%d %H:%M:%S") + "+09:00"
    o[["group", "ts", "forecast_kwh"]].to_csv(fc_path, index=False, float_format="%.4f")
    sf.to_parquet(os.path.join(args.out, "site_forecasts.parquet"), index=False)
    chk = pd.read_csv(fc_path)
    assert len(chk) == 17520 and chk.forecast_kwh.notna().all() and (chk.forecast_kwh >= 0).all()
    info = json.load(open(os.path.join(args.src0, "run_info.json"))) if os.path.exists(os.path.join(args.src0, "run_info.json")) else {}
    info.update(reblended_from=[os.path.abspath(p) for p in args.src], extra_network_members=([os.path.abspath(p) for p in args.extra_nets] or None), network_members_from=(os.path.abspath(args.nn_src) if args.nn_src else None), family_share=args.share, slope_beta=args.beta, trim=trim,
                day_shrink_reference=CF_REF, offday_correction=(OFFDAY if args.offday else None), month_factor=(month_factor or None))
    json.dump(info, open(os.path.join(args.out, "run_info.json"), "w"), indent=1)
    r = subprocess.run([sys.executable, os.path.join(HERE, "organiser_tools", "validate_submission.py"), "--composition", comp_path, "--forecast", fc_path, "--data", DATA], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    print(f"wrote {comp_path} and {fc_path}; annual energy " + ", ".join(f"{k} {v/1e6:.2f} GWh" for k, v in fc.groupby("group").forecast_kwh.sum().items()))
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
