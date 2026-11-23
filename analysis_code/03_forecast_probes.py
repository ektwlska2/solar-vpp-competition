"""Forecast probes on a time hold-out, scored with the competition metric (reference split, site-sum forecasts).

    python 03_forecast_probes.py --cut 2024-06-01 --exp M0 M3 S11 ...

Train on every row up to --cut (00:00 KST), predict the following 8,760 hours, and report W (penalty, 0.10 scale) for the
organisers' reference split on the sites with >= 90 % coverage in that year.  Results go to <CACHE>/results_<cut>.csv.
Predictions are saved as <CACHE>/pred_<cut>_<exp>.parquet so that 04/05 can reuse them.
"""
from __future__ import annotations
import argparse, os, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from oibc_lib import *

warnings.filterwarnings("ignore")

RAW = [f"{c}_r{r}" for c in A_COLS + B_COLS for r in (10, 17)]
BASE_A = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press"]       # organisers' baseline features
BASE_B = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob", "wb_total_liq", "wb_vis", "wb_wind_speed", "wb_dew_point"]
CTX_SUFFIX = ("_lag1", "_lead1", "_m3", "_m5", "_dmean", "_dmax", "_lag2", "_lead2", "_dmin", "_dstd", "_dam", "_dpm",
              "_lag3", "_lead3", "_lag4", "_lead4") + tuple(f"_h{h}" for h in range(9, 18))


def groups(cols):
    ctx = [c for c in cols if c.endswith(CTX_SUFFIX) and not c.startswith(("gfs_", "gefs_", "ecmwf_", "ecmwf18_", "nwp_", "rg_", "s1_", "jma_", "ldaps_"))]       # the national / regional NWP columns form their own group; stage-1 columns (s1_*, 08 --stage1) belong to no group and are added to the tree members explicitly
    fleet = [c for c in cols if c.startswith("fleet_")]
    blend = [c for c in cols if c.startswith("cloud_mean") or c.startswith("cloud_diff") or c.startswith("cloudA_r10") or c.startswith("cloudB_r10")]
    phy = [c for c in cols if c.startswith("ghi_")]
    dist = [c for c in ("wa_dist", "wb_dist", "cloud_idw_r17", "hum_idw_r17") if c in cols]
    hist = [c for c in cols if c.startswith("hist_")]
    nbr = [c for c in cols if c.startswith("nb_")]
    gfs = [c for c in cols if c.startswith(("gfs_", "gefs_", "aer_", "ecmwf_", "ecmwf18_", "nwp_", "rg_", "jma_", "ldaps_"))]      # jma_*: post-contest JMA GSM table (18 Nov 2026); before that fix its hourly columns were in no group
    return dict(raw=RAW, ctx=ctx, fleet=fleet, blend=blend, phy=phy, geo=GEO, dist=dist, hist=hist, nbr=nbr, gfs=gfs)


P_BASE = dict(n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
P_BIG = dict(n_estimators=800, learning_rate=0.05, num_leaves=255, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
P_HUGE = dict(n_estimators=1000, learning_rate=0.04, num_leaves=511, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
P_REG = dict(n_estimators=900, learning_rate=0.04, num_leaves=63, min_child_samples=400, subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0)


def fit_lgb(X, y, w=None, obj="l1", params=P_BASE, seed=0):
    m = lgb.LGBMRegressor(objective=obj, verbose=-1, n_jobs=10, random_state=seed, **params)
    return m.fit(X, y, sample_weight=w)


def fit_xgb(X, y, w=None, seed=0, params=None):
    import xgboost as xgb
    pr_ = dict(n_estimators=700, learning_rate=0.06, max_leaves=255, max_depth=0, grow_policy="lossguide", min_child_weight=100, subsample=0.8,
               colsample_bytree=0.8, max_bin=256, tree_method="hist", objective="reg:absoluteerror", n_jobs=10)
    pr_.update(params or {})
    return xgb.XGBRegressor(random_state=seed, **pr_).fit(X, y, sample_weight=w)


def load_panel(name="panel.parquet"):
    p = pd.read_parquet(f"{CACHE}/{name}")
    return p


def site_stats(tr):
    """Smooth, site-level descriptors learned from the training rows only (no site ID, no coordinates)."""
    q = tr[["site_code", "u", "hour", "wa_cloud_r17_dmean"]].dropna()
    p99 = q.groupby("site_code").u.quantile(0.99).rename("sp99")
    clear = q[q.wa_cloud_r17_dmean < 25]
    am = clear[clear.hour <= 12].groupby("site_code").u.mean()
    pm = clear[clear.hour >= 14].groupby("site_code").u.mean()
    asym = (am / pm).clip(0.3, 3.0).rename("sasym")
    return pd.concat([p99, asym], axis=1)


def drop_repeated_days(tr):
    """Remove the later copies of exactly repeated full-day profiles (stale-data artefacts, see 01b_review_checks.py)."""
    path = f"{CACHE}/repeated_days.csv"
    if not os.path.exists(path):
        return tr
    r = pd.read_csv(path, parse_dates=["dup"])
    bad = set(zip(r.site.astype(int), r.dup.dt.date))
    day = (pd.DatetimeIndex(tr.ts) - pd.Timedelta(hours=1)).tz_localize(None).normalize()
    key = list(zip(tr.site_code.astype(int).to_numpy(), day.date))
    keep = np.array([k not in bad for k in key])
    return tr[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="2024-06-01")
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--panel", default=None, help="override the panel file for every experiment")
    ap.add_argument("--tag", default="", help="suffix for the experiment name in results/pred files")
    ap.add_argument("--end", default=None, help="end of the scored window (default: cut + 1 year); use e.g. --cut 2025-03-01 --end 2025-06-01 for a short pseudo-test")
    ap.add_argument("--oof", action="store_true", help="write out-of-fold predictions for the training years instead of hold-out predictions")
    args = ap.parse_args()
    cut = K(f"{args.cut} 00:00")
    end = K(f"{args.end} 00:00") if args.end else cut + pd.DateOffset(years=1)
    smp = load_smp_train()
    sites = load_sites().set_index("site_code")
    cg = sites.current_group
    res_path = f"{CACHE}/results_{args.cut}.csv"
    results = pd.read_csv(res_path).set_index("exp") if os.path.exists(res_path) else pd.DataFrame()
    cache_panel = {}

    def get(name):
        if name not in cache_panel:
            cache_panel.clear()
            p = load_panel(name)
            full = p[["site_code", "ts", "energy", "cap", "elev"]].copy()
            # station-distance features (given in sites.csv) and inverse-distance blends of the two providers
            dd = sites[["wa_dist", "wb_dist"]].astype("float32")
            p = p.merge(dd, left_on="site_code", right_index=True, how="left")
            wa_, wb_ = 1.0 / np.maximum(p.wa_dist, 100.0), 1.0 / np.maximum(p.wb_dist, 100.0)
            p["cloud_idw_r17"] = ((p.wa_cloud_r17 * wa_ + p.wb_cloud_r17 * wb_) / (wa_ + wb_)).astype("float32")
            p["hum_idw_r17"] = ((p.wa_humidity_r17 * wa_ + p.wb_rel_hum_r17 * wb_) / (wa_ + wb_)).astype("float32")
            p = p[p.elev > 0].reset_index(drop=True)
            p["w"] = (hour_weights(pd.DatetimeIndex(p.ts), smp)).astype("float32")
            cache_panel[name] = (p, full)
        return cache_panel[name]

    for exp in args.exp:
        t0 = time.time()
        spec = EXPERIMENTS[exp]
        pname = args.panel or spec.get("panel", "panel.parquet")
        tagged = exp + args.tag
        p, full = get(pname)
        g = groups(p.columns)
        feats = spec["features"](g)
        tr = p[(p.ts <= cut) & p.u.notna()]
        if spec.get("site_cat"):
            cats = pd.CategoricalDtype(categories=sorted(p.site_code.unique()))
            p = p.assign(site_cat=p.site_code.astype(cats))
            tr = p[(p.ts <= cut) & p.u.notna()]
            va = p[(p.ts > cut) & (p.ts <= end)]
            feats = feats + ["site_cat"]
        if spec.get("statics"):
            ss = site_stats(tr)
            p = p.merge(ss, left_on="site_code", right_index=True, how="left")
            tr = p[(p.ts <= cut) & p.u.notna()]
            g = groups(p.columns)
            feats = spec["features"](g) + ["sp99", "sasym"]
        if spec.get("train_from"):
            tr = tr[tr.ts >= K(spec["train_from"])]
        if spec.get("drop_rows") is not None:
            tr = spec["drop_rows"](tr)
        va = p[(p.ts > cut) & (p.ts <= end)]
        if args.oof:
            # time-blocked out-of-fold predictions for every training row (7-day purge around each block)
            t_lo = K("2021-01-01 00:00")
            n_blocks = 3 if cut >= K("2024-01-01 00:00") else 2
            edges = [t_lo + (cut - t_lo) * i / n_blocks for i in range(n_blocks + 1)]
            purge = pd.Timedelta(days=7)
            outs = []
            for i in range(n_blocks):
                a_, b_ = edges[i], edges[i + 1]
                te = p[(p.ts > a_) & (p.ts <= b_)]
                trn = tr[~((tr.ts > a_ - purge) & (tr.ts <= b_ + purge))]
                wb_ = (trn.w / trn.w.mean()) if spec.get("weighted", True) else None
                ps = [fit_lgb(trn[feats], trn.u.clip(0, 1.2), wb_, spec.get("obj", "l1"), spec.get("params", P_BASE), sd).predict(te[feats]) for sd in spec.get("seeds", [0])]
                o = te[["site_code", "ts"]].copy()
                o["pred_u"] = np.clip(np.mean(ps, axis=0), 0, None).astype("float32")
                outs.append(o)
                print(f"  oof block {i}: train {len(trn):,} -> {len(te):,} rows", flush=True)
            pd.concat(outs).to_parquet(f"{CACHE}/oof_{args.cut}_{tagged}.parquet", index=False)
            print(f"{tagged}: saved oof_{args.cut}_{tagged}.parquet in {time.time()-t0:.0f}s", flush=True)
            continue
        y = tr.u.clip(0, 1.2)
        w = (tr.w / tr.w.mean()) if spec.get("weighted", True) else None
        csn_tr = csn_va = None
        if spec.get("target") == "cs":
            # model the clear-sky index kt = u / (GHI_clear/1000); weight by GHI_clear so the L1 loss still measures kWh
            csn_tr = np.maximum(tr.ghi_cs.to_numpy("float64") / 1000.0, 0.02)
            csn_va = np.maximum(va.ghi_cs.to_numpy("float64") / 1000.0, 0.02)
            y = pd.Series((tr.u.to_numpy("float64") / csn_tr).clip(0, 1.6), index=tr.index)
            w = pd.Series((tr.w.to_numpy("float64") * csn_tr), index=tr.index)
            w = w / w.mean()
        if spec.get("by_round"):                                  # organisers' baseline layout: one model per issue round
            preds = []
            for r in (10, 17):
                cols = [f"{c}_r{r}" for c in BASE_A + BASE_B] + GEO
                m = fit_lgb(tr[cols], y, None, spec["obj"], spec.get("params", P_BASE))
                preds.append(m.predict(va[cols]))
            pred = np.mean(preds, axis=0)
        else:
            preds = []
            if spec.get("weight_pow"):                    # emphasise the heavily weighted (midday, high-SMP) hours more
                w = w ** spec["weight_pow"]
                w = w / w.mean()
            if spec.get("cap_pow"):                       # align the loss with the metric: the penalty is in kWh of the group sum, so large sites matter more
                w = w * pd.Series(tr.cap.to_numpy("float64") ** spec["cap_pow"], index=w.index)
                w = w / w.mean()
            if spec.get("recency"):                       # half-life in years: older rows count less
                age = (cut - tr.ts).dt.total_seconds().to_numpy() / (365.25 * 86400)
                w = w * pd.Series(0.5 ** (age / spec["recency"]), index=w.index)
                w = w / w.mean()
            if spec.get("regime_w"):                      # up-weight rows from the newest data regime (e.g. provider B's rounds differ again from 2025-01)
                w = w * pd.Series(np.where(tr.ts > K(spec["regime_from"] + " 00:00"), spec["regime_w"], 1.0), index=w.index)
                w = w / w.mean()
            for seed in spec.get("seeds", [0]):
                if spec.get("model") == "xgb":
                    m = fit_xgb(tr[feats], y, w, seed, spec.get("xgb_params"))
                else:
                    m = fit_lgb(tr[feats], y, w, spec.get("obj", "l1"), spec.get("params", P_BASE), seed)
                preds.append(m.predict(va[feats]))
            pred = np.mean(preds, axis=0)
        if csn_va is not None:
            pred = pred * csn_va
        out = va[["site_code", "ts"]].copy()
        out["pred_u"] = np.clip(pred, 0, None).astype("float32")
        out.to_parquet(f"{CACHE}/pred_{args.cut}_{tagged}.parquet", index=False)
        d = error_matrix(out, full, cut, end)
        mask = (cg.reindex(d["sites"]) == "A").values
        row = dict(n_sites=len(d["sites"]), n_feat=len(feats), W_cur=W_of(d["E"], d["w"], mask), W_floor=W_floor(d["E"], d["w"]),
                   W_nopool=W_nopool(d["E"], d["w"]), seconds=round(time.time() - t0, 1))
        results.loc[tagged, list(row)] = list(row.values())
        if "M0" in results.index:
            results["score_vs_M0"] = 100 * (1 - results.W_cur / results.loc["M0", "W_cur"])
        results.index.name = "exp"
        results.to_csv(res_path)
        print(f"{tagged:18s} feats {row['n_feat']:3d}  W_cur {row['W_cur']/1e6:8.2f}M  floor {row['W_floor']/1e6:8.2f}M  "
              f"score {results.loc[tagged].get('score_vs_M0', float('nan')):+6.2f}  {row['seconds']:.0f}s", flush=True)


# --------------------------------------------------------------------------- experiment registry
def only(*keys):
    return lambda g: [c for k in keys for c in g[k]]


def without(fn, *drop):
    return lambda g: [c for c in fn(g) if c not in drop]


def nob10(fn):
    """Drop every feature built from provider B's round-10 forecast. In 2022-2024 B's two rounds are identical (88-92% of rows), from 2025-01 they differ again,
    so round 17 is the only B column whose meaning does not change between the training years and the scoring year."""
    return lambda g: [c for c in fn(g) if not (c.startswith("wb_") and "_r10" in c)]


EXPERIMENTS = {
    "M0": dict(by_round=True, obj="l2", train_from_years=2, weighted=False, features=only("geo"),
               train_from=None),   # train window handled below via drop_rows
    "M3": dict(features=only("raw", "geo")),
    # alignment tests on the base feature set
    "S11": dict(panel="panel_a1_b1.parquet", features=only("raw", "geo")),
    "S55": dict(panel="panel_a0.5_b0.5.parquet", features=only("raw", "geo")),
    "S10": dict(panel="panel_a1_b0.parquet", features=only("raw", "geo")),
    "S01": dict(panel="panel_a0_b1.parquet", features=only("raw", "geo")),
    # run-to-run noise of the identical model (LightGBM row/column subsampling)
    "M3_s1": dict(features=only("raw", "geo"), seeds=[1]),
    "M3_s2": dict(features=only("raw", "geo"), seeds=[2]),
    "M3_seeds": dict(features=only("raw", "geo"), seeds=[0, 1, 2]),
    "M3_L2": dict(features=only("raw", "geo"), obj="l2"),
    "M3_unw": dict(features=only("raw", "geo"), weighted=False),
    "M3_site": dict(features=only("raw", "geo"), site_cat=True),
    # feature families (run with --panel <best alignment>)
    "CTX": dict(features=only("raw", "ctx", "geo")),
    "AGG": dict(features=only("raw", "fleet", "blend", "phy", "geo")),
    "ALL": dict(features=only("raw", "ctx", "fleet", "blend", "phy", "geo")),
    "CTX_big": dict(features=only("raw", "ctx", "geo"), params=P_BIG),
    "CTX2_big": dict(panel="panel_ctx2.parquet", features=only("raw", "ctx", "geo"), params=P_BIG),
    "CTX_nocoord": dict(features=without(only("raw", "ctx", "geo"), "lat", "lon")),
    "CTX_nocoord_big": dict(features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG),
    "CTX2_nocoord": dict(panel="panel_ctx2.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon")),
    "ALL2_nocoord": dict(panel="panel_ctx2.parquet", features=without(only("raw", "ctx", "fleet", "blend", "phy", "geo"), "lat", "lon")),
    "CTX2_nocoord_big": dict(panel="panel_ctx2.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG),
    "CTX3_nocoord": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon")),
    "CTX3_nocoord_big": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG),
    # candidate final site model: extended context, no coordinates, default LightGBM settings
    "F": dict(panel="panel_ctx2.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon")),
    "F_seeds": dict(panel="panel_ctx2.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), seeds=[0, 1, 2]),
    # wide-context, high-capacity candidate
    "G": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG),
    "G_s1": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, seeds=[1]),
    "G_huge": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_HUGE),
    "G_seeds": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, seeds=[0, 1, 2]),
    "F_cs": dict(panel="panel_ctx2.parquet", features=without(only("raw", "ctx", "phy", "geo"), "lat", "lon"), target="cs"),
    "F_phy": dict(panel="panel_ctx2.parquet", features=without(only("raw", "ctx", "phy", "geo"), "lat", "lon")),
    # idea from pv_weather_feature_idea.md: distance-aware use of the given stations
    "F_dist": dict(panel="panel_ctx2.parquet", features=without(only("raw", "ctx", "geo", "dist"), "lat", "lon")),
    "F_dist_seeds": dict(panel="panel_ctx2.parquet", features=without(only("raw", "ctx", "geo", "dist"), "lat", "lon"), seeds=[0, 1, 2]),
    # previous-day regime (hist) and neighbour-station (nbr) features on top of the recommended model G
    "H_hist": dict(panel="panel_ctx4.parquet", features=without(only("raw", "ctx", "geo", "hist"), "lat", "lon"), params=P_BIG),
    "H_nbr": dict(panel="panel_ctx4.parquet", features=without(only("raw", "ctx", "geo", "nbr"), "lat", "lon"), params=P_BIG),
    "H_both": dict(panel="panel_ctx4.parquet", features=without(only("raw", "ctx", "geo", "hist", "nbr"), "lat", "lon"), params=P_BIG),
    "G_xgb": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), model="xgb"),
    "G_rec": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, recency=2.0),
    "G_unw": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, weighted=False),
    "H_xgb": dict(panel="panel_ctx4.parquet", features=without(only("raw", "ctx", "geo", "hist", "nbr"), "lat", "lon"), model="xgb"),
    "G_l2": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, obj="l2"),
    "G_extra": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=dict(P_BIG, extra_trees=True, n_estimators=1000)),
    "G_w15": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, weight_pow=1.5),
    "G_w2": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, weight_pow=2.0),
    # without provider B's round-10 columns only (31), pressure kept
    "G_nb": dict(panel="panel_ctx3.parquet", features=nob10(without(only("raw", "ctx", "geo"), "lat", "lon")), params=P_BIG, seeds=[0, 1]),
    # loss aligned with the metric: weight site rows by capacity^p (the penalty is on kWh of group sums)
    "G_cap1": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, cap_pow=1.0),
    "G_cap05": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, cap_pow=0.5),
    # regime tests for a short pseudo-test inside the newest data regime (--cut 2025-03-01 --end 2025-06-01): provider B's rounds differ again and its UV forecast is better from 2025-01
    "G_rw3": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, regime_from="2025-01-01", regime_w=3.0),
    "G_rw8": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, regime_from="2025-01-01", regime_w=8.0),
    "G_from2406": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, train_from="2024-06-01 01:00"),
    "G_from2312": dict(panel="panel_ctx3.parquet", features=without(only("raw", "ctx", "geo"), "lat", "lon"), params=P_BIG, train_from="2023-12-01 01:00"),
    "CTX2": dict(panel="panel_ctx2.parquet", features=only("raw", "ctx", "geo")),
    "CTX2_reg": dict(panel="panel_ctx2.parquet", features=only("raw", "ctx", "geo"), params=P_REG),
    "CTX2_phy": dict(panel="panel_ctx2.parquet", features=only("raw", "ctx", "phy", "blend", "geo")),
    "ALL_reg": dict(features=only("raw", "ctx", "fleet", "blend", "phy", "geo"), params=P_REG),
    "ALL_nocoord": dict(features=lambda g: [c for c in only("raw", "ctx", "fleet", "blend", "phy", "geo")(g) if c not in ("lat", "lon")]),
    "ALL_stat": dict(statics=True, features=lambda g: [c for c in only("raw", "ctx", "fleet", "blend", "phy", "geo")(g) if c not in ("lat", "lon", "cap")]),
    "ALL_clean": dict(features=only("raw", "ctx", "fleet", "blend", "phy", "geo"), drop_rows=lambda tr: drop_repeated_days(tr)),
    "ALL_seeds": dict(features=only("raw", "ctx", "fleet", "blend", "phy", "geo"), seeds=[0, 1, 2]),
}

if __name__ == "__main__":
    # M0 uses the two most recent years before the cut, like baseline.py (2023-06 -> 2025-05 for the real test)
    import sys
    cutarg = sys.argv[sys.argv.index("--cut") + 1] if "--cut" in sys.argv else "2024-06-01"
    EXPERIMENTS["M0"]["train_from"] = str((K(f"{cutarg} 00:00") - pd.DateOffset(years=2)).date()) + " 01:00"
    main()
