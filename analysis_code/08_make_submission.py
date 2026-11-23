"""Final pipeline: raw competition files -> <Name>_composition.csv + <Name>_forecast.csv (+ site-level forecasts for reuse).

    python 08_make_submission.py --out ../submissions/v1_structural          # train on all data, forecast the scoring year
    python 08_make_submission.py --mode holdout --cut 2024-06-01 --seeds 1    # regression test: same code path on a hold-out year
    python 08_make_submission.py --out ../submissions/v1_reference_split --composition reference \
        --reuse-forecasts ../submissions/v1_structural/site_forecasts.parquet             # same forecasts, organisers' reference split

What it does (see report section 6):
  1. forecast tables of provider A and B (rounds 10 and 17); rows missing in one round are filled from the other round
  2. station-level context features for the whole target day (neighbouring hours, day shape, hourly trajectory) -- all published
     before the 17:00 cut-off, computed per station series and merged to the sites; no coordinates, no observations, no SMP input
  3. LightGBM (L1 loss, SMP*k sample weights from the TRAINING SMP, 255 leaves, 800 trees), seeds averaged, utilisation x capacity,
     0 when the sun is below the horizon
  4. composition: sister sites alternated, remaining sites spread inside their WA station (needs sites.csv only)
  5. group forecasts = sum of the site forecasts; both files are checked with the organisers' validator
Nothing from the scoring year except the day-ahead forecast files is used.
"""
from __future__ import annotations
import argparse, importlib, json, os, subprocess, sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from oibc_lib import *
from blend_utils import family_blend, day_shrink, CF_REF

warnings.filterwarnings("ignore")
bp = importlib.import_module("02_build_panel")
pr = importlib.import_module("03_forecast_probes")

T0 = time.time()
SEED_OFFSET = int(os.environ.get("OIBC_SEED_OFFSET", "0"))   # post-contest noise control (18 Nov 2026): every seed + this offset; 0 = the original seeds (V13)
log = lambda m: print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
HERE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS = dict(n_estimators=800, learning_rate=0.05, num_leaves=255, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
XGB_PARAMS_NOTE = "pr.fit_xgb defaults: 700 trees, rate 0.06, max_leaves 255 (lossguide), min_child_weight 100, subsample/colsample 0.8, 256 bins, reg:absoluteerror"
FEATURE_RULE = pr.without(pr.only("raw", "ctx", "geo"), "lat", "lon")        # same rule as experiment G in 03_forecast_probes.py
DAY_HOURS = (5, 21)       # hour labels kept when building frames (sun is above the horizon only inside this range)
FEATURE_SETS = {
    "G": FEATURE_RULE,                                                                      # raw + same-day context + geometry (264)
    "H_hist": pr.without(pr.only("raw", "ctx", "geo", "hist"), "lat", "lon"),                # + previous-day regime features (300)
    "H_both": pr.without(pr.only("raw", "ctx", "geo", "hist", "nbr"), "lat", "lon"),         # + neighbour-station features (318)
    # the same sets + the national NOAA GFS / GEFS summaries (same value for every plant, 00 UTC cycle of the previous day; 13_gfs_national.py, 14_gefs_national.py)
    "G_nwp": lambda g: FEATURE_RULE(g) + g["gfs"],
    "H_hist_nwp": lambda g: pr.without(pr.only("raw", "ctx", "geo", "hist"), "lat", "lon")(g) + g["gfs"],
    "H_both_nwp": lambda g: pr.without(pr.only("raw", "ctx", "geo", "hist", "nbr"), "lat", "lon")(g) + g["gfs"],
    # network inputs: the same columns without the ECMWF and consensus columns (those exist only from 2024-03, a missing-value flag that shifts between training and scoring years
    # makes the network collapse: hold-out year 1 penalty 168.8M -> 229.5M; trees treat missing values as a learned branch and are not affected)
    # v7 network inputs: the ECMWF clear-sky-index columns are kept, but where ECMWF does not exist yet (before 2024-03) they are filled with the matching GEFS (else GFS) column of the same day
    # (fit_recipe, impute_ecmwf), so no missing-value flag shifts between the training years and the scoring year; hold-out (leave-one-month-out, Jan-May 2025): network alone -2.7%, blend -1.85% vs -1.15%
    "G_nwp_nn2": lambda g: FEATURE_RULE(g) + [c for c in g["gfs"] if not c.startswith(("ecmwf18_", "nwp_", "rg_ecmwf", "ldaps_")) and not (c.startswith("ecmwf_") and c.endswith(("_dsw", "_dsw_dsum")))],
    "H_hist_nwp_nn2": lambda g: pr.without(pr.only("raw", "ctx", "geo", "hist"), "lat", "lon")(g) + [c for c in g["gfs"] if not c.startswith(("ecmwf18_", "nwp_", "rg_ecmwf", "ldaps_")) and not (c.startswith("ecmwf_") and c.endswith(("_dsw", "_dsw_dsum")))],
    "G_nwp_nn": lambda g: FEATURE_RULE(g) + [c for c in g["gfs"] if not c.startswith(("ecmwf_", "ecmwf18_", "nwp_", "rg_ecmwf", "ldaps_"))],
    "H_hist_nwp_nn": lambda g: pr.without(pr.only("raw", "ctx", "geo", "hist"), "lat", "lon")(g) + [c for c in g["gfs"] if not c.startswith(("ecmwf_", "ecmwf18_", "nwp_", "rg_ecmwf", "ldaps_"))],
}
# (name, model family, feature set, seeds)
RECIPES = {
    "v1": [("lgb_G", "lgb", "G", 3)],
    "v2": [("lgb_Hhist", "lgb", "H_hist", 2), ("lgb_Hboth", "lgb", "H_both", 1), ("xgb_G", "xgb", "G", 1), ("xgb_Hboth", "xgb", "H_both", 1)],
    # v3 = v2 + national NWP summaries (GFS, GEFS, GEFS-Aerosols) as features of every member; v4 = v3 + the ECMWF IFS irradiance (same members, one more table)
    "v3": [("lgb_Hhist", "lgb", "H_hist_nwp", 2), ("lgb_Hboth", "lgb", "H_both_nwp", 1), ("xgb_G", "xgb", "G_nwp", 1), ("xgb_Hboth", "xgb", "H_both_nwp", 1)],
    "v4": [("lgb_Hhist", "lgb", "H_hist_nwp", 2), ("lgb_Hboth", "lgb", "H_both_nwp", 1), ("xgb_G", "xgb", "G_nwp", 1), ("xgb_Hboth", "xgb", "H_both_nwp", 1)],
    # v5 = v4 + multi-model consensus features, feature subsampling 0.4 in every member, rows of the newest data regime (from 2025-01-01) weighted x4, more seeds
    #      (the 18 UTC ECMWF cycle was tested and dropped: no gain)
    "v5": [("lgb_Hhist", "lgb", "H_hist_nwp", 3), ("lgb_Hboth", "lgb", "H_both_nwp", 2), ("xgb_G", "xgb", "G_nwp", 2), ("xgb_Hboth", "xgb", "H_both_nwp", 2)],
    # v6 = v5 + regional NWP summaries (fixed 1-degree cell grid, PCA + non-negative fleet weights fitted on training hours; regional_features.py) in every member
    #      + neural-network members (nn_member.py) whose family is blended with the tree family at a fixed share + a day-level shrink of the forecast towards the seasonal mean
    "v6": [("lgb_Hhist", "lgb", "H_hist_nwp", 3), ("lgb_Hboth", "lgb", "H_both_nwp", 2), ("xgb_G", "xgb", "G_nwp", 2), ("xgb_Hboth", "xgb", "H_both_nwp", 2),
           ("nn_G", "nn", "G_nwp_nn", 5), ("nn_Hhist", "nn", "H_hist_nwp_nn", 5)],
    "v6nn": [("nn_G", "nn", "G_nwp_nn", 5), ("nn_Hhist", "nn", "H_hist_nwp_nn", 5)],              # test only: the network members of v6 on their own
}
NWP_SOURCES = {"v3": ("gfs", "gefs", "aer"), "v4": ("gfs", "gefs", "aer", "ecmwf"), "v5": ("gfs", "gefs", "aer", "ecmwf"), "v6": ("gfs", "gefs", "aer", "ecmwf"), "v6nn": ("gfs", "gefs", "aer")}
CONSENSUS = {"v5", "v6"}                                                                # recipes that add the multi-model consensus features
RECIPE_PARAMS = {"v5": dict(lgb=dict(colsample_bytree=0.4), xgb=dict(colsample_bytree=0.4)), "v6": dict(lgb=dict(colsample_bytree=0.4), xgb=dict(colsample_bytree=0.4)), "v6nn": dict()}
REGIME = {"v5": ("2025-01-01 00:00", 4.0), "v6": ("2025-01-01 00:00", 1.0), "v6nn": ("2025-01-01 00:00", 4.0)}      # (start of the newest data regime, weight multiplier): provider B's rounds differ again and A/B agree closely from 2025-01. v6 (copied by v7/v8/v9/v9nn) uses 1.0 since 2026-11-10 (submissions V12/V13): the x4 weight had never been exercised by a hold-out and lost its first test
REGIONAL = {"v6": dict(sources=("gefs",), n_pc=4), "v6nn": dict(sources=("gefs",), n_pc=4)}                                # regional block: cell files nwp_cells_<source>.parquet in --nwp-dir
FAMILY_SHARE = {"v6": 0.35, "v6nn": 1.0}                                                        # weight of the neural-network family in the blend (fixed from hold-out windows, never from the leaderboard)
SLOPE = {"v6": 0.10, "v6nn": None}                                                              # day-level shrink: forecast x exp(-beta (cf_day - mean cf)), cf = predicted midday capacity factor
# v7 = v6 + the ECMWF IFS cells in the regional block (rg_ecmwf_* columns for the tree members only; the network inputs exclude them like every ECMWF column)
RECIPES["v7"] = RECIPES["v6"][:4] + [("nn_G", "nn", "G_nwp_nn2", 5), ("nn_Hhist", "nn", "H_hist_nwp_nn2", 5)]
RECIPES["v7nn"] = [("nn_G", "nn", "G_nwp_nn2", 5), ("nn_Hhist", "nn", "H_hist_nwp_nn2", 5)]             # the network members of v7 on their own (same inputs; used to refresh the networks of a finished v7 build)
NWP_SOURCES["v7"] = NWP_SOURCES["v6"]; CONSENSUS.add("v7"); RECIPE_PARAMS["v7"] = RECIPE_PARAMS["v6"]; REGIME["v7"] = REGIME["v6"]
REGIONAL["v7"] = dict(sources=("gefs", "ecmwf"), n_pc=4); FAMILY_SHARE["v7"] = 0.30; SLOPE["v7"] = SLOPE["v6"]       # share 0.30: optimum of the hold-out windows (0.2-0.4)
# v8 = v6 + principal components of the GEFS ensemble-mean cloud-cover cell field (4 columns, fitted on training hours): leave-one-month-out inside Jan-May 2025, trees 66.3M -> 65.1M (-1.9%)
RECIPES["v8"] = RECIPES["v6"]
NWP_SOURCES["v8"] = NWP_SOURCES["v6"]; CONSENSUS.add("v8"); RECIPE_PARAMS["v8"] = RECIPE_PARAMS["v6"]; REGIME["v8"] = REGIME["v6"]
REGIONAL["v8"] = dict(sources=("gefs",), n_pc=4, cloud_sources=("gefs",)); FAMILY_SHARE["v8"] = FAMILY_SHARE["v6"]; SLOPE["v8"] = SLOPE["v6"]
# v6obs / v8obs (V18, 2026-11-13) = v6 / v8 + the national OBSERVED irradiance of NASA POWER (24_power_national.py, source 'obs'; columns gfs_obs_* for every tree and network
# member).  Basis: the organisers' written answer to the user's inquiry (추가된 안내사항/organisers_answer_observed_weather_and_rerun.md), which allows evaluation-period
# reanalysis / "historical weather" values as inputs when they are a wide-area average applied identically to every plant.  v6 and v8 themselves are unchanged.
for _b in ("v6", "v8"):
    _k = _b + "obs"
    RECIPES[_k] = RECIPES[_b]; NWP_SOURCES[_k] = NWP_SOURCES[_b] + ("obs",); CONSENSUS.add(_k); RECIPE_PARAMS[_k] = RECIPE_PARAMS[_b]; REGIME[_k] = REGIME[_b]
    REGIONAL[_k] = REGIONAL[_b]; FAMILY_SHARE[_k] = FAMILY_SHARE[_b]; SLOPE[_k] = SLOPE[_b]
# v6x (post-contest study, 18 Nov 2026) = v6 + 12 more fields of the SAME GFS run as the gfs table (cloud layers, cloud water, precipitable water, visibility, sunshine duration,
# CAPE, humidity at 850 / 700 hPa, precipitation; 26_gfs_extra_national.py, columns gfs_x_* for every tree and network member).  v6 itself is unchanged.
RECIPES["v6x"] = RECIPES["v6"]; NWP_SOURCES["v6x"] = NWP_SOURCES["v6"] + ("gfsx",); CONSENSUS.add("v6x"); RECIPE_PARAMS["v6x"] = RECIPE_PARAMS["v6"]; REGIME["v6x"] = REGIME["v6"]
REGIONAL["v6x"] = REGIONAL["v6"]; FAMILY_SHARE["v6x"] = FAMILY_SHARE["v6"]; SLOPE["v6x"] = SLOPE["v6"]
RECIPES["v6j"] = RECIPES["v6"]; NWP_SOURCES["v6j"] = NWP_SOURCES["v6"] + ("jma",); CONSENSUS.add("v6j"); RECIPE_PARAMS["v6j"] = RECIPE_PARAMS["v6"]; REGIME["v6j"] = REGIME["v6"]   # post-contest: + JMA GSM (xHigh's third source)
REGIONAL["v6j"] = REGIONAL["v6"]; FAMILY_SHARE["v6j"] = FAMILY_SHARE["v6"]; SLOPE["v6j"] = SLOPE["v6"]
RECIPES["v6xj"] = RECIPES["v6"]; NWP_SOURCES["v6xj"] = NWP_SOURCES["v6"] + ("gfsx", "jma"); CONSENSUS.add("v6xj"); RECIPE_PARAMS["v6xj"] = RECIPE_PARAMS["v6"]; REGIME["v6xj"] = REGIME["v6"]   # post-contest: GFS extras + JMA
REGIONAL["v6xj"] = REGIONAL["v6"]; FAMILY_SHARE["v6xj"] = FAMILY_SHARE["v6"]; SLOPE["v6xj"] = SLOPE["v6"]
RECIPES["v6xl"] = RECIPES["v6"]; NWP_SOURCES["v6xl"] = NWP_SOURCES["v6"] + ("gfsx", "ldaps"); CONSENSUS.add("v6xl"); RECIPE_PARAMS["v6xl"] = RECIPE_PARAMS["v6"]; REGIME["v6xl"] = REGIME["v6"]   # post-contest: GFS extras + KMA UM-LDAPS (trees only)
REGIONAL["v6xl"] = REGIONAL["v6"]; FAMILY_SHARE["v6xl"] = FAMILY_SHARE["v6"]; SLOPE["v6xl"] = SLOPE["v6"]
# v9 = v6 trees + networks trained on the contest's group-level penalty (kind "nnm"), blended at share 0.5 (hold-out Jan-May 2025: plain network blend -3.6%, group-loss network blend -6.1%)
RECIPES["v9"] = RECIPES["v6"][:4] + [("nn_Gm", "nnm", "G_nwp_nn", 5), ("nn_Hhistm", "nnm", "H_hist_nwp_nn", 5)]
RECIPES["v9nn"] = [("nn_Gm", "nnm", "G_nwp_nn", 5), ("nn_Hhistm", "nnm", "H_hist_nwp_nn", 5)]
NWP_SOURCES["v9"] = NWP_SOURCES["v6"]; CONSENSUS.add("v9"); RECIPE_PARAMS["v9"] = RECIPE_PARAMS["v6"]; REGIME["v9"] = REGIME["v6"]
REGIONAL["v9"] = REGIONAL["v6"]; FAMILY_SHARE["v9"] = 0.5; SLOPE["v9"] = SLOPE["v6"]
NWP_SOURCES["v9nn"] = NWP_SOURCES["v6"]; CONSENSUS.add("v9nn"); RECIPE_PARAMS["v9nn"] = {}; REGIME["v9nn"] = REGIME["v6"]; REGIONAL["v9nn"] = REGIONAL["v6"]; FAMILY_SHARE["v9nn"] = 1.0; SLOPE["v9nn"] = None
# v6nn0 = the networks of v6 trained WITHOUT the x4 weight on the newest-regime rows (no REGIME entry): test of whether the weight makes the network disagree with the trees in the scoring year
RECIPES["v6nn0"] = RECIPES["v6nn"]
NWP_SOURCES["v6nn0"] = NWP_SOURCES["v6nn"]; RECIPE_PARAMS["v6nn0"] = {}; REGIONAL["v6nn0"] = REGIONAL["v6nn"]; FAMILY_SHARE["v6nn0"] = 1.0; SLOPE["v6nn0"] = None
for _k in ("v7nn",):
    NWP_SOURCES[_k] = NWP_SOURCES["v7"]; CONSENSUS.add(_k); RECIPE_PARAMS[_k] = {}; REGIME[_k] = REGIME["v7"]; REGIONAL[_k] = REGIONAL["v7"]; FAMILY_SHARE[_k] = 1.0; SLOPE[_k] = None


# ------------------------------------------------------------------------------------------------ forecast tables
def fill_other_round(f: pd.DataFrame, st: str, cols: list) -> pd.DataFrame:
    """Where a (station, ts) row or value is missing in round r, take it from the other round (both rounds were published before the cut-off)."""
    parts = {r: f[f["round"] == r].set_index([st, "ts"])[cols] for r in (10, 17)}
    idx = parts[10].index.union(parts[17].index)
    out = []
    for r, o in ((10, 17), (17, 10)):
        x = parts[r].reindex(idx).combine_first(parts[o].reindex(idx)).reset_index()
        x.insert(1, "round", r)
        out.append(x[[st, "round", "ts"] + cols])
    return pd.concat(out, ignore_index=True)


def load_forecasts(split: str):
    fa = pd.read_parquet(f"{DATA}/{split}/forecast_wa.parquet", columns=["wa_station", "round", "ts"] + A_COLS)
    fb = pd.read_parquet(f"{DATA}/{split}/forecast_wb.parquet", columns=["wb_station", "round", "ts"] + B_COLS)
    return fa, fb


def prepare_tables(fa, fb, lo=None, hi=None):
    """Window (lo, hi], then fill gaps from the other round."""
    if lo is not None:
        fa, fb = fa[(fa.ts > lo) & (fa.ts <= hi)], fb[(fb.ts > lo) & (fb.ts <= hi)]
    n_a, n_b = len(fa), len(fb)
    fa, fb = fill_other_round(fa, "wa_station", A_COLS), fill_other_round(fb, "wb_station", B_COLS)
    log(f"gap fill: A rows {n_a:,} -> {len(fa):,}, B rows {n_b:,} -> {len(fb):,}")
    return fa, fb


# ------------------------------------------------------------------------------------------------ composition
def build_composition(sites: pd.DataFrame, kind: str):
    cap = sites.capacity_kw.to_numpy("float64")
    if kind == "reference":
        mask = (sites.current_group == "A").to_numpy()
    else:
        mask = keyed_twin(sites.reset_index(drop=True), cap, "wa_station")
    c1, c2 = cap[mask].sum(), cap[~mask].sum()
    assert c1 >= 20000 and c2 >= 20000, f"capacity band violated: G1 {c1:,.1f} kW, G2 {c2:,.1f} kW"
    comp = pd.DataFrame({"site_id": sites.site_id.to_numpy(), "group": np.where(mask, "G1", "G2")})
    return comp, mask, (float(c1), float(c2))


# ------------------------------------------------------------------------------------------------ model
def impute_ecmwf(df: pd.DataFrame, feats: list) -> np.ndarray:
    """feature matrix for the networks: every `ecmwf_<q>` column that is missing is filled with `gefs_<q>` (else `gfs_<q>`) of the same hour (same quantity from another weather model, same day)."""
    X = df[feats].copy()
    for c in feats:
        if c.startswith("ecmwf_"):
            q = c[len("ecmwf_"):]
            for src in (f"gefs_{q}", f"gfs_{q}"):
                if src in df.columns:
                    X[c] = X[c].fillna(df[src])
                    break
    return X.to_numpy("float32")


NNM_LAM, NNM_HB = 2.0, 24            # weight of the group-level loss and hours per batch of the "nnm" network members (nn_member.py)


def fit_recipe(tr: pd.DataFrame, tg: pd.DataFrame, recipe: list, smp: pd.Series, seeds_override: int | None = None, params: dict | None = None, regime: tuple | None = None,
               group_map: dict | None = None) -> dict:
    """Fit every model of the recipe on the training frame, return {name: utilisation forecast on the target frame}."""
    y = tr.u.clip(0, 1.2)
    w = pd.Series(hour_weights(pd.DatetimeIndex(tr.ts), smp), index=tr.index)
    w = w / w.mean()
    if regime:
        w = w * pd.Series(np.where(tr.ts > K(regime[0]), regime[1], 1.0), index=w.index)
        w = w / w.mean()
    out = {}
    for name, kind, fs, n_seeds in recipe:
        feats = FEATURE_SETS[fs](pr.groups(tr.columns))
        if kind in ("lgb", "xgb"):
            feats = feats + [c for c in tr.columns if c.startswith("s1_")]                    # stage-1 features (present only with --stage1): tree members only
        missing = [c for c in feats if c not in tg.columns]
        assert not missing, f"target frame lacks features of {name}: {missing[:5]}"
        preds = []
        if kind in ("nn", "nnm"):                                                           # neural-network member: all seeds inside nn_member.fit_predict
            import nn_member
            t1 = time.time()
            Xtr_nn, Xte_nn = impute_ecmwf(tr, feats), impute_ecmwf(tg, feats)
            groups = None
            if kind == "nnm":                                                               # trained on the contest's penalty: group totals of the submitted composition, hour by hour
                groups = dict(ts=pd.factorize(tr.ts)[0], grp=tr.site_id.map(group_map).to_numpy("int64"), cap=tr.cap.to_numpy("float32"))
            out[name] = np.clip(nn_member.fit_predict(Xtr_nn, y.to_numpy("float32"), w.to_numpy("float64"), Xte_nn,
                                                      seeds=seeds_override or n_seeds, log=log, groups=groups, lam=NNM_LAM, hb=NNM_HB), 0.0, 1.0)
            log(f"{name}: fitted on {len(tr):,} rows x {len(feats)} features in {time.time()-t1:.0f}s")
            continue
        for sd in range(seeds_override or n_seeds):
            t1 = time.time()
            if kind == "lgb":
                m = lgb.LGBMRegressor(objective="l1", verbose=-1, n_jobs=10, random_state=sd + SEED_OFFSET, **{**PARAMS, **(params or {}).get("lgb", {})}).fit(tr[feats], y, sample_weight=w)
            else:
                m = pr.fit_xgb(tr[feats], y, w, sd + SEED_OFFSET, (params or {}).get("xgb"))
            preds.append(m.predict(tg[feats]))
            log(f"{name} seed {sd}: fitted on {len(tr):,} rows x {len(feats)} features in {time.time()-t1:.0f}s")
            del m
        out[name] = np.clip(np.mean(preds, axis=0), 0.0, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test", "holdout"], default="test")
    ap.add_argument("--cut", default="2024-06-01", help="hold-out mode: train up to this date (00:00 KST), forecast the following 8,760 h")
    ap.add_argument("--out", default=os.path.join(HERE_DIR, "..", "submissions", "v1_structural"))
    ap.add_argument("--recipe", choices=list(RECIPES), default="v4", help="model blend: v1 = LightGBM only, v2 = LightGBM + XGBoost blend with history/neighbour features, v3 = v2 + national NWP features (external_data/), v4 = v3 + ECMWF IFS irradiance, v5 = v4 + consensus features / regime weights, v6 = v5 + regional cells + neural-network family + day-level shrink")
    ap.add_argument("--name", default=None, help="prefix of the two submission files (default: the output folder's name, capitalised), so each upload is recognisable on the portal")
    ap.add_argument("--nwp-dir", default=os.path.join(HERE_DIR, "external_data"), help="folder with gfs_national_raw.parquet and gefs_national_raw.parquet (written by 13_/14_*.py; used by the *_nwp feature sets)")
    ap.add_argument("--seeds", type=int, default=None, help="override the number of seeds of every model (quick tests)")
    ap.add_argument("--end", default=None, help="hold-out mode: last day (00:00 KST) of the forecast window instead of cut + 1 year (e.g. the Jan-May 2025 pseudo-test)")
    ap.add_argument("--composition", choices=["structural", "reference"], default="structural")
    ap.add_argument("--reuse-forecasts", default=None, help="parquet with site-level forecasts from an earlier run (skips training)")
    ap.add_argument("--train-end", default=None, help="test mode only: train on data up to this date (00:00 KST) instead of everything before the scoring year, e.g. 2025-01-01 (Test D: models that never saw the 2025 rows)")
    ap.add_argument("--stage1", default=None, help="stage1_features.parquet written by 23_stage1_observed_weather.py (expected observed UV index / cloud at the plant's provider-B station); merged by station and hour, given to the TREE members only")
    ap.add_argument("--dump-frames", default=None, help="post-contest research (17 Nov 2026): write the training and target feature frames (all columns) to this folder and stop before any model is fitted")
    args = ap.parse_args()
    recipe = RECIPES[args.recipe]
    need4 = any(fs not in ("G", "G_nwp") for _, _, fs, _ in recipe)
    nwp = None
    if any("_nwp" in fs for _, _, fs, _ in recipe):
        import hashlib
        from gfs_features import gfs_context, gefs_context, aerosol_context, ecmwf_context, ecmwf18_context, nwp_consensus, obs_context, gfsx_context, jma_context, ldaps_context
        parts, nwp_files = [], {}
        table = {"gfs": ("gfs_national_raw.parquet", gfs_context), "gefs": ("gefs_national_raw.parquet", gefs_context),
                 "aer": ("aerosol_national_raw.parquet", aerosol_context), "ecmwf": ("ecmwf_national_raw.parquet", ecmwf_context),
                 "ecmwf18": ("ecmwf18_national_raw.parquet", ecmwf18_context),
                 "obs": ("obs_national_raw.parquet", obs_context),                                    # V18: observed national irradiance (24_power_national.py), only recipes v6obs / v8obs
                 "gfsx": ("gfsx_national_raw.parquet", gfsx_context),                                 # post-contest study: more fields of the same GFS run (26_gfs_extra_national.py), recipe v6x
                 "jma": ("jma_gsm_national_raw.parquet", jma_context),                                # post-contest study: JMA GSM 00 UTC of D-1 (27_jma_gsm_national.py), recipe v6j
                 "ldaps": ("nwp_cells_ldaps.parquet", ldaps_context)}                                 # post-contest study: KMA UM-LDAPS 00 UTC of D-1 national summaries (22_kma_ldaps.py), recipe v6xl
        for nm, fn in [table[k] for k in NWP_SOURCES.get(args.recipe, ("gfs", "gefs", "aer"))]:
            pth = os.path.join(args.nwp_dir, nm)
            assert os.path.exists(pth), f"missing {pth} (create it with 13_/14_/15_*.py or copy it from external_data/)"
            parts.append(fn(pth))
            nwp_files[nm] = hashlib.md5(open(pth, "rb").read()).hexdigest()
        nwp = pd.concat(parts, axis=1)
        if args.recipe in CONSENSUS:
            nwp = pd.concat([nwp, nwp_consensus(nwp)], axis=1)
        log(f"national NWP summaries: {nwp.shape[1]} columns from {args.nwp_dir}")
    sites = load_sites()
    smp_train = load_smp_train()
    smp_score = pd.read_csv(f"{DATA}/smp.csv")
    ts_axis = pd.DatetimeIndex(pd.to_datetime(smp_score.ts)).tz_convert(TZ)            # the 8,760 scoring hours (same axis as the validator)

    ts_axis = ts_axis.as_unit("ns")
    if args.mode == "holdout":
        cut = K(f"{args.cut} 00:00")
        end = K(f"{args.end} 00:00") if args.end else cut + pd.DateOffset(years=1)
        ts_axis = pd.date_range(cut + pd.Timedelta(hours=1), end, freq="h").as_unit("ns")
    else:
        cut, end = K("2025-06-01 00:00"), K("2026-06-01 00:00")
        assert len(ts_axis) == 8760 and ts_axis[0] == K("2025-06-01 01:00") and ts_axis[-1] == end
    tcut = cut                                                                             # end of the training data (= cut unless --train-end is given)
    if args.train_end:
        assert args.mode == "test", "--train-end is for test mode; hold-out mode has --cut"
        tcut = K(f"{args.train_end} 00:00")
        assert tcut < cut

    if args.reuse_forecasts:
        site_fc = pd.read_parquet(args.reuse_forecasts)
        site_fc["ts"] = pd.DatetimeIndex(site_fc.ts).tz_convert(TZ)
        log(f"reusing site forecasts from {args.reuse_forecasts}: {len(site_fc):,} rows")
    else:
        # ---- training frame (everything up to the cut)
        fa_all, fb_all = load_forecasts("train")
        fa_tr, fb_tr = prepare_tables(fa_all[fa_all.ts <= tcut], fb_all[fb_all.ts <= tcut])
        gen = pd.read_parquet(f"{DATA}/train/generation.parquet")
        gen_full = gen
        gen = gen[(gen.ts <= tcut) & gen.ts.dt.hour.between(*DAY_HOURS)]
        nb = {"wa": bp.compute_neighbors(fa_tr, "wa"), "wb": bp.compute_neighbors(fb_tr, "wb")} if need4 else None      # data-driven neighbour stations
        regional = None
        if args.recipe in REGIONAL:
            import regional_features as rf
            gp = gen_full[gen_full.ts <= tcut].merge(sites[["site_id", "site_code", "capacity_kw", "lat", "lon"]], on="site_id")
            gp["elev"], _, _ = solar(pd.DatetimeIndex(gp.ts), gp.lat.to_numpy(), gp.lon.to_numpy())
            panel_like = gp.rename(columns={"capacity_kw": "cap"})[["site_code", "ts", "energy", "cap", "elev"]]
            regional = rf.regional_block(args.nwp_dir, tcut, panel_like, sources=REGIONAL[args.recipe]["sources"], n_pc=REGIONAL[args.recipe]["n_pc"], cloud_sources=REGIONAL[args.recipe].get("cloud_sources", ()))      # PCA / fleet weights: training hours only
            nwp = pd.concat([nwp, regional], axis=1)
            log(f"regional NWP block: {regional.shape[1]} columns from {args.nwp_dir}")
        tr = bp.build_feature_frame(gen[["site_id", "ts", "energy"]], fa_tr, fb_tr, sites, 0, 0, True, True, need4, nb, fa_tr, fb_tr, gfs=nwp)
        tr["u"] = (tr.energy / tr.capacity_kw).astype("float32")
        tr = tr[(tr.elev > 0) & tr.u.notna()].reset_index(drop=True)
        log(f"training frame: {len(tr):,} daytime rows, {tr.shape[1]} columns; recipe {args.recipe}: " + ", ".join(f"{n}({fs}, {len(FEATURE_SETS[fs](pr.groups(tr.columns)))} feats)" for n, _, fs, _ in recipe))
        # ---- target frame (scoring year; in hold-out mode the next 8,760 training-file hours)
        if args.mode == "test":
            fa_te, fb_te = load_forecasts("test")
            fa_te, fb_te = prepare_tables(fa_te, fb_te)
        else:
            fa_te, fb_te = prepare_tables(fa_all, fb_all, cut, end)
        unit = fa_te.ts.dt.unit
        ts_day = ts_axis[ts_axis.hour.isin(range(DAY_HOURS[0], DAY_HOURS[1] + 1))].as_unit(unit)
        grid = pd.MultiIndex.from_product([sites.site_id, ts_day], names=["site_id", "ts"]).to_frame(index=False)
        # forecasts of the previous days (needed for the regime features of the first days): tail of the training tables + the target tables
        tail = cut - pd.Timedelta(days=45)
        fa_tl, fb_tl = (fa_tr, fb_tr) if tcut == cut else prepare_tables(fa_all[(fa_all.ts > tail) & (fa_all.ts <= cut)], fb_all[(fb_all.ts > tail) & (fb_all.ts <= cut)])   # forecasts of the 45 days before the scoring year: public inputs, needed for the previous-day features of its first days
        fa_hist, fb_hist = pd.concat([fa_tl[fa_tl.ts > tail], fa_te], ignore_index=True), pd.concat([fb_tl[fb_tl.ts > tail], fb_te], ignore_index=True)
        tg = bp.build_feature_frame(grid, fa_te, fb_te, sites, 0, 0, True, True, need4, nb, fa_hist, fb_hist, gfs=nwp)
        tg = tg[tg.elev > 0].reset_index(drop=True)
        if args.stage1:
            s1 = pd.read_parquet(args.stage1)
            s1_info = json.load(open(os.path.join(os.path.dirname(os.path.abspath(args.stage1)), "stage1_info.json")))
            assert not s1_info.get("smoke"), "this stage-1 file is a smoke run"
            assert pd.Timestamp(s1_info["train_end"]) == tcut, f"stage-1 file was trained to {s1_info['train_end']}, this run trains to {tcut}: use the matching file"
            n_tr, n_tg = len(tr), len(tg)
            tr = tr.merge(s1.assign(ts=s1.ts.astype(tr.ts.dtype)), on=["wb_station", "ts"], how="left")
            tg = tg.merge(s1.assign(ts=s1.ts.astype(tg.ts.dtype)), on=["wb_station", "ts"], how="left")
            assert len(tr) == n_tr and len(tg) == n_tg
            miss_tr, miss_tg = float(tr["s1_uv"].isna().mean()), float(tg["s1_uv"].isna().mean())
            log(f"stage-1 features: {sum(c.startswith('s1_') for c in tr.columns)} columns from {args.stage1} (trained to {s1_info['train_end']}); rows without a value: training {miss_tr:.4f}, target {miss_tg:.4f}")
            assert miss_tr < 0.005 and miss_tg < 0.005, "stage-1 values are missing for too many rows"
        allf = sorted({c for _, _, fs, _ in recipe for c in FEATURE_SETS[fs](pr.groups(tr.columns))})
        missing = [c for c in allf if c not in tg.columns]
        assert not missing, f"target frame lacks features: {missing[:5]}"
        nan_share = tg[allf].isna().mean()
        log(f"target frame: {len(tg):,} daytime rows; share of NaN per feature: max {nan_share.max():.4f}, mean {nan_share.mean():.5f} "
            f"(worst: {nan_share.idxmax()})")
        if args.dump_frames:                                                                 # post-contest research: the frames only, no models (default path unchanged)
            os.makedirs(args.dump_frames, exist_ok=True)
            tr.to_parquet(os.path.join(args.dump_frames, "train_frame.parquet"), index=False)
            tg.to_parquet(os.path.join(args.dump_frames, "target_frame.parquet"), index=False)
            json.dump(dict(mode=args.mode, cut=str(cut), end=str(end), recipe=args.recipe, train_end=str(tcut), n_train=int(len(tr)), n_target=int(len(tg)),
                           feature_sets={fs: FEATURE_SETS[fs](pr.groups(tr.columns)) for fs in FEATURE_SETS if "nn" in fs or fs in {f for _, _, f, _ in recipe}},
                           nwp_tables=(nwp_files if nwp is not None else None), built=time.strftime("%Y-%m-%d %H:%M:%S")),
                      open(os.path.join(args.dump_frames, "frames_info.json"), "w"), indent=1)
            log(f"frames written to {args.dump_frames}: training {len(tr):,} rows, target {len(tg):,} rows, {tr.shape[1]} columns; stopping (--dump-frames)")
            return
        # ---- models (blend = equal-weight mean of the models' utilisation forecasts)
        _, mask0, _ = build_composition(sites, args.composition)
        group_map = dict(zip(sites.site_id, mask0.astype(int)))
        preds_u = fit_recipe(tr, tg, recipe, smp_train, args.seeds, RECIPE_PARAMS.get(args.recipe), REGIME.get(args.recipe), group_map)
        tree_u = [v for n, v in preds_u.items() if not n.startswith("nn_")]
        nn_u = [v for n, v in preds_u.items() if n.startswith("nn_")]
        if nn_u and tree_u:                                                                  # family blend: (1 - share) x mean(tree members) + share x mean(network members)
            share = FAMILY_SHARE[args.recipe]
            blend_u = family_blend(tree_u, nn_u, share)
            log(f"family blend: {len(tree_u)} tree members x {1-share:.2f} + {len(nn_u)} network members x {share:.2f}")
            # guard: the two families must agree on the fleet level (a collapsed network shows up here, see the G_nwp_nn comment)
            fm, fn_ = np.mean(tree_u, axis=0) * tg.cap.to_numpy("float64"), np.mean(nn_u, axis=0) * tg.cap.to_numpy("float64")
            dkey = (pd.DatetimeIndex(tg.ts) - pd.Timedelta(hours=1)).normalize().to_numpy()
            dm, dn = pd.Series(fm).groupby(dkey).sum(), pd.Series(fn_).groupby(dkey).sum()
            rel = float(np.abs(dn - dm).sum() / dm.sum()); corr = float(np.corrcoef(dm, dn)[0, 1])
            log(f"agreement of the families (daily fleet energy): mean |network - trees| / trees = {rel:.3f}, correlation {corr:.3f}")
            assert rel < 0.15 and corr > 0.9, "the network family disagrees with the tree family: check its inputs"
        else:
            blend_u = family_blend(tree_u, nn_u, 0.0)
        beta = SLOPE.get(args.recipe)
        if beta:
            blend_u, f_row, cf_ref = day_shrink(blend_u, tg.ts, tg.cap.to_numpy("float64"), float(sites.capacity_kw.sum()), beta, cf_ref=CF_REF)      # fixed reference (training-period information), not the scoring period's own mean
            log(f"day-level shrink beta {beta}: reference cf {cf_ref:.3f} (fixed), factor range {f_row.min():.3f}..{f_row.max():.3f}")
        full = pd.MultiIndex.from_product([sites.site_id, ts_axis], names=["site_id", "ts"]).to_frame(index=False)
        cols = {"kwh": blend_u, **{f"kwh_{n}": p for n, p in preds_u.items()}}
        out_day = pd.DataFrame({"site_id": tg.site_id.to_numpy(), "ts": pd.DatetimeIndex(tg.ts).tz_convert(TZ).as_unit("ns"),
                                **{k: v * tg.cap.to_numpy("float64") for k, v in cols.items()}})
        site_fc = full.merge(out_day, on=["site_id", "ts"], how="left").fillna({k: 0.0 for k in cols})
        os.makedirs(args.out, exist_ok=True)
        site_fc.to_parquet(os.path.join(args.out, "site_forecasts.parquet"), index=False)
        used_sets = sorted({fs for _, _, fs, _ in recipe})
        json.dump(dict(mode=args.mode, cut=str(cut), recipe=args.recipe, models=[list(m) for m in recipe], seeds_override=args.seeds, lgb_params=PARAMS,
                       xgb_params=XGB_PARAMS_NOTE, recipe_params=RECIPE_PARAMS.get(args.recipe), regime_weight=REGIME.get(args.recipe), regional=REGIONAL.get(args.recipe), family_share=FAMILY_SHARE.get(args.recipe), slope_beta=SLOPE.get(args.recipe), nwp_tables=(nwp_files if nwp is not None else None), feature_sets={fs: FEATURE_SETS[fs](pr.groups(tr.columns)) for fs in used_sets},
                       n_train_rows=int(len(tr)), train_end=str(tcut), stage1=(os.path.abspath(args.stage1) if args.stage1 else None), built=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(args.out, "run_info.json"), "w"), indent=1)
        # ---- sanity: yearly yield per kW against the last full training year
        ref = gen_full[(gen_full.ts > K("2024-06-01 00:00")) & (gen_full.ts <= K("2025-06-01 00:00"))].merge(sites[["site_id", "capacity_kw"]], on="site_id")
        cov = ref.groupby("site_id").size() / 8760
        yld_hist = (ref.groupby("site_id").energy.sum() / ref.groupby("site_id").capacity_kw.first())[cov[cov >= 0.9].index]
        yld_fc = (site_fc.groupby("site_id").kwh.sum() / sites.set_index("site_id").capacity_kw)[yld_hist.index] if args.mode == "test" else None
        if yld_fc is not None:
            log(f"sanity: median annual yield kWh/kW, last full training year {yld_hist.median():.0f}  vs  forecast {yld_fc.median():.0f}")

    comp, mask, (c1, c2) = build_composition(sites, args.composition)
    log(f"composition ({args.composition}): G1 {int(mask.sum())} sites {c1:,.1f} kW ({c1/(c1+c2):.1%}), G2 {int((~mask).sum())} sites {c2:,.1f} kW")
    fc = site_fc.merge(comp, on="site_id").groupby(["group", "ts"], as_index=False).kwh.sum().rename(columns={"kwh": "forecast_kwh"})
    fc = fc.sort_values(["group", "ts"]).reset_index(drop=True)

    if args.mode == "holdout":
        gen_w = pd.read_parquet(f"{DATA}/train/generation.parquet")
        gen_w = gen_w[(gen_w.ts > cut) & (gen_w.ts <= end)].merge(sites[["site_id", "site_code", "capacity_kw", "lat", "lon"]], on="site_id")
        gen_w["elev"], _, _ = solar(pd.DatetimeIndex(gen_w.ts), gen_w.lat.to_numpy(), gen_w.lon.to_numpy())
        panel_like = gen_w.rename(columns={"capacity_kw": "cap"})[["site_code", "ts", "energy", "cap", "elev"]]
        pred_all = site_fc.merge(sites[["site_id", "site_code", "capacity_kw"]], on="site_id")
        pred_all["ts"] = pd.DatetimeIndex(pred_all.ts).as_unit(panel_like.ts.dt.unit)
        for col in [c for c in pred_all.columns if c.startswith("kwh_")]:                 # each model of the blend on its own
            q = pred_all.assign(pred_u=(pred_all[col] / pred_all.capacity_kw).astype("float32"))[["site_code", "ts", "pred_u"]]
            dq = error_matrix(q, panel_like, cut, end)
            mq = (sites.set_index("site_code").reindex(dq["sites"]).current_group == "A").to_numpy()
            print(f"  {col:16s} W (reference split) {W_of(dq['E'], dq['w'], mq)/1e6:.2f}M")
        pred = pred_all.assign(pred_u=(pred_all.kwh / pred_all.capacity_kw).astype("float32"))[["site_code", "ts", "pred_u"]]
        d = error_matrix(pred, panel_like, cut, end)
        U = d["sites"]
        s_u = sites.set_index("site_code").reindex(U)
        ref_mask = (s_u.current_group == "A").to_numpy()
        pipe_mask = comp.set_index("site_id").group.reindex(s_u.site_id).eq("G1").to_numpy()
        res = dict(n_sites=int(len(U)), W_reference=W_of(d["E"], d["w"], ref_mask), W_pipeline_split=W_of(d["E"], d["w"], pipe_mask),
                   W_floor=W_floor(d["E"], d["w"]), pipeline_split_cap_share=float(d["cap"][pipe_mask].sum() / d["cap"].sum()))
        print(json.dumps(res, indent=1))
        for name in ("G", "G_s1", "G_2seeds", "H_hist", "H_both", "G_xgb", "B_hist_xgb"):
            p = f"{CACHE}/results_{args.cut}.csv"
            if os.path.exists(p):
                r = pd.read_csv(p).set_index("exp")
                if name in r.index:
                    print(f"research harness {name}: W_cur {r.loc[name, 'W_cur']/1e6:.2f}M  floor {r.loc[name, 'W_floor']/1e6:.2f}M")
        print(f"this path     : W_reference-split {res['W_reference']/1e6:.2f}M  floor {res['W_floor']/1e6:.2f}M  (pipeline split on this subset: {res['W_pipeline_split']/1e6:.2f}M)")
        return

    # ---- write and check the two files
    os.makedirs(args.out, exist_ok=True)
    fc_out = fc.copy()
    fc_out["ts"] = fc_out.ts.dt.strftime("%Y-%m-%d %H:%M:%S") + "+09:00"
    name = args.name or os.path.basename(os.path.normpath(args.out)).capitalize()          # e.g. out=.../v2_blend -> V2_blend_composition.csv, V2_blend_forecast.csv
    comp_path, fc_path = os.path.join(args.out, f"{name}_composition.csv"), os.path.join(args.out, f"{name}_forecast.csv")
    comp.to_csv(comp_path, index=False)
    fc_out[["group", "ts", "forecast_kwh"]].to_csv(fc_path, index=False, float_format="%.4f")
    chk = pd.read_csv(fc_path)
    assert len(chk) == 17520 and chk.forecast_kwh.notna().all() and np.isfinite(chk.forecast_kwh).all() and (chk.forecast_kwh >= 0).all()
    assert chk.groupby("group").ts.nunique().eq(8760).all() and len(pd.read_csv(comp_path)) == 253
    log(f"wrote {comp_path} (253 rows) and {fc_path} (17,520 rows); annual energy G1 {fc[fc.group=='G1'].forecast_kwh.sum()/1e6:.2f} GWh, G2 {fc[fc.group=='G2'].forecast_kwh.sum()/1e6:.2f} GWh")
    r = subprocess.run([sys.executable, os.path.join(HERE_DIR, "organiser_tools", "validate_submission.py"), "--composition", comp_path, "--forecast", fc_path, "--data", DATA],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        sys.exit(f"organiser validator failed (exit {r.returncode})")


if __name__ == "__main__":
    main()
