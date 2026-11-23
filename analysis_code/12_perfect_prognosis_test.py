"""EXPERIMENT (negative result, see report section 5.1): 'perfect-prognosis' feature.  A model f_obs is trained on OBSERVED station weather -> utilisation (clean inputs, plenty of data); it is then applied to the
FORECAST values of the same variables (as if they were observations) and its output z is added as a feature to the normal forecast model G.
OOF (time-blocked) predictions for training rows, full-train model for hold-out rows.  Training years only; observed weather of the scoring year does not exist.

    python 12_perfect_prognosis_test.py --cut 2024-06-01 [--seeds 1]
"""
import argparse, time, importlib, os
import numpy as np, pandas as pd
from oibc_lib import *

pr = importlib.import_module("03_forecast_probes")
ap = argparse.ArgumentParser()
ap.add_argument("--cut", default="2024-06-01")
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--name", default="G_pp")
args = ap.parse_args()
cut = K(f"{args.cut} 00:00"); end = cut + pd.DateOffset(years=1)
smp = load_smp_train(); sites = load_sites(); cg = sites.set_index("site_code").current_group
T0 = time.time()
log = lambda m: print(f"[{time.time()-T0:5.0f}s] {m}", flush=True)

OA = ["wa_cloud", "wa_temp", "wa_humidity", "wa_wind_speed", "wa_rain"]
OB = ["wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_cloud", "wb_vis", "wb_wind_speed", "wb_dew_point", "wb_ceiling"]
GEO_PP = ["elev", "azim", "cosz", "doy_sin", "doy_cos", "hour", "cap"]

p = pd.read_parquet(f"{CACHE}/panel_ctx3.parquet")
full = p[["site_code", "ts", "energy", "cap", "elev"]].copy()
p = p[p.elev > 0].reset_index(drop=True)
p = p.merge(sites[["site_code", "wa_station", "wb_station"]], on="site_code", how="left")
oa = pd.read_parquet(f"{DATA}/train/observed_wa.parquet").drop_duplicates(["wa_station", "ts"])[["wa_station", "ts"] + OA]
ob = pd.read_parquet(f"{DATA}/train/observed_wb.parquet")[["wb_station", "ts"] + OB]
oa.loc[oa.wa_wind_speed > 50, "wa_wind_speed"] = np.nan
ob.loc[ob.wb_cloud > 100, "wb_cloud"] = np.nan
ob.loc[ob.wb_uv_idx > 15, "wb_uv_idx"] = np.nan
ob.loc[ob.wb_wind_speed > 150, "wb_wind_speed"] = np.nan
oa = oa.rename(columns={c: "o_" + c for c in OA}); ob = ob.rename(columns={c: "o_" + c for c in OB})
p = p.merge(oa, on=["wa_station", "ts"], how="left").merge(ob, on=["wb_station", "ts"], how="left")
p["w"] = hour_weights(pd.DatetimeIndex(p.ts), smp).astype("float32")
o_cols = ["o_" + c for c in OA + OB]
log(f"panel {len(p):,} rows; observed coverage {p[o_cols].notna().all(axis=1).mean():.3f}")

tr = p[(p.ts <= cut) & p.u.notna() & p[o_cols].notna().all(axis=1)]            # f_obs needs clean observations
va = p[(p.ts > cut) & (p.ts <= end)]
P_OBS = dict(n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
feats_obs = o_cols + GEO_PP


def as_if_observed(df, rnd):
    x = pd.DataFrame(index=df.index)
    for c in OA + OB:
        x["o_" + c] = df[f"{c}_r{rnd}"]
    for c in GEO_PP:
        x[c] = df[c]
    return x[feats_obs]


def fit_obs(rows):
    return pr.fit_lgb(rows[feats_obs], rows.u.clip(0, 1.2), rows.w / rows.w.mean(), "l1", P_OBS, 0)


# ---- OOF z for ALL training rows (also rows without clean observations), full-train z for the hold-out
z = pd.DataFrame(index=p.index, columns=["z10", "z17"], dtype="float32")
t_lo = K("2021-01-01 00:00"); n_blocks = 3
edges = [t_lo + (cut - t_lo) * i / n_blocks for i in range(n_blocks + 1)]
purge = pd.Timedelta(days=7)
for i in range(n_blocks):
    a_, b_ = edges[i], edges[i + 1]
    trn = tr[~((tr.ts > a_ - purge) & (tr.ts <= b_ + purge))]
    m = fit_obs(trn)
    te = p[(p.ts > a_) & (p.ts <= b_)]
    for r in (10, 17):
        z.loc[te.index, f"z{r}"] = m.predict(as_if_observed(te, r)).astype("float32")
    log(f"OOF block {i}: f_obs trained on {len(trn):,} rows, applied to {len(te):,}")
m = fit_obs(tr)
hv = p[(p.ts > cut)]
for r in (10, 17):
    z.loc[hv.index, f"z{r}"] = m.predict(as_if_observed(hv, r)).astype("float32")
log(f"full f_obs trained on {len(tr):,} rows")
p["z10"], p["z17"] = z.z10, z.z17

# sanity: z alone as a forecast (perfect-prognosis), and correlation with u on the hold-out
va = p[(p.ts > cut) & (p.ts <= end)]
print("corr(z17, u) hold-out %.3f ; corr(z10, u) %.3f" % (np.corrcoef(va.z17.fillna(0), va.u.fillna(0))[0, 1], np.corrcoef(va.z10.fillna(0), va.u.fillna(0))[0, 1]), flush=True)

g = pr.groups(p.columns)
feats = pr.without(pr.only("raw", "ctx", "geo"), "lat", "lon")(g) + ["z10", "z17"]
trm = p[(p.ts <= cut) & p.u.notna()]
y = trm.u.clip(0, 1.2); w = trm.w / trm.w.mean()
res_path = f"{CACHE}/results_{args.cut}.csv"


def score(pred, name):
    out = va[["site_code", "ts"]].copy(); out["pred_u"] = np.clip(pred, 0, None).astype("float32")
    out.to_parquet(f"{CACHE}/pred_{args.cut}_{name}.parquet", index=False)
    e = error_matrix(out, full, cut, end)
    m_ = (cg.reindex(e["sites"]) == "A").values
    W = W_of(e["E"], e["w"], m_)
    results = pd.read_csv(res_path).set_index("exp")
    row = dict(n_sites=len(e["sites"]), n_feat=len(feats), W_cur=W, W_floor=W_floor(e["E"], e["w"]), W_nopool=W_nopool(e["E"], e["w"]), seconds=round(time.time() - T0, 1))
    results.loc[name, list(row)] = list(row.values())
    results["score_vs_M0"] = 100 * (1 - results.W_cur / results.loc["M0", "W_cur"])
    results.index.name = "exp"
    results.to_csv(res_path)
    print(f"{name:14s} W {W/1e6:8.2f}M  score {results.loc[name, 'score_vs_M0']:+6.2f}", flush=True)


# z alone (perfect-prognosis forecast), for information
score(va.z17.fillna(0).to_numpy(), "PP_z17_alone")
preds = []
for sd in range(args.seeds):
    mm = pr.fit_lgb(trm[feats], y, w, "l1", pr.P_BIG, sd)
    preds.append(mm.predict(va[feats]))
    log(f"G + z fitted seed {sd}")
score(np.mean(preds, axis=0), args.name if args.seeds == 1 else args.name + f"_{args.seeds}s")
imp = pd.Series(mm.booster_.feature_importance("gain"), index=feats); imp = imp / imp.sum()
print("gain share of z10/z17: %.3f / %.3f ; top: %s" % (imp["z10"], imp["z17"], ", ".join(f"{k} {v:.3f}" for k, v in imp.sort_values(ascending=False).head(6).items())), flush=True)
