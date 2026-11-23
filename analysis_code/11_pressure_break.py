"""DIAGNOSTIC (not part of the submission pipeline): what does the 23 Jan 2026 change of provider A's ground-pressure forecast do to a model that uses pressure?

    python 11_pressure_break.py [--model lgb|xgb] [--no-ablation]

Provider A's ground-pressure forecast switches definition in the scoring-year files: from 2026-01-23 00:00, 26 of 61 stations (50% of capacity) drop by 2-58 hPa and stay low
(station-level instead of sea-level pressure).  Nothing like it exists in the training years, so the question is how much a model trained on the old definition suffers.
Replay: hold-out year 1 (train to 2024-05, score 2024-06 -> 2025-05); the break is replayed one year earlier (2025-01-23) with the station offsets measured in the scoring-year
forecast files (this is where this script reads scoring-year forecasts: to size the shift, never to build features or fit anything that is submitted).
Prints W for: LightGBM with pressure on clean features / on shifted features / without pressure features, and the share of gain the pressure features carry.
Writes <CACHE>/press_sim_2024-06-01.csv (LightGBM) or press_sim_2024-06-01_xgb.csv (XGBoost).
"""
import argparse, sys, time, importlib
import numpy as np, pandas as pd
from oibc_lib import *
pr = importlib.import_module("03_forecast_probes")
sub = importlib.import_module("08_make_submission")

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["lgb", "xgb"], default="lgb")
ap.add_argument("--no-ablation", action="store_true", help="skip the fit without pressure features")
args = ap.parse_args()
cut, end = K("2024-06-01 00:00"), K("2025-06-01 00:00")
smp = load_smp_train(); sites = load_sites().set_index("site_code"); cg = sites.current_group

# ---- station offsets from the test forecasts (diagnostic only)
fa, _ = sub.load_forecasts("train"); ta, _ = sub.load_forecasts("test")
F = pd.concat([fa, ta], ignore_index=True); F = F[F["round"] == 17]
F["day"] = F.ts.dt.tz_localize(None).dt.normalize()
d = F.groupby(["wa_station", "day"]).wa_ground_press.mean().unstack(0)
dev = d.sub(d.median(axis=1), axis=0)
off = (dev.loc["2026-02-01":].median() - dev.loc[:"2025-12-01"].median())
off = off.where(off < -2.0, 0.0)                       # only the stations that really changed
print("stations shifted:", int((off < 0).sum()), "of", len(off), "mean offset", round(off[off < 0].mean(), 1), flush=True)
site_off = sites.wa_station.map(off).fillna(0.0)

p = pd.read_parquet(f"{CACHE}/panel_ctx4.parquet")
full = p[["site_code", "ts", "energy", "cap", "elev"]].copy()
p = p[p.elev > 0].reset_index(drop=True)
p["w"] = hour_weights(pd.DatetimeIndex(p.ts), smp).astype("float32")
g = pr.groups(p.columns)
feats_all = pr.without(pr.only("raw", "ctx", "geo"), "lat", "lon")(g)          # recipe G (264 features)
press = [c for c in feats_all if "ground_press" in c]
feats_nop = [c for c in feats_all if c not in press]
print("features", len(feats_all), "pressure features", len(press), "without", len(feats_nop), flush=True)
tr = p[(p.ts <= cut) & p.u.notna()]
va = p[(p.ts > cut) & (p.ts <= end)].copy()
y = tr.u.clip(0, 1.2); w = tr.w / tr.w.mean()
va_d = va.copy()
brk = K("2025-01-23 00:00")
o = va_d.site_code.map(site_off).to_numpy("float32") * (va_d.ts > brk).to_numpy()
for c in press:
    if c.endswith("_dstd"):
        continue
    va_d[c] = va_d[c] + o
print("rows after the replayed break:", int((va_d.ts > brk).sum()), "rows with offset:", int((o != 0).sum()), flush=True)

def score(pred, name):
    out = va[["site_code", "ts"]].copy(); out["pred_u"] = np.clip(pred, 0, None).astype("float32")
    e = error_matrix(out, full, cut, end)
    m = (cg.reindex(e["sites"]) == "A").values
    pen = 0.10 * np.asarray(e["w"]) * (np.abs(e["E"][:, m].sum(1)) + np.abs(e["E"][:, ~m].sum(1)))
    sel = np.asarray(e["ts"] > brk)
    print(f"{name:34s} W {pen.sum()/1e6:8.2f}M   after the replayed break {pen[sel].sum()/1e6:7.2f}M   before {pen[~sel].sum()/1e6:7.2f}M", flush=True)
    return float(pen.sum()), float(pen[sel].sum())

fit = (lambda X: pr.fit_lgb(X, y, w, "l1", pr.P_BIG, 0)) if args.model == "lgb" else (lambda X: pr.fit_xgb(X, y, w, 0))
t0 = time.time()
m1 = fit(tr[feats_all])
print(f"fit with pressure {time.time()-t0:.0f}s", flush=True)
a = score(m1.predict(va[feats_all]), "with pressure, clean features")
b = score(m1.predict(va_d[feats_all]), "with pressure, break replayed")
imp = pd.Series(m1.booster_.feature_importance("gain") if args.model == "lgb" else m1.feature_importances_, index=feats_all); imp = imp / imp.sum()
print("share of total gain in pressure features: %.3f ; top pressure feature %s %.4f" % (imp[press].sum(), imp[press].idxmax(), imp[press].max()), flush=True)
print("top-10 gain features:", ", ".join(f"{k} {v:.3f}" for k, v in imp.sort_values(ascending=False).head(10).items()), flush=True)
res = {"with_clean": a, "with_drift": b}
if not args.no_ablation:
    t0 = time.time()
    m2 = fit(tr[feats_nop])
    print(f"fit without pressure {time.time()-t0:.0f}s", flush=True)
    res["nop_clean"] = score(m2.predict(va[feats_nop]), "without pressure, clean features")
pd.DataFrame(res, index=["W", "W_window"]).to_csv(f"{CACHE}/press_sim_2024-06-01{'' if args.model == 'lgb' else '_xgb'}.csv")
