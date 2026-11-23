"""Second-pass data checks: timestamp alignment, forecast usefulness for generation, duplicated / copied data.

Needs the panel from 02_build_panel.py.   Usage: python 01b_review_checks.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from oibc_lib import *

out = {}
sites = load_sites()
panel = pd.read_parquet(f"{CACHE}/panel.parquet")
H = pd.Timedelta(hours=1)

# ---------------------------------------------------------------- 1. alignment (lag scan)
fa = pd.read_parquet(f"{DATA}/train/forecast_wa.parquet")
fb = pd.read_parquet(f"{DATA}/train/forecast_wb.parquet")
oa = pd.read_parquet(f"{DATA}/train/observed_wa.parquet").drop_duplicates(["wa_station", "ts"])
ob = pd.read_parquet(f"{DATA}/train/observed_wb.parquet")


def lag_scan_fc_vs_obs(f, o, st, var, ovar=None, lags=range(-3, 4)):
    ovar = ovar or var
    f17 = f[f["round"] == 17][[st, "ts", var]]
    res = {}
    for k in lags:
        g = f17.assign(ts=f17.ts - k * H)                    # forecast made for t+k lined up with observation at t
        m = g.merge(o[[st, "ts", ovar]], on=[st, "ts"], suffixes=("_f", "_o"))
        x, y = m[f"{var}_f"] if f"{var}_f" in m else m[var], m[f"{ovar}_o"] if f"{ovar}_o" in m else m[ovar]
        ok = x.notna() & y.notna() & (y < 9000)
        res[k] = round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 4)
    return res


out["lag_fc_vs_obs"] = {
    "A temp": lag_scan_fc_vs_obs(fa, oa, "wa_station", "wa_temp"),
    "B temp": lag_scan_fc_vs_obs(fb, ob, "wb_station", "wb_temp"),
    "B uv": lag_scan_fc_vs_obs(fb, ob, "wb_station", "wb_uv_idx"),
}
print("corr(forecast made for t+k, observed at t); best lag should be 0")
for k, v in out["lag_fc_vs_obs"].items():
    print(f"  {k:7s}", v, " best lag:", max(v, key=v.get))

# diurnal profile: hour label of the peak of each series (a one-hour labelling offset shows up here)
prof = {
    "generation utilisation": panel.groupby("hour").u.mean(),
    "B UV forecast r17": fb[fb["round"] == 17].groupby(fb.ts.dt.hour).wb_uv_idx.mean(),
    "B UV observed": ob.groupby(ob.ts.dt.hour).wb_uv_idx.apply(lambda s: s.clip(upper=15).mean()),
    "A temp forecast r17": fa[fa["round"] == 17].groupby(fa.ts.dt.hour).wa_temp.mean(),
    "A temp observed": oa.groupby(oa.ts.dt.hour).wa_temp.mean(),
    "B temp forecast r17": fb[fb["round"] == 17].groupby(fb.ts.dt.hour).wb_temp.mean(),
    "B temp observed": ob.groupby(ob.ts.dt.hour).wb_temp.mean(),
}
out["peak_hour_label"] = {k: int(v.idxmax()) for k, v in prof.items()}
print("hour label of the daily peak:", out["peak_hour_label"])
print("B UV by hour 9..15  forecast:", prof["B UV forecast r17"].loc[9:15].round(2).to_dict(), "\n                    observed:", prof["B UV observed"].loc[9:15].round(2).to_dict())

# generation vs station series at lags (daytime, elevation > 15 deg)
d = panel.loc[panel.elev > 15, ["site_code", "ts", "u", "elev"]].merge(sites[["site_code", "wa_station", "wb_station"]], on="site_code")
d = d.sample(frac=0.35, random_state=0)


def lag_scan_gen(table, st, var, lags=range(-3, 4)):
    res = {}
    for k in lags:
        g = table[[st, "ts", var]].assign(ts=lambda t: t.ts - k * H)
        m = d.merge(g, on=[st, "ts"], how="inner")
        ok = m[var].notna()
        res[k] = round(float(np.corrcoef(m.loc[ok, "u"], m.loc[ok, var])[0, 1]), 4)
    return res


fb17 = fb[fb["round"] == 17]
fa17 = fa[fa["round"] == 17]
out["lag_gen"] = {
    "A cloud (forecast)": lag_scan_gen(fa17, "wa_station", "wa_cloud"),
    "B uv (forecast)": lag_scan_gen(fb17, "wb_station", "wb_uv_idx"),
    "B uv (observed)": lag_scan_gen(ob, "wb_station", "wb_uv_idx"),
    "A cloud (observed)": lag_scan_gen(oa, "wa_station", "wa_cloud"),
}
print("\ncorr(utilisation at t, weather value at t+k): the extreme should sit at lag 0")
for k, v in out["lag_gen"].items():
    print(f"  {k:20s}", v, " strongest |corr| lag:", max(v, key=lambda z: abs(v[z])))

# ---------------------------------------------------------------- 2. forecast usefulness for generation
# z-score utilisation and each weather field within (month, hour) cells, then correlate (daytime, elevation 25-70)
dd = panel[(panel.elev > 25) & (panel.elev < 70) & panel.u.notna()].copy()
dd["cell"] = dd.ts.dt.month * 100 + dd.hour.astype(int)
g = dd.groupby("cell")
dd["u_z"] = (dd.u - g.u.transform("mean")) / g.u.transform("std")
cands = ["wa_cloud_r17", "wa_cloud_r10", "wb_cloud_r17", "wb_cloud_r10", "cloud_mean_r17", "wa_humidity_r17", "wb_rel_hum_r17",
         "wb_uv_idx_r17", "wb_precip_prob_r17", "wb_ceiling_r17", "wb_vis_r17", "wa_cloud_r17_m3", "wa_cloud_r17_dmean", "fleet_wa_cloud_r17", "ghi_A_r17", "ghi_B_r17"]
res = {}
for c in cands:
    z = (dd[c] - dd.groupby("cell")[c].transform("mean")) / dd.groupby("cell")[c].transform("std")
    ok = z.notna() & dd.u_z.notna()
    res[c] = round(float(np.corrcoef(z[ok], dd.u_z[ok])[0, 1]), 3)
out["usefulness_corr"] = res
print("\ncorrelation with utilisation after removing month x hour means (daytime, elev 25-70 deg):")
for k, v in sorted(res.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {k:24s} {v:+.3f}")

# ---------------------------------------------------------------- 3. duplicates / copied data
# 3a generation duplicates between sites (identical daytime values at the same hour)
gp = panel[(panel.hour >= 8) & (panel.hour <= 16)].pivot_table(index="ts", columns="site_code", values="energy", aggfunc="first")
M = gp.to_numpy(dtype="float32")
cols = gp.columns.to_numpy()
pairs = []
for i in range(M.shape[1]):
    xi = M[:, i:i + 1]
    ok_i = ~np.isnan(xi) & (xi >= 10.0)                       # continuous range only: small values are quantised and coincide by chance
    eq = (M[:, i + 1:] == xi) & ok_i
    n_eq = eq.sum(0)
    n_both = (ok_i & (M[:, i + 1:] >= 10.0)).sum(0)
    for jj in np.where(n_eq >= 30)[0]:
        pairs.append((int(cols[i]), int(cols[i + 1 + jj]), int(n_eq[jj]), int(n_both[jj])))
pairs.sort(key=lambda t: -t[2])
out["dup_site_pairs"] = pairs[:15]
print("\nsite pairs with >= 30 identical daytime values >= 10 kWh at the same hour:", len(pairs), " top:", pairs[:8])
# near-duplicate series (very high correlation of utilisation)
U = (gp.to_numpy() / sites.set_index("site_code").capacity_kw.reindex(cols).to_numpy()[None, :]).astype("float32")
C = pd.DataFrame(U).corr(min_periods=500).to_numpy().copy()
np.fill_diagonal(C, 0)
iu = np.triu_indices_from(C, 1)
top = np.argsort(-np.nan_to_num(C[iu]))[:5]
out["top_corr_pairs"] = [(int(cols[iu[0][t]]), int(cols[iu[1][t]]), round(float(C[iu][t]), 4)) for t in top]
print("highest utilisation correlations between two sites:", out["top_corr_pairs"])

# 3b repeated days within a site
panel["day"] = (panel.ts - H).dt.normalize()
rep = 0
rep_examples = []
for sc, g in panel.groupby("site_code"):
    seen = {}
    for day, gd in g.groupby("day"):
        v = np.round(gd.energy.to_numpy(), 2)
        if (v > 0).sum() < 5 or len(v) < 20:
            continue
        key = v.tobytes()
        if key in seen:
            rep += 1
            if len(rep_examples) < 5:
                rep_examples.append((int(sc), str(seen[key].date()), str(day.date())))
        else:
            seen[key] = day
out["repeated_generation_days"] = dict(count=rep, examples=rep_examples)
print("\nidentical full-day generation profiles within a site:", rep, rep_examples)

# 3c repeated forecast days within a station / round, and test-year days equal to training days
def dup_forecast_days(f, st, cols_):
    n = 0
    ex = []
    for (s, r), g in f.groupby([st, "round"]):
        g = g.assign(day=(g.ts - H).dt.normalize())
        seen = {}
        for day, gd in g.groupby("day"):
            if len(gd) < 23:
                continue
            v = np.round(gd.sort_values("ts")[cols_].to_numpy(dtype="float64"), 3)
            if np.nanstd(v[:, 0]) == 0 and np.nanstd(v[:, 1]) == 0:
                continue
            key = v.tobytes()
            if key in seen:
                n += 1
                if len(ex) < 3:
                    ex.append((s, int(r), str(seen[key].date()), str(day.date())))
            else:
                seen[key] = day
    return n, ex


fa_te = pd.read_parquet(f"{DATA}/test/forecast_wa.parquet")
fb_te = pd.read_parquet(f"{DATA}/test/forecast_wb.parquet")
na, ea = dup_forecast_days(pd.concat([fa, fa_te]), "wa_station", ["wa_temp", "wa_humidity", "wa_cloud"])
nb, eb = dup_forecast_days(pd.concat([fb, fb_te]), "wb_station", ["wb_temp", "wb_rel_hum", "wb_cloud"])
out["repeated_forecast_days"] = dict(A=na, B=nb, A_ex=ea, B_ex=eb)
print("\nidentical forecast days within a station-round (train+test): A", na, ea, "| B", nb, eb)

# 3d identical station series
def dup_stations(f, st, var):
    w = f[f["round"] == 17].pivot_table(index="ts", columns=st, values=var)
    cols_ = w.columns
    X = w.to_numpy()
    n = 0
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            ok = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            if ok.sum() > 1000 and np.mean(X[ok, i] == X[ok, j]) > 0.99:
                n += 1
    return n


out["identical_stations"] = dict(A=dup_stations(fa, "wa_station", "wa_temp"), B=dup_stations(fb, "wb_station", "wb_temp"))
print("station pairs with identical temperature series:", out["identical_stations"])

# 3e test-year forecast days that equal a training-year forecast for the same station / calendar day
def test_vs_train_copy(ftr, fte, st, var):
    a = ftr[ftr["round"] == 17][[st, "ts", var]].copy()
    b = fte[fte["round"] == 17][[st, "ts", var]].copy()
    a["md"] = a.ts.dt.strftime("%m-%d %H")
    b["md"] = b.ts.dt.strftime("%m-%d %H")
    m = b.merge(a, on=[st, "md"], suffixes=("_te", "_tr"))
    return float(np.mean(m[f"{var}_te"].round(3) == m[f"{var}_tr"].round(3)))


out["test_equals_train_share"] = dict(A_temp=test_vs_train_copy(fa, fa_te, "wa_station", "wa_temp"), B_temp=test_vs_train_copy(fb, fb_te, "wb_station", "wb_temp"))
print("share of test-year forecast values equal to a training-year value at the same station/calendar hour:", out["test_equals_train_share"])

json.dump(out, open(f"{CACHE}/review_checks.json", "w"), indent=1, default=str)
