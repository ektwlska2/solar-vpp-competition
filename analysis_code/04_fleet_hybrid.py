"""Fleet-total ("hybrid") forecasts on a time hold-out.

Idea: the penalty is bounded below by the error of the pooled fleet, so forecast the *fleet total* with a median (L1) model
trained on fleet-level aggregates, then hand it to the two groups in proportion to the site-model shares.

    python 04_fleet_hybrid.py --cut 2024-06-01 --pred ALL [--panel panel.parquet] [--oof oof_2024-06-01_ALL.parquet] [--nmin 0]

Variants printed (penalty W, 0.10 scale, reference split and the sister split, same sites as 03):
  site_sum         sum of site forecasts (what 03 reports)
  fleet_raw        fleet L1 model on aggregated raw forecasts, shares from site model
  fleet_all        same with context / blend / physics features aggregated as well
  fleet_all_cap    same but shares proportional to capacity (ignores the site model completely)
  stack            fleet L1 model that also sees the out-of-fold site-sum forecast (needs --oof)
"""
from __future__ import annotations
import argparse, importlib, json, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from oibc_lib import *

warnings.filterwarnings("ignore")
probes = importlib.import_module("03_forecast_probes")

P_FLEET = dict(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.5, reg_lambda=5.0)


def agg_by_ts(df, cols, extra=()):
    """Capacity-weighted mean of `cols` over the sites present at each hour, plus capacity, energy and site count."""
    cap = df.cap.to_numpy("float64")
    g = df.groupby("ts", sort=True)
    out = {"capsum": g.cap.sum(), "esum": g.energy.sum(), "nsites": g.size()}
    key = df.ts
    for c in list(cols) + list(extra):
        v = df[c].to_numpy("float64")
        ok = ~np.isnan(v)
        num = pd.Series(np.where(ok, v * cap, 0.0), index=df.index).groupby(key).sum()
        den = pd.Series(np.where(ok, cap, 0.0), index=df.index).groupby(key).sum()
        out[c] = num / den.replace(0, np.nan)
    r = pd.DataFrame(out)
    r["ratio"] = r.esum / r.capsum
    return r


def fit_fleet(tr_a, feats, seed=0):
    w = tr_a.capsum * tr_a.w
    m = lgb.LGBMRegressor(objective="l1", verbose=-1, n_jobs=10, random_state=seed, **P_FLEET)
    return m.fit(tr_a[feats], tr_a.ratio, sample_weight=w / w.mean())


def hybrid_W(Pk, Ak, ts, w, T, masks, shares="site"):
    """Penalty of the hybrid forecast for each named partition. T: fleet total (kWh) indexed like ts."""
    present = ~np.isnan(Ak)
    Pp, Aa = np.where(present, np.nan_to_num(Pk), 0.0), np.where(present, Ak, 0.0)
    capv = None
    out = {}
    for name, mask in masks.items():
        if isinstance(shares, str) and shares == "site":
            s1 = Pp[:, mask].sum(1) / np.maximum(Pp.sum(1), 1e-9)
        else:
            s1 = shares[mask].sum() / shares.sum() * np.ones(len(ts))
        a = s1 * T - Aa[:, mask].sum(1)
        b = (1 - s1) * T - Aa[:, ~mask].sum(1)
        out[name] = (0.10 * float((w * (np.abs(a) + np.abs(b))).sum()), 0.10 * float((w * np.abs(a + b)).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="2024-06-01")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--panel", default="panel.parquet")
    ap.add_argument("--oof", default=None)
    ap.add_argument("--nmin", type=int, default=0, help="train the fleet model only on hours with at least this many sites reporting")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    cut = K(f"{args.cut} 00:00")
    end = cut + pd.DateOffset(years=1)
    smp = load_smp_train()
    sites = load_sites().set_index("site_code")
    p = pd.read_parquet(f"{CACHE}/{args.panel}")
    full = p[["site_code", "ts", "energy", "cap", "elev"]].copy()
    p = p[(p.elev > 0)].reset_index(drop=True)
    g = probes.groups(p.columns)
    base_cols = [c for c in probes.only("raw", "geo")(g) if c not in ("cap", "lat", "lon")]
    all_cols = [c for c in probes.only("raw", "ctx", "fleet", "blend", "phy", "geo")(g) if c not in ("cap", "lat", "lon")]
    val_pred = pd.read_parquet(f"{CACHE}/pred_{args.cut}_{args.pred}.parquet")
    dv = error_matrix(val_pred, full, cut, end)
    U, cap, ts_idx, wv = dv["sites"], dv["cap"], dv["ts"], dv["w"]
    su = sites.reindex(U)
    masks = {"reference": (su.current_group == "A").values, "sister": keyed_twin(su.reset_index(), cap)}
    # ----- aggregated training data
    tr = p[(p.ts <= cut) & p.u.notna()].copy()
    va = p[(p.ts > cut) & (p.ts <= end) & p.site_code.isin(U) & p.u.notna()].copy()
    for d in (tr, va):
        d["w"] = hour_weights(pd.DatetimeIndex(d.ts), smp).astype("float32")
    tr_pred = None
    if args.oof:
        oof = pd.read_parquet(f"{CACHE}/{args.oof}") if not args.oof.startswith("/") else pd.read_parquet(args.oof)
        tr = tr.merge(oof.rename(columns={"pred_u": "oof_u"}), on=["site_code", "ts"], how="left")
    va = va.merge(val_pred.rename(columns={"pred_u": "site_u"}), on=["site_code", "ts"], how="left")
    ta_base = agg_by_ts(tr, all_cols, extra=(["oof_u"] if args.oof else []))
    ta_base["w"] = hour_weights(pd.DatetimeIndex(ta_base.index), smp)
    va_a = agg_by_ts(va, all_cols, extra=["site_u"])
    print(f"[{time.time()-t0:5.0f}s] aggregated: train hours {len(ta_base):,}, val hours {len(va_a):,}", flush=True)
    ta = ta_base[ta_base.nsites >= args.nmin] if args.nmin else ta_base
    Pk, Ak = dv["Pk"], dv["Ak"]
    results = {}

    def evaluate(name, T_ratio, shares="site"):
        T = pd.Series(T_ratio * va_a.capsum.to_numpy(), index=va_a.index).reindex(ts_idx).fillna(0.0).to_numpy()
        ev = hybrid_W(Pk, Ak, ts_idx, wv, T, masks, shares)
        results[name] = {k: dict(W=v[0], floor=v[1]) for k, v in ev.items()}
        print(f"  {name:16s} " + "  ".join(f"{k}: W {v[0]/1e6:7.2f}M floor {v[1]/1e6:7.2f}M" for k, v in ev.items()), flush=True)

    # allocation diagnostic: give the groups the TRUE fleet total and split it by site-model shares; whatever penalty remains is
    # purely the error in how the total is divided between G1 and G2 (the part a better split / better share model can reduce)
    T_true = pd.Series(np.where(~np.isnan(Ak), Ak, 0.0).sum(1), index=ts_idx).to_numpy()
    ev = hybrid_W(Pk, Ak, ts_idx, wv, T_true, masks, "site")
    results["alloc_oracle_total"] = {k: dict(W=v[0], floor=v[1]) for k, v in ev.items()}
    print("  alloc_oracle_total " + "  ".join(f"{k}: W {v[0]/1e6:7.2f}M" for k, v in ev.items()), flush=True)
    # site-sum baseline (no fleet model): T = sum of site forecasts on the present set, shares site-based
    ssum = (va_a.site_u * va_a.capsum).to_numpy()
    evaluate("site_sum", va_a.site_u.to_numpy())
    for name, cols in (("fleet_raw", base_cols), ("fleet_all", all_cols)):
        m = fit_fleet(ta, cols)
        evaluate(name, np.clip(m.predict(va_a[cols]), 0, None))
        if name == "fleet_all":
            w_ = ta.capsum * ta.w
            m2 = lgb.LGBMRegressor(objective="l2", verbose=-1, n_jobs=10, random_state=0, **P_FLEET).fit(ta[cols], ta.ratio, sample_weight=w_ / w_.mean())
            evaluate("fleet_all_L2", np.clip(m2.predict(va_a[cols]), 0, None))
            evaluate("fleet_all_cap", np.clip(m.predict(va_a[cols]), 0, None), shares=cap)
            # bias diagnostic: how would a global multiplier change W (not used for selection)
            for alpha in (0.97, 1.03):
                evaluate(f"fleet_all x{alpha}", alpha * np.clip(m.predict(va_a[cols]), 0, None))
    # forecasting each group on its own: the fleet-trained model applied to the aggregated features of each group
    m_all = fit_fleet(ta, all_cols)
    gmask = pd.Series(masks["reference"], index=U)
    va["grp"] = va.site_code.map(gmask).astype(int)
    parts = {}
    for gval in (0, 1):
        sub = va[va.grp == gval]
        a_ = agg_by_ts(sub, all_cols)
        parts[gval] = (a_, np.clip(m_all.predict(a_[all_cols]), 0, None) * a_.capsum.to_numpy())
    idx = ts_idx
    e0 = parts[0][1] - parts[0][0].esum.to_numpy()
    e1 = parts[1][1] - parts[1][0].esum.to_numpy()
    s0 = pd.Series(e0, index=parts[0][0].index).reindex(idx).fillna(0.0).to_numpy()
    s1 = pd.Series(e1, index=parts[1][0].index).reindex(idx).fillna(0.0).to_numpy()
    results["group_direct"] = {"reference": dict(W=0.10 * float((wv * (np.abs(s0) + np.abs(s1))).sum()), floor=0.10 * float((wv * np.abs(s0 + s1)).sum()))}
    print(f"  group_direct       reference: W {results['group_direct']['reference']['W']/1e6:7.2f}M floor {results['group_direct']['reference']['floor']/1e6:7.2f}M", flush=True)
    if args.oof:
        ta_s = ta.dropna(subset=["oof_u"])
        cols_s = all_cols + ["oof_u"]
        va_s = va_a.rename(columns={"site_u": "oof_u"})
        m = fit_fleet(ta_s, cols_s)
        evaluate("stack", np.clip(m.predict(va_s[cols_s]), 0, None))
    json.dump(results, open(f"{CACHE}/hybrid_{args.cut}_{args.pred}{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
