"""Composition (G1/G2 split) experiments on a time hold-out.

    python 05_composition.py --cut 2024-06-01 --pred M3 [--oof <oof parquet>] [--tag x]

Candidates scored on the hold-out year (penalty W, 0.10 scale) for the sites with >= 90 % coverage:
  reference      organisers' current_group
  random         300 random balanced splits (mean / sd / min / max)
  twin_wa/_wb    greedy split inside each weather station
  sister         sister sites (same capacity and same WA+WB stations) are alternated across groups, the rest split by station
  oof_opt        local search on out-of-fold residuals of the years before the cut  (needs --oof)
  oof_best_rand  best of N random splits ranked on the out-of-fold residuals       (needs --oof)
  ceiling        local search on the hold-out residuals themselves (in-sample, not attainable)
plus the correlation between out-of-fold and hold-out excess across random splits (does selection transfer?).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from oibc_lib import *


def weighted_pair_corr(E, w, present):
    """Weighted correlation of site residual series, each pair measured on the hours where both sites report.
    Returns (corr NxN, overlap hours NxN, per-site weighted rms c_i)."""
    sw = np.sqrt(w)[:, None]
    Ew = E * sw
    P = present.astype(float)
    num = Ew.T @ Ew
    var_ij = ((E ** 2) * w[:, None]).T @ P                    # sum_t w e_i^2 over hours where j reports
    corr = num / np.sqrt(np.maximum(var_ij * var_ij.T, 1e-12))
    n = P.T @ P
    own = ((E ** 2) * w[:, None]).sum(0) / np.maximum((P * w[:, None]).sum(0), 1e-12)
    return corr, n, np.sqrt(own)


def pair_types(su):
    """0 sister (same capacity and stations), 1 same WA and WB, 2 same WA only, 3 same WB only, 4 other."""
    wa, wb = su.wa_station.astype(str).to_numpy(dtype=object), su.wb_station.astype(str).to_numpy(dtype=object)
    cp = su.capacity_kw.round(2).to_numpy()
    swa, swb = wa[:, None] == wa[None, :], wb[:, None] == wb[None, :]
    T = np.full(swa.shape, 4)
    T[swb & ~swa] = 3
    T[swa & ~swb] = 2
    T[swa & swb] = 1
    T[swa & swb & (cp[:, None] == cp[None, :])] = 0
    return T


def structured_cov(su, cap, Eo, wo, present_o, min_overlap=300, min_hours=800):
    """Residual covariance modelled by pair type: S_ij = rho_type * c_i * c_j, S_ii = c_i^2 (c_i from history or pooled capacity scaling)."""
    corr, n, c_own = weighted_pair_corr(Eo, wo, present_o)
    T = pair_types(su)
    hours = present_o.sum(0)
    iu = np.triu_indices(len(cap), 1)
    rho = {}
    for k in range(5):
        sel = (T[iu] == k) & (n[iu] >= min_overlap) & np.isfinite(corr[iu])
        rho[k] = float(np.average(np.clip(corr[iu][sel], -1, 1), weights=n[iu][sel])) if sel.sum() else 0.0
    good = hours >= min_hours
    sig_u = float(np.sqrt(np.sum(c_own[good] ** 2) / np.sum(cap[good] ** 2))) if good.any() else 0.1
    c = np.where(good, c_own, sig_u * cap)
    S = np.zeros((len(cap), len(cap)))
    for k in range(5):
        S[T == k] = rho[k]
    S = S * np.outer(c, c)
    np.fill_diagonal(S, c ** 2)
    return S, rho


def quad_search(S, cap, init, rng, lo=LO_FRAC, max_iter=400):
    """Minimise s'Ss over balanced +-1 assignments with greedy flips and swaps (vectorised)."""
    s = np.where(init, 1.0, -1.0)
    tot = cap.sum()
    d = np.diag(S)
    for _ in range(max_iter):
        g = S @ s - d * s
        share1 = cap[s > 0].sum()
        newshare = np.where(s > 0, share1 - cap, share1 + cap) / tot
        flip = np.where((newshare >= lo) & (newshare <= 1 - lo), -4 * s * g, np.inf)
        I, J = np.where(s > 0)[0], np.where(s < 0)[0]
        sw = -4 * (s[I] * g[I])[:, None] - 4 * (s[J] * g[J])[None, :] + 8 * S[np.ix_(I, J)] * (s[I][:, None] * s[J][None, :])
        nsw = (share1 - cap[I][:, None] + cap[J][None, :]) / tot
        sw = np.where((nsw >= lo) & (nsw <= 1 - lo), sw, np.inf)
        jf, (ii, jj) = int(np.argmin(flip)), np.unravel_index(int(np.argmin(sw)), sw.shape)
        if min(flip[jf], sw[ii, jj]) >= -1e-9:
            break
        if flip[jf] <= sw[ii, jj]:
            s[jf] = -s[jf]
        else:
            s[I[ii]], s[J[jj]] = -s[I[ii]], -s[J[jj]]
    return s > 0, float(s @ S @ s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="2024-06-01")
    ap.add_argument("--pred", required=True, help="prediction file name stem pred_<cut>_<pred>.parquet")
    ap.add_argument("--oof", default=None, help="path of out-of-fold predictions (site_code, ts, pred_u) for the years before the cut")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--n-select", type=int, default=2000)
    args = ap.parse_args()
    cut = K(f"{args.cut} 00:00")
    end = cut + pd.DateOffset(years=1)
    sites = load_sites().set_index("site_code")
    panel = pd.read_parquet(f"{CACHE}/panel.parquet", columns=["site_code", "ts", "energy", "cap", "elev"])
    val = pd.read_parquet(f"{CACHE}/pred_{args.cut}_{args.pred}.parquet")
    dv = error_matrix(val, panel, cut, end)
    U, cap, Ev, wv = dv["sites"], dv["cap"], dv["E"], dv["w"]
    su = sites.reindex(U)
    ref = (su.current_group == "A").values
    res = dict(n_sites=int(len(U)), cap_share=float(cap.sum() / sites.capacity_kw.sum()))
    res["W_nopool"], res["W_floor"], res["W_reference"] = W_nopool(Ev, wv), W_floor(Ev, wv), W_of(Ev, wv, ref)
    # the reference split is only comparable when it is balanced inside the evaluated subset (a lopsided split sits near the floor for free)
    res["ref_cap_share"] = float(cap[ref].sum() / cap.sum())
    res["ref_comparable"] = bool(0.40 <= res["ref_cap_share"] <= 0.60)
    same_share = []
    rng0 = np.random.default_rng(7)
    tries = 0
    while len(same_share) < 300 and tries < 300000 and res["ref_comparable"]:
        m = rng0.random(len(cap)) < 0.5
        tries += 1
        if abs(cap[m].sum() / cap.sum() - res["ref_cap_share"]) < 0.01:
            same_share.append(W_of(Ev, wv, m))
    if same_share:
        res["W_random_same_share_mean"], res["W_random_same_share_sd"] = float(np.mean(same_share)), float(np.std(same_share))
    rng = np.random.default_rng(1)
    rand_parts = [random_partition(cap, rng) for _ in range(max(args.n_random, args.n_select))]
    rw = np.array([W_of(Ev, wv, m) for m in rand_parts[:args.n_random]])
    res.update(W_random_mean=float(rw.mean()), W_random_sd=float(rw.std()), W_random_min=float(rw.min()), W_random_max=float(rw.max()))
    res["W_twin_wa"] = W_of(Ev, wv, twin_partition(su.wa_station.astype(str).values, cap))
    res["W_twin_wb"] = W_of(Ev, wv, twin_partition(su.wb_station.astype(str).values, cap))
    m_sis = keyed_twin(su.reset_index(), cap)
    res["W_sister"] = W_of(Ev, wv, m_sis) if feasible(cap, m_sis) else float("nan")
    m_sis_wb = keyed_twin(su.reset_index(), cap, "wb_station")
    res["W_sister_wb"] = W_of(Ev, wv, m_sis_wb) if feasible(cap, m_sis_wb) else float("nan")
    gid = sister_groups(su)
    res["n_sister_sites_in_universe"] = int(gid.notna().sum())
    res["sister_groups_kept_together_by_reference"] = int(np.mean([len(set(ref[(gid == g).values])) == 1 for g in gid.dropna().unique()]) * gid.dropna().nunique())
    res["sister_groups"] = int(gid.dropna().nunique())
    best = None
    for _ in range(6):
        m, W = local_search(Ev, wv, cap, random_partition(cap, rng), rng)
        best = (m, W) if best is None or W < best[1] else best
    res["W_ceiling_insample"] = best[1]
    if args.oof:
        oof = pd.read_parquet(args.oof)
        d0 = error_matrix(oof, panel, K("2021-01-01"), cut, min_cov=0.0, U=U)
        Eo, wo = np.nan_to_num(d0["E"]), d0["w"]
        bo = None
        for _ in range(6):
            m, W = local_search(Eo, wo, cap, random_partition(cap, rng), rng)
            bo = (m, W) if bo is None or W < bo[1] else bo
        res["W_oof_opt"] = W_of(Ev, wv, bo[0])
        # refine the sister split on residuals
        m2, _ = local_search(Eo, wo, cap, m_sis.copy(), rng, iters=40, swap_tries=1500)
        res["W_sister_then_oof_refine"] = W_of(Ev, wv, m2)
        # selection transfer: do splits that look good on past residuals look good on the unseen year?
        sel = rand_parts[:args.n_select]
        w_o = np.array([W_of(Eo, wo, m) for m in sel])
        w_v = np.array([W_of(Ev, wv, m) for m in sel])
        from scipy.stats import spearmanr
        res["selection_spearman"] = float(spearmanr(w_o, w_v).statistic)
        top = np.argsort(w_o)[:max(1, len(sel) // 100)]
        res["W_oof_best_of_random"] = float(w_v[top[0]])
        res["W_oof_top1pct_mean"] = float(w_v[top].mean())
        res["W_random_val_mean_n_select"] = float(w_v.mean())
        # structured-covariance anti-clustering: residual correlation by pair type, many restarts of a vectorised search
        present_o = ~np.isnan(d0["Ak"])
        S, rho = structured_cov(su, cap, Eo, wo, present_o)
        res["rho_by_type"] = rho
        cand = [quad_search(S, cap, random_partition(cap, rng), rng) for _ in range(40)] + [quad_search(S, cap, m_sis.copy(), rng), quad_search(S, cap, ref.copy(), rng)]
        mc, fc = min(cand, key=lambda t: t[1])
        res["W_cov_struct"] = W_of(Ev, wv, mc)
        res["cov_struct_agrees_with_sister"] = float(max(np.mean(mc == m_sis), np.mean(mc != m_sis)))
        fq = np.array([float((np.where(m, 1.0, -1.0)) @ S @ (np.where(m, 1.0, -1.0))) for m in sel])
        res["cov_criterion_spearman_with_val_W"] = float(spearmanr(fq, w_v).statistic)
        # (i) does the covariance criterion pick good splits out of random ones?  (ii) how far should the optimiser run from the sister split?
        top_f = np.argsort(fq)[:max(1, len(sel) // 100)]
        res["W_cov_top1pct_mean_val"] = float(w_v[top_f].mean())
        path = {}
        for k in (0, 2, 5, 10, 25, 100, 400):
            mk, fk = quad_search(S, cap, m_sis.copy(), rng, max_iter=k) if k else (m_sis.copy(), 0.0)
            path[str(k)] = W_of(Ev, wv, mk)
        res["W_val_along_optimiser_path_from_sister"] = path
        res["W_sister_on_oof"] = W_of(Eo, wo, m_sis)
        res["W_reference_on_oof"] = W_of(Eo, wo, ref)
    json.dump(res, open(f"{CACHE}/composition_{args.cut}_{args.pred}{args.tag}.json", "w"), indent=1)
    fl = res["W_floor"]
    print(f"reference split holds {res['ref_cap_share']:.1%} of the subset's capacity (comparable: {res['ref_comparable']})")
    print(f"universe {res['n_sites']} sites ({res['cap_share']:.1%} of capacity); sisters in universe {res['n_sister_sites_in_universe']} in {res['sister_groups']} groups; "
          f"kept together by the reference split: {res['sister_groups_kept_together_by_reference']}/{res['sister_groups']}")
    for k in ("W_floor", "W_ceiling_insample", "W_cov_struct", "W_oof_opt", "W_sister_then_oof_refine", "W_oof_best_of_random", "W_sister", "W_sister_wb", "W_twin_wa", "W_twin_wb", "W_random_mean", "W_reference"):
        if k in res:
            print(f"  {k:26s} {res[k]/1e6:9.2f}M  ({100*(res[k]/fl-1):+5.2f}% above floor)")
    print(f"  random sd {res['W_random_sd']/1e6:.2f}M  min {res['W_random_min']/1e6:.2f} max {res['W_random_max']/1e6:.2f}")
    if "rho_by_type" in res:
        print("  residual correlation by pair type (sister, same WA+WB, same WA, same WB, other):", {k: round(v, 3) for k, v in res["rho_by_type"].items()},
              "| cov-criterion vs hold-out W Spearman:", round(res["cov_criterion_spearman_with_val_W"], 3))
    if "W_cov_top1pct_mean_val" in res:
        print(f"  covariance criterion: top-1% of {args.n_select} random splits average {100*(res['W_cov_top1pct_mean_val']/fl-1):+.2f}% above floor on the hold-out (random mean {100*(res['W_random_val_mean_n_select']/fl-1):+.2f}%)")
        print("  hold-out W along the optimiser path from the sister split (iterations: excess%):", {k: round(100 * (v / fl - 1), 2) for k, v in res["W_val_along_optimiser_path_from_sister"].items()})
    if "selection_spearman" in res:
        print(f"  selection transfer (Spearman of OOF vs hold-out W across {args.n_select} random splits): {res['selection_spearman']:.3f}; "
              f"top-1% by OOF averages {100*(res['W_oof_top1pct_mean']/fl-1):+.2f}% vs random {100*(res['W_random_val_mean_n_select']/fl-1):+.2f}%")


if __name__ == "__main__":
    main()
