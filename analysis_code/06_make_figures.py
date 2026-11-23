"""Figures for the report. Reads the result files written by 01b/03/04/05 from <CACHE>; writes PNGs to <CACHE>/figs.

    python 06_make_figures.py --config figures_config.json
The config names which experiment is shown for each ladder step (see the bottom of this file for the default).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from oibc_lib import *

SURF, INK, INK2, MUTE, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150, "font.family": "DejaVu Sans"})
FIGS = f"{CACHE}/figs"
os.makedirs(FIGS, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{FIGS}/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_penalty_geometry():
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    E = 10
    a = np.linspace(-6, 16, 500)
    ax.plot(a, np.abs(a) + np.abs(E - a), color=BLUE, lw=2)
    ax.axvspan(0, E, color=BLUE, alpha=0.10, lw=0)
    ax.set_xlabel("G1 error  a  (kWh)   —   G2 error is  b = E − a,  with total error E = +10 fixed by the forecasts")
    ax.set_ylabel("Penalty  |a| + |b|")
    ax.annotate("Floor = |E|: both groups err the same way\n(a between 0 and E) — any such split is equally good", xy=(5, 10), xytext=(0.2, 15.3), fontsize=8.6, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
    ax.annotate("Opposite signs: penalty = |a − b|\n> |E|, the only part a split can remove", xy=(-4, 18), xytext=(-5.8, 22.3), fontsize=8.6, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
    ax.set_ylim(0, 28)
    ax.set_xlim(-6, 16)
    ax.xaxis.grid(False)
    save(fig, "f1_penalty_geometry")


def fig_fleet_overview():
    sites = load_sites()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.32))
    ax = axs[0]
    bins = [0, 50, 100.5, 200, 400, 700, 1000, 1400]
    labs = ["<50", "50-\n100", "100-\n200", "200-\n400", "400-\n700", "700-\n1000", ">1000"]
    cnt = pd.cut(sites.capacity_kw, bins).value_counts().sort_index()
    ax.bar(range(len(cnt)), cnt.values, color=BLUE, width=0.62)
    for i, v in enumerate(cnt.values):
        ax.text(i, v + 3, str(v), ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(cnt)))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("Number of sites")
    ax.set_title("253 sites: capacity (kW)")
    ax.xaxis.grid(False)
    ax = axs[1]
    panel = pd.read_parquet(f"{CACHE}/panel.parquet", columns=["site_code", "ts"])
    first = panel.groupby("site_code").ts.min()
    capk = sites.set_index("site_code").capacity_kw
    months = pd.period_range("2021-01", "2025-06", freq="M")
    capact = [capk.reindex(first[first <= m.end_time.tz_localize(TZ)].index).sum() / 1000 for m in months]
    ax.plot([m.to_timestamp() for m in months], capact, color=BLUE, lw=2)
    ax.set_ylabel("Capacity with history (MW)")
    ax.set_title("Fleet history builds up late")
    ax.set_ylim(0, 48)
    ax.text(pd.Timestamp("2021-03-01"), 44, "42.7 MW at the end of training", fontsize=8.5, color=INK2)
    x2 = pd.Timestamp("2023-06-01")
    ax.annotate("only ~20% of capacity\nhas >2 years of data", xy=(x2, np.interp(x2.value, [m.to_timestamp().value for m in months], capact)), xytext=(pd.Timestamp("2021-02-01"), 22),
                fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2021, 2026)])
    ax.set_xticklabels([str(y) for y in range(2021, 2026)], fontsize=8.5)
    save(fig, "f2_fleet_overview")


def fig_map():
    sites = load_sites()
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for g, col, lab in (("A", BLUE, "reference group A (134 sites)"), ("B", ORANGE, "reference group B (119 sites)")):
        s = sites[sites.current_group == g]
        ax.scatter(s.lon, s.lat, s=s.capacity_kw / 6 + 6, color=col, alpha=0.75, edgecolor=SURF, linewidth=0.6, label=lab)
    ax.set_xlabel("Longitude (shifted ≤ 2 km)")
    ax.set_ylabel("Latitude")
    ax.set_title("Sites cluster in the south-east")
    leg = ax.legend(frameon=False, loc="upper right", fontsize=8, markerscale=0.5, bbox_to_anchor=(1.0, 1.0))
    ax.set_xlim(126.4, 130.6)
    ax.set_aspect(1.25)
    save(fig, "f3_site_map")


def fig_usefulness():
    chk = json.load(open(f"{CACHE}/review_checks.json"))
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw=dict(wspace=0.55, width_ratios=[1, 1.25]))
    ax = axs[0]
    vars_ = ["Cloud", "Humidity", "Wind"]
    sk = json.load(open(f"{CACHE}/data_checks.json"))["skill_vs_observed"]
    A = [sk["wa_cloud"]["corr_clean"], sk["wa_humidity"]["corr_clean"], sk["wa_wind"]["corr_clean"]]
    B = [sk["wb_cloud"]["corr_clean"], sk["wb_humidity"]["corr_clean"], sk["wb_wind"]["corr_clean"]]
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2 - 0.01, A, w, color=BLUE, label="Provider A")
    ax.bar(x + w / 2 + 0.01, B, w, color=ORANGE, label="Provider B")
    for i, (a, b) in enumerate(zip(A, B)):
        ax.text(i - w / 2 - 0.01, a + 0.02, f"{a:.2f}", ha="center", fontsize=8.2, color=INK2)
        ax.text(i + w / 2 + 0.01, b + 0.02, f"{b:.2f}", ha="center", fontsize=8.2, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(vars_)
    ax.set_ylim(0, 1.15)
    ax.set_title("vs observed weather")
    ax.set_ylabel("Correlation")
    ax.xaxis.grid(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.2)
    ax = axs[1]
    u = chk["usefulness_corr"]
    items = [("A humidity", u["wa_humidity_r17"], BLUE), ("B humidity", u["wb_rel_hum_r17"], ORANGE), ("A+B cloud mean", u["cloud_mean_r17"], MUTE), ("A cloud, day mean", u["wa_cloud_r17_dmean"], BLUE),
             ("A cloud", u["wa_cloud_r17"], BLUE), ("B cloud", u["wb_cloud_r17"], ORANGE)]
    y = np.arange(len(items))[::-1]
    for yi, (lab, v, col) in zip(y, items):
        ax.barh(yi, abs(v), color=col, height=0.6)
        ax.text(abs(v) + 0.01, yi, f"{v:+.2f}", va="center", fontsize=8.4, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=8.4)
    ax.set_xlim(0, 0.8)
    ax.yaxis.grid(False)
    ax.set_title("vs generation (|corr|)")
    ax.set_xlabel("after removing month × hour means")
    save(fig, "f4_forecast_usefulness")


def fig_smp():
    chk = json.load(open(f"{CACHE}/data_checks.json")) if os.path.exists(f"{CACHE}/data_checks.json") else None
    tr = load_smp_train()
    te = pd.read_csv(f"{DATA}/smp.csv")
    te["ts"] = pd.to_datetime(te.ts)
    last = tr[tr.index > K("2024-06-01 00:00")]
    prof_tr = last.groupby(last.index.hour).mean()
    prof_te = te.groupby(te.ts.dt.hour).smp.mean()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.28, width_ratios=[1.5, 1]))
    ax = axs[0]
    ax.axvspan(11, 15, color=ORANGE, alpha=0.12, lw=0)
    ax.text(13, 176, "k = 3", ha="center", fontsize=8.5, color=INK2)
    ax.plot(prof_tr.index, prof_tr.values, color=BLUE, lw=2)
    ax.plot(prof_te.index, prof_te.values, color=ORANGE, lw=2)
    ax.text(16.3, 141.5, "training, last year", color=BLUE, fontsize=8.5)
    ax.text(16.3, 103, "scoring (test) year", color=ORANGE, fontsize=8.5)
    ax.set_xlabel("Hour label (ts)")
    ax.set_ylabel("Mean SMP (KRW/kWh)")
    ax.set_ylim(70, 185)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_title("SMP by hour of day")
    ax = axs[1]
    yr = tr.groupby(tr.index.year).mean()
    vals = [float(yr.loc[y]) for y in (2021, 2022, 2023, 2024, 2025)] + [float(te.smp.mean())]
    ax.bar(range(6), vals, color=[BLUE] * 5 + [ORANGE], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 4, f"{v:.0f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["2021", "2022", "2023", "2024", "2025*", "Test\nyear"], fontsize=8)
    ax.set_title("Mean SMP by year")
    ax.set_ylim(0, 225)
    ax.xaxis.grid(False)
    save(fig, "f5_smp")


def ladder_panel(ax, items, title, oracle=None):
    n = len(items) + (1 if oracle is not None else 0)
    y = np.arange(n)[::-1]
    for yi, (lab, v, kind) in zip(y, items):
        col = {"base": MUTE, "gain": BLUE, "null": GRID}[kind]
        ax.barh(yi, v, color=col, height=0.58)
        ax.text(max(v, 0) + 0.3, yi, f"{v:+.1f}", va="center", fontsize=8.6, color=INK2)
    labs = [i[0] for i in items]
    if oracle is not None:
        ax.barh(y[-1], oracle[1], color=MUTE, height=0.58, hatch="////", edgecolor="#c9c8c2", linewidth=0)
        ax.text(oracle[1] + 0.3, y[-1], f"{oracle[1]:+.1f}", va="center", fontsize=8.6, color=INK2)
        labs.append(oracle[0])
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8.3)
    ax.yaxis.grid(False)
    ax.set_title(title)


def fig_ladder(cfg):
    rb = pd.read_csv(f"{CACHE}/results_2024-06-01.csv").set_index("exp")
    ra = pd.read_csv(f"{CACHE}/results_2023-06-01.csv").set_index("exp") if os.path.exists(f"{CACHE}/results_2023-06-01.csv") else None
    sc = lambda r, k: float(r.loc[k, "score_vs_M0"])
    m3 = np.mean([sc(rb, k) for k in cfg["m3_runs"]])
    items = [("Baseline replica", 0.0, "base"), ("All weather fields, both rounds\n(mean of 3 runs)", m3, "gain"),
             ("+ same-day forecast context,\nno coordinates", sc(rb, cfg["ctx"]), "gain"),
             ("+ wide context, larger trees\n(2-seed average)", sc(rb, cfg["wide"]), "gain"),
             ("+ previous-day features,\nfour-model blend", sc(rb, cfg["final"]), "gain")]
    pj = f"{CACHE}/composition_2024-06-01_{cfg['comp_pred']['2024-06-01']}.json"
    if os.path.exists(pj):
        cj = json.load(open(pj))
        items.append(("+ sister-aware split\n(replaces reference split)", 100 * (1 - cj["W_sister"] / (float(rb.loc["M0", "W_cur"]))), "gain"))
    fig, axs = plt.subplots(1, 2 if ra is not None else 1, figsize=(10.2 if ra is not None else 6.6, 4.0), gridspec_kw=dict(wspace=1.05) if ra is not None else None)
    axs = np.atleast_1d(axs)
    ladder_panel(axs[0], items, f"Hold-out year 1: 2024-06 \u2192 2025-05 ({int(rb.loc['M0','n_sites'])} sites)", oracle=("Oracle: observed weather\n(not usable)", sc(rb, "ORACLE")) if "ORACLE" in rb.index else None)
    axs[0].set_xlabel("Normalised-score points vs baseline replica")
    axs[0].set_xlim(0, 34 if "ORACLE" in rb.index else 20)
    if ra is not None:
        m3a = np.mean([sc(ra, k) for k in ("M3", "M3_s1") if k in ra.index])
        itemsA = [("Baseline replica", 0.0, "base"), ("All weather fields, both rounds\n(mean of 2 runs)", m3a, "gain"),
                  ("+ same-day forecast context,\nno coordinates", sc(ra, "F"), "gain"), ("+ wide context, larger trees", sc(ra, cfg["a_wide"]), "gain"),
                  ("+ previous-day features,\nfour-model blend", sc(ra, cfg["a_final"]), "gain")]
        ladder_panel(axs[1], itemsA, f"Hold-out year 2: 2023-06 \u2192 2024-05 ({int(ra.loc['M0','n_sites'])} sites)")
        axs[1].set_xlabel("Normalised-score points vs baseline replica")
        axs[1].set_xlim(0, 20)
    fig.suptitle("What each change is worth (hold-out years; single runs unless stated, noise \u2248 \u00b10.4)", x=0.01, y=1.08, ha="left", fontsize=10.5, fontweight="bold")
    save(fig, "f6_forecast_ladder")


def fig_composition(cfg):
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.2), gridspec_kw=dict(wspace=0.95))
    for ax, cut, ttl in zip(axs, ("2024-06-01", "2023-06-01"), ("Hold-out year 1", "Hold-out year 2")):
        path = f"{CACHE}/composition_{cut}_{cfg['comp_pred'][cut]}.json"
        if not os.path.exists(path):
            ax.axis("off")
            continue
        r = json.load(open(path))
        fl = r["W_floor"]
        pct = lambda k: 100 * (r[k] / fl - 1)
        rows = [("Optimised on the hold-out year\n(ceiling, not attainable)", pct("W_ceiling_insample"), "ref"),
                ("Learned on past residuals,\nstructured covariance", pct("W_cov_struct"), "gain"),
                ("Sister sites alternated,\nrest split by station", pct("W_sister"), "gain"),
                ("Station-stratified (WB)", pct("W_twin_wb"), "gain"),
                ("Random split, mean of 300", pct("W_random_mean"), "base")]
        if r.get("ref_comparable", True):
            rows.append(("Reference split (current_group)", pct("W_reference"), "base"))
        y = np.arange(len(rows))[::-1]
        for yi, (lab, v, kind) in zip(y, rows):
            ax.barh(yi, v, color={"ref": MUTE, "gain": BLUE, "base": ORANGE}[kind], height=0.58)
            ax.text(v + 0.05, yi, f"+{v:.2f}%", va="center", fontsize=8.5, color=INK2)
        lo, hi = 100 * (r["W_random_min"] / fl - 1), 100 * (r["W_random_max"] / fl - 1)
        yr = y[4] - 0.42   # range whisker under the random-split bar
        ax.plot([lo, hi], [yr, yr], color=INK2, lw=1)
        ax.plot([lo, lo], [yr - 0.06, yr + 0.06], color=INK2, lw=1)
        ax.plot([hi, hi], [yr - 0.06, yr + 0.06], color=INK2, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([x[0] for x in rows], fontsize=8.2)
        ax.set_xlabel("Penalty above the pooled-fleet floor (%)")
        ax.yaxis.grid(False)
        ax.set_xlim(0, max(4.0, hi + 0.8))
        ax.set_title(f"{ttl} ({r['n_sites']} sites)")
        if not r.get("ref_comparable", True):
            ax.text(0.99, 0.97, f"reference split not shown:\ngroup A holds {100*r['ref_cap_share']:.0f}% of this\nsubset's capacity", transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=INK2)
    fig.suptitle("Composition headroom: penalty above the floor Σ SMP·k·|E_total|", x=0.01, y=1.08, ha="left", fontsize=10.5, fontweight="bold")
    save(fig, "f8_composition")


def fig_pair_types(cfg):
    path = f"{CACHE}/composition_2024-06-01_{cfg['comp_pred']['2024-06-01']}.json"
    if not os.path.exists(path):
        return
    r = json.load(open(path))
    if "rho_by_type" not in r:
        return
    rho = r["rho_by_type"]
    names = ["Sisters\n(same capacity,\nsame stations)", "Same WA and\nWB station", "Same WA\nstation only", "Same WB\nstation only", "Different\nstations"]
    vals = [rho[str(i)] for i in range(5)]
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.bar(range(5), vals, color=[BLUE] * 4 + [MUTE], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=8.2)
    ax.set_ylabel("Mean residual correlation")
    ax.set_ylim(0, 1.1)
    ax.xaxis.grid(False)
    ax.set_title("Forecast errors of site pairs, by pair type (out-of-fold residuals)")
    save(fig, "f9_pair_correlation")


def fig_penalty_by_hour_month(cfg):
    cut, end = K("2024-06-01 00:00"), K("2025-06-01 00:00")
    panel = pd.read_parquet(f"{CACHE}/panel.parquet", columns=["site_code", "ts", "energy", "cap", "elev"])
    pred = pd.read_parquet(f"{CACHE}/pred_2024-06-01_{cfg['final']}.parquet")
    d = error_matrix(pred, panel, cut, end)
    cg = load_sites().set_index("site_code").current_group
    mask = (cg.reindex(d["sites"]) == "A").values
    a, b = d["E"][:, mask].sum(1), d["E"][:, ~mask].sum(1)
    pen = 0.10 * d["w"] * (np.abs(a) + np.abs(b))
    df = pd.DataFrame({"hour": d["ts"].hour, "month": d["ts"].month, "pen": pen})
    byh, bym = df.groupby("hour").pen.sum(), df.groupby("month").pen.sum()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.25, width_ratios=[1.35, 1]))
    ax = axs[0]
    ax.bar(byh.index, byh.values / 1e6, color=[ORANGE if 11 <= i <= 15 else BLUE for i in byh.index], width=0.72)
    ax.set_xlabel("Hour label (ts)")
    ax.set_ylabel("Penalty W (M KRW)")
    ax.set_title("By hour (k = 3 in orange)")
    ax.set_xlim(4.4, 20.6)
    ax.set_xticks([6, 8, 10, 12, 14, 16, 18, 20])
    ax.xaxis.grid(False)
    share = byh.loc[11:15].sum() / byh.sum()
    ax.text(0.97, 0.62, f"hours 11-15:\n{share*100:.0f}% of W", transform=ax.transAxes, ha="right", fontsize=8.8, color=INK2)
    ax = axs[1]
    ax.bar(bym.index, bym.values / 1e6, color=BLUE, width=0.7)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(list("JFMAMJJASOND"), fontsize=8)
    ax.set_title("By month")
    ax.xaxis.grid(False)
    save(fig, "f7_penalty_by_hour_month")
    json.dump(dict(k3_share=float(share), worst_months=[int(m) for m in bym.sort_values(ascending=False).index[:2]]), open(f"{CACHE}/penalty_shares.json", "w"))


DEFAULT_CONFIG = dict(m3_runs=["M3", "M3_s1", "M3_s2"], ctx="CTX2_nocoord", wide="G_2seeds", final="B_v2", a_wide="G", a_final="A_four",
                      comp_pred={"2024-06-01": "B_v2", "2023-06-01": "A_four"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        cfg.update(json.load(open(args.config)))
    jobs = dict(geometry=fig_penalty_geometry, overview=fig_fleet_overview, map=fig_map, usefulness=fig_usefulness, smp=fig_smp,
                ladder=lambda: fig_ladder(cfg), composition=lambda: fig_composition(cfg), pairs=lambda: fig_pair_types(cfg), hourmonth=lambda: fig_penalty_by_hour_month(cfg))
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        fn()
        print("figure", name, "ok")


if __name__ == "__main__":
    main()
