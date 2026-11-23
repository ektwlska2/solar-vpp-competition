"""Family blend and day-level shrink of the final forecast (used by 08_make_submission.py and reblend_submission.py, so both give identical numbers).

family_blend(tree_u, nn_u, share)   (1 - share) x mean(tree members) + share x mean(network members), utilisation per row
day_shrink(u, ts, cap, total_cap, beta, cf_ref=CF_REF)
    factor exp(-beta (cf_day - cf_ref)) for every row; cf_day = predicted fleet capacity factor of the hour labels 11-15 of the day (sum of forecast kWh / (total capacity x 5 h));
    days without a forecast (cf <= 0.02) get factor 1.  cf_ref is a FIXED constant (CF_REF, the mean predicted midday capacity factor of the hold-out year 2024-06 -> 2025-05, i.e. training-period
    information).  Builds up to V10 centred the factor on the mean of the scoring period's own forecasts, which is a whole-period statistic (notice section 6: no normalising with the evaluation
    period as a whole); the two references differ by 0.003 (factor change 0.03 %).  The legacy option (cf_ref=None) was removed on 2026-11-11: the organisers do not rank files that contain a whole-period statistic; cf_ref must be a constant.
offday_correction(u, ts, cap, cfg=OFFDAY)
    grid-operator output reductions on low-demand days: plants > 500 kW produce less than the weather-only model says on weekends / public holidays in spring and autumn (see calendar_features.py);
    factor 1 - delta(month) x profile(hour label) for plants above big_kw on days off; every parameter comes from the hold-out windows 2023-24, 2024-25 and Jan-May 2025 (README, section "off-day correction").
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CF_REF = 0.50                                                                                    # fixed day-shrink reference (see day_shrink)
OFFDAY = dict(big_kw=500.0,
              delta={3: 0.15, 4: 0.15, 5: 0.15, 9: 0.08, 10: 0.08, 11: 0.08},                    # peak shortfall by month (spring: hold-out optimum 0.12-0.20; autumn: 0.08-0.20, taken at the low end)
              profile={11: 0.25, 12: 0.65, 13: 1.0, 14: 1.0, 15: 0.70, 16: 0.70, 17: 0.85, 18: 0.85})   # shape over the hour labels (out-of-fold ratio of plants > 500 kW on days off, spring 2025)


def family_blend(tree_u: list, nn_u: list, share: float, trim: dict | None = None, ts=None, cap=None) -> np.ndarray:
    """(1 - share) x tree family + share x network family.  With `trim` (dict(tol, hard, floor), needs ts and cap) the network share of a DAY is reduced when the two families disagree
    about the fleet's midday energy of that day (labels 11-15): factor 1 while |network/trees - 1| <= tol, `floor` once it is >= hard, linear in between (robust aggregation: a
    strong disagreement is more likely an extrapolation failure of the network than information; scoring year: on the days with |ratio - 1| > 0.12 an independent irradiance model fitted
    on past actuals sits closer to the trees (mean error 0.33 vs 0.47 kWh/kW at ratio <= 0.8, 0.15 vs 0.23 at ratio > 1.12); hold-out windows: neutral, +-0.5 %)."""
    if not (tree_u and nn_u):
        return np.mean(nn_u if nn_u else tree_u, axis=0)
    t, n = np.mean(tree_u, axis=0), np.mean(nn_u, axis=0)
    if trim is None:
        return (1 - share) * t + share * n
    ts = pd.DatetimeIndex(ts)
    hh = ts.hour.to_numpy()
    day_key = (ts - pd.Timedelta(hours=1)).normalize().to_numpy()
    mid = (hh >= 11) & (hh <= 15)
    et = pd.Series(t * cap).where(mid).groupby(day_key).sum()
    en = pd.Series(n * cap).where(mid).groupby(day_key).sum()
    r = (en / et.where(et > 0)).fillna(1.0)
    dev = (r - 1.0).abs() if trim.get("side", "both") == "both" else (1.0 - r).clip(lower=0.0)   # side "below": act only on days when the network family is below the trees (Test C, rule T)
    g = np.clip(1.0 - (1.0 - trim["floor"]) * (dev - trim["tol"]) / (trim["hard"] - trim["tol"]), trim["floor"], 1.0)
    w = share * g.reindex(day_key).to_numpy()
    return (1 - w) * t + w * n


def day_shrink(u: np.ndarray, ts: pd.DatetimeIndex, cap: np.ndarray, total_cap: float, beta: float, cf_ref: float | None = CF_REF):
    """returns (u * factor, factor per row, cf_ref).  cf_ref must be a fixed constant (CF_REF); no statistic of the forecast period is used."""
    ts = pd.DatetimeIndex(ts)
    hh = ts.hour.to_numpy()
    day_key = (ts - pd.Timedelta(hours=1)).normalize().to_numpy()
    fc_cap = pd.Series(u * cap).where((hh >= 11) & (hh <= 15))
    cf_day = fc_cap.groupby(day_key).sum() / (total_cap * 5.0)
    cf_day = cf_day.where(cf_day > 0.02)
    if cf_ref is None:
        raise ValueError("day_shrink: cf_ref must be a fixed constant (the whole-period reference was removed on 2026-11-11, notice section 6)")
    factor = np.exp(-beta * (cf_day.fillna(cf_ref) - cf_ref))
    f_row = factor.reindex(day_key).to_numpy()
    return u * f_row, f_row, cf_ref


def offday_correction(u: np.ndarray, ts: pd.DatetimeIndex, cap: np.ndarray, cfg: dict = OFFDAY):
    """returns (u * factor, factor per row).  Needs only the clock, the plant's capacity and the public holiday list (calendar_features.py)."""
    from calendar_features import calendar_columns
    ts = pd.DatetimeIndex(ts)
    off = calendar_columns(ts).cal_off.to_numpy().astype(bool)
    month = (ts - pd.Timedelta(hours=1)).month.to_numpy()
    hh = ts.hour.to_numpy()
    delta = pd.Series(month).map(cfg["delta"]).fillna(0.0).to_numpy()
    prof = pd.Series(hh).map(cfg["profile"]).fillna(0.0).to_numpy()
    factor = 1.0 - delta * prof * (off & (np.asarray(cap) > cfg["big_kw"]))
    return u * factor, factor
