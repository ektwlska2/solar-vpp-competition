"""Shared helpers for the OIBC 2026 analysis.

Paths
    OIBC_DATA   dataset folder (default: ../handover 2/01_participant_dataset next to this folder)
    OIBC_CACHE  where intermediate parquet/csv files go (default: ./_cache, safe to delete)

Everything here uses only the competition files. The solar-geometry function is the NOAA
approximation copied from the organisers' baseline.py.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("OIBC_DATA", os.path.normpath(os.path.join(HERE, "..", "handover 2", "01_participant_dataset")))
CACHE = os.environ.get("OIBC_CACHE", os.path.join(HERE, "_cache"))
os.makedirs(CACHE, exist_ok=True)
TZ = "Asia/Seoul"
TOTAL_KW = 42655.4
LO_FRAC = 20000.0 / TOTAL_KW          # each group must hold >= 46.9 % of capacity
K = lambda s: pd.Timestamp(s, tz=TZ)

# forecast columns used as model inputs (per issue round 10 and 17)
A_COLS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_ground_press", "wa_wind_speed", "wa_wind_dir", "wa_rain", "wa_snow"]
B_COLS = ["wb_cloud", "wb_temp", "wb_feel_temp", "wb_dew_point", "wb_wind_speed", "wb_wind_gust_spd", "wb_rel_hum",
          "wb_vis", "wb_ceiling", "wb_uv_idx", "wb_precip_prob", "wb_total_liq", "wb_snow"]
GEO = ["elev", "azim", "cosz", "doy_sin", "doy_cos", "hour", "cap", "lat", "lon"]


# --------------------------------------------------------------------------- geometry / prices
def solar(ts: pd.DatetimeIndex, lat, lon):
    """Solar elevation, azimuth, cos(zenith); ts is right-closed so the interval midpoint (-30 min) is used."""
    doy = ts.day_of_year.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = (hour - 0.5) * 60 + eqtime + 4 * (lon - 135.0)
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    cosz = np.clip(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha), -1, 1)
    elev = 90.0 - np.degrees(np.arccos(cosz))
    az = np.degrees(np.arctan2(-np.sin(ha), np.tan(decl) * np.cos(la) - np.sin(la) * np.cos(ha))) % 360
    return elev.astype("float32"), az.astype("float32"), cosz.astype("float32")


def haurwitz_ghi(cosz):
    """Clear-sky global horizontal irradiance (W/m2), Haurwitz (1945)."""
    c = np.clip(cosz, 0.0, 1.0).astype("float64")
    out = np.where(c > 0.01, 1098.0 * c * np.exp(-0.059 / np.maximum(c, 0.01)), 0.0)
    return out.astype("float32")


def load_smp_train() -> pd.Series:
    s = pd.read_csv(f"{DATA}/train/smp.csv")
    s["ts"] = pd.to_datetime(s.ts)
    return s.set_index("ts").smp


def hour_weights(ts: pd.DatetimeIndex, smp: pd.Series) -> np.ndarray:
    """w_t = SMP_t * k_t with k = 3 for hour labels 11..15. Missing SMP -> series mean."""
    base = smp.reindex(ts).fillna(smp.mean()).to_numpy(dtype="float64")
    return base * np.where(np.isin(ts.hour, [11, 12, 13, 14, 15]), 3.0, 1.0)


def load_sites() -> pd.DataFrame:
    s = pd.read_csv(f"{DATA}/sites.csv")
    s["site_code"] = s.site_id.str[2:].astype(int)
    return s


# --------------------------------------------------------------------------- metric helpers
def error_matrix(pred: pd.DataFrame, panel: pd.DataFrame, start, end, min_cov=0.9, U=None):
    """Site x hour error matrix (kWh, forecast - actual) for hours in (start, end], daytime rows only.

    pred   DataFrame[site_code, ts, pred_u]  utilisation forecasts
    panel  DataFrame[site_code, ts, energy, cap, elev]
    Hours where a site has no actual are set to 0 (excluded from forecast and actual alike).
    """
    a = panel[(panel.ts > start) & (panel.ts <= end)]
    n_hours = len(pd.date_range(start, end, freq="h", inclusive="right"))
    if U is None:
        cov = a.groupby("site_code").size() / n_hours
        U = cov[cov >= min_cov].index.values
    U = np.asarray(U)
    a = a[a.site_code.isin(U) & (a.elev > 0)]
    cap = panel.drop_duplicates("site_code").set_index("site_code").cap.reindex(U).to_numpy(dtype="float64")
    pr = pred[pred.site_code.isin(U)]
    P = pr.pivot(index="ts", columns="site_code", values="pred_u").reindex(columns=U)
    A = a.pivot(index="ts", columns="site_code", values="energy").reindex(index=P.index, columns=U)
    Pk = P.to_numpy() * cap[None, :]
    Ak = A.to_numpy()
    E = np.where(np.isnan(Ak), 0.0, np.nan_to_num(Pk) - Ak)
    smp = load_smp_train()
    w = hour_weights(P.index, smp)
    return dict(E=E, w=w, sites=U, cap=cap, ts=P.index, Pk=Pk, Ak=Ak)


def W_of(E, w, mask):
    a = E[:, mask].sum(1)
    b = E[:, ~mask].sum(1)
    return 0.10 * float((w * (np.abs(a) + np.abs(b))).sum())


def W_floor(E, w):
    return 0.10 * float((w * np.abs(E.sum(1))).sum())


def W_nopool(E, w):
    return 0.10 * float((w[:, None] * np.abs(E)).sum())


def feasible(cap, mask, lo=LO_FRAC):
    s = cap[mask].sum() / cap.sum()
    return lo <= s <= 1 - lo


def random_partition(cap, rng, lo=LO_FRAC):
    while True:
        m = rng.random(len(cap)) < 0.5
        if feasible(cap, m, lo):
            return m


def twin_partition(site_keys, cap):
    """Greedy station-stratified split: within each key (weather station) hand sites, largest first, to the lighter group."""
    m = np.zeros(len(cap), bool)
    c1 = c2 = 0.0
    df = pd.DataFrame({"i": range(len(cap)), "key": site_keys, "cap": cap}).sort_values(["key", "cap"], ascending=[True, False])
    for _, g in df.groupby("key", sort=True):
        for _, r in g.iterrows():
            if c1 <= c2:
                m[int(r.i)] = True
                c1 += r.cap
            else:
                c2 += r.cap
    return m


def local_search(E, w, cap, init, rng, lo=LO_FRAC, iters=80, swap_tries=4000):
    """Minimise sum_t w (|a|+|b|) over balanced partitions: greedy best flip, then random pair swaps."""
    mask = init.copy()
    tot = E.sum(1)
    a = E[:, mask].sum(1)
    cur = float((w * (np.abs(a) + np.abs(tot - a))).sum())
    N = len(cap)
    for _ in range(iters):
        sign = np.where(mask, -1.0, 1.0)
        An = a[:, None] + E * sign[None, :]
        cost = (w[:, None] * (np.abs(An) + np.abs(tot[:, None] - An))).sum(0)
        newcap = np.where(mask, cap[mask].sum() - cap, cap[mask].sum() + cap) / cap.sum()
        cost = np.where((newcap >= lo) & (newcap <= 1 - lo), cost, np.inf)
        j = int(np.argmin(cost))
        if cost[j] < cur - 1e-9:
            mask[j] = ~mask[j]
            a = a + sign[j] * E[:, j]
            cur = float(cost[j])
        else:
            break
    tries = 0
    improved = True
    while improved and tries < swap_tries:
        improved = False
        for ii, kk in zip(rng.integers(N, size=64), rng.integers(N, size=64)):
            if mask[ii] == mask[kk]:
                continue
            m2 = mask.copy()
            m2[ii] = ~m2[ii]
            m2[kk] = ~m2[kk]
            if not feasible(cap, m2, lo):
                continue
            a2 = a + (-E[:, ii] if mask[ii] else E[:, ii]) + (-E[:, kk] if mask[kk] else E[:, kk])
            c2 = float((w * (np.abs(a2) + np.abs(tot - a2))).sum())
            if c2 < cur - 1e-9:
                mask, a, cur = m2, a2, c2
                improved = True
        tries += 64
    return mask, 0.10 * cur


# --------------------------------------------------------------------------- sister sites
def sister_groups(sites):
    """Sister sites: same capacity AND same WA and WB stations as at least one other site (a farm registered as several units).
    Returns a Series (index = sites.index) holding a group key, NaN for sites without a sister."""
    key = sites.capacity_kw.round(2).astype(str) + "|" + sites.wa_station.astype(str) + "|" + sites.wb_station.astype(str)
    n = key.map(key.value_counts())
    return key.where(n > 1)


def keyed_twin(sites_u, cap, station="wa_station"):
    """Sisters are alternated across the two groups first, the remaining sites are split greedily inside each weather station."""
    gid = sister_groups(sites_u)
    keys = np.where(gid.notna(), "A_" + gid.astype(str), "B_" + sites_u[station].astype(str))
    return twin_partition(keys, cap)
