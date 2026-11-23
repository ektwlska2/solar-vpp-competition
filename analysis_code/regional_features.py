"""Regional NWP features: a FIXED grid of 1-degree cells over the peninsula and its sea (lat 33..39 x lon 125..130, 42 cell centres), the same columns for every plant.

Allowed by the contest guide, section 7 ("fixed area summaries, any number of areas, the same columns for every plant; coordinates and sun position may be inputs; weights fixed from
training-period values may be multiplied with the common area values and the result put identically into every plant"): nothing here looks at a plant's position or capacity.

Steps (functions)
  cell_kt(source)                 hourly clear-sky index of every fixed cell, from the stored grid points (20_nwp_cells.py)
  fit_pca(kt_train, k)            fixed loadings of the cell field from TRAINING hours only (never from the evaluation year)
  fit_fleet_weights(kt_all, y)    non-negative weights (training period) turning the cells into one fleet-wide index; y = fleet generation / clear-sky irradiance
  regional_context(...)           DataFrame of hourly features: PCs per source, fleet index per source and combined, day means
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from oibc_lib import *

CELL_LAT = np.arange(33.0, 39.01, 1.0)
CELL_LON = np.arange(125.0, 130.01, 1.0)
CELLS = [(a, b) for a in CELL_LAT for b in CELL_LON]                  # 42 fixed cell centres
LABELS = np.arange(1, 25)
MID = LABELS - 0.5


def assignment(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """[points x cells] averaging matrix: every native grid point belongs to the cell whose centre is within +-0.5 degree (half-open)."""
    M = np.zeros((len(lat), len(CELLS)), dtype="float64")
    for j, (a, b) in enumerate(CELLS):
        M[:, j] = ((lat >= a - 0.5) & (lat < a + 0.5) & (lon >= b - 0.5) & (lon < b + 0.5)).astype(float)
    s = M.sum(0)
    return M / np.maximum(s, 1.0)


def _interp(centres, V, mid=MID):
    """linear interpolation of rows V[n_windows, n_points] (window centres in hours after 00:00 KST) to the 24 hour-label mid points, NaN outside."""
    c = np.asarray(centres, dtype="float64")
    o = np.argsort(c)
    c, V = c[o], np.asarray(V, dtype="float64")[o]
    out = np.full((len(mid), V.shape[1]), np.nan)
    ok = (mid >= c[0]) & (mid <= c[-1])
    i = np.clip(np.searchsorted(c, mid[ok]) - 1, 0, len(c) - 2)
    w = (mid[ok] - c[i]) / (c[i + 1] - c[i])
    out[ok] = V[i] * (1 - w)[:, None] + V[i + 1] * w[:, None]
    return out


def _windows(rows, fld, kind):
    """3-hour mean fields (W/m2) and their centres from one day's rows. kind: 'gfs'/'gefs' = bucket averages with start a, end b; 'ecmwf' = accumulated J/m2 at steps."""
    centres, vals = [], []
    if kind == "ecmwf":
        r = rows.sort_values("step")
        st, arr = list(r.step), list(r[fld])
        for k in range(len(st) - 1):
            if st[k + 1] - st[k] == 3 and arr[k] is not None and arr[k + 1] is not None:
                centres.append((st[k] + st[k + 1]) / 2.0 - 27.0)
                vals.append(np.maximum((np.asarray(arr[k + 1], "float64") - np.asarray(arr[k], "float64")) / 10800.0, 0.0))
        return centres, vals
    r = rows.sort_values("fh").reset_index(drop=True)
    byfh = {int(x.fh): x for x in r.itertuples(index=False)}
    for fh, x in byfh.items():
        v = getattr(x, fld)
        if v is None or np.isnan(x.a):
            continue
        if x.b - x.a == 3:
            centres.append((x.a + x.b) / 2.0 - 15.0); vals.append(np.asarray(v, "float64"))
        elif x.b - x.a == 6 and (fh - 3) in byfh and getattr(byfh[fh - 3], fld) is not None and byfh[fh - 3].a == x.a:
            centres.append((x.b - 1.5) - 15.0); vals.append(np.maximum(2.0 * np.asarray(v, "float64") - np.asarray(getattr(byfh[fh - 3], fld), "float64"), 0.0))
    return centres, vals


def cell_kt(path: str, source: str) -> pd.DataFrame:
    """hourly clear-sky index of the 42 fixed cells. Index = hour labels (KST, right-closed), columns c_<lat>_<lon>."""
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"])
    geo = json.load(open(path.replace(".parquet", "_geo.json")))
    M = assignment(np.array(geo["lat"]), np.array(geo["lon"]))
    fld = "ssrd" if source == "ecmwf" else "dsw"
    days, blocks = [], []
    for day, g in raw.groupby("date"):
        c, v = _windows(g, fld, source)
        if len(c) < 2:
            continue
        dsw = _interp(c, np.vstack(v)) @ M                                        # [24, 42] cell values (W/m2)
        days.append(day); blocks.append(dsw)
    D = np.stack(blocks)                                                           # [days, 24, 42]
    ts = pd.DatetimeIndex([pd.Timestamp(d, tz=TZ) + pd.Timedelta(hours=int(h)) for d in days for h in LABELS])
    kt = np.full((len(ts), len(CELLS)), np.nan, dtype="float32")
    flat = D.reshape(-1, len(CELLS))
    for j, (a, b) in enumerate(CELLS):
        _, _, cz = solar(ts, a, b)
        cs = haurwitz_ghi(cz)
        ok = cs > 50
        kt[ok, j] = np.clip(flat[ok, j] / cs[ok], 0, 1.3)
    return pd.DataFrame(kt, index=ts, columns=[f"c_{int(a)}_{int(b)}" for a, b in CELLS])


def cell_kt_ldaps(path: str, var: str = "tdsw") -> pd.DataFrame:
    """hourly clear-sky index of the 120 fixed 0.5-degree cells of the KMA UM-LDAPS (1.5 km) from the stored 2-hourly samples (22_kma_ldaps.py): valid hours 08, 10, ..., 18 KST of the
    target day (1-hour mean flux ending at the valid hour = hour label), linear interpolation of the index in between, held constant to the edges of the day."""
    raw = pd.read_parquet(path)
    raw = raw[(raw["var"] == var) & raw.cells.notna()]
    geo = json.load(open(path.replace(".parquet", "_geo.json")))
    la, lo = np.array(geo["lat"]), np.array(geo["lon"])
    labels = np.arange(1, 25)
    blocks, days = [], []
    for day, g in raw.groupby("date"):
        g = g.sort_values("lead")
        if len(g) < 3:
            continue
        hv = (g["lead"].to_numpy() - 15).astype(float)                              # valid hour label of the target day
        V = np.vstack([np.asarray(c, dtype="float64") for c in g.cells])            # [n_leads, 120] W/m2
        ts_v = pd.DatetimeIndex([pd.Timestamp(day, tz=TZ) + pd.Timedelta(hours=int(h)) for h in hv])
        kt = np.full_like(V, np.nan)
        for j in range(V.shape[1]):
            _, _, cz = solar(ts_v, la[j], lo[j])
            cs = haurwitz_ghi(cz)
            ok = cs > 50
            kt[ok, j] = np.clip(V[ok, j] / cs[ok], 0, 1.3)
        full = np.full((24, V.shape[1]), np.nan)
        for j in range(V.shape[1]):
            m = np.isfinite(kt[:, j])
            if m.sum() >= 2:
                full[:, j] = np.interp(labels, hv[m], kt[m, j])                     # np.interp holds the edge values constant outside the sampled hours
        blocks.append(full); days.append(day)
    D = np.stack(blocks)
    ts = pd.DatetimeIndex([pd.Timestamp(d, tz=TZ) + pd.Timedelta(hours=int(h)) for d in days for h in labels])
    out = D.reshape(-1, D.shape[2]).astype("float32")
    # hours without sun get NaN like the other sources
    for j in range(out.shape[1]):
        _, _, cz = solar(ts, la[j], lo[j])
        out[haurwitz_ghi(cz) <= 50, j] = np.nan
    return pd.DataFrame(out, index=ts, columns=[f"c_{a:.2f}_{b:.2f}" for a, b in zip(la, lo)])


def cell_field(path: str, source: str, fld: str = "tcc") -> pd.DataFrame:
    """hourly value of a stored field (default total cloud cover, %) of the 42 fixed cells, same per-day construction as cell_kt (no clear-sky normalisation)."""
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"])
    geo = json.load(open(path.replace(".parquet", "_geo.json")))
    M = assignment(np.array(geo["lat"]), np.array(geo["lon"]))
    days, blocks = [], []
    for day, g in raw.groupby("date"):
        c, v = _windows(g, fld, source)
        if len(c) < 2:
            continue
        days.append(day); blocks.append(_interp(c, np.vstack(v)) @ M)
    D = np.stack(blocks)
    ts = pd.DatetimeIndex([pd.Timestamp(d, tz=TZ) + pd.Timedelta(hours=int(h)) for d in days for h in LABELS])
    out = D.reshape(-1, len(CELLS)).astype("float32")
    return pd.DataFrame(out, index=ts, columns=[f"c_{int(a)}_{int(b)}" for a, b in CELLS])


def fit_pca(kt: pd.DataFrame, k: int = 4):
    """mean and loadings of the first k principal components of the daytime cell field (training hours only)."""
    X = kt.dropna()
    mu = X.mean().to_numpy()
    U, S, Vt = np.linalg.svd(X.to_numpy() - mu, full_matrices=False)
    return mu, Vt[:k], (S[:k] ** 2) / (S ** 2).sum()


def apply_pca(kt: pd.DataFrame, mu, V, prefix: str) -> pd.DataFrame:
    Z = (np.nan_to_num(kt.to_numpy(), nan=0.0) - mu) @ V.T
    Z[kt.isna().all(axis=1).to_numpy()] = np.nan
    return pd.DataFrame(Z.astype("float32"), index=kt.index, columns=[f"{prefix}_pc{i+1}" for i in range(V.shape[0])])


def fit_fleet_weights(kt: pd.DataFrame, target: pd.Series) -> np.ndarray:
    """non-negative weights w (training hours only) with target ~ kt @ w, target = fleet generation per kW divided by the clear-sky irradiance (kW/m2)."""
    d = pd.concat([kt, target.rename("y")], axis=1, join="inner").dropna()
    w, _ = nnls(d[kt.columns].to_numpy("float64"), d.y.to_numpy("float64"))
    return w


# ------------------------------------------------------------------------------------------------------------------ the feature block used by the models
def fleet_target(panel: pd.DataFrame, cut) -> pd.Series:
    """hourly fleet generation per kW divided by the clear-sky irradiance in kW/m2, from TRAINING hours (<= cut) of the sites that report in that hour (>= 20 sites)."""
    p = panel[(panel.ts <= cut) & (panel.elev > 0) & panel.energy.notna()]
    g = p.groupby("ts").agg(e=("energy", "sum"), c=("cap", "sum"), n=("energy", "size"))
    g = g[g.n >= 20]
    _, _, cz = solar(pd.DatetimeIndex(g.index), 36.0, 127.8)
    cs = haurwitz_ghi(cz) / 1000.0
    y = pd.Series(np.where(cs > 0.05, (g.e / g.c).to_numpy() / np.maximum(cs, 0.05), np.nan), index=g.index)
    return y.clip(0, 1.6)


def _ctx(s: pd.Series, name: str) -> pd.DataFrame:
    """hourly value, neighbouring hours and day statistics INSIDE the same target day (nothing from the next day)."""
    d = pd.DataFrame({name: s})
    day = (s.index - pd.Timedelta(hours=1)).tz_localize(None).normalize()
    hr = s.index.hour
    sel = s.where((hr >= 9) & (hr <= 16))
    d[f"{name}_dmean"] = sel.groupby(day).transform("mean")
    d[f"{name}_dmax"] = sel.groupby(day).transform("max")
    d[f"{name}_dmin"] = sel.groupby(day).transform("min")
    for k in (1, 2):
        d[f"{name}_lag{k}"] = s.groupby(day).shift(k)
        d[f"{name}_lead{k}"] = s.groupby(day).shift(-k)
    return d


def regional_block(cache: str, cut, panel: pd.DataFrame, sources=("gfs", "gefs"), n_pc: int = 4, cells_raw: bool = False, cloud_sources=(), cloud_fleet: bool = False) -> pd.DataFrame:
    """Hourly regional features for the competition's hour labels. PCA loadings and non-negative fleet weights are fitted on TRAINING hours (<= cut) only and then applied identically to every
    hour and every plant (guide section 7)."""
    y = fleet_target(panel, cut)
    kts = {s: (cell_kt_ldaps(f"{cache}/nwp_cells_{s}.parquet") if s == "ldaps" else cell_kt(f"{cache}/nwp_cells_{s}.parquet", s))
           for s in sources if os.path.exists(f"{cache}/nwp_cells_{s}.parquet")}
    out = []
    for s, kt in kts.items():
        tr = kt[kt.index <= cut]
        mu, V, ev = fit_pca(tr, n_pc)
        pcs = apply_pca(kt, mu, V, f"rg_{s}")
        w = fit_fleet_weights(tr, y)
        idx = pd.Series(kt.to_numpy("float64") @ w, index=kt.index).where(kt.notna().any(axis=1))
        out.append(pcs)
        out.append(_ctx(idx.astype("float32"), f"rg_{s}_fleet"))
        if cells_raw:
            day = (kt.index - pd.Timedelta(hours=1)).tz_localize(None).normalize()
            arr = kt.to_numpy().copy()
            arr[~((kt.index.hour >= 9) & (kt.index.hour <= 16))] = np.nan                  # daytime core hours only
            dm = pd.DataFrame(arr, index=kt.index, columns=[f"rg_{s}_dm_{c}" for c in kt.columns]).groupby(day).transform("mean")
            out.append(dm)
    for s_ in cloud_sources:                                                                # principal components of the cloud-cover cell field (training hours only)
        if os.path.exists(f"{cache}/nwp_cells_{s_}.parquet"):
            cl = cell_field(f"{cache}/nwp_cells_{s_}.parquet", s_, "tcc")
            core = (cl.index.hour >= 9) & (cl.index.hour <= 16)                              # daytime core hours of the training period define the components
            mu, V, ev = fit_pca(cl[(cl.index <= cut) & core], n_pc)
            out.append(apply_pca(cl, mu, V, f"rg_{s_}_cc"))
            if cloud_fleet:                                                                  # one fleet-wide clear-fraction index (1 - cloud cover) with non-negative weights fitted on training hours
                clear = (1.0 - cl / 100.0).clip(0, 1)
                wc = fit_fleet_weights(clear[(clear.index <= cut) & core], y)
                idx_c = pd.Series(clear.to_numpy("float64") @ wc, index=clear.index)
                out.append(_ctx(idx_c.astype("float32"), f"rg_{s_}_cc_fleet"))
    if len([s for s in kts if s != "ecmwf"]) >= 2:
        base = [s for s in kts if s != "ecmwf"]
        K_ = pd.concat([kts[s].add_prefix(f"{s}_") for s in base], axis=1)
        tr = K_[K_.index <= cut].dropna(how="all")
        w = fit_fleet_weights(tr, y)
        idx = pd.Series(np.nan_to_num(K_.to_numpy("float64")) @ w, index=K_.index).where(K_.notna().any(axis=1))
        out.append(_ctx(idx.astype("float32"), "rg_all_fleet"))
    R = pd.concat(out, axis=1)
    return R.astype("float32")
