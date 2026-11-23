"""Build the modelling panel: one row per (site, hour) with forecast, context, aggregate and physics features.

Usage:  python 02_build_panel.py            (training panel from train/generation.parquet)
Output: <CACHE>/panel.parquet

Feature families
  raw      forecasts of provider A (8 vars) and B (13 vars) for issue rounds 10 and 17 at the site's own stations
  ctx      round-17 forecast of the SAME target day at neighbouring hours (t-1, t+1, 3h/5h mean) and day-level
           mean/max over hour labels 8-17.  All values of a target day are published before the 17:00 cut-off.
  agg      fleet-wide mean of the forecast over all stations (a broad-area summary), provider blends/disagreement
  phy      Haurwitz clear-sky irradiance and cloud-attenuated versions (Kasten-Czeplak factor)
"""
from __future__ import annotations
import sys, time
import numpy as np
import pandas as pd
from oibc_lib import *

CTX_A = ["wa_cloud", "wa_humidity"]
CTX_B = ["wb_cloud", "wb_uv_idx", "wb_rel_hum"]
FLEET_VARS = ["wa_cloud", "wa_humidity", "wb_cloud", "wb_uv_idx", "wb_rel_hum"]


def widen_rounds(f, st, cols):
    out = None
    for r in (10, 17):
        x = f[f["round"] == r].drop(columns="round").rename(columns={c: f"{c}_r{r}" for c in cols})
        out = x if out is None else out.merge(x, on=[st, "ts"], how="outer")
    return out


def apply_shift(f, st, cols, shift):
    """Re-time a forecast table: the value used for hour label t becomes f(t - shift).

    shift=1 treats a forecast labelled t as describing the hour ENDING at t+1 (hour-start labelling);
    shift=0.5 averages f(t) and f(t-1). Done per station and issue round on the hourly grid.
    """
    if shift == 0:
        return f
    grid = pd.date_range(f.ts.min(), f.ts.max(), freq="h").as_unit(f.ts.dt.unit)
    outs = []
    for r in (10, 17):
        x = f[f["round"] == r].set_index([st, "ts"]).sort_index()
        stations = x.index.get_level_values(0).unique()
        x = x[cols].reindex(pd.MultiIndex.from_product([stations, grid], names=[st, "ts"]))
        prev = x.groupby(level=0, sort=False).shift(1)
        y = prev if shift == 1 else 0.5 * x + 0.5 * prev
        y = y.dropna(how="all").reset_index()
        y.insert(1, "round", r)
        outs.append(y[[st, "round", "ts"] + cols])
    return pd.concat(outs, ignore_index=True)


def station_context(f, st, vars_, rnd=17):
    """Neighbour-hour and day-level features of the same issue round, computed on each station's hourly series."""
    x = f[f["round"] == rnd][[st, "ts"] + vars_]
    grid = pd.date_range(f.ts.min(), f.ts.max(), freq="h").as_unit(f.ts.dt.unit)
    stations = x[st].unique()
    x = x.set_index([st, "ts"]).sort_index().reindex(pd.MultiIndex.from_product([stations, grid], names=[st, "ts"]))
    g = x.groupby(level=0, sort=False)
    parts = [
        g.shift(1).add_suffix(f"_r{rnd}_lag1"),
        g.shift(-1).add_suffix(f"_r{rnd}_lead1"),
        g.transform(lambda s: s.rolling(3, center=True, min_periods=1).mean()).add_suffix(f"_r{rnd}_m3"),
        g.transform(lambda s: s.rolling(5, center=True, min_periods=2).mean()).add_suffix(f"_r{rnd}_m5"),
    ]
    ts = x.index.get_level_values("ts")
    dayid = (ts - pd.Timedelta(hours=1)).normalize()           # right-closed labels: 00:00 belongs to the previous day
    dmask = pd.Series(np.where((ts.hour >= 8) & (ts.hour <= 17), 1.0, np.nan), index=x.index)
    xm = x.mul(dmask, axis=0)
    keys = [x.index.get_level_values(0), dayid]
    parts.append(xm.groupby(keys).transform("mean").add_suffix(f"_r{rnd}_dmean"))
    parts.append(xm.groupby(keys).transform("max").add_suffix(f"_r{rnd}_dmax"))
    out = pd.concat(parts, axis=1).astype("float32")
    return out.reset_index().rename(columns={st: st})


EXTRA_A = ["wa_temp", "wa_rain", "wa_ground_press", "wa_wind_speed"]
EXTRA_B = ["wb_temp", "wb_ceiling", "wb_precip_prob", "wb_vis", "wb_total_liq", "wb_wind_speed"]


def station_context_ext(f, st, vars_, rnd):
    """Extra day-shape descriptors for the core variables: lag/lead 2, day min/std, morning (8-12) and afternoon (13-17) means."""
    x = f[f["round"] == rnd][[st, "ts"] + vars_]
    grid = pd.date_range(f.ts.min(), f.ts.max(), freq="h").as_unit(f.ts.dt.unit)
    stations = x[st].unique()
    x = x.set_index([st, "ts"]).sort_index().reindex(pd.MultiIndex.from_product([stations, grid], names=[st, "ts"]))
    g = x.groupby(level=0, sort=False)
    ts = x.index.get_level_values("ts")
    dayid = (ts - pd.Timedelta(hours=1)).normalize()
    keys = [x.index.get_level_values(0), dayid]

    def masked(lo, hi):
        return x.mul(pd.Series(np.where((ts.hour >= lo) & (ts.hour <= hi), 1.0, np.nan), index=x.index), axis=0)

    xm = masked(8, 17)
    parts = [g.shift(2).add_suffix(f"_r{rnd}_lag2"), g.shift(-2).add_suffix(f"_r{rnd}_lead2"),
             xm.groupby(keys).transform("min").add_suffix(f"_r{rnd}_dmin"), xm.groupby(keys).transform("std").add_suffix(f"_r{rnd}_dstd"),
             masked(8, 12).groupby(keys).transform("mean").add_suffix(f"_r{rnd}_dam"), masked(13, 17).groupby(keys).transform("mean").add_suffix(f"_r{rnd}_dpm")]
    return pd.concat(parts, axis=1).astype("float32").reset_index()


def station_context_wide(f, st, vars_, rnd=17):
    """Wider neighbour offsets (+-3, +-4 h) and the day trajectory (values at hour labels 9..17 of the same target day)."""
    x = f[f["round"] == rnd][[st, "ts"] + vars_]
    grid = pd.date_range(f.ts.min(), f.ts.max(), freq="h").as_unit(f.ts.dt.unit)
    stations = x[st].unique()
    x = x.set_index([st, "ts"]).sort_index().reindex(pd.MultiIndex.from_product([stations, grid], names=[st, "ts"]))
    g = x.groupby(level=0, sort=False)
    lags = pd.concat([g.shift(k).add_suffix(f"_r{rnd}_lag{k}") for k in (3, 4)] + [g.shift(-k).add_suffix(f"_r{rnd}_lead{k}") for k in (3, 4)], axis=1).astype("float32")
    lags = lags.reset_index()
    ts = x.index.get_level_values("ts")
    tmp = x.copy()
    tmp["day"] = (ts - pd.Timedelta(hours=1)).normalize()
    tmp["hh"] = ts.hour
    tmp = tmp.reset_index()
    traj = tmp[tmp.hh.between(9, 17)].pivot_table(index=[st, "day"], columns="hh", values=vars_, aggfunc="first")
    traj.columns = [f"{v}_r{rnd}_h{h}" for v, h in traj.columns]
    return lags, traj.astype("float32").reset_index()



# ----------------------------------------------------------------------------------------------- previous-day regime and spatial-neighbour features
HIST_SPEC = {  # provider -> (station column, rain, snow, temp, cloud, humidity) forecast variables
    "wa": ("wa_station", "wa_rain", "wa_snow", "wa_temp", "wa_cloud", "wa_humidity"),
    "wb": ("wb_station", "wb_total_liq", "wb_snow", "wb_temp", "wb_cloud", "wb_rel_hum"),
}


def station_history(f, prov, rnd=17):
    """Regime features from the forecasts of EARLIER days (published before the cut-off): daily rain / snow / temperature / cloud of D-1..D-3,
    3- and 7-day accumulations, days since rain and since snow. Keyed by (station, target day D = label date of ts - 1 h)."""
    st, rain, snow, temp, cloud, hum = HIST_SPEC[prov]
    x = f[f["round"] == rnd][[st, "ts", rain, snow, temp, cloud, hum]].copy()
    x["day"] = (x.ts - pd.Timedelta(hours=1)).dt.normalize()
    x["cl_day"] = x[cloud].where((x.ts.dt.hour >= 8) & (x.ts.dt.hour <= 17))
    g = x.groupby([st, "day"]).agg(rain=(rain, "sum"), snow=(snow, "sum"), tmin=(temp, "min"), tmax=(temp, "max"), cloud=("cl_day", "mean"), hum=(hum, "mean"))
    out = []
    for stn, d in g.groupby(level=0):
        d = d.droplevel(0)
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D", tz=d.index.tz))
        r = pd.DataFrame(index=d.index)
        for k in (1, 2, 3):
            for c in ("rain", "snow", "tmin", "cloud"):
                r[f"hist_{prov}_{c}_d{k}"] = d[c].shift(k)
        r[f"hist_{prov}_rain3"] = d.rain.shift(1).rolling(3, min_periods=1).sum()
        r[f"hist_{prov}_snow3"] = d.snow.shift(1).rolling(3, min_periods=1).sum()
        r[f"hist_{prov}_snow7"] = d.snow.shift(1).rolling(7, min_periods=1).sum()
        r[f"hist_{prov}_cloud7"] = d.cloud.shift(1).rolling(7, min_periods=2).mean()
        for name, cond in (("rain", d.rain > 2.0), ("snow", d.snow > 0.2)):
            c = cond.shift(1).fillna(False).to_numpy()
            idx = np.where(c, np.arange(len(c)), -10**6)
            last = np.maximum.accumulate(idx)
            since = np.arange(len(c)) - last
            r[f"hist_{prov}_since_{name}"] = np.where(last < 0, 30, np.minimum(since, 30)).astype("float32")
        r.insert(0, st, stn)
        out.append(r.reset_index(names="day"))
    return pd.concat(out, ignore_index=True)


def compute_neighbors(f, prov, k=3, rnd=17):
    """The k stations whose forecast cloud series correlate most with each station (data-driven, no coordinates)."""
    st = HIST_SPEC[prov][0]
    cloud = HIST_SPEC[prov][4]
    w = f[(f["round"] == rnd) & (f.ts.dt.hour >= 8) & (f.ts.dt.hour <= 17)].pivot_table(index="ts", columns=st, values=cloud)
    C = w.corr(min_periods=500).to_numpy().copy()
    np.fill_diagonal(C, -2)
    cols = list(w.columns)
    return {cols[i]: [cols[j] for j in np.argsort(-np.nan_to_num(C[i], nan=-2))[:k]] for i in range(len(cols))}


def station_nbr(f, prov, nb, rnd=17):
    """Mean forecast of the neighbour stations at the same hour (cloud, humidity, and a second variable), their day mean, and the own-minus-neighbour anomaly."""
    st = HIST_SPEC[prov][0]
    vars_ = {"wa": ["wa_cloud", "wa_humidity", "wa_wind_speed"], "wb": ["wb_cloud", "wb_uv_idx", "wb_rel_hum"]}[prov]
    x = f[f["round"] == rnd][[st, "ts"] + vars_]
    outs = []
    for v in vars_:
        w = x.pivot_table(index="ts", columns=st, values=v)
        nbm = pd.DataFrame({s: w[[c for c in nb[s] if c in w.columns]].mean(axis=1) for s in w.columns if s in nb})
        day = (nbm.index - pd.Timedelta(hours=1)).normalize()
        dmask = pd.Series(np.where((nbm.index.hour >= 8) & (nbm.index.hour <= 17), 1.0, np.nan), index=nbm.index)
        dmean = nbm.mul(dmask, axis=0).groupby(day).transform("mean")
        o = pd.concat({f"nb_{v}": nbm.stack(), f"nb_{v}_day": dmean.stack(), f"nb_{v}_anom": (w[nbm.columns] - nbm).stack()}, axis=1)
        outs.append(o)
    r = pd.concat(outs, axis=1).astype("float32")
    r.index.names = ["ts", st]
    return r.reset_index()


def build_feature_frame(base: pd.DataFrame, fa: pd.DataFrame, fb: pd.DataFrame, sites: pd.DataFrame,
                        shift_a: float = 0, shift_b: float = 0, ctx2: bool = False, ctx3: bool = False,
                        ctx4: bool = False, neighbors: dict | None = None, fa_hist: pd.DataFrame | None = None, fb_hist: pd.DataFrame | None = None,
                        gfs: pd.DataFrame | None = None) -> pd.DataFrame:
    """base: DataFrame[site_id, ts, (energy)]. fa/fb: forecast tables (train or test) with columns round, ts, station, vars."""
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
    fa = apply_shift(fa, "wa_station", A_COLS, shift_a)
    fb = apply_shift(fb, "wb_station", B_COLS, shift_b)
    keep = ["site_id", "capacity_kw", "lat", "lon", "wa_station", "wb_station", "current_group"]
    d = base.merge(sites[keep], on="site_id", how="left")
    fa_w = widen_rounds(fa[["wa_station", "round", "ts"] + A_COLS], "wa_station", A_COLS)
    fb_w = widen_rounds(fb[["wb_station", "round", "ts"] + B_COLS], "wb_station", B_COLS)
    d = d.merge(fa_w, on=["wa_station", "ts"], how="left").merge(fb_w, on=["wb_station", "ts"], how="left")
    log(f"raw merged {d.shape}")
    ctx_a = station_context(fa, "wa_station", CTX_A)
    ctx_b = station_context(fb, "wb_station", CTX_B)
    d = d.merge(ctx_a, on=["wa_station", "ts"], how="left").merge(ctx_b, on=["wb_station", "ts"], how="left")
    if ctx2:
        # round-10 context for the core variables, extra variables, and day-shape descriptors (all published before the cut-off)
        ex = [station_context(fa, "wa_station", CTX_A, 10), station_context(fb, "wb_station", CTX_B, 10),
              station_context(fa, "wa_station", EXTRA_A, 17), station_context(fb, "wb_station", EXTRA_B, 17),
              station_context_ext(fa, "wa_station", CTX_A, 17), station_context_ext(fb, "wb_station", CTX_B, 17)]
        for t in ex:
            st = "wa_station" if "wa_station" in t.columns else "wb_station"
            d = d.merge(t, on=[st, "ts"], how="left")
        log(f"context2 merged {d.shape}")
    if ctx3:
        d["day"] = (pd.DatetimeIndex(d.ts) - pd.Timedelta(hours=1)).normalize()
        for f_, st, vv in ((fa, "wa_station", CTX_A), (fb, "wb_station", CTX_B)):
            lags, traj = station_context_wide(f_, st, vv, 17)
            d = d.merge(lags, on=[st, "ts"], how="left").merge(traj, on=[st, "day"], how="left")
        d = d.drop(columns=["day"])
        log(f"context3 merged {d.shape}")
    if ctx4:
        # previous-day regime features (need the forecasts of earlier days: fa_hist / fb_hist = longer timeline) and neighbour-station features
        fa_h = fa_hist if fa_hist is not None else fa
        fb_h = fb_hist if fb_hist is not None else fb
        d["day"] = (pd.DatetimeIndex(d.ts) - pd.Timedelta(hours=1)).normalize()
        for prov, f_h in (("wa", fa_h), ("wb", fb_h)):
            st = HIST_SPEC[prov][0]
            h = station_history(f_h, prov)
            h["day"] = h["day"].dt.tz_convert(TZ) if h["day"].dt.tz is not None else h["day"]
            d = d.merge(h, on=[st, "day"], how="left")
        d = d.drop(columns=["day"])
        nb = neighbors or {"wa": compute_neighbors(fa_h, "wa"), "wb": compute_neighbors(fb_h, "wb")}
        for prov, f_t in (("wa", fa), ("wb", fb)):
            st = HIST_SPEC[prov][0]
            d = d.merge(station_nbr(f_t, prov, nb[prov]), on=[st, "ts"], how="left")
        log(f"context4 merged {d.shape}")
    log("context merged")
    fm = pd.concat([fa[fa["round"] == 17].groupby("ts")[["wa_cloud", "wa_humidity"]].mean(),
                    fb[fb["round"] == 17].groupby("ts")[["wb_cloud", "wb_uv_idx", "wb_rel_hum"]].mean()], axis=1)
    fm = fm.add_prefix("fleet_").add_suffix("_r17").astype("float32").reset_index()
    d = d.merge(fm, on="ts", how="left")
    ts = pd.DatetimeIndex(d.ts)
    d["elev"], d["azim"], d["cosz"] = solar(ts, d.lat.to_numpy(), d.lon.to_numpy())
    doy = ts.day_of_year.to_numpy()
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    d["hour"] = ts.hour.astype("int8")
    d["cap"] = d.capacity_kw.astype("float32")
    # blends / disagreement
    for r in (10, 17):
        d[f"cloud_mean_r{r}"] = ((d[f"wa_cloud_r{r}"] + d[f"wb_cloud_r{r}"]) / 2).astype("float32")
        d[f"cloud_diff_r{r}"] = (d[f"wa_cloud_r{r}"] - d[f"wb_cloud_r{r}"]).astype("float32")
    d["cloudA_r10_minus_r17"] = (d.wa_cloud_r10 - d.wa_cloud_r17).astype("float32")
    d["cloudB_r10_minus_r17"] = (d.wb_cloud_r10 - d.wb_cloud_r17).astype("float32")
    # physics: clear-sky GHI and cloud-attenuated GHI (Kasten & Czeplak 1980 factor)
    d["ghi_cs"] = haurwitz_ghi(d.cosz.to_numpy())
    for r in (10, 17):
        for p, c in (("A", f"wa_cloud_r{r}"), ("B", f"wb_cloud_r{r}")):
            n = np.clip(d[c].to_numpy(dtype="float32") / 100.0, 0, 1)
            d[f"ghi_{p}_r{r}"] = (d.ghi_cs.to_numpy() * (1 - 0.75 * n ** 3.4)).astype("float32")
        d[f"ghi_mean_r{r}"] = ((d[f"ghi_A_r{r}"] + d[f"ghi_B_r{r}"]) / 2).astype("float32")
    if gfs is not None:
        # national NOAA GFS summary (one value per hour for every plant, 00 UTC cycle of the previous day; see 13_gfs_national.py)
        g = gfs.copy()
        g.index = g.index.as_unit(pd.DatetimeIndex(d.ts).unit)
        d = d.merge(g, left_on="ts", right_index=True, how="left")
        log(f"gfs national features merged ({g.shape[1]} columns)")
    d["site_code"] = d.site_id.str[2:].astype(int).astype("int16")
    for c in d.columns:
        if d[c].dtype == "float64" and c not in ("energy", "lat", "lon", "capacity_kw"):
            d[c] = d[c].astype("float32")
    log(f"features done {d.shape}")
    return d


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift-a", type=float, default=0.0)
    ap.add_argument("--shift-b", type=float, default=0.0)
    ap.add_argument("--ctx2", action="store_true", help="extended context features (panel_ctx2.parquet)")
    ap.add_argument("--ctx3", action="store_true", help="ctx2 plus wider lags and the day trajectory (panel_ctx3.parquet)")
    ap.add_argument("--ctx4", action="store_true", help="ctx3 plus previous-day regime and neighbour-station features (panel_ctx4.parquet)")
    ap.add_argument("--gfs", default=None, help="with --ctx4: also merge the national GFS summary from this raw table (writes panel_ctx5.parquet)")
    args = ap.parse_args()
    sites = load_sites()
    gen = pd.read_parquet(f"{DATA}/train/generation.parquet")
    fa = pd.read_parquet(f"{DATA}/train/forecast_wa.parquet", columns=["wa_station", "round", "ts"] + A_COLS)
    fb = pd.read_parquet(f"{DATA}/train/forecast_wb.parquet", columns=["wb_station", "round", "ts"] + B_COLS)
    gfs = None
    if args.gfs:
        from gfs_features import gfs_context
        gfs = gfs_context(args.gfs)
    d = build_feature_frame(gen[["site_id", "ts", "energy"]], fa, fb, sites, args.shift_a, args.shift_b, args.ctx2 or args.ctx3 or args.ctx4, args.ctx3 or args.ctx4, args.ctx4, gfs=gfs)
    d["u"] = (d.energy / d.capacity_kw).astype("float32")
    d = d.drop(columns=["wa_station", "wb_station"])
    assert len(d) == len(gen), (len(d), len(gen))
    name = "panel.parquet" if (args.shift_a, args.shift_b) == (0, 0) else f"panel_a{args.shift_a:g}_b{args.shift_b:g}.parquet"
    if args.ctx2:
        name = "panel_ctx2.parquet"
    if args.ctx3:
        name = "panel_ctx3.parquet"
    if args.ctx4:
        name = "panel_ctx4.parquet"
    if args.gfs:
        name = "panel_ctx5.parquet"
    d.to_parquet(f"{CACHE}/{name}", index=False)
    print("saved", f"{CACHE}/{name}", d.shape)


if __name__ == "__main__":
    main()
