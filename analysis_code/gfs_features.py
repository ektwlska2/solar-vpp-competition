"""National NOAA GFS summary -> hourly, label-aligned model features (the same value for every plant).

Source table: <CACHE>/gfs_national_raw.parquet or external_data/gfs_national_raw.parquet, written by 13_gfs_national.py
(box mean of GFS 1-degree surface downward shortwave radiation and total cloud cover from the 00 UTC cycle of the day BEFORE each target day; see that script for the rules basis).

gfs_context(path) -> DataFrame indexed by the competition's hour labels (KST, right-closed), columns
    gfs_dsw, gfs_tcc, gfs_kt                       value at the hour (radiation W/m2, cloud %, clear-sky index)
    gfs_{kt,tcc}_{lag1,lag2,lead1,lead2}           neighbouring hours INSIDE the same target day (never the next day: that comes from a later cycle)
    gfs_{kt,tcc}_{dmean,dmin,dmax}                 day statistics over hour labels 9-16
    gfs_dsw_dsum                                   daily radiation sum over hour labels 6-19
Each target day is built from that day's own file set only, so a day's features cannot depend on any other day (checked in 10_test_causality.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from oibc_lib import *
import os
DSUM_MIN = 1 if os.environ.get("OIBC_DSUM_NAN", "0") == "1" else 0   # post-contest (18 Nov 2026, V19p): a day without any value gets an EMPTY day sum; default 0 = the sum of nothing is 0 (V6-V19 unchanged)

CENTRE = (36.0, 127.8)               # centre of the box, only used for the clear-sky normaliser


def gfs_hourly(raw_path: str) -> pd.DataFrame:
    raw = pd.read_parquet(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])
    labels = np.arange(1, 25)
    mid = labels - 0.5
    out = []
    for day, g in raw.groupby("date"):
        g = g.set_index("fh").sort_index()
        centres, vals = [], []
        for fh in g.index:
            a, b, v = g.at[fh, "a"], g.at[fh, "b"], g.at[fh, "dsw"]
            if np.isnan(v):
                continue
            if b - a == 3:                                              # already a 3-hour mean
                w, c = v, (a + b) / 2.0 - 15.0
            elif b - a == 6 and (fh - 3) in g.index and not np.isnan(g.at[fh - 3, "dsw"]) and g.at[fh - 3, "a"] == a:
                w, c = 2.0 * v - g.at[fh - 3, "dsw"], (b - 1.5) - 15.0  # de-accumulate the 6-hour bucket: mean of the last 3 hours
            else:
                continue
            centres.append(c); vals.append(max(w, 0.0))
        tcc = g.tcc.dropna()
        d = pd.DataFrame({"ts": pd.Timestamp(day, tz=TZ) + pd.to_timedelta(labels, unit="h")})
        d["gfs_dsw"] = np.interp(mid, centres, vals, left=np.nan, right=np.nan) if len(centres) >= 2 else np.nan
        d["gfs_tcc"] = np.interp(mid, (tcc.index.values - 15.0), tcc.values, left=np.nan, right=np.nan) if len(tcc) >= 2 else np.nan
        out.append(d)
    h = pd.concat(out, ignore_index=True)
    _, _, cz = solar(pd.DatetimeIndex(h.ts), CENTRE[0], CENTRE[1])
    cs = haurwitz_ghi(cz)
    h["gfs_kt"] = np.where(cs > 50, np.clip(h.gfs_dsw / np.maximum(cs, 50), 0, 1.3), np.nan)
    return h.set_index("ts")


def gfs_context(raw_path: str) -> pd.DataFrame:
    h = gfs_hourly(raw_path)
    h["day"] = (h.index - pd.Timedelta(hours=1)).tz_localize(None).normalize()
    hr = h.index.hour
    for c in ("gfs_kt", "gfs_tcc"):
        sel = h[c].where((hr >= 9) & (hr <= 16))
        for nm, fn in (("dmean", "mean"), ("dmin", "min"), ("dmax", "max")):
            h[f"{c}_{nm}"] = sel.groupby(h["day"]).transform(fn)
        for k in (1, 2):
            h[f"{c}_lag{k}"] = h.groupby("day")[c].shift(k)
            h[f"{c}_lead{k}"] = h.groupby("day")[c].shift(-k)
    h["gfs_dsw_dsum"] = h["gfs_dsw"].where((hr >= 6) & (hr <= 19)).groupby(h["day"]).transform("sum", min_count=DSUM_MIN)
    return h.drop(columns=["day"]).astype("float32")


# ------------------------------------------------------------------------------------------------------------- GEFS ensemble (14_gefs_national.py)
def gefs_hourly(raw_path: str) -> pd.DataFrame:
    raw = pd.read_parquet(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])
    labels = np.arange(1, 25)
    mid = labels - 0.5
    series = ("dsw_m", "dsw_sd", "tcc_m", "tcc_sd", "dsw_sp", "tcc_sp")
    out = []
    for day, g in raw.groupby("date"):
        g = g.set_index("fh").sort_index()
        d = pd.DataFrame({"ts": pd.Timestamp(day, tz=TZ) + pd.to_timedelta(labels, unit="h")})
        for s in series:
            centres, vals = [], []
            for fh in g.index:
                a, b, v = g.at[fh, "a"], g.at[fh, "b"], g.at[fh, s]
                if np.isnan(v) or np.isnan(a):
                    continue
                if b - a == 3:
                    w, c = v, (a + b) / 2.0 - 15.0
                elif b - a == 6 and (fh - 3) in g.index and not np.isnan(g.at[fh - 3, s]) and g.at[fh - 3, "a"] == a and s not in ("dsw_sp", "tcc_sp"):
                    w, c = 2.0 * v - g.at[fh - 3, s], (b - 1.5) - 15.0       # de-accumulate the 6-hour bucket mean
                elif b - a == 6:
                    w, c = v, (a + b) / 2.0 - 15.0                           # spreads cannot be de-accumulated: keep the bucket value at its centre
                else:
                    continue
                centres.append(c); vals.append(max(w, 0.0))
            if len(centres) >= 2:
                o = np.argsort(centres)
                d["gefs_" + s] = np.interp(mid, np.asarray(centres)[o], np.asarray(vals)[o], left=np.nan, right=np.nan)
            else:
                d["gefs_" + s] = np.nan
        out.append(d)
    h = pd.concat(out, ignore_index=True)
    _, _, cz = solar(pd.DatetimeIndex(h.ts), CENTRE[0], CENTRE[1])
    cs = haurwitz_ghi(cz)
    ok = cs > 50
    h["gefs_kt"] = np.where(ok, np.clip(h.gefs_dsw_m / np.maximum(cs, 50), 0, 1.3), np.nan)
    h["gefs_kt_sp"] = np.where(ok, h.gefs_dsw_sp / np.maximum(cs, 50), np.nan)           # ensemble spread in clear-sky-index units
    h["gefs_kt_patchy"] = np.where(ok, h.gefs_dsw_sd / np.maximum(cs, 50), np.nan)         # spatial standard deviation of the ensemble mean inside the box
    return h.set_index("ts")


def gefs_context(raw_path: str) -> pd.DataFrame:
    h = gefs_hourly(raw_path)
    h["day"] = (h.index - pd.Timedelta(hours=1)).tz_localize(None).normalize()
    hr = h.index.hour
    for c in ("gefs_kt", "gefs_tcc_m", "gefs_kt_sp"):
        sel = h[c].where((hr >= 9) & (hr <= 16))
        for nm, fn in (("dmean", "mean"), ("dmax", "max")):
            h[f"{c}_{nm}"] = sel.groupby(h["day"]).transform(fn)
    for c in ("gefs_kt", "gefs_tcc_m"):
        for k in (1, 2):
            h[f"{c}_lag{k}"] = h.groupby("day")[c].shift(k)
            h[f"{c}_lead{k}"] = h.groupby("day")[c].shift(-k)
    h["gefs_dsw_dsum"] = h["gefs_dsw_m"].where((hr >= 6) & (hr <= 19)).groupby(h["day"]).transform("sum", min_count=DSUM_MIN)
    return h.drop(columns=["day"]).astype("float32")


# ------------------------------------------------------------------------------------------------------------- GEFS-Aerosols (15_aerosol_national.py)
AEROSOL_VALID_FROM = "2023-01-01"        # the GEFS-Aerosols configuration changed around the end of 2022 (AOD level drops by about a third): earlier days are left empty


def aerosol_context(raw_path: str, dust: bool = False, valid_from: str = AEROSOL_VALID_FROM) -> pd.DataFrame:
    """Daily aerosol descriptors (same value for the 24 labels of the target day): AOD 550 nm and surface PM2.5 of the total aerosol, mean of the 12:00 and 15:00 KST forecasts.
    Dust-specific fields (AOD and PM2.5 of dust, dust share) are available with dust=True but are left out of the models: their winter 2025/26 level is 2-3 times above every earlier winter,
    which looks like a scheme change rather than weather."""
    raw = pd.read_parquet(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])
    cols = ["aod", "pm25"] + (["aod_dust", "pm25_dust"] if dust else [])
    d = raw.groupby("date")[cols].mean()
    d.columns = ["aer_" + c for c in cols]
    labels = np.arange(1, 25)
    idx = pd.DatetimeIndex([pd.Timestamp(day, tz=TZ) + pd.Timedelta(hours=int(h)) for day in d.index for h in labels])
    out = pd.DataFrame(np.repeat(d.to_numpy(), len(labels), axis=0), index=idx, columns=d.columns)
    if dust:
        out["aer_dust_share"] = out.aer_aod_dust / out.aer_aod.clip(lower=0.02)
    out.loc[out.index < pd.Timestamp(valid_from, tz=TZ)] = np.nan
    return out.astype("float32")


# ------------------------------------------------------------------------------------------------------------- more GEFS ensemble-mean fields (16_gefs_extra.py)
def gefs_extra_context(raw_path: str) -> pd.DataFrame:
    """Day-level descriptors from the 12:00 and 15:00 KST ensemble-mean forecasts (same value for the 24 labels of the target day):
    precipitable water, 2 m relative humidity, 2 m temperature (means of the two times) and the midday precipitation (09:00-15:00 KST, mm) with its ensemble spread."""
    raw = pd.read_parquet(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])
    d = raw.groupby("date")[["pwat", "rh2m", "t2m"]].mean()
    d.columns = ["gefs_x_pwat", "gefs_x_rh2m", "gefs_x_t2m"]
    p = raw[(raw.fh == 30) & (raw.a == 24) & (raw.b == 30)].set_index("date")           # the 24-30 h bucket = 09:00-15:00 KST of the target day
    d["gefs_x_apcp_mid"] = p["apcp"].reindex(d.index)
    d["gefs_x_apcp_sp"] = p["apcp_sp"].reindex(d.index)
    d["gefs_x_t2m"] = d["gefs_x_t2m"] - 273.15
    labels = np.arange(1, 25)
    idx = pd.DatetimeIndex([pd.Timestamp(day, tz=TZ) + pd.Timedelta(hours=int(h)) for day in d.index for h in labels])
    return pd.DataFrame(np.repeat(d.to_numpy(), len(labels), axis=0), index=idx, columns=d.columns).astype("float32")


# ------------------------------------------------------------------------------------------------------------- ECMWF IFS open data (17_ecmwf_national.py)
def ecmwf_hourly(raw_path: str, offset: float = 27.0, prefix: str = "ecmwf") -> pd.DataFrame:
    """offset = hours between the initialisation of the cycle and 00:00 KST of the target day (12 UTC of two days before: 27 h; 18 UTC of two days before: 21 h)."""
    raw = pd.read_parquet(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])
    labels = np.arange(1, 25)
    mid = labels - 0.5
    out = []
    for day, g in raw.groupby("date"):
        g = g.set_index("step").sort_index()
        centres, vals = [], []
        st = list(g.index)
        for a, b in zip(st[:-1], st[1:]):
            if b - a == 3 and not np.isnan(g.at[a, "ssrd"]) and not np.isnan(g.at[b, "ssrd"]):
                centres.append((a + b) / 2.0 - offset)
                vals.append(max((g.at[b, "ssrd"] - g.at[a, "ssrd"]) / 10800.0, 0.0))
        d = pd.DataFrame({"ts": pd.Timestamp(day, tz=TZ) + pd.to_timedelta(labels, unit="h")})
        d[f"{prefix}_dsw"] = np.interp(mid, centres, vals, left=np.nan, right=np.nan) if len(centres) >= 2 else np.nan
        out.append(d)
    h = pd.concat(out, ignore_index=True)
    _, _, cz = solar(pd.DatetimeIndex(h.ts), CENTRE[0], CENTRE[1])
    cs = haurwitz_ghi(cz)
    h[f"{prefix}_kt"] = np.where(cs > 50, np.clip(h[f"{prefix}_dsw"] / np.maximum(cs, 50), 0, 1.3), np.nan)
    return h.set_index("ts")


def ecmwf_context(raw_path: str, offset: float = 27.0, prefix: str = "ecmwf") -> pd.DataFrame:
    h = ecmwf_hourly(raw_path, offset, prefix)
    h["day"] = (h.index - pd.Timedelta(hours=1)).tz_localize(None).normalize()
    hr = h.index.hour
    sel = h[f"{prefix}_kt"].where((hr >= 9) & (hr <= 16))
    for nm, fn in (("dmean", "mean"), ("dmin", "min"), ("dmax", "max")):
        h[f"{prefix}_kt_{nm}"] = sel.groupby(h["day"]).transform(fn)
    for k in (1, 2):
        h[f"{prefix}_kt_lag{k}"] = h.groupby("day")[f"{prefix}_kt"].shift(k)
        h[f"{prefix}_kt_lead{k}"] = h.groupby("day")[f"{prefix}_kt"].shift(-k)
    h[f"{prefix}_dsw_dsum"] = h[f"{prefix}_dsw"].where((hr >= 6) & (hr <= 19)).groupby(h["day"]).transform("sum", min_count=DSUM_MIN)
    return h.drop(columns=["day"]).astype("float32")


def ecmwf18_context(raw_path: str) -> pd.DataFrame:
    """The 18 UTC cycle of two days before the target day (6 hours fresher than the 12 UTC one)."""
    return ecmwf_context(raw_path, offset=21.0, prefix="ecmwf18")


def ecmwf0_context(raw_path: str) -> pd.DataFrame:
    """The 00 UTC cycle of the day BEFORE the target day (12 hours fresher than the 12 UTC cycle of D-2; public 16:34 KST)."""
    return ecmwf_context(raw_path, offset=15.0, prefix="ecmwf0")


# ------------------------------------------------------------------------------------------------------------- GEFS snow depth (18_gefs_snow.py)
def snow_context(raw_path: str) -> pd.DataFrame:
    """Day-level snow descriptors from the ensemble-mean forecasts of the target day and of the two days before it (all published before the cut-off), same value for the 24 labels."""
    raw = pd.read_parquet(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])
    d = raw.set_index("date").sort_index()[["snod", "weasd"]].rename(columns={"snod": "gefs_s_snod", "weasd": "gefs_s_weasd"})
    d["gefs_s_snod_prev1"] = d.gefs_s_snod.shift(1)
    d["gefs_s_snod_max3"] = d.gefs_s_snod.rolling(3, min_periods=1).max()
    labels = np.arange(1, 25)
    idx = pd.DatetimeIndex([pd.Timestamp(day, tz=TZ) + pd.Timedelta(hours=int(h)) for day in d.index for h in labels])
    return pd.DataFrame(np.repeat(d.to_numpy(), len(labels), axis=0), index=idx, columns=d.columns).astype("float32")


# ------------------------------------------------------------------------------------------------------------- multi-model consensus
def nwp_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-model mean and spread of the national clear-sky index (GFS, GEFS, ECMWF; whichever exist for the day) for the hourly value, its neighbours and the day statistics.
    A single averaged feature is less noisy than three correlated ones from which a tree can use only one per split."""
    out = {}
    for suf in ("", "_lag1", "_lag2", "_lead1", "_lead2", "_dmean", "_dmax", "_dmin"):
        cols = [c for c in (f"gfs_kt{suf}", f"gefs_kt{suf}", f"ecmwf_kt{suf}", f"ecmwf18_kt{suf}", f"ecmwf0_kt{suf}", f"jma_kt{suf}") if c in df.columns]   # jma_*: only recipes with the JMA table (post-contest)
        if len(cols) < 2:
            continue
        x = df[cols].to_numpy("float64")
        with np.errstate(all="ignore"):
            out[f"nwp_kt{suf}"] = np.nanmean(x, axis=1)
            if suf in ("", "_dmean"):
                out[f"nwp_kt_sd{suf}"] = np.nanstd(x, axis=1)
    out["nwp_dsw"] = np.nanmean(df[[c for c in ("gfs_dsw", "gefs_dsw_m", "ecmwf_dsw", "ecmwf18_dsw", "ecmwf0_dsw", "jma_dsw") if c in df.columns]].to_numpy("float64"), axis=1)
    return pd.DataFrame(out, index=df.index).astype("float32")


# ------------------------------------------------------------------------------------------------------------- observed national irradiance (24_power_national.py; V18, 2026-11-13)
def obs_hourly(raw_path: str) -> pd.DataFrame:
    """NASA POWER hourly satellite-derived surface irradiance (CERES SYN1deg), mean of the fixed 42 points of the box 33-39 N / 125-130 E; index = the competition's KST hour
    labels (right-closed).  For the evaluation period these are OBSERVED values of the target day itself, used on the basis of the organisers' written answer
    (추가된 안내사항/organisers_answer_observed_weather_and_rerun.md: evaluation-period ERA5 / "historical weather" values, wide-area average, identical for every plant)."""
    raw = pd.read_parquet(raw_path)
    h = pd.DataFrame({"ts": pd.DatetimeIndex(raw.ts).tz_convert(TZ), "gfs_obs_dsw": raw.obs_dsw.to_numpy("float64")})
    _, _, cz = solar(pd.DatetimeIndex(h.ts), CENTRE[0], CENTRE[1])
    cs = haurwitz_ghi(cz)
    h["gfs_obs_kt"] = np.where(cs > 50, np.clip(h.gfs_obs_dsw / np.maximum(cs, 50), 0, 1.3), np.nan)     # the same clear-sky normaliser as gfs_kt
    return h.set_index("ts").sort_index()


def obs_context(raw_path: str) -> pd.DataFrame:
    """gfs_obs_dsw, gfs_obs_kt, gfs_obs_kt_{lag1,lag2,lead1,lead2} (inside the same target day), gfs_obs_kt_{dmean,dmin,dmax} over hour labels 9-16, gfs_obs_dsw_dsum over 6-19.
    The prefix gfs_ puts the columns into the national-NWP feature group, i.e. into every tree and network member of the recipes that list the source 'obs'."""
    h = obs_hourly(raw_path)
    h["day"] = (h.index - pd.Timedelta(hours=1)).tz_localize(None).normalize()
    hr = h.index.hour
    c = "gfs_obs_kt"
    sel = h[c].where((hr >= 9) & (hr <= 16))
    for nm, fn in (("dmean", "mean"), ("dmin", "min"), ("dmax", "max")):
        h[f"{c}_{nm}"] = sel.groupby(h["day"]).transform(fn)
    for k in (1, 2):
        h[f"{c}_lag{k}"] = h.groupby("day")[c].shift(k)
        h[f"{c}_lead{k}"] = h.groupby("day")[c].shift(-k)
    h["gfs_obs_dsw_dsum"] = h["gfs_obs_dsw"].where((hr >= 6) & (hr <= 19)).groupby(h["day"]).transform("sum", min_count=DSUM_MIN)
    return h.drop(columns=["day"]).astype("float32")


# ------------------------------------------------------------------------------------------------------------- more GFS fields (26_gfs_extra_national.py; post-contest study, 18 Nov 2026)
GFSX_INST = ("lcdc", "mcdc", "hcdc", "ccdc", "cwat", "pwat", "vis", "cape", "rh850", "rh700")


def gfsx_hourly(raw_path: str) -> pd.DataFrame:
    """Hourly national values (KST labels, right-closed) of the extra GFS fields of the 00 UTC run of the day before the target day (public about 12:45 KST).
    Instantaneous fields: interpolated to the middle of each hour (as gfs_tcc).  Precipitation: 3- / 6-hour buckets de-accumulated to a rate per hour.
    Sunshine duration (accumulated since the last 6-hourly reset at forecast hours 18, 24, 30, 36): the sunshine fraction of each 3-hour block, and per day the hours of
    sunshine 09-15 KST (the bucket ending at forecast hour 30, i.e. almost exactly the k = 3 hours) and over the whole day."""
    raw = pd.read_parquet(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])
    labels = np.arange(1, 25)
    mid = labels - 0.5
    out = []
    for day, g in raw.groupby("date"):
        g = g.set_index("fh").sort_index()
        d = pd.DataFrame({"ts": pd.Timestamp(day, tz=TZ) + pd.to_timedelta(labels, unit="h")})
        for c in GFSX_INST:
            s = g[c].dropna()
            d[f"gfs_x_{c}"] = np.interp(mid, s.index.values - 15.0, s.values, left=np.nan, right=np.nan) if len(s) >= 2 else np.nan
        cen, val = [], []
        for fh in g.index:                                                  # precipitation rate per hour of each 3-hour block
            a, v = g.at[fh, "apcp_a"], g.at[fh, "apcp"]
            if np.isnan(v) or np.isnan(a):
                continue
            if fh - a == 3:
                r = v / 3.0
            elif fh - a == 6 and (fh - 3) in g.index and g.at[fh - 3, "apcp_a"] == a and not np.isnan(g.at[fh - 3, "apcp"]):
                r = (v - g.at[fh - 3, "apcp"]) / 3.0
            else:
                continue
            cen.append(fh - 1.5 - 15.0); val.append(max(r, 0.0))
        d["gfs_x_prate"] = np.interp(mid, cen, val, left=np.nan, right=np.nan) if len(cen) >= 2 else np.nan
        cen, val = [], []
        for fh in g.index:                                                  # sunshine fraction of each 3-hour block
            v = g.at[fh, "sunsd"]
            if np.isnan(v):
                continue
            if fh % 6 == 3:
                sec = v
            elif (fh - 3) in g.index and not np.isnan(g.at[fh - 3, "sunsd"]):
                sec = v - g.at[fh - 3, "sunsd"]
            else:
                continue
            cen.append(fh - 1.5 - 15.0); val.append(float(np.clip(sec / 10800.0, 0.0, 1.0)))
        d["gfs_x_sunfrac"] = np.interp(mid, cen, val, left=np.nan, right=np.nan) if len(cen) >= 2 else np.nan
        sun = [g.at[fh, "sunsd"] if fh in g.index else np.nan for fh in (24, 30, 36)]
        d["gfs_x_sun_0915"] = sun[1] / 3600.0
        d["gfs_x_sun_day"] = np.nansum(sun) / 3600.0 if not np.isnan(sun).all() else np.nan
        out.append(d)
    return pd.concat(out, ignore_index=True).set_index("ts")


def gfsx_context(raw_path: str) -> pd.DataFrame:
    """gfsx_hourly + day means over hour labels 9-16 of the cloud / water / visibility / precipitation / sunshine columns (the same value for every hour of the day)."""
    h = gfsx_hourly(raw_path)
    h["day"] = (h.index - pd.Timedelta(hours=1)).tz_localize(None).normalize()
    hr = h.index.hour
    for c in ("lcdc", "mcdc", "hcdc", "cwat", "pwat", "vis", "prate", "sunfrac", "rh850"):
        h[f"gfs_x_{c}_dmean"] = h[f"gfs_x_{c}"].where((hr >= 9) & (hr <= 16)).groupby(h["day"]).transform("mean")
    return h.drop(columns=["day"]).astype("float32")


# ------------------------------------------------------------------------------------------------------------- JMA GSM (27_jma_gsm_national.py), post-contest study
def jma_hourly(raw_path: str) -> pd.DataFrame:
    """National box means of JMA's GSM, 00 UTC run of the day before (posted by 12:38 KST), at the competition's hour labels: lead L = label L-15.
    dsw is the mean flux of the hour ending at the label (exact). Instantaneous fields (cloud, t2, rh2) are averaged over the hour's start and end; precipitation is
    the accumulation of the hour. Per target day the older 0.25 x 0.2 deg product is used when it exists (runs up to 2024-09-17), else the 0.1 deg one
    (both give the same box means within about 0.5 %)."""
    raw = pd.read_parquet(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])
    has_old = set(raw.loc[raw["product"] == "old", "date"])
    raw = raw[(raw["product"] == "old") | ~raw["date"].isin(has_old)].sort_values(["date", "lead"])
    out = []
    for day, g in raw.groupby("date"):
        g = g.set_index("lead")
        d = pd.DataFrame({"label": np.arange(2, 24)})
        L = d.label.to_numpy() + 15
        d["jma_dsw"] = g.dsw.reindex(L).to_numpy()
        for c in ("tcc", "lcc", "mcc", "hcc", "t2", "rh2"):
            d[f"jma_{c}"] = 0.5 * (g[c].reindex(L - 1).to_numpy() + g[c].reindex(L).to_numpy())
        d["jma_prate"] = np.clip(g.apcp.reindex(L).to_numpy() - g.apcp.reindex(L - 1).to_numpy(), 0, None)
        d["ts"] = pd.Timestamp(day, tz=TZ) + pd.to_timedelta(d.label, unit="h")
        out.append(d.drop(columns=["label"]))
    h = pd.concat(out, ignore_index=True)
    _, _, cz = solar(pd.DatetimeIndex(h.ts), CENTRE[0], CENTRE[1])
    cs = haurwitz_ghi(cz)
    h["jma_kt"] = np.where(cs > 50, np.clip(h.jma_dsw / np.maximum(cs, 50), 0, 1.3), np.nan)
    return h.set_index("ts")


def jma_context(raw_path: str) -> pd.DataFrame:
    """jma_{dsw,kt,tcc,lcc,mcc,hcc,prate,t2,rh2} at the hour; jma_{kt,tcc}_{lag1,lag2,lead1,lead2} inside the same target day; jma_{kt,tcc,lcc}_{dmean,dmin,dmax}
    over hour labels 9-16; jma_dsw_dsum over 6-19. Each target day uses only its own run."""
    h = jma_hourly(raw_path)
    h["day"] = (h.index - pd.Timedelta(hours=1)).tz_localize(None).normalize()
    hr = h.index.hour
    for c in ("jma_kt", "jma_tcc", "jma_lcc"):
        sel = h[c].where((hr >= 9) & (hr <= 16))
        for nm, fn in (("dmean", "mean"), ("dmin", "min"), ("dmax", "max")):
            h[f"{c}_{nm}"] = sel.groupby(h["day"]).transform(fn)
    for c in ("jma_kt", "jma_tcc"):
        for k in (1, 2):
            h[f"{c}_lag{k}"] = h.groupby("day")[c].shift(k)
            h[f"{c}_lead{k}"] = h.groupby("day")[c].shift(-k)
    h["jma_dsw_dsum"] = h["jma_dsw"].where((hr >= 6) & (hr <= 19)).groupby(h["day"]).transform("sum", min_count=DSUM_MIN)
    return h.drop(columns=["day"]).astype("float32")


# ------------------------------------------------------------------------------------------------------------- KMA UM-LDAPS 1.5 km (22_kma_ldaps.py), post-contest study
def ldaps_context(raw_path: str) -> pd.DataFrame:
    """National summaries of KMA's UM-LDAPS (1.5 km), 09 KST run of the day before (callable from 15:00 KST per the API Hub guide), from its 120 fixed 0.5-degree cells
    (regional_features.cell_kt_ldaps; no plant coordinate): ldaps_kt = mean clear-sky index of the cells at the hour, ldaps_kt_sd = spread across the cells (how patchy the
    forecast sky is), ldaps_kt_{lag1,lag2,lead1,lead2} inside the same target day, ldaps_kt_{dmean,dmin,dmax} over hour labels 9-16. Stored samples are the labels 9, 11, ..., 17;
    the hours in between are interpolated. Exists for target days 2023-09-01 .. 2026-04-01 only, so it goes to the tree members only (network inputs exclude it, like ECMWF)."""
    import regional_features as rf
    cells = rf.cell_kt_ldaps(raw_path)
    h = pd.DataFrame({"ldaps_kt": cells.mean(axis=1), "ldaps_kt_sd": cells.std(axis=1)}, index=cells.index)
    h["day"] = (h.index - pd.Timedelta(hours=1)).tz_localize(None).normalize()
    hr = h.index.hour
    sel = h["ldaps_kt"].where((hr >= 9) & (hr <= 16))
    for nm, fn in (("dmean", "mean"), ("dmin", "min"), ("dmax", "max")):
        h[f"ldaps_kt_{nm}"] = sel.groupby(h["day"]).transform(fn)
    for k in (1, 2):
        h[f"ldaps_kt_lag{k}"] = h.groupby("day")["ldaps_kt"].shift(k)
        h[f"ldaps_kt_lead{k}"] = h.groupby("day")["ldaps_kt"].shift(-k)
    return h.drop(columns=["day"]).astype("float32")
