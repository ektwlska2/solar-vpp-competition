"""Training-derived reliability of the day-ahead forecasts at each provider station (organiser-provided station IDs, no coordinates):
mean absolute error of the 17:00 round against the observed value over daylight hours (08-16) up to the training cut-off.
Static per plant (via its WA / WB station); used as `skill_*` columns so that a model can weigh provider A against provider B per site."""
from __future__ import annotations
import numpy as np
import pandas as pd
from oibc_lib import *


def _mae(fc: pd.DataFrame, ob: pd.DataFrame, st: str, col: str, cut) -> pd.Series:
    f = fc[(fc["round"] == 17) & (fc.ts <= cut)][[st, "ts", col]].rename(columns={col: "f"})
    o = ob[ob.ts <= cut][[st, "ts", col]].rename(columns={col: "o"})
    m = f.merge(o, on=[st, "ts"]).dropna()
    m = m[(m.ts.dt.hour >= 8) & (m.ts.dt.hour <= 16)]
    return (m.f - m.o).abs().groupby(m[st]).mean()


def station_skill(cut, sites: pd.DataFrame) -> pd.DataFrame:
    """returns DataFrame[site_id, skill_wa_cloud, skill_wb_cloud, skill_wb_uv, skill_wa_temp]"""
    fa = pd.read_parquet(f"{DATA}/train/forecast_wa.parquet", columns=["wa_station", "round", "ts", "wa_cloud", "wa_temp"])
    oa = pd.read_parquet(f"{DATA}/train/observed_wa.parquet", columns=["wa_station", "ts", "wa_cloud", "wa_temp"])
    fb = pd.read_parquet(f"{DATA}/train/forecast_wb.parquet", columns=["wb_station", "round", "ts", "wb_cloud", "wb_uv_idx"])
    ob = pd.read_parquet(f"{DATA}/train/observed_wb.parquet", columns=["wb_station", "ts", "wb_cloud", "wb_uv_idx"])
    s = sites[["site_id", "wa_station", "wb_station"]].copy()
    s["skill_wa_cloud"] = s.wa_station.map(_mae(fa, oa, "wa_station", "wa_cloud", cut))
    s["skill_wa_temp"] = s.wa_station.map(_mae(fa, oa, "wa_station", "wa_temp", cut))
    s["skill_wb_cloud"] = s.wb_station.map(_mae(fb, ob, "wb_station", "wb_cloud", cut))
    s["skill_wb_uv"] = s.wb_station.map(_mae(fb, ob, "wb_station", "wb_uv_idx", cut))
    s["skill_ab_cloud_ratio"] = s.skill_wa_cloud / s.skill_wb_cloud
    return s[["site_id", "skill_wa_cloud", "skill_wb_cloud", "skill_wb_uv", "skill_wa_temp", "skill_ab_cloud_ratio"]].astype({c: "float32" for c in s.columns if c.startswith("skill_")})
