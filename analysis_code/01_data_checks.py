"""Data audit: reproduces the numbers quoted in section 3 of the report (sites, generation, forecasts, SMP).

    python 01_data_checks.py            (about a minute; prints to the terminal, writes <CACHE>/data_checks.json)
The timestamp, usefulness and duplicate checks live in 01b_review_checks.py.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from oibc_lib import *

out = {}
sites = load_sites()
cap = sites.set_index("site_code").capacity_kw
print("== sites")
print(f"{len(sites)} sites, {sites.capacity_kw.sum():,.1f} kW; <=100 kW: {(sites.capacity_kw <= 100).sum()}; 50-100 kW: {sites.capacity_kw.between(50, 100.0001).sum()}")
cs = sites.capacity_kw.sort_values(ascending=False)
print(f"largest 10 hold {cs.head(10).sum()/cs.sum():.1%}, largest 25 hold {cs.head(25).sum()/cs.sum():.1%}, max {cs.max():,.1f} kW")
g = sites.groupby("current_group").capacity_kw.agg(["count", "sum"])
print("reference split:", g.to_dict("index"))
print("sites per station (median / max): WA", int(sites.wa_station.value_counts().median()), int(sites.wa_station.value_counts().max()),
      " WB", int(sites.wb_station.value_counts().median()), int(sites.wb_station.value_counts().max()))
print("station distance km (median / max): A", sites.wa_dist.median() / 1000, sites.wa_dist.max() / 1000, " B", sites.wb_dist.median() / 1000, sites.wb_dist.max() / 1000)

print("\n== SMP")
tsmp = pd.read_csv(f"{DATA}/smp.csv")
tsmp["ts"] = pd.to_datetime(tsmp.ts)
trs = pd.read_csv(f"{DATA}/train/smp.csv")
trs["ts"] = pd.to_datetime(trs.ts)
print("mean SMP by year:", trs.groupby(trs.ts.dt.year).smp.mean().round(1).to_dict(), "| scoring year", round(tsmp.smp.mean(), 1))
print("k=3 hours:", int((tsmp.k == 3).sum()), f"= {(tsmp.k == 3).mean():.1%} of hours, {(tsmp.smp * tsmp.k)[tsmp.k == 3].sum() / (tsmp.smp * tsmp.k).sum():.1%} of SMP*k mass; zero-price hours:", int((tsmp.smp == 0).sum()))

print("\n== generation")
gen = pd.read_parquet(f"{DATA}/train/generation.parquet")
gen = gen.merge(sites[["site_id", "capacity_kw"]], on="site_id")
gen["u"] = gen.energy / gen.capacity_kw
print(f"{len(gen):,} rows; negatives {(gen.energy < 0).sum()} (min {gen.energy.min()}); u>1: {(gen.u > 1).sum()} (max {gen.u.max():.3f})")
sp = gen.groupby("site_id").agg(first=("ts", "min"), last=("ts", "max"), n=("ts", "size"))
sp["cov"] = sp.n / ((sp["last"] - sp["first"]).dt.total_seconds() / 3600 + 1)
print("coverage of each site's own span:", sp["cov"].min().round(5), "-", sp["cov"].max())
end = K("2025-06-01 00:00")
hist = (end - sp["first"]).dt.days
capi = sites.set_index("site_id").capacity_kw
for d in (180, 365, 730):
    sel = hist[hist < d].index
    print(f"  sites with < {d} days of history: {len(sel)} ({capi.reindex(sel).sum() / capi.sum():.1%} of capacity)")
h = gen.ts.dt.hour
print("mean utilisation by hour label:", gen.groupby(h).u.mean().round(3).to_dict())
mid = gen[(h >= 10) & (h <= 15)]
sd = mid.groupby(["site_id", mid.ts.dt.date]).energy.sum()
print(f"site-days with zero energy over hour labels 10-15: {(sd == 0).sum()} of {len(sd)} ({(sd == 0).mean():.2%})")
print("monthly mean utilisation (hours 6-19):", gen[(h >= 6) & (h <= 19)].groupby(gen.ts.dt.month).u.mean().round(3).to_dict())

print("\n== forecasts")
for nm in ("wa", "wb"):
    tr = pd.read_parquet(f"{DATA}/train/forecast_{nm}.parquet")
    te = pd.read_parquet(f"{DATA}/test/forecast_{nm}.parquet")
    st = f"{nm}_station"
    idx = pd.MultiIndex.from_product([sorted(te[st].unique()), [10, 17], pd.date_range("2025-06-01 01:00", "2026-06-01 00:00", freq="h", tz=TZ)])
    have = pd.MultiIndex.from_frame(te[[st, "round", "ts"]])
    miss = idx.difference(have)
    nan_rows = int(te.isna().any(axis=1).sum())
    print(f"{nm.upper()}: train rows {len(tr):,}, test rows {len(te):,}, absent test rows {len(miss)}, NaN test rows {nan_rows}")
    if nm == "wa":
        m = pd.DataFrame(list(miss), columns=["st", "r", "ts"])
        both = m.groupby(["st", "ts"]).r.nunique()
        aff = sites[sites.wa_station.isin(m.st.unique())]
        print(f"   station-hours with BOTH rounds absent: {int((both == 2).sum())}; sites on affected stations {len(aff)} ({aff.capacity_kw.sum() / sites.capacity_kw.sum():.1%} of capacity)")
    obs = pd.read_parquet(f"{DATA}/train/observed_{nm}.parquet").drop_duplicates([st, "ts"])
    for r in (10, 17):
        m = tr[tr["round"] == r].merge(obs, on=[st, "ts"], suffixes=("_f", "_o"))
        v = f"{nm}_cloud"
        ok = m[f"{v}_o"] <= 100
        print(f"   round {r}: cloud corr with observed {np.corrcoef(m.loc[ok, f'{v}_f'], m.loc[ok, f'{v}_o'])[0, 1]:.3f}, MAE {np.abs(m.loc[ok, f'{v}_f'] - m.loc[ok, f'{v}_o']).mean():.1f}")
# ---- forecast skill vs observed weather, with physically impossible observations removed (they dominate Pearson r otherwise)
print("\n== round-17 forecast vs observed (invalid observations removed: cloud > 100, humidity > 100, wind > 50 m/s (A) or > 200 km/h (B))")
valid = {"wa": dict(cloud=lambda x: x <= 100, humidity=lambda x: x <= 100, wind=lambda x: x < 50, temp=lambda x: x > -90),
         "wb": dict(cloud=lambda x: x <= 100, humidity=lambda x: x <= 100, wind=lambda x: x <= 200, temp=lambda x: x > -90)}
cols = {"wa": dict(cloud="wa_cloud", humidity="wa_humidity", wind="wa_wind_speed", temp="wa_temp"),
        "wb": dict(cloud="wb_cloud", humidity="wb_rel_hum", wind="wb_wind_speed", temp="wb_temp")}
skill = {}
for nm in ("wa", "wb"):
    tr = pd.read_parquet(f"{DATA}/train/forecast_{nm}.parquet")
    st = f"{nm}_station"
    obs = pd.read_parquet(f"{DATA}/train/observed_{nm}.parquet").drop_duplicates([st, "ts"])
    m = tr[tr["round"] == 17].merge(obs, on=[st, "ts"], suffixes=("_f", "_o"))
    for k, c in cols[nm].items():
        ok = valid[nm][k](m[f"{c}_o"]) & m[f"{c}_f"].notna()
        raw_r = np.corrcoef(m[f"{c}_f"][m[f"{c}_o"].notna()], m[f"{c}_o"].dropna())[0, 1]
        r = np.corrcoef(m.loc[ok, f"{c}_f"], m.loc[ok, f"{c}_o"])[0, 1]
        skill[f"{nm}_{k}"] = dict(corr_clean=float(r), corr_raw=float(raw_r), dropped=int((~ok).sum()), mae_clean=float(np.abs(m.loc[ok, f"{c}_f"] - m.loc[ok, f"{c}_o"]).mean()))
        print(f"   {nm.upper()} {k:9s} corr {r:.3f} (raw, uncleaned {raw_r:.3f}; {int((~ok).sum())} rows dropped)  MAE {skill[f'{nm}_{k}']['mae_clean']:.2f}")
out["skill_vs_observed"] = skill
json.dump(out, open(f"{CACHE}/data_checks.json", "w"), indent=1, default=str)
print("\nDone. Timestamp / usefulness / duplicate checks: python 01b_review_checks.py")
