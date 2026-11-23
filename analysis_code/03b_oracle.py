"""Oracle ceiling: the same kind of model fed OBSERVED station weather instead of day-ahead forecasts (not usable in the contest).

    python 03b_oracle.py --cut 2024-06-01
Shows how much penalty is due to weather-forecast error rather than to the model.  Output joins <CACHE>/results_<cut>.csv.
"""
from __future__ import annotations
import argparse, os, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from oibc_lib import *

warnings.filterwarnings("ignore")
OA = ["wa_cloud", "wa_temp", "wa_humidity", "wa_ground_press", "wa_wind_speed", "wa_rain"]
OB = ["wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_cloud", "wb_vis", "wb_wind_speed", "wb_dew_point", "wb_precip_1h", "wb_ceiling"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="2024-06-01")
    args = ap.parse_args()
    t0 = time.time()
    cut = K(f"{args.cut} 00:00")
    end = cut + pd.DateOffset(years=1)
    smp = load_smp_train()
    sites = load_sites()
    cg = sites.set_index("site_code").current_group
    p = pd.read_parquet(f"{CACHE}/panel.parquet", columns=["site_code", "ts", "energy", "u", "cap", "lat", "lon", "elev", "azim", "cosz", "doy_sin", "doy_cos", "hour"])
    full = p[["site_code", "ts", "energy", "cap", "elev"]].copy()
    p = p[p.elev > 0].merge(sites[["site_code", "wa_station", "wb_station"]], on="site_code", how="left")
    oa = pd.read_parquet(f"{DATA}/train/observed_wa.parquet").drop_duplicates(["wa_station", "ts"])[["wa_station", "ts"] + OA]
    ob = pd.read_parquet(f"{DATA}/train/observed_wb.parquet")[["wb_station", "ts"] + OB]
    oa = oa.rename(columns={c: "o_" + c for c in OA})
    ob = ob.rename(columns={c: "o_" + c for c in OB})
    oa.loc[oa.o_wa_wind_speed > 50, "o_wa_wind_speed"] = np.nan          # sentinel 9999 etc.
    p = p.merge(oa, on=["wa_station", "ts"], how="left").merge(ob, on=["wb_station", "ts"], how="left")
    feats = ["o_" + c for c in OA + OB] + GEO
    p["w"] = hour_weights(pd.DatetimeIndex(p.ts), smp).astype("float32")
    tr = p[(p.ts <= cut) & p.u.notna()]
    va = p[(p.ts > cut) & (p.ts <= end)]
    m = lgb.LGBMRegressor(objective="l1", n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=10)
    m.fit(tr[feats], tr.u.clip(0, 1.2), sample_weight=tr.w / tr.w.mean())
    out = va[["site_code", "ts"]].copy()
    out["pred_u"] = np.clip(m.predict(va[feats]), 0, None).astype("float32")
    out.to_parquet(f"{CACHE}/pred_{args.cut}_ORACLE.parquet", index=False)
    d = error_matrix(out, full, cut, end)
    mask = (cg.reindex(d["sites"]) == "A").values
    res_path = f"{CACHE}/results_{args.cut}.csv"
    results = pd.read_csv(res_path).set_index("exp") if os.path.exists(res_path) else pd.DataFrame()
    row = dict(n_sites=len(d["sites"]), n_feat=len(feats), W_cur=W_of(d["E"], d["w"], mask), W_floor=W_floor(d["E"], d["w"]), W_nopool=W_nopool(d["E"], d["w"]), seconds=round(time.time() - t0, 1))
    results.loc["ORACLE", list(row)] = list(row.values())
    results["score_vs_M0"] = 100 * (1 - results.W_cur / results.loc["M0", "W_cur"])
    results.index.name = "exp"
    results.to_csv(res_path)
    print(f"ORACLE  W_cur {row['W_cur']/1e6:.2f}M  score vs M0 {results.loc['ORACLE', 'score_vs_M0']:+.2f}")


if __name__ == "__main__":
    main()
