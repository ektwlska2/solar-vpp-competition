"""Average several prediction files (same cut) into one and score it like 03_forecast_probes.py.

    python 03c_blend.py --cut 2024-06-01 --inputs G G_s1 --name G_2seeds [--weights 1 1]
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from oibc_lib import *

ap = argparse.ArgumentParser()
ap.add_argument("--cut", default="2024-06-01")
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--weights", nargs="*", type=float, default=None)
ap.add_argument("--name", required=True)
args = ap.parse_args()
cut = K(f"{args.cut} 00:00")
end = cut + pd.DateOffset(years=1)
w = np.array(args.weights if args.weights else [1.0] * len(args.inputs))
w = w / w.sum()
preds = [pd.read_parquet(f"{CACHE}/pred_{args.cut}_{n}.parquet").set_index(["site_code", "ts"]).pred_u for n in args.inputs]
blend = sum(wi * p.reindex(preds[0].index) for wi, p in zip(w, preds)).rename("pred_u").reset_index()
blend.to_parquet(f"{CACHE}/pred_{args.cut}_{args.name}.parquet", index=False)
panel = pd.read_parquet(f"{CACHE}/panel.parquet", columns=["site_code", "ts", "energy", "cap", "elev"])
d = error_matrix(blend, panel, cut, end)
cg = load_sites().set_index("site_code").current_group
mask = (cg.reindex(d["sites"]) == "A").values
res_path = f"{CACHE}/results_{args.cut}.csv"
results = pd.read_csv(res_path).set_index("exp")
row = dict(n_sites=len(d["sites"]), n_feat=np.nan, W_cur=W_of(d["E"], d["w"], mask), W_floor=W_floor(d["E"], d["w"]), W_nopool=W_nopool(d["E"], d["w"]), seconds=0.0)
results.loc[args.name, list(row)] = list(row.values())
results["score_vs_M0"] = 100 * (1 - results.W_cur / results.loc["M0", "W_cur"])
results.index.name = "exp"
results.to_csv(res_path)
print(f"{args.name}: W_cur {row['W_cur']/1e6:.2f}M floor {row['W_floor']/1e6:.2f}M score {results.loc[args.name, 'score_vs_M0']:+.2f}")
