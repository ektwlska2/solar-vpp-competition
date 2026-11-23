"""Weighted average of the site-level forecasts of several finished submissions -> a new submission (same composition).

    python 19_mix_submissions.py --inputs ../submissions/v2_blend ../submissions/v5_nwp_cons --weights 0.4 0.6 --out ../submissions/v6_mix --name V6_mix

Why: a model that uses the external weather-model features and one that does not make partly independent errors, so their average beats both
(hold-out year 1: blend of the two best weight 0.75 on the external-feature model -0.7% penalty against using it alone; Jan-May 2025 pseudo-test: weight 0.5 -2.2%).
The penalty is convex in the forecast, so an average is never worse than the average of the penalties.
All inputs must share the same composition (they do: the structural split from sites.csv).
"""
from __future__ import annotations
import argparse, glob, os, subprocess, sys
import numpy as np
import pandas as pd
from oibc_lib import *

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="submission folders that contain site_forecasts.parquet and <Name>_composition.csv")
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    w = np.array(args.weights, dtype="float64")
    assert len(w) == len(args.inputs) and abs(w.sum() - 1.0) < 1e-9, "weights must sum to 1"
    comps = [pd.read_csv(glob.glob(os.path.join(d, "*_composition.csv"))[0]) for d in args.inputs]
    for c in comps[1:]:
        assert comps[0].equals(c), "the inputs have different compositions"
    comp = comps[0]
    sf = None
    for d, wi in zip(args.inputs, w):
        x = pd.read_parquet(os.path.join(d, "site_forecasts.parquet"))[["site_id", "ts", "kwh"]]
        x["ts"] = pd.DatetimeIndex(x.ts).tz_convert(TZ)
        x = x.set_index(["site_id", "ts"]).kwh
        sf = wi * x if sf is None else sf.add(wi * x.reindex(sf.index), fill_value=0.0)
    sf = sf.rename("kwh").reset_index()
    g = comp.set_index("site_id").group
    fc = sf.assign(group=sf.site_id.map(g)).groupby(["group", "ts"], as_index=False).kwh.sum().rename(columns={"kwh": "forecast_kwh"})
    fc = fc.sort_values(["group", "ts"]).reset_index(drop=True)
    os.makedirs(args.out, exist_ok=True)
    name = args.name or os.path.basename(os.path.normpath(args.out)).capitalize()
    comp_path, fc_path = os.path.join(args.out, f"{name}_composition.csv"), os.path.join(args.out, f"{name}_forecast.csv")
    comp.to_csv(comp_path, index=False)
    out = fc.copy()
    out["ts"] = out.ts.dt.strftime("%Y-%m-%d %H:%M:%S") + "+09:00"
    out[["group", "ts", "forecast_kwh"]].to_csv(fc_path, index=False, float_format="%.4f")
    sf.assign(kwh=sf.kwh).to_parquet(os.path.join(args.out, "site_forecasts.parquet"), index=False)
    chk = pd.read_csv(fc_path)
    assert len(chk) == 17520 and chk.forecast_kwh.notna().all() and (chk.forecast_kwh >= 0).all()
    with open(os.path.join(args.out, "run_info.json"), "w") as f:
        import json
        json.dump(dict(kind="mix", inputs=[os.path.abspath(d) for d in args.inputs], weights=list(map(float, w))), f, indent=1)
    r = subprocess.run([sys.executable, os.path.join(HERE, "organiser_tools", "validate_submission.py"), "--composition", comp_path, "--forecast", fc_path, "--data", DATA], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    print(f"wrote {comp_path} and {fc_path}; annual energy " + ", ".join(f"{k} {v/1e6:.2f} GWh" for k, v in fc.groupby("group").forecast_kwh.sum().items()))
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
