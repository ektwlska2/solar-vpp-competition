"""Stage 1 (new, 2026-11-12): the EXPECTED OBSERVED sky at every provider-B station, learned from the training-period station observations.

    python 23_stage1_observed_weather.py --out DIR                                   # production: train on all observations (to 2025-06-01), predict the scoring year
    python 23_stage1_observed_weather.py --out DIR --cut 2025-01-01 --end 2025-06-01 # hold-out: train on observations up to the cut only, predict (cut, end]
    python 23_stage1_observed_weather.py --out DIR --cut 2025-01-01 --end 2025-06-01 --smoke   # code rehearsal on 6 stations / 240 days / 30 trees (not for use)

Why: the plant models have to learn "forecast -> power" from plant records that are short (79 % of the capacity has under two years), while every one of the 73 provider-B stations has
hourly OBSERVATIONS back to 2021.  The observed UV index and cloud follow the plants' output much more closely than the forecasts do (day level, within station and month: observed UV
+0.72..+0.85, forecast UV +0.52..+0.73).  A model "forecasts -> observed UV / cloud" can therefore be trained on the long station records, and its prediction is handed to the plant
models as an extra input (08_make_submission.py --stage1).  The organisers' briefing names this use of the observations ("how wrong the forecast will be can be learned").

Rules kept:
  * observations are TRAINING TARGETS only; no observation is ever an input, and none exists for the scoring year;
  * inputs are the two providers' day-ahead forecasts (both rounds, issued before 17:00 of the previous day) and the national NWP summaries of external_data/ (same value for every station);
  * a prediction for the scoring year (or for the hold-out period) comes from a model fitted on observations up to the training end only;
  * a prediction for a TRAINING hour is out-of-fold: the calendar months are dealt into 5 folds ((year*12 + month) mod 5) and each month is predicted by the model that never saw it,
    so the plant models see stage-1 values of the same kind in training as in the forecast period;
  * the day-level context (mean / max of the day's values) uses the target day's own forecasts only; nothing is normalised over a period.

Output: DIR/stage1_features.parquet  (wb_station, ts, s1_uv, s1_uv_lag1, s1_uv_lead1, s1_uv_dmean, s1_uv_dmax, s1_cloud, s1_cloud_dmean; hour labels 5..21)
        DIR/stage1_info.json         (training end, rows, features, parameters, out-of-fold skill against the raw forecast, by year)
One "pseudo site" per provider-B station is used to build the inputs with the unchanged feature builder (02_build_panel.build_feature_frame): the station's own B forecasts, the A forecasts
of the provider-A station that most of its plants use, sun position at the mean coordinates of its plants (allowed: sun position from the provided coordinates).
"""
import argparse, importlib, json, os, sys, time, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from oibc_lib import *

warnings.filterwarnings("ignore")
bp = importlib.import_module("02_build_panel")
pr = importlib.import_module("03_forecast_probes")
sub = importlib.import_module("08_make_submission")

T0 = time.time()
log = lambda m: print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
HERE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGETS = {"uv": ("wb_uv_idx", 0.0, 15.0), "cloud": ("wb_cloud", 0.0, 100.0)}        # observed column, clip range (the observed files hold a few impossible values)
N_FOLDS = 5


def national_nwp(nwp_dir):
    from gfs_features import gfs_context, gefs_context, aerosol_context, ecmwf_context, ecmwf18_context, nwp_consensus
    table = {"gfs": ("gfs_national_raw.parquet", gfs_context), "gefs": ("gefs_national_raw.parquet", gefs_context), "aer": ("aerosol_national_raw.parquet", aerosol_context),
             "ecmwf": ("ecmwf_national_raw.parquet", ecmwf_context), "ecmwf18": ("ecmwf18_national_raw.parquet", ecmwf18_context)}
    parts = [fn(os.path.join(nwp_dir, nm)) for nm, fn in [table[k] for k in sub.NWP_SOURCES["v6"]]]
    nwp = pd.concat(parts, axis=1)
    return pd.concat([nwp, nwp_consensus(nwp)], axis=1)                                     # the same national columns as recipe v6 (no regional block: it needs the plants' generation)


def pseudo_sites(sites):
    rows = []
    for k, (st, g) in enumerate(sites.groupby("wb_station"), 1):
        wa = g.groupby("wa_station").capacity_kw.sum().idxmax()                             # the provider-A station most of this B station's capacity uses
        rows.append(dict(site_id=f"SB{k:03d}", capacity_kw=100.0, lat=float(g.lat.mean()), lon=float(g.lon.mean()), wa_station=wa, wb_station=st, current_group="A"))
    return pd.DataFrame(rows)


def frame(ps, fa, fb, nwp):
    unit = fb.ts.dt.unit
    ts = pd.DatetimeIndex(sorted(fb.ts.unique()))
    ts = ts[ts.hour.isin(range(sub.DAY_HOURS[0], sub.DAY_HOURS[1] + 1))].as_unit(unit)
    grid = pd.MultiIndex.from_product([ps.site_id, ts], names=["site_id", "ts"]).to_frame(index=False)
    return bp.build_feature_frame(grid, fa, fb, ps, 0, 0, True, True, False, gfs=nwp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cut", default=None, help="hold-out: train on observations up to this date (00:00 KST) only; default = the whole training period (to 2025-06-01)")
    ap.add_argument("--end", default=None, help="hold-out: last day (00:00 KST) of the period to predict; required with --cut")
    ap.add_argument("--nwp-dir", default=os.path.join(HERE_DIR, "external_data"))
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--smoke", action="store_true", help="code rehearsal: 6 stations, last 240 days before the training end, 30 trees; the output is not for use")
    args = ap.parse_args()
    assert (args.cut is None) == (args.end is None), "--cut and --end go together"
    tcut = K(f"{args.cut} 00:00") if args.cut else K("2025-06-01 00:00")
    sites = load_sites()
    ps = pseudo_sites(sites)
    if args.smoke:
        ps = ps.iloc[:6].reset_index(drop=True); args.trees = 30
    nwp = national_nwp(args.nwp_dir)
    log(f"{len(ps)} provider-B stations; national NWP summaries {nwp.shape[1]} columns; training end {tcut}")

    fa_all, fb_all = sub.load_forecasts("train")
    lo = (tcut - pd.Timedelta(days=240)) if args.smoke else fa_all.ts.min() - pd.Timedelta(hours=1)
    fa_tr, fb_tr = sub.prepare_tables(fa_all[(fa_all.ts > lo) & (fa_all.ts <= tcut)], fb_all[(fb_all.ts > lo) & (fb_all.ts <= tcut)])
    if args.cut:
        fa_te, fb_te = sub.prepare_tables(fa_all, fb_all, tcut, K(f"{args.end} 00:00"))
    else:
        fa_te, fb_te = sub.load_forecasts("test")
        fa_te, fb_te = sub.prepare_tables(fa_te, fb_te)
    tr = frame(ps, fa_tr, fb_tr, nwp)
    tg = frame(ps, fa_te, fb_te, nwp)
    assert tr.ts.max() <= tcut and tg.ts.min() > tcut, "training and target hours overlap"
    g = pr.groups(tr.columns)
    feats = [c for c in pr.only("raw", "ctx", "geo", "fleet", "blend", "phy")(g) if c not in ("lat", "lon", "cap")] + g["gfs"]
    missing = [c for c in feats if c not in tg.columns]
    assert not missing, f"target frame lacks stage-1 inputs: {missing[:5]}"
    assert not any(c.startswith(("obs_", "s1_")) for c in feats)
    log(f"frames: training {len(tr):,} station-hours, target {len(tg):,} station-hours ({tg.ts.min()} .. {tg.ts.max()}); {len(feats)} inputs")

    # ---- observations: training targets only
    ob = pd.read_parquet(f"{DATA}/train/observed_wb.parquet", columns=["wb_station", "ts"] + [v[0] for v in TARGETS.values()])
    ob["ts"] = pd.DatetimeIndex(ob.ts).tz_convert(TZ).as_unit(tr.ts.dt.unit)
    ob = ob[ob.ts <= tcut].drop_duplicates(["wb_station", "ts"]).rename(columns={v[0]: f"obs_{k}" for k, v in TARGETS.items()})
    n = len(tr)
    tr = tr.merge(ob, on=["wb_station", "ts"], how="left")
    assert len(tr) == n
    loc = (pd.DatetimeIndex(tr.ts) - pd.Timedelta(hours=1)).tz_convert(TZ)
    fold = ((loc.year * 12 + loc.month - 1) % N_FOLDS).to_numpy()
    params = dict(objective="l2", n_estimators=args.trees, learning_rate=0.06, num_leaves=127, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                  n_jobs=10, random_state=0, verbose=-1)
    Xtr, Xtg = tr[feats], tg[feats]
    info = dict(train_end=str(tcut), target_first=str(tg.ts.min()), target_last=str(tg.ts.max()), stations=int(len(ps)), n_train=int(len(tr)), n_target=int(len(tg)), inputs=feats,
                lgb_params=params, folds=f"(year*12 + month - 1) mod {N_FOLDS} of the target day", smoke=bool(args.smoke), skill={}, built=time.strftime("%Y-%m-%d %H:%M:%S"))
    out_tr, out_tg = tr[["wb_station", "ts"]].copy(), tg[["wb_station", "ts"]].copy()
    for k, (col, a, b) in TARGETS.items():
        y = tr[f"obs_{k}"].clip(a, b)
        has = y.notna().to_numpy()
        oof = np.full(len(tr), np.nan, "float32")
        for f in range(N_FOLDS):
            t1 = time.time()
            m = lgb.LGBMRegressor(**params).fit(Xtr[has & (fold != f)], y[has & (fold != f)])
            oof[fold == f] = m.predict(Xtr[fold == f])
            log(f"{k}: fold {f} fitted on {int((has & (fold != f)).sum()):,} rows, predicted {int((fold == f).sum()):,} in {time.time()-t1:.0f}s")
            del m
        t1 = time.time()
        m = lgb.LGBMRegressor(**params).fit(Xtr[has], y[has])
        out_tg[f"s1_{k}"] = np.clip(m.predict(Xtg), a, b).astype("float32")
        out_tr[f"s1_{k}"] = np.clip(oof, a, b)
        log(f"{k}: full model fitted on {int(has.sum()):,} rows in {time.time()-t1:.0f}s")
        del m
        # out-of-fold skill against the raw round-17 forecast of the same variable (hour labels 9..17), by year: a diagnostic of stage 1 only
        d = pd.DataFrame(dict(y=y.to_numpy(), p=out_tr[f"s1_{k}"].to_numpy(), r=tr[f"{col}_r17"].to_numpy(), yr=loc.year, hr=pd.DatetimeIndex(tr.ts).hour))
        d = d[(d.hr >= 9) & (d.hr <= 17)].dropna()
        sk = {}
        for yr, q in list(d.groupby("yr")) + [("all", d)]:
            sk[str(yr)] = dict(n=int(len(q)), corr_stage1=round(float(q.y.corr(q.p)), 4), corr_raw_forecast=round(float(q.y.corr(q.r)), 4),
                               rmse_stage1=round(float(np.sqrt(((q.y - q.p) ** 2).mean())), 4), rmse_raw_forecast=round(float(np.sqrt(((q.y - q.r.clip(a, b)) ** 2).mean())), 4))
            log(f"   {k} {yr}: observed vs out-of-fold stage 1 corr {sk[str(yr)]['corr_stage1']:.3f} rmse {sk[str(yr)]['rmse_stage1']:.3f} | vs raw forecast corr {sk[str(yr)]['corr_raw_forecast']:.3f} rmse {sk[str(yr)]['rmse_raw_forecast']:.3f}")
        info["skill"][k] = sk

    # ---- same-day context of the stage-1 values, per station (the day's own values only)
    s1 = pd.concat([out_tr, out_tg], ignore_index=True).sort_values(["wb_station", "ts"]).reset_index(drop=True)
    assert not s1.duplicated(["wb_station", "ts"]).any()
    day = (pd.DatetimeIndex(s1.ts) - pd.Timedelta(hours=1)).tz_convert(TZ).normalize()
    gk = [s1.wb_station.to_numpy(), np.asarray(day.tz_localize(None))]
    s1["s1_uv_dmean"] = s1.groupby(gk).s1_uv.transform("mean").astype("float32")
    s1["s1_uv_dmax"] = s1.groupby(gk).s1_uv.transform("max").astype("float32")
    s1["s1_cloud_dmean"] = s1.groupby(gk).s1_cloud.transform("mean").astype("float32")
    key = s1.set_index(["wb_station", "ts"]).s1_uv
    for nm, dt in (("lag1", -1), ("lead1", 1)):
        idx = pd.MultiIndex.from_arrays([s1.wb_station, s1.ts + pd.Timedelta(hours=dt)])
        s1[f"s1_uv_{nm}"] = key.reindex(idx).to_numpy("float32")
    cols = ["s1_uv", "s1_uv_lag1", "s1_uv_lead1", "s1_uv_dmean", "s1_uv_dmax", "s1_cloud", "s1_cloud_dmean"]
    os.makedirs(args.out, exist_ok=True)
    s1[["wb_station", "ts"] + cols].to_parquet(os.path.join(args.out, "stage1_features.parquet"), index=False)
    info["columns"] = cols; info["rows"] = int(len(s1))
    json.dump(info, open(os.path.join(args.out, "stage1_info.json"), "w"), indent=1)
    log(f"wrote {os.path.join(args.out, 'stage1_features.parquet')}: {len(s1):,} station-hours, {len(cols)} columns" + ("   SMOKE RUN: not for use" if args.smoke else ""))


if __name__ == "__main__":
    main()
