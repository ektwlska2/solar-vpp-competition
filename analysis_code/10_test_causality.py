"""Causality test of the feature builder: the features of a target day must not depend on forecasts of LATER days.

    python 10_test_causality.py

Builds all model features (same-day context, previous-day regime and neighbour-station features) for day D from the scoring-year
forecast tables, then
  1. destroys every forecast value of days AFTER D            -> every feature of day D must be identical
  2. destroys the forecasts of day D-1 (an earlier day)         -> the previous-day features must change (the test is not vacuous)
  3. destroys the forecasts of day D itself                     -> the same-day features must change
  4. national NWP summaries (external tables, recipe v3): the features of day D are rebuilt after destroying every raw row of all OTHER days -> identical;
  5. regional NWP cells (external tables, recipe v6): the cell clear-sky indices of day D are rebuilt after destroying the raw rows of all OTHER days -> identical;
     destroying the rows of day D itself -> they change
"""
from __future__ import annotations
import importlib
import numpy as np
import pandas as pd
from oibc_lib import *

bp = importlib.import_module("02_build_panel")
pr = importlib.import_module("03_forecast_probes")
sub = importlib.import_module("08_make_submission")

sites = load_sites()
fa_all, fb_all = sub.load_forecasts("train")
nb = {"wa": bp.compute_neighbors(fa_all, "wa"), "wb": bp.compute_neighbors(fb_all, "wb")}
tail = K("2025-06-01 00:00") - pd.Timedelta(days=45)
fa_tr, fb_tr = sub.prepare_tables(fa_all[fa_all.ts > tail], fb_all[fb_all.ts > tail])
fa_te, fb_te = sub.load_forecasts("test")
lo, hi = K("2025-06-01 00:00"), K("2025-06-13 00:00")
fa_te, fb_te = sub.prepare_tables(fa_te, fb_te, lo, hi)
D0, D1 = K("2025-06-06 00:00"), K("2025-06-07 00:00")          # day D = labels 01:00 .. 24:00 (the 00:00 label of 06-07 belongs to it)
sel = sites.iloc[::20].copy()
unit = fa_te.ts.dt.unit
hours = pd.date_range(lo + pd.Timedelta(hours=1), hi, freq="h").as_unit(unit)
grid = pd.MultiIndex.from_product([sel.site_id, hours], names=["site_id", "ts"]).to_frame(index=False)


def perturb(f, mask, seed):
    f = f.copy()
    rng = np.random.default_rng(seed)
    for c in f.columns:
        if c.startswith(("wa_", "wb_")) and not c.endswith("station"):
            f.loc[mask, c] = rng.uniform(0, 100, int(mask.sum()))
    return f


def build(fa_w, fb_w, fa_h, fb_h):
    d = bp.build_feature_frame(grid, fa_w, fb_w, sites, 0, 0, True, True, True, nb, fa_h, fb_h)
    return d[(d.ts > D0) & (d.ts <= D1) & (d.elev > 0)].sort_values(["site_id", "ts"]).reset_index(drop=True)


def tables(fa_w, fb_w):
    return fa_w, fb_w, pd.concat([fa_tr, fa_w], ignore_index=True), pd.concat([fb_tr, fb_w], ignore_index=True)


def run(mask_fn, seed):
    fa_w, fb_w = perturb(fa_te, mask_fn(fa_te), seed), perturb(fb_te, mask_fn(fb_te), seed + 1)
    return build(*tables(fa_w, fb_w))


base = build(*tables(fa_te, fb_te))
allf = sorted(set(sub.FEATURE_SETS["H_both"](pr.groups(base.columns))))
hist = [c for c in allf if c.startswith("hist_")]
same_day = [c for c in allf if c not in hist and not c.startswith("nb_")]
print(f"day {D0.date()}: {len(base)} daytime rows, {len(allf)} model features ({len(hist)} previous-day, {len([c for c in allf if c.startswith('nb_')])} neighbour)")
eq = lambda a, b, cols: np.isclose(a[cols].to_numpy("float64"), b[cols].to_numpy("float64"), equal_nan=True, rtol=0, atol=0)

later = run(lambda f: f.ts > D1, 1)
same = eq(base, later, allf)
print(f"1. destroying all forecasts AFTER the day: {(~same).sum()} differing cells of {same.size}")
prev = run(lambda f: (f.ts > D0 - pd.Timedelta(days=1)) & (f.ts <= D0), 3)
ch_hist = (~eq(base, prev, hist)).mean()
ch_same = (~eq(base, prev, same_day)).mean()
print(f"2. destroying the previous day's forecasts: {ch_hist:.3f} of previous-day-feature cells change, {ch_same:.3f} of same-day-feature cells change")
own = run(lambda f: (f.ts > D0) & (f.ts <= D1), 5)
print(f"3. destroying the day's own forecasts: {(~eq(base, own, same_day)).mean():.3f} of same-day-feature cells change")
assert (~same).sum() == 0, "features of a day depend on later forecasts: LEAKAGE"
assert ch_hist > 0.3 and ch_same == 0.0 and (~eq(base, own, same_day)).mean() > 0.5
print("PASS: features of a target day use only that day's and earlier days' forecasts")


# ---- 4. national NWP summaries (NOAA GFS / GEFS / GEFS-Aerosols tables in external_data/)
import os, tempfile
from gfs_features import gfs_context, gefs_context, aerosol_context
ext = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_data")
if all(os.path.exists(os.path.join(ext, n)) for n in ("gfs_national_raw.parquet", "gefs_national_raw.parquet")):
    day = pd.Timestamp("2025-06-06")
    aer_path = os.path.join(ext, "aerosol_national_raw.parquet")
    from gfs_features import ecmwf_context, gfsx_context
    tables = [("gfs_national_raw.parquet", gfs_context), ("gefs_national_raw.parquet", gefs_context)] + ([("aerosol_national_raw.parquet", aerosol_context)] if os.path.exists(aer_path) else []) \
        + ([("ecmwf_national_raw.parquet", ecmwf_context)] if os.path.exists(os.path.join(ext, "ecmwf_national_raw.parquet")) else []) \
        + ([("gfsx_national_raw.parquet", gfsx_context)] if os.path.exists(os.path.join(ext, "gfsx_national_raw.parquet")) else [])      # post-contest (18 Nov 2026): GFS extra fields (V19 / V19p)
    tmp = tempfile.mkdtemp()
    for name, fn in tables:
        raw = pd.read_parquet(os.path.join(ext, name))
        raw["date"] = pd.to_datetime(raw["date"])
        base = fn(os.path.join(ext, name))
        sel = (base.index > K("2025-06-06 00:00")) & (base.index <= K("2025-06-07 00:00"))
        rng = np.random.default_rng(7)
        num = [c for c in raw.columns if c not in ("date", "fh", "a", "b", "step")]
        def perturb(mask):
            r = raw.copy()
            for c in num:
                r.loc[mask, c] = rng.uniform(0, 1000, int(mask.sum()))
            p = os.path.join(tmp, name); r.to_parquet(p, index=False)
            return fn(p)
        others = perturb(raw["date"] != day)
        own = perturb(raw["date"] == day)
        same = np.isclose(base[sel].to_numpy("float64"), others[sel].to_numpy("float64"), equal_nan=True, rtol=0, atol=0)
        chg = (~np.isclose(base[sel].to_numpy("float64"), own[sel].to_numpy("float64"), equal_nan=True, rtol=0, atol=0)).mean()
        print(f"4. {name}: destroying all other days -> {(~same).sum()} differing cells of {same.size}; destroying the day itself -> {chg:.2f} of its cells change")
        assert same.all(), f"{name}: the features of a day depend on other days' rows"
        assert chg > 0.5
    print("PASS: national NWP features of a day depend on that day's own file set only (00 UTC cycle of the day before)")


# ---- 5. regional NWP cells (fixed 1-degree cells, recipe v6): per-day construction, no information from other days
import json, shutil
import regional_features as rf
for src in ("gefs", "gfs", "ecmwf"):
    path = os.path.join(ext, f"nwp_cells_{src}.parquet")
    if not os.path.exists(path):
        continue
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"])
    day = pd.Timestamp("2025-06-06")
    if not (raw["date"] == day).any():
        continue
    fld = "ssrd" if src == "ecmwf" else "dsw"
    rng = np.random.default_rng(11)
    tmp = tempfile.mkdtemp()
    shutil.copy(path.replace(".parquet", "_geo.json"), os.path.join(tmp, f"nwp_cells_{src}_geo.json"))

    def perturbed(mask):
        r = raw.copy()
        for i in np.where(mask.to_numpy())[0]:
            v = r.iloc[i][fld]
            if v is not None:
                r.at[r.index[i], fld] = rng.uniform(0, 1e6 if src == "ecmwf" else 1000, len(v)).astype("float32")
        p = os.path.join(tmp, f"nwp_cells_{src}.parquet"); r.to_parquet(p, index=False)
        return rf.cell_kt(p, src)
    base = rf.cell_kt(path, src)
    sel = (base.index > K("2025-06-06 00:00")) & (base.index <= K("2025-06-07 00:00"))
    others = perturbed(raw["date"] != day)
    own = perturbed(raw["date"] == day)
    same = np.isclose(base[sel].to_numpy("float64"), others[sel].to_numpy("float64"), equal_nan=True, rtol=0, atol=0)
    chg = (~np.isclose(base[sel].to_numpy("float64"), own[sel].to_numpy("float64"), equal_nan=True, rtol=0, atol=0)).mean()
    print(f"5. cells {src}: destroying all other days -> {(~same).sum()} differing cells of {same.size}; destroying the day itself -> {chg:.2f} of its cells change")
    assert same.all(), f"cells {src}: the cell indices of a day depend on other days' rows"
    assert chg > 0.3
print("PASS: regional NWP cell indices of a day depend on that day's own file set only; PCA / fleet weights are fitted on training hours (<= cut) in regional_features.regional_block")
