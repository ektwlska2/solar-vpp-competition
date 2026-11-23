#!/usr/bin/env python
import argparse, json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get("OIBC_PROJ", os.path.dirname(HERE))                         # the folder that holds analysis_code/
SP = os.environ.get("OIBC_SCRATCH", os.path.expanduser("~/oibc_scratch_backup"))    # scratch folder (frames in, GRU forecasts out)
os.environ.setdefault("OIBC_CACHE", f"{SP}/cache2")
sys.path.insert(0, f"{PROJ}/analysis_code")
from oibc_lib import hour_weights, load_smp_train, load_sites
import gru_member

ap = argparse.ArgumentParser()
ap.add_argument("--window", choices=["pt", "y1", "test"], required=True)       # test = the scoring year (frames of 08 --mode test --dump-frames)
ap.add_argument("--name", required=True)
ap.add_argument("--frames", default=None, help="frames folder: a name under <scratch>/post_contest (default frames_<window>) or an absolute path")
ap.add_argument("--out-dir", default=None, help="output folder (default <scratch>/post_contest)")
ap.add_argument("--feats", default="H_hist_nwp_nn", help="feature set of frames_info.json (the network inputs of V13: G_nwp_nn / H_hist_nwp_nn)")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--val-days", type=int, default=0)
ap.add_argument("--lam", type=float, default=0.0, help="weight of the group-total loss (0 = plant-hour loss only)")
ap.add_argument("--dpb", type=int, default=6, help="days per batch")
ap.add_argument("--batch", choices=["days", "random"], default="days", help="days = whole days per batch (needed for --lam); random = plant-days drawn from all days")
ap.add_argument("--bs", type=int, default=1024, help="plant-days per batch with --batch random")
ap.add_argument("--unk", type=float, default=0.1, help="share of training samples that use the unknown-plant embedding (1.0 = no plant identity at all)")
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--hid", type=int, default=96)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--drop", type=float, default=0.1)
args = ap.parse_args()

T0 = time.time()
say = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
F = args.frames if args.frames and os.path.isabs(args.frames) else f"{SP}/post_contest/{args.frames or 'frames_' + args.window}"
info = json.load(open(f"{F}/frames_info.json"))
feats = info["feature_sets"][args.feats]
base = ["site_id", "ts", "cap", "wa_station", "wb_station"]
tr = pd.read_parquet(f"{F}/train_frame.parquet", columns=list(dict.fromkeys(base + ["u"] + feats)))     # "cap" is both an id column and a feature: read it once
tg = pd.read_parquet(f"{F}/target_frame.parquet", columns=list(dict.fromkeys(base + feats)))
assert "u" not in tg.columns and "energy" not in tg.columns
say(f"window {args.window}: training {len(tr):,} rows to {info['train_end']}, target {len(tg):,} rows ({info['cut']} .. {info['end']}); {len(feats)} features ({args.feats})")
w = hour_weights(pd.DatetimeIndex(tr.ts), load_smp_train()); w = w / w.mean()          # SMP x k weights of the training hours (as the tree members; v6 has no regime weight)
group_of_site = None                                                                        # only the group-total loss (--lam > 0) needs the group of each plant
if args.lam > 0:
    sys.path.insert(0, f"{SP}/oibc_tests"); import score_lib
    sc = load_sites()[["site_id", "site_code"]]
    g = score_lib.structural_group()                                                        # the scorer's composition (structural split)
    group_of_site = {sid: int(g.get(code, "G2") == "G1") for sid, code in zip(sc.site_id, sc.site_code)}
u, gi = gru_member.fit_predict(tr, tg, feats, w, seeds=args.seeds, epochs=args.epochs, days_per_batch=args.dpb, lr=args.lr, d=args.d, hid=args.hid,
                               layers=args.layers, drop=args.drop, lam=args.lam, group_of_site=group_of_site, val_days=args.val_days, batch=args.batch, batch_size=args.bs, unk_p=args.unk, log=say)
col = f"kwh_gru_{args.name}"
out = pd.DataFrame({"site_id": tg.site_id.to_numpy(), "ts": pd.DatetimeIndex(tg.ts).tz_convert("Asia/Seoul"), col: u.astype("float64") * tg.cap.to_numpy("float64")})
OUT = args.out_dir or f"{SP}/post_contest"; os.makedirs(OUT, exist_ok=True)
stem = f"{OUT}/gru_{args.window}_{args.name}"
out.to_parquet(stem + ".parquet", index=False)
json.dump(dict(window=args.window, name=args.name, feats=args.feats, n_feats=len(feats), frames=info["cut"] + " .. " + info["end"], train_end=info["train_end"], **gi,
               built=time.strftime("%Y-%m-%d %H:%M:%S")), open(stem + ".json", "w"), indent=1)
say(f"wrote {stem}.parquet ({len(out):,} rows, column {col}); mean utilisation of the forecast {float(u.mean()):.4f}")
