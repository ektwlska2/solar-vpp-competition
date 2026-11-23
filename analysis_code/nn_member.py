"""Neural-network member of the blend: a multi-layer perceptron on the same inputs as the gradient-boosting members.

Why: a different model family makes partly independent errors, and the score penalises absolute error, so a convex blend of a tree model and a network beats both
(hold-out year 2024-06 -> 2025-05: network alone +3% penalty, blend 50/50 -4% against the tree model alone; see the code README).

fit_predict(Xtr, ytr, wtr, Xte, ...) -> utilisation forecast for the rows of Xte (mean of `seeds` networks, clipped to [0, 1.2]).
  * inputs: the training-set median / standard deviation (from a 400k-row sample) after clipping to the 1-99 % range +- 50 %, missing values -> 0 plus one 0/1 flag per column that has gaps
  * loss: weighted L1 (the same SMP x k hour weights as the tree models), output 1.2 * sigmoid(.)
  * 3 hidden layers (512 / 256 / 128, SiLU, dropout 0.05), AdamW lr 2e-3 with one-cycle schedule, weight decay 0.01, batch 4096, 10 epochs, seeds 0..n-1
Everything is computed from the training rows; no coordinates, no plant IDs, nothing from the scoring year except its day-ahead inputs.
"""
from __future__ import annotations
import os, time
import numpy as np


def _prep(X, med, sd, lo, hi, nan_cols):
    Z = (np.clip(X, lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo)) - med) / sd
    miss = np.isnan(X[:, nan_cols]).astype("float32")
    return np.concatenate([np.nan_to_num(Z, nan=0.0), miss], axis=1).astype("float32")


def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, wtr: np.ndarray, Xte: np.ndarray, seeds: int = 5, epochs: int = 10, width: int = 512, drop: float = 0.05,
                lr: float = 2e-3, wd: float = 1e-2, batch: int = 4096, log=None, groups: dict | None = None, lam: float = 2.0, hb: int = 24) -> np.ndarray:
    """groups (optional) = dict(ts=int array, grp=0/1 array, cap=kW array), aligned with the rows of Xtr: the network is then trained on the contest's own penalty as well: batches are made of
    whole hours (hb timestamps), and next to the per-plant weighted L1 the loss contains lam x the weighted L1 error of each GROUP's total (kW, relative to the group's reporting
    capacity), hour by hour, for the two groups of the submitted composition.  Hold-out (Jan-May 2025): network alone 68.8M -> 66.6M, blend with the tree model -3.6% -> -6.1%."""
    import torch
    import torch.nn as nn
    t0 = time.time()
    say = log or (lambda m: None)
    Xtr, Xte = Xtr.astype("float32"), Xte.astype("float32")
    rng = np.random.default_rng(0)
    sub = Xtr[rng.choice(len(Xtr), size=min(400_000, len(Xtr)), replace=False)]
    med = np.nanmedian(sub, axis=0)
    sd = np.nanstd(sub, axis=0); sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    q = np.nanpercentile(sub, [1, 99], axis=0)
    lo, hi = q[0], q[1]
    if os.environ.get("OIBC_FLAGS_FROM_TRAIN", "0") == "1":                  # post-contest (18 Nov 2026, V19p): gap flags chosen on the training rows only
        nan_cols = np.where(np.isnan(Xtr).any(0))[0]
    else:                                                                    # default (V6-V19): training or target rows
        nan_cols = np.where(np.isnan(Xtr).any(0) | np.isnan(Xte).any(0))[0]
    Ztr, Zte = _prep(Xtr, med, sd, lo, hi, nan_cols), _prep(Xte, med, sd, lo, hi, nan_cols)
    dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    Zt = torch.from_numpy(Ztr).to(dev)
    yt = torch.from_numpy(np.asarray(ytr, dtype="float32")).to(dev)
    wt = torch.from_numpy(np.asarray(wtr / np.mean(wtr), dtype="float32")).to(dev)
    Zv = torch.from_numpy(Zte).to(dev)
    say(f"nn: {Ztr.shape[0]:,} rows x {Ztr.shape[1]} inputs ({len(nan_cols)} with gaps), device {dev.type}, prepared in {time.time()-t0:.0f}s")
    n = len(Zt)
    if groups is not None:                                                            # rows sorted by hour: a batch = whole hours
        order = np.argsort(np.asarray(groups["ts"]), kind="stable")
        o_t = torch.from_numpy(order).to(dev)
        Zt, yt, wt = Zt[o_t], yt[o_t], wt[o_t]
        ts_s = np.asarray(groups["ts"])[order]
        Gt = torch.from_numpy(np.asarray(groups["grp"], dtype="int64")[order]).to(dev)
        capt = torch.from_numpy(np.asarray(groups["cap"], dtype="float32")[order]).to(dev)
        bounds = np.r_[0, np.where(np.diff(ts_s) != 0)[0] + 1, len(ts_s)]
        n_ts = len(bounds) - 1
        w_ts = torch.from_numpy(wt.detach().cpu().numpy()[bounds[:-1]]).to(dev)
        say(f"nn: group-level loss on {n_ts:,} hours (lam {lam}, {hb} hours per batch)")
    preds = []
    for s in range(seeds):
        off = int(os.environ.get("OIBC_SEED_OFFSET", "0"))         # post-contest noise control (18 Nov 2026); 0 = the original seeds
        torch.manual_seed(s + off)
        h = width
        net = nn.Sequential(nn.Linear(Zt.shape[1], h), nn.SiLU(), nn.Dropout(drop), nn.Linear(h, h // 2), nn.SiLU(), nn.Dropout(drop),
                            nn.Linear(h // 2, h // 4), nn.SiLU(), nn.Linear(h // 4, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        steps = epochs * ((n_ts // hb) if groups is not None else (n // batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
        for ep in range(epochs):
            net.train()
            if groups is None:
                perm = torch.randperm(n, device=dev)
                for i in range(0, n - batch + 1, batch):
                    idx = perm[i:i + batch]
                    out = 1.2 * torch.sigmoid(net(Zt[idx]).squeeze(1))
                    loss = (wt[idx] * (out - yt[idx]).abs()).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            else:
                perm = np.random.default_rng((s + off) * 1000 + ep).permutation(n_ts)
                for i in range(0, n_ts - hb + 1, hb):
                    sel = perm[i:i + hb]
                    idx = np.concatenate([np.arange(bounds[t], bounds[t + 1]) for t in sel])
                    seg = np.concatenate([np.full(bounds[t + 1] - bounds[t], k) for k, t in enumerate(sel)])
                    idx_t, seg_t = torch.from_numpy(idx).to(dev), torch.from_numpy(seg).to(dev)
                    out = 1.2 * torch.sigmoid(net(Zt[idx_t]).squeeze(1))
                    site_loss = (wt[idx_t] * (out - yt[idx_t]).abs()).mean()
                    e = capt[idx_t] * (out - yt[idx_t])
                    key = seg_t * 2 + Gt[idx_t]
                    E = torch.zeros(hb * 2, device=dev).index_add_(0, key, e)
                    C = torch.zeros(hb * 2, device=dev).index_add_(0, key, capt[idx_t]).clamp(min=1.0)
                    wts = w_ts[torch.from_numpy(sel).to(dev)].repeat_interleave(2)
                    loss = site_loss + lam * (wts * (E.abs() / C)).sum() / (wts.sum() + 1e-9)
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([1.2 * torch.sigmoid(net(Zv[i:i + 65536]).squeeze(1)) for i in range(0, len(Zv), 65536)]).cpu().numpy()
        preds.append(pv)
        say(f"nn seed {s}: done at {time.time()-t0:.0f}s")
        del net, opt
    return np.clip(np.mean(preds, axis=0), 0.0, 1.2).astype("float32")
