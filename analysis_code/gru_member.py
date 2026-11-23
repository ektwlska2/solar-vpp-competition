"""Day-sequence network member (post-contest study, 17 Nov 2026): a bidirectional GRU that reads one plant's whole day at once.

Why (organisers' review of the two leading teams): both blended tree models with a network whose training example is ONE PLANT x ONE TARGET DAY, the sequence of
its daytime hours (Team 1: plant/station encoder + 1-D convolution over the hours + bidirectional GRU; Kyle: bidirectional GRU over 24 hours, per plant or per zone).
Our own network (nn_member.py) scores every hour on its own, so the day's context reaches it only through hand-made columns, and it repeats the trees' mistakes.

Architecture, per sample (plant p, day d; T = 17 hour positions, labels 05..21):
    x[t]  (F inputs of hour t, the same columns and normalisation as nn_member, + a 'row present' flag)
      -> hour encoder: Linear(F -> d) + LayerNorm + SiLU + Dropout
      +  plant context: Linear([plant embedding, station-A embedding, station-B embedding, log capacity] -> d), added to every hour
      -> residual 1-D convolution over the hours (kernel 3, padding 1): each hour sees its neighbours
      -> bidirectional GRU (2 layers): each hour's state summarises the whole day, morning -> evening and evening -> morning
      -> head: LayerNorm + Linear + SiLU + Linear -> 1.2 x sigmoid = utilisation forecast of hour t
Loss: weighted L1 per plant-hour with the SMP x k hour weights of the tree members, over hours that exist;
      optional (lam > 0): + lam x weighted L1 of the two GROUP totals per hour (batches are whole days, so every plant of a day is in the same batch).
Unknown plants: during training 10 % of the samples use the 'unknown plant' embedding (index 0), which is what a plant never seen in training receives.
Nothing from the forecast window except its day-ahead inputs is used; the plant/station identities come from sites.csv.

fit_predict(tr, tg, feats, w_tr, ...) -> (utilisation forecast for the rows of tg, info dict)
"""
from __future__ import annotations
import os, time
import numpy as np
import pandas as pd

T0H, T1H = 5, 21                     # hour labels kept by 08_make_submission (DAY_HOURS)
T = T1H - T0H + 1


# ------------------------------------------------------------------------------------------------ inputs
def norm_stats(X: np.ndarray, seed: int = 0):
    """median / sd / 1-99 % range of the TRAINING rows (400k-row sample), as in nn_member."""
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), size=min(400_000, len(X)), replace=False)]
    with np.errstate(all="ignore"):
        med = np.nanmedian(sub, axis=0)
        sd = np.nanstd(sub, axis=0)
        q = np.nanpercentile(sub, [1, 99], axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    med, lo, hi = (np.where(np.isfinite(a), a, 0.0) for a in (med, q[0], q[1]))
    return med, sd, lo, hi


def prep(X: np.ndarray, med, sd, lo, hi, nan_cols) -> np.ndarray:
    Z = (np.clip(X, lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo)) - med) / sd
    miss = np.isnan(X[:, nan_cols]).astype("float32")
    return np.concatenate([np.nan_to_num(Z, nan=0.0), miss], axis=1).astype("float32")


def sequences(df: pd.DataFrame, Z: np.ndarray, site_ix: dict, st_ix: tuple, y=None, w=None):
    """rows (plant-hours) -> samples (plant-days) x T hour positions.  Samples are sorted by day, then plant.
    Returns dict(X [n, T, F+1] float16 incl. the presence flag, row_k / row_t (where each input row sits), day codes, static indices, y/w/m if given)."""
    ts = pd.DatetimeIndex(df.ts)
    if ts.tz is not None:
        ts = ts.tz_convert("Asia/Seoul").tz_localize(None)
    pos = (ts.hour - T0H).to_numpy()
    assert pos.min() >= 0 and pos.max() < T, "hour labels outside 05..21"
    day = ts.normalize()
    key = pd.DataFrame({"site": df.site_id.to_numpy(), "day": day})
    samp = key.drop_duplicates().sort_values(["day", "site"]).reset_index(drop=True)
    samp["k"] = np.arange(len(samp))
    k = key.merge(samp, on=["site", "day"], how="left").k.to_numpy()
    n, F = len(samp), Z.shape[1]
    X = np.zeros((n, T, F + 1), np.float16)
    X[k, pos, :F] = Z
    X[k, pos, F] = 1.0
    st = df[["site_id", "wa_station", "wb_station", "cap"]].drop_duplicates("site_id").set_index("site_id")
    out = dict(X=X, row_k=k, row_t=pos, day=pd.factorize(samp.day, sort=True)[0], days=np.sort(samp.day.unique()),
               site=samp.site.map(lambda s: site_ix.get(s, 0)).to_numpy("int64"),
               wa=samp.site.map(st.wa_station).map(lambda s: st_ix[0].get(s, 0)).to_numpy("int64"),
               wb=samp.site.map(st.wb_station).map(lambda s: st_ix[1].get(s, 0)).to_numpy("int64"),
               lcap=np.log(samp.site.map(st.cap).to_numpy("float32")), cap=samp.site.map(st.cap).to_numpy("float32"), site_id=samp.site.to_numpy())
    if y is not None:
        Y = np.zeros((n, T), np.float32); Wt = np.zeros((n, T), np.float32); M = np.zeros((n, T), np.float32)
        Y[k, pos] = y; Wt[k, pos] = w; M[k, pos] = 1.0
        out.update(y=Y, w=Wt, m=M)
    return out


# ------------------------------------------------------------------------------------------------ model
def make_model(n_in, n_site, n_wa, n_wb, d=128, hid=96, layers=2, drop=0.1, e_site=16, e_st=8):
    import torch
    import torch.nn as nn

    class DayGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, d), nn.LayerNorm(d), nn.SiLU(), nn.Dropout(drop))
            self.e_site, self.e_wa, self.e_wb = nn.Embedding(n_site + 1, e_site), nn.Embedding(n_wa + 1, e_st), nn.Embedding(n_wb + 1, e_st)
            self.stat = nn.Linear(e_site + 2 * e_st + 1, d)
            self.conv = nn.Conv1d(d, d, kernel_size=3, padding=1)
            self.gru = nn.GRU(d, hid, num_layers=layers, batch_first=True, bidirectional=True, dropout=drop if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.LayerNorm(2 * hid), nn.Linear(2 * hid, 64), nn.SiLU(), nn.Linear(64, 1))

        def forward(self, x, site, wa, wb, lcap):
            s = self.stat(torch.cat([self.e_site(site), self.e_wa(wa), self.e_wb(wb), lcap[:, None]], dim=1))
            h = self.inp(x) + s[:, None, :]
            h = h + torch.nn.functional.silu(self.conv(h.transpose(1, 2)).transpose(1, 2))
            o, _ = self.gru(h)
            return 1.2 * torch.sigmoid(self.head(o).squeeze(-1))

    return DayGRU()


# ------------------------------------------------------------------------------------------------ fit / predict
def fit_predict(tr: pd.DataFrame, tg: pd.DataFrame, feats: list, w_tr: np.ndarray, seeds: int = 3, epochs: int = 25, days_per_batch: int = 6,
                lr: float = 2e-3, wd: float = 1e-2, d: int = 128, hid: int = 96, layers: int = 2, drop: float = 0.1, lam: float = 0.0,
                group_of_site: dict | None = None, val_days: int = 0, unk_p: float = 0.1, flag_min: float = 0.005, batch: str = "days", batch_size: int = 1024, log=None):
    """tr needs site_id, ts, cap, wa_station, wb_station, u and `feats`; tg the same without u.  w_tr: hour weights of the training rows (mean 1).
    val_days > 0: the last val_days training days are held out of training and their weighted L1 is logged after every epoch (to choose the epoch count)."""
    import torch
    say = log or (lambda m: None)
    t0 = time.time()
    dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    Xtr, Xte = tr[feats].to_numpy("float32"), tg[feats].to_numpy("float32")
    med, sd, lo, hi = norm_stats(Xtr)
    if os.environ.get("OIBC_FLAGS_FROM_TRAIN", "0") == "1":                                              # post-contest (18 Nov 2026, V19p): training rows only
        nan_cols = np.where(np.isnan(Xtr).mean(0) >= flag_min)[0]
    else:                                                                                                # default (V19): training or target rows
        nan_cols = np.where((np.isnan(Xtr).mean(0) >= flag_min) | (np.isnan(Xte).mean(0) >= flag_min))[0]       # a 0/1 gap flag only where >= flag_min of the rows are missing
    Ztr, Zte = prep(Xtr, med, sd, lo, hi, nan_cols), prep(Xte, med, sd, lo, hi, nan_cols)
    del Xtr, Xte
    sites_tr = np.sort(tr.site_id.unique())
    site_ix = {s: i + 1 for i, s in enumerate(sites_tr)}
    st_ix = tuple({s: i + 1 for i, s in enumerate(np.sort(pd.concat([tr[c], tg[c]]).unique()))} for c in ("wa_station", "wb_station"))
    S = sequences(tr, Ztr, site_ix, st_ix, tr.u.clip(0, 1.2).to_numpy("float32"), np.asarray(w_tr, "float32"))
    G = sequences(tg, Zte, site_ix, st_ix)
    del Ztr, Zte
    n_unseen = int((G["site"] == 0).sum())
    say(f"gru: {len(S['X']):,} training plant-days x {T} hours x {S['X'].shape[2]} inputs ({len(nan_cols)} gap flags), target {len(G['X']):,} plant-days "
        f"({n_unseen:,} of plants never seen in training), device {dev.type}, prepared in {time.time()-t0:.0f}s")
    grp = np.zeros(len(S["X"]), np.int64)
    if lam > 0:
        assert group_of_site is not None
        grp = np.array([group_of_site.get(s, 0) for s in S["site_id"]], np.int64)
    # day ranges (samples are sorted by day)
    nd = int(S["day"].max()) + 1
    starts = np.searchsorted(S["day"], np.arange(nd)); ends = np.searchsorted(S["day"], np.arange(nd), side="right")
    tr_days = np.arange(nd - val_days) if val_days else np.arange(nd)
    va_days = np.arange(nd - val_days, nd) if val_days else np.array([], int)
    to = lambda a: torch.from_numpy(np.array(a, copy=True)).to(dev)
    n = len(S["X"])                                                                       # index n = the dummy sample
    add1 = lambda a: np.concatenate([a, np.zeros((1,) + a.shape[1:], a.dtype)])
    Xd = to(add1(S["X"]))                                                                 # float16 on the device; cast per batch
    Yd, Wd, Md = to(add1(S["y"])), to(add1(S["w"])), to(add1(S["m"]))
    sited, wad, wbd, lcd, capd = (to(add1(S[k])) for k in ("site", "wa", "wb", "lcap", "cap"))
    grpd = to(add1(grp))
    assert batch in ("days", "random") and not (batch == "random" and lam > 0), "the group-total loss needs whole days in a batch"
    B_fix = days_per_batch * int((ends - starts).max()) if batch == "days" else batch_size      # every training batch has exactly this many samples
    tr_samples = np.concatenate([np.arange(starts[q], ends[q]) for q in tr_days])
    CH = 2048                                                                             # fixed chunk size for validation / prediction
    say(f"gru: batches of {days_per_batch} whole days padded to {B_fix} samples" if batch == "days" else f"gru: batches of {B_fix} plant-days drawn at random from all training days")

    def batch_of(days):
        ix = np.concatenate([np.arange(starts[q], ends[q]) for q in days])
        dd = np.concatenate([np.full(ends[q] - starts[q], j) for j, q in enumerate(days)])
        pad = B_fix - len(ix)
        return to(np.r_[ix, np.full(pad, n)].astype("int64")), to(np.r_[dd, np.full(pad, days_per_batch)].astype("int64"))

    def chunks(idx):                                                                      # fixed-size chunks, the last one padded with the dummy sample
        for i in range(0, len(idx), CH):
            c = idx[i:i + CH]
            yield to(np.r_[c, np.full(CH - len(c), n)].astype("int64")), len(c)

    va_np = np.concatenate([np.arange(starts[q], ends[q]) for q in va_days]) if val_days else None
    preds, curves = [], []
    for s in range(seeds):
        torch.manual_seed(s)
        rng = np.random.default_rng(1000 + s)
        net = make_model(S["X"].shape[2], len(sites_tr), len(st_ix[0]), len(st_ix[1]), d, hid, layers, drop).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        n_b = int(np.ceil(len(tr_days) / days_per_batch)) if batch == "days" else int(np.ceil(len(tr_samples) / B_fix))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * n_b, pct_start=0.15)
        curve = []
        for ep in range(epochs):
            net.train()
            perm = rng.permutation(tr_days if batch == "days" else tr_samples)
            tl, tw = 0.0, 0.0
            for b in range(n_b):
                bd = perm[b * days_per_batch:(b + 1) * days_per_batch] if batch == "days" else perm[b * B_fix:(b + 1) * B_fix]
                if batch == "days":
                    ix, dd = batch_of(bd)
                else:
                    ix, dd = to(np.r_[bd, np.full(B_fix - len(bd), n)].astype("int64")), None
                site_b = sited[ix]
                if unk_p > 0:
                    site_b = torch.where(torch.rand(B_fix, device=dev) < unk_p, torch.zeros_like(site_b), site_b)
                out = net(Xd[ix].float(), site_b, wad[ix], wbd[ix], lcd[ix])
                y, w, m = Yd[ix], Wd[ix], Md[ix]
                wm = w * m
                loss = (wm * (out - y).abs()).sum() / wm.sum().clamp(min=1e-6)
                if lam > 0:
                    key = ((dd[:, None] * T + torch.arange(T, device=dev)[None, :]) * 2 + grpd[ix][:, None]).reshape(-1)
                    nk = (days_per_batch + 1) * T * 2
                    cap = capd[ix][:, None]
                    E = torch.zeros(nk, device=dev).index_add_(0, key, (cap * (out - y) * m).reshape(-1))
                    C = torch.zeros(nk, device=dev).index_add_(0, key, (cap * m).reshape(-1))
                    Wk = torch.zeros(nk, device=dev).index_add_(0, key, wm.reshape(-1)) / torch.zeros(nk, device=dev).index_add_(0, key, m.reshape(-1)).clamp(min=1.0)
                    loss = loss + lam * (Wk * E.abs() / C.clamp(min=1.0)).sum() / Wk.sum().clamp(min=1e-6)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sched.step()
                tl += float(loss.detach()) * len(bd); tw += len(bd)
            msg = f"gru seed {s} epoch {ep + 1:2d}/{epochs}: train loss {tl / tw:.5f}"
            if val_days:
                net.eval()
                num, den = 0.0, 0.0
                with torch.no_grad():
                    for ci, k in chunks(va_np):
                        out = net(Xd[ci].float(), sited[ci], wad[ci], wbd[ci], lcd[ci])
                        wm = Wd[ci] * Md[ci]
                        num += float((wm * (out - Yd[ci]).abs()).sum()); den += float(wm.sum())
                vl = num / den
                curve.append(vl)
                msg += f", validation weighted L1 {vl:.5f}"
            say(msg + f"  [{time.time() - t0:.0f}s" + (f", GPU memory {torch.mps.driver_allocated_memory() / 1e9:.1f} GB]" if dev.type == "mps" else "]"))
            if dev.type == "mps":
                torch.mps.empty_cache()
        net.eval()
        ng = len(G["X"])
        Xg, sg, wag, wbg, lcg = (to(add1(G[k])) for k in ("X", "site", "wa", "wb", "lcap"))
        parts = []
        with torch.no_grad():
            for i in range(0, ng, CH):
                c = np.arange(i, min(i + CH, ng)); ci = to(np.r_[c, np.full(CH - len(c), ng)].astype("int64"))
                parts.append(net(Xg[ci].float(), sg[ci], wag[ci], wbg[ci], lcg[ci])[:len(c)].cpu().numpy())
        pg = np.concatenate(parts)
        preds.append(pg[G["row_k"], G["row_t"]])
        curves.append(curve)
        del net, opt, Xg
        if dev.type == "mps":
            torch.mps.empty_cache()
    u = np.clip(np.mean(preds, axis=0), 0.0, 1.0).astype("float32")
    return u, dict(seconds=round(time.time() - t0), n_train_samples=int(len(S["X"])), n_inputs=int(S["X"].shape[2]), n_gap_flags=int(len(nan_cols)),
                   unseen_target_samples=n_unseen, val_curves=curves, settings=dict(seeds=seeds, epochs=epochs, days_per_batch=days_per_batch, lr=lr, wd=wd, d=d,
                   hid=hid, layers=layers, drop=drop, lam=lam, val_days=val_days, unk_p=unk_p, flag_min=flag_min, batch=batch, batch_samples=B_fix))
