"""Train the unified IMU velocity model and validate with the official TartanIMU scorer.

  python train.py                          # train on train, validate on val (model selection)
  python train.py --splits train,val       # final fit on everything (fixed schedule, no selection)

Checkpoints + logs go to ./runs/<name>/.
"""
from __future__ import annotations

import argparse, json, math, time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import WIN, PLATFORMS, PLAT2ID, read_index, load_split, build_solution, score_predictions
from model import IMUNet, CHUNK_WIN, TOK, predict_trajectory

p = argparse.ArgumentParser()
p.add_argument("--name", default="ctx_tcn_gru")
p.add_argument("--splits", default="train")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--steps", type=int, default=300, help="optimizer steps per epoch")
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=1.5e-3)
p.add_argument("--wd", type=float, default=0.02)
p.add_argument("--chunk", type=int, default=CHUNK_WIN)
p.add_argument("--width", type=int, default=128)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval-every", type=int, default=2)
p.add_argument("--rot-deg", type=float, default=15.0, help="max sensor-mount rotation augmentation")
p.add_argument("--plat-probs", default="0.25,0.25,0.25,0.25", help="sampling probability per platform (car,dog,drone,human)")
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
run = Path("runs") / args.name; run.mkdir(parents=True, exist_ok=True)
print("device", device, "| run", run, "| args", vars(args))

# --------------------------------------------------------------------------- data
T = args.chunk
trajs = []                                                # list of dicts: imu, win_v, dense_v, plat, n_win
for split in args.splits.split(","):
    for tid, d in load_split(split, keys=("imu", "vel_body")).items():
        n = d["n_win"]
        vb = d["vel_body"][: n * WIN]
        trajs.append({"id": tid, "imu": d["imu"][: n * WIN].astype(np.float32), "n_win": n,
                      "win_v": vb.reshape(n, WIN, 3).mean(1).astype(np.float32),
                      "dense_v": vb.reshape(n * TOK, WIN // TOK, 3).mean(1).astype(np.float32),
                      "plat": PLAT2ID[tid.split("_")[0]]})
by_plat = {i: [k for k, t in enumerate(trajs) if t["plat"] == i] for i in range(4)}
plat_w = {i: np.array([trajs[k]["n_win"] for k in ks], float) for i, ks in by_plat.items()}
plat_w = {i: w / w.sum() for i, w in plat_w.items()}
print({PLATFORMS[i]: len(ks) for i, ks in by_plat.items()}, "trajectories;", sum(t["n_win"] for t in trajs), "windows")

rng = np.random.default_rng(args.seed)
plat_probs = np.array([float(x) for x in args.plat_probs.split(",")]); plat_probs /= plat_probs.sum()


def sample_batch(B):
    """Platform-balanced random chunks of T windows (edge-padded + masked if trajectory is shorter)."""
    X = np.empty((B, T * WIN, 6), np.float32); Yw = np.empty((B, T, 3), np.float32)
    Yd = np.empty((B, T * TOK, 3), np.float32); M = np.ones((B, T), np.float32); P = np.empty(B, np.int64)
    for b in range(B):
        pl = rng.choice(4, p=plat_probs)
        t = trajs[rng.choice(by_plat[pl], p=plat_w[pl])]
        n = t["n_win"]
        if n >= T:
            s = rng.integers(0, n - T + 1)
            X[b] = t["imu"][s * WIN:(s + T) * WIN]; Yw[b] = t["win_v"][s:s + T]; Yd[b] = t["dense_v"][s * TOK:(s + T) * TOK]
        else:
            X[b, : n * WIN] = t["imu"]; X[b, n * WIN:] = t["imu"][-1]
            Yw[b, :n] = t["win_v"]; Yw[b, n:] = 0; Yd[b, : n * TOK] = t["dense_v"]; Yd[b, n * TOK:] = 0; M[b, n:] = 0
        P[b] = t["plat"]
    return X, Yw, Yd, M, P


def rand_rotation(B, max_deg):
    """Small random rotations (B,3,3): uniform random axis, angle ~ U(0, max_deg)."""
    axis = torch.randn(B, 3, device=device); axis = axis / axis.norm(dim=1, keepdim=True)
    ang = torch.rand(B, 1, device=device) * math.radians(max_deg)
    K = torch.zeros(B, 3, 3, device=device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -axis[:, 2], axis[:, 1], axis[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -axis[:, 0], -axis[:, 1], axis[:, 0]
    I = torch.eye(3, device=device).expand(B, 3, 3)
    s, c = ang.sin().view(B, 1, 1), ang.cos().view(B, 1, 1)
    return I + s * K + (1 - c) * (K @ K)


def augment(X, Yw, Yd):
    """Physically consistent augmentation on device. X:(B,L,6) Yw:(B,T,3) Yd:(B,N,3)."""
    B = X.shape[0]
    R = rand_rotation(B, args.rot_deg)                                   # re-mount the sensor
    acc, gyr = X[..., :3] @ R.transpose(1, 2), X[..., 3:] @ R.transpose(1, 2)
    Yw, Yd = Yw @ R.transpose(1, 2), Yd @ R.transpose(1, 2)
    acc = acc * (1 + 0.02 * torch.randn(B, 1, 3, device=device)) + 0.15 * torch.randn(B, 1, 3, device=device)
    gyr = gyr * (1 + 0.02 * torch.randn(B, 1, 3, device=device)) + 0.02 * torch.randn(B, 1, 3, device=device)
    acc = acc + 0.05 * torch.randn_like(acc); gyr = gyr + 0.004 * torch.randn_like(gyr)
    return torch.cat([acc, gyr], -1), Yw, Yd


def vhuber(pred, tgt, beta=0.25, mask=None):
    e = torch.linalg.vector_norm(pred - tgt, dim=-1)
    l = torch.where(e < beta, 0.5 * e.square() / beta, e - 0.5 * beta)
    return l.mean() if mask is None else (l * mask).sum() / mask.sum().clamp(min=1)


# --------------------------------------------------------------------------- validation
val_sol, val_trajs = None, None
if "val" not in args.splits.split(","):
    val_sol = build_solution("val")
    val_trajs = load_split("val", keys=("imu",))
    val_idx = read_index("val")


def evaluate(m):
    m.eval()
    preds = []
    for tid, g in val_idx.groupby("traj_id", sort=False):
        v = predict_trajectory(m, val_trajs[tid]["imu"], device, chunk_win=T)
        preds.append(pd.DataFrame({"window_id": g["window_id"].to_numpy(), "vx": v[g.win_idx, 0], "vy": v[g.win_idx, 1], "vz": v[g.win_idx, 2]}))
    m.train()
    return score_predictions(val_sol, pd.concat(preds, ignore_index=True))


# --------------------------------------------------------------------------- train
model = IMUNet(width=args.width).to(device)
ema = deepcopy(model).eval()
for q in ema.parameters():
    q.requires_grad_(False)
print(f"{sum(q.numel() for q in model.parameters())/1e6:.2f}M params")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.99))
total_steps = args.epochs * args.steps
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.08, div_factor=20, final_div_factor=200)
ema_decay = 0.998
log, best = [], float("inf")
t0 = time.time()
for ep in range(1, args.epochs + 1):
    model.train(); tot = {"win": 0., "dense": 0., "drift": 0., "plat": 0.}
    for it in range(args.steps):
        X, Yw, Yd, M, P = sample_batch(args.batch)
        X, Yw, Yd = (torch.from_numpy(a).to(device) for a in (X, Yw, Yd))
        M, P = torch.from_numpy(M).to(device), torch.from_numpy(P).to(device)
        X, Yw, Yd = augment(X, Yw, Yd)
        dense, plat = model(X.transpose(1, 2))
        pw = model.to_windows(dense)
        l_win = vhuber(pw, Yw, mask=M)
        l_dense = vhuber(dense, Yd, mask=M.repeat_interleave(TOK, dim=1))
        cum = torch.cumsum((pw - Yw) * M[..., None], dim=1)                    # integrated body-frame error
        l_drift = (torch.linalg.vector_norm(cum, dim=-1) / torch.sqrt(torch.arange(1, T + 1, device=device))).mean()
        l_plat = F.cross_entropy(plat, P)
        loss = l_win + 0.5 * l_dense + 0.2 * l_drift + 0.05 * l_plat
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step(); sched.step()
        with torch.no_grad():
            d = min(ema_decay, (1 + ep * args.steps + it) / (10 + ep * args.steps + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        for k, v in zip(tot, (l_win, l_dense, l_drift, l_plat)):
            tot[k] += v.item() / args.steps
    rec = {"epoch": ep, "lr": sched.get_last_lr()[0], "min": (time.time() - t0) / 60, **{f"l_{k}": round(v, 4) for k, v in tot.items()}}
    if val_sol is not None and (ep % args.eval_every == 0 or ep == args.epochs):
        s, pp = evaluate(ema)
        rec["val_score"] = round(s, 4); rec.update({f"{k}_score": round(v, 3) for k, v in pp["score"].items()})
        rec.update({f"{k}_ave": round(v, 3) for k, v in pp["ave"].items()}); rec.update({f"{k}_ate": round(v, 3) for k, v in pp["ate20"].items()})
        if s < best:
            best = s; torch.save({"model": ema.state_dict(), "args": vars(args), "val_score": s, "epoch": ep}, run / "best.pt")
    torch.save({"model": ema.state_dict(), "args": vars(args), "epoch": ep}, run / "last.pt")
    log.append(rec); pd.DataFrame(log).to_csv(run / "log.csv", index=False)
    print(json.dumps(rec))
print("best val score", best)
