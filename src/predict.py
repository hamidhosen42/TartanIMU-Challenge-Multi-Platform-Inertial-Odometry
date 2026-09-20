"""Generate a Kaggle submission from one or more checkpoints (checkpoint ensembling = mean of velocities).

  python predict.py --ckpt runs/v1/best.pt [runs/v2/best.pt ...] --out submission.csv [--val]

--val also scores the checkpoint(s) on the val split with the official metric before predicting test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import DATA, read_index, load_split, build_solution, score_predictions
from model import IMUNet, predict_trajectory

p = argparse.ArgumentParser()
p.add_argument("--ckpt", nargs="+", required=True)
p.add_argument("--out", default="submission.csv")
p.add_argument("--stride", type=int, default=2, help="chunk stride in windows (smaller = more overlap averaging)")
p.add_argument("--val", action="store_true")
p.add_argument("--tta-deg", type=float, default=0.0, help="rotation TTA: ±deg about each body axis (0 = off)")
args = p.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

models = []
for c in args.ckpt:
    ck = torch.load(c, map_location="cpu")
    m = IMUNet(width=ck["args"].get("width", 128), ctx_layers=ck["args"].get("ctx_layers", 2), physics=bool(ck["args"].get("physics", 0)), lag=bool(ck["args"].get("lag", 0))).to(device).eval()
    m.load_state_dict(ck["model"]); models.append((m, ck["args"].get("chunk", 16)))
    print(f"loaded {c}: epoch {ck.get('epoch')}, val_score {ck.get('val_score', 'n/a')}")


def tta_rotations(deg):
    """Identity plus ±deg rotations about each body axis (sensor re-mount TTA)."""
    from scipy.spatial.transform import Rotation
    Rs = [np.eye(3)]
    if deg > 0:
        for ax in "xyz":
            for sgn in (1, -1):
                Rs.append(Rotation.from_euler(ax, sgn * deg, degrees=True).as_matrix())
    return [R.astype(np.float32) for R in Rs]


def predict_traj_tta(m, imu, cw):
    vs = []
    for R in tta_rotations(args.tta_deg):
        x = np.concatenate([imu[:, :3] @ R.T, imu[:, 3:] @ R.T], axis=1)       # rotate sensor frame
        vs.append(predict_trajectory(m, x, device, chunk_win=cw, stride_win=args.stride) @ R)  # rotate velocity back
    return np.mean(vs, axis=0)


def predict_split(split):
    idx = read_index(split); trajs = load_split(split, keys=("imu",)); out = []
    for tid, g in idx.groupby("traj_id", sort=False):
        v = np.mean([predict_traj_tta(m, trajs[tid]["imu"], cw) for m, cw in models], axis=0)
        w = g["win_idx"].to_numpy()
        out.append(pd.DataFrame({"window_id": g["window_id"].to_numpy(), "vx": v[w, 0], "vy": v[w, 1], "vz": v[w, 2]}))
    return pd.concat(out, ignore_index=True)


if args.val:
    s, pp = score_predictions(build_solution("val"), predict_split("val"))
    print(f"val TartanIMU score: {s:.4f}\n{pp.round(3)}")

sub = pd.read_csv(DATA / "sample_submission.csv")[["window_id"]].merge(predict_split("test"), on="window_id", how="left")
assert len(sub) == 30644 and not sub.isna().any().any(), "submission incomplete"
sub.to_csv(args.out, index=False)
speed = np.linalg.norm(sub[["vx", "vy", "vz"]].to_numpy(), axis=1)
print(f"wrote {args.out}: {len(sub)} rows; speed median {np.median(speed):.3f}, p99 {np.quantile(speed, .99):.3f}, max {speed.max():.3f}")
