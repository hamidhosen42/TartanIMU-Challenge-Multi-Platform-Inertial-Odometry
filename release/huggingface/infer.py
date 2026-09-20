"""TartanIMU Challenge — inference entry point (single unified model, one weight set).

    python infer.py --traj_dir <dir with test_*.npz> --windows <index/test_windows.csv> --out submission.csv

Reads only `imu` (N,6) = [ax, ay, az, gx, gy, gz] from each trajectory .npz plus the window order from the CSV.
No platform label, pose, timestamps or other metadata are used. Each trajectory is processed independently.
CUDA if available, otherwise CPU (about 8 min for the 89-trajectory test set on 8 CPU cores; <1 min on a GPU).
"""
from __future__ import annotations

import argparse, hashlib, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import IMUNet, predict_trajectory

p = argparse.ArgumentParser()
p.add_argument("--traj_dir", required=True, help="directory containing <traj_id>.npz files (scanned recursively)")
p.add_argument("--windows", required=True, help="CSV with columns window_id, traj_id, win_idx")
p.add_argument("--out", default="submission.csv")
p.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent / "model.pt"))
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--stride", type=int, default=2, help="chunk stride in windows for overlap averaging (2 = submitted setting)")
args = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # determinism across GPUs
device = torch.device(args.device)
ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
a = ck["args"]
model = IMUNet(width=a.get("width", 128), ctx_layers=a.get("ctx_layers", 2), physics=bool(a.get("physics", 0)), lag=bool(a.get("lag", 0))).to(device).eval()
model.load_state_dict(ck["model"])
print(f"checkpoint {args.checkpoint}: {sum(q.numel() for q in model.parameters())/1e6:.2f}M params, sha256 {hashlib.sha256(open(args.checkpoint,'rb').read()).hexdigest()[:16]}…, device {device}")

idx = pd.read_csv(args.windows)
files = {f.stem: f for f in Path(args.traj_dir).rglob("*.npz")}
t0, out = time.time(), []
for tid, g in idx.groupby("traj_id", sort=False):
    tid = str(tid).replace(".npz", "")
    with np.load(files[tid]) as d:
        imu = d["imu"].astype(np.float32)
    v = predict_trajectory(model, imu, device, chunk_win=a.get("chunk", 16), stride_win=args.stride)
    w = g["win_idx"].to_numpy()
    out.append(pd.DataFrame({"window_id": g["window_id"].to_numpy(), "vx": v[w, 0], "vy": v[w, 1], "vz": v[w, 2]}))
sub = pd.concat(out, ignore_index=True).sort_values("window_id")
assert len(sub) == len(idx) and np.isfinite(sub[["vx", "vy", "vz"]].to_numpy()).all()
sub.to_csv(args.out, index=False)
print(f"wrote {args.out}: {len(sub)} windows, {idx.traj_id.nunique()} trajectories in {time.time()-t0:.0f}s; md5 {hashlib.md5(open(args.out,'rb').read()).hexdigest()}")
