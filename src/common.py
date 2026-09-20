"""Shared data utilities for the TartanIMU challenge.

Data conventions (from the official starter kit):
  imu (N,6) = [ax, ay, az, gx, gy, gz]  (m/s^2, rad/s), 200 Hz, body frame, gravity retained
  window k of a trajectory = imu[k*200:(k+1)*200]; target = mean vel_body over that window
  platform_id: car=0, dog=1, drone=2, human=3
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # project root
DATA = ROOT / "data"
WIN = 200
PLATFORMS = ["car", "dog", "drone", "human"]
PLAT2ID = {p: i for i, p in enumerate(PLATFORMS)}


def read_index(split: str) -> pd.DataFrame:
    """window index (+ targets for train/val), one row per window, sorted by traj/win_idx."""
    df = pd.read_csv(DATA / "index" / f"{split}_windows.csv").dropna(axis=1, how="all")
    if split != "test":
        df = df.merge(pd.read_csv(DATA / "index" / f"{split}_targets.csv"), on="window_id")
    return df.sort_values(["traj_id", "win_idx"]).reset_index(drop=True)


def traj_path(split: str, traj_id: str) -> Path:
    if split == "test":
        return DATA / "test" / f"{traj_id}.npz"
    return DATA / split / traj_id.split("_")[0] / f"{traj_id}.npz"


def load_split(split: str, keys=("imu",)) -> dict[str, dict[str, np.ndarray]]:
    """Load every trajectory of a split into memory: {traj_id: {key: array}}."""
    idx = read_index(split)
    out = {}
    for tid in idx["traj_id"].unique():
        with np.load(traj_path(split, tid)) as d:
            out[tid] = {k: d[k] for k in keys if k in d.files}
            out[tid]["n_win"] = len(d["ts"]) // WIN
    return out


def build_solution(split: str) -> pd.DataFrame:
    """Build the scorer's `solution` frame for a labelled split (val/train).

    Per window: quaternion at the window's mid sample, ground-truth position at the window's
    last sample (the integrated path point), dt = window duration.
    """
    idx = read_index(split)
    rows = []
    for tid, g in idx.groupby("traj_id", sort=False):
        with np.load(traj_path(split, tid)) as d:
            quat, pos, ts, fs = d["quat"], d["pos"], d["ts"], float(d["fs"])
        w = g["win_idx"].to_numpy()
        s, e, m = w * WIN, w * WIN + WIN - 1, w * WIN + WIN // 2
        rows.append(pd.DataFrame({
            "window_id": g["window_id"].to_numpy(), "traj_id": tid, "win_idx": w,
            "platform": g["platform"].to_numpy(),
            "qx": quat[m, 0], "qy": quat[m, 1], "qz": quat[m, 2], "qw": quat[m, 3],
            "gx": pos[e, 0], "gy": pos[e, 1], "gz": pos[e, 2],
            "dt": ts[e] - ts[s] + 1.0 / fs,
            "vx_gt": g["vx"].to_numpy(), "vy_gt": g["vy"].to_numpy(), "vz_gt": g["vz"].to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def score_predictions(solution: pd.DataFrame, pred: pd.DataFrame, per_platform: bool = True):
    """Official TartanIMU score + per-platform AVE / ATE20 breakdown."""
    from kaggle_metric import score, _ate_traj, _ave_traj, AVE_REF, ATE_REF

    total = score(solution, pred, "window_id")
    if not per_platform:
        return total, None
    m = solution.merge(pred.rename(columns={"vx": "vx_pred", "vy": "vy_pred", "vz": "vz_pred"}), on="window_id")
    recs = []
    for tid, g in m.groupby("traj_id", sort=False):
        g = g.sort_values("win_idx")
        recs.append({"platform": g["platform"].iloc[0], "traj_id": tid, "ate20": _ate_traj(g), "ave": _ave_traj(g)})
    pp = pd.DataFrame(recs).groupby("platform")[["ave", "ate20"]].mean()
    pp["score"] = 0.6 * pp["ave"] / AVE_REF + 0.4 * pp["ate20"] / ATE_REF
    return total, pp
