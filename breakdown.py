"""Val breakdown by platform and drone source (A = racing drone, B = the rest) for one or more checkpoints.
  python breakdown.py runs/v1/best.pt [runs/v3/best.pt ...]
"""
import sys, numpy as np, pandas as pd, torch
from common import load_split, read_index, build_solution
from kaggle_metric import _ate_traj, _ave_traj, AVE_REF, ATE_REF
from model import IMUNet, predict_trajectory
pd.set_option("display.width", 250)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
sol = build_solution("val"); idx = read_index("val"); trajs = load_split("val", keys=("imu",))
for ckpt in sys.argv[1:]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False); a = ck["args"]
    m = IMUNet(width=a.get("width", 128), ctx_layers=a.get("ctx_layers", 2), physics=bool(a.get("physics", 0)), lag=bool(a.get("lag", 0))).to(device).eval(); m.load_state_dict(ck["model"])
    preds = []
    for tid, g in idx.groupby("traj_id", sort=False):
        v = predict_trajectory(m, trajs[tid]["imu"], device, chunk_win=a.get("chunk", 16), stride_win=2)
        w = g["win_idx"].to_numpy(); preds.append(pd.DataFrame({"window_id": g["window_id"].to_numpy(), "vx_pred": v[w, 0], "vy_pred": v[w, 1], "vz_pred": v[w, 2]}))
    mm = sol.merge(pd.concat(preds), on="window_id")
    recs = []
    for tid, g in mm.groupby("traj_id", sort=False):
        g = g.sort_values("win_idx"); p = g["platform"].iloc[0]
        grp = p if p != "drone" else ("drone-A" if int(tid[-4:]) <= 8 else "drone-B")
        recs.append({"platform": p, "group": grp, "ave": _ave_traj(g), "ate20": _ate_traj(g)})
    r = pd.DataFrame(recs)
    pp = r.groupby("platform")[["ave", "ate20"]].mean(); score = 0.6 * pp["ave"].mean() / AVE_REF + 0.4 * pp["ate20"].mean() / ATE_REF
    gg = r.groupby("group")[["ave", "ate20"]].mean(); gg["n_traj"] = r.groupby("group").size()
    print(f"\n== {ckpt} (epoch {ck.get('epoch')}): val score {score:.4f}")
    print(gg.round(3).to_string())
