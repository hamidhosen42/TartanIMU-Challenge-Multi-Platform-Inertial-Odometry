"""Exploratory analysis of the TartanIMU challenge data.

Writes figures + a markdown summary to ./analysis/.  Run:  python analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATA, WIN, PLATFORMS, read_index, traj_path, load_split

OUT = DATA.parent / "analysis"
OUT.mkdir(exist_ok=True)
COLORS = {"car": "#2a78d6", "dog": "#eb6834", "drone": "#1baf7a", "human": "#4a3aa7"}
CH = ["ax", "ay", "az", "gx", "gy", "gz"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})
lines: list[str] = ["# TartanIMU challenge — data analysis\n"]


def md(s=""):
    lines.append(s)


# ----------------------------------------------------------------------------- 1. inventory
tr, va, te = read_index("train"), read_index("val"), read_index("test")
md("## 1. Inventory\n")
inv = []
for name, df in [("train", tr), ("val", va)]:
    g = df.groupby("platform")
    inv.append(pd.DataFrame({"split": name, "trajectories": g["traj_id"].nunique(), "windows": g.size(),
                             "hours": g.size() / 3600,
                             "median_traj_s": g.apply(lambda x: x.groupby("traj_id").size().median()),
                             "min_traj_s": g.apply(lambda x: x.groupby("traj_id").size().min()),
                             "max_traj_s": g.apply(lambda x: x.groupby("traj_id").size().max())}))
inv = pd.concat(inv).reset_index()
md(inv.round(2).to_markdown(index=False))
tl = te.groupby("traj_id").size()
md(f"\nTest: {te.traj_id.nunique()} trajectories, {len(te)} windows ({len(te)/3600:.2f} h); "
   f"trajectory length s: min {tl.min()}, median {tl.median():.0f}, max {tl.max()}. No platform labels.\n")
md("Windows are contiguous, non-overlapping 1 s slices of a trajectory, and test trajectories are given whole "
   "→ a model may legitimately use temporal context around each window (past *and* future).\n")

# ----------------------------------------------------------------------------- 2. targets
md("## 2. Targets (body-frame velocity)\n")
lab = pd.concat([tr.assign(split="train"), va.assign(split="val")])
lab["speed"] = np.linalg.norm(lab[["vx", "vy", "vz"]].to_numpy(), axis=1)
q = lambda p: (lambda s: s.quantile(p))
ts = lab.groupby(["split", "platform"]).agg(
    vx_mean=("vx", "mean"), vy_mean=("vy", "mean"), vz_mean=("vz", "mean"),
    vx_std=("vx", "std"), vy_std=("vy", "std"), vz_std=("vz", "std"),
    speed_med=("speed", "median"), speed_p95=("speed", q(.95)), speed_max=("speed", "max"),
    still_frac=("speed", lambda s: (s < 0.1).mean())).reset_index()
md(ts.round(3).to_markdown(index=False))
md("\n* car: dominantly forward (+vx) motion, tiny vy/vz → strong 'non-holonomic' prior.\n"
   "* dog: forward-biased but with substantial lateral/vertical components (gait).\n"
   "* drone: fastest and fully 3-D; largest speed spread → dominates the all-zero error.\n"
   "* human: slow handheld motion, most 'still' windows; sensor orientation is arbitrary.\n")

fig, axes = plt.subplots(1, 4, figsize=(13, 3))
for ax, p in zip(axes, PLATFORMS):
    for split, ls in [("train", "-"), ("val", "--")]:
        s = np.sort(lab[(lab.platform == p) & (lab.split == split)].speed.to_numpy())
        ax.plot(s, np.linspace(0, 1, len(s)), ls, color=COLORS[p], lw=2, label=split)
    ax.set_title(p); ax.set_xlabel("speed (m/s)"); ax.set_xlim(0, np.quantile(lab[lab.platform == p].speed, .995))
axes[0].set_ylabel("ECDF"); axes[0].legend(frameon=False)
fig.suptitle("Body-frame speed distribution per platform (train solid, val dashed)")
fig.tight_layout(); fig.savefig(OUT / "speed_ecdf.png"); plt.close(fig)

# ----------------------------------------------------------------------------- 3. IMU statistics
md("## 3. IMU channel statistics (per platform, train)\n")
train = load_split("train", keys=("imu", "vel_body", "quat"))
stats = []
for p in PLATFORMS:
    imu = np.concatenate([d["imu"][: d["n_win"] * WIN] for t, d in train.items() if t.startswith(p)])
    row = {"platform": p}
    for i, c in enumerate(CH):
        row[f"{c}_mean"] = imu[:, i].mean(); row[f"{c}_std"] = imu[:, i].std()
    row["|a|_mean"] = np.linalg.norm(imu[:, :3], axis=1).mean()
    stats.append(row)
stats = pd.DataFrame(stats)
md(stats.round(3).to_markdown(index=False))
md("\nMean accelerometer vector ≈ gravity direction in the body frame: +z for car/dog/drone (upright mounts), "
   "but tilted/-y for human (handheld, arbitrary). Gyro/accel std differ by an order of magnitude between "
   "platforms → per-trajectory vibration signature is a strong platform cue the network can learn implicitly.\n")

# spectra
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
freqs = np.fft.rfftfreq(WIN, 1 / 200)
for p in PLATFORMS:
    ids = [t for t in train if t.startswith(p)]
    rng = np.random.default_rng(0)
    wins = []
    for t in rng.choice(ids, size=min(20, len(ids)), replace=False):
        imu = train[t]["imu"][: train[t]["n_win"] * WIN].reshape(-1, WIN, 6)
        wins.append(imu[rng.choice(len(imu), size=min(30, len(imu)), replace=False)])
    w = np.concatenate(wins); w = w - w.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(w, axis=1)) ** 2 / WIN
    for ax, sl, name in [(axes[0], slice(0, 3), "accelerometer"), (axes[1], slice(3, 6), "gyroscope")]:
        ax.semilogy(freqs[1:], np.median(P[:, :, sl].mean(2), 0)[1:], color=COLORS[p], lw=2, label=p)
        ax.set_title(f"median {name} power spectrum"); ax.set_xlabel("Hz")
axes[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "spectra.png"); plt.close(fig)

# ----------------------------------------------------------------------------- 4. temporal structure of the target
md("## 4. Temporal structure — how much does context help?\n")
md("Autocorrelation of the per-window velocity between windows k and k+lag (train). High values mean the "
   "velocity changes slowly relative to the 1 s window, so neighbouring windows carry information about "
   "the current one.\n")
ac = []
for p in PLATFORMS:
    row = {"platform": p}
    for lag in [1, 2, 5, 10]:
        num = den = 0.0
        for t, d in train.items():
            if not t.startswith(p):
                continue
            v = d["vel_body"][: d["n_win"] * WIN].reshape(-1, WIN, 3).mean(1)
            if len(v) <= lag:
                continue
            a, b = v[:-lag] - v.mean(0), v[lag:] - v.mean(0)
            num += (a * b).sum(); den += (v - v.mean(0)).pow(2).sum() if hasattr(v, "pow") else ((v - v.mean(0)) ** 2).sum()
        row[f"lag{lag}"] = num / den
    ac.append(row)
md(pd.DataFrame(ac).round(3).to_markdown(index=False))

# Quantify the "1 s window loses sub-second curvature" floor and how a smoothed prediction would do.
md("\nSimple predictability baselines (val, per-window mean Euclidean error, m/s):\n")
val = load_split("val", keys=("imu", "vel_body"))
rows = []
for p in PLATFORMS:
    errs = {"platform": p, "zero": [], "platform_mean": [], "prev_window": [], "neighbour_avg": []}
    mean_v = lab[(lab.split == "train") & (lab.platform == p)][["vx", "vy", "vz"]].mean().to_numpy()
    for t, d in val.items():
        if not t.startswith(p):
            continue
        v = d["vel_body"][: d["n_win"] * WIN].reshape(-1, WIN, 3).mean(1)
        errs["zero"].append(np.linalg.norm(v, axis=1).mean())
        errs["platform_mean"].append(np.linalg.norm(v - mean_v, axis=1).mean())
        if len(v) > 2:
            errs["prev_window"].append(np.linalg.norm(v[1:] - v[:-1], axis=1).mean())
            errs["neighbour_avg"].append(np.linalg.norm(v[1:-1] - 0.5 * (v[:-2] + v[2:]), axis=1).mean())
    rows.append({k: (np.mean(x) if isinstance(x, list) else x) for k, x in errs.items()})
md(pd.DataFrame(rows).round(3).to_markdown(index=False))
md("\n`prev_window` = oracle that copies the previous window's true velocity; `neighbour_avg` = oracle average of "
   "both neighbours. Errors are far below the all-zero level for every platform → the target is strongly "
   "temporally coherent, and a model that sees several seconds of context should beat a per-window model.\n")

# ----------------------------------------------------------------------------- 5. gravity / orientation view
md("## 5. Gravity leakage into the body frame\n")
md("Body-frame velocity is coupled to orientation: when the sensor tilts, gravity moves between accelerometer "
   "axes. Per-window mean accelerometer vs. mean velocity (train, 3000 random windows per platform):\n")
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
rng = np.random.default_rng(1)
for ax, p in zip(axes, PLATFORMS):
    ids = [t for t in train if t.startswith(p)]
    A, V = [], []
    for t in ids:
        d = train[t]; n = d["n_win"]
        A.append(d["imu"][: n * WIN].reshape(n, WIN, 6)[:, :, :3].mean(1)); V.append(d["vel_body"][: n * WIN].reshape(n, WIN, 3).mean(1))
    A, V = np.concatenate(A), np.concatenate(V)
    sel = rng.choice(len(A), size=min(3000, len(A)), replace=False)
    ax.scatter(A[sel, 0], V[sel, 0], s=4, alpha=.35, color=COLORS[p], edgecolors="none")
    ax.set_title(p); ax.set_xlabel("mean a_x in window (m/s²)"); ax.set_ylabel("v_x (m/s)")
fig.suptitle("Forward velocity vs mean forward acceleration (pitch → gravity leakage)")
fig.tight_layout(); fig.savefig(OUT / "accel_vs_vel.png"); plt.close(fig)

# ----------------------------------------------------------------------------- 6. what does the test set look like?
md("## 6. Test-set composition (platform is hidden)\n")
md("Per-trajectory IMU summary features (channel means/stds, |a| stats, gyro energy) are enough to separate the "
   "platforms almost perfectly on val. This is *analysis only* — the rules forbid routing to per-platform "
   "models — but it tells us the test set's likely composition and that a unified network can infer the "
   "embodiment from the signal itself.\n")


def traj_feats(imu):
    a, g = imu[:, :3], imu[:, 3:]
    an, gn = np.linalg.norm(a, axis=1), np.linalg.norm(g, axis=1)
    f = np.concatenate([imu.mean(0), imu.std(0), np.quantile(an, [.05, .5, .95]), np.quantile(gn, [.05, .5, .95]),
                        [np.abs(np.diff(a, axis=0)).mean(), np.abs(np.diff(g, axis=0)).mean()]])
    return f


from sklearn.ensemble import RandomForestClassifier
Xtr = np.stack([traj_feats(d["imu"]) for d in train.values()]); ytr = [t.split("_")[0] for t in train]
Xva = np.stack([traj_feats(d["imu"]) for d in val.values()]); yva = [t.split("_")[0] for t in val]
clf = RandomForestClassifier(500, random_state=0).fit(Xtr, ytr)
acc = (clf.predict(Xva) == np.array(yva)).mean()
test = load_split("test", keys=("imu",))
Xte = np.stack([traj_feats(d["imu"]) for d in test.values()])
pte = clf.predict(Xte)
comp = pd.DataFrame({"traj_id": list(test), "pred_platform": pte, "windows": [d["n_win"] for d in test.values()]})
summ = comp.groupby("pred_platform").agg(trajectories=("traj_id", "size"), windows=("windows", "sum"))
md(f"Trajectory-level platform classifier: val accuracy = {acc:.3f} ({len(yva)} val trajectories).\n")
md(summ.to_markdown())
comp.to_csv(OUT / "test_platform_guess.csv", index=False)
md("\nThe test split is roughly platform-balanced by trajectory count, so — together with the macro-averaged "
   "metric — every platform matters equally; drone (largest all-zero error) and car (largest ATE) offer the most "
   "score headroom.\n")

# ----------------------------------------------------------------------------- 7. train/val shift
md("## 7. Train → val shift\n")
md("Val speed distributions (dashed in `speed_ecdf.png`) differ visibly from train for dog and drone; and the "
   "val split's drone trajectories are fewer/shorter. Expect val ↔ public-LB gaps of a few hundredths.\n")

# ----------------------------------------------------------------------------- 8. sample trajectory
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, p in zip(axes, PLATFORMS):
    t = next(t for t in train if t.startswith(p))
    with np.load(traj_path("train", t)) as d:
        pos = d["pos"]
    ax.plot(pos[:, 0], pos[:, 1], color=COLORS[p], lw=1.2); ax.set_aspect("equal"); ax.set_title(f"{t} ({len(pos)/200:.0f} s)")
fig.suptitle("Ground-truth XY paths of the first trajectory of each platform"); fig.tight_layout()
fig.savefig(OUT / "paths.png"); plt.close(fig)

# ----------------------------------------------------------------------------- 9. takeaways
md("## 9. Modelling implications\n")
md("""1. **Use context.** Windows come in whole trajectories; velocity is highly autocorrelated. Run a sequence
   model over long chunks (e.g. 16 s) and predict all windows at once, with overlap-averaging at inference.
2. **Unified model with implicit platform inference.** Vibration/gravity signatures identify the platform; an
   auxiliary platform-classification head makes the shared features embodiment-aware without routing.
3. **Macro-averaged metric → platform-balanced sampling** during training, not window-proportional.
4. **ATE20 (40 %) punishes correlated bias**, so add a loss on integrated velocity over multi-window segments,
   not only per-window error.
5. **Physically consistent augmentation**: small random sensor-mount rotations (rotate IMU *and* target
   velocity), gyro/accel bias, scale and noise. Do not mirror/time-reverse (breaks the forward prior).
6. Gravity is retained in the accelerometer, so orientation changes leak into the signal — the network can use
   it as a tilt cue; do not remove the mean acceleration.
""")
(OUT / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
