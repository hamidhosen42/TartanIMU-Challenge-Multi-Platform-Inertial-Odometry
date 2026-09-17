# TartanIMU Challenge — Multi-Platform Inertial Odometry

Predict the mean 3-D **body-frame velocity** of every 1 s window (200 Hz, 6-axis IMU) with **one model** shared
across car / dog / drone / human. Score = `0.6·AVE/0.7356 + 0.4·ATE20/3.116`, macro-averaged over platforms
(lower is better; all-zero = 1.0). Competition: https://www.kaggle.com/competitions/tartan-imu-challenge-iros2026

## Layout

| File | Purpose |
| --- | --- |
| `data/` | competition data (`kaggle competitions download -c tartan-imu-challenge-iros2026`) |
| `common.py` | data loading, window index, val "solution" builder, official-metric wrapper |
| `kaggle_metric.py` | the organisers' exact leaderboard scorer (from the TartanIMU starter kit) |
| `analysis.py` → `analysis/` | EDA: inventory, target/IMU statistics, temporal autocorrelation, test composition, figures, `summary.md` |
| `model.py` | `IMUNet` (conv stem → dilated TCN → transformer context → dense 20 Hz velocity) + sliding-chunk trajectory inference |
| `train.py` | platform-balanced chunk training with physical augmentation, EMA weights, official-metric validation |
| `predict.py` | checkpoint (ensemble) → `submission.csv`, optional val self-scoring |
| `checkpoints/full_long_swa.pt`, `configs/full_long.json` | ranked checkpoint (SHA-256 in `REPORT.md`) and its exact config |
| `REPORT.md`, `requirements.txt` | organisers' team report + environment |
| `notebooks/tartanimu_v1_submission.ipynb` | **self-contained notebook** (Kaggle/local) reproducing v1 → `submission_v1.csv` |

## Approach

1. **Context, not isolated windows.** Test trajectories are given whole and windows are contiguous, so the model
   reads 16 s chunks (past *and* future) and predicts every window in the chunk; at inference chunks slide with
   overlap and are Hann-weighted-averaged. (Velocity lag-1 autocorrelation is 0.85 car / 0.78 dog / 0.56 human.)
2. **Unified network, implicit embodiment.** No platform input; an auxiliary platform-classification head makes the
   shared features embodiment-aware (the vibration/gravity signature identifies the platform with ~100 % accuracy).
3. **Metric-aware training.** Platform-balanced sampling (macro-average), vector-Huber window loss, dense 20 Hz
   loss, and an *integrated-error* ("drift") loss that mirrors ATE20's sensitivity to correlated bias.
4. **Physically consistent augmentation**: small random sensor-mount rotations applied to IMU *and* target
   velocity, accel/gyro bias, scale and white noise.

## Results (public leaderboard)

| Submission | Val (official metric) | Public LB |
| --- | --- | --- |
| all-zero reference | 1.015 | 1.054 |
| v1 — 16 s context, width 128, train only | 0.2247 | 0.378 |
| full — v1 recipe on train+val | — | 0.3625 |
| ens3 — v1 + v2 (32 s, width 160) + full | 0.2171 (v1+v2) | 0.3592 |
| ens4 — ens3 + full2 (v2 recipe on train+val) | — | 0.3544 |
| **full_long — v1 recipe on train+val, 60 epochs, EMA+SWA (single model, ranked)** | — | **0.3466** |

## Run

```bash
python analysis.py                                   # EDA → analysis/summary.md + figures
python train.py --name v1                            # train on train, validate on val → runs/v1/best.pt
python predict.py --ckpt runs/v1/best.pt --val       # val score + submission.csv
kaggle competitions submit -c tartan-imu-challenge-iros2026 -f submission.csv -m "v1"
python train.py --name full_long --splits train,val --epochs 60 --steps 250 --swa-from 50   # ranked model → runs/full_long/swa.pt
python predict.py --ckpt checkpoints/full_long_swa.pt --out submission.csv                  # reproduce the ranked submission
```
# TartanIMU-Challenge-Multi-Platform-Inertial-Odometry
