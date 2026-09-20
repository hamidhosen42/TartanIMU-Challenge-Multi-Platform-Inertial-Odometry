---
license: apache-2.0
tags: [imu, inertial-odometry, velocity-estimation, tartanimu, iros-2026]
library_name: pytorch
---

# TartanIMU Challenge — single unified inertial-odometry model (team Hack2Publish)

One model, one shared set of weights, for car / dog / drone / human. From raw 6-axis IMU (200 Hz, gravity retained)
it predicts the mean 3-D **body-frame velocity** of every 1 s window. Built for the IROS 2026
[TartanIMU Challenge](https://www.kaggle.com/competitions/tartan-imu-challenge-iros2026).

| file | what |
| --- | --- |
| `model.pt` | **ranked entry** — Kaggle submission **56402554** (`submission_final_kaggle_w_uni.csv`, public 0.28609). 3.9 M params. SHA-256 in `SHA256SUMS`. |
| `model_secondary_final_v4_160.pt` | second selected Kaggle entry **56332266** (public 0.28696): same pipeline, width-128 / 2-layer variant, length-weighted sampling. 1.8 M params. |
| `infer.py` | inference entry point (test trajectory directory + window index CSV → submission CSV) |
| `model.py` | architecture + sliding-chunk trajectory inference |
| `requirements.txt` | pinned environment |

## Inference

```bash
pip install -r requirements.txt
python infer.py --traj_dir /path/to/test --windows /path/to/index/test_windows.csv --out submission.csv
# second entry:
python infer.py --checkpoint model_secondary_final_v4_160.pt --traj_dir ... --windows ... --out submission_v4_160.csv
```

* Reads **only** `imu` (N,6) = `[ax, ay, az, gx, gy, gz]` from each `.npz` and the window order from the CSV. No platform
  label, pose, timestamps or metadata. Trajectories are processed independently (never concatenated).
* One GPU with < 2 GB VRAM, ≈ 1 min for the 89-trajectory test set; CPU also works (`--device cpu`, tens of minutes).
  No internet access needed. TF32 is disabled for determinism; GPU-to-GPU differences are ≤ 1e-3 m/s.
* Prints the MD5 of the written CSV. The ranked entry's submitted CSV has MD5 `fd12d4f25d7b52d44aaa4e4a14534378`
  (produced with these weights on a Kaggle P100/T4; re-running on another device reproduces it to ≤ 0.005 m/s per window).

## Method (short)

Raw 200 Hz IMU chunks of 16 s → strided conv stem (200 → 20 Hz tokens) + deterministic strap-down features
(gyro-integrated relative orientation, de-gravitated acceleration, integrated velocity change, under both gyro-z sign
hypotheses) → 8 residual depthwise-dilated temporal-convolution blocks (≈ 13 s receptive field) → 3-layer transformer
encoder over the chunk → per-token velocity head (dense 20 Hz), averaged to one vector per 1 s window. At inference,
chunks slide over the whole trajectory with a stride of 2 windows and overlapping predictions are Hann-weight averaged.

Training (challenge train + val only, no external data, no pretrained weights): AdamW + OneCycle, 160 epochs × 250 steps
of 64 chunks, EMA (0.998) + SWA over the last 20 epochs; loss = vector-Huber window velocity + 0.5 dense 20 Hz Huber +
0.2 integrated-error (drift) loss + 0.05 auxiliary platform cross-entropy (training signal only, never used for routing);
sampling uniform over platforms and uniform over trajectories within a platform; augmentation = random sensor-mount
rotations (≤ 15°, ≤ 45° for drone chunks) applied to IMU and target, physically exact time dilation (× 0.77–1.3), IMU
bias / scale / white noise.

## Compliance

* Single model, single weight set at inference. No ensembling, no test-time augmentation, no platform classifier or
  routing, no output scaling / smoothing / calibration, no test-time adaptation, no external data, no pretrained weights.
* Weight averaging: EMA during training and a uniform average of the EMA weights over the last 20 epochs of the same run.
* Training code (exact commit), configs and the technical report: https://github.com/hamidhosen42/TartanIMU-Challenge-Multi-Platform-Inertial-Odometry

Team: **Hack2Publish** (Md. Hamid Hosen, Esfer Sami; Kaggle user hosen42). Data: TartanIMU Challenge, CMU AirLab.
