# 10th place — One model, four bodies: trajectory context beats per-window regression

**Team Hack2Publish** — Md. Hamid Hosen, Esfer Sami, Kahakashan Ashraf, Foysal Emon Shanto

| | |
|---|---|
| Public LB | **0.28609** (10 / 131) |
| Official scoring service, all 89 sequences | **0.21903** (macro AVE 0.180 m/s, ATE20 0.563 m, RTE@5s 0.868 m) |
| Organisers' baseline, same service | 0.538 (AVE 0.461, ATE20 1.261) |
| Model | one network, one weight set, 3.9 M parameters, no platform input |
| Weights + inference | https://huggingface.co/mdhamidhosen/tartanimu-unified-hosen42 |
| Code | https://github.com/hamidhosen42/TartanIMU-Challenge-Multi-Platform-Inertial-Odometry |

No ensembling, no test-time augmentation, no platform routing, no external data, no pretrained weights.

---

## TL;DR

We stopped treating the 1 s window as the unit of prediction. The network reads **16 s of a trajectory at a time**,
outputs a dense 20 Hz velocity for the whole span, and overlapping spans are averaged. That single change took our
held-out score from ~0.30 to 0.2247. Everything after that was smaller: longer schedules (→ 0.2030), sampling the
data the way the metric averages it (→ 0.1982), a slightly wider trunk. We also spent the last week finding out
**why the drone term refuses to move**, and the answer is in the data, not the model.

---

## 1. The observation that decided the design

Before training anything we measured how much a window tells us about its neighbours:

| platform | lag-1 autocorrelation of the window velocity |
|---|---|
| car | 0.85 |
| dog | 0.78 |
| human | 0.56 |
| drone | 0.21 |

And an **oracle** that predicts each window as the average of its two neighbours' *true* velocities gets (val, mean
per-window error): car 0.198, dog 0.153, human 0.168, drone 0.939 m/s — against 0.594 / 0.497 / 0.263 / 1.556 for
predicting zero. So the information is there in the neighbourhood, and a per-window model throws it away.

The test trajectories are given whole, so using context is legal. One second is not enough to estimate gyroscope
bias, low-frequency tilt, or the embodiment; sixteen seconds is.

---

## 2. Model

```
raw IMU chunk (16 s, 200 Hz, 6 ch, gravity retained)
   → conv stem (k9/s2, k11/s5) → 20 Hz tokens, width 192
   + parameter-free strap-down features (added through a 1×1 projection)
   → 8 residual depthwise-dilated TCN blocks (d = 1,2,4,8,16,32,1,2 → RF ≈ 13 s, GroupNorm)
   → 3-layer pre-norm transformer encoder (4 heads, learned positions) over the 320 tokens
   → linear head → dense 20 Hz body-frame velocity
   → mean of the 20 tokens inside a window → v̂ for that window
```

Details that matter:

* **No mean subtraction.** Each channel is divided by a fixed constant (3 m/s² accel, 0.4 rad/s gyro) and that's it.
  Gravity stays in the signal as the only tilt cue, and rotation augmentation stays exact.
* **Strap-down features** (no learned parameters): the gyro is integrated into a per-token relative orientation
  (parallel prefix product of rotation matrices), the accelerometer is rotated into the chunk-start frame, its chunk
  mean is taken as a gravity estimate, and the de-gravitated acceleration and its integral are rotated back into the
  current body frame. Computed for **both signs of ω_z** (see §6), projected and added to the tokens.
* **Auxiliary platform head** (4-way, CE weight 0.05) exists only in the training graph. It is never evaluated at
  inference and nothing depends on it — the embodiment is inferred implicitly.

---

## 3. Training

**Loss** `L = L_win + 0.5·L_dense + 0.2·L_drift + 0.05·L_plat`

| term | what |
|---|---|
| `L_win` | vector Huber (β = 0.25 m/s) on the per-window velocity error — the AVE term |
| `L_dense` | same on the 20 Hz output vs 10-sample means of `vel_body` |
| `L_drift` | `mean_t ‖Σ_{k≤t}(v̂_k − v_k)‖ / √t` — integrated body-frame error, the correlated bias that ATE20 sees and a per-window loss does not |
| `L_plat` | auxiliary platform cross-entropy |

**Sampling shaped after the metric.** The score is macro-averaged over platforms and averaged *per trajectory*
within a platform. So: draw a platform uniformly, then a **trajectory uniformly within that platform**, then a random
chunk. Length-proportional sampling (our first version) under-trains exactly the short drone clips that carry most of
the error — switching to uniform was worth 0.005 on val.

**Augmentation, physically exact.**

* *Re-mounting the sensor*: a random rotation (uniform axis, angle U(0°, 15°); U(0°, 45°) for drone chunks) applied to
  accelerometer, gyroscope **and the velocity label** — exactly what a differently mounted sensor would have measured.
* *Time dilation*: resample by `s ~ logU(1/1.3, 1.3)`; `v → s·v`, `ω → s·ω`, dynamic acceleration `→ s²`, while the
  gravity component (known from the training quaternion) is kept. Produces realistic faster/slower flights.
* Accel/gyro scale 2 %, bias 0.15 m/s² / 0.02 rad/s, white noise 0.05 m/s² / 0.004 rad/s.

**Optimisation.** AdamW (wd 0.02), OneCycle (peak 1.5e-3, 8 % warm-up), batch 64 chunks, 250 steps/epoch, clip 2.0,
EMA 0.998, then a plain average of the EMA weights over the last 20 epochs (SWA). Development runs: 60 epochs on
`train` only. Final model: **160 epochs on train+val**, fixed schedule, no early stopping, no checkpoint picking.
Trained by a Kaggle script kernel on a **T4 in 3.6 h**.

---

## 4. Inference

For each test trajectory: 16-window chunks slid with a **stride of 2 windows**, one forward pass each, and the
per-window outputs of overlapping chunks averaged with a raised-cosine weight (plus a small floor), so a window is
trusted most from the chunk where it sits near the centre. Trajectories shorter than 16 windows are edge-padded and
the padding discarded. Trajectories are never concatenated.

Nothing else: no TTA, no scaling, no smoothing, no clipping, no test-time adaptation. **26 s for the whole test set
on a T4** (~1,200 windows/s), 30 min on a 10-core CPU, < 2 GB GPU / 0.6 GB RAM.

---

## 5. What actually moved the score

All numbers are TartanIMU scores on the held-out `val` split (80 trajectories, organisers' scorer), models trained on
`train` only.

| change | val | Δ |
|---|---|---|
| per-window 1-D ResNet (starter-kernel style) | ≈0.30 | — |
| **16 s context + dense output + overlap averaging** (30 ep) | **0.2247** | **−0.08** |
| 32 s context + wider trunk instead (30 ep) | 0.2293 | worse |
| + drone augmentation (45° rotations, time dilation) | 0.2233 | −0.001 |
| **+ 60 epochs instead of 30** | **0.2030** | **−0.02** |
| **+ uniform per-trajectory sampling** | **0.1982** | **−0.005** |
| wider trunk (192, 3 layers) instead | 0.1987 | −0.004 |
| drone sampling share 40 % instead | 0.2013 | worse |
| vibration-amplitude augmentation instead | 0.2065 | worse |

On the public leaderboard, the train+val refits tracked the schedule length: **0.3625** (30 ep) → **0.3466** (60) →
**0.2975** (100) → **0.2870** (160). 240 epochs gave nothing more (0.2897).

### Things that did **not** work

* **Strap-down features alone.** Integrating the raw gyro over 16 s drifts by 5–20 m/s under uncompensated bias, so
  the network's own short-horizon integration is already better. (We kept the features; they cost nothing.)
* **A learned per-chunk time-shift head**, supervised with IMU↔ground-truth offsets we measured on train. It predicted
  the offsets of *unseen* recordings well (−66/−48 ms predicted vs −40/−70 ms measured) but did not improve the score.
* **Rotation TTA** (0.2246 vs 0.2247), **3-window smoothing** (ATE worse on every platform), **denser overlap**
  (stride 1 vs 4: identical).
* **Prediction ensembling** improved val to 0.2171 and public to 0.354 (from 0.363) — but it is not a single weight
  set, so it could not be a ranked entry.

---

## 6. Where the error lives (the part we'd want the organisers to see)

Official per-platform numbers of our submission:

| platform | ATE20 (m) | AVE (m/s) | RTE@5s (m) | share of score |
|---|---|---|---|---|
| Wheeled / Car | 0.369 | 0.071 | 0.323 | 12.0 % |
| Handheld / Human | 0.490 | 0.053 | 0.270 | 12.1 % |
| Legged / Quadruped | 0.358 | 0.074 | 0.353 | 12.1 % |
| **Aerial / Drone** | **1.037** | **0.522** | **2.528** | **63.8 %** |
| Average | 0.563 | 0.180 | 0.868 | |

Per sequence the median AVE is 0.173 m/s, the best five are 0.015–0.029 m/s, and the worst five are 0.66–2.90 m/s —
all drone. The error is concentrated, not spread.

Splitting the drone data by source explains it. **Drone-A** (the racing drone: `drone_train_0000–0042`,
`drone_val_0000–0008`, ~30 min) sits at 0.45–1.0 m/s **at every speed**; **drone-B** (the rest, 3.4 h) sits at
0.15–0.4 m/s and is *more* accurate at 3–4 m/s than when hovering. Two measurements on the released train/val files:

1. **Drone-B's frame conventions differ from everything else.** Its gyroscope **z axis is inverted** relative to the
   body frame of its own quaternion and velocity (ground-truth yaw rate = −ω_z, R² = 0.97), and its accelerometer
   dynamics are **not a rotation** of the ground-truth specific force under any axis relabelling (R² at 2 Hz is
   negative; unconstrained 3×3 fits give axis gains 0.6–2.1). The other three platforms and drone-A have identity
   conventions with R² 0.97–0.99. The network copes because the source is trivially recognisable from the signal
   (its high-frequency accelerometer content is ~10× lower), but the physics it learns there does not transfer to
   drone-A.
2. **Every racing flight is recorded twice, by two IMUs** (one level, one pitched ≈45°, e.g. `drone_val_0005` and
   `0006`). Cross-correlating |ω| with the quaternion-derived rate shows the two streams are offset from the ground
   truth by **different amounts**: ≈ −10 ms for one IMU type, −40…−70 ms for the other. After aligning in time and
   fitting the best rotation between them, **their velocity labels still disagree by 0.15–0.9 m/s** (≈5° of mounting
   inconsistency at 4–7 m/s). No IMU-only model can beat that disagreement, and ~a quarter of the test drone
   trajectories are of this type. Several dog trajectories also carry 40–150 ms offsets.

The other irreducible piece is **hover drift**: for the slowest drone-B sequences our AVE equals the standard
deviation of their own ground-truth velocity, i.e. the model predicts the mean, because ±0.15 m/s oscillations at
0.1 Hz produce ~0.1 m/s² under 4–6 m/s² of vibration.

---

## 7. The public leaderboard is noisy — validate locally

Two runs of the **identical recipe** on different GPUs scored **0.2870** and **0.3052** publicly. Our wide model
scored 0.2893 (160 ep) and 0.2897 (240 ep); the ranked recipe scored 0.2861 on a Kaggle T4 and 0.2876 on a laptop.
That ±0.01–0.02 is larger than most differences we were trying to detect, because the public split is dominated by a
handful of short racing-drone clips under per-trajectory averaging.

So every decision was made on `val` with the organisers' exact scorer (`kaggle_metric_tartanimu_score.py` from the
starter kit — our val floor 0.013 and all-zero 1.015 match the published references), and never on the leaderboard.
Our two selected entries are the **val-validated recipe** first and public score second.

---

## 8. Reproduce it

```bash
pip install torch==2.14.0 numpy==2.5.2 pandas==3.0.5 huggingface_hub
huggingface-cli download mdhamidhosen/tartanimu-unified-hosen42 --local-dir tartanimu_model
cd tartanimu_model
python infer.py --traj_dir /path/to/test --windows /path/to/index/test_windows.csv --out submission.csv
# -> md5 fd12d4f25d7b52d44aaa4e4a14534378  (our ranked submission 56402554)
```

Training (from the GitHub repo):

```bash
python src/train.py --name final --splits train,val --epochs 160 --steps 250 \
  --physics 1 --dilate 1.3 --rot-deg-drone 45 --boost-a 1 --traj-uniform 1 \
  --swa-from 140 --width 192 --ctx-layers 3
python src/predict.py --ckpt runs/final/swa.pt --out submission.csv
python src/breakdown.py runs/final/swa.pt     # val score per platform and per drone source
```

The repo also contains the Kaggle script kernel that trained the ranked checkpoint (`release/kaggle_kernel/`), the
exact configs, the EDA, and the technical report with the full 89-sequence table.

---

## 9. What we'd do next

* Estimate **bias, gravity and time offset explicitly**, with an uncertainty, instead of hoping the network absorbs
  them — that is where the racing-drone error lives.
* Use the fact that each racing flight has **two IMU recordings** as a self-supervised consistency signal.
* For benchmark design: verify and document per-source synchronisation and frame conventions, and consider a
  genuinely held-out platform or a causal track — with recognisable sources, a platform-blind test set measures
  source recognition as much as generalisation.

Thanks to the CMU AirLab team for the dataset, the exact scorer and the per-sequence scoring service.
