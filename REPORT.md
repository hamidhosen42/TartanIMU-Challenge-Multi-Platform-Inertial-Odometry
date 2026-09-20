# TartanIMU Challenge — Team Report

## Submission identification

| Field | Your answer |
| --- | --- |
| Team name (exactly as on the leaderboard) | _TODO: fill in from the leaderboard_ |
| Members (name — affiliation) | Md. Hamid Hosen — _TODO: affiliation_ |
| Contact email | hamidhosen8444@gmail.com |
| Submission you want ranked (Kaggle submission ID, or the submission filename + its UTC timestamp) | **56394522** — `submission_final_w_uni_160.csv`, 2026-09-20 12:54 UTC (our two selected Kaggle entries are this one and **56332266** `submission_final_v4_160.csv`, 2026-09-18 08:17 UTC; both checkpoints are included) |
| Public score of that submission | 0.28763 (second entry: 0.28696) |
| Score you expect us to reproduce | same ± 0.005 (single seed, EMA+SWA weights; MPS vs CUDA numerics differ slightly) |
| Code repository or archive (a private link is fine) | https://github.com/hamidhosen42/TartanIMU-Challenge-Multi-Platform-Inertial-Odometry |
| Commit SHA that produced the checkpoint | `a9a53840d23a8a426a05d5d616bead0e2f77325e` (training code; checkpoints are committed in the following commit) |
| Checkpoint file name(s) | `checkpoints/final_w_uni_160_swa.pt` (ranked); `checkpoints/final_v4_160_swa.pt` (second selected entry); `checkpoints/MANIFEST.json` |
| Checkpoint SHA-256 | `89a0db50a1300b47de87141e0871dc87836135b75cf1025ca1aeeba7b47006ea` (ranked); `f1eb208625e2c161d88baeaf3ec5203db7455d63bd8bfa24e1b424851b507a05` (second) |
| Config file path inside the repository | `configs/final_w_uni_160.json` / `configs/final_v4_160.json` (exact argparse namespaces; also stored inside each checkpoint under `args`). Reproduce with `python train.py --name final_w_uni_160 --splits train,val --epochs 160 --steps 250 --physics 1 --dilate 1.3 --rot-deg-drone 45 --boost-a 1 --traj-uniform 1 --swa-from 140 --width 192 --ctx-layers 3` |
| Total training cost (GPU type × hours) | Apple M5 laptop GPU (MPS) × ≈9 h for the ranked run (shared with another run for part of it); ≈45 h total across all experiments, plus ≈6 h on a Colab L4 |

## Artifact checklist

- [x] **Final checkpoint(s)** — `checkpoints/final_w_uni_160_swa.pt`, `checkpoints/final_v4_160_swa.pt`
- [x] **Training code** — `train.py`, `model.py`, `common.py` at the commit above
- [x] **The exact config / hyper-parameter file** — `configs/final_w_uni_160.json`, `configs/final_v4_160.json`
- [x] **Inference script** — `predict.py` (`python predict.py --ckpt checkpoints/final_w_uni_160_swa.pt --out submission.csv`; reproduces the submitted CSV bit-exactly on MPS)
- [x] **Environment** — `requirements.txt` (Python 3.12.11)
- [x] **This report.**

## Compliance statement

- [x] All ranked predictions come from a single model with one shared set of weights.

| Question | Yes / No | If yes, describe |
| --- | --- | --- |
| Weight averaging across checkpoints (soup, EMA, SWA)? | Yes | EMA of the weights during training (decay 0.998), then a uniform average of the EMA weights at the end of each of the last 21 epochs (epochs 140–160) of the *same* run (`swa_n = 21` stored in the checkpoint). One weight set results. |
| Test-time augmentation, output scaling, or calibration? | No | Only overlap averaging of sliding 16 s chunks (stride 2 windows, Hann weights) over each test trajectory — the same procedure is used for validation. No scaling, clipping or calibration. |
| Any per-platform behavior — and is it internal routing or separate models? | No | No platform input, no routing. An auxiliary 4-way platform classification head is trained (loss weight 0.05) and its output is discarded at inference. |
| Pretrained weights not included in the release? | No | Trained from scratch. |
| External data (public or private) beyond the challenge dataset? | No | |
| Anything else that changes the numbers and is not in the training code? | No | Earlier leaderboard entries (`submission_ens3/ens4.csv`) were prediction averages of several runs and are **not** the submission we ask to rank. |

Signed (name, date): Md. Hamid Hosen, 2026-09-_TODO_

---

## Technical report

## 1. Backbone

A context model over whole trajectories: raw 200 Hz IMU (6 channels, fixed scaling, gravity retained) → strided conv stem (200 → 20 Hz tokens), plus a projection of deterministic strap-down features (gyro-integrated relative orientation, de-gravitated acceleration and integrated velocity change within the chunk, computed under both gyro-z sign hypotheses) added to the tokens → 8 residual depthwise-dilated temporal-convolution blocks (dilations 1,2,4,8,16,32,1,2; ≈13 s receptive field) → 3-layer transformer encoder (4 heads, learned positions) over the whole 16 s chunk → per-token linear head giving dense 20 Hz body-frame velocity, averaged to one vector per 1 s window. Width 192, 3.9 M parameters (the second selected entry is the width-128 / 2-layer variant with 1.8 M parameters). Relative to the released ResNet-LSTM baseline the differences are (i) the model reads past *and future* context across many windows of the same trajectory instead of one isolated 1 s window, and (ii) it has no platform input or per-platform heads — embodiment is inferred implicitly (an auxiliary classification head is used only as a training signal).

## 2. Loss

`L = L_win + 0.5·L_dense + 0.2·L_drift + 0.05·L_plat`

- `L_win`: vector Huber (β = 0.25 m/s) on the per-window mean velocity (masked for padded windows) — the AVE term.
- `L_dense`: the same Huber on the 20 Hz dense output against 20 Hz-averaged ground-truth `vel_body`.
- `L_drift`: mean over t of ‖Σ_{k≤t}(v̂_k − v_k)‖ / √t within the chunk — an integrated body-frame error that mirrors ATE20's sensitivity to correlated bias.
- `L_plat`: cross-entropy of the auxiliary platform head.
Weights were set once by hand (not tuned or scheduled).

## 3. Data handling

- Windows are used as given (k·200 … (k+1)·200); trajectories are kept whole in memory and training samples are random 16-window (3 200-sample) chunks. Trajectories shorter than 16 windows are edge-padded and masked.
- **Metric-shaped sampling**: each chunk picks a platform uniformly (the metric is macro-averaged over platforms), then a
  trajectory *uniformly within the platform* (the metric averages per trajectory, so short trajectories count as much as
  long ones; sampling ∝ length — used until v6 — under-trained the short, hardest drone trajectories: val 0.2030 → 0.1982).
- Augmentation (all on device, per chunk): random sensor-mount rotation — uniform axis, angle U(0°, 15°), U(0°, 45°) for drone chunks — applied identically to accelerometer, gyroscope **and** the target velocity; accelerometer scale 1 + N(0, 0.02) and bias N(0, 0.15 m/s²); gyroscope scale 1 + N(0, 0.02) and bias N(0, 0.02 rad/s); white noise σ = 0.05 m/s² / 0.004 rad/s.
- Physically exact time dilation of training chunks by a factor logU(1/1.3, 1.3): velocity ×s, gyro ×s, dynamic acceleration ×s² with the gravity component (from the training quaternion) kept, sequence resampled.
- Input normalisation: fixed division by (3, 3, 3, 0.4, 0.4, 0.4); no mean subtraction (keeps the rotation augmentation exact and gravity available as a tilt cue).
- Split: the ranked checkpoint is trained on **train + val** (the released splits, concatenated) with a fixed schedule; model design and all hyper-parameters were chosen on runs trained on `train` only and scored on `val` with the organisers' scorer.

## 4. Training schedule

AdamW (β = 0.9/0.99, weight decay 0.02), OneCycle LR (peak 1.5e-3, 8 % warm-up, final 1.5e-3/4000), batch 64 chunks × 16 windows, 250 optimizer steps per epoch, **160 epochs** (model-selection runs: 60 epochs), gradient clipping 2.0, EMA of weights (decay 0.998), SWA over epochs 140–160. Apple M5 laptop (MPS backend, FP32), ≈3 min/epoch for the wide model → ≈8–9 h wall-clock for the ranked run.

## 5. Model selection — how did you choose which checkpoint to submit?

Design decisions were made on `val` with the organisers' exact scorer (`kaggle_metric.py`), scored per platform and
separately for the two drone sources (`breakdown.py`). Val progression of the train-only runs (60 epochs unless noted):
v1 0.2247 (30 ep) → v3 +drone augmentation 0.2233 → v4 = v3 for 60 epochs **0.2030** → v6 wide model 0.1987 →
v7 = v4 + uniform per-trajectory sampling 0.1982. Rejected on val: 32 s context (0.2293 vs 0.2247), a learned IMU/GT
time-shift head, strap-down physics features alone, rotation TTA, smoothing.

The ranked checkpoint is the final EMA+SWA weights of one fixed-length train+val run of a val-chosen recipe — no early
stopping and no checkpoint picking are possible on it, since `val` is inside its training set.

We uploaded 12 submissions in total (1 earlier baseline, 2 prediction ensembles that are *not* eligible, and 9 single
models). An important observation for the analysis paper: **the public leaderboard is noisy at the ±0.01–0.02 level for
this model family.** Two runs of the *identical* recipe (v4, 160 epochs, train+val) on different hardware (Apple M5 vs.
an L4 GPU, hence different random paths) scored 0.2870 and 0.3052; the wide 240-epoch and 160-epoch models scored
0.2897 / 0.2893. Val differences of a few thousandths therefore do not transfer to the public split, which is dominated by
a handful of short racing-drone trajectories. We consequently chose the two final entries by *val-validated recipe*
first and public score second, and we did not tune anything against the leaderboard.

## 6. Inference-time processing

`predict.py`: for each test trajectory, 16-window chunks are slid with a stride of 2 windows; every chunk is run through the network once (no TTA); per-window outputs are averaged across overlapping chunks with a raised-cosine weight (+0.05 floor) so each window is trusted most from the chunk in which it is central. Trajectories shorter than 16 windows are edge-padded and the padding discarded. No scaling, clipping, smoothing or per-platform constants. Test trajectories are processed independently (never concatenated).

## 7. External resources

- Organisers' starter kit: `kaggle_metric_tartanimu_score.py` (used verbatim for local validation) and data conventions.
- PyTorch, NumPy, pandas, SciPy, scikit-learn. No pretrained weights, no external data.
- The publicly shared inference notebook of the 0.429 entry (leaonwang, "TartanIMU V25b") was read for its description of the context-chunk / overlap-averaging idea; no code or weights from it were used.

## 8. ★ What did NOT work

- Tried a bidirectional GRU as the context module → ≈1 s/step on Apple MPS (3× slower than a 2-layer transformer, same accuracy in a short comparison) → replaced; cost ½ h.
- Tried rotation test-time augmentation (±5° and ±10° about each body axis, outputs rotated back, 7 passes) → val 0.2246 / 0.2248 vs 0.2247 without → no gain, dropped; cost ½ h.
- Tried a larger context/width (32 s chunks, width 160, drone sampling weight 0.34) → val 0.2293 vs 0.2247 (worse; drone AVE did not improve despite more drone samples) → kept the small model; cost 3 h GPU.
- Per-trajectory platform classification from hand-crafted IMU statistics reaches 100 % on val — but the rules forbid routing, so it was used only as analysis (test composition ≈ 18 car / 15 dog / 46 drone / 10 human trajectories).
- Prediction-averaging of 2–4 runs improved val 0.2247 → 0.2171 and public 0.363 → 0.354, but is not a single weight set, so it is not the ranked submission; a single 60-epoch run with SWA (0.347) ended up beating it anyway.
- Drone-targeted augmentation (sensor-mount rotations up to 45° for drones, ×4 oversampling of the racing-drone source, physically exact time dilation ×0.77–1.3) → val 0.2247 → 0.2233 overall but **no change on the racing-drone flights** (AVE 0.674 → 0.676); cost 1 h.
- Strap-down "physics" input features (gyro-integrated relative orientation, de-gravitated acceleration, integrated velocity change over the 16 s chunk, under both gyro-z sign hypotheses) → kept in the final model but no measurable gain; raw 16 s integration drifts by 5–20 m/s because of uncompensated gyro bias/scale, so the network's implicit short-horizon integration is already better; cost 2 h.
- A learned per-chunk IMU↔ground-truth **time-shift head** (supervised by lags measured on train) predicts the offsets of unseen recordings well (−66/−48 ms predicted vs −40/−70 ms measured for DAVIS-type recordings, −14/−10 vs −10 ms for Snapdragon-type, +52 vs +95 ms for an offset dog trajectory) but did not improve the score (val 0.223 vs 0.212 at equal epochs) → dropped; cost 2 h.
- Rotation TTA, prediction smoothing (3-window moving average: ATE worse on every platform), denser overlap (stride 1 vs 4: identical) → no gain.
- Longer training was the single largest lever: 30 → 60 epochs of the same recipe gave val 0.2233 → 0.2030 (drone-B AVE 0.328 → 0.271, car 0.129 → 0.117, human 0.085 → 0.073); public LB 0.3625 (30 ep) → 0.3466 (60 ep) → 0.2975 (100 ep) → 0.2870 (160 ep). Going to 240 epochs or a 1.9× wider trunk (val 0.1987) gave no further public gain (0.2897 / 0.2893) — within the ±0.01–0.02 leaderboard noise measured above.
- Tried making the second drone source's IMU frame consistent with its target frame (its gyro-z is inverted and its accelerometer dynamics fit no axis relabeling — unconstrained 3×3 fits show axis gains of 0.6–2.1), so that the racing-drone source could benefit from the 3.4 h of the other source → not feasible; cost 2 h of analysis.

## 9. ★ If you had to name one component that mattered most, what would it be?

Reading the trajectory as a sequence — predicting all windows of a 16 s chunk jointly with bidirectional context and overlap-averaging at inference. The target is strongly autocorrelated (lag-1 velocity autocorrelation 0.85 car / 0.78 dog / 0.56 human) and an oracle that merely averages the two neighbouring windows' true velocities already halves the all-zero error; the same architecture applied to isolated 1 s windows cannot estimate gyro/accelerometer bias, low-frequency tilt or the embodiment reliably. Everything else (drift loss, rotation augmentation, EMA/SWA) gave increments of a few hundredths at most.

## 10. Anything else we should know

- All training was done on a laptop (Apple M5, MPS); CUDA numerics will differ at the 3rd–4th decimal.
- Known failure mode: the racing-drone source (train `drone_train_0000–0042`, val `drone_val_0000–0008`; UZH-FPV-like) — per-trajectory AVE 0.45–1.0 m/s vs 0.15–0.4 for the other drone source, and it is ≈25 % of the test drone trajectories.
- **Data observations that may interest the organisers** (all measured from the released train/val files):
  1. In the second drone source (`drone_train_0043+`, 246 trajectories) the **gyroscope z-axis is inverted** relative to the quaternion/velocity body frame (GT yaw rate = −gyro_z, R² 0.97), while the accelerometer is not; the x/y gyro axes match the body frame once the low roll/pitch excitation is accounted for. A network can only cope with this by recognising the source from the signal.
  2. The racing-drone flights appear **twice each, recorded by two different IMUs** (one level, one pitched ≈45°, e.g. `drone_val_0005/0006` and `0007/0008`). Their IMU streams are offset from the ground truth by **different amounts: ≈ −10 ms for one IMU type and ≈ −40…−70 ms for the other**, and after time alignment and the best-fit rotation the two recordings' velocity targets still disagree by 0.15–0.9 m/s (≈5° of mount-rotation inconsistency at 4–7 m/s). This puts a floor of several tenths of a m/s on that source that no IMU-only model can pass, and it dominates the drone term.
  3. Several dog trajectories have IMU↔ground-truth offsets of 40–150 ms.
  4. The two drone sources are trivially separable from the signal (high-frequency accelerometer content differs by 10×), so a "platform-blind" test set is not source-blind.
- The val→public gap (0.225 → 0.36–0.38) was much larger than any val-measured improvement; a per-trajectory breakdown of the public split would have helped teams understand what generalises.
