# TartanIMU Challenge — Team Report

## Submission identification

| Field | Your answer |
| --- | --- |
| Team name (exactly as on the leaderboard) | _TODO: fill in from the leaderboard_ |
| Members (name — affiliation) | Md. Hamid Hosen — _TODO: affiliation_ |
| Contact email | hamidhosen8444@gmail.com |
| Submission you want ranked (Kaggle submission ID, or the submission filename + its UTC timestamp) | Kaggle submission ID **56313744** — `submission_full_long.csv`, 2026-09-17 21:12 UTC |
| Public score of that submission | 0.34661 |
| Score you expect us to reproduce | same ± 0.005 (single seed, EMA+SWA weights; MPS vs CUDA numerics differ slightly) |
| Code repository or archive (a private link is fine) | https://github.com/hamidhosen42/TartanIMU-Challenge-Multi-Platform-Inertial-Odometry |
| Commit SHA that produced the checkpoint | `6bea7bb50113db6c04b0d616d42fa90fba27f72b` (training code; the checkpoint itself is committed in the following commit) |
| Checkpoint file name(s) | `checkpoints/full_long_swa.pt` (in the repository) |
| Checkpoint SHA-256 | `0131e2ebe662fb78f269875c07eafececec2966bc0b6e542ce96b1d2f56fe352` |
| Config file path inside the repository | `configs/full_long.json` (the exact argparse namespace; also stored inside the checkpoint under `args`). Reproduce with `python train.py --name full_long --splits train,val --epochs 60 --steps 250 --swa-from 50` |
| Total training cost (GPU type × hours) | Apple M5 laptop GPU (MPS) × 1.9 h for the ranked run; ≈10 h total across all experiments |

## Artifact checklist

- [x] **Final checkpoint(s)** — `checkpoints/full_long_swa.pt`
- [x] **Training code** — `train.py`, `model.py`, `common.py` at the commit above
- [x] **The exact config / hyper-parameter file** — `configs/full_long.json`
- [x] **Inference script** — `predict.py` (`python predict.py --ckpt checkpoints/full_long_swa.pt --out submission.csv`)
- [x] **Environment** — `requirements.txt` (Python 3.12.11)
- [x] **This report.**

## Compliance statement

- [x] All ranked predictions come from a single model with one shared set of weights.

| Question | Yes / No | If yes, describe |
| --- | --- | --- |
| Weight averaging across checkpoints (soup, EMA, SWA)? | Yes | EMA of the weights during training (decay 0.998), then a uniform average of the EMA weights at the end of each of the last 11 epochs (epochs 50–60) of the *same* run (`swa_n = 11` stored in the checkpoint). One weight set results. |
| Test-time augmentation, output scaling, or calibration? | No | Only overlap averaging of sliding 16 s chunks (stride 2 windows, Hann weights) over each test trajectory — the same procedure is used for validation. No scaling, clipping or calibration. |
| Any per-platform behavior — and is it internal routing or separate models? | No | No platform input, no routing. An auxiliary 4-way platform classification head is trained (loss weight 0.05) and its output is discarded at inference. |
| Pretrained weights not included in the release? | No | Trained from scratch. |
| External data (public or private) beyond the challenge dataset? | No | |
| Anything else that changes the numbers and is not in the training code? | No | Earlier leaderboard entries (`submission_ens3/ens4.csv`) were prediction averages of several runs and are **not** the submission we ask to rank. |

Signed (name, date): Md. Hamid Hosen, 2026-09-_TODO_

---

## Technical report

## 1. Backbone

A context model over whole trajectories: raw 200 Hz IMU (6 channels, fixed scaling, gravity retained) → strided conv stem (200 → 20 Hz tokens) → 8 residual depthwise-dilated temporal-convolution blocks (dilations 1,2,4,8,16,32,1,2; ≈13 s receptive field) → 2-layer transformer encoder (4 heads, learned positions) over the whole 16 s chunk → per-token linear head giving dense 20 Hz body-frame velocity, averaged to one vector per 1 s window. 1.78 M parameters. Relative to the released ResNet-LSTM baseline the differences are (i) the model reads past *and future* context across many windows of the same trajectory instead of one isolated 1 s window, and (ii) it has no platform input or per-platform heads — embodiment is inferred implicitly (an auxiliary classification head is used only as a training signal).

## 2. Loss

`L = L_win + 0.5·L_dense + 0.2·L_drift + 0.05·L_plat`

- `L_win`: vector Huber (β = 0.25 m/s) on the per-window mean velocity (masked for padded windows) — the AVE term.
- `L_dense`: the same Huber on the 20 Hz dense output against 20 Hz-averaged ground-truth `vel_body`.
- `L_drift`: mean over t of ‖Σ_{k≤t}(v̂_k − v_k)‖ / √t within the chunk — an integrated body-frame error that mirrors ATE20's sensitivity to correlated bias.
- `L_plat`: cross-entropy of the auxiliary platform head.
Weights were set once by hand (not tuned or scheduled).

## 3. Data handling

- Windows are used as given (k·200 … (k+1)·200); trajectories are kept whole in memory and training samples are random 16-window (3 200-sample) chunks. Trajectories shorter than 16 windows are edge-padded and masked.
- **Platform-balanced sampling**: each chunk picks a platform uniformly (the metric is macro-averaged), then a trajectory with probability ∝ length.
- Augmentation (all on device, per chunk): random sensor-mount rotation — uniform axis, angle U(0°, 15°) — applied identically to accelerometer, gyroscope **and** the target velocity; accelerometer scale 1 + N(0, 0.02) and bias N(0, 0.15 m/s²); gyroscope scale 1 + N(0, 0.02) and bias N(0, 0.02 rad/s); white noise σ = 0.05 m/s² / 0.004 rad/s.
- Input normalisation: fixed division by (3, 3, 3, 0.4, 0.4, 0.4); no mean subtraction (keeps the rotation augmentation exact and gravity available as a tilt cue).
- Split: the ranked checkpoint is trained on **train + val** (the released splits, concatenated) with a fixed schedule; model design and all hyper-parameters were chosen on runs trained on `train` only and scored on `val` with the organisers' scorer.

## 4. Training schedule

AdamW (β = 0.9/0.99, weight decay 0.02), OneCycle LR (peak 1.5e-3, 8 % warm-up, final 1.5e-3/4000), batch 64 chunks × 16 windows, 250 optimizer steps per epoch, **60 epochs** (v1 model-selection runs: 30 epochs), gradient clipping 2.0, EMA of weights (decay 0.998), SWA over epochs 50–60. Apple M5 laptop (MPS backend, FP32), ≈0.45 s/step → 116 min wall-clock for the ranked run.

## 5. Model selection — how did you choose which checkpoint to submit?

Design decisions were made on `val` with the organisers' exact scorer (`kaggle_metric.py`), never on the leaderboard: v1 (16 s context, width 128) reached val 0.2247 and v2 (32 s context, width 160, drone-heavier sampling) 0.2293, so the v1 recipe was kept. The ranked checkpoint is the final EMA+SWA weights of one fixed-length train+val run — no early stopping and no checkpoint picking are possible on it, since `val` is inside its training set. In total we uploaded 6 submissions to Kaggle (1 earlier baseline, v1, full, two prediction-ensembles, and the ranked single model). Public scores: v1 0.378 → full (train+val, 30 ep) 0.363 → ensembles 0.359 / 0.354 → ranked single model (train+val, 60 ep, SWA) **0.347**. Public LB tracked val ordering but with a large offset (val 0.225 ↔ LB 0.378), consistent with the organisers' note that the public split is harder than private (baseline 0.637 public / 0.456 private).

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
- Longer training was the single largest lever: 30 → 60 epochs of the same recipe gave val 0.2233 → 0.2030 (drone-B AVE 0.328 → 0.271, car 0.129 → 0.117, human 0.085 → 0.073).

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
