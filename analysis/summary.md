# TartanIMU challenge — data analysis

## 1. Inventory

| platform   | split   |   trajectories |   windows |   hours |   median_traj_s |   min_traj_s |   max_traj_s |
|:-----------|:--------|---------------:|----------:|--------:|----------------:|-------------:|-------------:|
| car        | train   |             44 |     23533 |    6.54 |             470 |          106 |         2108 |
| dog        | train   |             36 |     14915 |    4.14 |             255 |           20 |         3870 |
| drone      | train   |            289 |     14213 |    3.95 |              60 |           12 |           60 |
| human      | train   |             26 |     29270 |    8.13 |            1119 |          958 |         1334 |
| car        | val     |             12 |      7348 |    2.04 |             466 |           91 |         2032 |
| dog        | val     |             13 |      6301 |    1.75 |             258 |           22 |         1687 |
| drone      | val     |             48 |      2327 |    0.65 |              60 |           17 |           60 |
| human      | val     |              7 |      7738 |    2.15 |            1049 |         1029 |         1310 |

Test: 89 trajectories, 30644 windows (8.51 h); trajectory length s: min 14, median 60, max 2618. No platform labels.

Windows are contiguous, non-overlapping 1 s slices of a trajectory, and test trajectories are given whole → a model may legitimately use temporal context around each window (past *and* future).

## 2. Targets (body-frame velocity)

| split   | platform   |   vx_mean |   vy_mean |   vz_mean |   vx_std |   vy_std |   vz_std |   speed_med |   speed_p95 |   speed_max |   still_frac |
|:--------|:-----------|----------:|----------:|----------:|---------:|---------:|---------:|------------:|------------:|------------:|-------------:|
| train   | car        |     0.337 |     0.002 |    -0.01  |    0.63  |    0.077 |    0.03  |       0.427 |       1.21  |       4.011 |        0.151 |
| train   | dog        |     0.141 |     0.008 |     0.001 |    0.459 |    0.151 |    0.056 |       0.209 |       1.008 |       5.323 |        0.424 |
| train   | drone      |     0.175 |    -0.032 |     0.111 |    1.454 |    1.355 |    0.688 |       0.908 |       4.619 |      11.258 |        0.042 |
| train   | human      |     0.11  |    -0.027 |    -0.083 |    0.278 |    0.123 |    0.243 |       0.112 |       0.95  |       3.952 |        0.479 |
| val     | car        |     0.227 |     0.002 |    -0.005 |    0.629 |    0.08  |    0.027 |       0.381 |       1.199 |       5.1   |        0.159 |
| val     | dog        |     0.31  |     0.005 |     0.003 |    0.526 |    0.123 |    0.054 |       0.31  |       1.385 |       1.823 |        0.341 |
| val     | drone      |     0.073 |     0.15  |     0.124 |    1.229 |    1.466 |    0.664 |       1.141 |       4.622 |       9.108 |        0.042 |
| val     | human      |     0.142 |    -0.012 |    -0.099 |    0.302 |    0.097 |    0.243 |       0.094 |       1.065 |       1.45  |        0.51  |

* car: dominantly forward (+vx) motion, tiny vy/vz → strong 'non-holonomic' prior.
* dog: forward-biased but with substantial lateral/vertical components (gait).
* drone: fastest and fully 3-D; largest speed spread → dominates the all-zero error.
* human: slow handheld motion, most 'still' windows; sensor orientation is arbitrary.

## 3. IMU channel statistics (per platform, train)

| platform   |   ax_mean |   ax_std |   ay_mean |   ay_std |   az_mean |   az_std |   gx_mean |   gx_std |   gy_mean |   gy_std |   gz_mean |   gz_std |   |a|_mean |
|:-----------|----------:|---------:|----------:|---------:|----------:|---------:|----------:|---------:|----------:|---------:|----------:|---------:|-----------:|
| car        |     0.211 |    1.721 |     0.028 |    2.21  |     9.764 |    1.359 |    -0.002 |    0.12  |    -0.003 |    0.108 |     0     |    0.26  |     10.14  |
| dog        |     0.101 |    2.567 |    -0.11  |    2.344 |     9.703 |    2.508 |    -0     |    0.15  |    -0     |    0.162 |    -0.004 |    0.227 |     10.306 |
| drone      |    -0.327 |    2.142 |     0.035 |    1.902 |     9.736 |    1.128 |     0.006 |    0.479 |    -0.01  |    0.437 |     0.198 |    0.962 |     10.141 |
| human      |    -1.668 |    3.114 |    -8.64  |    2.117 |     1.097 |    2.58  |     0     |    0.355 |    -0.003 |    0.773 |     0.005 |    0.383 |      9.904 |

Mean accelerometer vector ≈ gravity direction in the body frame: +z for car/dog/drone (upright mounts), but tilted/-y for human (handheld, arbitrary). Gyro/accel std differ by an order of magnitude between platforms → per-trajectory vibration signature is a strong platform cue the network can learn implicitly.

## 4. Temporal structure — how much does context help?

Autocorrelation of the per-window velocity between windows k and k+lag (train). High values mean the velocity changes slowly relative to the 1 s window, so neighbouring windows carry information about the current one.

| platform   |   lag1 |   lag2 |   lag5 |   lag10 |
|:-----------|-------:|-------:|-------:|--------:|
| car        |  0.848 |  0.716 |  0.524 |   0.343 |
| dog        |  0.777 |  0.606 |  0.44  |   0.272 |
| drone      |  0.208 | -0.043 | -0.118 |   0.004 |
| human      |  0.557 |  0.358 |  0.239 |   0.189 |

Simple predictability baselines (val, per-window mean Euclidean error, m/s):

| platform   |   zero |   platform_mean |   prev_window |   neighbour_avg |
|:-----------|-------:|----------------:|--------------:|----------------:|
| car        |  0.594 |           0.546 |         0.279 |           0.198 |
| dog        |  0.497 |           0.479 |         0.219 |           0.153 |
| drone      |  1.556 |           1.575 |         1.371 |           0.939 |
| human      |  0.263 |           0.295 |         0.21  |           0.168 |

`prev_window` = oracle that copies the previous window's true velocity; `neighbour_avg` = oracle average of both neighbours. Errors are far below the all-zero level for every platform → the target is strongly temporally coherent, and a model that sees several seconds of context should beat a per-window model.

## 5. Gravity leakage into the body frame

Body-frame velocity is coupled to orientation: when the sensor tilts, gravity moves between accelerometer axes. Per-window mean accelerometer vs. mean velocity (train, 3000 random windows per platform):

## 6. Test-set composition (platform is hidden)

Per-trajectory IMU summary features (channel means/stds, |a| stats, gyro energy) are enough to separate the platforms almost perfectly on val. This is *analysis only* — the rules forbid routing to per-platform models — but it tells us the test set's likely composition and that a unified network can infer the embodiment from the signal itself.

Trajectory-level platform classifier: val accuracy = 1.000 (80 val trajectories).

| pred_platform   |   trajectories |   windows |
|:----------------|---------------:|----------:|
| car             |             18 |      8787 |
| dog             |             15 |      6609 |
| drone           |             46 |      3984 |
| human           |             10 |     11264 |

The test split is roughly platform-balanced by trajectory count, so — together with the macro-averaged metric — every platform matters equally; drone (largest all-zero error) and car (largest ATE) offer the most score headroom.

## 7. Train → val shift

Val speed distributions (dashed in `speed_ecdf.png`) differ visibly from train for dog and drone; and the val split's drone trajectories are fewer/shorter. Expect val ↔ public-LB gaps of a few hundredths.

## 9. Modelling implications

1. **Use context.** Windows come in whole trajectories; velocity is highly autocorrelated. Run a sequence
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
