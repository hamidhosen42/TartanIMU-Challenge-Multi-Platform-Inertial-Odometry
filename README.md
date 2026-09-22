# TartanIMU Challenge — Multi-Platform Inertial Odometry (team Hack2Publish)

**Team Hack2Publish** — Md. Hamid Hosen, Esfer Sami, Kahakashan Ashraf, Foysal Emon Shanto.

Predict the mean 3-D **body-frame velocity** of every 1 s window (200 Hz, 6-axis IMU) with **one model** shared
across car / dog / drone / human. Score = `0.6·AVE/0.7356 + 0.4·ATE20/3.116`, macro-averaged over platforms
(lower is better; all-zero = 1.0).

| | |
| --- | --- |
| Competition | https://www.kaggle.com/competitions/tartan-imu-challenge-iros2026 (IROS 2026 workshop *Beyond Exteroception*) |
| **Model weights + inference (Hugging Face)** | **https://huggingface.co/mdhamidhosen/tartanimu-unified-hosen42** |
| Result | public leaderboard **0.2861** (rank 10 / 131 at close); official full-test score **0.219** (organisers' baseline 0.538) |
| Technical report | [`docs/Hack2Publish_TartanIMU_report.pdf`](docs/Hack2Publish_TartanIMU_report.pdf) · paper draft [`docs/Hack2Publish_paper_draft.pdf`](docs/Hack2Publish_paper_draft.pdf) |

## Quick start (inference only)

```bash
pip install torch==2.14.0 numpy==2.5.2 pandas==3.0.5 huggingface_hub
huggingface-cli download mdhamidhosen/tartanimu-unified-hosen42 --local-dir tartanimu_model
cd tartanimu_model
python infer.py --traj_dir /path/to/test --windows /path/to/index/test_windows.csv --out submission.csv
```
Reads only the `imu` array of each trajectory; one GPU (<2 GB) does the 89-trajectory test set in ~30 s, a CPU in ~30 min.

## Repository layout

```
.
├── README.md · LICENSE (Apache-2.0) · NOTICE · CITATION.cff · requirements.txt
├── src/                       training / inference / analysis code
│   ├── common.py              data loading, window index, val "solution" builder, official-metric wrapper
│   ├── model.py               IMUNet (conv stem + strap-down features → dilated TCN → transformer → dense 20 Hz velocity)
│   ├── train.py               chunk training: metric-shaped sampling, physical augmentation, EMA + SWA, val scoring
│   ├── predict.py             checkpoint → submission.csv (optional val self-scoring)
│   ├── breakdown.py           val score per platform and per drone source
│   ├── analysis.py            EDA → analysis/
│   ├── kaggle_metric.py       the organisers' exact leaderboard scorer (starter kit, unchanged)
│   └── score_official.py      scores a CSV on the organisers' per-sequence scoring service
├── models/
│   ├── checkpoints/           the two selected checkpoints + MANIFEST.json (SHA-256 / MD5)
│   └── configs/               exact argparse configuration of each checkpoint
├── release/
│   ├── huggingface/           source of the Hugging Face release (infer.py, model.py, weights, model card)
│   └── kaggle_kernel/         the Kaggle script kernel that trained the ranked checkpoint
├── docs/
│   ├── Hack2Publish_TartanIMU_report.pdf    technical report (IEEE template, official-service tables)
│   ├── Hack2Publish_paper_draft.pdf         standalone paper draft
│   ├── report/                LaTeX sources + make_tables.py
│   └── paper/                 LaTeX sources
├── results/
│   ├── official_scores/       scoring-service outputs (overall / per platform / per sequence)
│   └── submissions/           the two selected submission CSVs
├── notebooks/                 Colab experiment notebook; self-contained v1 notebook
├── analysis/                  EDA summary + figures
├── data/                      competition data (ignored; `kaggle competitions download -c tartan-imu-challenge-iros2026`)
└── runs/                      training logs (ignored)
```

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
| full_long — v1 recipe on train+val, 60 epochs, EMA+SWA | — | 0.3466 |
| final_v4 — v4 recipe (drone aug + physics feats), train+val, 100 ep | — | 0.2975 |
| **final_v4_160 — same, 160 ep (selected entry #2)** | val recipe 0.2030 | **0.2870** |
| final_v6_160 / final_colab — wide model, 160 / 240 ep | val recipe 0.1987 | 0.2893 / 0.2897 |
| final_n_uni_160 — v4 + uniform per-trajectory sampling, 160 ep | val recipe 0.1982 | 0.2978 |
| final_w_uni_160 — wide + uniform sampling, 160 ep (laptop) | — | 0.2876 |
| **final_kaggle_w_uni — same recipe trained on a Kaggle GPU (ranked, selected entry #1)** | — | **0.2861** |

**Official full-test scores** (organisers' scoring service, all 89 sequences): ranked entry **0.21903** (AVE 0.180 m/s, ATE20 0.563 m; car 0.071 / human 0.053 / dog 0.074 / drone 0.522 m/s), second entry 0.22189.

Public-LB noise is ±0.01–0.02 for this family (identical recipes on different GPUs: 0.2870 vs 0.3052), so entries were chosen by val-validated recipe first.

## Authors

Team **Hack2Publish**: Md. Hamid Hosen (lead, mdhamidhosen4@gmail.com), Esfer Sami, Kahakashan Ashraf, Foysal Emon Shanto.

## License

Code, documents and released weights: [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for third-party components.
The competition data is not redistributed here and remains under the challenge's own terms.

## Citation

```bibtex
@misc{hack2publish2026tartanimu,
  title  = {One Model, Four Bodies: Trajectory-Context Inertial Velocity Estimation Across Cars, Quadrupeds, Drones and Handheld Sensors},
  author = {Hosen, Md. Hamid and Sami, Esfer and Ashraf, Kahakashan and Shanto, Foysal Emon},
  year   = {2026},
  note   = {Team Hack2Publish entry to the TartanIMU Challenge, IROS 2026},
  url    = {https://github.com/hamidhosen42/TartanIMU-Challenge-Multi-Platform-Inertial-Odometry}
}
```
(also in [CITATION.cff](CITATION.cff))

## References

- S. Zhao, S. Zhou, R. Blanchard, Y. Qiu, W. Wang, S. Scherer. *Tartan IMU: A Light Foundation Model for Inertial Positioning in Robotics.* CVPR 2025.
- Y. Zhao et al. *TartanIMU Challenge: Multi-Platform Inertial Odometry.* Kaggle / IROS 2026 workshop *Beyond Exteroception*, 2026. https://www.kaggle.com/competitions/tartan-imu-challenge-iros2026
- TartanIMU starter kit (metric, data conventions): https://github.com/superxslam/TartanIMU
- S. Herath, H. Yan, Y. Furukawa. *RoNIN: Robust Neural Inertial Navigation in the Wild.* ICRA 2020.
- W. Liu et al. *TLIO: Tight Learned Inertial Odometry.* IEEE RA-L 2020.
- Y. Qiu et al. *AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability.* arXiv:2501.15659, 2025.
- J. Sturm et al. *A Benchmark for the Evaluation of RGB-D SLAM Systems.* IROS 2012 (RTE definition).
- The public Kaggle kernel *[TS] TCMPIO — Unified Temporal ResNet* (nomannic19) was used as the reference for the data-loading conventions and the per-window baseline.

## Run (training)

```bash
python analysis.py                                   # EDA → analysis/summary.md + figures
python train.py --name v1                            # train on train, validate on val → runs/v1/best.pt
python predict.py --ckpt runs/v1/best.pt --val       # val score + submission.csv
kaggle competitions submit -c tartan-imu-challenge-iros2026 -f submission.csv -m "v1"
python train.py --name final_w_uni_160 --splits train,val --epochs 160 --steps 250 --physics 1 --dilate 1.3 --rot-deg-drone 45 --boost-a 1 --traj-uniform 1 --swa-from 140 --width 192 --ctx-layers 3   # ranked model
python predict.py --ckpt checkpoints/final_kaggle_w_uni_swa.pt --out submission.csv   # reproduce the ranked submission
python breakdown.py runs/<name>/swa.pt                                            # val breakdown of a train-only run
```
# TartanIMU-Challenge-Multi-Platform-Inertial-Odometry
