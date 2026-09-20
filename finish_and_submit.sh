#!/bin/zsh
# waits for both final runs; submits each when done (independent of the Claude session)
cd "/Users/md.hamidhosen/Documents/Kaggle/TartanIMU Challenge- Multi-Platform Inertial Odometry"
submit() {
  name=$1; msg=$2
  while pgrep -f "train.py --name $name " > /dev/null || pgrep -f "train.py --name $name$" > /dev/null; do sleep 120; done
  sleep 30
  if [ -f runs/$name/swa.pt ]; then
    python3 predict.py --ckpt runs/$name/swa.pt --out submission_$name.csv > predict_$name.log 2>&1
    kaggle competitions submit -c tartan-imu-challenge-iros2026 -f submission_$name.csv -m "$msg" >> predict_$name.log 2>&1
    echo "$(date) submitted $name" >> finish_and_submit.log
  else
    echo "$(date) $name has no swa.pt" >> finish_and_submit.log
  fi
}
submit final_n_uni_160 "final_n_uni_160: v4 recipe + uniform per-trajectory sampling, train+val, 160 ep, EMA+SWA — single model" &
submit final_w_uni_160 "final_w_uni_160: wide (w192, 3 ctx) + uniform per-trajectory sampling, train+val, 160 ep, EMA+SWA — single model" &
wait
