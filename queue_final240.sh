#!/bin/zsh
# after final_v6_160 finishes: submit it, then start the 240-epoch wide run
cd "/Users/md.hamidhosen/Documents/Kaggle/TartanIMU Challenge- Multi-Platform Inertial Odometry"
while pgrep -f "train.py --name final_v6_160" > /dev/null; do sleep 120; done
sleep 30
python3 predict.py --ckpt runs/final_v6_160/swa.pt --out submission_final_v6_160.csv > predict_final_v6_160.log 2>&1
kaggle competitions submit -c tartan-imu-challenge-iros2026 -f submission_final_v6_160.csv -m "final_v6_160: wide (w192, 3 ctx) v4 recipe on train+val, 160 ep, EMA+SWA(140-160) — single model" >> predict_final_v6_160.log 2>&1
nohup python3 train.py --name final_v6_240 --splits train,val --epochs 240 --steps 250 --physics 1 --dilate 1.3 --rot-deg-drone 45 --boost-a 4 --swa-from 210 --width 192 --ctx-layers 3 > runs_final_v6_240.log 2>&1 &
