#!/bin/zsh
cd "/Users/md.hamidhosen/Documents/Kaggle/TartanIMU Challenge- Multi-Platform Inertial Odometry"
while pgrep -f "train.py --name full" > /dev/null; do sleep 30; done
python3 train.py --name v2 --chunk 32 --width 160 --batch 32 --steps 300 --epochs 30 --plat-probs 0.22,0.22,0.34,0.22 --eval-every 2 > runs_v2.log 2>&1
