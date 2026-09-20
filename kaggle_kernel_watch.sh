#!/bin/zsh
cd "/Users/md.hamidhosen/Documents/Kaggle/TartanIMU Challenge- Multi-Platform Inertial Odometry"
K=hosen42/tartanimu-final-w-uni
while true; do
  s=$(kaggle kernels status $K 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z]+' | head -1)
  echo "$(date) $s" >> kaggle_kernel_watch.log
  case "$s" in
    *COMPLETE*) break;;
    *ERROR*|*CANCEL*) echo "kernel failed: $s" >> kaggle_kernel_watch.log; kaggle kernels output $K -p kaggle_out >> kaggle_kernel_watch.log 2>&1; exit 1;;
  esac
  sleep 300
done
mkdir -p kaggle_out && kaggle kernels output $K -p kaggle_out >> kaggle_kernel_watch.log 2>&1
if [ -f kaggle_out/submission.csv ]; then
  cp kaggle_out/submission.csv submission_final_kaggle_w_uni.csv
  kaggle competitions submit -c tartan-imu-challenge-iros2026 -f submission_final_kaggle_w_uni.csv -m "final_kaggle_w_uni: wide + uniform sampling, train+val, 160 ep, EMA+SWA — single model (trained on Kaggle GPU)" >> kaggle_kernel_watch.log 2>&1
  echo "$(date) submitted" >> kaggle_kernel_watch.log
fi
