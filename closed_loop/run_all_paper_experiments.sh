#!/bin/bash
# run_all_paper_experiments.sh
# 只跑 full_system，后面的实验由 run_gpu2.sh / run_gpu3.sh 在其他卡上跑
set -e
cd /root/autodl-tmp/pytorch-cifar/closed_loop
mkdir -p logs
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Starting full_system at $(date)"
echo "=========================================="

# 如果 full_system 进程已经在跑（由之前的启动），等待它
while ps aux | grep -v grep | grep "run_experiments.py --mode full_system" > /dev/null 2>&1; do
    echo "[$(date)] Waiting for full_system to finish..."
    sleep 60
done
echo "[$(date)] full_system done (or not running). Starting it now..."

python run_experiments.py --mode full_system --runs 3 --rounds 5 2>&1 | tee -a logs/full_system.log

echo ""
echo "=========================================="
echo "full_system COMPLETED at $(date)"
echo "=========================================="
