#!/bin/bash
# GPU 卡 3: No SEF + No CFM 消融实验
set -e
cd /root/autodl-tmp/pytorch-cifar/closed_loop
mkdir -p logs
export PYTHONUNBUFFERED=1

echo "[$(date)] Starting GPU3 batch..."

echo "[$(date)] full_system --disable_sef starting..."
python run_experiments.py --mode full_system --disable_sef --runs 1 --rounds 5 2>&1 | tee logs/full_no_sef.log

echo "[$(date)] full_system --disable_cfm starting..."
python run_experiments.py --mode full_system --disable_cfm --runs 1 --rounds 5 2>&1 | tee logs/full_no_cfm.log

echo "[$(date)] GPU3 batch done."
