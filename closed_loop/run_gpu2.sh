#!/bin/bash
# GPU 卡 2: Random Search + Single Turn + Zero Memory
set -e
cd /root/autodl-tmp/pytorch-cifar/closed_loop
mkdir -p logs
export PYTHONUNBUFFERED=1

echo "[$(date)] Starting GPU2 batch..."

echo "[$(date)] random_search starting..."
python run_experiments.py --mode random_search --runs 1 --rounds 5 2>&1 | tee logs/random_search.log

echo "[$(date)] single_turn starting..."
python run_experiments.py --mode single_turn --runs 3 2>&1 | tee logs/single_turn.log

echo "[$(date)] zero_memory starting..."
python run_experiments.py --mode zero_memory --runs 1 --rounds 5 2>&1 | tee logs/zero_memory.log

echo "[$(date)] GPU2 batch done."
