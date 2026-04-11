#!/bin/bash
set -e
cd /root/autodl-tmp/pytorch-cifar/closed_loop
mkdir -p logs
export PYTHONUNBUFFERED=1
nohup python run_experiments.py --mode full_system --runs 3 --rounds 5 > logs/full_system.log 2>&1 &
echo $! > logs/full_system.pid
echo "Started full_system, PID=$(cat logs/full_system.pid)"
