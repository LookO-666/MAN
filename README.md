# MAN: Closed-Loop LLM-Guided CIFAR-10 Research on PyTorch

This repository extends the original CIFAR-10 training codebase with a closed-loop experimentation framework for model improvement research. It adds automated idea generation, experiment execution, result analysis, and iterative refinement across multiple research strategies.

## Project Overview

The codebase contains two main parts:

- Original PyTorch CIFAR-10 training pipeline based on ResNet and related architectures
- A `closed_loop/` framework for running iterative AI-guided research experiments

The closed-loop system supports:

- Full closed-loop search with mini-experiment screening and feedback
- Ablation variants such as `no_memory`, `no_mini_exp`, `no_llm_analysis`, `linear`, and `no_feedback`
- Baselines such as `random`, `openloop`, and `single_shot` / zero-shot style search

## Repository Structure

```text
.
├── main.py                      # Original CIFAR-10 training entry
├── models/                      # Backbone model definitions
├── utils.py                     # Training utilities
├── run_baselines.py             # Baseline experiment runner
├── results/                     # Baseline and comparison results
└── closed_loop/
    ├── loop.py                  # Main closed-loop research driver
    ├── runner.py                # Experiment execution wrapper
    ├── llm_client.py            # LLM API interaction
    ├── prompts.py               # Prompt templates
    ├── signal_extractor.py      # Early-signal extraction for screening
    ├── experience_memory.py     # Failure/experience memory
    ├── config.py                # Global experiment config
    └── experiments/             # Outputs from iterative experiments
```

## Installation

Recommended environment:

- Python 3.8+
- PyTorch compatible with your CUDA environment

Install dependencies manually if needed:

```bash
pip install torch torchvision tqdm requests
```

## Original CIFAR-10 Training

Train the baseline model:

```bash
python main.py
```

Resume training:

```bash
python main.py --resume --lr 0.01
```

## Closed-Loop Research Workflow

Main entry:

```bash
cd closed_loop
python loop.py
```

Run a specific method variant:

```bash
python loop.py --method full
python loop.py --method no_memory
python loop.py --method no_mini_exp
python loop.py --method linear
python loop.py --method random
python loop.py --method single_shot
```

## Configuration

Core configuration is defined in `closed_loop/config.py`.

Important options include:

- Number of rounds
- Number of candidates per round
- Number of survivors
- Mini/full experiment epochs
- LLM API endpoint and model name
- Method variant

## Result Summaries

Current recorded summaries in this repository include:

- `results/zeroshot/summary.json`: best test accuracy `82.66`
- `results/openloop/summary.json`: best test accuracy `82.85`
- `results/random/summary.json`: best test accuracy `83.59`
- `closed_loop/experiments/full/final_summary.json`: baseline `95.51`, best `95.51`, improvement `0.0`

## Notes

- Dataset files, checkpoints, and `.pth` weights are excluded via `.gitignore`
- Some experiment directories use linked `data` paths
- The repository currently contains experiment outputs and logs for multiple rounds and ablation settings

## Acknowledgement

This project is built on top of the original [`kuangliu/pytorch-cifar`](https://github.com/kuangliu/pytorch-cifar) repository and extends it with a closed-loop LLM-guided experimentation framework.
