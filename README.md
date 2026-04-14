# MAN: LLM-Driven Closed-Loop Neural Architecture Improvement

MAN (Model Augmentation Network) is a closed-loop AI research system that uses LLM-guided iterative experimentation to automatically improve deep learning models. Starting from a ResNet-18 baseline on CIFAR-10 (95.51% test accuracy), the system generates improvement ideas, implements them as code, runs experiments, analyzes results, and feeds lessons back into the next round — all without human intervention.

## Key Features

- **Closed-Loop Iteration**: Multi-round research cycle — idea generation → code implementation → mini-experiment screening → full training → result analysis → memory update
- **Structured Early Feedback (SEF)**: 8-dimensional signal extraction from 5-epoch mini-experiments to predict long-term potential before committing to full 200-epoch training
- **Component-level Failure Memory (CFM)**: Structured experience memory that stores failure types, mechanism hypotheses, avoid patterns, and salvageable components — not just flat text logs
- **REFINE / DISCARD Triage**: A 4-quadrant rescue framework that decides whether a failed idea should be refined (overfitting potential, slow-starter, unstable peak) or discarded (dead end)
- **Exploration-Exploitation Budgeting**: Dynamically allocates idea slots between new exploration and refinement of promising prior ideas
- **Multiple Ablation Variants**: Supports systematic ablation studies for paper-ready comparisons

## Repository Structure

```
.
├── main.py                          # CIFAR-10 training entry (ResNet-18 baseline)
├── models/                          # Backbone definitions (ResNet, VGG, DenseNet, etc.)
├── utils.py                         # Training utilities (progress bar, init)
├── run_baselines.py                 # Baseline runners: random / zeroshot / openloop
├── results/                         # Saved baseline, baselines, and closed-loop logs
│   ├── baseline_200ep/              #   ResNet-18 200-epoch baseline (95.51%)
│   ├── random/                      #   Random hyperparameter-search baseline
│   ├── zeroshot/                    #   Single-shot LLM baseline
│   ├── openloop/                    #   Multi-iteration LLM without feedback baseline
│   ├── full_run1/                   #   Closed-loop full system run #1 logs
│   ├── full_run2/                   #   Closed-loop full system run #2 logs
│   ├── no_cfm/                      #   Ablation logs (without CFM)
│   ├── no_sef/                      #   Ablation logs (without SEF)
│   └── random_200ep/                #   Random-strategy 200-epoch run logs
│
└── closed_loop/                     # Core closed-loop system
    ├── loop.py                      #   Main research loop driver
    ├── runner.py                    #   Experiment execution & sandboxing
    ├── llm_client.py                #   DeepSeek API client with retry & token tracking
    ├── prompts.py                   #   All LLM prompt templates
    ├── signal_extractor.py          #   8-dim early signal extraction (SEF)
    ├── experience_memory.py         #   Structured failure memory (CFM)
    ├── config.py                    #   Global experiment configuration
    ├── run_experiments.py           #   Unified experiment orchestrator
    ├── run_all_paper_experiments.sh #   Full system batch runner
    ├── run_gpu2.sh                  #   Baselines batch (random, single-turn, zero-memory)
    └── run_gpu3.sh                  #   Ablation batch (no-SEF, no-CFM)
```

## Installation

**Requirements**: Python 3.8+, PyTorch (with CUDA), OpenAI-compatible API access

```bash
pip install torch torchvision tqdm openai numpy
```

## Quick Start

### 1. Train the Baseline

```bash
python main.py --epoch 200
```

This trains ResNet-18 on CIFAR-10 for 200 epochs and saves results to `results.json`.

### 2. Run the Full Closed-Loop System

```bash
cd closed_loop
python loop.py --method full --num_rounds 5
```

### 3. Run Baselines for Comparison

```bash
# Random hyperparameter search
python run_baselines.py --strategy random

# Single-shot LLM modification (no feedback)
python run_baselines.py --strategy zeroshot

# Multi-iteration LLM without feedback loop
python run_baselines.py --strategy openloop
```

### 4. Run Paper Experiments (Unified Orchestrator)

```bash
cd closed_loop

# Full system with 3 independent runs
python run_experiments.py --mode full_system --runs 3 --rounds 5

# Ablation: disable Structured Early Feedback
python run_experiments.py --mode full_system --disable_sef --runs 1 --rounds 5

# Ablation: disable Component-level Failure Memory
python run_experiments.py --mode full_system --disable_cfm --runs 1 --rounds 5

# Baselines
python run_experiments.py --mode random_search --runs 1 --rounds 5
python run_experiments.py --mode single_turn --runs 3
python run_experiments.py --mode zero_memory --runs 1 --rounds 5
```

## Method Variants

| Method | Description |
|--------|-------------|
| `full` | Complete system: SEF screening + CFM memory + LLM analysis + refinement |
| `no_memory` | Disable experience memory — each round starts from scratch |
| `no_mini_exp` | Skip mini-experiment screening — LLM picks survivors by judgment alone |
| `no_llm_analysis` | Skip LLM analysis — rank by val_acc only (degrades to Hyperband-style) |
| `no_feedback` | Each round independent — no history in prompts |
| `linear` | 1 idea per round, run full experiment directly (simulates AI Scientist) |
| `random` | Random hyperparameter search, no LLM |
| `single_shot` | LLM gives one plan, run once, done |

## Ablation Switches

| Flag | Effect |
|------|--------|
| `--disable_sef` | Degrade early signal extraction to simple val_acc only (no 8-dim features) |
| `--disable_cfm` | Degrade memory to flat (idea, accuracy) pairs — no constraints, no mechanism hypotheses |

## Configuration

Core settings are in `closed_loop/config.py`:

- `num_rounds`: Number of research iterations (default: 5)
- `num_candidates`: Ideas generated per round (default: 6)
- `num_survivors`: Ideas promoted to full training (default: 2)
- `mini_epochs` / `full_epochs`: 5 / 200
- `model`: LLM model name (default: `deepseek-chat`)
- `method`: Experiment variant

LLM API settings can be overridden via environment variables (`DEEPSEEK_API_KEY`) or CLI arguments.

## How It Works

```
Round N:
  ┌─────────────────────────────────────────────────────┐
  │ 1. Idea Generation (LLM + memory constraints)       │
  │    → 6 candidate ideas (mix of new + refined)       │
  │                                                     │
  │ 2. Code Implementation (LLM)                        │
  │    → Complete training scripts for each idea         │
  │                                                     │
  │ 3. Mini-Experiment Screening (5 epochs + SEF)       │
  │    → 8-dim signal extraction per candidate           │
  │    → LLM-based triage: SELECT / REJECT              │
  │    → REFINE or DISCARD rejected ideas                │
  │    → Top 2 survivors advance                         │
  │                                                     │
  │ 4. Full Training (200 epochs)                        │
  │    → Run survivors with full budget                  │
  │                                                     │
  │ 5. Result Analysis (LLM)                             │
  │    → Lessons learned, failure types, mechanism hyp.  │
  │    → Update experience memory (CFM)                  │
  │    → Suggestions for Round N+1                       │
  └─────────────────────────────────────────────────────┘
```

## First LLM Invocation: Entry Point, Call Chain, and Input Payload

### Entry Point

The first LLM call happens in **Step 1 (Idea Generation)** of the very first round, triggered from `closed_loop/loop.py`.

### Call Chain

```
closed_loop/loop.py  run_full_loop()          # line 349
  └─ loop.py         for round_num in ...:    # line 390  (Round 1)
       └─ loop.py    _generate_ideas_llm()    # line 423  (Step 1: Generating ideas)
            └─ loop.py  call_llm(...)         # line 718  ← first actual LLM call
                 └─ llm_client.py  call_llm() # line 26   (sends OpenAI chat.completions.create)
```

> **Note – `single_shot` mode**: `run_full_loop` immediately delegates to `_run_single_shot()` (line 387), which also calls `_generate_ideas_llm` as its first action, so the first LLM call is structurally identical.  
> **`random` mode**: No LLM is used; ideas come from a pre-defined list (`RANDOM_MODIFICATIONS`).

### Concrete Input Payload (Round 1, Step 1)

The call at `loop.py:718` is:
```python
response = call_llm("你是一位ML研究专家。", prompt, **llm_kw)
```

This translates to the following `messages` array sent to the OpenAI-compatible API (`llm_client.py:40-43`):

```json
[
  {
    "role": "system",
    "content": "你是一位ML研究专家。"
  },
  {
    "role": "user",
    "content": "<IDEA_GENERATION_PROMPT — see closed_loop/prompts.py:3>"
  }
]
```

### How the User Prompt Is Assembled

The user message is built at `loop.py:710-717` by filling `IDEA_GENERATION_PROMPT` (defined in `closed_loop/prompts.py:3-48`) with six context variables:

| Variable               | Source                                                              | Value on Round 1 (fresh run)           |
|------------------------|---------------------------------------------------------------------|----------------------------------------|
| `{baseline_acc}`       | `results/baseline_200ep/results.json` → `best_test_acc`            | `95.51` (%)                            |
| `{history_summary}`    | `build_history_summary(all_round_results)` — `loop.py:62`          | `"暂无历史实验结果（这是第一轮）。"`   |
| `{experience_lessons}` | `ExperienceMemory.format_for_prompt(...)` — `experience_memory.py:181` | `"暂无历史经验。"`                 |
| `{generation_guidance}`| `ExperienceMemory.format_generation_guidance(...)`                  | `"暂无额外约束。"`                     |
| `{previous_suggestions}`| `build_suggestions_summary(all_round_results)` — `loop.py:108`   | `"暂无历史建议。"`                     |
| `{num_ideas}`          | `config["num_candidates"]` — `config.py:12`                        | `6`                                    |

The resulting first-round user message is therefore the `IDEA_GENERATION_PROMPT` template with all placeholders filled as above — asking the LLM to propose **6 concrete, diverse improvement ideas** for the ResNet-18 / CIFAR-10 baseline (starting from 95.51 % accuracy), with no historical constraints yet loaded into the prompt.

### Additional Parameters

| Parameter      | Value                                     |
|----------------|-------------------------------------------|
| `model`        | `deepseek-chat` (default; override via `OPENAI_MODEL` env var) |
| `temperature`  | `0.7`                                     |
| `max_retries`  | `3` (with exponential back-off)           |

## Acknowledgement

Built on top of [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar). The closed-loop experimentation framework and LLM integration are original contributions.

## License

MIT
