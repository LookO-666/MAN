#!/usr/bin/env python3
"""
Baseline dispatcher for Closed-Loop AI Research experiments.

Usage:
    python run_baselines.py --strategy random
    python run_baselines.py --strategy zeroshot
    python run_baselines.py --strategy openloop
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call an OpenAI-compatible LLM API (DeepSeek / Qwen / GPT / etc.).

    Reads configuration from environment variables:
        OPENAI_API_KEY   – API key (required)
        OPENAI_BASE_URL  – Base URL, e.g. https://api.deepseek.com/v1
        OPENAI_MODEL     – Model name, e.g. deepseek-chat, qwen-plus
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("OPENAI_MODEL", "deepseek-chat")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Export it before running zeroshot / openloop strategies."
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_python_code(text: str) -> str:
    """Robustly extract a Python code block from LLM output.

    Tries, in order:
        1. ```python ... ```
        2. ``` ... ```
        3. Raw text (fallback)
    """
    # Try ```python ... ```
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ```
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: return everything (may fail, but caller handles errors)
    return text.strip()


# ---------------------------------------------------------------------------
# File backup / restore helpers
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_PATH = os.path.join(PROJECT_DIR, "models", "resnet.py")
RESNET_BACKUP = os.path.join(PROJECT_DIR, "models", "resnet_backup.py")


def backup_resnet():
    shutil.copy2(RESNET_PATH, RESNET_BACKUP)


def restore_resnet():
    if os.path.exists(RESNET_BACKUP):
        shutil.copy2(RESNET_BACKUP, RESNET_PATH)


def read_resnet() -> str:
    with open(RESNET_PATH, "r") as f:
        return f.read()


def write_resnet(code: str):
    with open(RESNET_PATH, "w") as f:
        f.write(code)


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def run_training(lr: float = 0.1, weight_decay: float = 5e-4,
                 epochs: int = 5, out_dir: str = "results/tmp") -> Optional[dict]:
    """Run main.py and return the parsed results.json, or None on failure."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "main.py",
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--epoch", str(epochs),
        "--out_dir", out_dir,
    ]
    try:
        subprocess.run(cmd, cwd=PROJECT_DIR, check=True, timeout=600)
        results_path = os.path.join(out_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                return json.load(f)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  [ERROR] Training failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Pretty logging
# ---------------------------------------------------------------------------

SEP = "=" * 60

def log_header(strategy: str):
    print(f"\n{SEP}")
    print(f"  Baseline Strategy: {strategy.upper()}")
    print(SEP)


def log_iter(iteration: int, total: int, extra: str = ""):
    print(f"\n--- Iteration {iteration}/{total} {extra}---")


def log_result(acc):
    if acc is not None:
        print(f"  >> Test Accuracy: {acc:.2f}%")
    else:
        print("  >> Test Accuracy: N/A (run failed)")


def log_best(best_acc):
    print(f"\n{SEP}")
    print(f"  Best Accuracy across all iterations: {best_acc:.2f}%")
    print(SEP)


# ---------------------------------------------------------------------------
# Strategy: random
# ---------------------------------------------------------------------------

def strategy_random():
    log_header("random")
    num_iters = 5
    best_acc = 0.0
    all_results = []

    for i in range(1, num_iters + 1):
        lr = round(random.uniform(0.001, 0.1), 5)
        wd = round(random.uniform(1e-4, 1e-3), 6)
        log_iter(i, num_iters, f"(lr={lr}, wd={wd}) ")

        out_dir = os.path.join(PROJECT_DIR, "results", "random", f"iter_{i}")
        result = run_training(lr=lr, weight_decay=wd, epochs=5, out_dir=out_dir)

        acc = result["best_test_acc"] if result else None
        log_result(acc)

        all_results.append({"iter": i, "lr": lr, "weight_decay": wd, "best_test_acc": acc})
        if acc is not None and acc > best_acc:
            best_acc = acc

    log_best(best_acc)

    summary_path = os.path.join(PROJECT_DIR, "results", "random", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"strategy": "random", "best_acc": best_acc, "runs": all_results}, f, indent=2)
    print(f"  Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Strategy: zeroshot
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert deep learning researcher. "
    "You will be given a ResNet implementation for CIFAR-10. "
    "Your task is to propose a single architectural modification "
    "(e.g., adding attention modules, changing activation functions, "
    "modifying skip connections, etc.) to improve test accuracy. "
    "Return the COMPLETE modified Python file inside a ```python``` code block. "
    "The file must be self-contained and keep the same public API "
    "(ResNet18, ResNet34, … functions must still exist)."
)


def strategy_zeroshot():
    log_header("zeroshot")
    backup_resnet()
    original_code = read_resnet()

    user_prompt = (
        "Here is the current models/resnet.py:\n\n"
        f"```python\n{original_code}\n```\n\n"
        "Please propose ONE architectural improvement to boost CIFAR-10 accuracy "
        "and return the full modified file."
    )

    print("  Calling LLM (single shot) ...")
    try:
        response = call_llm(SYSTEM_PROMPT, user_prompt)
        new_code = extract_python_code(response)
        write_resnet(new_code)
        print("  models/resnet.py overwritten with LLM code.")

        out_dir = os.path.join(PROJECT_DIR, "results", "zeroshot")
        result = run_training(epochs=5, out_dir=out_dir)
        acc = result["best_test_acc"] if result else None
        log_result(acc)

        summary = {"strategy": "zeroshot", "best_test_acc": acc}
        summary_path = os.path.join(out_dir, "summary.json")
        os.makedirs(out_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved to {summary_path}")

    except Exception as e:
        print(f"  [ERROR] zeroshot failed: {e}")
    finally:
        restore_resnet()
        print("  models/resnet.py restored to original.")


# ---------------------------------------------------------------------------
# Strategy: openloop
# ---------------------------------------------------------------------------

def strategy_openloop():
    log_header("openloop")
    backup_resnet()
    original_code = read_resnet()

    num_iters = 5
    best_acc = 0.0
    all_results = []

    # Pre-generate diverse idea prompts so LLM gets a different angle each time
    idea_angles = [
        "Add a channel attention mechanism (e.g., SE-block style) to each residual block.",
        "Replace ReLU activations with a more modern activation function like SiLU/Swish or GELU.",
        "Add a spatial attention module after each residual stage.",
        "Introduce dropout or stochastic depth regularization into the residual blocks.",
        "Modify the stem (first conv layer) and add a multi-scale feature fusion before the classifier.",
    ]

    for i in range(1, num_iters + 1):
        log_iter(i, num_iters, f"[openloop] ")

        user_prompt = (
            "Here is the baseline models/resnet.py:\n\n"
            f"```python\n{original_code}\n```\n\n"
            f"Idea direction: {idea_angles[i - 1]}\n\n"
            "Please implement this idea and return the COMPLETE modified file. "
            "Keep the same public API (ResNet18, ResNet34, … must still exist)."
        )

        try:
            print("  Calling LLM ...")
            response = call_llm(SYSTEM_PROMPT, user_prompt)
            new_code = extract_python_code(response)
            write_resnet(new_code)
            print("  models/resnet.py overwritten with LLM code.")

            out_dir = os.path.join(PROJECT_DIR, "results", "openloop", f"iter_{i}")
            result = run_training(epochs=5, out_dir=out_dir)
            acc = result["best_test_acc"] if result else None
            log_result(acc)

            all_results.append({"iter": i, "idea": idea_angles[i - 1], "best_test_acc": acc})
            if acc is not None and acc > best_acc:
                best_acc = acc

        except Exception as e:
            print(f"  [ERROR] Iteration {i} failed: {e}")
            all_results.append({"iter": i, "idea": idea_angles[i - 1], "best_test_acc": None})
        finally:
            restore_resnet()
            print("  models/resnet.py restored to original.")

    log_best(best_acc)

    summary_path = os.path.join(PROJECT_DIR, "results", "openloop", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"strategy": "openloop", "best_acc": best_acc, "runs": all_results}, f, indent=2)
    print(f"  Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STRATEGIES = {
    "random": strategy_random,
    "zeroshot": strategy_zeroshot,
    "openloop": strategy_openloop,
}


def main():
    parser = argparse.ArgumentParser(description="Run baseline strategies for Closed-Loop AI Research")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=list(STRATEGIES.keys()),
        help="Baseline strategy to run: random | zeroshot | openloop",
    )
    args = parser.parse_args()

    print(f"Project directory: {PROJECT_DIR}")
    print(f"Strategy selected: {args.strategy}")

    STRATEGIES[args.strategy]()


if __name__ == "__main__":
    main()
