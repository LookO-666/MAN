"""
Unified experiment orchestration script for the Neuro-Symbolic AI Scientist system.

Supports baselines, ablations, and the full system via a single entry point.

Usage examples:
    # Full system, 3 independent runs, 5 rounds each
    python run_experiments.py --mode full_system --runs 3 --rounds 5

    # Random search baseline
    python run_experiments.py --mode random_search --runs 5 --rounds 5

    # Single-turn baseline (one-shot LLM)
    python run_experiments.py --mode single_turn --runs 5

    # Zero-memory baseline (LLM without history context)
    python run_experiments.py --mode zero_memory --runs 3 --rounds 5

    # Ablation: disable Structured Early Feedback (SEF)
    python run_experiments.py --mode full_system --disable_sef --runs 3 --rounds 5

    # Ablation: disable Component-level Failure Memory (CFM)
    python run_experiments.py --mode full_system --disable_cfm --runs 3 --rounds 5
"""

import argparse
import os
import sys
import time
import traceback

# Ensure the closed_loop package root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from loop import run_full_loop
from llm_client import _reset_token_usage  # we will add this helper


# ---------------------------------------------------------------------------
# Mode → config mapping
# ---------------------------------------------------------------------------

MODE_TO_METHOD = {
    "random_search": "random",
    "single_turn": "single_shot",
    "zero_memory": "no_memory",
    "full_system": "full",
}


def build_config(args) -> dict:
    """Build a runtime config dict from CLI args, layered on top of CONFIG."""
    config = CONFIG.copy()

    # --- Mode mapping ---
    method = MODE_TO_METHOD[args.mode]
    config["method"] = method

    # --- Rounds override ---
    config["num_rounds"] = args.rounds

    # --- Single-turn special handling ---
    if args.mode == "single_turn":
        config["num_rounds"] = 1  # forced

    # --- Zero-memory: inject physical cutoff flag ---
    if args.mode == "zero_memory":
        config["force_empty_memory"] = True

    # --- Ablation switches ---
    config["disable_sef"] = args.disable_sef
    config["disable_cfm"] = args.disable_cfm

    # --- API key override ---
    if args.api_key:
        config["api_key"] = args.api_key

    return config


def run_single(config: dict, run_id: int, total_runs: int):
    """Execute one complete run of the experiment loop."""
    # Per-run output directory
    mode_tag = config["method"]
    if config.get("disable_sef"):
        mode_tag += "_no_sef"
    if config.get("disable_cfm"):
        mode_tag += "_no_cfm"

    if total_runs > 1:
        out_dir = os.path.join("experiments", f"{mode_tag}_run{run_id}")
    else:
        out_dir = os.path.join("experiments", mode_tag)

    config["output_dir"] = out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Inject run_id into config so the CSV logger can reference it
    config["run_id"] = run_id

    print(f"\n{'#'*70}")
    print(f"# RUN {run_id}/{total_runs}  |  mode={config['method']}  |  rounds={config['num_rounds']}")
    print(f"# disable_sef={config.get('disable_sef', False)}  disable_cfm={config.get('disable_cfm', False)}")
    print(f"# output_dir={out_dir}")
    print(f"{'#'*70}\n")

    run_full_loop(config)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified experiment orchestrator for paper data collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["random_search", "single_turn", "zero_memory", "full_system"],
        help="Experiment mode / baseline variant.",
    )
    parser.add_argument(
        "--disable_sef",
        action="store_true",
        help="Ablation: disable Structured Early Feedback (8-dim signal extraction).",
    )
    parser.add_argument(
        "--disable_cfm",
        action="store_true",
        help="Ablation: disable Component-level Failure Memory (degrade to flat text).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent runs (for averaging). Default: 1.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of rounds per run (overrides config). Default: 5.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Override LLM API key.",
    )

    args = parser.parse_args()
    config_template = build_config(args)

    total_t0 = time.time()
    for run_id in range(1, args.runs + 1):
        config = config_template.copy()
        try:
            # Reset LLM token counter between runs
            try:
                from llm_client import _reset_token_usage
                _reset_token_usage()
            except ImportError:
                pass

            run_single(config, run_id, args.runs)

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user. Exiting gracefully.")
            sys.exit(1)
        except Exception:
            print(f"\n💥 RUN {run_id} CRASHED:")
            traceback.print_exc()
            print("Continuing to next run...\n")
            continue

    elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"All {args.runs} run(s) completed in {elapsed:.0f}s ({elapsed/60:.1f} min).")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
