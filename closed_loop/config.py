"""Configuration for the closed-loop AI research iteration system."""

import os

CONFIG = {
    # Experiment setup
    "baseline_code_path": "../main.py",
    "baseline_results_path": "../results/baseline_200ep/results.json",
    "output_dir": "experiments",

    # Loop parameters
    "num_rounds": 5,
    "num_candidates": 6,
    "num_survivors": 2,
    "mini_epochs": 5,
    "full_epochs": 200,
    "seed": 42,

    # LLM settings
    "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "temperature": 0.7,

    # Method variant (for ablation)
    # "full"            = complete method (mini-exp screening + LLM analysis + failure memory)
    # "no_mini_exp"     = skip mini-experiment, LLM picks ideas directly
    # "no_llm_analysis" = skip LLM analysis, rank by val_acc only (degrades to Hyperband)
    # "no_memory"       = skip failure memory
    # "linear"          = 1 idea per round, run full experiment directly (simulates AI Scientist)
    # "no_feedback"     = each round independent, no history in prompt
    # "random"          = random search, no LLM
    # "single_shot"     = LLM gives one plan, run once, done
    "method": "full",
}
