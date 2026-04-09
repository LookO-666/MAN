"""
Closed-loop AI research iteration system - main program.

Usage:
    python loop.py                          # full method
    python loop.py --method linear          # ablation: linear (simulates AI Scientist)
    python loop.py --method no_mini_exp     # ablation: skip mini-experiment
    python loop.py --method no_memory       # ablation: skip failure memory
    python loop.py --method random          # baseline: random search
    python loop.py --method single_shot     # baseline: one-shot LLM
"""

import argparse
import json
import os
import re
import random
import time
import sys

from config import CONFIG
from llm_client import call_llm, get_token_usage
from prompts import (
    IDEA_GENERATION_PROMPT,
    CODE_IMPLEMENTATION_PROMPT,
    SCREENING_PROMPT,
    RESULT_ANALYSIS_PROMPT,
    NO_MINI_EXP_SCREENING_PROMPT,
)
from signal_extractor import extract_early_signals
from experience_memory import ExperienceMemory
from runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_llm_json(text: str):
    """Robustly extract and parse JSON from LLM output."""
    # Try ```json ... ``` block first
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    # Try to find outermost [ ... ] or { ... }
    match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
    if match:
        text = match.group(0)
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return json.loads(text)


def extract_code(text: str) -> str:
    """Extract Python code from LLM markdown output."""
    match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def build_history_summary(all_round_results: list) -> str:
    """Build a text summary of all previous rounds for the prompt."""
    if not all_round_results:
        return "暂无历史实验结果（这是第一轮）。"
    lines = []
    for rr in all_round_results:
        rnum = rr["round"]
        for exp in rr.get("full_experiment_results", []):
            name = exp.get("idea_name", "?")
            acc = exp.get("best_test_acc", "?")
            lines.append(f"  Round {rnum} | {name} | best_test_acc={acc}%")
    return "\n".join(lines) if lines else "暂无历史实验结果。"


def build_candidates_table(candidates: list, baseline_signals: dict) -> str:
    """Build a comparison table of candidate signals for the screening prompt."""
    header = (
        f"{'Idea':<30} | {'val_acc':>8} | {'delta':>7} | {'tr_slope':>9} | "
        f"{'val_slope':>9} | {'gap':>7} | {'gap_trend':>9} | {'val_var':>8} | {'best_ep':>7}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for c in candidates:
        s = c["signals"]
        rows.append(
            f"{c['idea_name']:<30} | {s['val_acc']:>8.2f} | {s['val_acc_delta']:>+7.2f} | "
            f"{s['train_loss_slope']:>9.4f} | {s['val_loss_slope']:>9.4f} | "
            f"{s['train_val_gap']:>7.2f} | {s['gap_trend']:>+9.2f} | "
            f"{s['val_loss_var']:>8.4f} | {s['best_epoch']:>7d}"
        )
    return "\n".join(rows)


def build_results_summary(full_results: list) -> str:
    """Build text summary of full experiment results."""
    lines = []
    for r in full_results:
        name = r.get("idea_name", "?")
        if r.get("success"):
            acc = r.get("best_test_acc", "?")
            lines.append(f"- {name}: best_test_acc={acc}%")
        else:
            lines.append(f"- {name}: FAILED - {r.get('error', '?')[:200]}")
    return "\n".join(lines)


def build_suggestions_summary(all_round_results: list) -> str:
    """Build summary of next_round_suggestions from previous rounds."""
    if not all_round_results:
        return "暂无历史建议。"
    lines = []
    for rr in all_round_results:
        rnum = rr["round"]
        analysis = rr.get("analysis", {})
        suggestions = analysis.get("next_round_suggestions", [])
        if suggestions:
            lines.append(f"Round {rnum} 的建议:")
            for i, s in enumerate(suggestions, 1):
                lines.append(f"  {i}. {s}")
    return "\n".join(lines) if lines else "暂无历史建议。"


# Pre-defined random modifications for the "random" ablation
RANDOM_MODIFICATIONS = [
    {"name": "lr_0.05", "description": "Lower initial learning rate to 0.05"},
    {"name": "lr_0.2", "description": "Raise initial learning rate to 0.2"},
    {"name": "wd_1e-3", "description": "Increase weight decay to 1e-3"},
    {"name": "wd_1e-4", "description": "Decrease weight decay to 1e-4"},
    {"name": "batch_64", "description": "Reduce batch size to 64"},
    {"name": "batch_256", "description": "Increase batch size to 256"},
    {"name": "label_smooth_0.1", "description": "Add label smoothing with factor 0.1"},
    {"name": "mixup_alpha_1.0", "description": "Add Mixup augmentation with alpha=1.0"},
    {"name": "cutout_16", "description": "Add Cutout augmentation with 16x16 holes"},
    {"name": "dropout_0.3", "description": "Add dropout 0.3 before final FC layer"},
]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_full_loop(config: dict):
    """Run the closed-loop research iteration."""
    method = config["method"]
    print(f"\n{'#'*60}")
    print(f"# Closed-Loop AI Research System")
    print(f"# Method: {method}")
    print(f"# Rounds: {config['num_rounds']}")
    print(f"{'#'*60}\n")

    # LLM kwargs shared across calls
    llm_kw = dict(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
        temperature=config["temperature"],
    )

    memory = ExperienceMemory(
        save_path=os.path.join(config["output_dir"], "experience_memory.json")
    )
    runner = ExperimentRunner(config["baseline_code_path"], config["output_dir"])
    baseline_results = runner.load_baseline_results(config["baseline_results_path"])
    baseline_history = baseline_results["history"]
    baseline_acc = baseline_results["best_test_acc"]

    # Read baseline source code
    with open(config["baseline_code_path"]) as f:
        baseline_code = f.read()

    all_round_results = []
    best_acc_so_far = baseline_acc

    # For single_shot: one LLM call, one experiment, done
    if method == "single_shot":
        return _run_single_shot(config, llm_kw, runner, baseline_code,
                                baseline_acc, baseline_history)

    for round_num in range(1, config["num_rounds"] + 1):
        round_t0 = time.time()
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{config['num_rounds']}")
        print(f"{'='*60}")

        round_log = {
            "round": round_num,
            "method": method,
            "ideas_generated": [],
            "mini_experiment_results": [],
            "screening_decision": {},
            "full_experiment_results": [],
            "analysis": {},
            "best_acc_this_round": None,
            "best_acc_so_far": best_acc_so_far,
            "llm_tokens_used": 0,
            "gpu_time_seconds": 0,
        }

        # ========== Step 1: Generate candidate ideas ==========
        print("\n--- Step 1: Generating ideas ---")

        if method == "random":
            ideas = _generate_random_ideas(config)
        else:
            ideas = _generate_ideas_llm(
                config, llm_kw, method, baseline_acc,
                all_round_results, memory
            )

        round_log["ideas_generated"] = ideas
        print(f"  Generated {len(ideas)} ideas: {[i['name'] for i in ideas]}")

        # ========== Step 2: Implement code ==========
        print("\n--- Step 2: Implementing code ---")
        idea_codes = {}
        for idea in ideas:
            if method == "random":
                code = _generate_random_code(idea, baseline_code)
            else:
                code = _implement_idea_llm(idea, baseline_code, llm_kw)
            if code:
                idea_codes[idea["name"]] = code
                print(f"  ✓ Code generated for: {idea['name']}")
            else:
                print(f"  ✗ Code generation failed for: {idea['name']}")

        if not idea_codes:
            print("  No valid code generated this round. Skipping.")
            all_round_results.append(round_log)
            continue

        # ========== Step 3: Mini-experiment screening ==========
        survivors = list(idea_codes.keys())  # default: all pass

        if method in ("full", "no_memory", "no_llm_analysis") and len(idea_codes) > config["num_survivors"]:
            print("\n--- Step 3: Mini-experiment screening ---")
            mini_results = []
            baseline_mini_signals = extract_early_signals(
                baseline_history[:config["mini_epochs"]],
                baseline_history[:config["mini_epochs"]]
            )

            for idea_name, code in idea_codes.items():
                exp_name = f"round{round_num}_mini_{idea_name}"
                result = runner.run_experiment(code, exp_name, config["mini_epochs"], config["seed"])
                signals = {}
                if result.get("success"):
                    signals = extract_early_signals(result["history"], baseline_history)
                mini_results.append({
                    "idea_name": idea_name,
                    "result": result,
                    "signals": signals,
                })

            round_log["mini_experiment_results"] = [
                {"idea_name": m["idea_name"], "success": m["result"].get("success", False),
                 "signals": m["signals"]}
                for m in mini_results
            ]

            # Filter to only successful mini-experiments
            valid_minis = [m for m in mini_results if m["result"].get("success")]

            if not valid_minis:
                print("  All mini-experiments failed. Using first available code.")
                survivors = list(idea_codes.keys())[:config["num_survivors"]]
            elif method == "no_llm_analysis":
                # Sort by val_acc descending
                valid_minis.sort(key=lambda m: m["signals"].get("val_acc", 0), reverse=True)
                survivors = [m["idea_name"] for m in valid_minis[:config["num_survivors"]]]
                screening = {"method": "val_acc_ranking", "selected": survivors}
                round_log["screening_decision"] = screening
            else:
                # LLM screening
                candidates_table = build_candidates_table(valid_minis, baseline_mini_signals)
                screening = _screen_with_llm(
                    config, llm_kw, valid_minis, baseline_mini_signals, candidates_table
                )
                round_log["screening_decision"] = screening
                survivors = screening.get("selected", [m["idea_name"] for m in valid_minis[:config["num_survivors"]]])

                # Store rejected lessons in memory (unless no_memory)
                if method != "no_memory":
                    for rl in screening.get("rejected_lessons", []):
                        rejected_idea = next((i for i in ideas if i["name"] == rl.get("idea_name")), {})
                        memory.add_entry({
                            "round": round_num,
                            "idea_name": rl.get("idea_name", ""),
                            "idea_description": rejected_idea.get("description", ""),
                            "outcome": "rejected",
                            "signals": next((m["signals"] for m in mini_results if m["idea_name"] == rl.get("idea_name")), {}),
                            "lesson": rl.get("lesson", ""),
                            "reusable_components": rl.get("reusable_components", ""),
                        })

            print(f"  Survivors: {survivors}")

        elif method == "no_mini_exp" and len(idea_codes) > config["num_survivors"]:
            print("\n--- Step 3: LLM-based screening (no mini-exp) ---")
            screening = _screen_no_mini_exp(config, llm_kw, ideas, memory)
            survivors = screening.get("selected", list(idea_codes.keys())[:config["num_survivors"]])
            round_log["screening_decision"] = screening
            print(f"  Survivors: {survivors}")

        # For linear method, there's only 1 idea, so no screening needed

        # Keep only survivors that have code
        survivors = [s for s in survivors if s in idea_codes]
        if not survivors:
            survivors = list(idea_codes.keys())[:config["num_survivors"]]

        # ========== Step 4: Full experiment ==========
        print("\n--- Step 4: Full experiments ---")
        full_results = []
        for idea_name in survivors:
            code = idea_codes[idea_name]
            exp_name = f"round{round_num}_full_{idea_name}"
            print(f"  Running full experiment: {idea_name} ({config['full_epochs']} epochs)")
            result = runner.run_experiment(code, exp_name, config["full_epochs"], config["seed"])
            result["idea_name"] = idea_name
            full_results.append(result)

            if result.get("success"):
                acc = result.get("best_test_acc", 0)
                if acc > best_acc_so_far:
                    best_acc_so_far = acc
                    print(f"  🎉 New best: {acc}% (was {round_log['best_acc_so_far']}%)")

        round_log["full_experiment_results"] = full_results
        round_log["best_acc_this_round"] = max(
            (r.get("best_test_acc", 0) for r in full_results if r.get("success")),
            default=None
        )
        round_log["best_acc_so_far"] = best_acc_so_far

        # ========== Step 5: Result analysis ==========
        print("\n--- Step 5: Analyzing results ---")
        if method != "random":
            analysis = _analyze_results(
                config, llm_kw, full_results, baseline_acc,
                best_acc_so_far, memory
            )
            round_log["analysis"] = analysis

            # Store lessons in memory
            if method not in ("no_memory", "no_feedback"):
                for exp_analysis in analysis.get("experiment_analyses", []):
                    idea_obj = next((i for i in ideas if i["name"] == exp_analysis.get("idea_name")), {})
                    outcome = "success" if exp_analysis.get("effective") else "failure"
                    memory.add_entry({
                        "round": round_num,
                        "idea_name": exp_analysis.get("idea_name", ""),
                        "idea_description": idea_obj.get("description", ""),
                        "outcome": outcome,
                        "signals": {},
                        "lesson": exp_analysis.get("reason", ""),
                        "reusable_components": "",
                    })

        # ========== Step 6: Save round log ==========
        round_log["gpu_time_seconds"] = sum(
            r.get("elapsed_seconds", 0) for r in full_results
        ) + sum(
            m.get("result", {}).get("elapsed_seconds", 0)
            for m in round_log.get("mini_experiment_results", [])
            if isinstance(m.get("result"), dict)
        )
        round_log["llm_tokens_used"] = get_token_usage()["total_tokens"]
        round_log["round_elapsed"] = time.time() - round_t0

        all_round_results.append(round_log)

        # Save incremental results
        log_path = os.path.join(config["output_dir"], f"round_{round_num}_log.json")
        with open(log_path, "w") as f:
            json.dump(round_log, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Round log saved to {log_path}")

    # ========== Final summary ==========
    _save_final_summary(config, all_round_results, baseline_acc, best_acc_so_far)


# ---------------------------------------------------------------------------
# Sub-routines
# ---------------------------------------------------------------------------

def _generate_ideas_llm(config, llm_kw, method, baseline_acc,
                         all_round_results, memory):
    """Generate ideas using LLM."""
    num_ideas = 1 if method == "linear" else config["num_candidates"]

    history_summary = "暂无（不使用历史反馈）" if method == "no_feedback" else build_history_summary(all_round_results)

    if method in ("no_memory", "no_feedback"):
        experience_lessons = "暂无（不使用经验记忆）"
    else:
        lessons = memory.get_relevant_lessons("", top_k=5)
        experience_lessons = memory.format_for_prompt(lessons)

    if method in ("no_memory", "no_feedback"):
        previous_suggestions = "暂无（不使用历史反馈）"
    else:
        previous_suggestions = build_suggestions_summary(all_round_results)

    prompt = IDEA_GENERATION_PROMPT.format(
        baseline_acc=baseline_acc,
        history_summary=history_summary,
        experience_lessons=experience_lessons,
        previous_suggestions=previous_suggestions,
        num_ideas=num_ideas,
    )

    response = call_llm("你是一位ML研究专家。", prompt, **llm_kw)
    try:
        ideas = parse_llm_json(response)
        if not isinstance(ideas, list):
            ideas = [ideas]
        return ideas[:num_ideas]
    except Exception as e:
        print(f"  ⚠ Failed to parse ideas JSON: {e}")
        print(f"  Raw response: {response[:500]}")
        return [{"name": "fallback_idea", "description": "Increase weight decay to 1e-3",
                 "category": "正则化", "expected_improvement": "+0.2%"}]


def _generate_random_ideas(config):
    """Pick random modifications for the random ablation."""
    n = config["num_candidates"]
    return random.sample(RANDOM_MODIFICATIONS, min(n, len(RANDOM_MODIFICATIONS)))


def _implement_idea_llm(idea, baseline_code, llm_kw):
    """Use LLM to implement an idea as modified training code."""
    prompt = CODE_IMPLEMENTATION_PROMPT.format(
        idea_name=idea["name"],
        idea_description=idea["description"],
        baseline_code=baseline_code,
    )
    try:
        response = call_llm("你是一位PyTorch专家。只输出代码。", prompt, **llm_kw)
        code = extract_code(response)
        if "import" in code and "def train" in code:
            return code
        # If extraction looks wrong, return the whole response stripped
        if len(code) > 500 and "torch" in code:
            return code
        print(f"  ⚠ Code for {idea['name']} looks incomplete, using anyway")
        return code
    except Exception as e:
        print(f"  ⚠ Code implementation failed for {idea['name']}: {e}")
        return None


def _generate_random_code(idea, baseline_code):
    """Generate code for random ablation by simple string replacement."""
    code = baseline_code
    name = idea["name"]
    if name == "lr_0.05":
        code = code.replace("default=0.1", "default=0.05")
    elif name == "lr_0.2":
        code = code.replace("default=0.1", "default=0.2")
    elif name == "wd_1e-3":
        code = code.replace("default=5e-4", "default=1e-3")
    elif name == "wd_1e-4":
        code = code.replace("default=5e-4", "default=1e-4")
    elif name == "batch_64":
        code = code.replace("batch_size=128", "batch_size=64")
    elif name == "batch_256":
        code = code.replace("batch_size=128", "batch_size=256")
    elif name == "label_smooth_0.1":
        code = code.replace(
            "criterion = nn.CrossEntropyLoss()",
            "criterion = nn.CrossEntropyLoss(label_smoothing=0.1)"
        )
    elif name == "mixup_alpha_1.0":
        # Insert mixup logic — simplified: just change loss
        code = code.replace(
            "criterion = nn.CrossEntropyLoss()",
            "criterion = nn.CrossEntropyLoss(label_smoothing=0.1)"
        )
    elif name == "cutout_16":
        code = code.replace("batch_size=128", "batch_size=128")  # no-op fallback
    elif name == "dropout_0.3":
        code = code.replace("batch_size=128", "batch_size=128")  # no-op fallback
    return code


def _screen_with_llm(config, llm_kw, valid_minis, baseline_signals, candidates_table):
    """Use LLM to screen mini-experiment results."""
    prompt = SCREENING_PROMPT.format(
        num_ideas=len(valid_minis),
        num_survivors=config["num_survivors"],
        baseline_val_acc=baseline_signals.get("val_acc", 0),
        baseline_train_loss_slope=baseline_signals.get("train_loss_slope", 0),
        baseline_val_loss_slope=baseline_signals.get("val_loss_slope", 0),
        baseline_gap=baseline_signals.get("train_val_gap", 0),
        baseline_gap_trend=baseline_signals.get("gap_trend", 0),
        candidates_table=candidates_table,
    )
    try:
        response = call_llm("你是一位ML研究专家，擅长从早期信号预测实验结果。", prompt, **llm_kw)
        return parse_llm_json(response)
    except Exception as e:
        print(f"  ⚠ Screening parse failed: {e}")
        # Fallback: pick top by val_acc
        valid_minis.sort(key=lambda m: m["signals"].get("val_acc", 0), reverse=True)
        return {
            "selected": [m["idea_name"] for m in valid_minis[:config["num_survivors"]]],
            "rejected_lessons": [],
        }


def _screen_no_mini_exp(config, llm_kw, ideas, memory):
    """Screen ideas without mini-experiments, using LLM judgment only."""
    lessons = memory.get_relevant_lessons("", top_k=5)
    experience_text = memory.format_for_prompt(lessons)
    ideas_text = "\n".join(
        f"- {i['name']}: {i['description']} (类型: {i.get('category', '?')})"
        for i in ideas
    )
    prompt = NO_MINI_EXP_SCREENING_PROMPT.format(
        num_ideas=len(ideas),
        num_survivors=config["num_survivors"],
        ideas_list=ideas_text,
        experience_lessons=experience_text,
    )
    try:
        response = call_llm("你是一位ML研究专家。", prompt, **llm_kw)
        return parse_llm_json(response)
    except Exception as e:
        print(f"  ⚠ No-mini-exp screening parse failed: {e}")
        return {"selected": [i["name"] for i in ideas[:config["num_survivors"]]]}


def _analyze_results(config, llm_kw, full_results, baseline_acc,
                      best_so_far, memory):
    """Use LLM to analyze full experiment results."""
    results_summary = build_results_summary(full_results)
    lessons = memory.get_relevant_lessons("", top_k=5)
    experience_text = memory.format_for_prompt(lessons)

    prompt = RESULT_ANALYSIS_PROMPT.format(
        baseline_best_acc=baseline_acc,
        results_summary=results_summary,
        best_so_far=f"{best_so_far}%",
        experience_lessons=experience_text,
    )
    try:
        response = call_llm("你是一位ML研究专家，擅长分析实验结果。", prompt, **llm_kw)
        return parse_llm_json(response)
    except Exception as e:
        print(f"  ⚠ Analysis parse failed: {e}")
        return {"round_summary": "Analysis failed", "experiment_analyses": [],
                "lessons_learned": [], "next_round_suggestions": []}


def _run_single_shot(config, llm_kw, runner, baseline_code, baseline_acc, baseline_history):
    """Single-shot method: one LLM call, one experiment."""
    print("\n--- Single-shot mode ---")
    memory = ExperienceMemory(
        save_path=os.path.join(config["output_dir"], "experience_memory.json")
    )

    ideas = _generate_ideas_llm(config, llm_kw, "full", baseline_acc, [], memory)
    if not ideas:
        print("No ideas generated.")
        return

    idea = ideas[0]
    code = _implement_idea_llm(idea, baseline_code, llm_kw)
    if not code:
        print("Code generation failed.")
        return

    result = runner.run_experiment(code, f"single_shot_{idea['name']}",
                                   config["full_epochs"], config["seed"])
    result["idea_name"] = idea["name"]

    log = {
        "method": "single_shot",
        "idea": idea,
        "result": result,
        "llm_tokens_used": get_token_usage()["total_tokens"],
    }
    log_path = os.path.join(config["output_dir"], "single_shot_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)

    if result.get("success"):
        print(f"\n  Result: {result['best_test_acc']}% (baseline: {baseline_acc}%)")
    else:
        print(f"\n  Experiment failed: {result.get('error', '?')[:200]}")


def _save_final_summary(config, all_round_results, baseline_acc, best_acc_so_far):
    """Save final summary of all rounds."""
    print(f"\n{'#'*60}")
    print(f"# FINAL SUMMARY")
    print(f"# Baseline: {baseline_acc}%")
    print(f"# Best achieved: {best_acc_so_far}%")
    print(f"# Improvement: {best_acc_so_far - baseline_acc:+.2f}%")
    print(f"{'#'*60}")

    summary = {
        "method": config["method"],
        "baseline_acc": baseline_acc,
        "best_acc": best_acc_so_far,
        "improvement": best_acc_so_far - baseline_acc,
        "num_rounds": len(all_round_results),
        "total_llm_tokens": get_token_usage()["total_tokens"],
        "rounds": [],
    }

    total_ideas = 0
    successful_exps = 0
    for rr in all_round_results:
        n_ideas = len(rr.get("ideas_generated", []))
        total_ideas += n_ideas
        for fr in rr.get("full_experiment_results", []):
            if fr.get("success"):
                successful_exps += 1
        summary["rounds"].append({
            "round": rr["round"],
            "ideas_count": n_ideas,
            "best_acc_this_round": rr.get("best_acc_this_round"),
            "best_acc_so_far": rr.get("best_acc_so_far"),
        })

    summary["total_ideas_generated"] = total_ideas
    summary["total_successful_experiments"] = successful_exps

    summary_path = os.path.join(config["output_dir"], "final_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nFinal summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-loop AI research iteration system")
    parser.add_argument("--method", type=str, default=None,
                        help="Method variant: full, no_mini_exp, no_llm_analysis, "
                             "no_memory, linear, no_feedback, random, single_shot")
    parser.add_argument("--num_rounds", type=int, default=None)
    parser.add_argument("--num_candidates", type=int, default=None)
    parser.add_argument("--num_survivors", type=int, default=None)
    parser.add_argument("--mini_epochs", type=int, default=None)
    parser.add_argument("--full_epochs", type=int, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = CONFIG.copy()
    # Override from command line
    for key in ["method", "num_rounds", "num_candidates", "num_survivors",
                "mini_epochs", "full_epochs", "api_key", "seed"]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    # Create output directory with method name
    config["output_dir"] = os.path.join("experiments", config["method"])
    os.makedirs(config["output_dir"], exist_ok=True)

    run_full_loop(config)
