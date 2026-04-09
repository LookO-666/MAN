"""Experiment runner: execute training scripts and collect results."""

import subprocess
import json
import os
import shutil
import time


class ExperimentRunner:
    def __init__(self, baseline_code_path: str, output_base_dir: str):
        self.baseline_code_path = os.path.abspath(baseline_code_path)
        self.output_base_dir = os.path.abspath(output_base_dir)
        # Locate project root (where models/ and utils.py live)
        self.project_root = os.path.dirname(self.baseline_code_path)
        os.makedirs(self.output_base_dir, exist_ok=True)

    def run_experiment(self, modified_code: str, experiment_name: str,
                       epochs: int, seed: int = 42) -> dict:
        """
        Run a training experiment with modified code.

        1. Create experiment directory
        2. Write modified code as train.py
        3. Copy models/ and utils.py into the directory
        4. Symlink data/ to avoid copying dataset
        5. Execute with subprocess
        6. Return results dict or error info
        """
        exp_dir = os.path.join(self.output_base_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)

        # Write the modified training script
        train_path = os.path.join(exp_dir, "train.py")
        with open(train_path, "w") as f:
            f.write(modified_code)

        # Copy models/ directory
        dst_models = os.path.join(exp_dir, "models")
        if os.path.exists(dst_models):
            shutil.rmtree(dst_models)
        shutil.copytree(os.path.join(self.project_root, "models"), dst_models)

        # Copy utils.py
        shutil.copy2(os.path.join(self.project_root, "utils.py"),
                      os.path.join(exp_dir, "utils.py"))

        # Symlink data/ to avoid copying the dataset
        data_link = os.path.join(exp_dir, "data")
        data_src = os.path.join(self.project_root, "data")
        if not os.path.exists(data_link):
            os.symlink(data_src, data_link)

        # Determine timeout: mini-exp 5min, full-exp 90min
        timeout = 300 if epochs <= 10 else 5400

        cmd = [
            "python3", "train.py",
            f"--epoch={epochs}",
            f"--out_dir=.",
        ]

        print(f"  [Runner] Executing: {' '.join(cmd)} in {exp_dir}")
        print(f"  [Runner] Timeout: {timeout}s")

        t0 = time.time()
        try:
            # Use Popen for real-time stdout streaming
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=exp_dir,
                bufsize=1,  # line-buffered
            )

            stderr_lines = []
            # Stream stdout in real-time, collect stderr
            import threading

            def _read_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line)

            t = threading.Thread(target=_read_stderr, daemon=True)
            t.start()

            for line in proc.stdout:
                print(f"    {line}", end="", flush=True)

            proc.wait(timeout=timeout)
            t.join(timeout=5)
            elapsed = time.time() - t0

            if proc.returncode != 0:
                error_msg = "".join(stderr_lines)[-1000:] or "Unknown error"
                print(f"  [Runner] FAILED ({elapsed:.0f}s): {error_msg[:200]}")
                return {
                    "success": False,
                    "error": error_msg,
                    "elapsed_seconds": elapsed,
                    "experiment_name": experiment_name,
                }

            # Read results.json
            results_path = os.path.join(exp_dir, "results.json")
            if not os.path.exists(results_path):
                return {
                    "success": False,
                    "error": "results.json not found after training",
                    "elapsed_seconds": elapsed,
                    "experiment_name": experiment_name,
                }

            with open(results_path) as f:
                data = json.load(f)

            data["success"] = True
            data["elapsed_seconds"] = elapsed
            data["experiment_name"] = experiment_name
            print(f"  [Runner] OK ({elapsed:.0f}s) best_test_acc={data.get('best_test_acc', '?')}%")
            return data

        except subprocess.TimeoutExpired:
            proc.kill()
            elapsed = time.time() - t0
            print(f"  [Runner] TIMEOUT after {elapsed:.0f}s")
            return {
                "success": False,
                "error": f"Experiment timed out after {timeout}s",
                "elapsed_seconds": elapsed,
                "experiment_name": experiment_name,
            }
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [Runner] EXCEPTION: {e}")
            return {
                "success": False,
                "error": str(e),
                "elapsed_seconds": elapsed,
                "experiment_name": experiment_name,
            }

    def load_baseline_results(self, path: str) -> dict:
        """Load baseline results.json."""
        abs_path = os.path.join(os.path.dirname(self.baseline_code_path), "..",
                                "pytorch-cifar", path.lstrip("../"))
        # Try the path as-is first (relative to closed_loop dir)
        for candidate in [path, os.path.abspath(path)]:
            if os.path.exists(candidate):
                with open(candidate) as f:
                    return json.load(f)
        raise FileNotFoundError(f"Baseline results not found at {path}")
