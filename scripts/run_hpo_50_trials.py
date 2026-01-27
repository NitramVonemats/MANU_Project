#!/usr/bin/env python3
"""
Run HPO with 50 trials for all datasets and all algorithms.
This script runs all 6 algorithms × 6 datasets × 50 trials = 1800 total training runs.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Configuration
TRIALS = 50
EPOCHS = 50
PATIENCE = 12
SEED = 42

# Datasets (excluding clintox which has TDC bug)
DATASETS = [
    "Caco2_Wang",
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "tox21",
    "herg",
]

# All HPO algorithms
ALGORITHMS = ["random", "pso", "abc", "ga", "sa", "hc"]

def run_hpo(dataset: str, algo: str):
    """Run HPO for a single dataset-algorithm combination."""
    print(f"\n{'='*80}")
    print(f"🚀 Running: {dataset} with {algo.upper()} - {TRIALS} trials")
    print(f"{'='*80}\n")

    # Map algorithm names to module names
    algo_module_map = {
        "random": "random_search",
        "pso": "pso",
        "abc": "abc",
        "ga": "genetic",
        "sa": "simulated_annealing",
        "hc": "hill_climbing",
    }

    module_name = algo_module_map.get(algo, algo)

    cmd = [
        sys.executable, "-m", f"optimization.algorithms.{module_name}",
        "--dataset", dataset,
        "--trials", str(TRIALS),
        "--epochs", str(EPOCHS),
        "--patience", str(PATIENCE),
        "--seed", str(SEED),
    ]

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✅ Completed {dataset} - {algo} in {elapsed/60:.1f} minutes")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Failed {dataset} - {algo} after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False, elapsed
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted {dataset} - {algo}")
        return None, time.time() - start_time


def main():
    """Run all HPO combinations."""
    total_start = time.time()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create log file
    log_file = Path("hpo_50_trials.log")

    print(f"\n{'#'*80}")
    print(f"# HPO WITH 50 TRIALS - ALL DATASETS & ALGORITHMS")
    print(f"{'#'*80}")
    print(f"\nStarted: {start_datetime}")
    print(f"\nConfiguration:")
    print(f"  Trials: {TRIALS}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Patience: {PATIENCE}")
    print(f"  Datasets: {len(DATASETS)}")
    print(f"  Algorithms: {len(ALGORITHMS)}")
    print(f"\n  Total HPO runs: {len(DATASETS)} × {len(ALGORITHMS)} = {len(DATASETS) * len(ALGORITHMS)}")
    print(f"  Total training runs: {len(DATASETS) * len(ALGORITHMS) * TRIALS} = {len(DATASETS) * len(ALGORITHMS) * TRIALS}")

    results = []
    total_runs = len(DATASETS) * len(ALGORITHMS)
    current_run = 0

    for dataset in DATASETS:
        for algo in ALGORITHMS:
            current_run += 1
            print(f"\n\n{'='*80}")
            print(f"[{current_run}/{total_runs}] {dataset} - {algo.upper()}")
            print(f"{'='*80}")

            success, elapsed = run_hpo(dataset, algo)

            results.append({
                "dataset": dataset,
                "algorithm": algo,
                "success": success,
                "time_seconds": elapsed,
            })

            # Log progress
            with open(log_file, "a") as f:
                status = "SUCCESS" if success else "FAILED" if success is False else "INTERRUPTED"
                f.write(f"{datetime.now().isoformat()} | {dataset:30s} | {algo:8s} | {status:12s} | {elapsed/60:.1f}m\n")

            if success is None:  # Interrupted
                break

            # Brief pause
            time.sleep(2)

        if success is None:
            break

    # Final summary
    total_elapsed = time.time() - total_start
    end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n\n{'#'*80}")
    print("# FINAL SUMMARY")
    print(f"{'#'*80}")
    print(f"\nStarted: {start_datetime}")
    print(f"Ended:   {end_datetime}")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")

    successful = sum(1 for r in results if r["success"] is True)
    failed = sum(1 for r in results if r["success"] is False)
    interrupted = sum(1 for r in results if r["success"] is None)

    print(f"\nResults: ✅ {successful} successful | ❌ {failed} failed | ⚠️ {interrupted} interrupted")

    print(f"\n{'='*80}")
    print("Detailed Results:")
    print(f"{'='*80}")
    for r in results:
        status = "✅" if r["success"] else "❌" if r["success"] is False else "⚠️"
        print(f"  {status} {r['dataset']:30s} [{r['algorithm']:8s}] - {r['time_seconds']/60:.1f}m")

    # Save final summary
    with open(log_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"FINAL SUMMARY\n")
        f.write(f"Total time: {total_elapsed/60:.1f} minutes\n")
        f.write(f"Results: {successful} successful, {failed} failed, {interrupted} interrupted\n")
        f.write(f"{'='*80}\n")

    print(f"\n📊 Log saved to: {log_file}")


if __name__ == "__main__":
    main()
