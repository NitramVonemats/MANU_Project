#!/usr/bin/env python
"""Script to check model comparison progress"""
import json
import os
from pathlib import Path
import time

def find_latest_results():
    """Find the latest results directory"""
    reports_dir = Path("reports/model_comparison")
    if not reports_dir.exists():
        return None

    dirs = sorted([d for d in reports_dir.iterdir() if d.is_dir()], reverse=True)
    for d in dirs:
        results_file = d / "results.json"
        if results_file.exists():
            return results_file
    return None

def main():
    print("=" * 70)
    print("MODEL COMPARISON PROGRESS CHECKER")
    print("=" * 70)

    results_file = find_latest_results()
    if not results_file:
        print("No results found yet!")
        return

    print(f"Results directory: {results_file.parent.name}")
    print()

    with open(results_file) as f:
        results = json.load(f)

    total = 5670
    completed = len(results)
    percent = (completed / total) * 100

    print(f"Progress: {completed}/{total} ({percent:.2f}%)")

    if completed > 0:
        last = results[-1]
        print(f"\nLast completed experiment:")
        print(f"  Dataset: {last['dataset']}")
        print(f"  Model: {last['model_type']}")
        print(f"  Hyperparams: layers={last['hyperparameters']['num_layers']}, "
              f"hidden={last['hyperparameters']['hidden_dim']}, "
              f"lr={last['hyperparameters']['learning_rate']}, "
              f"dropout={last['hyperparameters']['dropout']}")
        print(f"  Head dims: {last['hyperparameters']['head_dims']}")
        print(f"  Training time: {last['train_time']:.1f}s")

        if last.get('is_classification'):
            print(f"  Test AUC: {last['test_metrics']['auc_roc']:.4f}")
        else:
            print(f"  Test RMSE: {last['test_metrics']['rmse']:.4f}")
            print(f"  Test R2: {last['test_metrics']['r2']:.4f}")

        # Estimate time remaining
        avg_time = sum(r['train_time'] for r in results) / len(results)
        remaining = total - completed
        est_time = remaining * avg_time
        hours = est_time / 3600
        print(f"\nEstimated time remaining: {hours:.1f} hours ({est_time/60:.1f} minutes)")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
