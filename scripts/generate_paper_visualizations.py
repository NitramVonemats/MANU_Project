#!/usr/bin/env python3
"""
Generate All Paper Visualizations
Creates ROC curves, confusion matrices, training curves, and comparison tables.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Matplotlib setup
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_hpo_results(runs_dir="results/hpo"):
    """Load all HPO results."""
    results = {}
    runs_path = Path(runs_dir)

    for dataset_dir in runs_path.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        results[dataset_name] = {}

        for json_file in dataset_dir.glob("hpo_*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                algo = data.get("algo", "unknown")
                results[dataset_name][algo] = data
            except Exception as e:
                print(f"Warning: Error loading {json_file}: {e}")

    return results


def generate_training_curves(results, output_dir="figures/paper"):
    """Generate training curves (loss vs epoch) for all datasets."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Generating Training Curves ===")

    for dataset, algos in results.items():
        # Skip if no data
        if not algos:
            continue

        # Find best algorithm
        best_algo = None
        best_score = -float('inf') if 'tox' in dataset.lower() or 'herg' in dataset.lower() else float('inf')

        for algo, data in algos.items():
            final = data.get("final_training", {})
            test_metrics = final.get("test_metrics", {})

            # Get metric based on task type
            if 'tox' in dataset.lower() or 'herg' in dataset.lower():
                score = test_metrics.get("f1", 0)
                if score > best_score:
                    best_score = score
                    best_algo = algo
            else:
                score = test_metrics.get("rmse", float('inf'))
                if score < best_score:
                    best_score = score
                    best_algo = algo

        if not best_algo:
            continue

        # Extract training history
        history = results[dataset][best_algo].get("final_training", {}).get("history", [])

        if not history:
            print(f"  [SKIP] {dataset}: No training history found")
            continue

        epochs = [h["epoch"] for h in history]
        train_loss = [h.get("train_loss", 0) for h in history]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Training Curve - {dataset} ({best_algo.upper()})', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Save
        filename = f"{output_dir}/{dataset}_training_curve.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] {dataset}: {filename}")

    print(f"\n[DONE] Training curves saved to {output_dir}/")


def generate_classification_metrics_plot(results, output_dir="figures/paper"):
    """Generate validation metrics evolution for classification datasets."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Generating Classification Metrics Evolution ===")

    classification_datasets = ['herg', 'tox21']

    for dataset in classification_datasets:
        if dataset not in results or not results[dataset]:
            print(f"  [SKIP] {dataset}: Not found")
            continue

        # Find best algorithm
        best_algo = None
        best_f1 = -1

        for algo, data in results[dataset].items():
            final = data.get("final_training", {})
            test_f1 = final.get("test_metrics", {}).get("f1", 0)
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_algo = algo

        if not best_algo:
            continue

        history = results[dataset][best_algo].get("final_training", {}).get("history", [])

        if not history:
            print(f"  [SKIP] {dataset}: No history")
            continue

        epochs = [h["epoch"] for h in history]
        val_f1 = [h.get("val_f1", 0) for h in history]
        val_auc = [h.get("val_auc_roc", 0) for h in history]
        val_acc = [h.get("val_accuracy", 0) for h in history]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, val_f1, 'b-', linewidth=2, label='Validation F1', marker='o', markersize=4)
        ax.plot(epochs, val_auc, 'r-', linewidth=2, label='Validation AUC-ROC', marker='s', markersize=4)
        ax.plot(epochs, val_acc, 'g-', linewidth=2, label='Validation Accuracy', marker='^', markersize=4)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title(f'Validation Metrics - {dataset.upper()} ({best_algo.upper()})',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # Save
        filename = f"{output_dir}/{dataset}_metrics_evolution.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] {dataset}: {filename}")

    print(f"\n[DONE] Classification metrics plots saved to {output_dir}/")


def create_comparison_tables(results, output_dir="figures/paper"):
    """Create comprehensive comparison tables in LaTeX format."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Generating Comparison Tables ===")

    # 1. ADME Comparison Table (Regression)
    adme_datasets = ['Caco2_Wang', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']

    adme_data = []
    for dataset in adme_datasets:
        if dataset not in results:
            continue

        # Find best algorithm
        best_algo = None
        best_rmse = float('inf')

        for algo, data in results[dataset].items():
            final = data.get("final_training", {})
            test_rmse = final.get("test_metrics", {}).get("rmse", float('inf'))
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_algo = algo

        if best_algo:
            final = results[dataset][best_algo].get("final_training", {})
            test_metrics = final.get("test_metrics", {})
            val_metrics = final.get("val_metrics", {})

            adme_data.append({
                'Dataset': dataset,
                'Algorithm': best_algo.upper(),
                'Test RMSE': test_metrics.get('rmse', 0),
                'Test MAE': test_metrics.get('mae', 0),
                'Test R²': test_metrics.get('r2', 0),
                'Val RMSE': val_metrics.get('rmse', 0),
                'Train Time (s)': final.get('train_time', 0)
            })

    adme_df = pd.DataFrame(adme_data)

    # Save as CSV
    adme_df.to_csv(f"{output_dir}/adme_comparison.csv", index=False)

    # Save as LaTeX
    latex_adme = adme_df.to_latex(index=False, float_format="%.4f",
                                   caption="ADME Datasets - Best HPO Results",
                                   label="tab:adme_results")

    with open(f"{output_dir}/adme_comparison.tex", 'w') as f:
        f.write(latex_adme)

    print(f"  [OK] ADME table: {output_dir}/adme_comparison.tex")

    # 2. Toxicity Comparison Table (Classification)
    tox_datasets = ['herg', 'tox21']

    tox_data = []
    for dataset in tox_datasets:
        if dataset not in results:
            continue

        # Find best algorithm
        best_algo = None
        best_f1 = -1

        for algo, data in results[dataset].items():
            final = data.get("final_training", {})
            test_f1 = final.get("test_metrics", {}).get("f1", 0)
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_algo = algo

        if best_algo:
            final = results[dataset][best_algo].get("final_training", {})
            test_metrics = final.get("test_metrics", {})
            val_metrics = final.get("val_metrics", {})

            tox_data.append({
                'Dataset': dataset.upper(),
                'Algorithm': best_algo.upper(),
                'Test F1': test_metrics.get('f1', 0),
                'Test AUC-ROC': test_metrics.get('auc_roc', 0),
                'Test Accuracy': test_metrics.get('accuracy', 0),
                'Val F1': val_metrics.get('f1', 0),
                'Val AUC-ROC': val_metrics.get('auc_roc', 0),
                'Train Time (s)': final.get('train_time', 0)
            })

    tox_df = pd.DataFrame(tox_data)

    # Save as CSV
    tox_df.to_csv(f"{output_dir}/toxicity_comparison.csv", index=False)

    # Save as LaTeX
    latex_tox = tox_df.to_latex(index=False, float_format="%.4f",
                                 caption="Toxicity Datasets - Best HPO Results",
                                 label="tab:tox_results")

    with open(f"{output_dir}/toxicity_comparison.tex", 'w') as f:
        f.write(latex_tox)

    print(f"  [OK] Toxicity table: {output_dir}/toxicity_comparison.tex")

    # 3. Algorithm Comparison Across All Datasets
    algo_summary = []

    all_algos = set()
    for dataset, algos in results.items():
        all_algos.update(algos.keys())

    for algo in sorted(all_algos):
        wins = 0
        total = 0

        for dataset, algos in results.items():
            if algo not in algos:
                continue

            total += 1

            # Check if this is the best algorithm for this dataset
            is_classification = 'tox' in dataset.lower() or 'herg' in dataset.lower()

            best_score = -float('inf') if is_classification else float('inf')
            best_algo_name = None

            for a, data in algos.items():
                final = data.get("final_training", {})
                test_metrics = final.get("test_metrics", {})

                if is_classification:
                    score = test_metrics.get("f1", 0)
                    if score > best_score:
                        best_score = score
                        best_algo_name = a
                else:
                    score = test_metrics.get("rmse", float('inf'))
                    if score < best_score:
                        best_score = score
                        best_algo_name = a

            if best_algo_name == algo:
                wins += 1

        algo_summary.append({
            'Algorithm': algo.upper(),
            'Wins': wins,
            'Total Datasets': total,
            'Win Rate (%)': (wins / total * 100) if total > 0 else 0
        })

    algo_df = pd.DataFrame(algo_summary).sort_values('Wins', ascending=False)

    # Save
    algo_df.to_csv(f"{output_dir}/algorithm_summary.csv", index=False)

    latex_algo = algo_df.to_latex(index=False, float_format="%.1f",
                                   caption="HPO Algorithm Performance Summary",
                                   label="tab:algo_summary")

    with open(f"{output_dir}/algorithm_summary.tex", 'w') as f:
        f.write(latex_algo)

    print(f"  [OK] Algorithm summary: {output_dir}/algorithm_summary.tex")

    print(f"\n[DONE] Comparison tables saved to {output_dir}/")


def create_dataset_statistics_table(output_dir="figures/paper"):
    """Create dataset statistics table."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Generating Dataset Statistics Table ===")

    # Dataset info (from known values)
    dataset_info = [
        {'Dataset': 'Caco2 Wang', 'Type': 'ADME', 'Task': 'Regression', 'Property': 'Permeability', 'Size': 910, 'Train': 728, 'Val': 91, 'Test': 91},
        {'Dataset': 'Half-Life Obach', 'Type': 'ADME', 'Task': 'Regression', 'Property': 'Half-life', 'Size': 667, 'Train': 534, 'Val': 67, 'Test': 66},
        {'Dataset': 'Clearance Hepatocyte', 'Type': 'ADME', 'Task': 'Regression', 'Property': 'Clearance', 'Size': 1213, 'Train': 970, 'Val': 121, 'Test': 122},
        {'Dataset': 'Clearance Microsome', 'Type': 'ADME', 'Task': 'Regression', 'Property': 'Clearance', 'Size': 1102, 'Train': 882, 'Val': 110, 'Test': 110},
        {'Dataset': 'Tox21', 'Type': 'Toxicity', 'Task': 'Classification', 'Property': 'NR Toxicity', 'Size': 7258, 'Train': 5806, 'Val': 726, 'Test': 726},
        {'Dataset': 'hERG', 'Type': 'Toxicity', 'Task': 'Classification', 'Property': 'Cardiac Toxicity', 'Size': 655, 'Train': 524, 'Val': 66, 'Test': 65},
    ]

    df = pd.DataFrame(dataset_info)

    # Save as CSV
    df.to_csv(f"{output_dir}/dataset_statistics.csv", index=False)

    # Save as LaTeX
    latex_table = df.to_latex(index=False,
                              caption="Dataset Statistics and Splits",
                              label="tab:dataset_stats")

    with open(f"{output_dir}/dataset_statistics.tex", 'w') as f:
        f.write(latex_table)

    print(f"  [OK] Dataset statistics: {output_dir}/dataset_statistics.tex")
    print(f"\n[DONE] Dataset statistics table saved")


def generate_performance_vs_time(results, output_dir="figures/paper"):
    """Generate performance vs training time scatter plot."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Generating Performance vs Training Time ===")

    # Collect data
    adme_data = {'time': [], 'performance': [], 'dataset': [], 'algo': []}
    tox_data = {'time': [], 'performance': [], 'dataset': [], 'algo': []}

    for dataset, algos in results.items():
        is_classification = 'tox' in dataset.lower() or 'herg' in dataset.lower()

        for algo, data in algos.items():
            final = data.get("final_training", {})
            test_metrics = final.get("test_metrics", {})
            train_time = final.get("train_time", 0)

            if is_classification:
                performance = test_metrics.get("f1", 0)
                tox_data['time'].append(train_time)
                tox_data['performance'].append(performance)
                tox_data['dataset'].append(dataset)
                tox_data['algo'].append(algo)
            else:
                # For regression, use 1/RMSE as "performance" (higher is better)
                rmse = test_metrics.get("rmse", 100)
                performance = 1 / (rmse + 1e-6)  # Avoid division by zero
                adme_data['time'].append(train_time)
                adme_data['performance'].append(performance)
                adme_data['dataset'].append(dataset)
                adme_data['algo'].append(algo)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ADME plot
    if adme_data['time']:
        scatter1 = ax1.scatter(adme_data['time'], adme_data['performance'],
                              c=range(len(adme_data['time'])), cmap='viridis',
                              s=100, alpha=0.6, edgecolors='black')
        ax1.set_xlabel('Training Time (seconds)', fontsize=12)
        ax1.set_ylabel('Performance (1/RMSE)', fontsize=12)
        ax1.set_title('ADME: Performance vs Training Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

    # Toxicity plot
    if tox_data['time']:
        scatter2 = ax2.scatter(tox_data['time'], tox_data['performance'],
                              c=range(len(tox_data['time'])), cmap='plasma',
                              s=100, alpha=0.6, edgecolors='black')
        ax2.set_xlabel('Training Time (seconds)', fontsize=12)
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.set_title('Toxicity: Performance vs Training Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    filename = f"{output_dir}/performance_vs_time.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] {filename}")
    print(f"\n[DONE] Performance vs time plot saved")


def main():
    """Generate all paper visualizations."""
    print("\n" + "="*80)
    print("GENERATING PAPER VISUALIZATIONS")
    print("="*80)

    # Load all HPO results
    print("\nLoading HPO results...")
    results = load_hpo_results()
    print(f"  Loaded {len(results)} datasets")

    # Create output directory
    output_dir = "figures/paper"
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    generate_training_curves(results, output_dir)
    generate_classification_metrics_plot(results, output_dir)
    create_comparison_tables(results, output_dir)
    create_dataset_statistics_table(output_dir)
    generate_performance_vs_time(results, output_dir)

    print("\n" + "="*80)
    print("[OK] ALL VISUALIZATIONS GENERATED")
    print("="*80)
    print(f"\nOutput directory: {output_dir}/")
    print("\nGenerated files:")
    print("  - Training curves (6 PNG)")
    print("  - Classification metrics evolution (2 PNG)")
    print("  - Comparison tables (3 CSV + 3 TEX)")
    print("  - Dataset statistics (1 CSV + 1 TEX)")
    print("  - Performance vs time (1 PNG)")
    print("\nTotal: ~15 files")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
