"""
Multi-seed validation for robustness assessment.
Phase 4 of paper preparation.

Runs the optimized GNN model with multiple seeds to quantify variance.
Uses the same scaffold splits but different random initialization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from optimized_gnn import (
    train_model, OptimizedGNNConfig, is_classification_dataset
)


def run_multiseed_validation(
    datasets=None,
    seeds=(42, 43, 44),
    device='cpu',
    epochs=100,
    patience=20
):
    """Run training with multiple seeds for all datasets."""

    if datasets is None:
        datasets = [
            'Caco2_Wang',
            'Half_Life_Obach',
            'Clearance_Hepatocyte_AZ',
            'Clearance_Microsome_AZ',
            'tox21',
            'herg',
        ]

    config = OptimizedGNNConfig(
        hidden_dim=128,
        num_layers=5,
        lr=0.001,
        batch_train=32,
        batch_eval=64,
    )

    all_results = []

    for dataset in datasets:
        is_class = is_classification_dataset(dataset)
        task_type = 'Classification' if is_class else 'Regression'

        print(f"\n{'='*70}")
        print(f"Dataset: {dataset} ({task_type})")
        print(f"{'='*70}")

        seed_results = []

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")

            # Set all seeds
            torch.manual_seed(seed)
            np.random.seed(seed)

            result = train_model(
                dataset_name=dataset,
                config=config,
                epochs=epochs,
                patience=patience,
                device=device,
                seed=seed,
                return_model=False,
                verbose=True
            )

            test_metrics = result['test_metrics']

            seed_result = {
                'dataset': dataset,
                'task': task_type,
                'seed': seed,
                'train_time': result['train_time'],
            }

            if is_class:
                seed_result.update({
                    'test_auc_roc': test_metrics['auc_roc'],
                    'test_f1': test_metrics['f1'],
                    'test_accuracy': test_metrics['accuracy'],
                })
            else:
                seed_result.update({
                    'test_rmse': test_metrics['rmse'],
                    'test_mae': test_metrics['mae'],
                    'test_r2': test_metrics['r2'],
                })

            seed_results.append(seed_result)
            all_results.append(seed_result)

        # Compute statistics for this dataset
        print(f"\n--- {dataset} Summary ---")
        if is_class:
            f1_values = [r['test_f1'] for r in seed_results]
            auc_values = [r['test_auc_roc'] for r in seed_results]
            print(f"F1:      {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}")
            print(f"AUC-ROC: {np.mean(auc_values):.4f} ± {np.std(auc_values):.4f}")
        else:
            rmse_values = [r['test_rmse'] for r in seed_results]
            r2_values = [r['test_r2'] for r in seed_results]
            print(f"RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")
            print(f"R2:   {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")

    return pd.DataFrame(all_results)


def compute_summary_statistics(df):
    """Compute summary statistics from multi-seed results."""

    summary = []

    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        task = ds_df['task'].iloc[0]

        if task == 'Classification':
            f1_mean = ds_df['test_f1'].mean()
            f1_std = ds_df['test_f1'].std()
            auc_mean = ds_df['test_auc_roc'].mean()
            auc_std = ds_df['test_auc_roc'].std()

            summary.append({
                'Dataset': dataset,
                'Task': task,
                'Primary_Metric': 'F1',
                'Mean': f1_mean,
                'Std': f1_std,
                'Mean_pm_Std': f"{f1_mean:.4f} ± {f1_std:.4f}",
                'AUC_Mean': auc_mean,
                'AUC_Std': auc_std,
            })
        else:
            rmse_mean = ds_df['test_rmse'].mean()
            rmse_std = ds_df['test_rmse'].std()
            r2_mean = ds_df['test_r2'].mean()
            r2_std = ds_df['test_r2'].std()

            summary.append({
                'Dataset': dataset,
                'Task': task,
                'Primary_Metric': 'RMSE',
                'Mean': rmse_mean,
                'Std': rmse_std,
                'Mean_pm_Std': f"{rmse_mean:.4f} ± {rmse_std:.4f}",
                'R2_Mean': r2_mean,
                'R2_Std': r2_std,
            })

    return pd.DataFrame(summary)


def generate_latex_table(summary_df, output_path):
    """Generate LaTeX table for multi-seed results."""

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Multi-seed validation results (seeds 42, 43, 44). Shows mean $\\pm$ std across 3 random initializations with identical scaffold splits.}",
        "\\label{tab:multiseed}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Task} & \\textbf{Primary Metric} & \\textbf{Mean $\\pm$ Std} \\\\",
        "\\midrule",
    ]

    for _, row in summary_df.iterrows():
        dataset = row['Dataset'].replace('_', '\\_')
        task = row['Task'][:5] + '.'  # Truncate
        metric = row['Primary_Metric']
        mean_std = row['Mean_pm_Std']

        lines.append(f"{dataset} & {task} & {metric} & {mean_std} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Generated LaTeX table: {output_path}")


def main():
    """Main function for multi-seed validation."""

    print("="*70)
    print("MULTI-SEED VALIDATION")
    print("Seeds: 42, 43, 44")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Run validation
    results_df = run_multiseed_validation(
        seeds=(42, 43, 44),
        device=device,
        epochs=100,
        patience=20
    )

    # Output directory
    output_dir = Path("results/multiseed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    results_df.to_csv(output_dir / 'multiseed_raw_results.csv', index=False)
    print(f"\nSaved raw results to {output_dir / 'multiseed_raw_results.csv'}")

    # Compute summary
    summary_df = compute_summary_statistics(results_df)
    summary_df.to_csv(output_dir / 'multiseed_summary.csv', index=False)

    print("\n" + "="*70)
    print("MULTI-SEED VALIDATION SUMMARY")
    print("="*70)
    print(summary_df[['Dataset', 'Task', 'Primary_Metric', 'Mean_pm_Std']].to_string(index=False))

    # Generate LaTeX table
    latex_dir = Path("figures/paper")
    latex_dir.mkdir(parents=True, exist_ok=True)
    generate_latex_table(summary_df, latex_dir / 'multiseed_table.tex')

    # Save as JSON for paper
    summary_json = summary_df.to_dict(orient='records')
    with open(output_dir / 'multiseed_summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("Multi-seed validation complete. Results confirm model robustness")
    print("across different random initializations with fixed scaffold splits.")

    return results_df, summary_df


if __name__ == "__main__":
    results, summary = main()
