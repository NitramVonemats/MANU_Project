#!/usr/bin/env python3
"""
Generate Hyperparameter Sensitivity Analysis
Analyzes how each hyperparameter affects model performance using trial history data.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")


def load_trial_histories(results_dir: str) -> dict:
    """Load trial histories from HPO results files."""
    results = {}

    results_path = Path(results_dir)
    for json_file in results_path.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        dataset_name = json_file.stem.replace("hpo_extended_", "").replace("_results", "")

        # Collect trials from all algorithms
        all_trials = []
        for algo, algo_data in data.items():
            if 'trial_history' in algo_data:
                for trial in algo_data['trial_history']:
                    trial_record = trial['params'].copy()
                    trial_record['algorithm'] = algo

                    # Get validation score
                    if 'result' in trial:
                        result = trial['result']
                        if 'val_f1' in result:
                            trial_record['val_score'] = result['val_f1']
                            trial_record['test_score'] = result.get('test_f1', np.nan)
                            trial_record['metric'] = 'F1'
                        elif 'val_rmse' in result:
                            trial_record['val_score'] = -result['val_rmse']  # Negative for consistency
                            trial_record['test_score'] = -result.get('test_rmse', np.nan)
                            trial_record['metric'] = 'RMSE'

                    all_trials.append(trial_record)

        if all_trials:
            results[dataset_name] = pd.DataFrame(all_trials)

    return results


def plot_param_vs_performance(df: pd.DataFrame, param: str, dataset: str, output_dir: str):
    """Plot single hyperparameter vs performance."""
    if param not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # For log-scale params
    if param in ['lr', 'weight_decay']:
        df[f'{param}_log'] = np.log10(df[param])
        x_data = df[f'{param}_log']
        xlabel = f'log10({param})'
    else:
        x_data = df[param]
        xlabel = param

    # Scatter plot with regression line
    ax.scatter(x_data, df['val_score'], alpha=0.5, s=40)

    # Add trend line
    z = np.polyfit(x_data, df['val_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.4f})')

    # Calculate correlation
    corr = np.corrcoef(x_data, df['val_score'])[0, 1]

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset}: {param} Sensitivity (r={corr:.3f})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"{output_dir}/sensitivity_{dataset}_{param}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    return corr


def plot_combined_sensitivity(all_data: dict, output_dir: str):
    """Create combined sensitivity analysis across all datasets."""

    params_to_analyze = ['hidden_dim', 'num_layers', 'lr', 'weight_decay']

    # Calculate correlations for all params across datasets
    correlation_matrix = {}

    for dataset, df in all_data.items():
        correlation_matrix[dataset] = {}
        for param in params_to_analyze:
            if param not in df.columns:
                correlation_matrix[dataset][param] = np.nan
                continue

            # Transform log-scale params
            if param in ['lr', 'weight_decay']:
                x_data = np.log10(df[param])
            else:
                x_data = df[param]

            try:
                corr = np.corrcoef(x_data, df['val_score'])[0, 1]
                correlation_matrix[dataset][param] = corr
            except:
                correlation_matrix[dataset][param] = np.nan

    # Create heatmap
    corr_df = pd.DataFrame(correlation_matrix).T

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Correlation with Performance'}, ax=ax)

    ax.set_title('Hyperparameter Sensitivity Analysis\n(Correlation with Validation Score)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Hyperparameter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    filename = f"{output_dir}/param_sensitivity_heatmap.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] {filename}")

    return corr_df


def plot_param_distributions(all_data: dict, output_dir: str):
    """Plot hyperparameter distributions with performance overlay."""

    params_to_analyze = ['hidden_dim', 'num_layers', 'lr', 'weight_decay']

    for dataset, df in all_data.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, param in enumerate(params_to_analyze):
            ax = axes[idx]

            if param not in df.columns:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
                ax.set_title(f'{param}')
                continue

            x_data = df[param].copy()

            # Transform log-scale params for visualization
            if param in ['lr', 'weight_decay']:
                x_data = np.log10(x_data)
                xlabel = f'log10({param})'
            else:
                xlabel = param

            # Scatter with color coding by performance
            scatter = ax.scatter(x_data, df['val_score'],
                               c=df['val_score'], cmap='viridis',
                               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('Val Score', fontsize=11)
            ax.set_title(f'{param}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{dataset.upper()}: Hyperparameter vs Performance',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = f"{output_dir}/param_sensitivity_{dataset}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] {filename}")


def plot_learning_rate_analysis(all_data: dict, output_dir: str):
    """Detailed learning rate analysis."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (dataset, df) in enumerate(all_data.items()):
        ax = axes[idx]

        if 'lr' not in df.columns:
            continue

        lr_log = np.log10(df['lr'])

        # Scatter plot
        scatter = ax.scatter(lr_log, df['val_score'],
                           c=df['num_layers'], cmap='viridis',
                           alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

        # Best points
        best_idx = df['val_score'].idxmax()
        ax.scatter(lr_log[best_idx], df['val_score'][best_idx],
                  c='red', s=200, marker='*', zorder=5, label='Best')

        # Add trend line
        z = np.polyfit(lr_log, df['val_score'], 2)
        p = np.poly1d(z)
        x_line = np.linspace(lr_log.min(), lr_log.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='Trend')

        ax.set_xlabel('log10(Learning Rate)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Num Layers', fontsize=10)

    plt.suptitle('Learning Rate Sensitivity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = f"{output_dir}/learning_rate_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] {filename}")


def generate_summary_table(all_data: dict, output_dir: str):
    """Generate summary statistics table."""

    summary_rows = []

    for dataset, df in all_data.items():
        best_trial = df.loc[df['val_score'].idxmax()]

        row = {
            'Dataset': dataset,
            'Best Hidden Dim': best_trial.get('hidden_dim', 'N/A'),
            'Best Num Layers': best_trial.get('num_layers', 'N/A'),
            'Best LR': f"{best_trial.get('lr', 0):.2e}",
            'Best Weight Decay': f"{best_trial.get('weight_decay', 0):.2e}",
            'Best Val Score': f"{best_trial['val_score']:.4f}",
            'Algorithm': best_trial.get('algorithm', 'N/A'),
            'Total Trials': len(df)
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save as CSV
    csv_path = f"{output_dir}/param_sensitivity_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  [OK] {csv_path}")

    return summary_df


def main():
    """Generate all hyperparameter sensitivity visualizations."""
    print("\n" + "="*80)
    print("GENERATING HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("="*80)

    # Create output directory
    output_dir = "figures/paper"
    os.makedirs(output_dir, exist_ok=True)

    # Load data from extended HPO results
    print("\nLoading trial histories...")

    all_data = {}

    # Try to load hERG results
    herg_dir = Path("results/hpo_extended/herg")
    if herg_dir.exists():
        herg_data = load_trial_histories(str(herg_dir))
        all_data.update(herg_data)

    # Try to load Caco2 results
    caco2_dir = Path("results/hpo_extended/caco2_wang")
    if caco2_dir.exists():
        caco2_data = load_trial_histories(str(caco2_dir))
        all_data.update(caco2_data)

    if not all_data:
        print("[ERROR] No HPO trial data found!")
        return

    print(f"  Loaded data for {len(all_data)} datasets")
    for dataset, df in all_data.items():
        print(f"    - {dataset}: {len(df)} trials")

    # Generate visualizations
    print("\nGenerating sensitivity plots...")

    # Individual parameter plots
    plot_param_distributions(all_data, output_dir)

    # Combined heatmap
    corr_df = plot_combined_sensitivity(all_data, output_dir)

    # Learning rate analysis
    plot_learning_rate_analysis(all_data, output_dir)

    # Summary table
    summary_df = generate_summary_table(all_data, output_dir)

    print("\n" + "="*80)
    print("[OK] HYPERPARAMETER SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}/")
    print("\nGenerated files:")
    print("  - param_sensitivity_heatmap.png")
    print("  - param_sensitivity_herg.png")
    print("  - param_sensitivity_caco2_wang.png")
    print("  - learning_rate_analysis.png")
    print("  - param_sensitivity_summary.csv")

    print("\n" + "="*80)
    print("CORRELATION SUMMARY")
    print("="*80)
    print(corr_df.to_string())

    print("\n" + "="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
