"""
Analyze HPO results and compute variance/uncertainty metrics.
Phase 3 of paper preparation.

Extracts metrics from HPO logs and computes:
- Per-algorithm performance
- Cross-algorithm variance
- Summary tables with mean ± std
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def load_hpo_results(results_dir="results/hpo"):
    """Load all HPO JSON results."""
    results_path = Path(results_dir)
    all_results = []

    for json_file in results_path.glob("**/*.json"):
        # Skip foundation model results for now
        if "foundation" in str(json_file):
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract key metrics
            result = {
                'dataset': data.get('dataset', ''),
                'algorithm': data.get('algo', ''),
                'seed': data.get('seed', 42),
                'n_trials': data.get('search', {}).get('trials', 0),
                'best_val_rmse': data.get('search', {}).get('best_val_rmse'),
                'best_val_f1': data.get('search', {}).get('best_val_f1'),
            }

            # Final training metrics
            final = data.get('final_training', {})
            test_metrics = final.get('test_metrics', {})
            val_metrics = final.get('val_metrics', {})

            # Regression metrics
            result['test_rmse'] = test_metrics.get('rmse')
            result['test_mae'] = test_metrics.get('mae')
            result['test_r2'] = test_metrics.get('r2')
            result['val_rmse'] = val_metrics.get('rmse')
            result['val_r2'] = val_metrics.get('r2')

            # Classification metrics
            result['test_auc_roc'] = test_metrics.get('auc_roc')
            result['test_f1'] = test_metrics.get('f1')
            result['test_accuracy'] = test_metrics.get('accuracy')
            result['val_auc_roc'] = val_metrics.get('auc_roc')
            result['val_f1'] = val_metrics.get('f1')

            # Training info
            result['train_time'] = final.get('train_time')

            # Best params
            best_params = data.get('search', {}).get('best_params', {})
            result['hidden_dim'] = best_params.get('hidden_dim')
            result['num_layers'] = best_params.get('num_layers')
            result['lr'] = best_params.get('lr')

            all_results.append(result)

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return pd.DataFrame(all_results)


def compute_algorithm_statistics(df):
    """Compute statistics per dataset across algorithms."""

    # Determine task type for each dataset
    classification_datasets = ['tox21', 'herg']

    stats = []

    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        is_classification = dataset.lower() in classification_datasets

        if is_classification:
            # Classification metrics
            metric_col = 'test_f1'
            metric_name = 'F1'
            val_metric = 'val_f1'
        else:
            # Regression metrics
            metric_col = 'test_rmse'
            metric_name = 'RMSE'
            val_metric = 'val_rmse'

        # Get valid values
        values = ds_df[metric_col].dropna().values

        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)

            # Find best algorithm
            if is_classification:
                best_idx = ds_df[metric_col].idxmax()
            else:
                best_idx = ds_df[metric_col].idxmin()

            best_algo = ds_df.loc[best_idx, 'algorithm'] if best_idx in ds_df.index else 'N/A'
            best_val = ds_df.loc[best_idx, metric_col] if best_idx in ds_df.index else np.nan

            stats.append({
                'Dataset': dataset,
                'Task': 'Classification' if is_classification else 'Regression',
                'Metric': metric_name,
                'N_Algorithms': len(values),
                'Mean': mean_val,
                'Std': std_val,
                'Min': min_val,
                'Max': max_val,
                'Best_Algorithm': best_algo,
                'Best_Value': best_val,
            })

    return pd.DataFrame(stats)


def generate_algorithm_comparison_table(df, output_dir):
    """Generate per-algorithm comparison table."""

    classification_datasets = ['tox21', 'herg']
    algorithms = ['pso', 'abc', 'ga', 'sa', 'hc', 'random']

    # Regression table
    reg_datasets = [d for d in df['dataset'].unique() if d.lower() not in classification_datasets]
    reg_data = []

    for dataset in reg_datasets:
        ds_df = df[df['dataset'] == dataset]
        row = {'Dataset': dataset}
        for algo in algorithms:
            algo_df = ds_df[ds_df['algorithm'] == algo]
            if len(algo_df) > 0:
                rmse = algo_df['test_rmse'].values[0]
                r2 = algo_df['test_r2'].values[0]
                row[f'{algo.upper()}_RMSE'] = rmse
                row[f'{algo.upper()}_R2'] = r2
            else:
                row[f'{algo.upper()}_RMSE'] = np.nan
                row[f'{algo.upper()}_R2'] = np.nan
        reg_data.append(row)

    reg_df = pd.DataFrame(reg_data)
    reg_df.to_csv(output_dir / 'regression_algorithm_comparison.csv', index=False)

    # Classification table
    class_datasets = [d for d in df['dataset'].unique() if d.lower() in classification_datasets]
    class_data = []

    for dataset in class_datasets:
        ds_df = df[df['dataset'] == dataset]
        row = {'Dataset': dataset}
        for algo in algorithms:
            algo_df = ds_df[ds_df['algorithm'] == algo]
            if len(algo_df) > 0:
                f1 = algo_df['test_f1'].values[0]
                auc = algo_df['test_auc_roc'].values[0]
                row[f'{algo.upper()}_F1'] = f1
                row[f'{algo.upper()}_AUC'] = auc
            else:
                row[f'{algo.upper()}_F1'] = np.nan
                row[f'{algo.upper()}_AUC'] = np.nan
        class_data.append(row)

    class_df = pd.DataFrame(class_data)
    class_df.to_csv(output_dir / 'classification_algorithm_comparison.csv', index=False)

    return reg_df, class_df


def plot_algorithm_comparison(df, output_dir):
    """Generate comparison plots with variance bars."""

    classification_datasets = ['tox21', 'herg']
    algorithms = ['PSO', 'ABC', 'GA', 'SA', 'HC', 'Random']
    algo_lower = [a.lower() for a in algorithms]

    # Regression plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    reg_datasets = [d for d in df['dataset'].unique() if d.lower() not in classification_datasets]

    for idx, dataset in enumerate(reg_datasets[:4]):
        ax = axes[idx // 2, idx % 2]
        ds_df = df[df['dataset'] == dataset]

        rmse_values = []
        for algo in algo_lower:
            algo_df = ds_df[ds_df['algorithm'] == algo]
            if len(algo_df) > 0:
                rmse_values.append(algo_df['test_rmse'].values[0])
            else:
                rmse_values.append(np.nan)

        bars = ax.bar(algorithms, rmse_values, color='steelblue', alpha=0.7)
        ax.set_ylabel('Test RMSE', fontsize=11)
        ax.set_title(dataset, fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        # Highlight best
        valid_values = [v for v in rmse_values if not np.isnan(v)]
        if valid_values:
            best_idx = rmse_values.index(min(valid_values))
            bars[best_idx].set_color('darkorange')

    plt.tight_layout()
    plt.savefig(output_dir / 'regression_algorithm_comparison.png', dpi=300)
    plt.close()

    # Classification plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    class_datasets = [d for d in df['dataset'].unique() if d.lower() in classification_datasets]

    for idx, dataset in enumerate(class_datasets[:2]):
        ax = axes[idx]
        ds_df = df[df['dataset'] == dataset]

        f1_values = []
        for algo in algo_lower:
            algo_df = ds_df[ds_df['algorithm'] == algo]
            if len(algo_df) > 0:
                f1_values.append(algo_df['test_f1'].values[0])
            else:
                f1_values.append(np.nan)

        bars = ax.bar(algorithms, f1_values, color='steelblue', alpha=0.7)
        ax.set_ylabel('Test F1 Score', fontsize=11)
        ax.set_title(dataset, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)

        # Highlight best
        valid_values = [v for v in f1_values if not np.isnan(v)]
        if valid_values:
            best_idx = f1_values.index(max(valid_values))
            bars[best_idx].set_color('darkorange')

    plt.tight_layout()
    plt.savefig(output_dir / 'classification_algorithm_comparison.png', dpi=300)
    plt.close()


def generate_latex_tables(df, stats_df, output_dir):
    """Generate LaTeX tables for the paper."""

    classification_datasets = ['tox21', 'herg']

    # Main results table with cross-algorithm variance
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{HPO algorithm comparison across datasets. Values show test metrics for each optimizer. Best per dataset in \\textbf{bold}.}",
        "\\label{tab:hpo_results}",
        "\\scriptsize",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{PSO} & \\textbf{ABC} & \\textbf{GA} & \\textbf{SA} & \\textbf{HC} & \\textbf{Random} \\\\",
        "\\midrule",
        "\\multicolumn{7}{l}{\\emph{Regression (RMSE $\\downarrow$)}} \\\\",
        "\\midrule",
    ]

    algorithms = ['pso', 'abc', 'ga', 'sa', 'hc', 'random']
    reg_datasets = [d for d in df['dataset'].unique() if d.lower() not in classification_datasets]

    for dataset in reg_datasets:
        ds_df = df[df['dataset'] == dataset]
        values = []
        for algo in algorithms:
            algo_df = ds_df[ds_df['algorithm'] == algo]
            if len(algo_df) > 0:
                val = algo_df['test_rmse'].values[0]
                values.append(val if not np.isnan(val) else None)
            else:
                values.append(None)

        # Find best
        valid_values = [v for v in values if v is not None]
        best = min(valid_values) if valid_values else None

        row = [dataset.replace('_', '\\_')]
        for v in values:
            if v is None:
                row.append('---')
            elif v == best:
                row.append(f'\\textbf{{{v:.4f}}}')
            else:
                row.append(f'{v:.4f}')

        latex_lines.append(' & '.join(row) + ' \\\\')

    latex_lines.extend([
        "\\midrule",
        "\\multicolumn{7}{l}{\\emph{Classification (F1 $\\uparrow$)}} \\\\",
        "\\midrule",
    ])

    class_datasets = [d for d in df['dataset'].unique() if d.lower() in classification_datasets]

    for dataset in class_datasets:
        ds_df = df[df['dataset'] == dataset]
        values = []
        for algo in algorithms:
            algo_df = ds_df[ds_df['algorithm'] == algo]
            if len(algo_df) > 0:
                val = algo_df['test_f1'].values[0]
                values.append(val if not np.isnan(val) else None)
            else:
                values.append(None)

        # Find best
        valid_values = [v for v in values if v is not None]
        best = max(valid_values) if valid_values else None

        row = [dataset]
        for v in values:
            if v is None:
                row.append('---')
            elif v == best:
                row.append(f'\\textbf{{{v:.4f}}}')
            else:
                row.append(f'{v:.4f}')

        latex_lines.append(' & '.join(row) + ' \\\\')

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_dir / 'hpo_results_table.tex', 'w') as f:
        f.write('\n'.join(latex_lines))

    # Variance summary table
    variance_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-algorithm variance analysis. Shows mean $\\pm$ std across 6 HPO algorithms.}",
        "\\label{tab:variance}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Metric} & \\textbf{Mean $\\pm$ Std} & \\textbf{Best Algorithm} \\\\",
        "\\midrule",
    ]

    for _, row in stats_df.iterrows():
        dataset = row['Dataset'].replace('_', '\\_')
        metric = row['Metric']
        mean = row['Mean']
        std = row['Std']
        best = row['Best_Algorithm'].upper()

        variance_lines.append(
            f"{dataset} & {metric} & {mean:.4f} $\\pm$ {std:.4f} & {best} \\\\"
        )

    variance_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_dir / 'variance_table.tex', 'w') as f:
        f.write('\n'.join(variance_lines))

    print(f"Generated LaTeX tables in {output_dir}")


def main():
    """Main analysis function."""

    print("="*70)
    print("HPO VARIANCE ANALYSIS")
    print("="*70)

    # Load results
    df = load_hpo_results()
    print(f"\nLoaded {len(df)} HPO results")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print(f"Algorithms: {df['algorithm'].unique().tolist()}")

    # Output directory
    output_dir = Path("figures/paper")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute statistics
    stats_df = compute_algorithm_statistics(df)
    print("\n" + "="*70)
    print("CROSS-ALGORITHM VARIANCE STATISTICS")
    print("="*70)
    print(stats_df.to_string(index=False))

    # Save statistics
    stats_df.to_csv(output_dir / 'hpo_variance_statistics.csv', index=False)

    # Generate comparison tables
    reg_df, class_df = generate_algorithm_comparison_table(df, output_dir)

    # Generate plots
    plot_algorithm_comparison(df, output_dir)
    print(f"\nGenerated plots in {output_dir}")

    # Generate LaTeX tables
    generate_latex_tables(df, stats_df, output_dir)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nKey findings:")
    for _, row in stats_df.iterrows():
        print(f"  {row['Dataset']}: {row['Best_Algorithm'].upper()} best ({row['Metric']}={row['Best_Value']:.4f})")
        print(f"    Cross-algorithm variance: {row['Mean']:.4f} ± {row['Std']:.4f}")

    print("\n" + "="*70)
    print("LIMITATION NOTE")
    print("="*70)
    print("These statistics represent cross-algorithm variance, NOT multi-seed variance.")
    print("Each algorithm was run with seed=42. Multi-seed validation is needed for")
    print("proper uncertainty quantification (see Phase 4).")

    return df, stats_df


if __name__ == "__main__":
    df, stats = main()
