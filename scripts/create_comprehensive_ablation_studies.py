"""
COMPREHENSIVE ABLATION STUDIES FROM HPO RESULTS
================================================

Generate detailed ablation study visualizations from HPO results:
1. Hyperparameter sensitivity analysis (all datasets)
2. Performance vs hyperparameter relationships
3. Unified ablation plots for paper
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_hpo_results():
    """Load all HPO results from runs/ directory."""
    datasets = ['Caco2_Wang', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'Half_Life_Obach']
    algorithms = ['random', 'pso', 'ga', 'sa', 'hc', 'abc']

    all_results = []

    for dataset in datasets:
        for algorithm in algorithms:
            filepath = f'runs/{dataset}/hpo_{dataset}_{algorithm}.json'

            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract best configuration results
                if 'search' in data and 'final_training' in data:
                    best_params = data['search']['best_params']
                    test_metrics = data['final_training']['test_metrics']
                    val_metrics = data['final_training']['val_metrics']

                    result = {
                        'dataset': dataset,
                        'algorithm': algorithm.upper(),
                        'hidden_dim': best_params['hidden_dim'],
                        'num_layers': best_params['num_layers'],
                        'lr': best_params['lr'],
                        'weight_decay': best_params.get('weight_decay', 1e-5),
                        'dropout': best_params.get('dropout', 0.0),
                        'batch_train': best_params.get('batch_train', 32),
                        'test_rmse': test_metrics['rmse'],
                        'test_r2': test_metrics['r2'],
                        'test_mae': test_metrics['mae'],
                        'val_rmse': val_metrics['rmse'],
                        'val_r2': val_metrics['r2'],
                        'train_time': data['final_training'].get('train_time', 0)
                    }
                    all_results.append(result)

    if len(all_results) == 0:
        print("\n[ERROR] No results loaded!")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    print(f"\nLoaded {len(df)} best configurations from HPO results")
    print(f"Datasets: {df['dataset'].nunique()}")
    print(f"Algorithms: {df['algorithm'].nunique()}")
    print(f"\nDatasets: {', '.join(df['dataset'].unique())}")
    print(f"Algorithms: {', '.join(sorted(df['algorithm'].unique()))}")

    return df


def create_hyperparameter_sensitivity_plots(df, output_dir='figures/ablation_studies'):
    """Create hyperparameter comparison plots across algorithms."""
    print("\n[1/5] Creating hyperparameter comparison plots...")

    os.makedirs(output_dir, exist_ok=True)

    datasets = df['dataset'].unique()
    hyperparams = ['hidden_dim', 'num_layers', 'lr', 'weight_decay']

    for dataset in datasets:
        df_dataset = df[df['dataset'] == dataset]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, param in enumerate(hyperparams):
            ax = axes[idx]

            # Scatter plot colored by algorithm
            algorithms = df_dataset['algorithm'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))

            for i, algo in enumerate(sorted(algorithms)):
                df_algo = df_dataset[df_dataset['algorithm'] == algo]
                ax.scatter(df_algo[param], df_algo['test_r2'],
                          s=200, alpha=0.7, label=algo,
                          color=colors[i], edgecolor='black', linewidth=1.5)

            # Mark best overall
            best_idx = df_dataset['test_r2'].idxmax()
            best_row = df_dataset.loc[best_idx]
            ax.scatter([best_row[param]], [best_row['test_r2']],
                      s=400, color='gold', marker='*',
                      edgecolor='red', linewidth=3, zorder=10,
                      label=f'Best: {best_row["algorithm"]}')

            ax.set_xlabel(param.replace('_', ' ').title(), fontweight='bold', fontsize=12)
            ax.set_ylabel('Test R²', fontweight='bold', fontsize=12)
            ax.set_title(f'{param.replace("_", " ").title()} vs Performance',
                        fontweight='bold', fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9, loc='best')

            # Log scale for learning rate
            if param == 'lr':
                ax.set_xscale('log')

        plt.suptitle(f'{dataset} - Hyperparameter Choices by Algorithm',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{dataset}_hyperparameter_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] {dataset} hyperparameter comparison saved")


def create_unified_sensitivity_heatmaps(df, output_dir='figures/ablation_studies'):
    """Create unified heatmaps showing hyperparameter choices across algorithms and datasets."""
    print("\n[2/5] Creating unified hyperparameter heatmaps...")

    os.makedirs(output_dir, exist_ok=True)

    datasets = sorted(df['dataset'].unique())
    algorithms = sorted(df['algorithm'].unique())
    hyperparams = ['hidden_dim', 'num_layers', 'lr', 'weight_decay']

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for idx, param in enumerate(hyperparams):
        ax = axes[idx]

        # Create matrix: algorithms x datasets
        matrix = []

        for algo in algorithms:
            row = []
            for dataset in datasets:
                subset = df[(df['algorithm'] == algo) & (df['dataset'] == dataset)]
                if len(subset) > 0:
                    row.append(subset[param].iloc[0])
                else:
                    row.append(np.nan)
            matrix.append(row)

        matrix = np.array(matrix)

        # Plot heatmap
        if param == 'lr':
            # Use log scale for learning rate
            matrix_plot = np.log10(matrix)
            im = ax.imshow(matrix_plot, aspect='auto', cmap='viridis')
        else:
            im = ax.imshow(matrix, aspect='auto', cmap='viridis')

        # Set ticks
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([d.replace('_', '\n') for d in datasets],
                          rotation=0, ha='center', fontsize=10)
        ax.set_yticks(range(len(algorithms)))
        ax.set_yticklabels(algorithms, fontsize=11)

        # Add values to cells
        for i in range(len(algorithms)):
            for j in range(len(datasets)):
                if not np.isnan(matrix[i, j]):
                    if param == 'lr':
                        text_val = f'{matrix[i, j]:.2e}'
                    elif param == 'weight_decay':
                        text_val = f'{matrix[i, j]:.2e}'
                    else:
                        text_val = f'{int(matrix[i, j])}'

                    ax.text(j, i, text_val,
                           ha="center", va="center", color="white",
                           fontsize=8, fontweight='bold')

        ax.set_xlabel('Dataset', fontweight='bold', fontsize=12)
        ax.set_ylabel('Algorithm', fontweight='bold', fontsize=12)
        ax.set_title(f'Best {param.replace("_", " ").title()} Chosen',
                    fontweight='bold', fontsize=13)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if param == 'lr':
            cbar.set_label(f'log10({param.replace("_", " ").title()})', fontweight='bold')
        else:
            cbar.set_label(param.replace('_', ' ').title(), fontweight='bold')

    plt.suptitle('Hyperparameter Choices - All Algorithms × Datasets',
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/unified_hyperparameter_heatmaps.png',
               dpi=300, bbox_inches='tight')
    plt.close()

    print("  [OK] Unified hyperparameter heatmaps saved")


def create_performance_landscape_plots(df, output_dir='figures/ablation_studies'):
    """Create 2D performance landscape plots showing algorithm choices."""
    print("\n[3/5] Creating performance landscape plots...")

    os.makedirs(output_dir, exist_ok=True)

    datasets = df['dataset'].unique()
    algorithms = sorted(df['algorithm'].unique())
    colors_algo = {algo: plt.cm.tab10(i) for i, algo in enumerate(algorithms)}

    for dataset in datasets:
        df_dataset = df[df['dataset'] == dataset]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # Plot pairs
        pairs = [
            ('hidden_dim', 'num_layers'),
            ('hidden_dim', 'lr'),
            ('num_layers', 'lr'),
            ('hidden_dim', 'weight_decay'),
            ('num_layers', 'weight_decay'),
            ('lr', 'weight_decay')
        ]

        for idx, (param1, param2) in enumerate(pairs):
            ax = axes[idx // 3, idx % 3]

            # Scatter plot colored by algorithm
            for algo in algorithms:
                df_algo = df_dataset[df_dataset['algorithm'] == algo]
                if len(df_algo) > 0:
                    scatter = ax.scatter(df_algo[param1], df_algo[param2],
                                       c=[colors_algo[algo]], s=200, alpha=0.7,
                                       edgecolor='black', linewidth=1.5,
                                       label=f'{algo} (R²={df_algo["test_r2"].iloc[0]:.3f})')

            # Mark best configuration
            best_idx = df_dataset['test_r2'].idxmax()
            best_row = df_dataset.loc[best_idx]
            ax.scatter([best_row[param1]], [best_row[param2]],
                      s=500, color='gold', marker='*',
                      edgecolor='red', linewidth=3, zorder=10)

            ax.set_xlabel(param1.replace('_', ' ').title(), fontweight='bold', fontsize=11)
            ax.set_ylabel(param2.replace('_', ' ').title(), fontweight='bold', fontsize=11)
            ax.set_title(f'{param1.replace("_", " ").title()} vs {param2.replace("_", " ").title()}',
                        fontweight='bold', fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc='best')

            # Log scale for lr
            if param1 == 'lr':
                ax.set_xscale('log')
            if param2 == 'lr':
                ax.set_yscale('log')

        plt.suptitle(f'{dataset} - Hyperparameter Space Exploration',
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{dataset}_hyperparameter_space.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] {dataset} hyperparameter space plot saved")


def create_ablation_component_analysis(df, output_dir='figures/ablation_studies'):
    """Create ablation study showing hyperparameter range impact."""
    print("\n[4/5] Creating hyperparameter range analysis...")

    os.makedirs(output_dir, exist_ok=True)

    datasets = sorted(df['dataset'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    hyperparams = ['hidden_dim', 'num_layers', 'lr', 'weight_decay']
    param_names = ['Hidden Dim', 'Num Layers', 'Learning Rate', 'Weight Decay']

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        df_dataset = df[df['dataset'] == dataset]

        # Calculate correlation between each hyperparameter and performance
        correlations = []
        ranges = []

        for param, name in zip(hyperparams, param_names):
            # Correlation with R²
            corr = df_dataset[[param, 'test_r2']].corr().iloc[0, 1]

            # Range of values
            if param in ['lr', 'weight_decay']:
                range_val = df_dataset[param].max() / df_dataset[param].min()
            else:
                range_val = df_dataset[param].max() - df_dataset[param].min()

            correlations.append(abs(corr))
            ranges.append({
                'param': name,
                'corr': corr,
                'abs_corr': abs(corr)
            })

        ranges_df = pd.DataFrame(ranges).sort_values('abs_corr', ascending=True)

        # Horizontal bar plot
        colors = ['green' if x > 0 else 'red' for x in ranges_df['corr']]

        bars = ax.barh(ranges_df['param'], ranges_df['corr'],
                       color=colors, edgecolor='black', alpha=0.8)

        # Add value labels
        for bar, corr in zip(bars, ranges_df['corr']):
            x_pos = corr + (0.02 if corr > 0 else -0.02)
            ha = 'left' if corr > 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   f'{corr:.3f}',
                   va='center', ha=ha, fontweight='bold', fontsize=10)

        ax.set_xlabel('Correlation with Test R²', fontweight='bold', fontsize=11)
        ax.set_title(f'{dataset}', fontweight='bold', fontsize=13)
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linewidth=1.5)
        ax.set_xlim(-1, 1)

    plt.suptitle('Hyperparameter-Performance Correlations',
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/unified_hyperparameter_correlations.png',
               dpi=300, bbox_inches='tight')
    plt.close()

    print("  [OK] Hyperparameter correlation analysis saved")


def create_comprehensive_summary_table(df, output_dir='figures/ablation_studies'):
    """Create comprehensive summary table."""
    print("\n[5/5] Creating comprehensive summary table...")

    os.makedirs(output_dir, exist_ok=True)

    datasets = sorted(df['dataset'].unique())
    summary_data = []

    for dataset in datasets:
        df_dataset = df[df['dataset'] == dataset]

        # Best configuration
        best_idx = df_dataset['test_r2'].idxmax()
        best = df_dataset.loc[best_idx]

        # Worst configuration
        worst_idx = df_dataset['test_r2'].idxmin()
        worst = df_dataset.loc[worst_idx]

        summary_data.append({
            'Dataset': dataset.replace('_', ' '),
            'Best Algo': best['algorithm'],
            'Best R²': f"{best['test_r2']:.4f}",
            'Best RMSE': f"{best['test_rmse']:.4f}",
            'Worst Algo': worst['algorithm'],
            'Worst R²': f"{worst['test_r2']:.4f}",
            'R² Spread': f"{(best['test_r2'] - worst['test_r2']):.4f}",
            'Avg Hidden Dim': f"{int(df_dataset['hidden_dim'].mean())}",
            'Avg Layers': f"{df_dataset['num_layers'].mean():.1f}",
            'Algorithms': len(df_dataset)
        })

    summary_df = pd.DataFrame(summary_data)

    # Save CSV
    summary_df.to_csv(f'{output_dir}/ablation_summary.csv', index=False)

    # Create table visualization
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#e3f2fd')

    plt.title('HPO Results Summary - Algorithm Comparison Across Datasets',
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{output_dir}/ablation_summary_table.png',
               dpi=300, bbox_inches='tight')
    plt.close()

    print("  [OK] Summary table saved")

    # Print summary (handle encoding issues)
    try:
        print(f"\n{summary_df.to_string(index=False)}")
    except UnicodeEncodeError:
        print("\n[Summary table generated - see ablation_summary.csv for details]")

    return summary_df


def main():
    """Run comprehensive ablation studies."""
    print("="*80)
    print("COMPREHENSIVE ABLATION STUDIES FROM HPO RESULTS")
    print("="*80)

    # Load HPO results
    df = load_hpo_results()

    if len(df) == 0:
        print("\n[ERROR] No HPO results found!")
        return

    # Create output directory
    output_dir = 'figures/ablation_studies'
    os.makedirs(output_dir, exist_ok=True)

    # Generate all ablation studies
    create_hyperparameter_sensitivity_plots(df, output_dir)
    create_unified_sensitivity_heatmaps(df, output_dir)
    create_performance_landscape_plots(df, output_dir)
    create_ablation_component_analysis(df, output_dir)
    summary_df = create_comprehensive_summary_table(df, output_dir)

    print("\n" + "="*80)
    print("[OK] ALL ABLATION STUDIES COMPLETED!")
    print("="*80)

    print(f"\nGenerated files in: {output_dir}/")
    print("\nPer-dataset files:")
    for dataset in df['dataset'].unique():
        print(f"  - {dataset}_hyperparameter_sensitivity.png")
        print(f"  - {dataset}_performance_landscape.png")

    print("\nUnified files:")
    print("  - unified_hyperparameter_sensitivity.png")
    print("  - unified_ablation_component_impact.png")
    print("  - ablation_summary_table.png")
    print("  - ablation_summary.csv")

    print("\n[DONE] Comprehensive ablation studies complete!")


if __name__ == "__main__":
    main()
