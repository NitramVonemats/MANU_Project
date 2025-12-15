"""
ABLATION STUDIES FROM HPO RESULTS
==================================

Generate ablation study visualizations from HPO results.
Shows how different hyperparameters affect model performance.

Analyses:
1. Hidden Dimensions Impact
2. Graph Layers Impact
3. Learning Rate Impact
4. Weight Decay Impact
5. Algorithm Sensitivity
"""

import os
import json
import pandas as pd
import numpy as np

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Create output directory
os.makedirs('figures/ablation_studies', exist_ok=True)


def load_hpo_results():
    """Load all HPO results from runs/ directory."""
    results = []

    datasets = ['Caco2_Wang', 'Clearance_Hepatocyte_AZ',
                'Clearance_Microsome_AZ', 'Half_Life_Obach']
    algorithms = ['random', 'pso', 'ga', 'sa', 'hc', 'abc']

    for dataset in datasets:
        for algo in algorithms:
            path = f'runs/{dataset}/hpo_{dataset}_{algo}.json'
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                    results.append({
                        'Dataset': dataset,
                        'Algorithm': algo.upper(),
                        'Test RMSE': data['final_training']['test_metrics']['rmse'],
                        'Test R²': data['final_training']['test_metrics']['r2'],
                        'Val RMSE': data['final_training']['val_metrics']['rmse'],
                        'Train Time': data['final_training']['train_time'],
                        'Hidden Dim': data['search']['best_params']['hidden_dim'],
                        'Num Layers': data['search']['best_params']['num_layers'],
                        'Learning Rate': data['search']['best_params']['lr'],
                        'Weight Decay': data['search']['best_params']['weight_decay'],
                    })

    return pd.DataFrame(results)


def plot_hidden_dim_impact(df):
    """Analyze impact of hidden dimensions."""
    print("\n[1/5] Creating Hidden Dimensions Impact analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    datasets = df['Dataset'].unique()
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']

    # Subplot 1: Scatter plot - Hidden Dim vs Test RMSE
    ax = axes[0, 0]
    for dataset, color in zip(datasets, colors):
        subset = df[df['Dataset'] == dataset]
        # Filter out extreme RMSE values
        subset = subset[subset['Test RMSE'] < 100]
        ax.scatter(subset['Hidden Dim'], subset['Test RMSE'],
                  s=100, alpha=0.6, label=dataset, color=color, edgecolor='black')

    ax.set_xlabel('Hidden Dimensions', fontweight='bold')
    ax.set_ylabel('Test RMSE', fontweight='bold')
    ax.set_title('(A) Hidden Dimensions vs Test RMSE', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.legend()
    ax.grid(alpha=0.3)

    # Subplot 2: Scatter plot - Hidden Dim vs Test R²
    ax = axes[0, 1]
    for dataset, color in zip(datasets, colors):
        subset = df[df['Dataset'] == dataset]
        ax.scatter(subset['Hidden Dim'], subset['Test R²'],
                  s=100, alpha=0.6, label=dataset, color=color, edgecolor='black')

    ax.set_xlabel('Hidden Dimensions', fontweight='bold')
    ax.set_ylabel('Test R²', fontweight='bold')
    ax.set_title('(B) Hidden Dimensions vs Test R²', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)

    # Subplot 3: Best hidden dim per dataset
    ax = axes[1, 0]
    best_configs = []
    for dataset in datasets:
        subset = df[df['Dataset'] == dataset]
        subset_filtered = subset[subset['Test RMSE'] < 100]
        if len(subset_filtered) > 0:
            best = subset_filtered.loc[subset_filtered['Test RMSE'].idxmin()]
            best_configs.append({
                'Dataset': dataset,
                'Best Hidden Dim': best['Hidden Dim'],
                'Test RMSE': best['Test RMSE']
            })

    best_df = pd.DataFrame(best_configs)
    bars = ax.bar(range(len(best_df)), best_df['Best Hidden Dim'],
                 color=colors[:len(best_df)], edgecolor='black', alpha=0.8)

    ax.set_xticks(range(len(best_df)))
    ax.set_xticklabels(best_df['Dataset'], rotation=45, ha='right')
    ax.set_ylabel('Best Hidden Dimensions', fontweight='bold')
    ax.set_title('(C) Optimal Hidden Dimensions per Dataset', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, hd, rmse) in enumerate(zip(bars, best_df['Best Hidden Dim'], best_df['Test RMSE'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(hd)}\n(RMSE={rmse:.3f})',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 4: Distribution of hidden dims tested
    ax = axes[1, 1]
    hidden_dims = df['Hidden Dim'].unique()
    counts = df['Hidden Dim'].value_counts().sort_index()

    ax.bar(counts.index, counts.values, color='skyblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Hidden Dimensions', fontweight='bold')
    ax.set_ylabel('Frequency (HPO runs)', fontweight='bold')
    ax.set_title('(D) Hidden Dimensions Explored by HPO', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Ablation Study: Hidden Dimensions Impact', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('figures/ablation_studies/01_hidden_dimensions_impact.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/ablation_studies/01_hidden_dimensions_impact.png")
    plt.close()


def plot_num_layers_impact(df):
    """Analyze impact of number of layers."""
    print("\n[2/5] Creating Graph Layers Impact analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    datasets = df['Dataset'].unique()
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']

    # Subplot 1: Scatter plot - Num Layers vs Test RMSE
    ax = axes[0, 0]
    for dataset, color in zip(datasets, colors):
        subset = df[df['Dataset'] == dataset]
        subset = subset[subset['Test RMSE'] < 100]
        ax.scatter(subset['Num Layers'], subset['Test RMSE'],
                  s=100, alpha=0.6, label=dataset, color=color, edgecolor='black')

    ax.set_xlabel('Number of GNN Layers', fontweight='bold')
    ax.set_ylabel('Test RMSE', fontweight='bold')
    ax.set_title('(A) Number of Layers vs Test RMSE', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(df['Num Layers'].unique())

    # Subplot 2: Scatter plot - Num Layers vs Test R²
    ax = axes[0, 1]
    for dataset, color in zip(datasets, colors):
        subset = df[df['Dataset'] == dataset]
        ax.scatter(subset['Num Layers'], subset['Test R²'],
                  s=100, alpha=0.6, label=dataset, color=color, edgecolor='black')

    ax.set_xlabel('Number of GNN Layers', fontweight='bold')
    ax.set_ylabel('Test R²', fontweight='bold')
    ax.set_title('(B) Number of Layers vs Test R²', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(df['Num Layers'].unique())

    # Subplot 3: Best num layers per dataset
    ax = axes[1, 0]
    best_configs = []
    for dataset in datasets:
        subset = df[df['Dataset'] == dataset]
        subset_filtered = subset[subset['Test RMSE'] < 100]
        if len(subset_filtered) > 0:
            best = subset_filtered.loc[subset_filtered['Test RMSE'].idxmin()]
            best_configs.append({
                'Dataset': dataset,
                'Best Num Layers': best['Num Layers'],
                'Test RMSE': best['Test RMSE']
            })

    best_df = pd.DataFrame(best_configs)
    bars = ax.bar(range(len(best_df)), best_df['Best Num Layers'],
                 color=colors[:len(best_df)], edgecolor='black', alpha=0.8)

    ax.set_xticks(range(len(best_df)))
    ax.set_xticklabels(best_df['Dataset'], rotation=45, ha='right')
    ax.set_ylabel('Best Number of Layers', fontweight='bold')
    ax.set_title('(C) Optimal Number of Layers per Dataset', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.grid(axis='y', alpha=0.3)
    ax.set_yticks([2, 3, 4, 5])

    # Add value labels
    for i, (bar, nl, rmse) in enumerate(zip(bars, best_df['Best Num Layers'], best_df['Test RMSE'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(nl)} layers\n(RMSE={rmse:.3f})',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 4: Distribution of layers tested
    ax = axes[1, 1]
    layer_counts = df['Num Layers'].value_counts().sort_index()

    ax.bar(layer_counts.index, layer_counts.values, color='lightcoral', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Number of GNN Layers', fontweight='bold')
    ax.set_ylabel('Frequency (HPO runs)', fontweight='bold')
    ax.set_title('(D) Number of Layers Explored by HPO', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(layer_counts.index)

    plt.suptitle('Ablation Study: Graph Layers Impact', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('figures/ablation_studies/02_graph_layers_impact.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/ablation_studies/02_graph_layers_impact.png")
    plt.close()


def plot_learning_rate_impact(df):
    """Analyze impact of learning rate."""
    print("\n[3/5] Creating Learning Rate Impact analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    datasets = df['Dataset'].unique()
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']

    # Subplot 1: Scatter plot - LR vs Test RMSE (log scale)
    ax = axes[0, 0]
    for dataset, color in zip(datasets, colors):
        subset = df[df['Dataset'] == dataset]
        subset = subset[subset['Test RMSE'] < 100]
        ax.scatter(subset['Learning Rate'], subset['Test RMSE'],
                  s=100, alpha=0.6, label=dataset, color=color, edgecolor='black')

    ax.set_xlabel('Learning Rate (log scale)', fontweight='bold')
    ax.set_ylabel('Test RMSE', fontweight='bold')
    ax.set_title('(A) Learning Rate vs Test RMSE', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    # Subplot 2: Scatter plot - LR vs Test R²
    ax = axes[0, 1]
    for dataset, color in zip(datasets, colors):
        subset = df[df['Dataset'] == dataset]
        ax.scatter(subset['Learning Rate'], subset['Test R²'],
                  s=100, alpha=0.6, label=dataset, color=color, edgecolor='black')

    ax.set_xlabel('Learning Rate (log scale)', fontweight='bold')
    ax.set_ylabel('Test R²', fontweight='bold')
    ax.set_title('(B) Learning Rate vs Test R²', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.set_xscale('log')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)

    # Subplot 3: Best LR per dataset
    ax = axes[1, 0]
    best_configs = []
    for dataset in datasets:
        subset = df[df['Dataset'] == dataset]
        subset_filtered = subset[subset['Test RMSE'] < 100]
        if len(subset_filtered) > 0:
            best = subset_filtered.loc[subset_filtered['Test RMSE'].idxmin()]
            best_configs.append({
                'Dataset': dataset,
                'Best LR': best['Learning Rate'],
                'Test RMSE': best['Test RMSE']
            })

    best_df = pd.DataFrame(best_configs)
    bars = ax.barh(range(len(best_df)), best_df['Best LR'],
                  color=colors[:len(best_df)], edgecolor='black', alpha=0.8)

    ax.set_yticks(range(len(best_df)))
    ax.set_yticklabels(best_df['Dataset'])
    ax.set_xlabel('Best Learning Rate', fontweight='bold')
    ax.set_title('(C) Optimal Learning Rate per Dataset', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, lr, rmse) in enumerate(zip(bars, best_df['Best LR'], best_df['Test RMSE'])):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'  {lr:.1e} (RMSE={rmse:.3f})',
               ha='left', va='center', fontsize=9, fontweight='bold')

    # Subplot 4: LR distribution
    ax = axes[1, 1]
    lr_bins = np.logspace(np.log10(df['Learning Rate'].min()),
                          np.log10(df['Learning Rate'].max()), 10)
    ax.hist(df['Learning Rate'], bins=lr_bins, edgecolor='black', alpha=0.7, color='lightgreen')
    ax.set_xlabel('Learning Rate (log scale)', fontweight='bold')
    ax.set_ylabel('Frequency (HPO runs)', fontweight='bold')
    ax.set_title('(D) Learning Rates Explored by HPO', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.set_xscale('log')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Ablation Study: Learning Rate Impact', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('figures/ablation_studies/03_learning_rate_impact.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/ablation_studies/03_learning_rate_impact.png")
    plt.close()


def plot_weight_decay_impact(df):
    """Analyze impact of weight decay."""
    print("\n[4/5] Creating Weight Decay Impact analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    datasets = df['Dataset'].unique()
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']

    # Subplot 1: Scatter plot - WD vs Test RMSE
    ax = axes[0, 0]
    for dataset, color in zip(datasets, colors):
        subset = df[df['Dataset'] == dataset]
        subset = subset[subset['Test RMSE'] < 100]
        ax.scatter(subset['Weight Decay'], subset['Test RMSE'],
                  s=100, alpha=0.6, label=dataset, color=color, edgecolor='black')

    ax.set_xlabel('Weight Decay (log scale)', fontweight='bold')
    ax.set_ylabel('Test RMSE', fontweight='bold')
    ax.set_title('(A) Weight Decay vs Test RMSE', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    # Subplot 2: Scatter plot - WD vs Test R²
    ax = axes[0, 1]
    for dataset, color in zip(datasets, colors):
        subset = df[df['Dataset'] == dataset]
        ax.scatter(subset['Weight Decay'], subset['Test R²'],
                  s=100, alpha=0.6, label=dataset, color=color, edgecolor='black')

    ax.set_xlabel('Weight Decay (log scale)', fontweight='bold')
    ax.set_ylabel('Test R²', fontweight='bold')
    ax.set_title('(B) Weight Decay vs Test R²', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.set_xscale('log')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)

    # Subplot 3: Best WD per dataset
    ax = axes[1, 0]
    best_configs = []
    for dataset in datasets:
        subset = df[df['Dataset'] == dataset]
        subset_filtered = subset[subset['Test RMSE'] < 100]
        if len(subset_filtered) > 0:
            best = subset_filtered.loc[subset_filtered['Test RMSE'].idxmin()]
            best_configs.append({
                'Dataset': dataset,
                'Best WD': best['Weight Decay'],
                'Test RMSE': best['Test RMSE']
            })

    best_df = pd.DataFrame(best_configs)
    bars = ax.barh(range(len(best_df)), best_df['Best WD'],
                  color=colors[:len(best_df)], edgecolor='black', alpha=0.8)

    ax.set_yticks(range(len(best_df)))
    ax.set_yticklabels(best_df['Dataset'])
    ax.set_xlabel('Best Weight Decay', fontweight='bold')
    ax.set_title('(C) Optimal Weight Decay per Dataset', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, wd, rmse) in enumerate(zip(bars, best_df['Best WD'], best_df['Test RMSE'])):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'  {wd:.1e} (RMSE={rmse:.3f})',
               ha='left', va='center', fontsize=9, fontweight='bold')

    # Subplot 4: WD distribution
    ax = axes[1, 1]
    wd_bins = np.logspace(np.log10(df['Weight Decay'].min()),
                          np.log10(df['Weight Decay'].max()), 10)
    ax.hist(df['Weight Decay'], bins=wd_bins, edgecolor='black', alpha=0.7, color='plum')
    ax.set_xlabel('Weight Decay (log scale)', fontweight='bold')
    ax.set_ylabel('Frequency (HPO runs)', fontweight='bold')
    ax.set_title('(D) Weight Decay Values Explored by HPO', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.set_xscale('log')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Ablation Study: Weight Decay Impact', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('figures/ablation_studies/04_weight_decay_impact.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/ablation_studies/04_weight_decay_impact.png")
    plt.close()


def plot_algorithm_sensitivity(df):
    """Analyze algorithm sensitivity to hyperparameters."""
    print("\n[5/5] Creating Algorithm Sensitivity analysis...")

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    datasets = df['Dataset'].unique()
    algorithms = df['Algorithm'].unique()

    # Subplots for each dataset
    for idx, dataset in enumerate(datasets):
        ax = fig.add_subplot(gs[idx // 2, idx % 3])
        subset = df[df['Dataset'] == dataset]
        subset_filtered = subset[subset['Test RMSE'] < 100]

        if len(subset_filtered) > 0:
            algo_performance = subset_filtered.groupby('Algorithm')['Test RMSE'].agg(['mean', 'std']).sort_values('mean')

            x = range(len(algo_performance))
            y = algo_performance['mean']
            yerr = algo_performance['std']

            colors_algo = ['#FF6B6B' if 'ABC' in alg else '#4ECDC4' if 'SA' in alg else '#45B7D1' if 'GA' in alg
                          else '#FFA07A' if 'PSO' in alg else '#98D8C8' if 'HC' in alg else '#CCCCCC'
                          for alg in algo_performance.index]

            bars = ax.barh(x, y, xerr=yerr, color=colors_algo, edgecolor='black', alpha=0.8, capsize=5)

            ax.set_yticks(x)
            ax.set_yticklabels(algo_performance.index)
            ax.set_xlabel('Mean Test RMSE ± Std', fontweight='bold')
            ax.set_title(f'({chr(65+idx)}) {dataset}', fontsize=11, fontweight='bold', loc='left')
            ax.grid(axis='x', alpha=0.3)

            # Highlight best
            ax.get_children()[0].set_linewidth(3)

    plt.suptitle('Ablation Study: Algorithm Sensitivity Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('figures/ablation_studies/05_algorithm_sensitivity.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/ablation_studies/05_algorithm_sensitivity.png")
    plt.close()


def main():
    """Generate all ablation studies from HPO results."""
    print("="*80)
    print("GENERATING ABLATION STUDIES FROM HPO RESULTS")
    print("="*80)

    # Load HPO results
    print("\nLoading HPO results...")
    df = load_hpo_results()

    if df.empty:
        print("\n[ERROR] No HPO results found!")
        print("Please run HPO first using scripts/run_hpo.py")
        return

    print(f"Loaded {len(df)} HPO results")
    print(f"Datasets: {df['Dataset'].unique().tolist()}")
    print(f"Algorithms: {df['Algorithm'].unique().tolist()}")

    # Generate ablation studies
    plot_hidden_dim_impact(df)
    plot_num_layers_impact(df)
    plot_learning_rate_impact(df)
    plot_weight_decay_impact(df)
    plot_algorithm_sensitivity(df)

    print("\n" + "="*80)
    print("[OK] ALL ABLATION STUDIES GENERATED!")
    print("="*80)
    print("\nGenerated files in figures/ablation_studies/:")
    print("  1. 01_hidden_dimensions_impact.png")
    print("  2. 02_graph_layers_impact.png")
    print("  3. 03_learning_rate_impact.png")
    print("  4. 04_weight_decay_impact.png")
    print("  5. 05_algorithm_sensitivity.png")
    print("\n✅ All ablation studies show results for ALL 4 ADME datasets!")


if __name__ == "__main__":
    main()
