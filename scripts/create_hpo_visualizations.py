"""
Create unified HPO results visualizations.
All HPO results on unified plots for easy comparison.
"""
import os
import json
import pandas as pd
import numpy as np

# Set matplotlib backend before importing pyplot
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
os.makedirs('figures/hpo', exist_ok=True)


def load_hpo_results():
    """Load all HPO results from runs/ directory."""
    regression_results = []
    classification_results = []

    # ADME datasets (regression)
    adme_datasets = ['Caco2_Wang', 'Clearance_Hepatocyte_AZ',
                     'Clearance_Microsome_AZ', 'Half_Life_Obach']
    # Toxicity datasets (classification)
    tox_datasets = ['tox21', 'herg']

    algorithms = ['random', 'pso', 'ga', 'sa', 'hc', 'abc']

    # Load ADME (regression) results
    for dataset in adme_datasets:
        for algo in algorithms:
            path = f'runs/{dataset}/hpo_{dataset}_{algo}.json'
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                    regression_results.append({
                        'Dataset': dataset,
                        'Algorithm': algo.upper(),
                        'Test RMSE': data['final_training']['test_metrics']['rmse'],
                        'Test MAE': data['final_training']['test_metrics']['mae'],
                        'Test R²': data['final_training']['test_metrics']['r2'],
                        'Val RMSE': data['final_training']['val_metrics']['rmse'],
                        'Train Time (s)': data['final_training']['train_time'],
                        'HPO Trials': data['search']['trials'],
                        'Hidden Dim': data['search']['best_params']['hidden_dim'],
                        'Num Layers': data['search']['best_params']['num_layers'],
                        'Learning Rate': data['search']['best_params']['lr'],
                        'Weight Decay': data['search']['best_params']['weight_decay'],
                    })
                print(f"Loaded (regression): {dataset} - {algo}")

    # Load Toxicity (classification) results
    for dataset in tox_datasets:
        for algo in algorithms:
            path = f'runs/{dataset}/hpo_{dataset}_{algo}.json'
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                    classification_results.append({
                        'Dataset': dataset,
                        'Algorithm': algo.upper(),
                        'Test AUC-ROC': data['final_training']['test_metrics']['auc_roc'],
                        'Test F1': data['final_training']['test_metrics']['f1'],
                        'Test Accuracy': data['final_training']['test_metrics']['accuracy'],
                        'Val AUC-ROC': data['final_training']['val_metrics']['auc_roc'],
                        'Val F1': data['final_training']['val_metrics']['f1'],
                        'Train Time (s)': data['final_training']['train_time'],
                        'HPO Trials': data['search']['trials'],
                        'Hidden Dim': data['search']['best_params']['hidden_dim'],
                        'Num Layers': data['search']['best_params']['num_layers'],
                        'Learning Rate': data['search']['best_params']['lr'],
                        'Weight Decay': data['search']['best_params']['weight_decay'],
                    })
                print(f"Loaded (classification): {dataset} - {algo}")

    return pd.DataFrame(regression_results), pd.DataFrame(classification_results)


def plot_algorithm_performance(results_df):
    """Create unified algorithm performance comparison."""
    print("\n[1/4] Creating algorithm performance comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: Test RMSE comparison
    ax = axes[0, 0]
    pivot = results_df.pivot(index='Algorithm', columns='Dataset', values='Test RMSE')

    x = np.arange(len(pivot))
    width = 0.2
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']

    for i, (dataset, color) in enumerate(zip(pivot.columns, colors)):
        offset = (i - len(pivot.columns)/2) * width + width/2
        bars = ax.bar(x + offset, pivot[dataset], width, label=dataset, color=color,
                     edgecolor='black', linewidth=0.8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height < 100:  # Only label reasonable values
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel('Test RMSE (lower is better)', fontweight='bold')
    ax.set_title('(A) Test RMSE by Algorithm', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, min(80, results_df['Test RMSE'].max() * 1.1))

    # Subplot 2: Test R² comparison
    ax = axes[0, 1]
    pivot = results_df.pivot(index='Algorithm', columns='Dataset', values='Test R²')

    x = np.arange(len(pivot))
    for i, (dataset, color) in enumerate(zip(pivot.columns, colors)):
        offset = (i - len(pivot.columns)/2) * width + width/2
        bars = ax.bar(x + offset, pivot[dataset], width, label=dataset, color=color,
                     edgecolor='black', linewidth=0.8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., max(height, 0) + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel('Test R² (higher is better)', fontweight='bold')
    ax.set_title('(B) Test R² by Algorithm', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Subplot 3: Training time comparison
    ax = axes[1, 0]
    pivot = results_df.pivot(index='Algorithm', columns='Dataset', values='Train Time (s)')

    x = np.arange(len(pivot))
    for i, (dataset, color) in enumerate(zip(pivot.columns, colors)):
        offset = (i - len(pivot.columns)/2) * width + width/2
        ax.bar(x + offset, pivot[dataset], width, label=dataset, color=color,
              edgecolor='black', linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax.set_title('(C) Training Time by Algorithm', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Subplot 4: Performance vs Time trade-off
    ax = axes[1, 1]
    datasets = results_df['Dataset'].unique()

    for dataset, color in zip(datasets, colors):
        subset = results_df[results_df['Dataset'] == dataset]
        # Filter out extreme RMSE values for better visualization
        subset_filtered = subset[subset['Test RMSE'] < 100]
        ax.scatter(subset_filtered['Train Time (s)'], subset_filtered['Test RMSE'],
                  s=120, alpha=0.7, label=dataset, color=color, edgecolor='black', linewidth=1)

        # Annotate algorithm names
        for _, row in subset_filtered.iterrows():
            ax.annotate(row['Algorithm'], (row['Train Time (s)'], row['Test RMSE']),
                       fontsize=7, alpha=0.8, ha='center', va='bottom')

    ax.set_xlabel('Training Time (seconds)', fontweight='bold')
    ax.set_ylabel('Test RMSE (lower is better)', fontweight='bold')
    ax.set_title('(D) Performance vs Training Time Trade-off', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle('HPO Algorithm Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('figures/hpo/01_algorithm_performance.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/hpo/01_algorithm_performance.png")
    plt.close()


def plot_best_hyperparameters(results_df):
    """Create unified best hyperparameters heatmaps."""
    print("\n[2/4] Creating best hyperparameters heatmaps...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    hyperparams = [
        ('Hidden Dim', 'YlOrRd'),
        ('Num Layers', 'YlGnBu'),
        ('Learning Rate', 'Purples'),
        ('Weight Decay', 'Greens')
    ]

    for idx, (param, cmap) in enumerate(hyperparams):
        ax = axes[idx // 2, idx % 2]

        pivot = results_df.pivot(index='Algorithm', columns='Dataset', values=param)

        if param in ['Learning Rate', 'Weight Decay']:
            # Use scientific notation
            im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')
            for i in range(len(pivot)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        text = ax.text(j, i, f'{val:.1e}',
                                      ha="center", va="center", color="black", fontsize=9)
        else:
            # Use regular numbers
            im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')
            for i in range(len(pivot)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        text = ax.text(j, i, f'{int(val)}',
                                      ha="center", va="center", color="black", fontsize=9)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(param, fontweight='bold')

        ax.set_title(f'({chr(65+idx)}) Best {param}', fontsize=12, fontweight='bold', loc='left', pad=15)

    plt.suptitle('Best Hyperparameters Discovered by HPO', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('figures/hpo/02_best_hyperparameters.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/hpo/02_best_hyperparameters.png")
    plt.close()


def plot_winner_analysis(results_df):
    """Create unified winner analysis."""
    print("\n[3/4] Creating winner analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Find best algorithm per dataset
    winners = []
    for dataset in results_df['Dataset'].unique():
        dataset_results = results_df[results_df['Dataset'] == dataset]
        # Filter out extreme values
        dataset_results_filtered = dataset_results[dataset_results['Test RMSE'] < 100]
        if len(dataset_results_filtered) > 0:
            best = dataset_results_filtered.loc[dataset_results_filtered['Test RMSE'].idxmin()]
            worst_rmse = dataset_results['Test RMSE'].max()
            if worst_rmse < 100:  # Only calculate improvement if reasonable
                improvement = (worst_rmse - best['Test RMSE']) / worst_rmse * 100
            else:
                improvement = 0
            winners.append({
                'Dataset': dataset,
                'Winner': best['Algorithm'],
                'Test RMSE': best['Test RMSE'],
                'Test R²': best['Test R²'],
                'Improvement': improvement
            })

    winners_df = pd.DataFrame(winners)

    # Subplot 1: Best algorithm by dataset
    ax = axes[0]
    colors_dict = {'ABC': '#FF6B6B', 'SA': '#4ECDC4', 'GA': '#45B7D1',
                   'PSO': '#FFA07A', 'HC': '#98D8C8', 'RANDOM': '#CCCCCC'}
    bar_colors = [colors_dict.get(w, '#CCCCCC') for w in winners_df['Winner']]

    bars = ax.bar(range(len(winners_df)), winners_df['Test RMSE'],
                 color=bar_colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    ax.set_xticks(range(len(winners_df)))
    ax.set_xticklabels(winners_df['Dataset'], rotation=45, ha='right')
    ax.set_ylabel('Test RMSE (lower is better)', fontweight='bold')
    ax.set_title('(A) Best Algorithm per Dataset', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.grid(axis='y', alpha=0.3)

    # Add algorithm labels and R² on bars
    for i, (bar, winner, r2) in enumerate(zip(bars, winners_df['Winner'], winners_df['Test R²'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{winner}\n(R²={r2:.3f})',
                ha='center', va='bottom', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Subplot 2: Algorithm win count
    ax = axes[1]
    win_counts = winners_df['Winner'].value_counts()

    bars = ax.bar(range(len(win_counts)), win_counts.values,
                 color=[colors_dict.get(alg, '#CCCCCC') for alg in win_counts.index],
                 edgecolor='black', linewidth=1.5, alpha=0.8)

    ax.set_xticks(range(len(win_counts)))
    ax.set_xticklabels(win_counts.index)
    ax.set_ylabel('Number of Datasets Won', fontweight='bold')
    ax.set_title('(B) Algorithm Win Count', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(win_counts.values) + 1)

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, win_counts.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.suptitle('HPO Winner Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/hpo/03_winner_analysis.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/hpo/03_winner_analysis.png")
    plt.close()


def create_summary_table(results_df):
    """Create HPO summary statistics table."""
    print("\n[4/4] Creating HPO summary statistics table...")

    # Find best per dataset
    summary_data = []
    for dataset in results_df['Dataset'].unique():
        dataset_results = results_df[results_df['Dataset'] == dataset]
        dataset_results_filtered = dataset_results[dataset_results['Test RMSE'] < 100]

        if len(dataset_results_filtered) > 0:
            best = dataset_results_filtered.loc[dataset_results_filtered['Test RMSE'].idxmin()]
            summary_data.append({
                'Dataset': dataset,
                'Best Algorithm': best['Algorithm'],
                'Test RMSE': f"{best['Test RMSE']:.4f}",
                'Test R²': f"{best['Test R²']:.4f}",
                'Val RMSE': f"{best['Val RMSE']:.4f}",
                'Train Time': f"{best['Train Time (s)']:.1f}s",
                'Hidden Dim': int(best['Hidden Dim']),
                'Num Layers': int(best['Num Layers']),
                'Learning Rate': f"{best['Learning Rate']:.1e}",
            })

    summary_df = pd.DataFrame(summary_data)

    # Create table visualization
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12]*len(summary_df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)

    # Color header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows alternating
    for i in range(1, len(summary_df) + 1):
        color = '#E7E6E6' if i % 2 == 1 else '#F4B084'
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_edgecolor('black')
            table[(i, j)].set_linewidth(1)

    plt.title('HPO Best Results Summary (Best Configuration per Dataset)',
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig('figures/hpo/04_summary_table.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/hpo/04_summary_table.png")
    plt.close()

    # Also save as CSV
    summary_df.to_csv('figures/hpo/hpo_best_results.csv', index=False)
    print("  Saved: figures/hpo/hpo_best_results.csv")


def plot_classification_performance(clf_df):
    """Create classification performance comparison for toxicity datasets."""
    print("\n[5/6] Creating classification performance comparison...")

    if clf_df.empty:
        print("  No classification results found, skipping...")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ['#E74C3C', '#9B59B6']  # Red for tox21, purple for herg

    # Subplot 1: Test AUC-ROC comparison
    ax = axes[0, 0]
    pivot = clf_df.pivot(index='Algorithm', columns='Dataset', values='Test AUC-ROC')

    x = np.arange(len(pivot))
    width = 0.35

    for i, (dataset, color) in enumerate(zip(pivot.columns, colors)):
        offset = (i - len(pivot.columns)/2) * width + width/2
        bars = ax.bar(x + offset, pivot[dataset], width, label=dataset, color=color,
                     edgecolor='black', linewidth=0.8)
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel('Test AUC-ROC (higher is better)', fontweight='bold')
    ax.set_title('(A) Test AUC-ROC by Algorithm', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    # Subplot 2: Test F1 comparison
    ax = axes[0, 1]
    pivot = clf_df.pivot(index='Algorithm', columns='Dataset', values='Test F1')

    for i, (dataset, color) in enumerate(zip(pivot.columns, colors)):
        offset = (i - len(pivot.columns)/2) * width + width/2
        bars = ax.bar(x + offset, pivot[dataset], width, label=dataset, color=color,
                     edgecolor='black', linewidth=0.8)
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel('Test F1 Score (higher is better)', fontweight='bold')
    ax.set_title('(B) Test F1 Score by Algorithm', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Subplot 3: Training time comparison
    ax = axes[1, 0]
    pivot = clf_df.pivot(index='Algorithm', columns='Dataset', values='Train Time (s)')

    for i, (dataset, color) in enumerate(zip(pivot.columns, colors)):
        offset = (i - len(pivot.columns)/2) * width + width/2
        ax.bar(x + offset, pivot[dataset], width, label=dataset, color=color,
              edgecolor='black', linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax.set_title('(C) Training Time by Algorithm', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Subplot 4: AUC-ROC vs F1 scatter
    ax = axes[1, 1]
    for dataset, color in zip(clf_df['Dataset'].unique(), colors):
        subset = clf_df[clf_df['Dataset'] == dataset]
        ax.scatter(subset['Test AUC-ROC'], subset['Test F1'],
                  s=120, alpha=0.7, label=dataset, color=color, edgecolor='black', linewidth=1)
        for _, row in subset.iterrows():
            ax.annotate(row['Algorithm'], (row['Test AUC-ROC'], row['Test F1']),
                       fontsize=7, alpha=0.8, ha='center', va='bottom')

    ax.set_xlabel('Test AUC-ROC', fontweight='bold')
    ax.set_ylabel('Test F1 Score', fontweight='bold')
    ax.set_title('(D) AUC-ROC vs F1 Trade-off', fontsize=12, fontweight='bold', loc='left', pad=15)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.0)

    plt.suptitle('HPO Classification Performance (Toxicity Datasets)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('figures/hpo/05_classification_performance.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/hpo/05_classification_performance.png")
    plt.close()


def create_classification_summary(clf_df):
    """Create classification summary table for toxicity datasets."""
    print("\n[6/6] Creating classification summary table...")

    if clf_df.empty:
        print("  No classification results found, skipping...")
        return

    summary_data = []
    for dataset in clf_df['Dataset'].unique():
        dataset_results = clf_df[clf_df['Dataset'] == dataset]
        best = dataset_results.loc[dataset_results['Test AUC-ROC'].idxmax()]
        summary_data.append({
            'Dataset': dataset,
            'Best Algorithm': best['Algorithm'],
            'Test AUC-ROC': f"{best['Test AUC-ROC']:.4f}",
            'Test F1': f"{best['Test F1']:.4f}",
            'Test Accuracy': f"{best['Test Accuracy']:.4f}",
            'Val AUC-ROC': f"{best['Val AUC-ROC']:.4f}",
            'Train Time': f"{best['Train Time (s)']:.1f}s",
            'Hidden Dim': int(best['Hidden Dim']),
            'Num Layers': int(best['Num Layers']),
        })

    summary_df = pd.DataFrame(summary_data)

    fig, ax = plt.subplots(figsize=(18, 3))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.11]*len(summary_df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)

    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#E74C3C')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(summary_df) + 1):
        color = '#FADBD8' if i % 2 == 1 else '#F5B7B1'
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_edgecolor('black')
            table[(i, j)].set_linewidth(1)

    plt.title('HPO Best Results - Toxicity Classification',
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig('figures/hpo/06_classification_summary.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/hpo/06_classification_summary.png")
    plt.close()

    summary_df.to_csv('figures/hpo/hpo_classification_results.csv', index=False)
    print("  Saved: figures/hpo/hpo_classification_results.csv")


def main():
    """Main function to create all HPO visualizations."""
    print("="*80)
    print("CREATING UNIFIED HPO VISUALIZATIONS")
    print("All HPO results on unified plots for easy comparison")
    print("="*80)

    # Load HPO results
    print("\nLoading HPO results from runs/...")
    reg_df, clf_df = load_hpo_results()

    if reg_df.empty and clf_df.empty:
        print("\n[ERROR] No HPO results found!")
        print("Please run HPO first using scripts/run_hpo.py")
        return

    print(f"\nLoaded {len(reg_df)} regression results (ADME)")
    print(f"Loaded {len(clf_df)} classification results (Toxicity)")

    if not reg_df.empty:
        print(f"ADME Datasets: {reg_df['Dataset'].unique().tolist()}")
        print(f"Algorithms: {reg_df['Algorithm'].unique().tolist()}")

    if not clf_df.empty:
        print(f"Toxicity Datasets: {clf_df['Dataset'].unique().tolist()}")

    # Create regression visualizations (ADME)
    if not reg_df.empty:
        print("\n--- ADME (Regression) Visualizations ---")
        plot_algorithm_performance(reg_df)
        plot_best_hyperparameters(reg_df)
        plot_winner_analysis(reg_df)
        create_summary_table(reg_df)

    # Create classification visualizations (Toxicity)
    if not clf_df.empty:
        print("\n--- Toxicity (Classification) Visualizations ---")
        plot_classification_performance(clf_df)
        create_classification_summary(clf_df)

    print("\n" + "="*80)
    print("[OK] ALL HPO VISUALIZATIONS CREATED!")
    print("="*80)
    print("\nGenerated files in figures/hpo/:")
    print("  ADME (Regression):")
    print("    1. 01_algorithm_performance.png")
    print("    2. 02_best_hyperparameters.png")
    print("    3. 03_winner_analysis.png")
    print("    4. 04_summary_table.png")
    print("    5. hpo_best_results.csv")
    print("  Toxicity (Classification):")
    print("    6. 05_classification_performance.png")
    print("    7. 06_classification_summary.png")
    print("    8. hpo_classification_results.csv")
    print("\n[OK] All HPO results are now on unified plots for easy comparison!")


if __name__ == "__main__":
    main()
