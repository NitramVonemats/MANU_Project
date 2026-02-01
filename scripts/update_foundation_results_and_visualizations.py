"""
UPDATE FOUNDATION RESULTS AND VISUALIZATIONS
=============================================

After MolCLR pretrained benchmark completes, this script:
1. Updates foundation_comparison_COMPLETE.csv with new MolCLR results
2. Regenerates all foundation comparison figures
3. Creates summary report
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

RESULTS_DIR = os.path.join(project_root, 'results', 'foundation_benchmark')
FIGURES_DIR = os.path.join(project_root, 'figures', 'foundation')

# Dataset info
REGRESSION_DATASETS = ['Caco2_Wang', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']
CLASSIFICATION_DATASETS = ['tox21', 'herg']
ALL_DATASETS = REGRESSION_DATASETS + CLASSIFICATION_DATASETS


def load_latest_molclr_results():
    """Load most recent MolCLR pretrained results"""
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('molclr_pretrained_results_')]
    if not files:
        raise FileNotFoundError("No MolCLR pretrained results found. Run benchmark first.")
    latest = sorted(files)[-1]
    path = os.path.join(RESULTS_DIR, latest)
    print(f"Loading MolCLR results from: {latest}")
    return pd.read_csv(path)


def update_foundation_comparison():
    """Update foundation comparison CSV with new MolCLR results"""
    # Load existing results
    existing_path = os.path.join(RESULTS_DIR, 'foundation_comparison_COMPLETE.csv')
    df_existing = pd.read_csv(existing_path)

    # Load new MolCLR results
    df_molclr = load_latest_molclr_results()

    # Remove old MolCLR rows
    df_updated = df_existing[df_existing['model'] != 'MolCLR'].copy()

    # Rename model to MolCLR (was MolCLR-Pretrained)
    df_molclr['model'] = 'MolCLR'

    # Append new results
    df_updated = pd.concat([df_updated, df_molclr], ignore_index=True)

    # Save updated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f'foundation_comparison_UPDATED_{timestamp}.csv')
    df_updated.to_csv(output_path, index=False)

    # Also update the main file
    df_updated.to_csv(existing_path, index=False)
    print(f"Updated: {existing_path}")

    return df_updated


def generate_foundation_figures(df):
    """Generate all foundation comparison figures"""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

    # 1. GNN vs Foundation Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Regression subplot
    ax1 = axes[0]
    reg_data = df[(df['task_type'] == 'regression') & (df['status'] == 'success')]
    models = ['GNN-Best', 'Morgan-FP', 'ChemBERTa', 'MolCLR', 'MolE-FP']

    for i, dataset in enumerate(REGRESSION_DATASETS):
        subset = reg_data[reg_data['dataset'] == dataset]
        rmses = []
        for model in models:
            row = subset[subset['model'] == model]
            rmses.append(row['test_rmse'].values[0] if len(row) > 0 else np.nan)
        x = np.arange(len(models))
        ax1.bar(x + i*0.15, rmses, width=0.15, label=dataset)

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Test RMSE (lower is better)')
    ax1.set_title('ADME Regression Performance')
    ax1.set_xticks(np.arange(len(models)) + 0.3)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_yscale('log')

    # Classification subplot
    ax2 = axes[1]
    clf_data = df[(df['task_type'] == 'classification') & (df['status'] == 'success')]

    for i, dataset in enumerate(CLASSIFICATION_DATASETS):
        subset = clf_data[clf_data['dataset'] == dataset]
        aucs = []
        for model in models:
            row = subset[subset['model'] == model]
            aucs.append(row['test_auc'].values[0] if len(row) > 0 else np.nan)
        x = np.arange(len(models))
        ax2.bar(x + i*0.35, aucs, width=0.35, label=dataset)

    ax2.set_xlabel('Model')
    ax2.set_ylabel('Test AUC-ROC (higher is better)')
    ax2.set_title('Toxicity Classification Performance')
    ax2.set_xticks(np.arange(len(models)) + 0.175)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(loc='lower right')
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    ax2.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'gnn_vs_foundation_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: gnn_vs_foundation_comparison.png")

    # 2. Performance Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Regression heatmap (normalized RMSE)
    reg_matrix = []
    for dataset in REGRESSION_DATASETS:
        row = []
        subset = reg_data[reg_data['dataset'] == dataset]
        for model in models:
            r = subset[subset['model'] == model]
            row.append(r['test_rmse'].values[0] if len(r) > 0 else np.nan)
        # Normalize by max
        row = np.array(row)
        row = row / np.nanmax(row)
        reg_matrix.append(row)

    reg_matrix = np.array(reg_matrix)
    im1 = axes[0].imshow(reg_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_yticks(range(len(REGRESSION_DATASETS)))
    axes[0].set_yticklabels(REGRESSION_DATASETS)
    axes[0].set_title('Regression RMSE (normalized, lower=green)')
    plt.colorbar(im1, ax=axes[0])

    # Add values
    for i in range(len(REGRESSION_DATASETS)):
        for j in range(len(models)):
            val = reg_matrix[i, j]
            if not np.isnan(val):
                axes[0].text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9)

    # Classification heatmap (AUC)
    clf_matrix = []
    for dataset in CLASSIFICATION_DATASETS:
        row = []
        subset = clf_data[clf_data['dataset'] == dataset]
        for model in models:
            r = subset[subset['model'] == model]
            row.append(r['test_auc'].values[0] if len(r) > 0 else np.nan)
        clf_matrix.append(row)

    clf_matrix = np.array(clf_matrix)
    im2 = axes[1].imshow(clf_matrix, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_yticks(range(len(CLASSIFICATION_DATASETS)))
    axes[1].set_yticklabels(CLASSIFICATION_DATASETS)
    axes[1].set_title('Classification AUC (higher=green)')
    plt.colorbar(im2, ax=axes[1])

    # Add values
    for i in range(len(CLASSIFICATION_DATASETS)):
        for j in range(len(models)):
            val = clf_matrix[i, j]
            if not np.isnan(val):
                axes[1].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'performance_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: performance_heatmap.png")

    # 3. Foundation Model Ranking
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count wins per model
    wins = {m: 0 for m in models}
    for dataset in REGRESSION_DATASETS:
        subset = reg_data[reg_data['dataset'] == dataset]
        if len(subset) > 0:
            best = subset.loc[subset['test_rmse'].idxmin(), 'model']
            if best in wins:
                wins[best] += 1

    for dataset in CLASSIFICATION_DATASETS:
        subset = clf_data[clf_data['dataset'] == dataset]
        if len(subset) > 0:
            best = subset.loc[subset['test_auc'].idxmax(), 'model']
            if best in wins:
                wins[best] += 1

    # Plot
    colors = ['#2ecc71' if m == 'GNN-Best' else '#3498db' for m in models]
    bars = ax.bar(models, [wins[m] for m in models], color=colors)
    ax.set_xlabel('Model')
    ax.set_ylabel('Number of Dataset Wins')
    ax.set_title('Foundation Model Ranking: Dataset Wins')
    ax.set_ylim(0, 6)

    # Add value labels
    for bar, m in zip(bars, models):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{wins[m]}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'foundation_ranking.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: foundation_ranking.png")

    # 4. Summary comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate average normalized performance
    avg_perf = {}
    for model in models:
        reg_scores = []
        for dataset in REGRESSION_DATASETS:
            subset = reg_data[reg_data['dataset'] == dataset]
            all_rmse = subset['test_rmse'].dropna()
            if len(all_rmse) > 0:
                best_rmse = all_rmse.min()
                row = subset[subset['model'] == model]
                if len(row) > 0:
                    reg_scores.append(best_rmse / row['test_rmse'].values[0])

        clf_scores = []
        for dataset in CLASSIFICATION_DATASETS:
            subset = clf_data[clf_data['dataset'] == dataset]
            row = subset[subset['model'] == model]
            if len(row) > 0:
                clf_scores.append(row['test_auc'].values[0])

        avg_perf[model] = {
            'reg': np.mean(reg_scores) if reg_scores else 0,
            'clf': np.mean(clf_scores) if clf_scores else 0
        }

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, [avg_perf[m]['reg'] for m in models], width, label='Regression (norm)')
    rects2 = ax.bar(x + width/2, [avg_perf[m]['clf'] for m in models], width, label='Classification (AUC)')

    ax.set_xlabel('Model')
    ax.set_ylabel('Average Performance')
    ax.set_title('Summary: Average Performance by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'summary_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: summary_comparison.png")


def print_summary(df):
    """Print summary of results"""
    print("\n" + "="*70)
    print("UPDATED FOUNDATION MODEL COMPARISON")
    print("="*70)

    print("\n--- REGRESSION (RMSE - lower is better) ---")
    print(f"{'Dataset':<30} {'GNN-Best':<12} {'Morgan-FP':<12} {'ChemBERTa':<12} {'MolCLR':<12} {'MolE-FP':<12}")
    print("-"*90)

    reg_data = df[(df['task_type'] == 'regression') & (df['status'] == 'success')]
    for dataset in REGRESSION_DATASETS:
        subset = reg_data[reg_data['dataset'] == dataset]
        row = f"{dataset:<30}"
        for model in ['GNN-Best', 'Morgan-FP', 'ChemBERTa', 'MolCLR', 'MolE-FP']:
            r = subset[subset['model'] == model]
            val = f"{r['test_rmse'].values[0]:.4f}" if len(r) > 0 else "N/A"
            row += f" {val:<12}"
        print(row)

    print("\n--- CLASSIFICATION (AUC - higher is better) ---")
    print(f"{'Dataset':<30} {'GNN-Best':<12} {'Morgan-FP':<12} {'ChemBERTa':<12} {'MolCLR':<12} {'MolE-FP':<12}")
    print("-"*90)

    clf_data = df[(df['task_type'] == 'classification') & (df['status'] == 'success')]
    for dataset in CLASSIFICATION_DATASETS:
        subset = clf_data[clf_data['dataset'] == dataset]
        row = f"{dataset:<30}"
        for model in ['GNN-Best', 'Morgan-FP', 'ChemBERTa', 'MolCLR', 'MolE-FP']:
            r = subset[subset['model'] == model]
            val = f"{r['test_auc'].values[0]:.4f}" if len(r) > 0 else "N/A"
            row += f" {val:<12}"
        print(row)


def main():
    print("Updating Foundation Model Results with Pretrained MolCLR...")

    # Update results
    df = update_foundation_comparison()

    # Generate figures
    generate_foundation_figures(df)

    # Print summary
    print_summary(df)

    print("\n" + "="*70)
    print("UPDATE COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
