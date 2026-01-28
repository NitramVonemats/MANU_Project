"""
FOUNDATION MODEL COMPARISON VISUALIZATIONS
==========================================

Creates publication-ready figures comparing foundation models with GNN results:
1. gnn_vs_foundation_comparison.png - Bar chart comparing all models
2. foundation_ranking.png - Ranking of foundation models per dataset
3. performance_heatmap.png - Heatmap of models x datasets
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configuration
RESULTS_DIR = os.path.join(project_root, 'results', 'foundation_benchmark')
OUTPUT_DIR = os.path.join(project_root, 'figures', 'foundation')

# Datasets
ADME_DATASETS = ['Caco2_Wang', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']
TOX_DATASETS = ['tox21', 'herg']

# Color palette
MODEL_COLORS = {
    'GNN-Best': '#2ecc71',      # Green - highlight GNN
    'Morgan-FP': '#3498db',     # Blue
    'ChemBERTa': '#e74c3c',     # Red
    'BioMed-IBM': '#9b59b6',    # Purple
    'MolCLR': '#f39c12',        # Orange
    'MolE-FP': '#1abc9c',       # Teal
}


def load_results():
    """Load benchmark results"""
    # Try to find the latest results file
    results_file = os.path.join(RESULTS_DIR, 'foundation_comparison_COMPLETE.csv')

    if not os.path.exists(results_file):
        # Look for timestamped files
        files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('foundation_comparison') and f.endswith('.csv')]
        if files:
            files.sort(reverse=True)  # Most recent first
            results_file = os.path.join(RESULTS_DIR, files[0])
        else:
            raise FileNotFoundError(f"No results found in {RESULTS_DIR}")

    print(f"Loading results from: {results_file}")
    df = pd.read_csv(results_file)

    # Filter to successful runs only
    df = df[df['status'] == 'success']

    return df


def plot_gnn_vs_foundation_comparison(df, output_path):
    """
    Figure 1: GNN vs Foundation Models Comparison
    4-panel figure showing performance comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GNN vs Foundation Models Comparison', fontsize=14, fontweight='bold')

    # Get models in order
    models = ['GNN-Best', 'Morgan-FP', 'ChemBERTa', 'BioMed-IBM', 'MolCLR', 'MolE-FP']
    models = [m for m in models if m in df['model'].values]

    # Panel A: ADME RMSE
    ax1 = axes[0, 0]
    adme_df = df[df['dataset'].isin(ADME_DATASETS)]

    if not adme_df.empty:
        pivot_rmse = adme_df.pivot_table(index='dataset', columns='model', values='test_rmse')
        pivot_rmse = pivot_rmse.reindex(columns=[m for m in models if m in pivot_rmse.columns])

        x = np.arange(len(pivot_rmse.index))
        width = 0.12
        n_models = len(pivot_rmse.columns)

        for i, model in enumerate(pivot_rmse.columns):
            offset = (i - n_models/2 + 0.5) * width
            color = MODEL_COLORS.get(model, '#95a5a6')
            ax1.bar(x + offset, pivot_rmse[model], width, label=model, color=color, edgecolor='black', linewidth=0.5)

        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Test RMSE (↓ better)')
        ax1.set_title('(A) ADME Regression - RMSE', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.replace('_', '\n') for d in pivot_rmse.index], fontsize=8)
        ax1.legend(loc='upper right', fontsize=7)
        ax1.grid(axis='y', alpha=0.3)

    # Panel B: ADME R²
    ax2 = axes[0, 1]
    if not adme_df.empty:
        pivot_r2 = adme_df.pivot_table(index='dataset', columns='model', values='test_r2')
        pivot_r2 = pivot_r2.reindex(columns=[m for m in models if m in pivot_r2.columns])

        for i, model in enumerate(pivot_r2.columns):
            offset = (i - n_models/2 + 0.5) * width
            color = MODEL_COLORS.get(model, '#95a5a6')
            ax2.bar(x + offset, pivot_r2[model], width, label=model, color=color, edgecolor='black', linewidth=0.5)

        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Test R² (↑ better)')
        ax2.set_title('(B) ADME Regression - R²', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.replace('_', '\n') for d in pivot_r2.index], fontsize=8)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=7)
        ax2.grid(axis='y', alpha=0.3)

    # Panel C: Toxicity AUC-ROC
    ax3 = axes[1, 0]
    tox_df = df[df['dataset'].isin(TOX_DATASETS)]

    if not tox_df.empty and 'test_auc' in tox_df.columns:
        pivot_auc = tox_df.pivot_table(index='dataset', columns='model', values='test_auc')
        pivot_auc = pivot_auc.reindex(columns=[m for m in models if m in pivot_auc.columns])

        x = np.arange(len(pivot_auc.index))
        n_models = len(pivot_auc.columns)

        for i, model in enumerate(pivot_auc.columns):
            offset = (i - n_models/2 + 0.5) * width
            color = MODEL_COLORS.get(model, '#95a5a6')
            ax3.bar(x + offset, pivot_auc[model], width, label=model, color=color, edgecolor='black', linewidth=0.5)

        ax3.set_xlabel('Dataset')
        ax3.set_ylabel('Test AUC-ROC (↑ better)')
        ax3.set_title('(C) Toxicity Classification - AUC-ROC', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(pivot_auc.index, fontsize=9)
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax3.legend(loc='upper right', fontsize=7)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0.4, 1.0)

    # Panel D: Toxicity F1
    ax4 = axes[1, 1]
    if not tox_df.empty and 'test_f1' in tox_df.columns:
        pivot_f1 = tox_df.pivot_table(index='dataset', columns='model', values='test_f1')
        pivot_f1 = pivot_f1.reindex(columns=[m for m in models if m in pivot_f1.columns])

        for i, model in enumerate(pivot_f1.columns):
            offset = (i - n_models/2 + 0.5) * width
            color = MODEL_COLORS.get(model, '#95a5a6')
            ax4.bar(x + offset, pivot_f1[model], width, label=model, color=color, edgecolor='black', linewidth=0.5)

        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Test F1 Score (↑ better)')
        ax4.set_title('(D) Toxicity Classification - F1 Score', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(pivot_f1.index, fontsize=9)
        ax4.legend(loc='upper right', fontsize=7)
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_foundation_ranking(df, output_path):
    """
    Figure 2: Foundation Model Ranking per Dataset
    Shows which model performs best on each dataset
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Foundation Model Rankings by Dataset', fontsize=14, fontweight='bold')

    # Panel A: ADME Rankings (by RMSE - lower is better)
    ax1 = axes[0]
    adme_df = df[df['dataset'].isin(ADME_DATASETS)]

    if not adme_df.empty:
        # Rank models by RMSE (ascending)
        rankings = []
        for dataset in ADME_DATASETS:
            subset = adme_df[adme_df['dataset'] == dataset].copy()
            if not subset.empty:
                subset = subset.sort_values('test_rmse')
                for rank, (_, row) in enumerate(subset.iterrows(), 1):
                    rankings.append({
                        'dataset': dataset,
                        'model': row['model'],
                        'rank': rank,
                        'value': row['test_rmse']
                    })

        if rankings:
            rank_df = pd.DataFrame(rankings)
            pivot = rank_df.pivot_table(index='model', columns='dataset', values='rank')

            # Reorder
            model_order = pivot.mean(axis=1).sort_values().index
            pivot = pivot.reindex(model_order)

            sns.heatmap(pivot, annot=True, cmap='RdYlGn_r', ax=ax1, fmt='.0f',
                       cbar_kws={'label': 'Rank (1=Best)'}, vmin=1, vmax=len(pivot))
            ax1.set_title('(A) ADME Datasets - Ranking by RMSE', fontweight='bold')
            ax1.set_xlabel('Dataset')
            ax1.set_ylabel('Model')

    # Panel B: Toxicity Rankings (by AUC - higher is better)
    ax2 = axes[1]
    tox_df = df[df['dataset'].isin(TOX_DATASETS)]

    if not tox_df.empty and 'test_auc' in tox_df.columns:
        rankings = []
        for dataset in TOX_DATASETS:
            subset = tox_df[tox_df['dataset'] == dataset].copy()
            if not subset.empty:
                subset = subset.sort_values('test_auc', ascending=False)
                for rank, (_, row) in enumerate(subset.iterrows(), 1):
                    rankings.append({
                        'dataset': dataset,
                        'model': row['model'],
                        'rank': rank,
                        'value': row['test_auc']
                    })

        if rankings:
            rank_df = pd.DataFrame(rankings)
            pivot = rank_df.pivot_table(index='model', columns='dataset', values='rank')

            model_order = pivot.mean(axis=1).sort_values().index
            pivot = pivot.reindex(model_order)

            sns.heatmap(pivot, annot=True, cmap='RdYlGn_r', ax=ax2, fmt='.0f',
                       cbar_kws={'label': 'Rank (1=Best)'}, vmin=1, vmax=len(pivot))
            ax2.set_title('(B) Toxicity Datasets - Ranking by AUC-ROC', fontweight='bold')
            ax2.set_xlabel('Dataset')
            ax2.set_ylabel('Model')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_performance_heatmap(df, output_path):
    """
    Figure 3: Performance Heatmap
    Models x Datasets performance matrix
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Performance Heatmap: Models × Datasets', fontsize=14, fontweight='bold')

    # Panel A: ADME (R² scores)
    ax1 = axes[0]
    adme_df = df[df['dataset'].isin(ADME_DATASETS)]

    if not adme_df.empty:
        pivot = adme_df.pivot_table(index='model', columns='dataset', values='test_r2')

        # Annotate with values
        annot = pivot.applymap(lambda x: f'{x:.3f}' if pd.notna(x) else '')

        sns.heatmap(pivot, annot=annot, fmt='', cmap='RdYlGn', ax=ax1,
                   center=0, cbar_kws={'label': 'R² Score'}, vmin=-1, vmax=1)
        ax1.set_title('(A) ADME Datasets - R² Score\n(green=better, red=worse)', fontweight='bold')
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Model')

        # Rotate x labels
        ax1.set_xticklabels([d.replace('_', '\n') for d in pivot.columns], rotation=0, fontsize=8)

    # Panel B: Toxicity (AUC scores)
    ax2 = axes[1]
    tox_df = df[df['dataset'].isin(TOX_DATASETS)]

    if not tox_df.empty and 'test_auc' in tox_df.columns:
        pivot = tox_df.pivot_table(index='model', columns='dataset', values='test_auc')

        annot = pivot.applymap(lambda x: f'{x:.3f}' if pd.notna(x) else '')

        sns.heatmap(pivot, annot=annot, fmt='', cmap='RdYlGn', ax=ax2,
                   center=0.5, cbar_kws={'label': 'AUC-ROC'}, vmin=0.4, vmax=1.0)
        ax2.set_title('(B) Toxicity Datasets - AUC-ROC\n(green=better, red=worse)', fontweight='bold')
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Model')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_comparison(df, output_path):
    """
    Figure 4: Summary Comparison Table as Image
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Create summary data
    summary_data = []

    for dataset in ADME_DATASETS + TOX_DATASETS:
        subset = df[df['dataset'] == dataset]
        if subset.empty:
            continue

        is_tox = dataset in TOX_DATASETS

        if is_tox:
            metric_col = 'test_auc'
            best_idx = subset[metric_col].idxmax() if metric_col in subset.columns else None
        else:
            metric_col = 'test_rmse'
            best_idx = subset[metric_col].idxmin() if metric_col in subset.columns else None

        if best_idx is not None:
            best_row = subset.loc[best_idx]
            winner = best_row['model']

            if is_tox:
                summary_data.append({
                    'Dataset': dataset,
                    'Task': 'Classification',
                    'Winner': winner,
                    'AUC-ROC': f"{best_row.get('test_auc', 'N/A'):.4f}" if pd.notna(best_row.get('test_auc')) else 'N/A',
                    'F1': f"{best_row.get('test_f1', 'N/A'):.4f}" if pd.notna(best_row.get('test_f1')) else 'N/A',
                })
            else:
                summary_data.append({
                    'Dataset': dataset,
                    'Task': 'Regression',
                    'Winner': winner,
                    'RMSE': f"{best_row.get('test_rmse', 'N/A'):.4f}" if pd.notna(best_row.get('test_rmse')) else 'N/A',
                    'R²': f"{best_row.get('test_r2', 'N/A'):.4f}" if pd.notna(best_row.get('test_r2')) else 'N/A',
                })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Create table
        table = ax.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#3498db'] * len(summary_df.columns)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style header
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        # Highlight GNN-Best wins
        for i, row in enumerate(summary_df.itertuples(), 1):
            if 'GNN' in row.Winner:
                for j in range(len(summary_df.columns)):
                    table[(i, j)].set_facecolor('#d5f5e3')

    ax.set_title('Summary: Best Model per Dataset', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all foundation model comparison visualizations"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print("FOUNDATION MODEL COMPARISON VISUALIZATIONS")
    print("="*70)

    try:
        df = load_results()
        print(f"Loaded {len(df)} results")
        print(f"Datasets: {df['dataset'].unique().tolist()}")
        print(f"Models: {df['model'].unique().tolist()}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run 'python scripts/run_complete_foundation_benchmark.py' first")
        return

    print("\nGenerating visualizations...")

    # Generate all plots
    plot_gnn_vs_foundation_comparison(
        df,
        os.path.join(OUTPUT_DIR, 'gnn_vs_foundation_comparison.png')
    )

    plot_foundation_ranking(
        df,
        os.path.join(OUTPUT_DIR, 'foundation_ranking.png')
    )

    plot_performance_heatmap(
        df,
        os.path.join(OUTPUT_DIR, 'performance_heatmap.png')
    )

    plot_summary_comparison(
        df,
        os.path.join(OUTPUT_DIR, 'summary_comparison.png')
    )

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
