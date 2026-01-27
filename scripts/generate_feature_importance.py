#!/usr/bin/env python3
"""
Generate Feature Importance Visualization
Aggregates feature correlations across all datasets.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_all_correlations():
    """Load correlation data from all datasets."""
    base_dir = Path("figures/per_dataset_analysis")

    all_correlations = []

    for dataset_dir in base_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        corr_file = dataset_dir / "correlation_stats.csv"

        if not corr_file.exists():
            continue

        try:
            df = pd.read_csv(corr_file)
            df['Dataset'] = dataset_dir.name
            all_correlations.append(df)
        except Exception as e:
            print(f"Warning: Could not load {corr_file}: {e}")

    if not all_correlations:
        return None

    combined = pd.concat(all_correlations, ignore_index=True)
    return combined


def plot_feature_importance_ranking(correlations, output_dir="figures/paper"):
    """Plot feature importance ranking across all datasets."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Generating Feature Importance Ranking ===")

    # Calculate average absolute correlation for each feature
    feature_importance = correlations.groupby('Feature')['Abs_Correlation'].mean().sort_values(ascending=False)

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(feature_importance))
    colors = plt.cm.RdYlGn_r(feature_importance / feature_importance.max())

    bars = ax.barh(y_pos, feature_importance.values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_importance.index, fontsize=11)
    ax.set_xlabel('Average Absolute Correlation with Target', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Ranking (Averaged Across All Datasets)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, val) in enumerate(feature_importance.items()):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    filename = f"{output_dir}/feature_importance_ranking.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] {filename}")

    # Also save as CSV
    feature_importance.to_csv(f"{output_dir}/feature_importance.csv", header=['Importance'])
    print(f"  [OK] {output_dir}/feature_importance.csv")

    return feature_importance


def plot_feature_importance_heatmap(correlations, output_dir="figures/paper"):
    """Plot feature importance heatmap across datasets."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Generating Feature Importance Heatmap ===")

    # Pivot to create dataset x feature matrix
    pivot = correlations.pivot_table(index='Feature', columns='Dataset',
                                     values='Abs_Correlation', aggfunc='mean')

    # Sort by average importance
    avg_importance = pivot.mean(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[avg_importance.index]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                linewidths=0.5, cbar_kws={'label': 'Absolute Correlation'},
                ax=ax)

    ax.set_title('Feature Importance Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    filename = f"{output_dir}/feature_importance_heatmap.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] {filename}")


def plot_top_features_per_dataset(correlations, output_dir="figures/paper", top_n=5):
    """Plot top N features for each dataset."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Generating Top {top_n} Features Per Dataset ===")

    # Get top features for each dataset
    datasets = correlations['Dataset'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, dataset in enumerate(sorted(datasets)):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Get top features for this dataset
        dataset_data = correlations[correlations['Dataset'] == dataset]
        top_features = dataset_data.nlargest(top_n, 'Abs_Correlation')

        # Plot
        y_pos = np.arange(len(top_features))
        colors = plt.cm.viridis(top_features['Abs_Correlation'] / top_features['Abs_Correlation'].max())

        ax.barh(y_pos, top_features['Abs_Correlation'].values, color=colors,
                edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['Feature'].values, fontsize=9)
        ax.set_xlabel('Abs Correlation', fontsize=10)
        ax.set_title(f'{dataset} (Top {top_n})', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, val in enumerate(top_features['Abs_Correlation'].values):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)

    # Hide unused subplots
    for idx in range(len(datasets), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Top Features by Dataset', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = f"{output_dir}/top_features_per_dataset.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] {filename}")


def main():
    """Generate all feature importance visualizations."""
    print("\n" + "="*80)
    print("GENERATING FEATURE IMPORTANCE VISUALIZATIONS")
    print("="*80)

    # Load correlations
    print("\nLoading feature correlations...")
    correlations = load_all_correlations()

    if correlations is None:
        print("[ERROR] No correlation data found!")
        return

    print(f"  Loaded correlations for {correlations['Dataset'].nunique()} datasets")
    print(f"  Total features: {correlations['Feature'].nunique()}")

    # Create output directory
    output_dir = "figures/paper"
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    plot_feature_importance_ranking(correlations, output_dir)
    plot_feature_importance_heatmap(correlations, output_dir)
    plot_top_features_per_dataset(correlations, output_dir, top_n=5)

    print("\n" + "="*80)
    print("[OK] FEATURE IMPORTANCE VISUALIZATIONS GENERATED")
    print("="*80)
    print(f"\nOutput directory: {output_dir}/")
    print("\nGenerated files:")
    print("  - feature_importance_ranking.png")
    print("  - feature_importance_heatmap.png")
    print("  - top_features_per_dataset.png")
    print("  - feature_importance.csv")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
