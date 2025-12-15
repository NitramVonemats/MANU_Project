"""
COMPREHENSIVE DATASET ANALYSIS - ALL DATASETS
==============================================

Apply all analyses to ALL datasets (ADME + Toxicity).
This ensures consistency across the entire benchmark.

Analyses:
1. Tanimoto Similarity Analysis
2. Label Distribution Analysis
3. Feature-Label Correlation Analysis
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_all_datasets():
    """Load all datasets from datasets/ directory."""
    datasets = {}

    # ADME datasets
    adme_dir = 'datasets/adme'
    if os.path.exists(adme_dir):
        for filename in os.listdir(adme_dir):
            if filename.endswith('.csv'):
                name = filename.replace('.csv', '')
                path = os.path.join(adme_dir, filename)
                df = pd.read_csv(path)
                df['Type'] = 'ADME'
                df['Task'] = 'Regression'
                datasets[name] = df
                print(f"Loaded {name}: {len(df)} molecules")

    # Toxicity datasets
    tox_dir = 'datasets/toxicity'
    if os.path.exists(tox_dir):
        for filename in os.listdir(tox_dir):
            if filename.endswith('.csv'):
                name = filename.replace('.csv', '')
                path = os.path.join(tox_dir, filename)
                df = pd.read_csv(path)
                df['Type'] = 'Toxicity'
                df['Task'] = 'Classification'
                datasets[name] = df
                print(f"Loaded {name}: {len(df)} molecules")

    return datasets


def compute_tanimoto_similarities(dataset_name, df, output_dir='figures/per_dataset_analysis'):
    """Compute and visualize Tanimoto similarities."""
    print(f"\n  Computing Tanimoto similarities for {dataset_name}...")

    os.makedirs(f'{output_dir}/{dataset_name}', exist_ok=True)

    # Generate fingerprints
    fps = []
    valid_indices = []

    for idx, smiles in enumerate(df['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
            valid_indices.append(idx)

    print(f"    Generated {len(fps)} fingerprints")

    # Sample if too large
    if len(fps) > 1000:
        sample_indices = np.random.choice(len(fps), 1000, replace=False)
        fps_sample = [fps[i] for i in sample_indices]
    else:
        fps_sample = fps

    # Compute pairwise similarities
    n = len(fps_sample)
    similarities = []

    for i in range(n):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(fps_sample[i], fps_sample[j])
            similarities.append(sim)

    similarities = np.array(similarities)

    # Statistics
    stats = {
        'mean': similarities.mean(),
        'median': np.median(similarities),
        'std': similarities.std(),
        'min': similarities.min(),
        'max': similarities.max()
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(similarities, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.3f}")
    ax.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.3f}")
    ax.set_xlabel('Tanimoto Similarity', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'Tanimoto Similarity Distribution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Box plot
    ax = axes[1]
    bp = ax.boxplot([similarities], vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax.set_ylabel('Tanimoto Similarity', fontweight='bold')
    ax.set_title(f'Similarity Statistics', fontweight='bold')
    ax.set_xticklabels([f'{dataset_name}'])
    ax.grid(axis='y', alpha=0.3)

    # Add statistics text
    stats_text = f"Mean: {stats['mean']:.3f}\nMedian: {stats['median']:.3f}\nStd: {stats['std']:.3f}"
    ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{dataset_name} - Molecular Similarity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}/tanimoto_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save statistics
    pd.DataFrame([stats]).to_csv(f'{output_dir}/{dataset_name}/similarity_stats.csv', index=False)

    print(f"    Mean similarity: {stats['mean']:.3f}")
    return stats


def analyze_label_distribution(dataset_name, df, output_dir='figures/per_dataset_analysis'):
    """Analyze and visualize label distribution."""
    print(f"\n  Analyzing label distribution for {dataset_name}...")

    os.makedirs(f'{output_dir}/{dataset_name}', exist_ok=True)

    task = df['Task'].iloc[0]
    labels = df['Y'].values

    if task == 'Regression':
        # Regression: distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Histogram
        ax = axes[0, 0]
        ax.hist(labels, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
        ax.set_xlabel('Target Value', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Label Distribution', fontweight='bold')
        ax.grid(alpha=0.3)

        # Box plot
        ax = axes[0, 1]
        bp = ax.boxplot([labels], vert=True, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][0].set_alpha(0.7)
        ax.set_ylabel('Target Value', fontweight='bold')
        ax.set_title('Box Plot', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Q-Q plot
        ax = axes[1, 0]
        from scipy import stats as scipy_stats
        scipy_stats.probplot(labels, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
        ax.grid(alpha=0.3)

        # Violin plot
        ax = axes[1, 1]
        parts = ax.violinplot([labels], positions=[0], widths=0.7, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightcoral')
            pc.set_alpha(0.7)
        ax.set_ylabel('Target Value', fontweight='bold')
        ax.set_title('Violin Plot', fontweight='bold')
        ax.set_xticks([0])
        ax.set_xticklabels([dataset_name])
        ax.grid(axis='y', alpha=0.3)

        # Statistics
        stats_dict = {
            'mean': labels.mean(),
            'median': np.median(labels),
            'std': labels.std(),
            'min': labels.min(),
            'max': labels.max(),
            'skewness': pd.Series(labels).skew(),
            'kurtosis': pd.Series(labels).kurt()
        }

    else:
        # Classification: class balance
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar plot
        ax = axes[0]
        unique, counts = np.unique(labels, return_counts=True)
        colors = ['lightgreen', 'salmon']
        bars = ax.bar(['Negative (0)', 'Positive (1)'], counts, color=colors, edgecolor='black', alpha=0.8)

        # Add percentage labels
        total = counts.sum()
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = 100 * count / total
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Class Balance', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Pie chart
        ax = axes[1]
        ax.pie(counts, labels=['Negative (0)', 'Positive (1)'], autopct='%1.1f%%',
              colors=colors, startangle=90, wedgeprops=dict(edgecolor='black'))
        ax.set_title('Class Distribution', fontweight='bold')

        stats_dict = {
            'class_0': counts[0],
            'class_1': counts[1] if len(counts) > 1 else 0,
            'balance_ratio': counts[1] / counts[0] if len(counts) > 1 else 0,
            'total': total
        }

    plt.suptitle(f'{dataset_name} - Label Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}/label_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save statistics
    pd.DataFrame([stats_dict]).to_csv(f'{output_dir}/{dataset_name}/label_stats.csv', index=False)

    print(f"    Distribution analyzed")
    return stats_dict


def analyze_feature_correlations(dataset_name, df, output_dir='figures/per_dataset_analysis'):
    """Analyze feature-label correlations."""
    print(f"\n  Analyzing feature correlations for {dataset_name}...")

    os.makedirs(f'{output_dir}/{dataset_name}', exist_ok=True)

    task = df['Task'].iloc[0]

    # Select molecular features
    feature_cols = ['MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'NumRotatableBonds',
                   'NumAromaticRings', 'FractionCSP3']

    # Check which features exist
    available_features = [f for f in feature_cols if f in df.columns]

    if len(available_features) < 3:
        print(f"    Not enough features available, skipping")
        return None

    X = df[available_features].values
    y = df['Y'].values

    # Compute correlations
    correlations = []
    for i, feature in enumerate(available_features):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append(corr)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot of correlations
    ax = axes[0]
    colors = ['green' if c > 0 else 'red' for c in correlations]
    bars = ax.barh(available_features, correlations, color=colors, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Correlation with Target', fontweight='bold')
    ax.set_title('Feature-Target Correlations', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Scatter plot of top feature
    ax = axes[1]
    top_idx = np.argmax(np.abs(correlations))
    top_feature = available_features[top_idx]
    top_corr = correlations[top_idx]

    ax.scatter(X[:, top_idx], y, alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
    ax.set_xlabel(f'{top_feature}', fontweight='bold')
    ax.set_ylabel('Target Value', fontweight='bold')
    ax.set_title(f'Top Feature: {top_feature} (r={top_corr:.3f})', fontweight='bold')
    ax.grid(alpha=0.3)

    # Add regression line
    z = np.polyfit(X[:, top_idx], y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(X[:, top_idx].min(), X[:, top_idx].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)

    plt.suptitle(f'{dataset_name} - Feature Correlation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}/feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save correlations
    corr_df = pd.DataFrame({
        'Feature': available_features,
        'Correlation': correlations,
        'Abs_Correlation': np.abs(correlations)
    }).sort_values('Abs_Correlation', ascending=False)

    corr_df.to_csv(f'{output_dir}/{dataset_name}/correlation_stats.csv', index=False)

    print(f"    Top feature: {top_feature} (r={top_corr:.3f})")
    return corr_df


def main():
    """Run all analyses on all datasets."""
    print("="*80)
    print("COMPREHENSIVE DATASET ANALYSIS - ALL DATASETS")
    print("="*80)

    # Load datasets
    print("\nLoading all datasets...")
    datasets = load_all_datasets()

    if not datasets:
        print("\n[ERROR] No datasets found!")
        print("Please ensure datasets are in:")
        print("  - datasets/adme/")
        print("  - datasets/toxicity/")
        return

    print(f"\nFound {len(datasets)} datasets to analyze\n")

    # Create output directory
    output_dir = 'figures/per_dataset_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Run analyses for each dataset
    all_results = {}

    for dataset_name, df in datasets.items():
        print(f"\n{'='*80}")
        print(f"ANALYZING: {dataset_name}")
        print(f"  Type: {df['Type'].iloc[0]}, Task: {df['Task'].iloc[0]}, Size: {len(df)}")
        print(f"{'='*80}")

        results = {}

        # 1. Tanimoto Similarity
        try:
            results['similarity'] = compute_tanimoto_similarities(dataset_name, df, output_dir)
        except Exception as e:
            print(f"    [ERROR] Tanimoto analysis failed: {e}")

        # 2. Label Distribution
        try:
            results['labels'] = analyze_label_distribution(dataset_name, df, output_dir)
        except Exception as e:
            print(f"    [ERROR] Label analysis failed: {e}")

        # 3. Feature Correlations
        try:
            results['correlations'] = analyze_feature_correlations(dataset_name, df, output_dir)
        except Exception as e:
            print(f"    [ERROR] Correlation analysis failed: {e}")

        all_results[dataset_name] = results

    # Generate summary
    print(f"\n{'='*80}")
    print("[OK] ALL DATASET ANALYSES COMPLETED!")
    print(f"{'='*80}")

    print(f"\nGenerated files in: {output_dir}/")
    print("\nPer dataset:")
    for dataset_name in datasets.keys():
        print(f"  {dataset_name}/")
        print(f"    - tanimoto_similarity.png")
        print(f"    - similarity_stats.csv")
        print(f"    - label_distribution.png")
        print(f"    - label_stats.csv")
        print(f"    - feature_correlations.png")
        print(f"    - correlation_stats.csv")

    print(f"\n✅ All analyses applied to all {len(datasets)} datasets!")


if __name__ == "__main__":
    main()
