"""
TANIMOTO SIMILARITY ANALYSIS
============================

Анализа на молекуларна сличност со Tanimoto коефициент користејќи Morgan fingerprints.
Ова е клучно за разбирање на:
- Колку се разнолични молекулите во dataset
- Дали постои correlation помеѓу сличност и предвидувања
- Дали test set е доволно различен од train set
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tdc.single_pred import ADME
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Визуелизација стил
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def generate_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    Генерирај Morgan fingerprint за SMILES string

    Args:
        smiles: SMILES string
        radius: Радиус на fingerprint (default 2 = ECFP4)
        n_bits: Број на битови (default 2048)

    Returns:
        Morgan fingerprint или None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return fp
    except:
        return None


def calculate_tanimoto_matrix(fingerprints: List, max_size: int = 500):
    """
    Пресметај Tanimoto similarity matrix

    Args:
        fingerprints: Листа на fingerprints
        max_size: Максимален број на молекули за matrix (заради меморија)

    Returns:
        Similarity matrix
    """
    n = min(len(fingerprints), max_size)
    fps_subset = fingerprints[:n]

    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            sim = DataStructs.TanimotoSimilarity(fps_subset[i], fps_subset[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    return similarity_matrix


def analyze_dataset_similarity(dataset_name: str, output_dir: str = "figures/similarity"):
    """
    Анализирај Tanimoto similarity за dataset

    Args:
        dataset_name: Име на dataset
        output_dir: Директориум за зачувување на графици
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Tanimoto Similarity Analysis: {dataset_name}")
    print(f"{'='*80}\n")

    # Вчитај податоци
    print("Вчитување на податоци...")
    data_api = ADME(name=dataset_name)
    split = data_api.get_split(method="scaffold")

    # Приправи податоци
    train_df = split["train"].dropna()
    test_df = split["test"].dropna()

    # Резолвирај колони
    smiles_col = None
    for col in ["Drug", "SMILES", "smiles", "X"]:
        if col in train_df.columns:
            smiles_col = col
            break

    target_col = None
    for col in ["Y", "target", "y", "Label", "label"]:
        if col in train_df.columns:
            target_col = col
            break

    print(f"SMILES column: {smiles_col}, Target column: {target_col}")
    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    # Генерирај fingerprints
    print("\nГенерирање на Morgan fingerprints (ECFP4, radius=2, 2048 bits)...")
    train_fps = []
    train_smiles = []
    train_targets = []

    for idx, row in train_df.iterrows():
        fp = generate_morgan_fingerprint(row[smiles_col])
        if fp is not None:
            train_fps.append(fp)
            train_smiles.append(row[smiles_col])
            train_targets.append(row[target_col])

    test_fps = []
    test_smiles = []
    test_targets = []

    for idx, row in test_df.iterrows():
        fp = generate_morgan_fingerprint(row[smiles_col])
        if fp is not None:
            test_fps.append(fp)
            test_smiles.append(row[smiles_col])
            test_targets.append(row[target_col])

    print(f"Генерирани fingerprints: Train={len(train_fps)}, Test={len(test_fps)}")

    # ============ ВИЗУЕЛИЗАЦИЈА 1: Pairwise Similarity Distribution ============
    print("\n1. Пресметување на pairwise similarities во train set...")
    n_sample = min(300, len(train_fps))
    sample_indices = np.random.choice(len(train_fps), n_sample, replace=False)
    train_sample_fps = [train_fps[i] for i in sample_indices]

    train_similarities = []
    for i in range(len(train_sample_fps)):
        for j in range(i+1, len(train_sample_fps)):
            sim = DataStructs.TanimotoSimilarity(train_sample_fps[i], train_sample_fps[j])
            train_similarities.append(sim)

    # Similarity matrix visualization
    print("   Креирање на similarity matrix...")
    n_viz = min(100, len(train_fps))
    sim_matrix = calculate_tanimoto_matrix(train_fps, max_size=n_viz)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap
    im = axes[0].imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title(f'Tanimoto Similarity Matrix (Train Set)\n{dataset_name} - {n_viz} molecules',
                      fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Molecule Index')
    axes[0].set_ylabel('Molecule Index')
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label('Tanimoto Similarity', rotation=270, labelpad=20)

    # Distribution histogram
    axes[1].hist(train_similarities, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(train_similarities), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(train_similarities):.3f}')
    axes[1].axvline(np.median(train_similarities), color='orange', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(train_similarities):.3f}')
    axes[1].set_xlabel('Tanimoto Similarity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Distribution of Pairwise Similarities\n{dataset_name} - {len(train_similarities):,} pairs',
                      fontweight='bold', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_similarity_matrix.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_similarity_matrix.png")
    plt.close()

    # ============ ВИЗУЕЛИЗАЦИЈА 2: Train vs Test Similarity ============
    print("\n2. Анализа на Train-Test similarity...")

    # Пресметај max similarity за секоја test молекула спрема train set
    test_max_similarities = []
    test_mean_similarities = []

    n_test_sample = min(200, len(test_fps))
    n_train_sample = min(300, len(train_fps))

    for i in range(n_test_sample):
        sims = []
        for j in range(n_train_sample):
            sim = DataStructs.TanimotoSimilarity(test_fps[i], train_fps[j])
            sims.append(sim)
        test_max_similarities.append(max(sims))
        test_mean_similarities.append(np.mean(sims))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Max similarity distribution
    axes[0].hist(test_max_similarities, bins=40, color='coral', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(test_max_similarities), color='darkred', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(test_max_similarities):.3f}')
    axes[0].set_xlabel('Maximum Tanimoto Similarity to Train Set')
    axes[0].set_ylabel('Frequency (Test Molecules)')
    axes[0].set_title(f'Test Set Similarity to Train Set\n{dataset_name} - Max Similarity',
                      fontweight='bold', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Mean similarity distribution
    axes[1].hist(test_mean_similarities, bins=40, color='lightcoral', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(test_mean_similarities), color='darkred', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(test_mean_similarities):.3f}')
    axes[1].set_xlabel('Mean Tanimoto Similarity to Train Set')
    axes[1].set_ylabel('Frequency (Test Molecules)')
    axes[1].set_title(f'Test Set Similarity to Train Set\n{dataset_name} - Mean Similarity',
                      fontweight='bold', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_train_test_similarity.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_train_test_similarity.png")
    plt.close()

    # ============ ВИЗУЕЛИЗАЦИЈА 3: Similarity vs Target Correlation ============
    print("\n3. Анализа на Similarity-Target correlation...")

    # Sample pairs and compute similarity + target difference
    n_pairs = min(1000, len(train_fps) * (len(train_fps) - 1) // 2)

    similarities = []
    target_diffs = []

    for _ in range(n_pairs):
        i, j = np.random.choice(len(train_fps), 2, replace=False)
        sim = DataStructs.TanimotoSimilarity(train_fps[i], train_fps[j])
        target_diff = abs(train_targets[i] - train_targets[j])
        similarities.append(sim)
        target_diffs.append(target_diff)

    # Create bins for similarity
    sim_bins = np.linspace(0, 1, 11)
    bin_centers = (sim_bins[:-1] + sim_bins[1:]) / 2
    bin_means = []
    bin_stds = []

    for i in range(len(sim_bins) - 1):
        mask = (np.array(similarities) >= sim_bins[i]) & (np.array(similarities) < sim_bins[i+1])
        if mask.sum() > 0:
            bin_means.append(np.mean(np.array(target_diffs)[mask]))
            bin_stds.append(np.std(np.array(target_diffs)[mask]))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot
    axes[0].scatter(similarities, target_diffs, alpha=0.3, s=10, color='steelblue')
    axes[0].set_xlabel('Tanimoto Similarity')
    axes[0].set_ylabel('Absolute Target Difference')
    axes[0].set_title(f'Molecular Similarity vs Target Difference\n{dataset_name} - {n_pairs:,} pairs',
                      fontweight='bold', fontsize=12)
    axes[0].grid(alpha=0.3)

    # Correlation
    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(similarities, target_diffs)
    spearman_r, spearman_p = spearmanr(similarities, target_diffs)

    axes[0].text(0.05, 0.95,
                 f'Pearson r: {pearson_r:.3f} (p={pearson_p:.2e})\nSpearman ρ: {spearman_r:.3f} (p={spearman_p:.2e})',
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

    # Binned plot
    axes[1].errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-',
                     capsize=5, capthick=2, linewidth=2, markersize=8, color='darkgreen')
    axes[1].set_xlabel('Tanimoto Similarity (binned)')
    axes[1].set_ylabel('Mean Absolute Target Difference')
    axes[1].set_title(f'Similarity vs Target Difference (Binned)\n{dataset_name}',
                      fontweight='bold', fontsize=12)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_similarity_target_correlation.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_similarity_target_correlation.png")
    plt.close()

    # ============ СТАТИСТИЧКИ РЕЗИМЕ ============
    print(f"\n{'='*80}")
    print(f"РЕЗИМЕ: {dataset_name}")
    print(f"{'='*80}")
    print(f"\nTrain Set Internal Similarity:")
    print(f"  Mean:   {np.mean(train_similarities):.4f}")
    print(f"  Median: {np.median(train_similarities):.4f}")
    print(f"  Std:    {np.std(train_similarities):.4f}")
    print(f"  Min:    {np.min(train_similarities):.4f}")
    print(f"  Max:    {np.max(train_similarities):.4f}")

    print(f"\nTest Set Similarity to Train:")
    print(f"  Mean Max:  {np.mean(test_max_similarities):.4f}")
    print(f"  Mean Avg:  {np.mean(test_mean_similarities):.4f}")
    print(f"  Median Max: {np.median(test_max_similarities):.4f}")

    print(f"\nSimilarity-Target Correlation:")
    print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})")

    if pearson_r < -0.2:
        print("  OK: Негативна корелација: Поголема сличност → Помала разлика во targets")
    elif abs(pearson_r) < 0.1:
        print("  WARNING:  Слаба корелација: Сличност не предвидува target сличност")

    # Зачувај резултати
    results = {
        'dataset': dataset_name,
        'train_sim_mean': np.mean(train_similarities),
        'train_sim_median': np.median(train_similarities),
        'train_sim_std': np.std(train_similarities),
        'test_max_sim_mean': np.mean(test_max_similarities),
        'test_mean_sim_mean': np.mean(test_mean_similarities),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(f'{output_dir}/{dataset_name}_similarity_stats.csv', index=False)
    print(f"\n💾 Статистики зачувани: {dataset_name}_similarity_stats.csv")


def analyze_all_datasets():
    """Анализирај сите datasets"""
    datasets = [
        "Half_Life_Obach",
        "Clearance_Hepatocyte_AZ",
        "Clearance_Microsome_AZ",
        "Caco2_Wang"
    ]

    all_results = []

    for dataset_name in datasets:
        try:
            analyze_dataset_similarity(dataset_name)
            print(f"\nOK: Завршено: {dataset_name}\n")
        except Exception as e:
            print(f"\nERROR: Грешка за {dataset_name}: {e}\n")

    print(f"\n{'='*80}")
    print("TANIMOTO SIMILARITY ANALYSIS - КОМПЛЕТИРАНО!")
    print(f"{'='*80}")


if __name__ == "__main__":
    analyze_all_datasets()
