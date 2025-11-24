"""
FEATURE-LABEL CORRELATION ANALYSIS
===================================

Детална анализа на корелација помеѓу ADME молекуларни features и target labels.
Вклучува:
- Correlation heatmaps (Pearson и Spearman)
- Feature importance ranking
- Scatter plots на топ корелации
- Feature distribution analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tdc.single_pred import ADME
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def compute_adme_descriptors(smiles: str):
    """
    Пресметај ADME дескриптори за SMILES

    Returns:
        Dictionary со ADME дескриптори
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        descriptors = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': rdMolDescriptors.CalcNumHBD(mol),
            'HBA': rdMolDescriptors.CalcNumHBA(mol),
            'TPSA': rdMolDescriptors.CalcTPSA(mol),
            'RotatableBonds': Descriptors.NumRotatableBonds(mol),
            'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'AliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
            'Heteroatoms': Descriptors.NumHeteroatoms(mol),
            'HeavyAtoms': rdMolDescriptors.CalcNumHeavyAtoms(mol),
            'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
            'MolMR': Descriptors.MolMR(mol),
            'BertzCT': Descriptors.BertzCT(mol),
            'Chi0v': Descriptors.Chi0v(mol),
            'Rings': rdMolDescriptors.CalcNumRings(mol),
            'Lipinski_MW': int(descriptors['MW'] > 500) if 'MW' in locals() else 0,
            'Lipinski_LogP': int(Descriptors.MolLogP(mol) > 5),
            'Lipinski_HBD': int(rdMolDescriptors.CalcNumHBD(mol) > 5),
            'Lipinski_HBA': int(rdMolDescriptors.CalcNumHBA(mol) > 10),
        }

        # Fix Lipinski flags
        descriptors['Lipinski_MW'] = int(descriptors['MW'] > 500)

        return descriptors
    except:
        return None


def resolve_columns(df):
    """Резолвирај SMILES и target колони"""
    smiles_col = None
    for col in ["Drug", "SMILES", "smiles", "X"]:
        if col in df.columns:
            smiles_col = col
            break

    target_col = None
    for col in ["Y", "target", "y", "Label", "label"]:
        if col in df.columns:
            target_col = col
            break

    return smiles_col, target_col


def analyze_feature_correlations(dataset_name: str, output_dir: str = "figures/correlations"):
    """
    Анализирај feature-label корелации за dataset

    Args:
        dataset_name: Име на dataset
        output_dir: Директориум за зачувување
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Feature-Label Correlation Analysis: {dataset_name}")
    print(f"{'='*80}\n")

    # Вчитај податоци
    print("Вчитување на податоци...")
    data_api = ADME(name=dataset_name)
    split = data_api.get_split(method="scaffold")

    train_df = split["train"].dropna()
    test_df = split["test"].dropna()

    smiles_col, target_col = resolve_columns(train_df)
    print(f"SMILES column: {smiles_col}, Target column: {target_col}")
    print(f"Train: {len(train_df)} samples")

    # Пресметај ADME features за train set
    print("\nПресметување на ADME дескриптори...")
    features_list = []

    for idx, row in train_df.iterrows():
        smiles = row[smiles_col]
        target = row[target_col]
        descriptors = compute_adme_descriptors(smiles)

        if descriptors is not None:
            descriptors['Target'] = target
            descriptors['log_Target'] = np.log(max(1e-6, target))
            features_list.append(descriptors)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(train_df)} molecules...")

    features_df = pd.DataFrame(features_list)
    print(f"\nКреиран DataFrame со {len(features_df)} молекули и {len(features_df.columns)} features")

    # Remove constant features
    features_df = features_df.loc[:, features_df.std() > 0]
    print(f"После отстранување на константни features: {len(features_df.columns)} features")

    # ============ ВИЗУЕЛИЗАЦИЈА 1: Correlation Heatmap (Pearson) ============
    print("\n1. Pearson Correlation Heatmap...")

    # Compute Pearson correlations
    corr_pearson = features_df.corr(method='pearson')

    # Extract correlations with target
    target_corr_pearson = corr_pearson['Target'].drop(['Target', 'log_Target']).sort_values(key=abs, ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Full heatmap
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool), k=1)
    sns.heatmap(corr_pearson, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[0],
                vmin=-1, vmax=1)
    axes[0].set_title(f'Pearson Correlation Matrix\n{dataset_name}', fontweight='bold', fontsize=13)

    # Top features vs Target
    top_n = min(15, len(target_corr_pearson))
    top_features = target_corr_pearson.head(top_n)

    colors = ['green' if x > 0 else 'red' for x in top_features.values]
    axes[1].barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_yticks(range(len(top_features)))
    axes[1].set_yticklabels(top_features.index)
    axes[1].set_xlabel('Pearson Correlation with Target')
    axes[1].set_title(f'Top {top_n} Features - Pearson Correlation\n{dataset_name}',
                      fontweight='bold', fontsize=13)
    axes[1].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[1].grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_pearson_correlation.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_pearson_correlation.png")
    plt.close()

    # ============ ВИЗУЕЛИЗАЦИЈА 2: Correlation Heatmap (Spearman) ============
    print("\n2. Spearman Correlation Heatmap...")

    # Compute Spearman correlations
    corr_spearman = features_df.corr(method='spearman')

    # Extract correlations with target
    target_corr_spearman = corr_spearman['Target'].drop(['Target', 'log_Target']).sort_values(key=abs, ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Full heatmap
    mask = np.triu(np.ones_like(corr_spearman, dtype=bool), k=1)
    sns.heatmap(corr_spearman, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[0],
                vmin=-1, vmax=1)
    axes[0].set_title(f'Spearman Correlation Matrix\n{dataset_name}', fontweight='bold', fontsize=13)

    # Top features vs Target
    top_features_spear = target_corr_spearman.head(top_n)

    colors = ['green' if x > 0 else 'red' for x in top_features_spear.values]
    axes[1].barh(range(len(top_features_spear)), top_features_spear.values, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_yticks(range(len(top_features_spear)))
    axes[1].set_yticklabels(top_features_spear.index)
    axes[1].set_xlabel('Spearman Correlation with Target')
    axes[1].set_title(f'Top {top_n} Features - Spearman Correlation\n{dataset_name}',
                      fontweight='bold', fontsize=13)
    axes[1].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[1].grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_spearman_correlation.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_spearman_correlation.png")
    plt.close()

    # ============ ВИЗУЕЛИЗАЦИЈА 3: Scatter Plots - Top Correlations ============
    print("\n3. Scatter Plots за топ корелации...")

    # Get top 6 features
    top_6_features = target_corr_pearson.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, feature in enumerate(top_6_features):
        x = features_df[feature]
        y = features_df['Target']

        axes[i].scatter(x, y, alpha=0.5, s=30, color='steelblue', edgecolor='black', linewidth=0.5)

        # Fit regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        axes[i].plot(x_line, p(x_line), "r--", linewidth=2, label='Linear Fit')

        # Compute statistics
        r_pearson, p_pearson = pearsonr(x, y)
        r_spearman, p_spearman = spearmanr(x, y)

        axes[i].set_xlabel(feature, fontsize=11)
        axes[i].set_ylabel('Target', fontsize=11)
        axes[i].set_title(f'{feature} vs Target\nPearson: {r_pearson:.3f} (p={p_pearson:.2e})',
                         fontweight='bold', fontsize=11)
        axes[i].grid(alpha=0.3)
        axes[i].legend(fontsize=9)

    plt.suptitle(f'Top 6 Feature-Target Correlations\n{dataset_name}', fontweight='bold', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_scatter_plots.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_scatter_plots.png")
    plt.close()

    # ============ ВИЗУЕЛИЗАЦИЈА 4: Feature Distributions ============
    print("\n4. Feature Distributions...")

    top_4_features = target_corr_pearson.head(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, feature in enumerate(top_4_features):
        data = features_df[feature]

        # Histogram + KDE
        axes[i].hist(data, bins=40, color='steelblue', alpha=0.6, edgecolor='black', density=True, label='Histogram')

        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        # Statistics
        axes[i].axvline(data.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
        axes[i].axvline(data.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')

        axes[i].set_xlabel(feature, fontsize=11)
        axes[i].set_ylabel('Density', fontsize=11)
        axes[i].set_title(f'Distribution of {feature}\nCorrelation with Target: {target_corr_pearson[feature]:.3f}',
                         fontweight='bold', fontsize=11)
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3)

    plt.suptitle(f'Feature Distributions - Top Correlations\n{dataset_name}', fontweight='bold', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_feature_distributions.png")
    plt.close()

    # ============ ВИЗУЕЛИЗАЦИЈА 5: Feature Importance Comparison ============
    print("\n5. Feature Importance Comparison (Pearson vs Spearman)...")

    # Compare top 10 from both methods
    comparison_df = pd.DataFrame({
        'Pearson': target_corr_pearson.head(10),
        'Spearman': target_corr_spearman.head(10)
    }).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(comparison_df))
    width = 0.35

    bars1 = ax.barh(x - width/2, comparison_df['Pearson'].abs(), width,
                    label='Pearson', color='steelblue', alpha=0.8)
    bars2 = ax.barh(x + width/2, comparison_df['Spearman'].abs(), width,
                    label='Spearman', color='coral', alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(comparison_df.index)
    ax.set_xlabel('Absolute Correlation with Target', fontsize=12)
    ax.set_title(f'Feature Importance: Pearson vs Spearman\n{dataset_name}', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_feature_importance_comparison.png")
    plt.close()

    # ============ ВИЗУЕЛИЗАЦИЈА 6: Pairwise Feature Correlations (Top Features) ============
    print("\n6. Pairwise Feature Correlations...")

    top_10_features = target_corr_pearson.head(10).index.tolist()
    top_10_features.append('Target')

    corr_subset = features_df[top_10_features].corr(method='pearson')

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title(f'Pairwise Correlations - Top 10 Features + Target\n{dataset_name}',
                 fontweight='bold', fontsize=13)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_pairwise_correlations.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_pairwise_correlations.png")
    plt.close()

    # ============ СТАТИСТИЧКИ РЕЗИМЕ ============
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS РЕЗИМЕ: {dataset_name}")
    print(f"{'='*80}")

    print(f"\nTOP 10 FEATURES - PEARSON CORRELATION:")
    for i, (feature, corr) in enumerate(target_corr_pearson.head(10).items(), 1):
        print(f"  {i:2d}. {feature:20s}  r = {corr:+.4f}")

    print(f"\nTOP 10 FEATURES - SPEARMAN CORRELATION:")
    for i, (feature, corr) in enumerate(target_corr_spearman.head(10).items(), 1):
        print(f"  {i:2d}. {feature:20s}  ρ = {corr:+.4f}")

    # Statistical significance
    print(f"\nSTATISTICAL SIGNIFICANCE (Top 5 Pearson):")
    for feature in target_corr_pearson.head(5).index:
        r, p = pearsonr(features_df[feature], features_df['Target'])
        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {feature:20s}  r = {r:+.4f}, p = {p:.2e} {significance}")

    # Save correlation statistics
    corr_stats = pd.DataFrame({
        'Feature': target_corr_pearson.index,
        'Pearson_r': target_corr_pearson.values,
        'Spearman_rho': [target_corr_spearman.get(f, np.nan) for f in target_corr_pearson.index],
    })

    # Add p-values for top features
    p_values = []
    for feature in target_corr_pearson.index:
        try:
            _, p = pearsonr(features_df[feature], features_df['Target'])
            p_values.append(p)
        except:
            p_values.append(np.nan)

    corr_stats['Pearson_p'] = p_values

    corr_stats.to_csv(f'{output_dir}/{dataset_name}_correlation_stats.csv', index=False)
    print(f"\n💾 Correlation статистики зачувани: {dataset_name}_correlation_stats.csv")

    # Save full features DataFrame for later analysis
    features_df.to_csv(f'{output_dir}/{dataset_name}_features_data.csv', index=False)
    print(f"💾 Features DataFrame зачуван: {dataset_name}_features_data.csv")


def analyze_all_datasets():
    """Анализирај сите datasets"""
    datasets = [
        "Half_Life_Obach",
        "Clearance_Hepatocyte_AZ",
        "Clearance_Microsome_AZ",
        "Caco2_Wang"
    ]

    for dataset_name in datasets:
        try:
            analyze_feature_correlations(dataset_name)
            print(f"\nOK: Завршено: {dataset_name}\n")
        except Exception as e:
            print(f"\nERROR: Грешка за {dataset_name}: {e}\n")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("FEATURE-LABEL CORRELATION ANALYSIS - КОМПЛЕТИРАНО!")
    print(f"{'='*80}")


if __name__ == "__main__":
    analyze_all_datasets()
