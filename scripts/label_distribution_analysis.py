"""
LABEL DISTRIBUTION ANALYSIS
===========================

Детална анализа на дистрибуција на target вредности (labels) за ADME datasets.
Вклучува:
- Histograms во original и log space
- Statistical summaries
- Outlier detection
- Cross-dataset comparisons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tdc.single_pred import ADME
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


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


def detect_outliers(data, method='iqr', threshold=1.5):
    """
    Детекција на outliers

    Args:
        data: Array of values
        method: 'iqr', 'zscore', or 'percentile'
        threshold: Threshold for outlier detection

    Returns:
        Boolean mask of outliers
    """
    if method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (data < lower) | (data > upper)

    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        return z_scores > threshold

    elif method == 'percentile':
        lower = np.percentile(data, 2.5)
        upper = np.percentile(data, 97.5)
        return (data < lower) | (data > upper)


def analyze_dataset_labels(dataset_name: str, output_dir: str = "figures/labels"):
    """
    Анализирај label distribution за dataset

    Args:
        dataset_name: Име на dataset
        output_dir: Директориум за зачувување
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Label Distribution Analysis: {dataset_name}")
    print(f"{'='*80}\n")

    # Вчитај податоци
    print("Вчитување на податоци...")
    data_api = ADME(name=dataset_name)
    split = data_api.get_split(method="scaffold")

    train_df = split["train"].dropna()
    test_df = split["test"].dropna()

    smiles_col, target_col = resolve_columns(train_df)
    print(f"Target column: {target_col}")
    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    # Извлечи target вредности
    train_targets = train_df[target_col].values
    test_targets = test_df[target_col].values
    all_targets = np.concatenate([train_targets, test_targets])

    # ============ ВИЗУЕЛИЗАЦИЈА 1: Original vs Log Distribution ============
    print("\n1. Original vs Log Space Distribution...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Original space - Train
    axes[0, 0].hist(train_targets, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(train_targets), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(train_targets):.3f}')
    axes[0, 0].axvline(np.median(train_targets), color='orange', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(train_targets):.3f}')
    axes[0, 0].set_xlabel('Target Value (Original Space)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Train Set Distribution - Original Space\n{dataset_name}',
                         fontweight='bold', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Original space - Test
    axes[0, 1].hist(test_targets, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(test_targets), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(test_targets):.3f}')
    axes[0, 1].axvline(np.median(test_targets), color='orange', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(test_targets):.3f}')
    axes[0, 1].set_xlabel('Target Value (Original Space)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Test Set Distribution - Original Space\n{dataset_name}',
                         fontweight='bold', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Log space - Train
    train_log = np.log(np.clip(train_targets, 1e-6, None))
    axes[1, 0].hist(train_log, bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(train_log), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(train_log):.3f}')
    axes[1, 0].axvline(np.median(train_log), color='orange', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(train_log):.3f}')
    axes[1, 0].set_xlabel('log(Target Value)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Train Set Distribution - Log Space\n{dataset_name}',
                         fontweight='bold', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Log space - Test
    test_log = np.log(np.clip(test_targets, 1e-6, None))
    axes[1, 1].hist(test_log, bins=50, color='darkorange', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(test_log), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(test_log):.3f}')
    axes[1, 1].axvline(np.median(test_log), color='orange', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(test_log):.3f}')
    axes[1, 1].set_xlabel('log(Target Value)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Test Set Distribution - Log Space\n{dataset_name}',
                         fontweight='bold', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_distribution_comparison.png")
    plt.close()

    # ============ ВИЗУЕЛИЗАЦИЈА 2: Box Plots и Violin Plots ============
    print("\n2. Box Plots и Violin Plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'Target': np.concatenate([train_targets, test_targets]),
        'Split': ['Train']*len(train_targets) + ['Test']*len(test_targets)
    })

    plot_data_log = pd.DataFrame({
        'log(Target)': np.concatenate([train_log, test_log]),
        'Split': ['Train']*len(train_log) + ['Test']*len(test_log)
    })

    # Box plot - Original space
    sns.boxplot(data=plot_data, x='Split', y='Target', ax=axes[0, 0], palette=['steelblue', 'coral'])
    axes[0, 0].set_title(f'Box Plot - Original Space\n{dataset_name}', fontweight='bold', fontsize=12)
    axes[0, 0].grid(alpha=0.3, axis='y')

    # Violin plot - Original space
    sns.violinplot(data=plot_data, x='Split', y='Target', ax=axes[0, 1], palette=['steelblue', 'coral'])
    axes[0, 1].set_title(f'Violin Plot - Original Space\n{dataset_name}', fontweight='bold', fontsize=12)
    axes[0, 1].grid(alpha=0.3, axis='y')

    # Box plot - Log space
    sns.boxplot(data=plot_data_log, x='Split', y='log(Target)', ax=axes[1, 0],
                palette=['darkgreen', 'darkorange'])
    axes[1, 0].set_title(f'Box Plot - Log Space\n{dataset_name}', fontweight='bold', fontsize=12)
    axes[1, 0].grid(alpha=0.3, axis='y')

    # Violin plot - Log space
    sns.violinplot(data=plot_data_log, x='Split', y='log(Target)', ax=axes[1, 1],
                   palette=['darkgreen', 'darkorange'])
    axes[1, 1].set_title(f'Violin Plot - Log Space\n{dataset_name}', fontweight='bold', fontsize=12)
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_boxplots_violinplots.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_boxplots_violinplots.png")
    plt.close()

    # ============ ВИЗУЕЛИЗАЦИЈА 3: Outlier Detection ============
    print("\n3. Outlier Detection...")

    # Detect outliers using multiple methods
    outliers_iqr = detect_outliers(all_targets, method='iqr', threshold=1.5)
    outliers_zscore = detect_outliers(all_targets, method='zscore', threshold=3)
    outliers_percentile = detect_outliers(all_targets, method='percentile')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # IQR method
    axes[0].scatter(range(len(all_targets)), all_targets, c=outliers_iqr,
                    cmap='RdYlGn_r', alpha=0.6, s=20)
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Target Value')
    axes[0].set_title(f'Outlier Detection: IQR Method\n{dataset_name}\n{outliers_iqr.sum()} outliers ({100*outliers_iqr.sum()/len(all_targets):.1f}%)',
                      fontweight='bold', fontsize=11)
    axes[0].grid(alpha=0.3)

    # Z-score method
    axes[1].scatter(range(len(all_targets)), all_targets, c=outliers_zscore,
                    cmap='RdYlGn_r', alpha=0.6, s=20)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Target Value')
    axes[1].set_title(f'Outlier Detection: Z-Score Method\n{dataset_name}\n{outliers_zscore.sum()} outliers ({100*outliers_zscore.sum()/len(all_targets):.1f}%)',
                      fontweight='bold', fontsize=11)
    axes[1].grid(alpha=0.3)

    # Percentile method
    axes[2].scatter(range(len(all_targets)), all_targets, c=outliers_percentile,
                    cmap='RdYlGn_r', alpha=0.6, s=20)
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Target Value')
    axes[2].set_title(f'Outlier Detection: Percentile Method\n{dataset_name}\n{outliers_percentile.sum()} outliers ({100*outliers_percentile.sum()/len(all_targets):.1f}%)',
                      fontweight='bold', fontsize=11)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_outlier_detection.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_outlier_detection.png")
    plt.close()

    # ============ ВИЗУЕЛИЗАЦИЈА 4: Q-Q Plot (Normalnost тест) ============
    print("\n4. Q-Q Plot (Normalност тест)...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Q-Q plot - Original space
    stats.probplot(all_targets, dist="norm", plot=axes[0])
    axes[0].set_title(f'Q-Q Plot - Original Space\n{dataset_name}', fontweight='bold', fontsize=12)
    axes[0].grid(alpha=0.3)

    # Q-Q plot - Log space
    all_log = np.log(np.clip(all_targets, 1e-6, None))
    stats.probplot(all_log, dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot - Log Space\n{dataset_name}', fontweight='bold', fontsize=12)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_qqplot.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: {dataset_name}_qqplot.png")
    plt.close()

    # ============ СТАТИСТИЧКИ РЕЗИМЕ ============
    print(f"\n{'='*80}")
    print(f"СТАТИСТИЧКИ РЕЗИМЕ: {dataset_name}")
    print(f"{'='*80}")

    print(f"\nORIGINAL SPACE:")
    print(f"  Train Set:")
    print(f"    Mean:   {np.mean(train_targets):.4f}")
    print(f"    Median: {np.median(train_targets):.4f}")
    print(f"    Std:    {np.std(train_targets):.4f}")
    print(f"    Min:    {np.min(train_targets):.4f}")
    print(f"    Max:    {np.max(train_targets):.4f}")
    print(f"    Skew:   {stats.skew(train_targets):.4f}")
    print(f"    Kurt:   {stats.kurtosis(train_targets):.4f}")

    print(f"\n  Test Set:")
    print(f"    Mean:   {np.mean(test_targets):.4f}")
    print(f"    Median: {np.median(test_targets):.4f}")
    print(f"    Std:    {np.std(test_targets):.4f}")
    print(f"    Min:    {np.min(test_targets):.4f}")
    print(f"    Max:    {np.max(test_targets):.4f}")

    print(f"\nLOG SPACE:")
    print(f"  Train Set:")
    print(f"    Mean:   {np.mean(train_log):.4f}")
    print(f"    Median: {np.median(train_log):.4f}")
    print(f"    Std:    {np.std(train_log):.4f}")
    print(f"    Skew:   {stats.skew(train_log):.4f}")
    print(f"    Kurt:   {stats.kurtosis(train_log):.4f}")

    print(f"\n  Test Set:")
    print(f"    Mean:   {np.mean(test_log):.4f}")
    print(f"    Median: {np.median(test_log):.4f}")
    print(f"    Std:    {np.std(test_log):.4f}")

    # Normality tests
    _, p_shapiro_orig = stats.shapiro(train_targets[:min(5000, len(train_targets))])
    _, p_shapiro_log = stats.shapiro(train_log[:min(5000, len(train_log))])

    print(f"\nNORMALITY TESTS (Shapiro-Wilk):")
    print(f"  Original space: p={p_shapiro_orig:.2e} {'OK: Normal' if p_shapiro_orig > 0.05 else 'ERROR: Not normal'}")
    print(f"  Log space:      p={p_shapiro_log:.2e} {'OK: Normal' if p_shapiro_log > 0.05 else 'ERROR: Not normal'}")

    print(f"\nOUTLIERS:")
    print(f"  IQR method:        {outliers_iqr.sum()} ({100*outliers_iqr.sum()/len(all_targets):.1f}%)")
    print(f"  Z-score method:    {outliers_zscore.sum()} ({100*outliers_zscore.sum()/len(all_targets):.1f}%)")
    print(f"  Percentile method: {outliers_percentile.sum()} ({100*outliers_percentile.sum()/len(all_targets):.1f}%)")

    # Train-Test distribution comparison
    _, p_ks = stats.ks_2samp(train_targets, test_targets)
    print(f"\nTRAIN-TEST DISTRIBUTION COMPARISON (Kolmogorov-Smirnov test):")
    print(f"  p-value: {p_ks:.2e} {'OK: Same distribution' if p_ks > 0.05 else 'WARNING:  Different distributions'}")

    # Save statistics
    stats_dict = {
        'dataset': dataset_name,
        'train_mean': np.mean(train_targets),
        'train_median': np.median(train_targets),
        'train_std': np.std(train_targets),
        'train_min': np.min(train_targets),
        'train_max': np.max(train_targets),
        'train_skew': stats.skew(train_targets),
        'train_kurt': stats.kurtosis(train_targets),
        'test_mean': np.mean(test_targets),
        'test_median': np.median(test_targets),
        'test_std': np.std(test_targets),
        'log_train_mean': np.mean(train_log),
        'log_train_std': np.std(train_log),
        'log_train_skew': stats.skew(train_log),
        'shapiro_orig_p': p_shapiro_orig,
        'shapiro_log_p': p_shapiro_log,
        'outliers_iqr_count': outliers_iqr.sum(),
        'outliers_iqr_pct': 100*outliers_iqr.sum()/len(all_targets),
        'ks_test_p': p_ks,
    }

    stats_df = pd.DataFrame([stats_dict])
    stats_df.to_csv(f'{output_dir}/{dataset_name}_label_stats.csv', index=False)
    print(f"\n💾 Статистики зачувани: {dataset_name}_label_stats.csv")


def compare_all_datasets(output_dir: str = "figures/labels"):
    """Споредба на сите datasets"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    datasets = [
        "Half_Life_Obach",
        "Clearance_Hepatocyte_AZ",
        "Clearance_Microsome_AZ",
        "Caco2_Wang"
    ]

    all_data = []

    print(f"\n{'='*80}")
    print("CROSS-DATASET COMPARISON")
    print(f"{'='*80}\n")

    for dataset_name in datasets:
        try:
            data_api = ADME(name=dataset_name)
            split = data_api.get_split(method="scaffold")

            train_df = split["train"].dropna()
            test_df = split["test"].dropna()

            _, target_col = resolve_columns(train_df)

            train_targets = train_df[target_col].values
            test_targets = test_df[target_col].values

            for split_name, targets in [('Train', train_targets), ('Test', test_targets)]:
                for target in targets:
                    all_data.append({
                        'Dataset': dataset_name,
                        'Split': split_name,
                        'Target': target,
                        'log_Target': np.log(max(1e-6, target))
                    })

        except Exception as e:
            print(f"WARNING:  Грешка за {dataset_name}: {e}")

    df_all = pd.DataFrame(all_data)

    # ============ CROSS-DATASET ВИЗУЕЛИЗАЦИИ ============
    print("\n1. Cross-Dataset Distribution Comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Box plots - Original space
    sns.boxplot(data=df_all, x='Dataset', y='Target', hue='Split', ax=axes[0, 0],
                palette=['steelblue', 'coral'])
    axes[0, 0].set_title('Target Distribution Across Datasets - Original Space',
                         fontweight='bold', fontsize=13)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(alpha=0.3, axis='y')

    # Box plots - Log space
    sns.boxplot(data=df_all, x='Dataset', y='log_Target', hue='Split', ax=axes[0, 1],
                palette=['darkgreen', 'darkorange'])
    axes[0, 1].set_title('Target Distribution Across Datasets - Log Space',
                         fontweight='bold', fontsize=13)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(alpha=0.3, axis='y')

    # Violin plots - Original space
    sns.violinplot(data=df_all, x='Dataset', y='Target', hue='Split', ax=axes[1, 0],
                   palette=['steelblue', 'coral'], split=True)
    axes[1, 0].set_title('Target Distribution Across Datasets - Violin Plot (Original)',
                         fontweight='bold', fontsize=13)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(alpha=0.3, axis='y')

    # Violin plots - Log space
    sns.violinplot(data=df_all, x='Dataset', y='log_Target', hue='Split', ax=axes[1, 1],
                   palette=['darkgreen', 'darkorange'], split=True)
    axes[1, 1].set_title('Target Distribution Across Datasets - Violin Plot (Log)',
                         fontweight='bold', fontsize=13)
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cross_dataset_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   OK: Зачувано: cross_dataset_comparison.png")
    plt.close()

    print(f"\n{'='*80}")
    print("CROSS-DATASET ANALYSIS - КОМПЛЕТИРАНО!")
    print(f"{'='*80}")


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
            analyze_dataset_labels(dataset_name)
            print(f"\nOK: Завршено: {dataset_name}\n")
        except Exception as e:
            print(f"\nERROR: Грешка за {dataset_name}: {e}\n")

    # Cross-dataset comparison
    try:
        compare_all_datasets()
    except Exception as e:
        print(f"\nERROR: Грешка при cross-dataset comparison: {e}\n")

    print(f"\n{'='*80}")
    print("LABEL DISTRIBUTION ANALYSIS - КОМПЛЕТИРАНО!")
    print(f"{'='*80}")


if __name__ == "__main__":
    analyze_all_datasets()
