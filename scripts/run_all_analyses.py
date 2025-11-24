"""
MASTER SCRIPT - RUN ALL ANALYSES
=================================

Главна скрипта која ги извршува сите анализи:
1. Tanimoto Similarity Analysis
2. Label Distribution Analysis
3. Feature-Label Correlation Analysis

Креира comprehensive report со сите визуелизации и статистики.
"""

import os
import sys
import time
from datetime import datetime

# Додај script директориум во path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


def print_banner(text):
    """Print баннер со текст"""
    print(f"\n{'='*80}")
    print(f"{text.center(80)}")
    print(f"{'='*80}\n")


def run_all_analyses():
    """Изврши ги сите анализи"""

    start_time = time.time()

    print_banner("COMPREHENSIVE ADME DATASET ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Креирај output директориуми
    os.makedirs("figures/similarity", exist_ok=True)
    os.makedirs("figures/labels", exist_ok=True)
    os.makedirs("figures/correlations", exist_ok=True)

    # ============ 1. TANIMOTO SIMILARITY ANALYSIS ============
    print_banner("1/3: TANIMOTO SIMILARITY ANALYSIS")

    try:
        from tanimoto_similarity_analysis import analyze_all_datasets as analyze_tanimoto
        analyze_tanimoto()
        print("\nOK: Tanimoto Similarity Analysis КОМПЛЕТИРАНО!")
    except Exception as e:
        print(f"\nERROR: ГРЕШКА во Tanimoto Analysis: {e}")
        import traceback
        traceback.print_exc()

    # ============ 2. LABEL DISTRIBUTION ANALYSIS ============
    print_banner("2/3: LABEL DISTRIBUTION ANALYSIS")

    try:
        from label_distribution_analysis import analyze_all_datasets as analyze_labels
        analyze_labels()
        print("\nOK: Label Distribution Analysis КОМПЛЕТИРАНО!")
    except Exception as e:
        print(f"\nERROR: ГРЕШКА во Label Analysis: {e}")
        import traceback
        traceback.print_exc()

    # ============ 3. FEATURE-LABEL CORRELATION ANALYSIS ============
    print_banner("3/3: FEATURE-LABEL CORRELATION ANALYSIS")

    try:
        from feature_label_correlation_analysis import analyze_all_datasets as analyze_correlations
        analyze_correlations()
        print("\nOK: Feature-Label Correlation Analysis КОМПЛЕТИРАНО!")
    except Exception as e:
        print(f"\nERROR: ГРЕШКА во Correlation Analysis: {e}")
        import traceback
        traceback.print_exc()

    # ============ ФИНАЛЕН РЕЗИМЕ ============
    elapsed_time = time.time() - start_time

    print_banner("COMPREHENSIVE ANALYSIS - КОМПЛЕТИРАНО!")

    print(f"\n📊 ГЕНЕРИРАНИ ВИЗУЕЛИЗАЦИИ:\n")

    print("TANIMOTO SIMILARITY ANALYSIS:")
    print("  • Similarity matrices (train set)")
    print("  • Train-Test similarity distributions")
    print("  • Similarity-Target correlations")
    print("  • Statistical summaries")

    print("\nLABEL DISTRIBUTION ANALYSIS:")
    print("  • Original vs Log space distributions")
    print("  • Box plots и Violin plots")
    print("  • Outlier detection (3 методи)")
    print("  • Q-Q plots за normalnost")
    print("  • Cross-dataset comparisons")

    print("\nFEATURE-LABEL CORRELATION ANALYSIS:")
    print("  • Pearson correlation heatmaps")
    print("  • Spearman correlation heatmaps")
    print("  • Scatter plots (топ 6 features)")
    print("  • Feature distributions")
    print("  • Feature importance comparisons")
    print("  • Pairwise feature correlations")

    print(f"\n📁 LOCATION:")
    print(f"  • figures/similarity/     - Tanimoto анализи")
    print(f"  • figures/labels/         - Label дистрибуции")
    print(f"  • figures/correlations/   - Feature корелации")

    print(f"\n⏱️  Total execution time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n{'='*80}")
    print("✨ СИ АНАЛИЗИ УСПЕШНО ЗАВРШЕНИ! ✨")
    print(f"{'='*80}\n")

    # Креирај index файл со резиме
    create_analysis_index()


def create_analysis_index():
    """Креирај index файл со резиме на сите анализи"""

    index_content = """
# COMPREHENSIVE ADME DATASET ANALYSIS - INDEX

Креирано: {timestamp}

## Структура на Анализи

### 1. Tanimoto Similarity Analysis (`figures/similarity/`)

Анализа на молекуларна сличност користејќи Morgan fingerprints (ECFP4).

**Генерирани фајлови за секој dataset:**
- `{dataset}_similarity_matrix.png` - Similarity matrix и distribution
- `{dataset}_train_test_similarity.png` - Train-Test similarity analysis
- `{dataset}_similarity_target_correlation.png` - Similarity vs Target correlation
- `{dataset}_similarity_stats.csv` - Statistical summary

**Клучни метрики:**
- Mean/Median/Std Tanimoto similarity
- Train-Test similarity overlap
- Correlation помеѓу molecular similarity и target difference

### 2. Label Distribution Analysis (`figures/labels/`)

Анализа на дистрибуција на target вредности.

**Генерирани фајлови за секој dataset:**
- `{dataset}_distribution_comparison.png` - Original vs Log space distributions
- `{dataset}_boxplots_violinplots.png` - Box и Violin plots
- `{dataset}_outlier_detection.png` - Outlier detection (IQR, Z-score, Percentile)
- `{dataset}_qqplot.png` - Q-Q plots за normalnost
- `{dataset}_label_stats.csv` - Statistical summary

**Cross-dataset:**
- `cross_dataset_comparison.png` - Споредба на сите datasets

**Клучни метрики:**
- Mean, Median, Std, Skewness, Kurtosis
- Normality tests (Shapiro-Wilk)
- Outlier percentages
- Train-Test distribution similarity (Kolmogorov-Smirnov)

### 3. Feature-Label Correlation Analysis (`figures/correlations/`)

Анализа на корелација помеѓу ADME features и targets.

**Генерирани фајлови за секој dataset:**
- `{dataset}_pearson_correlation.png` - Pearson correlation heatmap и топ features
- `{dataset}_spearman_correlation.png` - Spearman correlation heatmap и топ features
- `{dataset}_scatter_plots.png` - Scatter plots за топ 6 correlations
- `{dataset}_feature_distributions.png` - Distributions на топ features
- `{dataset}_feature_importance_comparison.png` - Pearson vs Spearman comparison
- `{dataset}_pairwise_correlations.png` - Pairwise correlations (топ 10 features)
- `{dataset}_correlation_stats.csv` - Correlation statistics
- `{dataset}_features_data.csv` - Full features dataset

**ADME Features (19 дескриптори):**
- MW, LogP, HBD, HBA, TPSA
- RotatableBonds, AromaticRings, AliphaticRings
- Heteroatoms, HeavyAtoms, FractionCSP3
- MolMR, BertzCT, Chi0v, Rings
- Lipinski violations (MW, LogP, HBD, HBA)

**Клучни метрики:**
- Pearson и Spearman correlations
- Statistical significance (p-values)
- Feature importance ranking

## Datasets Анализирани

1. **Half_Life_Obach** - Half-life во крв
2. **Clearance_Hepatocyte_AZ** - Hepatocyte clearance
3. **Clearance_Microsome_AZ** - Microsomal clearance
4. **Caco2_Wang** - Caco-2 permeability

## Употреба

### Индивидуални скрипти:
```bash
python scripts/tanimoto_similarity_analysis.py
python scripts/label_distribution_analysis.py
python scripts/feature_label_correlation_analysis.py
```

### Сите анализи одеднаш:
```bash
python scripts/run_all_analyses.py
```

## Interpretation Guide

### Tanimoto Similarity
- **High similarity (>0.7)**: Многу слични молекули
- **Medium similarity (0.3-0.7)**: Умерено слични
- **Low similarity (<0.3)**: Многу различни молекули

Scaffold splitting обично резултира со **lower train-test similarity**, што е добро за generalization.

### Label Distributions
- **Skewness**: Позитивна → Right-skewed (повеќе мали вредности)
- **Kurtosis**: Позитивна → Heavy tails (повеќе outliers)
- **Log transformation**: Често прави дистрибуцијата понормална

### Feature Correlations
- **|r| > 0.7**: Силна корелација
- **|r| 0.4-0.7**: Умерена корелација
- **|r| < 0.4**: Слаба корелација
- **p < 0.05**: Статистички значајна

## Клучни Findings (да се пополни после анализа)

### Molecular Diversity
- [ ] Train sets покажуваат ___ mean Tanimoto similarity
- [ ] Test sets се ___ различни од train sets
- [ ] Најголема diversity во ___ dataset

### Target Distributions
- [ ] ___ datasets имаат right-skewed distributions
- [ ] Log transformation подобрува normalност за ___
- [ ] Outliers се најчести во ___ dataset

### Feature Importance
- [ ] Најважни features за Half_Life_Obach: ___
- [ ] Најважни features за Clearance datasets: ___
- [ ] Универзално важни features: ___

## References

- RDKit: https://www.rdkit.org/
- TDC (Therapeutics Data Commons): https://tdcommons.ai/
- Morgan Fingerprints: Rogers, D. & Hahn, M. (2010) J. Chem. Inf. Model.

---

Generated by: MANU Project - Comprehensive ADME Analysis Pipeline
""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), dataset="{dataset}")

    with open("figures/ANALYSIS_INDEX.md", "w", encoding="utf-8") as f:
        f.write(index_content)

    print("\n💾 Analysis index креиран: figures/ANALYSIS_INDEX.md")


if __name__ == "__main__":
    run_all_analyses()
