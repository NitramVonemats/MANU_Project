# Comprehensive ADME Dataset Analysis

Оваа документација опишува ги сите имплементирани анализи и визуелизации за ADME предвидување со GNN.

## 📁 Структура на Скрипти

```
scripts/
├── tanimoto_similarity_analysis.py       # Tanimoto similarity анализа
├── label_distribution_analysis.py        # Label дистрибуција анализа
├── feature_label_correlation_analysis.py # Feature-label корелации
├── create_publication_figures.py         # Publication-quality figures
├── run_all_analyses.py                   # Master скрипта (извршува ги сите)
└── README_ANALYSES.md                    # Ова
```

## 🚀 Quick Start

### Извршување на сите анализи:
```bash
cd scripts
python run_all_analyses.py
```

### Извршување на индивидуални анализи:
```bash
# Tanimoto similarity анализа
python tanimoto_similarity_analysis.py

# Label distribution анализа
python label_distribution_analysis.py

# Feature-label correlation анализа
python feature_label_correlation_analysis.py

# Publication figures (не бара dataset loading)
python create_publication_figures.py
```

## 📊 Генерирани Визуелизации

### 1. Tanimoto Similarity Analysis (`figures/similarity/`)

**Цел**: Анализа на молекуларна сличност помеѓу compounds користејќи Morgan fingerprints (ECFP4).

**Генерирани фајлови по dataset:**
- `{dataset}_similarity_matrix.png` - Heatmap и distribution
- `{dataset}_train_test_similarity.png` - Train-Test similarity analysis
- `{dataset}_similarity_target_correlation.png` - Similarity vs Target correlation
- `{dataset}_similarity_stats.csv` - Statistical summary

**Интерпретација:**
- **High similarity (>0.7)**: Молекулите се многу слични
- **Negative correlation**: Поголема сличност → помала разлика во targets

---

### 2. Label Distribution Analysis (`figures/labels/`)

**Цел**: Детална анализа на дистрибуција на target вредности.

**Генерирани фајлови по dataset:**
- `{dataset}_distribution_comparison.png` - Original vs Log space
- `{dataset}_boxplots_violinplots.png` - Box и Violin plots
- `{dataset}_outlier_detection.png` - 3 методи (IQR, Z-score, Percentile)
- `{dataset}_qqplot.png` - Normalност тест
- `{dataset}_label_stats.csv` - Statistical summary

**Cross-dataset:**
- `cross_dataset_comparison.png` - Споредба на сите 4 datasets

---

### 3. Feature-Label Correlation Analysis (`figures/correlations/`)

**Цел**: Анализа на корелација помеѓу ADME features и targets.

**Генерирани фајлови по dataset:**
- `{dataset}_pearson_correlation.png` - Pearson correlations
- `{dataset}_spearman_correlation.png` - Spearman correlations
- `{dataset}_scatter_plots.png` - Топ 6 корелации
- `{dataset}_feature_distributions.png` - Топ 4 feature distributions
- `{dataset}_feature_importance_comparison.png` - Pearson vs Spearman
- `{dataset}_pairwise_correlations.png` - Топ 10 features heatmap
- `{dataset}_correlation_stats.csv` - Correlation statistics
- `{dataset}_features_data.csv` - Full features dataset

**ADME Features (19 дескриптори):**
MW, LogP, HBD, HBA, TPSA, RotatableBonds, AromaticRings, AliphaticRings,
Heteroatoms, HeavyAtoms, FractionCSP3, MolMR, BertzCT, Chi0v, Rings,
Lipinski violations (MW, LogP, HBD, HBA)

---

### 4. Publication-Quality Figures (`figures/publication/`)

**Генерирани фајлови:**
- `gnn_architecture_diagram.png` - Детална GNN архитектура
- `performance_summary.png` - Резултати (RMSE, R², MAE)
- `ablation_study.png` - Design decisions (4 панели)
- `methodology_flowchart.png` - Целосен pipeline

**Карактеристики:**
- 300 DPI резолуција
- Publication-ready styling
- Готови за LaTeX/Word документи

---

## 🔍 Datasets Анализирани

1. **Half_Life_Obach** - Half-life во крв (667 compounds)
2. **Clearance_Hepatocyte_AZ** - Hepatocyte clearance (1,213 compounds)
3. **Clearance_Microsome_AZ** - Microsomal clearance (1,102 compounds)
4. **Caco2_Wang** - Caco-2 permeability (906 compounds)

---

## 📦 Dependencies

```bash
pip install matplotlib seaborn scipy rdkit pandas numpy tdc torch torch_geometric
```

---

## 🛠️ Troubleshooting

### Problem: ModuleNotFoundError
```bash
pip install matplotlib seaborn scipy rdkit
```

### Problem: Unicode грешки на Windows
- Користи `chcp 65001` пред извршување
- Или користи `python -X utf8 script.py`

### Problem: Memory грешка
- Скриптите користат sampling (max 500 molecules)
- За поголеми datasets, намалете `max_size` параметарот

---

## 📝 How to Use Results

### За Research Paper:
1. `figures/publication/` за methods section
2. `figures/correlations/` за feature analysis
3. `figures/similarity/` за dataset characterization
4. `figures/labels/` за data distribution

### За Презентација:
1. `methodology_flowchart.png` (overview)
2. `gnn_architecture_diagram.png` (model)
3. `ablation_study.png` (design)
4. `performance_summary.png` (results)

---

Generated: 2025-11-24
Project: MANU - Molecular ADME Prediction with Graph Neural Networks
