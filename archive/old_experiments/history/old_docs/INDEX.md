# Визуелизации - Индекс

**Вкупно:** 66 фајлови (41 PNG + 24 CSV + 1 summary)

---

## 📂 COMPARATIVE (Unified plots)

Локација: `figures/comparative/`

| Фајл | Опис | За Paper |
|------|------|----------|
| `01_dataset_overview.png` | 4-panel: Dataset overview | Dataset Section |
| `02_label_distributions.png` | ADME + Tox distributions | Dataset Section |
| `03_feature_importance.png` | Feature importance heatmap | Results Section |
| `04_tanimoto_similarity.png` | Molecular similarity | Dataset Section |
| `05_summary_table.png` | Summary statistics | Dataset Section |

---

## 📂 HPO (Hyperparameter Optimization Results)

Локација: `figures/hpo/`

| Фајл | Опис | За Paper |
|------|------|----------|
| `01_algorithm_performance.png` | RMSE, R², Time comparison | Results Section |
| `02_best_hyperparameters.png` | Optimal hyperparameters | Results Section |
| `03_winner_analysis.png` | Best algorithm per dataset | Results Section |
| `04_summary_table.png` | HPO summary table | Results Section |

---

## 📂 ABLATION STUDIES

Локација: `figures/ablation_studies/`

### Per-Dataset (8 фајлови):
- `{dataset}_hyperparameter_comparison.png` - 4 datasets
- `{dataset}_hyperparameter_space.png` - 4 datasets

### Unified (4 фајлови):
- `unified_hyperparameter_correlations.png`
- `unified_hyperparameter_heatmaps.png`
- `ablation_summary_table.png`
- `ablation_summary.csv`

За paper: **Results Section (Ablation)**

---

## 📂 PER-DATASET ANALYSIS

Локација: `figures/per_dataset_analysis/{dataset}/`

За секој од **7 datasets** (6 фајлови):

| Фајл | Опис |
|------|------|
| `tanimoto_similarity.png` | Similarity histogram + boxplot |
| `similarity_stats.csv` | Similarity statistics |
| `label_distribution.png` | Label distribution plots |
| `label_stats.csv` | Label statistics |
| `feature_correlations.png` | Feature-label correlations |
| `correlation_stats.csv` | Correlation statistics |

**Datasets:**
- Caco2_Wang
- Half_Life_Obach
- Clearance_Hepatocyte_AZ
- Clearance_Microsome_AZ
- Tox21
- hERG
- ClinTox

За paper: **Dataset Section (детали)**

---

## 🚀 Брзи команди

```bash
# Отвори comparative plots
explorer figures\comparative

# Отвори HPO results
explorer figures\hpo

# Отвори ablation studies
explorer figures\ablation_studies

# Отвори Caco2 analysis
explorer figures\per_dataset_analysis\Caco2_Wang

# Отвори сите
explorer figures
```

---

## 📊 За Paper Writing

### Dataset Section:
1. `comparative/01_dataset_overview.png`
2. `comparative/02_label_distributions.png`
3. `comparative/04_tanimoto_similarity.png`
4. `comparative/05_summary_table.png`
5. Per-dataset details: `per_dataset_analysis/{dataset}/`

### Results Section:
1. `hpo/01_algorithm_performance.png`
2. `hpo/02_best_hyperparameters.png`
3. `hpo/03_winner_analysis.png`
4. `comparative/03_feature_importance.png`
5. Ablation: `ablation_studies/unified_*.png`

### Supplementary:
- Per-dataset detailed analysis
- CSV табели со статистики
