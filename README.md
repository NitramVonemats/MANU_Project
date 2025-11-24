# GNN-based ADME Prediction - Optimized Pipeline

Comprehensive Graph Neural Network pipeline за предвидување на ADME (Absorption, Distribution, Metabolism, Excretion) својства на молекули.

## 📁 Project Structure

```
MANU_Project/
│
├── src/                                      # Core Model
│   └── optimized_gnn.py                     # Оптимизиран GNN модел (5L, 128H)
│
├── scripts/                                  # Analysis Scripts
│   ├── tanimoto_similarity_analysis.py      # Tanimoto similarity анализа
│   ├── label_distribution_analysis.py       # Label дистрибуција
│   ├── feature_label_correlation_analysis.py # Feature-label корелации
│   ├── create_publication_figures.py        # Publication-quality фигури
│   ├── run_all_analyses.py                  # Master скрипта
│   └── README_ANALYSES.md                   # Документација за анализи
│
├── figures/                                  # Visualizations
│   ├── publication/                         # Publication-ready фигури (4 фајла)
│   ├── similarity/                          # Tanimoto анализи (16 фајла)
│   ├── labels/                              # Label дистрибуции (21 фајл)
│   └── correlations/                        # Feature корелации (32 фајла)
│
├── docs/                                     # Documentation
│   ├── METHODOLOGY.md                       # Comprehensive методологија (395 lines)
│   └── FINAL_REPORT.md                      # Финален извештај (430 lines)
│
├── GNN_test/                                 # Core Infrastructure
│   ├── graph/                               # Graph construction и featurization
│   ├── models/                              # Model architectures
│   ├── services/                            # Training services
│   ├── functional/                          # Utilities (metrics, transforms)
│   ├── configs/                             # Hyperparameter configs
│   ├── visualizations/                      # Visualization scripts
│   └── archive/                             # Archived old code
│       ├── old_tests/                       # Old test files
│       ├── old_models/                      # Old model versions
│       ├── old_scripts/                     # Old analysis scripts
│       └── README.md                        # Archive documentation
│
└── requirements.txt                          # Dependencies
```

## 🔬 Datasets

Сите datasets од Therapeutics Data Commons (TDC):

| Dataset | Compounds | Property |
|---------|-----------|----------|
| Half_Life_Obach | 667 | Half-life во крв |
| Clearance_Hepatocyte_AZ | 1,213 | Hepatocyte clearance |
| Clearance_Microsome_AZ | 1,102 | Microsomal clearance |
| Caco2_Wang | 906 | Caco-2 permeability |

## 🏆 Best Results

| Dataset | Test RMSE | Test R² | Test MAE |
|---------|-----------|---------|----------|
| **Half_Life_Obach** | 0.8388 | 0.2765 | 0.65 |
| **Clearance_Hepatocyte_AZ** | 1.1921 | 0.0868 | 0.92 |
| **Clearance_Microsome_AZ** | 1.0184 | 0.3208 | 0.78 |
| **Caco2_Wang** | 17.686 | 0.3357 | - |

**Оптимална конфигурација**: Graph architecture, 5 layers, 128 hidden channels

## ✨ Key Findings

1. **Graph модел е најдобар** - 20-100× подобар од GCN, GAT, GIN, SAGE
2. **Edge features го влошуваат performance** - 3.5× worse
3. **Dropout не е потребен** - За мали datasets
4. **Оптимални хиперпараметри**:
   - Layers: 5
   - Hidden channels: 128
   - Learning rate: 0.001
   - NO edge features
   - NO dropout

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install torch torch-geometric rdkit PyTDC pandas numpy scikit-learn matplotlib seaborn scipy

# Or use requirements.txt
pip install -r requirements.txt
```

### Running the Optimized Model

```bash
# Run benchmark on all datasets
python src/optimized_gnn.py
```

### Running Comprehensive Analyses

```bash
# Run ALL analyses (Tanimoto + Labels + Correlations)
cd scripts
python run_all_analyses.py

# Or run individual analyses
python scripts/tanimoto_similarity_analysis.py          # ~5-8 min
python scripts/label_distribution_analysis.py           # ~3-5 min
python scripts/feature_label_correlation_analysis.py    # ~8-12 min

# Generate publication figures (fast, no dataset loading)
python scripts/create_publication_figures.py
```

## 📊 Comprehensive Analyses

### 1. Tanimoto Similarity Analysis
- Morgan fingerprints (ECFP4)
- Pairwise similarity matrices
- Train-Test similarity distributions
- Similarity-target correlations

**Output**: `figures/similarity/` (16 фајлови)

### 2. Label Distribution Analysis
- Original vs Log space distributions
- Box plots и Violin plots
- Outlier detection (IQR, Z-score, Percentile)
- Q-Q plots (normalност)
- Cross-dataset comparisons

**Output**: `figures/labels/` (21 фајл)

### 3. Feature-Label Correlation Analysis
- 19 ADME дескриптори (MW, LogP, HBD, HBA, TPSA, ...)
- Pearson и Spearman correlations
- Scatter plots, feature distributions
- Feature importance ranking

**Output**: `figures/correlations/` (32 фајла)

### 4. Publication-Quality Figures
- GNN architecture diagram
- Performance summary (RMSE, R², MAE)
- Ablation study (4 панели)
- Methodology flowchart

**Output**: `figures/publication/` (4 фајла)

## 📖 Documentation

- **`docs/METHODOLOGY.md`** - Comprehensive methodology (395 lines)
  - Datasets, preprocessing, splits
  - 8 model architectures tested
  - Training procedure, hyperparameters
  - Ablation studies, baselines
  - Statistical analysis

- **`docs/FINAL_REPORT.md`** - Final report (430 lines)
  - Top 5 models, architecture comparisons
  - Performance analysis, recommendations

- **`scripts/README_ANALYSES.md`** - Analysis documentation
  - How to run analyses
  - Interpretation guide
  - Troubleshooting

## 🛠️ Development

### Project Organization
- **Active code**: `src/`, `scripts/`, `GNN_test/`
- **Archived code**: `GNN_test/archive/` (old versions, test files)
- **Documentation**: `docs/`, `README.md`
- **Results**: `figures/`, CSV files

### Contributing
Контакт за прашања или подобрувања.

## 📚 References

1. **TDC**: Therapeutics Data Commons - https://tdcommons.ai/
2. **RDKit**: Open-Source Cheminformatics - https://www.rdkit.org/
3. **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

---

**Generated**: 2025-11-24
**Project**: MANU - Molecular ADME Prediction with Graph Neural Networks
**Best Model**: Graph (5 layers, 128 hidden channels)
