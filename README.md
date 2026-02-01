# MANU - Systematic HPO Benchmark for Molecular GNNs

**Systematic Hyperparameter Optimization for Molecular Property Prediction with Graph Neural Networks**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

A comprehensive benchmark comparing seven HPO algorithms (including TPE/Bayesian optimization) for GNN-based ADMET property prediction across six datasets from the Therapeutics Data Commons (TDC). Includes comparisons with foundation models (ChemBERTa, MolCLR) and multi-seed statistical validation.

### Key Statistics

| Metric | Value |
|--------|-------|
| **Datasets** | 6 (4 ADME + 2 Toxicity) |
| **Total Molecules** | 11,805 |
| **HPO Algorithms** | 7 (Random, PSO, ABC, GA, SA, HC, TPE) |
| **Trials per Run** | 50 |
| **Total HPO Runs** | 42 |
| **Total Model Evaluations** | 2,100 |
| **Multi-Seed Validation** | 5 seeds per dataset |
| **Foundation Models** | ChemBERTa (fine-tuned), MolCLR, Morgan-FP |

---

## Key Findings

1. **Random Search competitive for regression** - Wins on 2/4 ADME datasets (Caco2, Clearance_Microsome)
2. **TPE excels on complex clearance tasks** - Best on Clearance_Hepatocyte (47.52 vs 68.22 RMSE)
3. **Metaheuristic algorithms excel on classification** - SA wins on Tox21, ABC wins on hERG
4. **ChemBERTa fine-tuning improves toxicity prediction** - AUC 0.79 on hERG, 0.73 on Tox21
5. **No universal winner** - Algorithm selection should be task-dependent
6. **50 trials is sufficient** - Diminishing returns beyond this budget

---

## Results (50 Trials)

### ADME Regression (Test RMSE - lower is better)

| Dataset | Random | PSO | ABC | GA | SA | HC | TPE | ChemBERTa-FT |
|---------|--------|-----|-----|----|----|----|----|--------------|
| Caco2_Wang | **0.0027** | 0.0031 | 0.0029 | 0.0031 | 0.0029 | 0.0030 | 0.526 | 0.506 |
| Half_Life_Obach | 22.31 | **21.66** | 21.66 | 21.66 | 23.70 | 24.52 | 98.47 | 21.99 |
| Clearance_Hepatocyte | 68.22 | 70.21 | 72.04 | 71.34 | 72.04 | 72.04 | **47.52** | 49.39 |
| Clearance_Microsome | **38.75** | 42.76 | 42.29 | 42.29 | 40.94 | 41.63 | 39.04 | 43.25 |

### Toxicity Classification (Test AUC-ROC - higher is better)

| Dataset | Random | PSO | ABC | GA | SA | HC | TPE | ChemBERTa-FT |
|---------|--------|-----|-----|----|----|----|----|--------------|
| Tox21 | 0.713 | 0.692 | 0.735 | 0.735 | **0.743** | 0.652 | 0.742 | 0.735 |
| hERG | 0.747 | 0.747 | **0.825** | 0.747 | 0.802 | 0.821 | 0.745 | 0.791 |

### Winner Summary

| Algorithm | Wins | Datasets |
|-----------|------|----------|
| Random Search | 2/6 | Caco2, Clearance_Microsome |
| PSO | 1/6 | Half_Life (tie with ABC, GA) |
| TPE | 1/6 | Clearance_Hepatocyte |
| SA | 1/6 | Tox21 |
| ABC | 1/6 | hERG |

---

## Multi-Seed Validation (n=5 seeds)

Statistical robustness with 95% confidence intervals:

| Dataset | Task | Mean ± Std | 95% CI |
|---------|------|------------|--------|
| Caco2_Wang | Regression | 0.631 ± 0.065 | [0.550, 0.712] |
| Half_Life_Obach | Regression | 42.82 ± 41.09 | [-8.19, 93.84] |
| Clearance_Hepatocyte | Regression | 48.46 ± 3.12 | [44.58, 52.33] |
| Clearance_Microsome | Regression | 52.17 ± 9.64 | [40.20, 64.14] |
| Tox21 | Classification | 0.774 ± 0.061 | [0.698, 0.850] |
| hERG | Classification | 0.760 ± 0.062 | [0.684, 0.837] |

---

## Foundation Model Comparison

| Model | Caco2 (RMSE) | Half_Life (RMSE) | Tox21 (AUC) | hERG (AUC) |
|-------|--------------|------------------|-------------|------------|
| GNN-Best | **0.0027** | **21.66** | **0.743** | **0.825** |
| Morgan-FP | 0.614 | 22.12 | 0.722 | 0.611 |
| ChemBERTa | 0.496 | 27.39 | 0.728 | 0.770 |
| ChemBERTa-FT | 0.506 | 21.99 | 0.735 | **0.791** |
| MolE-FP | 0.670 | 25.01 | 0.675 | 0.672 |
| MolCLR | 0.713 | 21.97 | 0.538 | 0.504 |

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/NitramVonemats/MANU_Project.git
cd MANU_Project

# Install dependencies
pip install -r requirements.txt
```

### Run HPO (50 trials, all datasets)

```bash
python scripts/run_hpo_50_trials.py
```

### Run TPE Benchmark (Bayesian optimization)

```bash
python scripts/run_tpe_benchmark.py
```

### Run ChemBERTa Fine-tuning

```bash
python scripts/run_chemberta_finetune.py
```

### Run Multi-Seed Validation

```bash
python scripts/run_multi_seed_validation.py
```

### Generate Publication Figures

```bash
python scripts/generate_publication_figures.py
```

### Generate Visualizations

```bash
python scripts/create_hpo_visualizations.py
```

---

## Visualizations

### HPO Figures (`figures/hpo/`)

| Figure | Description |
|--------|-------------|
| `01_algorithm_performance.png` | ADME algorithm comparison |
| `02_best_hyperparameters.png` | Optimal hyperparameter heatmaps |
| `03_winner_analysis.png` | Winner per dataset analysis |
| `05_classification_performance.png` | Toxicity classification results |

### Publication Figures (`figures/paper/`)

| Figure | Description |
|--------|-------------|
| `hpo_comparison_with_tpe.png` | HPO algorithm comparison including TPE |
| `learning_curves.png` | Training convergence curves |
| `confusion_matrices.png` | Classification confusion matrices |
| `foundation_comparison_with_finetune.png` | Foundation model benchmarks |
| `multi_seed_boxplots.png` | Multi-seed validation distributions |
| `tpe_optimization_history.png` | TPE Bayesian optimization progress |

### Algorithm Performance (ADME)
![Algorithm Performance](figures/hpo/01_algorithm_performance.png)

### Winner Analysis
![Winner Analysis](figures/hpo/03_winner_analysis.png)

### Classification Performance (Toxicity)
![Classification Performance](figures/hpo/05_classification_performance.png)

### Foundation Model Comparison
![Foundation Comparison](figures/paper/foundation_comparison_with_finetune.png)

---

## Project Structure

```
MANU_Project/
├── optimized_gnn.py              # Main GNN implementation
├── adme_gnn/                     # Core GNN module
│   └── models/                   # Model implementations
├── optimization/                 # HPO algorithms
│   └── algorithms/              # PSO, ABC, GA, SA, HC, Random
├── scripts/
│   ├── run_hpo_50_trials.py     # 50-trial HPO runner
│   ├── run_tpe_benchmark.py     # TPE Bayesian optimization
│   ├── run_chemberta_finetune.py # ChemBERTa fine-tuning
│   ├── run_multi_seed_validation.py # Multi-seed validation
│   ├── generate_publication_figures.py # Publication figures
│   └── create_hpo_visualizations.py
├── runs/                         # HPO results (JSON)
│   ├── Caco2_Wang/              # Algorithm results per dataset
│   └── ...
├── results/
│   ├── tpe_benchmark/           # TPE optimization results
│   ├── chemberta_finetune/      # ChemBERTa fine-tuning results
│   └── multi_seed/              # Multi-seed validation results
├── figures/
│   ├── hpo/                     # HPO visualizations
│   └── paper/                   # Publication-ready figures
├── paper/                        # LaTeX paper
├── docs/                         # Documentation
├── DOCUMENTATION.md              # Complete project documentation
└── README.md
```

---

## Datasets

| Dataset | Task | Molecules | Metric |
|---------|------|-----------|--------|
| Caco2_Wang | Permeability | 910 | RMSE, R² |
| Half_Life_Obach | Half-life | 667 | RMSE, R² |
| Clearance_Hepatocyte | Clearance | 1,213 | RMSE, R² |
| Clearance_Microsome | Clearance | 1,102 | RMSE, R² |
| Tox21 | Toxicity | 7,258 | AUC-ROC, F1 |
| hERG | Cardiotoxicity | 655 | AUC-ROC, F1 |

---

## HPO Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| **Random** | Baseline | Uniform random sampling |
| **PSO** | Swarm | Particle Swarm Optimization |
| **ABC** | Swarm | Artificial Bee Colony |
| **GA** | Evolutionary | Genetic Algorithm |
| **SA** | Probabilistic | Simulated Annealing |
| **HC** | Local Search | Hill Climbing |
| **TPE** | Bayesian | Tree-structured Parzen Estimator (Optuna) |

---

## Practitioner Recommendations

| Task Type | Recommended | Reason |
|-----------|-------------|--------|
| **Regression** | Random Search | Fast, competitive results |
| **Classification** | SA or ABC | Better handles class imbalance |
| **Limited Budget** | TPE | Best sample efficiency |
| **Toxicity Prediction** | ChemBERTa-FT | Strong pretrained representations |
| **Quick Baseline** | Morgan-FP | Simple, interpretable |

---

## Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Complete project documentation
- **[docs/STATUS/](docs/STATUS/)** - Status reports
- **[paper/](paper/)** - LaTeX paper files

---

## License

MIT License

---

## Acknowledgments

- [Therapeutics Data Commons (TDC)](https://tdcommons.ai/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [NiaPy](https://github.com/NiaOrg/NiaPy)
- [Optuna](https://optuna.org/) - TPE optimization
- [Hugging Face Transformers](https://huggingface.co/) - ChemBERTa

---

*Last Updated: 2026-01-30*
*Total Compute: ~40 hours | 2,100+ model evaluations | 5-seed validation*
