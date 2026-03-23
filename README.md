Systematic HPO Benchmark for Molecular GNNs

**Framework for Benchmarking and Optimization of Small Molecule Foundation Models for ADMET**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.4+-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Authors:** Martin, Mila, Viktorija, Ilinka
> **Paper:** [`paper_1/main.tex`](paper_1/main.tex)


---

## Overview

MANU is a reproducible benchmarking framework that systematically evaluates **seven hyperparameter optimization (HPO) strategies** for Graph Neural Networks (GNNs) on **six ADMET datasets** from the Therapeutics Data Commons (TDC). It additionally compares optimized GNNs against frozen foundation model baselines (ChemBERTa, MolCLR, Morgan-FP, MolE-FP), and provides multi-seed statistical validation with confidence intervals.

The framework answers two core questions:
1. **Which HPO algorithm should practitioners choose** for GNN-based molecular property prediction under scaffold-split evaluation?
2. **Can task-specific GNNs with systematic HPO match or exceed frozen pretrained foundation models?**

### Key Statistics

| Metric | Value |
|--------|-------|
| **Datasets** | 6 (4 ADME regression + 2 Toxicity classification) |
| **Total molecules** | 11,805 |
| **HPO algorithms** | 7 (Random, PSO, ABC, GA, SA, HC, TPE) |
| **Trials per run** | 50 |
| **Total HPO runs** | 42 (6 datasets x 7 algorithms) |
| **Total model evaluations** | 2,100+ |
| **Multi-seed validation** | 5 seeds per dataset |
| **Foundation model baselines** | 4 (ChemBERTa, MolCLR, Morgan-FP, MolE-FP) |
| **GNN backbone** | GCN (GraphConv) |
| **Evaluation protocol** | Scaffold split (Bemis-Murcko, 80/10/10) |
| **Hardware** | NVIDIA RTX 3060, i7-8700K, 16 GB RAM |
| **Total compute** | ~45 hours |

---

## Key Findings

1. **No universal optimizer exists.** Random Search wins on 3/4 regression tasks; metaheuristics (SA, ABC) win on classification. Algorithm choice is task-dependent.
2. **Random Search is a strong baseline.** No metaheuristic achieves statistically significant improvement over Random Search (Wilcoxon signed-rank, p > 0.05) under a 50-trial budget with scaffold split.
3. **Scaffold-split evaluation changes optimizer rankings.** The noisy validation landscape induced by scaffold split reduces the advantage of adaptive metaheuristics compared to random-split settings.
4. **GNNs outperform frozen foundation models on toxicity.** hERG: GNN AUC=0.825 vs ChemBERTa 0.770. Tox21: GNN AUC=0.742 vs ChemBERTa 0.728.
5. **Structure-only models fail on complex PK.** Hepatocyte clearance R^2 = -1.02 (worse than predicting the mean). Foundation models provide a more stable starting point on this task.
6. **Dataset difficulty varies dramatically.** From hERG (AUC=0.825, strong) to Hepatocyte clearance (R^2=-1.02, impossible).

---

## Results

### HPO Algorithm Comparison (50 Trials, Seed 42)

#### ADME Regression (Test RMSE -- lower is better)

| Dataset | PSO | ABC | GA | SA | HC | Random | TPE |
|---------|-----|-----|----|----|-----|--------|-----|
| Caco2_Wang | 0.0031 | 0.0029 | 0.0031 | 0.0029 | 0.0030 | **0.0027** | 0.0030 |
| Half_Life_Obach | **21.66** | **21.66** | **21.66** | 23.70 | 24.52 | 22.31 | 22.34 |
| Clearance_Hepatocyte_AZ | 70.21 | 72.04 | 71.34 | 72.04 | 72.04 | 68.22 | **52.16** |
| Clearance_Microsome_AZ | 42.76 | 42.29 | 42.29 | 40.94 | 41.63 | **38.75** | 44.34 |

#### Toxicity Classification (Test AUC-ROC -- higher is better)

| Dataset | PSO | ABC | GA | SA | HC | Random | TPE |
|---------|-----|-----|----|----|-----|--------|-----|
| Tox21 (NR-AR) | 0.692 | 0.735 | 0.735 | **0.742** | 0.652 | 0.713 | 0.705 |
| hERG | 0.747 | **0.825** | 0.747 | 0.802 | 0.821 | 0.747 | 0.772 |

> **Note:** TPE uses Optuna and additionally searches over dropout (8-dim space), while NiaPy-based algorithms share a 7-dim search space.

### Multi-Seed Validation (5 Seeds)

| Dataset | Task | Metric | Mean +/- Std (95% CI) |
|---------|------|--------|----------------------|
| Caco2_Wang | Regr. | RMSE | 0.0033 +/- 0.0005 (0.0027--0.0039) |
| Half_Life_Obach | Regr. | RMSE | 20.05 +/- 1.17 (18.61--21.50) |
| Clearance_Hepatocyte_AZ | Regr. | RMSE | 52.37 +/- 2.87 (48.81--55.93) |
| Clearance_Microsome_AZ | Regr. | RMSE | 53.46 +/- 13.56 (36.63--70.30) |
| Tox21 (NR-AR) | Class. | AUC | 0.711 +/- 0.012 (0.696--0.727) |
| hERG | Class. | AUC | 0.805 +/- 0.022 (0.778--0.832) |

### Foundation Model Comparison

| Model | Caco2 (R^2) | Half_Life (RMSE) | Clear_Hep (RMSE) | Clear_Micro (RMSE) | Tox21 (AUC) | hERG (AUC) |
|-------|------------|------------------|-------------------|---------------------|-------------|------------|
| **GNN-Best** | 0.48 | **21.66** | 68.22 | **38.75** | **0.743** | **0.825** |
| Morgan-FP | -- | 22.12 | **48.36** | 40.36 | 0.722 | 0.611 |
| ChemBERTa | 0.48 | 27.39 | **47.31** | 42.56 | 0.728 | 0.770 |
| MolE-FP | -- | 25.01 | **47.22** | 41.79 | 0.675 | 0.672 |
| MolCLR | -- | 21.71 | 48.92 | 42.19 | 0.452 | 0.401 |

> Caco2 comparison uses R^2 (scale-invariant) because GNN reports RMSE in original units while foundation models use z-score-normalised space.

---

## Quick Start

### Installation

```bash
git clone https://github.com/NitramVonemats/MANU_Project.git
cd MANU_Project
pip install -r requirements.txt
```

### Run HPO Benchmark (50 trials, all algorithms, all datasets)

```bash
python scripts/run_hpo_50_trials.py
```

### Run TPE Benchmark (Optuna)

```bash
python scripts/run_tpe_benchmark.py
```

### Run Foundation Model Baselines

```bash
python scripts/run_complete_foundation_benchmark.py
python scripts/run_chemberta_finetune.py
```

### Run Multi-Seed Validation

```bash
python scripts/run_multi_seed_validation.py
```

### Generate Visualizations

```bash
python scripts/create_hpo_visualizations.py
python scripts/create_foundation_comparison_plots.py
```

---

## Project Structure

```
MANU/
|-- paper_1/                          # LaTeX paper
|   |-- main.tex                      # Main manuscript
|   |-- refs.bib                      # Bibliography
|   `-- images/                       # Paper figures (PNG)
|
|-- src/core/                         # Core source code
|   |-- optimized_gnn.py              # GNN model, training, evaluation
|   `-- model_comparison.py           # Model comparison utilities
|
|-- optimization/                     # HPO framework
|   |-- space.py                      # 7-dim search space definition
|   |-- problem.py                    # NiaPy problem wrapper
|   |-- runner.py                     # HPO execution runner
|   |-- foundation_problem.py         # Foundation model HPO wrapper
|   |-- foundation_runner.py          # Foundation model HPO runner
|   `-- algorithms/                   # Algorithm implementations
|       |-- pso.py                    # Particle Swarm Optimization
|       |-- genetic.py                # Genetic Algorithm
|       |-- abc.py                    # Artificial Bee Colony
|       |-- simulated_annealing.py    # Simulated Annealing
|       |-- hill_climbing.py          # Hill Climbing
|       `-- random_search.py          # Random Search
|
|-- scripts/                          # Execution and analysis scripts
|   |-- run_hpo_50_trials.py          # Main HPO runner (50 trials)
|   |-- run_tpe_benchmark.py          # TPE via Optuna
|   |-- run_multi_seed_validation.py  # 5-seed validation
|   |-- run_chemberta_finetune.py     # ChemBERTa fine-tuning
|   |-- run_complete_foundation_benchmark.py
|   |-- create_hpo_visualizations.py  # HPO figures
|   |-- create_foundation_comparison_plots.py
|   |-- statistical_significance_tests.py
|   `-- analyses/                     # Detailed analysis scripts
|
|-- runs/                             # HPO results (JSON, per dataset/algo)
|   |-- Caco2_Wang/                   # 6 algo result files
|   |-- Half_Life_Obach/
|   |-- Clearance_Hepatocyte_AZ/
|   |-- Clearance_Microsome_AZ/
|   |-- tox21/
|   `-- herg/
|
|-- results/                          # Processed results
|   |-- multi_seed/                   # 5-seed validation results
|   |-- tpe_benchmark/                # TPE results (6 datasets)
|   |-- foundation_benchmark/         # Foundation model comparison CSV
|   |-- chemberta_finetune/           # ChemBERTa fine-tuning results
|   |-- figures/                      # Generated tables and figures
|   `-- hpo/                          # Processed HPO results
|
|-- datasets/                         # Raw datasets (CSV)
|   |-- adme/                         # 4 ADME regression datasets
|   `-- toxicity/                     # Tox21, hERG, ClinTox
|
|-- external/MolCLR/                  # MolCLR pretrained checkpoints
|-- figures/paper/                    # Generated LaTeX tables
|-- archive/                          # Old experiments and scripts
|-- requirements.txt                  # Python dependencies
`-- README.md                         # This file
```

---

## Datasets

All datasets are from the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) ADMET benchmark.

| Dataset | Task | Molecules | Primary Metric | Difficulty |
|---------|------|-----------|----------------|------------|
| Caco2_Wang | Permeability (regression) | 910 | RMSE, R^2 | Moderate (R^2=0.48) |
| Half_Life_Obach | Half-life (regression) | 667 | RMSE, R^2 | Very Hard (R^2=0.004) |
| Clearance_Hepatocyte_AZ | Clearance (regression) | 1,213 | RMSE, R^2 | Impossible (R^2=-1.02) |
| Clearance_Microsome_AZ | Clearance (regression) | 1,102 | RMSE, R^2 | Weak (R^2=0.19) |
| Tox21 (NR-AR) | Toxicity (classification) | 7,258 | AUC-ROC | Moderate (3.5% pos) |
| hERG | Cardiotoxicity (classification) | 655 | AUC-ROC | Good (AUC=0.825) |

Splitting: Bemis-Murcko scaffold split (80/10/10 train/val/test), seed 42.

---

## HPO Algorithms

| Algorithm | Type | Framework | Config |
|-----------|------|-----------|--------|
| Random Search | Baseline | NiaPy | Uniform sampling |
| PSO | Swarm intelligence | NiaPy | pop=16, C1=2.0, C2=2.0, w=0.7 |
| ABC | Swarm intelligence | NiaPy | colony=16, limit=50 |
| GA | Evolutionary | NiaPy | pop=16, mutation=0.1, crossover=0.8 |
| SA | Probabilistic | NiaPy | T0=50, alpha=0.99 |
| HC | Local search | NiaPy | Greedy, single init |
| TPE | Bayesian | Optuna | 10 startup + 40 TPE, median pruning |

### Search Space (7 dimensions for NiaPy, 8 for TPE)

| Hyperparameter | Range | Type |
|----------------|-------|------|
| Hidden dimensions | {64, 96, 128, 192, 256, 384, 512} | Categorical |
| Number of layers | {3, 4, 5, 6, 7} | Categorical |
| MLP head layer 1 | {128, 192, 256, 384, 512} | Categorical |
| MLP head layer 2 | {64, 96, 128, 192, 256} | Categorical |
| MLP head layer 3 | {32, 48, 64, 96, 128} | Categorical |
| Learning rate | [1e-4, 1e-2] | Log-uniform |
| Weight decay | [1e-6, 1e-2] | Log-uniform |
| Dropout (TPE only) | [0.0, 0.5] | Uniform |

---

## Practitioner Recommendations

| Scenario | Recommended Algorithm | Reason |
|----------|----------------------|--------|
| Regression (general) | Random Search or PSO | Fast, competitive; Random wins 3/4 ADME tasks |
| Classification / toxicity | SA or ABC | Better handles class imbalance; wins on both tox tasks |
| Complex metabolic endpoints | TPE (Optuna) | Best sample efficiency on Clearance_Hepatocyte |
| Quick baseline | Morgan-FP + MLP | Simple, interpretable, no GPU needed |
| Limited compute budget | Random Search | Zero optimizer overhead, competitive with 50 trials |

---

## Documentation

Full project documentation: [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) (~40 pages)

Covers: methodology, model architecture, datasets, HPO algorithms, foundation models, results analysis, reproducibility, and API reference.

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{manu2025,
  title={Framework for Benchmarking and Optimization of Small Molecule Foundation Models for ADMET},
  author={Martin and Mila and Adrian and Viktorija and Ilinka},
  year={2025}
}
```

---

## License

MIT License

## Acknowledgments

- [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) -- Datasets and benchmarks
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) -- GNN framework
- [NiaPy](https://github.com/NiaOrg/NiaPy) -- Metaheuristic algorithms
- [Optuna](https://optuna.org/) -- TPE optimization
- [Hugging Face Transformers](https://huggingface.co/) -- ChemBERTa
- [RDKit](https://www.rdkit.org/) -- Molecular featurization

---

*Last updated: 2026-03-23*
