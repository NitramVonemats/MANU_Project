# Task-Dependent Performance of GNNs and Pretrained Models in ADMET Prediction

A reproducible benchmark of graph neural networks (GNNs), frozen pretrained molecular
representations, and fingerprint baselines for ADMET property prediction under
**scaffold-split** evaluation, together with a systematic comparison of **seven
hyperparameter-optimization (HPO) algorithms**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3+-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Authors:** Martin Stamenov, Mila Gjurovska, Viktorija Vodilovska, Ilinka Ivanoska
> **Manuscript:** [`paper_final/main.tex`](paper_final/main.tex) · compiled [`paper_final/main.pdf`](paper_final/main.pdf)
> **Result provenance:** [`PAPER_RESULTS_NAVIGATION.md`](PAPER_RESULTS_NAVIGATION.md)

---

## Overview

This project studies *when molecular structure alone is sufficient* for reliable ADMET
prediction and *how model and optimizer choice should be tailored to the endpoint*. Under
a unified scaffold-split protocol, it compares five model families — task-specific GNNs,
the frozen pretrained transformer **ChemBERTa**, the frozen contrastive GNN encoder
**MolCLR**, **Morgan (ECFP)** fingerprints, and **MolE-style** learned fingerprints — on
six Therapeutics Data Commons (TDC) ADMET benchmarks (four ADME regression tasks, two
toxicity classification tasks). It also benchmarks seven HPO strategies (Random Search,
PSO, ABC, GA, SA, Hill Climbing, and Optuna's TPE) at a fixed 50-trial budget.

> **Note on pretrained baselines.** ChemBERTa and MolCLR are evaluated as **frozen
> feature extractors** (embeddings extracted once, with a trainable MLP head); they are
> *not* fully fine-tuned. Results should be read in that setting.

### Benchmark at a glance

| Item | Value |
|---|---|
| Datasets | 6 (4 ADME regression + 2 toxicity classification) |
| Molecules (TDC catalogued / used after graph conversion) | 11,805 / 10,627 |
| Model families | 5 (GNN, ChemBERTa, MolCLR, Morgan-FP, MolE-FP) |
| GNN backbone | GraphConv (selected from 8 candidate architectures) |
| HPO algorithms | 7 (Random, PSO, ABC, GA, SA, HC, TPE) |
| HPO budget | 50 trials per algorithm per dataset |
| HPO training runs | 7 × 6 × 50 = **2,100** |
| Multi-seed validation | 5 seeds `[42, 123, 456, 789, 1011]` |
| Evaluation protocol | Bemis–Murcko scaffold split (≈70/8/22 train/val/test) |
| Hardware | NVIDIA RTX 3060, Intel i7-8700K, 16 GB RAM (≈45 h total) |

---

## Key findings

1. **Task-dependent learnability (three tiers).** Performance varies sharply by endpoint
   rather than by model complexity.
2. **Structure-driven endpoints.** GNNs reach practically useful accuracy and match or
   exceed the frozen pretrained encoders — hERG AUC = **0.825**, Caco-2 R² = **0.48**.
3. **Moderately structure-driven endpoints.** On Tox21 NR-AR (AUC = 0.742) and microsomal
   clearance (R² = 0.191), performance is modest and similar across families.
4. **Weakly structure-driven endpoints.** On half-life and hepatocyte clearance, all
   evaluated models yield near-zero or negative R² (GNN hepatocyte R² = −1.02), indicating
   limited signal in structure-only inputs *within this benchmark*.
5. **Random Search is a strong HPO baseline.** No metaheuristic shows a consistent
   improvement over Random Search at the 50-trial budget: bootstrap 95% confidence
   intervals on paired relative improvements span or fall below zero for all five
   metaheuristics, and Hill Climbing loses on all six datasets. (Reported via paired
   bootstrap CIs and effect sizes, not null-hypothesis significance tests.)

---

## Headline results

All numbers below are taken from the manuscript tables; see
[`PAPER_RESULTS_NAVIGATION.md`](PAPER_RESULTS_NAVIGATION.md) for the exact source file of
each value, and run `python scripts/audit_paper_final_numbers.py` to re-verify them.

### Model-family comparison (Table IV — R² for regression, AUC-ROC for classification; best per task in **bold**)

| Task | Metric | GNN-Best | ChemBERTa | Morgan-FP | MolE-FP | MolCLR |
|---|---|---|---|---|---|---|
| hERG | AUC ↑ | **0.825** | 0.770 | 0.611 | 0.672 | 0.401 |
| Caco-2 | R² ↑ | **0.481** | 0.478 | 0.200 | 0.047 | −0.189 |
| Tox21 (NR-AR) | AUC ↑ | **0.742** | 0.728 | 0.722 | 0.675 | 0.452 |
| Clearance Microsome | R² ↑ | **0.191** | 0.024 | 0.122 | 0.059 | 0.041 |
| Half-Life | R² ↑ | **0.004** | −0.594 | −0.039 | −0.329 | −0.001 |
| Clearance Hepatocyte | R² ↑ | −1.019 | 0.029 | −0.015 | **0.032** | −0.039 |

### HPO algorithm comparison (Table VIII — regression: test RMSE ↓; classification: test AUC-ROC ↑; best per row in **bold**)

| Dataset | PSO | ABC | GA | SA | HC | Random | TPE |
|---|---|---|---|---|---|---|---|
| Caco2_Wang (RMSE) | 0.0031 | 0.0029 | 0.0031 | 0.0029 | 0.0030 | **0.0027** | 0.0029 |
| Half_Life_Obach (RMSE) | 21.66 | 21.66 | 21.66 | 23.70 | 24.52 | 22.31 | **21.48** |
| Clearance_Hepatocyte_AZ (RMSE) | 70.21 | 72.04 | 71.34 | 72.04 | 72.04 | **68.22** | 80.32 |
| Clearance_Microsome_AZ (RMSE) | 42.76 | 42.29 | 42.29 | 40.94 | 41.63 | **38.75** | 40.89 |
| Tox21 (NR-AR) (AUC) | 0.692 | 0.735 | 0.735 | **0.742** | 0.652 | 0.713 | 0.722 |
| hERG (AUC) | 0.747 | **0.825** | 0.747 | 0.802 | 0.821 | 0.747 | 0.756 |

> TPE (Optuna) searches a separate space that also includes dropout; the NiaPy-based
> algorithms share the 7-dimensional space below.

### HPO vs. Random Search (Table X — paired relative improvement, n = 6 datasets)

| Algorithm | W/T/L | Mean Δ (%) | 95% CI (%) | d_z |
|---|---|---|---|---|
| PSO | 2/0/4 | −5.08 | [−10.21, +0.05] | −0.70 |
| ABC | 1/0/5 | −3.86 | [−6.90, −0.49] | −0.89 |
| GA  | 2/0/4 | −3.98 | [−9.13, +0.92] | −0.57 |
| SA  | 1/0/5 | −3.94 | [−6.01, −0.96] | −1.09 |
| HC  | 0/0/6 | −6.96 | [−9.38, −4.09] | −1.89 |

> Improvement direction is normalized so that positive favors the optimizer (RMSE:
> Random − Algo; F1: Algo − Random), expressed as a percentage of the Random-Search
> baseline. CIs are from 10,000 percentile bootstrap resamples.

### Multi-seed validation (Table XI — mean ± std, 95% CI, n = 5 seeds)

| Dataset | Task | Metric | Mean ± Std (95% CI) |
|---|---|---|---|
| Caco2_Wang | Regr. | RMSE | 0.0026 ± 0.0001 (0.0026–0.0027) |
| Half_Life_Obach | Regr. | RMSE | 20.72 ± 1.42 (19.48–21.96) |
| Clearance_Hepatocyte_AZ | Regr. | RMSE | 49.87 ± 1.15 (48.86–50.88) |
| Clearance_Microsome_AZ | Regr. | RMSE | 42.02 ± 3.36 (39.08–44.97) |
| Tox21 (NR-AR) | Class. | AUC | 0.716 ± 0.012 (0.706–0.727) |
| hERG | Class. | AUC | 0.804 ± 0.018 (0.789–0.819) |

---

## Datasets

All datasets are from the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) ADMET
benchmark, partitioned with a Bemis–Murcko scaffold split (≈70/8/22 train/val/test, with a
fixed seed-42 train/validation carve).

| Dataset | Task | TDC | Used | Metric | Note |
|---|---|---|---|---|---|
| Caco2_Wang | Permeability (regression) | 910 | 819 | RMSE / R² | strongest regression task (R² ≈ 0.48) |
| Half_Life_Obach | Half-life (regression) | 667 | 601 | RMSE / R² | weak (R² ≈ 0.00) |
| Clearance_Hepatocyte_AZ | Clearance (regression) | 1,213 | 1,092 | RMSE / R² | weak (R² < 0) |
| Clearance_Microsome_AZ | Clearance (regression) | 1,102 | 992 | RMSE / R² | weak–moderate (R² ≈ 0.19) |
| Tox21 (NR-AR) | Toxicity (classification) | 7,258 | 6,533 | AUC-ROC / F1 | imbalanced (≈4.2% positive) |
| hERG | Cardiotoxicity (classification) | 655 | 590 | AUC-ROC / F1 | ≈69% blockers; AUC ≈ 0.825 |

---

## Repository structure

```
.
├── paper_final/                  # IEEE manuscript (authoritative results)
│   ├── main.tex                  #   source
│   ├── refs.bib                  #   bibliography
│   ├── main.pdf                  #   compiled PDF
│   └── images/                   #   figures
├── src/
│   ├── core/                     # optimized_gnn.py (GNN model/train/eval), model_comparison.py
│   └── utils/
├── optimization/                 # HPO framework
│   ├── space.py                  #   7-dim GNN search space
│   ├── problem.py, runner.py     #   NiaPy problem wrapper + runner
│   ├── foundation_*.py           #   frozen pretrained / fingerprint evaluation
│   └── algorithms/               #   pso, genetic, abc, simulated_annealing, hill_climbing, random_search
├── scripts/                      # runners, figure generation, analysis, audit
│   ├── run_hpo_50_trials.py      #   main NiaPy HPO runner (50 trials)
│   ├── run_tpe_benchmark.py      #   TPE via Optuna
│   ├── run_multi_seed_validation.py
│   ├── run_complete_foundation_benchmark.py, run_chemberta_finetune.py
│   ├── regenerate_paper_figures.py     # rebuilds paper_final/images/*
│   └── audit_paper_final_numbers.py    # cross-checks manuscript numbers vs sources
├── runs/                         # raw HPO results (JSON) per dataset/algorithm
├── results/multi_seed/           # 5-seed validation outputs
├── datasets/{adme,toxicity}/     # TDC datasets (CSV)
├── figures/                      # generated tables + paper-source evidence files
├── archive/                      # earlier experiments (incl. TPE & foundation logs)
├── external/MolCLR/              # MolCLR pretrained checkpoints
├── requirements.txt, environment.yml
├── PAPER_RESULTS_NAVIGATION.md   # maps each table/figure to its source file
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/NitramVonemats/MANU_Project.git
cd MANU_Project
pip install -r requirements.txt          # or: conda env create -f environment.yml
```

Core dependencies: Python ≥ 3.8, PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.3, RDKit ≥ 2022.9,
PyTDC ≥ 0.4, NiaPy ≥ 2.0, Optuna ≥ 3.0, Transformers ≥ 4.30.

## Reproducing the experiments

```bash
# NiaPy HPO benchmark (Random/PSO/ABC/GA/SA/HC), 50 trials, all datasets
python scripts/run_hpo_50_trials.py

# TPE (Optuna) benchmark
python scripts/run_tpe_benchmark.py

# Frozen pretrained / fingerprint baselines
python scripts/run_complete_foundation_benchmark.py

# Multi-seed validation (5 seeds)
python scripts/run_multi_seed_validation.py

# Rebuild the paper figures from the result files
python scripts/regenerate_paper_figures.py

# Re-verify every manuscript number against its source (expects: TOTAL FAILURES: 0)
python scripts/audit_paper_final_numbers.py
```

## Building the paper

```bash
cd paper_final
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
# or, with a self-contained engine: tectonic main.tex
```

---

## HPO configuration

| Algorithm | Type | Framework | Configuration |
|---|---|---|---|
| Random Search | baseline | NiaPy | uniform sampling |
| PSO | swarm | NiaPy | population 16 |
| ABC | swarm | NiaPy | colony 16 |
| GA | evolutionary | NiaPy | population 16 |
| SA | annealing | NiaPy | T₀ = 1.0, exponential cooling α = 0.99 |
| HC | local search | NiaPy | greedy, single initialization |
| TPE | Bayesian | Optuna | 10 startup trials, median pruning |

Each GNN trial: Adam, batch size 32, ≤ 50 epochs with early stopping (patience 12),
gradient clipping (max-norm 1.0); MSE loss for regression, BCE-with-logits for
classification.

### Search space (7-dim, NiaPy)

| Hyperparameter | Range | Type |
|---|---|---|
| Hidden dimensions | {64, 96, 128, 192, 256, 384, 512} | categorical |
| Number of layers | {3, 4, 5, 6, 7} | categorical |
| MLP head layer 1 | {128, 192, 256, 384, 512} | categorical |
| MLP head layer 2 | {64, 96, 128, 192, 256} | categorical |
| MLP head layer 3 | {32, 48, 64, 96, 128} | categorical |
| Learning rate | [1e-4, 1e-2] | log-uniform |
| Weight decay | [1e-6, 1e-2] | log-uniform |

TPE additionally searches dropout ∈ [0.0, 0.5] over a separate (non-identical) space.

---

## Citation

If you use this benchmark, please cite the paper:

```bibtex
@article{stamenov2026admet,
  title  = {Task-Dependent Performance of GNNs and Pretrained Models in ADMET Prediction},
  author = {Stamenov, Martin and Gjurovska, Mila and Vodilovska, Viktorija and Ivanoska, Ilinka},
  year   = {2026}
  % TODO: add journal/volume/pages/DOI once published.
}
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

- [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) — datasets and benchmarks
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) — GNN framework
- [NiaPy](https://github.com/NiaOrg/NiaPy) — nature-inspired optimization algorithms
- [Optuna](https://optuna.org/) — TPE hyperparameter optimization
- [Hugging Face Transformers](https://huggingface.co/) — ChemBERTa
- [RDKit](https://www.rdkit.org/) — molecular featurization
