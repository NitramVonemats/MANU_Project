# Reproducibility Checklist

## MANU Project: Systematic HPO Benchmark for Molecular Property Prediction with GNNs

This document provides all information needed to reproduce the experiments in this paper.

---

## 1. Random Seeds

| Component | Seed Value | Purpose |
|-----------|------------|---------|
| Data splits | 42 | TDC scaffold split |
| Train/Val split | 42 | Validation set creation |
| Model initialization | 42 | PyTorch random seed |
| NumPy operations | 42 | Random operations |
| Multi-seed validation | 42, 43, 44 | Robustness check |

**Note:** All main experiments use seed=42. Multi-seed validation (Phase 4) uses seeds 42, 43, 44.

---

## 2. Data Splits

- **Split Method:** Scaffold split (Bemis-Murcko scaffolds)
- **Split Source:** TDC's built-in `get_split(method='scaffold')`
- **Train/Val/Test Proportions:**
  - TDC provides train (80%) and test (20%)
  - We further split train into train (90%) and val (10%)
  - Final proportions: ~72% train, ~8% val, ~20% test

### Dataset Sizes (After SMILES Filtering)

| Dataset | Total | Train | Val | Test |
|---------|-------|-------|-----|------|
| Caco2_Wang | 819 | 574 | 63 | 182 |
| Half_Life_Obach | 601 | 420 | 46 | 135 |
| Clearance_Hepatocyte_AZ | 1,092 | 765 | 84 | 243 |
| Clearance_Microsome_AZ | 992 | 694 | 77 | 221 |
| Tox21 (NR-AR) | 6,533 | 4,572 | 508 | 1,453 |
| hERG | 590 | 413 | 45 | 132 |
| **Total** | **10,627** | 7,438 | 823 | 2,366 |

**Data Filtering:** Compounds with invalid SMILES (failing RDKit parsing) are excluded.

---

## 3. Hardware Configuration

### Experiments Run On:
- **CPU:** Intel Core i7 (or similar)
- **RAM:** 16+ GB recommended
- **GPU:** Not required (CPU-only experiments)
- **OS:** Windows 10/11 or Linux
- **Python:** 3.9+

### Timing Notes:
- Single model training: ~1-5 minutes (CPU)
- Full HPO run (10 trials): ~30-60 minutes per dataset
- Complete benchmark: ~6-12 hours (CPU)

---

## 4. Software Versions

See `requirements.txt` for exact versions. Key dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| torch-geometric | >=2.4.0 | Graph neural networks |
| rdkit | >=2023.03 | Molecular processing |
| PyTDC | >=0.4.0 | Dataset access |
| niapy | >=2.0.0 | Optimization algorithms |
| scikit-learn | >=1.3.0 | Metrics, preprocessing |

---

## 5. Hyperparameter Search Space

```python
SEARCH_SPACE = {
    'hidden_dim': (64, 384),      # GNN hidden dimension
    'num_layers': (2, 8),         # Number of GNN layers
    'head_dims': [(256, 128, 64), (128, 64, 32), ...],  # MLP head
    'lr': (1e-4, 1e-2),           # Learning rate (log scale)
    'weight_decay': (1e-6, 1e-2), # L2 regularization (log scale)
    'batch_train': [16, 32, 64],  # Training batch size
}
```

### HPO Algorithms Tested:
1. PSO (Particle Swarm Optimization)
2. ABC (Artificial Bee Colony)
3. GA (Genetic Algorithm)
4. SA (Simulated Annealing)
5. HC (Hill Climbing)
6. Random Search (baseline)

**Trials per algorithm:** 10

---

## 6. Training Protocol

```python
# Default configuration
config = {
    'epochs': 100,
    'patience': 20,           # Early stopping
    'val_fraction': 0.1,      # From train set
    'batch_train': 32,
    'batch_eval': 64,
    'max_grad_norm': 1.0,     # Gradient clipping
}

# Optimizer
optimizer = Adam(lr=0.001, weight_decay=0.0)

# Scheduler
scheduler = ReduceLROnPlateau(patience=10, factor=0.5)
```

---

## 7. Evaluation Metrics

### Regression Tasks:
- **RMSE** (Root Mean Squared Error) - primary metric
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

### Classification Tasks:
- **F1 Score** - primary metric
- **AUC-ROC** (Area Under ROC Curve)
- **Accuracy**

---

## 8. Known Limitations

1. **Single-seed main evaluation:** Most experiments use seed=42 only. Multi-seed validation was performed with 3 seeds (42, 43, 44) for robustness assessment.

2. **CPU-only training:** All experiments were run on CPU. GPU training is supported but not used in reported results.

3. **Limited HPO budget:** 10 trials per algorithm due to computational constraints.

4. **No cross-validation:** We use fixed scaffold splits rather than k-fold CV.

5. **Foundation models not optimized:** ChemBERTa and Morgan fingerprints use default configurations, not HPO-tuned.

---

## 9. Reproducing Results

### Quick Start:

```bash
# 1. Clone repository
git clone https://github.com/NitramVonemats/MANU_Project.git
cd MANU_Project

# 2. Create environment
conda env create -f environment.yml
conda activate manu

# 3. Run benchmark
python scripts/run_hpo.py --dataset Caco2_Wang --algorithm pso

# 4. Generate figures
python scripts/generate_real_evaluation_figures.py

# 5. Verify dataset counts
python scripts/verify_class_balance.py --all
```

### Full Benchmark:

```bash
# Run all HPO experiments (takes several hours)
for ds in Caco2_Wang Half_Life_Obach Clearance_Hepatocyte_AZ Clearance_Microsome_AZ tox21 herg; do
    for algo in pso abc ga sa hc random; do
        python scripts/run_hpo.py --dataset $ds --algorithm $algo
    done
done
```

---

## 10. File Structure

```
MANU_Project/
├── optimized_gnn.py          # Main model and training code
├── scripts/
│   ├── run_hpo.py            # HPO runner
│   ├── verify_class_balance.py
│   ├── generate_real_evaluation_figures.py
│   ├── analyze_hpo_variance.py
│   └── run_multiseed_validation.py
├── optimization/
│   ├── problem.py            # HPO problem definition
│   ├── space.py              # Hyperparameter space
│   └── algorithms/           # Optimization algorithms
├── results/
│   ├── hpo/                  # HPO results (JSON)
│   └── predictions/          # Model predictions (NPZ)
├── figures/paper/            # Publication figures
├── docs/                     # LaTeX sections
├── requirements.txt
├── environment.yml
└── REPRODUCIBILITY_CHECKLIST.md
```

---

## 11. Contact

For questions about reproducibility, please open an issue on GitHub or contact the authors.

---

*Last updated: 2026-01-20*
