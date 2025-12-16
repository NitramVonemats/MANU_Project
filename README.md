# MANU - Molecular ADME Prediction with GNNs

Molecular property prediction using Graph Neural Networks with hyperparameter optimization.

---

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

# Run analyses
cd scripts
python run_all_analyses.py

# Run HPO (GNN)
python scripts/run_hpo.py --dataset Caco2_Wang --algorithm abc --trials 50

# Run HPO (Foundation Models)
python scripts/run_foundation_hpo.py --model morgan --dataset Caco2_Wang --algo pso --trials 30

# Run EVERYTHING (Benchmarks for GNN + Foundation Models)
# 1. Run all GNN benchmarks (4 ADME + 3 Toxicity datasets)
python scripts/run_all_hpo.py --config config_benchmark.yaml

# 2. Run all Foundation Model benchmarks (4 ADME + 3 Toxicity datasets)
python scripts/run_foundation_batch.py --config config_foundation_benchmark.yaml
```

---

## Datasets

**ADME (4 datasets, 3,892 molecules):**
- Caco2_Wang (910) - Permeability
- Half_Life_Obach (667) - Half-life
- Clearance_Hepatocyte_AZ (1,213) - Hepatic clearance
- Clearance_Microsome_AZ (1,102) - Microsomal clearance

**Toxicity (3 datasets, 9,391 molecules):**
- Tox21 (7,258) - Nuclear receptor toxicity
- hERG (655) - Cardiac toxicity
- ClinTox (1,478) - Clinical trial toxicity

**Total: 13,283 molecules**

---

## Results

Best HPO results (Test RMSE):

| Dataset | Algorithm | RMSE | R² |
|---------|-----------|------|-----|
| Caco2_Wang | ABC | 0.529 | 0.529 |
| Half_Life_Obach | SA | 0.847 | 0.192 |
| Clearance_Microsome_AZ | ABC | 1.018 | 0.321 |
| Clearance_Hepatocyte_AZ | SA | 1.192 | -0.097 |

---

## Project Structure

```
MANU_Project/
├── README.md                 # This file
├── adme_gnn/                 # Main package (models, data, training)
├── optimization/             # HPO algorithms (ABC, GA, PSO, SA, HC, Random)
├── scripts/                  # Analysis and visualization scripts
├── figures/                  # All visualizations (66 files)
│   ├── comparative/          # Unified plots
│   ├── hpo/                  # HPO results
│   ├── ablation_studies/     # Hyperparameter analysis
│   └── per_dataset_analysis/ # Per-dataset details
├── runs/                     # HPO JSON results
├── reports/                  # Benchmark reports
├── docs/                     # Documentation
│   ├── GUIDES.md            # Usage guides
│   └── PROJECT_STRUCTURE.md # Detailed structure
└── archive/summaries/        # Old logs and summaries
```

---

## Documentation

- **[GUIDES.md](docs/GUIDES.md)** - How to run analyses, HPO, visualizations
- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Full project structure

---

## What's Done

✅ 7 datasets prepared (13,283 molecules)
✅ Molecular features computed (19 descriptors)
✅ 66 visualizations created
✅ 24 HPO runs completed (4 datasets × 6 algorithms)
✅ Benchmark reports generated
✅ GNN training pipeline implemented

---

## Requirements

```
python >= 3.8
torch >= 2.0.0
torch-geometric >= 2.3.0
rdkit >= 2022.9.0
PyTDC >= 1.0.0
```

Install: `pip install -r requirements.txt`
