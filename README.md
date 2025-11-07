# GNN Benchmark for ADME Prediction

## Project Structure

```
├── src/                          # Source code
│   ├── optimized_gnn.py         # Optimized model
│   └── molecular_gnn.py         # Full experimental version
│
├── scripts/                      # Analysis scripts
│   ├── analyze_gnn_results.py
│   ├── visualize_results.py
│   └── export_best_models.py
│
├── results/                      # Experimental results (CSV)
│   ├── SUMMARY_BEST_MODELS.csv
│   ├── MODEL_STATISTICS.csv
│   ├── HYPERPARAMETER_ANALYSIS.csv
│   └── best_models_*.csv
│
├── figures/                      # Visualizations (PNG)
│   └── [9 figures]
│
│
└── GNN_test/                     # Original experiments
    └── [39 CSV files + code]
```

## Datasets

- Half_Life_Obach (667 compounds)
- Clearance_Hepatocyte_AZ (1,213 compounds)
- Clearance_Microsome_AZ (1,102 compounds)

## Best Results

| Dataset | Test RMSE | Test R² |
|---------|-----------|---------|
| Half_Life_Obach | 0.8388 | 0.2765 |
| Clearance_Hepatocyte_AZ | 1.1921 | 0.0868 |
| Clearance_Microsome_AZ | 1.0184 | 0.3208 |
|Caco2_Wang|1.768614e+01|0.335670|

Configuration: Graph model, 5 layers, 128 hidden channels

## Key Findings

1. Graph model outperforms other architectures
2. Edge features degrade performance (3.5x worse)
3. Dropout not needed for small datasets
4. Optimal: 5 layers, 128 hidden channels, LR=0.001

## Installation

```bash
# 1. (Optional but recommended) create a fresh conda env
conda create -n tdc_env python=3.10
conda activate tdc_env

# 2. Install Python requirements (PyTorch, torch_geometric, RDKit, PyTDC, Niapy, etc.)
pip install -r requirements.txt

# 3. (If you see a RequestsDependencyWarning) install a charset detector
pip install charset-normalizer
```

The project expects CUDA if available, but automatically falls back to CPU.

## Running the Optimized GNN

The main training entry point lives in `GNN_test/optimized_gnn.py`.

```bash
# Train on the default benchmark suite (Half_Life_Obach, Clearance_Hepatocyte_AZ, Clearance_Microsome_AZ)
python GNN_test/optimized_gnn.py

# Train a single dataset with custom settings
python -c "from GNN_test.optimized_gnn import train_model; train_model('Half_Life_Obach', epochs=80, patience=15, device='auto')"
```

### Configuration

`train_model` accepts an `OptimizedGNNConfig` (see file for full list). Example:

```python
from GNN_test.optimized_gnn import OptimizedGNNConfig, train_model

cfg = OptimizedGNNConfig(
    hidden_dim=192,
    num_layers=4,
    head_dims=(256, 128, 64),
    lr=5e-4,
    weight_decay=1e-4,
    batch_train=64,
    batch_eval=128,
    val_fraction=0.15,
)

train_model('Half_Life_Obach', config=cfg, epochs=120, patience=20, device='auto')
```

## Hyperparameter Optimisation (HPO)

The `hpo/` folder contains Niapy-based search drivers. Each script shares the same CLI:

```bash
# Random search (5 trials) on Caco2_Wang
python -m hpo.random_search --dataset Caco2_Wang --trials 5 --epochs 40 --patience 8 --device auto

# Genetic algorithm (population 24, 60 trials)
python -m hpo.ga --dataset Caco2_Wang --pop 24 --trials 60 --epochs 60 --patience 12

# Particle swarm optimisation
python -m hpo.pso --dataset Half_Life_Obach --pop 20 --trials 50

# Artificial bee colony, Hill Climb, Simulated Annealing are analogous
python -m hpo.abc --dataset Clearance_Hepatocyte_AZ --trials 40
python -m hpo.hc  --dataset Clearance_Microsome_AZ --trials 40
python -m hpo.sa  --dataset Half_Life_Obach --trials 40
```

Outputs land in `runs/<dataset>/hpo_<dataset>_<algo>.json` and record the search space, best parameters, full training metrics, and history.

**Tip:** The scripts reuse a cached dataset split per run. Delete the cache or change the random seed if you need a fresh split.

## Analysis

```bash
# Generate aggregate reports / plots from CSV logs
python scripts/analyze_gnn_results.py
python scripts/summary_report.py
python scripts/visualize_results.py
```

## Analysis Results

See `results/` directory for detailed CSV files and `figures/` for visualizations.
