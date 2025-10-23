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

## Usage

```bash
# Install dependencies
pip install torch torch-geometric rdkit pytdc pandas numpy scikit-learn matplotlib seaborn

# Run optimized model
python src/optimized_gnn.py

# Analyze results
python scripts/analyze_gnn_results.py
```

## Analysis Results

See `results/` directory for detailed CSV files and `figures/` for visualizations.
