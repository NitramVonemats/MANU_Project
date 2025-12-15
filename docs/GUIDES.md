# Usage Guides - MANU Project

Овој документ содржи практични упатства за користење на проектот.

---

## 📊 Running Dataset Analyses

### Сите анализи одеднаш:
```bash
cd scripts
python run_all_analyses.py
```

### Индивидуални анализи:
```bash
# Tanimoto similarity анализа
python scripts/tanimoto_similarity_analysis.py

# Label distribution анализа
python scripts/label_distribution_analysis.py

# Feature-label correlation анализа
python scripts/feature_label_correlation_analysis.py

# Publication figures
python scripts/create_publication_figures.py
```

**Излез:** Фигури во `figures/` директориум (comparative, per_dataset_analysis, hpo, ablation_studies)

Детали: Види `scripts/README_ANALYSES.md`

---

## 🔬 Running Hyperparameter Optimization

### Single dataset HPO:
```bash
python scripts/run_hpo.py --dataset Caco2_Wang --algorithm abc --trials 50
```

### Сите datasets и algorithms:
```bash
python run_all_hpo.py
```

**Параметри:**
- `--dataset`: Caco2_Wang, Half_Life_Obach, Clearance_Hepatocyte_AZ, Clearance_Microsome_AZ
- `--algorithm`: abc, ga, pso, sa, hc, random
- `--trials`: Број на trials (default: 50)

**Излез:**
- JSON резултати во `runs/{dataset}/hpo_{dataset}_{algorithm}.json`
- Најдобри хиперпараметри во консола

---

## 📈 Creating Visualizations

### HPO visualizations:
```bash
python scripts/create_hpo_visualizations.py
```

**Излез:** `figures/hpo/` (algorithm performance, best hyperparameters, winner analysis)

### Ablation studies:
```bash
python scripts/create_comprehensive_ablation_studies.py
```

**Излез:** `figures/ablation_studies/` (hyperparameter comparison, space exploration)

### Unified comparative plots:
```bash
python scripts/create_unified_visualizations.py
```

**Излез:** `figures/comparative/` (dataset overview, label distributions, feature importance)

---

## 🚀 Running Benchmarks

### Benchmark report:
```bash
python scripts/benchmark_report.py
```

**Излез:** `reports/benchmark_{timestamp}/` (summary, detailed comparison, plots)

---

## 📦 Datasets

### Locations:
- **TDC cache:** `~/.tdc/` (автоматски се симнуваат)
- **Processed:** `datasets/` (ако има custom processing)

### Available datasets:
**ADME (regression):**
- Caco2_Wang (910 molecules)
- Half_Life_Obach (667 molecules)
- Clearance_Hepatocyte_AZ (1,213 molecules)
- Clearance_Microsome_AZ (1,102 molecules)

**Toxicity (classification):**
- Tox21 (7,258 molecules)
- hERG (655 molecules)
- ClinTox (1,478 molecules)

---

## 🛠️ Model Training

### Using optimized_gnn.py:
```bash
python optimized_gnn.py
```

### Using adme_gnn package:
```python
from adme_gnn.models import GNN
from adme_gnn.training import Trainer
from adme_gnn.data import load_dataset

# Load dataset
data = load_dataset('Caco2_Wang')

# Create model
model = GNN(
    model_name='GCN',
    num_features=data.num_features,
    hidden_channels=128,
    num_layers=5
)

# Train
trainer = Trainer(model, data)
results = trainer.train(epochs=100)
```

---

## 📊 Results Location

```
figures/
├── comparative/          # Unified plots (all datasets)
├── hpo/                  # HPO results
├── ablation_studies/     # Hyperparameter analysis
└── per_dataset_analysis/ # Per-dataset detailed analysis

reports/
└── benchmark_{timestamp}/ # Benchmark reports

runs/
└── {dataset}/            # HPO run JSONs
```

---

## 🔍 Troubleshooting

### CUDA out of memory:
```python
# Намали batch size
--batch_size 32

# Намали hidden dimensions
--hidden_channels 64
```

### ModuleNotFoundError:
```bash
pip install -r requirements.txt
```

### Unicode errors (Windows):
```bash
chcp 65001
python -X utf8 script.py
```

---

## 📝 Next Steps

1. ✅ Datasets подготвени (7 datasets, 13,283 molecules)
2. ✅ Анализи завршени (similarity, labels, correlations)
3. ✅ HPO завршен (6 algorithms × 4 datasets)
4. ✅ Визуелизации креирани (65 фигури)
5. ⏳ Model comparison (Phase 2 - optional)
6. 📄 Paper writing (користи фигури од `figures/`)

---

**За повеќе детали:** Види `archive/summaries/` за старите summary фајлови.
