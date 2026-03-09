# MANU Methodology

## Experimental Design

### Overview
This project systematically evaluates **seven hyperparameter optimization (HPO) algorithms** across **six ADMET datasets** using a **Graph Neural Network (GNN)** backbone with consistent evaluation protocols.

### Key Design Choices

#### 1. Evaluation Protocol: Scaffold-Based Splitting
- **Why:** Realistic drug discovery scenario (models must generalize to novel chemical series)
- **Method:** Bemis–Murcko scaffolds separate compounds with different core structures
- **Split ratio:** 80% train / 10% validation / 10% test
- **Reproducibility:** Fixed random seed (42) for all splits

**Impact:** Scaffold splitting substantially increases task difficulty compared to random splitting

#### 2. Model Architecture: Graph Convolutional Network (GCN)
```
Input (Molecular Graph)
    ↓
Graph Representation:
  - Atoms (nodes): 8 features each
    * Atomic number
    * Degree
    * Formal charge
    * Hybridization
    * Aromaticity
    * Ring membership
    * Hydrogen count
    * Atomic mass
  - Bonds (edges): type, aromatic flag
    ↓
Graph Convolutional Layers (variable depth)
  - BatchNorm + ReLU activation
  - Concatenated global mean & max pooling
    ↓
MLP Prediction Head
  - 3 layers with batch normalization
  - Output: 1 (regression) or 2 (classification)
```

#### 3. Training Configuration

| Parameter | Regression | Classification |
|-----------|-----------|--------|
| **Max Epochs** | 100 | 50 |
| **Early Stopping** | 12 epochs patience | 12 epochs patience |
| **Optimizer** | Adam (default params) | Adam (default params) |
| **Batch Size** | 32 | 32 |
| **Loss Function** | MSE | BCE with logits |
| **Class Weighting** | N/A | Yes (for Tox21 imbalance) |
| **Gradient Clipping** | max_norm=1.0 | max_norm=1.0 |

#### 4. Target Transformations

**Regression Tasks (ADME):**
- Non-Caco2: Apply log transformation with ε=10⁻³ offset
- All: Z-score normalization using train-set statistics
- Normalization applied to train+val, reused for test (prevents leakage)

**Classification Tasks (Toxicity):**
- Binary encoding (0/1)
- No transformation
- Class weight = (negative samples) / (positive samples) for Tox21

### Hyperparameter Optimization Setup

#### Search Space

| Hyperparameter | Search Space | Type |
|---|---|---|
| **Hidden Dimensions** | {64, 96, 128, 192, 256, 384, 512} | Categorical |
| **Number of Layers** | {3, 4, 5, 6, 7} | Categorical |
| **Learning Rate** | [10⁻⁴, 10⁻²] | Log-uniform |
| **Weight Decay** | [10⁻⁶, 10⁻²] | Log-uniform |
| **Dropout Rate** | [0.0, 0.5] | Uniform |
| **MLP Head Dims** | 3 categorical presets | Preset |

**Total search space size:** ~250,000 possible configurations

#### Optimization Budget
- **50 trials** per algorithm-dataset combination
- **6 datasets** × **7 algorithms** = **42 combinations**
- **Total runs:** 42 × 50 = **2,100+ training runs**
- **Hardware:** NVIDIA RTX 3060, Intel i7-8700K, 16GB RAM
- **Total computation time:** ~45 hours

### Algorithm Selection & Configuration

#### 1. **Random Search (NiaPy)**
- Baseline: uniform sampling from search space
- Population: N/A (per-trial sampling)
- Trials: 50

#### 2. **Particle Swarm Optimization (PSO) (NiaPy)**
- Mimics swarm behavior (birds, fish)
- Balance exploration vs exploitation
- **Config:** Population=16, C1=2.0, C2=2.0, w=0.7
- **Trials:** 50 (3+ rounds of 16 particles)

#### 3. **Simulated Annealing (SA) (NiaPy)**
- Escapes local optima via temperature schedule
- **Config:** T₀=50, cooling α=0.99
- **Trials:** 50

#### 4. **Genetic Algorithm (GA) (NiaPy)**
- Evolution via selection, crossover, mutation
- **Config:** Population=16, mutation_rate=0.1, crossover_rate=0.8
- **Trials:** 50 (~3+ generations)

#### 5. **Artificial Bee Colony (ABC) (NiaPy)**
- Mimics honeybee foraging
- **Config:** Colony size=16, limit=50 (scouts trigger limit)
- **Trials:** 50

#### 6. **Hill Climbing (HC) (NiaPy)**
- Greedy local search baseline
- Single random init, only accept improvements
- **Trials:** 50

#### 7. **Tree-structured Parzen Estimator (TPE) (Optuna)**
- Bayesian optimization: models P(good) and P(bad) hyperparameters
- **Config:** 10 startup random trials + 40 TPE trials
- Median pruning of underperforming trials
- **Trials:** 50

### Model Selection Strategy

For each algorithm-dataset pair:
1. Run 50 trials
2. **Select best hyperparameters** based on:
   - **Regression:** Minimum validation RMSE
   - **Classification:** Maximum validation AUC
3. Retrain with selected hyperparameters on full train+val
4. Evaluate once on held-out test set
5. Report test metrics

**Key:** No test-set touching during HPO (prevents data leakage)

### Multi-Seed Validation

To assess robustness:
1. Take best hyperparameters per algorithm-dataset
2. Retrain 5 times with different random seeds: {42, 43, 44, 45, 46}
3. Report mean ± std deviation of test metrics

**Purpose:** Quantify variance due to random initialization

### Foundation Model Baselines

Compared against:
1. **Morgan Fingerprints (ECFP4)** + MLP predictor
2. **ChemBERTa** (frozen encoder + trainable MLP head)
3. **ChemBERTa-FT** (limited fine-tuning, small learning rate sweep)
4. **MolCLR** (contrastive graph embeddings)
5. **MolE** (multi-task pretrained GNN)

**Note:** Foundation models used as frozen feature extractors (unfair advantage to GNNs which got full HPO)

### Evaluation Metrics

#### Regression Tasks
- **RMSE:** Root Mean Squared Error (main metric for HPO)
- **MAE:** Mean Absolute Error
- **R²:** Coefficient of determination (interpretation: explained variance)

#### Classification Tasks
- **AUC-ROC:** Area Under ROC Curve (main metric for HPO)
- **Accuracy:** Proportion of correct predictions
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** To assess imbalance effects

### Data Quality & Reproducibility

#### Reproducibility Measures
✓ Fixed random seeds for all stochastic operations
✓ Deterministic data splitting
✓ Hardware specs documented
✓ All hyperparameters saved in JSON
✓ Training histories recorded epoch-by-epoch

#### Result Validation
✓ Multi-seed stability checks
✓ Realistic training curves (non-monotonic)
✓ Test metrics worse than validation (expected overfitting)
✓ Statistical significance testing (Wilcoxon signed-rank)

### Software & Dependencies

```
Python ≥ 3.8
PyTorch ≥ 2.0
PyTorch Geometric ≥ 2.3
RDKit ≥ 2022.9
PyTDC ≥ 1.0
NiaPy ≥ 2.0 (for metaheuristic algorithms)
Optuna ≥ 3.0 (for TPE)
Transformers ≥ 4.30 (for ChemBERTa)
```

### Statistical Analysis

#### Significance Testing
- **Test:** Wilcoxon signed-rank test
- **Null hypothesis:** No difference between algorithm pairs
- **p-value threshold:** 0.05
- **Effect size:** Rank-biserial correlation (r)

#### Confidence Intervals
- Computed from multi-seed results
- 95% CI for each metric

## Validation Approach

### 1. Internal Consistency
- Verify test metrics from saved results match reported values
- Check training histories are complete (not truncated)
- Ensure epochs match early stopping pattern

### 2. External Consistency
- Compare with TDC leaderboard results
- Validate against published benchmarks
- Cross-check dataset statistics (n_samples, class balance)

### 3. Numerical Plausibility
- R² should be in [-∞, 1]
- RMSE should be positive
- AUC should be in [0, 1]
- F1 should be in [0, 1]

### 4. Statistical Soundness
- Test RMSE ≥ validation RMSE (expected generalization gap)
- Multi-seed variance should be non-zero
- No single algorithm uniformly best (expected)

## Limitations & Caveats

### Known Limitations
1. **Fixed GNN architecture:** Only tested GraphConv (not GAT, GIN, others)
2. **Limited HPO budget:** 50 trials may not fully explore 250k config space
3. **Frozen foundation models:** Not fine-tuned with equal budget
4. **Single seed for main:** Multi-seed only for best configs
5. **Small datasets:** Some datasets have <1000 compounds (noisy)

### Acknowledged Challenges
- **Clearance tasks:** Very difficult (R² < 0) - may need additional data modalities
- **Tox21 imbalance:** 3.5% positive class - hard to predict minority
- **Half-Life:** Multiple unknown factors beyond structure (PK complexity)

## Reproducibility

To reproduce results:
```bash
# Install dependencies
pip install -r requirements.txt

# Run HPO for a single dataset
python scripts/run_hpo_benchmark.py \
  --dataset Caco2_Wang \
  --n_trials 50 \
  --seed 42

# Run all datasets with all algorithms
python scripts/run_hpo_benchmark.py --all

# Multi-seed validation
python scripts/run_multiseed_validation.py --dataset Caco2_Wang --n_seeds 5
```

---

**Last Updated:** 2026-03-09
