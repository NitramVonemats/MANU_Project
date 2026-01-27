# Experimental Setup & Fairness - For Methods Section

**Date:** 2026-01-19
**Purpose:** Answer critical questions for Methods/Appendix sections

---

## 🎯 Critical Questions & Answers

### **1. Splits - Дали се исти за сите модели?**

#### **Answer: ДА - Completely consistent splits ✅**

**Evidence:**
- All models (GNN + Foundation) use **seed=42** for reproducibility
- TDC built-in splits for consistency
- 80% train / 10% val / 10% test (standard split)

**Code Reference:** `optimized_gnn.py` lines 314-342
```python
# prepare_dataset() function
random_state = 42  # Fixed seed
dataset.get_split(method='random', seed=random_state, frac=[0.8, 0.1, 0.1])
```

**Foundation Models:**
- `scripts/analyses/benchmark_foundation_models.py` uses **SAME splits**
- Loads datasets with `seed=42`
- Same train/val/test split ratios

---

### **2. Seed-ови и Variance - Има ли mean ± std?**

#### **Answer: Partial - Seed fixed, но нема multi-run variance ⚠️**

**Current Status:**
- ✅ **Fixed seed=42** for reproducibility
- ❌ **NO multi-run variance** (each HPO run is single execution)
- ⚠️ **HPO trials=10** give variance WITHIN hyperparameter search, not across random seeds

**What We Have:**
- Variance across **hyperparameters** (10 trials per algorithm)
- Variance across **algorithms** (6 algorithms tested)

**What We DON'T Have:**
- Variance across **random seeds** (no seed=[42, 43, 44, ...])
- Mean ± Std across multiple runs

**Recommendation for Paper:**
```
"All experiments use fixed random seed (seed=42) for reproducibility.
Variance is reported across hyperparameter trials (N=10) and algorithms (N=6).
Future work: Multi-seed evaluation for statistical significance."
```

**Note:** Single-seed is ACCEPTABLE for MSc thesis, but for top-tier:
- Run with 3-5 different seeds
- Report mean ± std
- Statistical significance tests (t-test, Wilcoxon)

---

### **3. HPO Leakage - Дали HPO "видел" тест сет?**

#### **Answer: NO leakage - Test set completely held out ✅**

**Evidence:**

**HPO Process:**
1. **Train split**: Model training (80% of data)
2. **Val split**: HPO optimization metric (10% of data) ← HPO uses THIS
3. **Test split**: NEVER seen during HPO (10% of data)

**Code Reference:** `optimization/problem.py` lines 64-73
```python
def _evaluate(self, x):
    # Train model with hyperparameters x
    result = train_model(...)

    # HPO uses VAL metrics ONLY
    if self.is_classification:
        f1 = result["val_metrics"].get("f1", 0.0)  # ← VAL, not TEST!
        return -float(f1)
    else:
        return float(result["val_metrics"]["rmse"])  # ← VAL, not TEST!
```

**Final Evaluation:**
After HPO finds best hyperparameters on VAL set:
- Model trained ONLY on train (80%) ← Best checkpoint from early stopping
- Evaluate on test (10%) ← **This is what we report!**
- ❌ **NO retraining on train+val** (keeps training budget consistent across all models)

**Conclusion:** **NO DATA LEAKAGE** ✅ + **FAIR TRAINING BUDGET** (all models use 80%)

---

### **4. Macro/Micro за Tox21 - Multi-task metrics?**

#### **Answer: Currently single-task (NR-AR), should note this ⚠️**

**Current Implementation:**
- Tox21 has **12 tasks** (nuclear receptor assays)
- We use **1 task: NR-AR** (Nuclear Receptor - Androgen Receptor)

**Code Reference:** `optimized_gnn.py` line 327
```python
tox_labels = {
    'tox21': 'NR-AR',  # Single task selected
    'herg': None,       # Single task (binary)
    'clintox': 'CT_TOX' # Single task
}
```

**Metrics:**
- **F1**: Macro-averaged (binary classification)
- **AUC-ROC**: Single-task
- **NOT**: Micro-averaged across 12 tasks

**For Paper - Must State:**
```
"For Tox21, we evaluate on the NR-AR assay (nuclear receptor - androgen receptor)
as a representative single-task classification problem. Multi-task learning across
all 12 Tox21 assays is left for future work."
```

**Improvement Needed (Future):**
- Evaluate on all 12 tasks
- Report macro/micro-averaged metrics
- Multi-task learning architecture

---

### **5. Foundation Models - Фер evaluation?**

#### **Answer: YES, fair but with important notes ✅⚠️**

**What's Fair:**
✅ **Same splits**: All models use seed=42, same train/val/test
✅ **Same datasets**: Identical SMILES and labels
✅ **Same metrics**: AUC-ROC, F1, RMSE, R²
✅ **Same preprocessing**: SMILES normalization

**What's Different (MUST MENTION IN PAPER):**

#### **ChemBERTa:**
- **Pretrained**: Yes (77M molecules from ZINC)
- **Fine-tuning**: ❌ NO - Used as **feature extractor only**
- **Downstream**: Simple MLP (2 layers) on frozen embeddings
- **Reason**: Computational constraints (CPU-only, no GPU fine-tuning)

#### **Morgan Fingerprints:**
- **Pretrained**: No (classical baseline)
- **Features**: ECFP4 (radius=2, 2048 bits)
- **Downstream**: Random Forest classifier/regressor

#### **GNN (Ours):**
- **Pretrained**: No (trained from scratch)
- **Architecture**: GAT/GCN with 19 node features
- **HPO**: Full hyperparameter optimization (6 algorithms × 10 trials)

**Critical Disclaimers for Paper:**

```
"Foundation models (ChemBERTa, Morgan-FP) are evaluated under the following conditions:

1. ChemBERTa: Used as frozen feature extractor with simple MLP head.
   Full fine-tuning was not performed due to computational constraints.

2. Morgan Fingerprints: Classical baseline with default hyperparameters (radius=2).

3. GNN: Trained from scratch with systematic HPO (6 algorithms, 10 trials each).

Therefore, our comparison demonstrates GNN superiority under our specific
experimental setup (limited computational resources, systematic HPO for GNN only).
A fully fair comparison would require:
- Fine-tuning ChemBERTa with similar HPO budget
- HPO for Morgan-FP downstream models
- GPU-accelerated training for all methods"
```

**Why This is Still Valid:**

- **Real-world scenario**: Many researchers have limited resources
- **Fair within constraints**: All models evaluated on same data/splits
- **Transparent**: We openly state limitations
- **Novel contribution**: Systematic HPO comparison (main contribution)

---

## 📝 For Methods Section

### **Experimental Setup (Template)**

```latex
\subsection{Experimental Setup}

\textbf{Data Splits:} All datasets were split into training (80\%), validation (10\%),
and test (10\%) sets using a fixed random seed (seed=42) for reproducibility, following
standard protocols from the Therapeutics Data Commons (TDC)~\cite{tdc2021}. The same
splits were used consistently across all models (GNN, ChemBERTa, Morgan Fingerprints)
to ensure fair comparison.

\textbf{Training Protocol:} All models were trained exclusively on the training set
(80\% of data). The validation set (10\%) was used for model selection, early stopping,
and hyperparameter optimization metric evaluation. The test set (10\%) remained completely
held out during all training and optimization procedures to prevent data leakage.
Importantly, no models were retrained on combined train+validation sets after
hyperparameter selection, ensuring consistent training budget (80\%) across all
approaches.

\textbf{Hyperparameter Optimization:} For GNN models, we performed systematic HPO
using six metaheuristic algorithms: Random Search (baseline), Particle Swarm Optimization
(PSO), Artificial Bee Colony (ABC), Genetic Algorithm (GA), Simulated Annealing (SA),
and Hill Climbing (HC). Each algorithm was run for 10 trials (60 total evaluations per
dataset). Validation F1 score (classification) or RMSE (regression) guided the optimization
process. Early stopping with patience=10 epochs was employed to prevent overfitting.

\textbf{Foundation Model Evaluation:} ChemBERTa~\cite{chemberta2020} was used as a
frozen feature extractor (no fine-tuning) due to computational constraints, with a
2-layer MLP downstream predictor. Morgan Fingerprints (ECFP4, radius=2, 2048 bits)
served as classical baseline with Random Forest downstream models. Therefore, our
comparison evaluates GNN models with systematic HPO against pretrained foundation
models in feature-extraction mode without task-specific fine-tuning.

\textbf{Evaluation Metrics:} For regression tasks (ADME properties), we report Root
Mean Squared Error (RMSE), Mean Absolute Error (MAE), and coefficient of determination
(R²). For classification tasks (toxicity prediction), we report F1 score, Area Under
ROC Curve (AUC-ROC), and Accuracy. All metrics are computed on the held-out test set.
For Tox21, we evaluate the NR-AR (Nuclear Receptor - Androgen Receptor) assay as a
representative single-task classification problem.

\textbf{Implementation Details:} GNN models were implemented using PyTorch Geometric~\cite{pyg}
with Graph Attention Networks (GAT)~\cite{gat2018} and Graph Convolutional Networks
(GCN)~\cite{gcn2017} architectures. Node features included 19 molecular descriptors
(TPSA, LogP, MW, HBA, HBD, etc.). Models were trained with Adam optimizer, learning
rate selected via HPO (range: 1e-4 to 1e-2), batch size 32, and maximum 50 epochs.
All experiments were conducted on CPU (Intel Core i7) due to resource constraints.

\textbf{Reproducibility:} Fixed random seed (seed=42), deterministic operations where
possible. Source code and experimental results available at [GitHub URL].
```

---

## ⚠️ Limitations to Acknowledge

### **In Methods:**
1. Single random seed (no variance across seeds)
2. ChemBERTa not fine-tuned (frozen features only)
3. Tox21 single-task (NR-AR only, not all 12 tasks)
4. CPU-only training (no GPU acceleration)

### **In Discussion:**
```
"Our results demonstrate GNN superiority under specific experimental conditions:
(1) limited computational resources (CPU-only),
(2) systematic HPO for GNN but not foundation models,
(3) single random seed initialization.

Future work should address these limitations by:
- Multi-seed evaluation with statistical significance testing
- Full fine-tuning of foundation models with comparable HPO budgets
- Multi-task learning for Tox21 (all 12 assays)
- GPU-accelerated training for fair computational comparison"
```

---

## ✅ What Makes This Still Valid

### **Strengths:**
1. ✅ **Consistent splits** across all models
2. ✅ **No data leakage** (test completely held out)
3. ✅ **Systematic HPO** (novel contribution)
4. ✅ **Transparent limitations** (clearly stated)
5. ✅ **Reproducible** (fixed seed, code available)

### **Contribution:**
- **Main**: Systematic HPO benchmarking (6 algorithms, 6 datasets)
- **Secondary**: GNN vs Foundation models under constrained resources
- **Practical**: Real-world scenario (limited compute)

---

## 📊 Summary Table for Paper

| Aspect | GNN (Ours) | ChemBERTa | Morgan-FP |
|--------|------------|-----------|-----------|
| **Pretrained** | No | Yes (77M) | No |
| **Fine-tuning** | N/A | No (frozen) | N/A |
| **HPO** | Yes (6 alg, 10 trials) | No | No |
| **Splits** | seed=42 | seed=42 | seed=42 |
| **Metrics** | RMSE, F1, AUC | Same | Same |
| **Compute** | CPU | CPU | CPU |

---

**Conclusion for Paper:**

"Under our experimental setup (limited resources, systematic HPO for GNN only),
GNN models outperform foundation models on all tested datasets. This demonstrates
that systematic hyperparameter optimization can enable simpler architectures to
achieve competitive or superior performance compared to pretrained foundation models
in feature-extraction mode."

---

*Generated: 2026-01-19*
*Purpose: Transparent reporting for Methods/Discussion sections*
