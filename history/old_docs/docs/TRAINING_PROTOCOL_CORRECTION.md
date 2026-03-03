# Training Protocol Correction - CRITICAL UPDATE

**Date:** 2026-01-19
**Status:** ✅ CORRECTED

---

## 🔴 Error Discovered

**Previous (INCORRECT) Documentation:**
> "After HPO finds best hyperparameters on VAL set:
> - Retrain on train+val (90%)
> - Evaluate on test (10%)"

This was **WRONG** and not reflected in the actual code implementation.

---

## ✅ Actual Implementation (Verified from Code)

### **GNN Models** (`optimized_gnn.py`)
```python
# Lines 810-870
for epoch in range(1, epochs + 1):
    train_loss = train_epoch(model, train_loader, ...)  # ← train_loader = 80% only
    val_metrics = evaluate(..., val_loader, ...)        # ← val_loader = 10% for early stopping

    if no_improvement >= patience:
        break  # Early stopping

# Final evaluation (line 865-870)
test_metrics = evaluate(..., test_loader, ...)  # ← test_loader = 10% for final metrics
```

**Training Budget:** **80% (train set ONLY)**

### **Foundation Models** (`benchmark_foundation_models.py`)
```python
# Line 127
predictor.fit(X_train, y_train_scaled)  # ← X_train = 80% only

# Lines 131-135 (evaluation)
val_preds_proba = predictor.predict_proba(X_valid)   # ← 10% validation
test_preds_proba = predictor.predict_proba(X_test)   # ← 10% test (final)
```

**Training Budget:** **80% (train set ONLY)**

---

## ✅ Corrected Protocol

### **Data Splits (All Models):**
- **Train: 80%** → Used for model training
- **Validation: 10%** → Used for:
  - Early stopping (GNN)
  - HPO objective evaluation (GNN)
  - Validation metrics (Foundation)
- **Test: 10%** → Used ONLY for final evaluation (completely held out)

### **After HPO:**
- ❌ **NO retraining on train+val (90%)**
- ✅ **Model uses best checkpoint from training on 80%**
- ✅ **Test set (10%) used only for final reported metrics**

---

## 🎯 Why This is GOOD

### **Fair Comparison:**
All models (GNN, ChemBERTa, Morgan-FP) use the **same training budget (80%)**.

### **No Unfair Advantage:**
If only GNN retrained on 90%, it would have an unfair advantage over foundation models.

### **Standard Practice:**
Many papers use this protocol (train on train, select on val, evaluate on test).

---

## 📝 Corrected Documentation

### **1. Updated Files:**

#### ✅ `docs/EXPERIMENTAL_SETUP_FAIRNESS.md`
- Removed incorrect statement about 90% retraining
- Added explicit note: "NO retraining on train+val"
- Emphasized fair training budget across all models

#### ✅ `docs/latex_methods_experimental_setup.tex` (NEW)
- Complete LaTeX Methods section ready for paper
- Explicitly states: "All models trained exclusively on training set (80%)"
- Emphasizes: "No models retrained on combined train+validation sets"

#### ✅ `docs/latex_discussion_limitations.tex` (NEW)
- Limitations section for Discussion
- Addresses single seed, foundation model constraints, etc.

---

## 📄 For Your Paper - Use This Exact Wording

### **Methods Section (Experimental Setup):**

```latex
\textbf{Training Protocol.}
All models were trained exclusively on the training set (80\% of data). The validation
set (10\%) was reserved for model selection, early stopping, and hyperparameter
optimization metric evaluation. The test set (10\%) remained completely held out during
all training and hyperparameter tuning procedures to prevent data leakage. Importantly,
\emph{no models were retrained on combined train+validation sets after hyperparameter
selection}, ensuring a consistent training budget (80\% of data) across all approaches
for fair comparison.
```

### **Key Points to State:**
1. ✅ All models trained on **80% only**
2. ✅ Validation (10%) used for early stopping + HPO
3. ✅ Test (10%) completely held out
4. ✅ **NO retraining on 90%** after HPO
5. ✅ **Fair training budget** across all approaches

---

## 🔍 Verification Checklist

- [x] Code verified (`optimized_gnn.py:810-870`)
- [x] Foundation models verified (`benchmark_foundation_models.py:127`)
- [x] Documentation corrected (`EXPERIMENTAL_SETUP_FAIRNESS.md`)
- [x] LaTeX Methods section created (`latex_methods_experimental_setup.tex`)
- [x] LaTeX Discussion section created (`latex_discussion_limitations.tex`)
- [x] README checked (no incorrect statements found)

---

## 📊 Summary Table

| Model Type | Training Data | Validation Use | Test Use | Retrain on 90%? |
|------------|---------------|----------------|----------|-----------------|
| **GNN** | 80% | Early stopping, HPO | Final eval | ❌ NO |
| **ChemBERTa** | 80% | Validation metrics | Final eval | ❌ NO |
| **Morgan-FP** | 80% | Validation metrics | Final eval | ❌ NO |

**Conclusion:** ✅ **FAIR COMPARISON** - All models use same 80% training budget.

---

## 🎯 Key Takeaway for Reviewers

**If a reviewer asks:**
> "Did you retrain on train+val after finding best hyperparameters?"

**Answer:**
> "No. To ensure fair comparison, all models (GNN, ChemBERTa, Morgan Fingerprints) were
> trained exclusively on the training set (80%). The validation set was used only for
> model selection and hyperparameter optimization, while the test set remained completely
> held out. This protocol ensures consistent training budget across all approaches and
> prevents any model from having an unfair advantage."

---

**Status:** ✅ Documentation corrected and ready for paper submission.
**Next Steps:** Copy LaTeX sections from `docs/latex_methods_experimental_setup.tex`
and `docs/latex_discussion_limitations.tex` directly into your paper.
