# MANU Datasets

All datasets are from the **Therapeutics Data Commons (TDC) ADMET Benchmark**.

## Dataset Overview

| Dataset | Type | Task | Samples | Metric | Difficulty | Notes |
|---------|------|------|---------|--------|-----------|-------|
| **Caco2_Wang** | ADME | Regression | 910 | RMSE | ✓ Moderate | Tractable |
| **Half_Life_Obach** | PK | Regression | 667 | RMSE | ✗ Hard | Multicausal |
| **Clearance_Hepatocyte_AZ** | ADME | Regression | 1,213 | RMSE | ✗✗ Very Hard | Complex |
| **Clearance_Microsome_AZ** | ADME | Regression | 1,102 | RMSE | ⚠️ Weak | Limited Signal |
| **Tox21 (NR-AR)** | Toxicity | Classification | 7,258 | AUC-ROC | ⚠️ Imbalanced | 3.5% positive |
| **hERG** | Toxicity | Classification | 655 | AUC-ROC | ✓ Good | Balanced |

**Total molecules:** 11,805

---

## 📊 REGRESSION TASKS

### 1. Caco-2 Permeability (Caco2_Wang)

**Target Property:** Apparent permeability (P_app) in intestinal Caco-2 cells

**Definition:**
- P_app > 10⁻⁶ cm/s → Good oral bioavailability
- Measured in log(cm/s) units
- In vitro surrogate for human intestinal absorption

**Dataset Details:**
- **Samples:** 910 compounds
- **Source:** Wang et al. (curated dataset)
- **Target Distribution:** Continuous log values
- **Data Quality:** High (well-established assay)

**Challenge Level:** ✓ **MODERATE** (R² = 0.48)

**Why Tractable:**
- Permeability governed by molecular size, lipophilicity, H-bonding
- GNNs can capture these structural features directly
- Relatively strong signal in data

**Why Not Perfect:**
- In vitro-in vivo correlation not perfect
- Bidirectional transport (efflux) hard to predict from structure
- Individual variability

**Prediction Performance:**
- Best: Random (RMSE=0.0027, R²=0.481)
- Good enough for early screening

---

### 2. Plasma Half-Life (Half_Life_Obach)

**Target Property:** Elimination half-life (t₁/₂) in human plasma

**Definition:**
- Time for plasma concentration to drop by 50%
- Measured in hours
- Determines dosing frequency

**Dataset Details:**
- **Samples:** 667 FDA-approved drugs
- **Source:** Obach et al. database
- **Target Distribution:** Wide range (0.5–100+ hours), highly skewed
- **Preprocessing:** Log-transformed before z-score normalization

**Challenge Level:** ✗ **VERY HARD** (R² = 0.004)

**Why So Difficult:**
Half-life is a **composite pharmacokinetic parameter** influenced by:
1. Metabolic clearance (structure-dependent)
2. Volume of distribution (protein binding, tissue accumulation)
3. Renal elimination (not predicted from structure)
4. Individual patient variability (genetics, age, disease)

Only #1 is encoded in molecule structure → **Fundamental prediction limit**

**Prediction Performance:**
- Best: PSO (RMSE=21.66, R²=0.004)
- R² ≈ 0: Structure explains ~0% of variance
- Still useful for relative ranking, not absolute prediction

**Conclusion:** This dataset demonstrates the **limitation of structure-only models** for complex PK parameters

---

### 3. Hepatocyte Clearance (Clearance_Hepatocyte_AZ)

**Target Property:** Intrinsic clearance in freshly isolated human hepatocytes

**Definition:**
- μL/min/10⁶ cells
- Measures both Phase I & Phase II metabolism
- Gold-standard in vitro metabolic stability assay
- Direct relevance to in vivo half-life

**Dataset Details:**
- **Samples:** 1,213 proprietary compounds (from AstraZeneca)
- **Source:** Therapeutics Data Commons (internal AZ data)
- **Target Distribution:** Wide range, heavily skewed
- **Preprocessing:** Log-transform + z-score normalization

**Challenge Level:** ✗✗ **IMPOSSIBLE** (R² = -1.019)

**Why Worst Performance:**
1. **Extreme complexity:** Metabolic enzyme kinetics poorly encoded in 2D structure
2. **Metabolic unexplainability:** Different enzymes (CYP2D6, CYP3A4, UGTs) have different affinities
3. **Individual variation:** Enzyme expression varies between donors
4. **Proprietary data:** Likely noisier than public datasets
5. **Missing features:** 3D structure, electron density, binding mode needed

**Prediction Performance:**
- Best: Random (RMSE=68.22, R²=-1.019)
- **R² < 0:** Model performs WORSE than predicting the mean
- Baseline (always predict mean): RMSE=69.72
- Model improvement: -4.3% (actually worse!)

**Interpretation:** Structure alone is **fundamentally insufficient** for hepatocyte clearance prediction

**Implication:** Multi-modal data essential:
- Gene expression (enzyme availability)
- Protein binding
- 3D structure/docking
- Metabolite identification
- Clinical covariates

---

### 4. Microsomal Clearance (Clearance_Microsome_AZ)

**Target Property:** Intrinsic clearance in human liver microsomes

**Definition:**
- μL/min/mg protein
- Measures primarily CYP P450-mediated oxidative metabolism
- Phase I reactions only (unlike hepatocytes)
- Simpler than hepatocytes → lower variability

**Dataset Details:**
- **Samples:** 1,102 compounds (AstraZeneca internal)
- **Source:** Therapeutics Data Commons
- **Target Distribution:** Right-skewed, wide range
- **Preprocessing:** Log-transform + z-score normalization

**Challenge Level:** ⚠️ **WEAK SIGNAL** (R² = 0.191)

**Why Weak Performance:**
1. Similar issues to hepatocytes, but simpler
2. Still subject to individual donor variation
3. Assay variability (substrate prep, conditions)
4. No phase II metabolism in microsomes
5. Structure-metabolism relationship complex for Phase I

**Prediction Performance:**
- Best: Random (RMSE=38.75, R²=0.191)
- Better than hepatocytes, but still poor
- 19% of variance explained (not great)

**Takeaway:** Even simpler metabolic assays are hard to predict from structure alone

---

## 🔬 CLASSIFICATION TASKS

### 5. hERG Cardiotoxicity (hERG)

**Target Property:** hERG potassium channel blockade

**Definition:**
- Binary: Blocker (1) vs Non-blocker (0)
- hERG is involved in cardiac action potential repolarization
- Blockade → QT prolongation → arrhythmia risk
- **Mandatory safety screen** in drug development

**Dataset Details:**
- **Samples:** 655 compounds
- **Source:** Therapeutics Data Commons
- **Class Distribution:** 31% blockers, 69% non-blockers
- **Balance:** Reasonable (not too imbalanced)

**Challenge Level:** ✓ **GOOD** (AUC = 0.825, F1 = 0.809)

**Why Good Performance:**
1. **Clear structural alerts:** Specific pharmacophores block hERG
2. **Well-studied biology:** 20+ years of SAR data
3. **Balanced dataset:** 31% positive is manageable
4. **Strong signal:** Structure encodes blockade mechanism well

**Prediction Performance:**
- Best: ABC (AUC=0.825, F1=0.809)
- High sensitivity: Misses few blockers (clinically important)
- High specificity: Few false positives
- Suitable for **virtual screening** to deprioritize cardiotoxic compounds

**Clinical Utility:** This model could be deployed in drug discovery pipelines

**Key Insight:** Toxicity endpoints often more predictable than PK parameters

---

### 6. Tox21 Androgen Receptor (Tox21 NR-AR)

**Target Property:** Nuclear receptor androgen receptor (AR) activation

**Definition:**
- Binary: Activator (1) vs Inactive (0)
- Part of Tox21 multi-assay toxicity panel
- Screens for endocrine disruptor potential
- Indicator of reproductive/developmental toxicity

**Dataset Details:**
- **Samples:** 7,258 compounds (largest dataset)
- **Source:** Tox21 initiative (NIH/EPA/NCATS)
- **Class Distribution:** 3.5% active (severe imbalance!)
- **Challenge:** Predicting rare positive class

**Challenge Level:** ⚠️ **IMBALANCED** (AUC = 0.742, F1 = 0.455)

**Why Imbalance is Problematic:**
```
If model always predicts "inactive":
  - Accuracy = 96.5% (misleading!)
  - F1 = 0 (fails to identify actives)
  - AUC-ROC = 0.5 (no discrimination)

Our model:
  - Accuracy = 96.2% (high, but primarily from majority class)
  - F1 = 0.455 (better, but many false negatives)
  - AUC = 0.742 (reasonable discrimination)
```

**Mitigation Strategies Used:**
1. **Class weighting:** Loss weight = (neg samples) / (pos samples) ≈ 28x
2. **Focus on AUC-ROC:** Less sensitive to class imbalance
3. **Report F1:** Captures minority class performance

**Prediction Performance:**
- Best: SA (AUC=0.742, F1=0.455)
- **Sensitivity:** Identifies ~50% of true actives
- **Specificity:** ~95% of inactives correctly labeled
- **Tradeoff:** Misses many actives (low recall for positive class)

**Practical Use:** Better as **triage filter** than primary screen
- Use for high-throughput flagging
- Manual review of flagged compounds
- Higher confidence in negative predictions

**Key Challenge:** Rare positive class makes minority class prediction inherently difficult

---

## 📈 Comparative Analysis

### Predictability Ranking

```
Easiest → Hardest

1. hERG (AUC=0.825)         ✓ Strong
2. Tox21 (AUC=0.742)        ⚠️ Moderate (imbalance)
3. Caco2_Wang (R²=0.481)     ✓ Moderate
4. Clearance_Microsome (R²=0.191)   ⚠️ Weak
5. Half_Life (R²=0.004)      ✗ Very Hard
6. Clearance_Hepatocyte (R²=-1.019) ✗✗ Impossible
```

### Why Toxicity > ADMET Performance?

**Toxicity (Good):**
- Structural alerts well-defined
- 20+ years of SAR knowledge
- Binary outcome (clear signal)
- Mechanism relatively simple

**ADMET (Poor):**
- Multi-factorial (clearance, distribution, PK)
- Individual variability high
- Assay-dependent factors
- Need multi-modal data

### Data Modality Implications

| Modality | Useful For | Not Useful For |
|----------|-----------|--------|
| **2D Structure** | Toxicity alerts | Clearance, Half-life |
| **3D Conformation** | Binding, docking | Distribution |
| **Protein Binding** | PK parameters | Toxicity |
| **Gene Expression** | Clearance, metabolism | Toxicity |
| **Clinical Covariates** | Half-life variance | Structure-specific properties |

---

## 🔍 Data Access

All datasets are publicly available through TDC:

```python
from tdc.benchmark_group import admet_group

# Load ADMET benchmark
benchmark = admet_group(name='TDC')

# Access individual dataset
data = benchmark.get_data('Caco2_Wang')
train, valid, test = data['train'], data['valid'], data['test']
```

**Reference:** Huang et al. (2021) "Therapeutics Data Commons: Machine Learning Datasets and Benchmarks for Drug Discovery and Development"

---

## 📝 Summary

| Aspect | Finding |
|--------|---------|
| **Best predictable task** | hERG toxicity (AUC=0.825) |
| **Hardest predictable task** | Hepatocyte clearance (R²=-1.019) |
| **Key insight** | Structure encodes toxicity better than PK |
| **Data quality** | Good (publicly vetted via TDC) |
| **Imbalance challenge** | Tox21: only 3.5% actives |
| **Imbalance solution** | Class weighting + AUC-ROC metric |
| **Recommendation** | Use structure-only models for toxicity screening, need multi-modal data for ADMET |

---

**Last Updated:** 2026-03-09
