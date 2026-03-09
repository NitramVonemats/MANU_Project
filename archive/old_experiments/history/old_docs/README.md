# Visualization Dashboard

**For Professor Presentation**

---

## 📊 Quick Start

### Generate All Visualizations:

```bash
cd GNN_test

# 1. Generate architecture diagrams (always works)
python visualizations/create_architecture_diagrams.py

# 2. Generate performance plots (needs results from Phase 2)
python visualizations/create_all_plots.py
```

---

## 📁 Output Structure

```
visualizations/
├── plots/                          # Generated figures
│   ├── architecture_gnn_only.png
│   ├── architecture_foundation_only.png
│   ├── architecture_hybrid.png
│   ├── model_comparison_test_r2.png
│   ├── model_comparison_test_rmse.png
│   ├── performance_heatmap.png
│   ├── model_type_comparison.png
│   ├── training_time_analysis.png
│   ├── foundation_ranking.png
│   └── metric_correlation.png
│
├── create_all_plots.py             # Main plotting script
├── create_architecture_diagrams.py # Architecture diagrams
└── README.md                       # This file
```

---

## 🎨 Visualization Types

### 1. Architecture Diagrams

**Files:**
- `architecture_gnn_only.png`
- `architecture_foundation_only.png`
- `architecture_hybrid.png`

**What they show:**
- Visual flowcharts of each model architecture
- Input → Processing → Output pipeline
- Dimensions at each layer
- Color-coded components

**When to use:**
- Explaining "How does your model work?"
- Showing differences between approaches
- Illustrating fusion strategy

---

### 2. Model Comparison Bar Charts

**Files:**
- `model_comparison_test_r2.png`
- `model_comparison_test_rmse.png`
- `model_comparison_test_spearman.png`

**What they show:**
- Performance of each model on each dataset
- Error bars showing variability across seeds
- Direct comparison: which model is best?

**When to use:**
- Answering "Which model performs best?"
- Showing statistical significance
- Comparing across datasets

**Example interpretation:**
```
Half_Life_Obach:
  GNN-only:      R²=0.091 ± 0.041  ← Phase 1 baseline
  MolFormer:     R²=0.15  ± 0.03   ← Better!
  GNN+MolFormer: R²=0.18  ± 0.025  ← Best! Hybrid wins
```

---

### 3. Performance Heatmap

**File:** `performance_heatmap.png`

**What it shows:**
- R² scores for all model × dataset combinations
- Color gradient: green = good, red = bad
- Easy to spot best model for each dataset

**When to use:**
- Quick overview of all results
- Identifying patterns across datasets
- Spotting which models are consistent

---

### 4. Model Type Comparison

**File:** `model_type_comparison.png`

**What it shows:**
- Box plots comparing GNN-only vs Foundation-only vs Hybrid
- Distribution of performance within each category
- Statistical differences between approaches

**When to use:**
- Testing hypothesis: "Does fusion help?"
- Comparing architectural strategies
- Showing variability vs consistency

---

### 5. Training Time Analysis

**File:** `training_time_analysis.png`

**What it shows:**
- Left panel: Training time per model
- Right panel: Efficiency (R² per second)
- Trade-off between speed and accuracy

**When to use:**
- Discussing computational costs
- Choosing models for production
- Justifying resource requirements

**Example insights:**
```
Morgan-FP:   Fast but low accuracy
GNN-only:    Medium speed, good accuracy
MolFormer:   Slow but high accuracy
Hybrid:      Slowest, but best results
```

---

### 6. Foundation Model Ranking

**File:** `foundation_ranking.png`

**What it shows:**
- Average rank of each model across all datasets
- Lower rank = better (1st, 2nd, 3rd...)
- Overall winner across all tasks

**When to use:**
- Concluding "Which foundation model is best overall?"
- Cross-dataset generalization
- Final recommendation

---

### 7. Metric Correlation

**File:** `metric_correlation.png`

**What it shows:**
- Scatter plots: R² vs RMSE, R² vs Spearman, etc.
- Regression lines showing relationships
- Correlation coefficients

**When to use:**
- Validating results (do metrics agree?)
- Understanding what R² really means
- Spotting outliers

---

## 🎓 For Your Presentation

### Recommended Slide Flow:

**Slide 1: Problem**
- "Which foundation model is best for ADME prediction?"

**Slide 2-4: Architectures**
- Show `architecture_*.png` diagrams
- Explain GNN, Foundation, Hybrid approaches

**Slide 5: Main Results**
- Show `model_comparison_test_r2.png`
- Highlight best models

**Slide 6: Cross-Dataset Performance**
- Show `performance_heatmap.png`
- Discuss consistency

**Slide 7: Model Type Analysis**
- Show `model_type_comparison.png`
- Answer: "Does fusion help?" (YES/NO based on results)

**Slide 8: Efficiency**
- Show `training_time_analysis.png`
- Discuss trade-offs

**Slide 9: Conclusion**
- Show `foundation_ranking.png`
- Final recommendation

---

## 📊 Interpreting Results

### Scenario A: Hybrid Models Win (Expected)

```
Ranking:
1. GNN + MolFormer      ← WINNER
2. GNN + ChemBERTa
3. MolFormer-only
4. GNN-only
5. ChemBERTa-only
```

**Conclusion:**
"Fusion of graph structure and pretrained knowledge works! Recommend hybrid architectures for production."

---

### Scenario B: Foundation-only Wins

```
Ranking:
1. MolFormer-only       ← WINNER
2. ChemBERTa-only
3. GNN + MolFormer
4. GNN-only
```

**Conclusion:**
"Pretrained knowledge dominates. Graph structure adds noise. Recommend foundation-only models."

---

### Scenario C: GNN-only Still Best

```
Ranking:
1. GNN-only             ← WINNER (Phase 1)
2. GNN + MolFormer
3. MolFormer-only
```

**Conclusion:**
"Task-specific learning outperforms pretrained models. Foundation models didn't transfer well to ADME."

---

## 🔧 Customization

### Change Color Scheme:

Edit `create_all_plots.py`:
```python
# Line ~19
sns.set_palette("husl")  # Change to "Set2", "viridis", "coolwarm", etc.
```

### Add More Metrics:

```python
# Add to plot functions:
plot_model_comparison(df, 'val_r2')      # Validation R²
plot_model_comparison(df, 'test_mae')    # Mean Absolute Error
```

### Filter Specific Models:

```python
# Before plotting:
df = df[df['model_name'].str.contains('Hybrid')]  # Only hybrid models
df = df[df['dataset'] == 'Half_Life_Obach']       # Only one dataset
```

---

## ❓ Troubleshooting

### "No results found!"

**Problem:** Haven't run Phase 2 yet.

**Solution:**
```bash
# Option 1: Run quick test first (10 epochs, fast)
python test_phase2_quick.py

# Option 2: Run full benchmark (slow)
python phase2_foundation_benchmark.py
```

---

### "Architecture diagrams look weird"

**Problem:** Matplotlib version issues.

**Solution:**
```bash
pip install --upgrade matplotlib
```

Or just use the diagrams as-is - they're still informative!

---

### "I want different plots"

Edit `create_all_plots.py` and add your own plotting functions. Template:

```python
def plot_my_custom_viz(df):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Your plotting code here

    filename = OUTPUT_DIR / 'my_custom_plot.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Call it
plot_my_custom_viz(df)
```

---

## 📚 Dependencies

Required packages:
```bash
pip install pandas numpy matplotlib seaborn
```

All should already be installed from Phase 1/2 requirements.

---

## 🎉 Pro Tips

1. **High DPI:** All plots are 300 DPI - perfect for presentations
2. **Vector graphics:** For even better quality, change `.png` to `.pdf` in save commands
3. **Batch generation:** Run both scripts in sequence to get everything
4. **Live updates:** Re-run after each new Phase 2 experiment to update plots

---

**Questions?** Check main `README.md` or `phase2_foundation_benchmark.py`

Good luck with your presentation! 🚀
