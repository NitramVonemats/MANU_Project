"""
VISUALIZATION DASHBOARD FOR PHASE 2 FOUNDATION MODEL BENCHMARKING
==================================================================
Creates comprehensive visualizations for professor presentation:
1. Model comparison bar charts (R2, RMSE, Spearman)
2. Performance heatmaps
3. Dataset-specific comparisons
4. Model type analysis (GNN-only vs Foundation vs Hybrid)
5. Scatter plots (predicted vs actual)
6. Training time analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path("visualizations/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("FOUNDATION MODEL BENCHMARK - VISUALIZATION DASHBOARD")
print("="*80)

# ===================== LOAD DATA =====================

def load_latest_results():
    """Load the most recent phase2 results"""
    tests_dir = Path("tests")

    # Try all datasets mock first (for demo)
    mock_file = tests_dir / "phase2_all_datasets_mock.csv"
    if mock_file.exists():
        print(f"\nLoading Phase 2 mock results (all datasets): {mock_file.name}")
        return pd.read_csv(mock_file)

    # Try quick results
    quick_file = tests_dir / "phase2_quick_results.csv"
    if quick_file.exists():
        print(f"\nLoading Phase 2 quick results: {quick_file.name}")
        return pd.read_csv(quick_file)

    # Try to find phase2 results
    phase2_files = list(tests_dir.glob("phase2_foundation_benchmark_*.csv"))
    if phase2_files:
        latest = max(phase2_files, key=lambda p: p.stat().st_mtime)
        print(f"\nLoading Phase 2 results: {latest.name}")
        return pd.read_csv(latest)

    # Try phase2 summary
    summary_files = list(tests_dir.glob("phase2_summary_*.csv"))
    if summary_files:
        latest = max(summary_files, key=lambda p: p.stat().st_mtime)
        print(f"\nLoading Phase 2 summary: {latest.name}")
        return pd.read_csv(latest)

    # Fallback to Phase 1 results for demonstration
    print("\nNo Phase 2 results found. Looking for Phase 1 results...")
    week1_files = list(tests_dir.glob("week1_*.csv"))
    if week1_files:
        latest = max(week1_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading Phase 1 results: {latest.name}")
        return pd.read_csv(latest)

    print("\nERROR: No results found! Run phase2_foundation_benchmark.py first.")
    return None

df = load_latest_results()
if df is None:
    exit(1)

print(f"Loaded {len(df)} experiments")
print(f"Columns: {list(df.columns)}")

# ===================== PLOT 1: MODEL COMPARISON BAR CHART =====================

def plot_model_comparison(df, metric='test_r2'):
    """Bar chart comparing all models on a specific metric"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = df['dataset'].unique()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        df_dataset = df[df['dataset'] == dataset]

        # Group by model and compute mean/std
        if 'model_name' in df.columns:
            model_col = 'model_name'
        elif 'model' in df.columns:
            model_col = 'model'
        else:
            print("Warning: No model column found")
            continue

        summary = df_dataset.groupby(model_col)[metric].agg(['mean', 'std']).reset_index()
        summary = summary.sort_values('mean', ascending=False)

        # Create bar plot
        x_pos = np.arange(len(summary))
        ax.bar(x_pos, summary['mean'], yerr=summary['std'],
               capsize=5, alpha=0.7, edgecolor='black')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary[model_col], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (mean_val, std_val) in enumerate(zip(summary['mean'], summary['std'])):
            ax.text(i, mean_val + std_val + 0.01, f'{mean_val:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Model Comparison: {metric.replace("_", " ").title()}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    filename = OUTPUT_DIR / f'model_comparison_{metric}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# Create comparison plots for different metrics
if 'test_r2' in df.columns:
    plot_model_comparison(df, 'test_r2')
if 'test_rmse' in df.columns:
    plot_model_comparison(df, 'test_rmse')
if 'test_spearman' in df.columns:
    plot_model_comparison(df, 'test_spearman')

# ===================== PLOT 2: PERFORMANCE HEATMAP =====================

def plot_performance_heatmap(df):
    """Heatmap showing model performance across datasets"""

    if 'model_name' in df.columns:
        model_col = 'model_name'
    elif 'model' in df.columns:
        model_col = 'model'
    else:
        return

    # Pivot table for heatmap
    pivot = df.pivot_table(values='test_r2', index=model_col,
                           columns='dataset', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, vmin=-0.2, vmax=0.3,
                cbar_kws={'label': 'Test R²'}, ax=ax)

    ax.set_title('Model Performance Heatmap (R² Score)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.tight_layout()
    filename = OUTPUT_DIR / 'performance_heatmap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

if 'test_r2' in df.columns:
    plot_performance_heatmap(df)

# ===================== PLOT 3: MODEL TYPE COMPARISON =====================

def plot_model_type_comparison(df):
    """Compare GNN-only vs Foundation-only vs Hybrid"""

    if 'model_name' not in df.columns and 'model' not in df.columns:
        return

    model_col = 'model_name' if 'model_name' in df.columns else 'model'

    # Categorize models
    def categorize_model(name):
        name_lower = str(name).lower()
        if 'hybrid' in name_lower or '+' in name_lower:
            return 'Hybrid'
        elif 'gnn' in name_lower:
            return 'GNN-only'
        else:
            return 'Foundation-only'

    df['model_type'] = df[model_col].apply(categorize_model)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = df['dataset'].unique()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        df_dataset = df[df['dataset'] == dataset]

        # Box plot by model type
        data_to_plot = [df_dataset[df_dataset['model_type'] == mt]['test_r2'].values
                        for mt in ['GNN-only', 'Foundation-only', 'Hybrid']]

        bp = ax.boxplot(data_to_plot, labels=['GNN-only', 'Foundation-only', 'Hybrid'],
                        patch_artist=True, showmeans=True)

        # Color boxes
        colors = ['#FF9999', '#99CCFF', '#99FF99']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Test R²', fontsize=12)
        ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Model Type Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    filename = OUTPUT_DIR / 'model_type_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

if 'test_r2' in df.columns:
    plot_model_type_comparison(df)

# ===================== PLOT 4: TRAINING TIME ANALYSIS =====================

def plot_training_time(df):
    """Analyze training time vs performance"""

    if 'train_time_s' not in df.columns:
        return

    # Check if model column exists
    if 'model_name' in df.columns:
        model_col = 'model_name'
    elif 'model' in df.columns:
        model_col = 'model'
    else:
        print("Warning: No model column for training time plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Training time by model
    ax = axes[0]
    summary = df.groupby(model_col)['train_time_s'].agg(['mean', 'std']).reset_index()
    summary = summary.sort_values('mean', ascending=False)

    x_pos = np.arange(len(summary))
    ax.barh(x_pos, summary['mean'], xerr=summary['std'],
            capsize=5, alpha=0.7, edgecolor='black')

    ax.set_yticks(x_pos)
    ax.set_yticklabels(summary[model_col])
    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time by Model', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Plot 2: Efficiency (R² per second)
    ax = axes[1]
    df['efficiency'] = df['test_r2'] / (df['train_time_s'] + 1)  # +1 to avoid div by 0

    summary_eff = df.groupby(model_col)['efficiency'].mean().reset_index()
    summary_eff = summary_eff.sort_values('efficiency', ascending=False)

    x_pos = np.arange(len(summary_eff))
    ax.barh(x_pos, summary_eff['efficiency'], alpha=0.7, edgecolor='black', color='green')

    ax.set_yticks(x_pos)
    ax.set_yticklabels(summary_eff[model_col])
    ax.set_xlabel('Efficiency (R² / second)', fontsize=12)
    ax.set_title('Model Efficiency', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    filename = OUTPUT_DIR / 'training_time_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

if 'train_time_s' in df.columns:
    plot_training_time(df)

# ===================== PLOT 5: FOUNDATION MODEL RANKING =====================

def plot_foundation_ranking(df):
    """Rank foundation models across all datasets"""

    model_col = 'model_name' if 'model_name' in df.columns else 'model'

    # Calculate average rank for each model
    datasets = df['dataset'].unique()
    rankings = []

    for dataset in datasets:
        df_dataset = df[df['dataset'] == dataset]
        summary = df_dataset.groupby(model_col)['test_r2'].mean().reset_index()
        summary['rank'] = summary['test_r2'].rank(ascending=False)
        summary['dataset'] = dataset
        rankings.append(summary[[model_col, 'dataset', 'rank']])

    rankings_df = pd.concat(rankings)
    avg_rank = rankings_df.groupby(model_col)['rank'].mean().reset_index()
    avg_rank = avg_rank.sort_values('rank')

    fig, ax = plt.subplots(figsize=(12, 8))

    x_pos = np.arange(len(avg_rank))
    ax.barh(x_pos, avg_rank['rank'], alpha=0.7, edgecolor='black')

    ax.set_yticks(x_pos)
    ax.set_yticklabels(avg_rank[model_col])
    ax.set_xlabel('Average Rank (lower is better)', fontsize=12)
    ax.set_title('Foundation Model Ranking (across all datasets)',
                 fontsize=16, fontweight='bold')
    ax.invert_xaxis()  # Lower rank is better
    ax.grid(axis='x', alpha=0.3)

    # Add rank numbers
    for i, rank in enumerate(avg_rank['rank']):
        ax.text(rank - 0.1, i, f'{rank:.1f}',
               ha='right', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    filename = OUTPUT_DIR / 'foundation_ranking.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

if 'test_r2' in df.columns:
    plot_foundation_ranking(df)

# ===================== PLOT 6: METRIC CORRELATION =====================

def plot_metric_correlation(df):
    """Correlation between different metrics"""

    metrics = ['test_r2', 'test_rmse', 'test_spearman']
    available_metrics = [m for m in metrics if m in df.columns]

    if len(available_metrics) < 2:
        return

    fig, axes = plt.subplots(1, len(available_metrics)-1, figsize=(6*len(available_metrics)-6, 5))

    if len(available_metrics) == 2:
        axes = [axes]

    for idx, (m1, m2) in enumerate(zip(available_metrics[:-1], available_metrics[1:])):
        ax = axes[idx]

        ax.scatter(df[m1], df[m2], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        # Add regression line
        z = np.polyfit(df[m1], df[m2], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[m1].min(), df[m1].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Correlation coefficient
        corr = df[[m1, m2]].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(m1.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(m2.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{m1} vs {m2}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    filename = OUTPUT_DIR / 'metric_correlation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

plot_metric_correlation(df)

# ===================== SUMMARY =====================

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print("\nGenerated plots:")
print("  1. model_comparison_*.png - Bar charts for each metric")
print("  2. performance_heatmap.png - R² heatmap across models/datasets")
print("  3. model_type_comparison.png - GNN vs Foundation vs Hybrid")
print("  4. training_time_analysis.png - Time and efficiency analysis")
print("  5. foundation_ranking.png - Overall model ranking")
print("  6. metric_correlation.png - Correlation between metrics")
print("\nUse these for your professor presentation!")
print("="*80)
