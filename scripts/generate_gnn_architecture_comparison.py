#!/usr/bin/env python3
"""
Generate GNN Architecture Comparison Visualizations
Based on results from architecture selection phase testing different GNN types.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 150

# Create output directory
output_dir = Path("figures/paper-sources-2")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Data from MODEL_STATISTICS.csv and archive/summaries
# ============================================================================

# Caco2_Wang results (R² values from DOCUMENTATION_COMPLETE.md)
caco2_data = {
    'Model': ['GCN', 'GraphSAGE', 'GIN', 'TAG', 'SGC'],
    'Test_R2': [0.30, 0.36, 0.04, 0.21, 0.16],
    'Training_Time': [30, 45, 85, 94, 35]
}

# Half_Life_Obach results
half_life_data = {
    'Model': ['Graph', 'GCN', 'TAG', 'GIN', 'SGC', 'Transformer', 'GAT', 'SAGE'],
    'N_Experiments': [16, 20, 20, 17, 27, 9, 15, 49],
    'Mean_RMSE': [16.64, 15.30, 30.67, 18.02, 17.99, 12.88, 474.68, 19.49],
    'Min_RMSE': [0.839, 0.949, 0.959, 0.985, 1.065, 1.027, 17.22, 17.29],
    'Max_R2': [0.384, 0.468, 0.404, 0.392, 0.399, 0.327, 0.370, 0.365]
}

# Clearance_Hepatocyte_AZ results
hepatocyte_data = {
    'Model': ['Graph', 'TAG', 'GCN', 'Transformer', 'GIN', 'SGC', 'SAGE', 'GAT'],
    'N_Experiments': [9, 11, 10, 4, 7, 12, 19, 7],
    'Mean_RMSE': [39.44, 36.35, 47.23, 25.67, 50.45, 169.28, 219.84, 67.49],
    'Min_RMSE': [1.192, 1.230, 1.221, 1.277, 1.339, 1.242, 49.69, 50.79],
    'Max_R2': [0.087, 0.027, 0.041, -0.030, -0.126, 0.009, -0.072, -0.120]
}

# Clearance_Microsome_AZ results
microsome_data = {
    'Model': ['Graph', 'TAG', 'GIN', 'Transformer', 'GCN', 'SGC', 'SAGE', 'GAT'],
    'N_Experiments': [9, 11, 8, 4, 9, 12, 19, 7],
    'Mean_RMSE': [33.01, 26.84, 32.35, 20.93, 28.20, 33.99, 44.20, 43.33],
    'Min_RMSE': [1.018, 1.041, 1.075, 1.150, 1.198, 1.235, 37.43, 39.79],
    'Max_R2': [0.321, 0.291, 0.243, 0.149, 0.283, 0.259, 0.245, 0.147]
}

# Classification datasets (from archive/summaries/РЕЗИМЕ_ПОДГОТВЕНО.md)
tox21_data = {
    'Model': ['GCN', 'GAT', 'GraphSAGE'],
    'Test_AUC': [0.823, 0.789, 0.801],
    'Test_F1': [0.756, 0.712, 0.734]
}

herg_data = {
    'Model': ['GAT', 'GCN', 'GraphSAGE'],
    'Test_AUC': [0.789, 0.776, 0.768],
    'Test_F1': [0.712, 0.698, 0.689]
}

# ============================================================================
# Figure 1: Comprehensive GNN Architecture Comparison (2x3 Grid)
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Regression datasets (RMSE)
# Caco2 - R² (higher is better)
ax = axes[0, 0]
df_caco2 = pd.DataFrame(caco2_data)
colors_caco2 = sns.color_palette("husl", len(df_caco2))
bars = ax.bar(df_caco2['Model'], df_caco2['Test_R2'], color=colors_caco2, edgecolor='black', linewidth=0.5)
max_idx = df_caco2['Test_R2'].idxmax()
bars[max_idx].set_edgecolor('gold')
bars[max_idx].set_linewidth(3)
ax.set_title('Caco2_Wang\n(Test R² ↑)', fontsize=11, fontweight='bold')
ax.set_xlabel('GNN Architecture')
ax.set_ylabel('Test R²')
ax.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, df_caco2['Test_R2']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Other regression datasets (RMSE - lower is better)
datasets = ['Half_Life_Obach', 'Clearance_Hepatocyte', 'Clearance_Microsome']
data_list = [half_life_data, hepatocyte_data, microsome_data]
model_order = ['Graph', 'GCN', 'TAG', 'GIN', 'SGC', 'Transformer', 'GAT', 'SAGE']
colors = sns.color_palette("husl", len(model_order))

for idx, (dataset, data) in enumerate(zip(datasets, data_list)):
    if idx < 2:
        ax = axes[0, idx + 1]
    else:
        ax = axes[1, 0]

    df = pd.DataFrame(data)
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    df = df.sort_values('Model')

    bars = ax.bar(df['Model'], df['Min_RMSE'], color=colors, edgecolor='black', linewidth=0.5)
    min_idx = df['Min_RMSE'].idxmin()
    best_model = df.loc[min_idx, 'Model']
    for i, bar in enumerate(bars):
        if df.iloc[i]['Model'] == best_model:
            bar.set_edgecolor('gold')
            bar.set_linewidth(3)

    ax.set_title(f'{dataset}\n(Best RMSE ↓)', fontsize=11, fontweight='bold')
    ax.set_xlabel('GNN Architecture')
    ax.set_ylabel('Test RMSE (log)')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, min(5, df['Min_RMSE'].max() * 1.2))

    for i, (bar, val) in enumerate(zip(bars, df['Min_RMSE'])):
        if val < 5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Row 2: Classification datasets (AUC)
# Tox21
ax = axes[1, 1]
df_tox21 = pd.DataFrame(tox21_data)
colors_tox21 = sns.color_palette("Set2", len(df_tox21))
bars = ax.bar(df_tox21['Model'], df_tox21['Test_AUC'], color=colors_tox21, edgecolor='black', linewidth=0.5)
max_idx = df_tox21['Test_AUC'].idxmax()
bars[max_idx].set_edgecolor('gold')
bars[max_idx].set_linewidth(3)
ax.set_title('Tox21 (NR-AR)\n(Test AUC ↑)', fontsize=11, fontweight='bold')
ax.set_xlabel('GNN Architecture')
ax.set_ylabel('Test AUC-ROC')
ax.set_ylim(0.7, 0.9)
for bar, val in zip(bars, df_tox21['Test_AUC']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
           f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# hERG
ax = axes[1, 2]
df_herg = pd.DataFrame(herg_data)
colors_herg = sns.color_palette("Set2", len(df_herg))
bars = ax.bar(df_herg['Model'], df_herg['Test_AUC'], color=colors_herg, edgecolor='black', linewidth=0.5)
max_idx = df_herg['Test_AUC'].idxmax()
bars[max_idx].set_edgecolor('gold')
bars[max_idx].set_linewidth(3)
ax.set_title('hERG (Cardiotoxicity)\n(Test AUC ↑)', fontsize=11, fontweight='bold')
ax.set_xlabel('GNN Architecture')
ax.set_ylabel('Test AUC-ROC')
ax.set_ylim(0.7, 0.85)
for bar, val in zip(bars, df_herg['Test_AUC']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
           f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'gnn_architecture_comparison_all_datasets.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {output_dir / 'gnn_architecture_comparison_all_datasets.png'}")

# ============================================================================
# Figure 1b: Original 3-dataset comparison (keep for backwards compatibility)
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

datasets = ['Half_Life_Obach', 'Clearance_Hepatocyte', 'Clearance_Microsome']
data_list = [half_life_data, hepatocyte_data, microsome_data]

for idx, (ax, dataset, data) in enumerate(zip(axes, datasets, data_list)):
    df = pd.DataFrame(data)
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    df = df.sort_values('Model')

    bars = ax.bar(df['Model'], df['Min_RMSE'], color=colors, edgecolor='black', linewidth=0.5)

    min_idx = df['Min_RMSE'].idxmin()
    best_model = df.loc[min_idx, 'Model']
    for i, bar in enumerate(bars):
        if df.iloc[i]['Model'] == best_model:
            bar.set_edgecolor('gold')
            bar.set_linewidth(3)

    ax.set_title(f'{dataset}\n(Best RMSE log-scale)', fontsize=11, fontweight='bold')
    ax.set_xlabel('GNN Architecture')
    ax.set_ylabel('Test RMSE (log-scale)' if idx == 0 else '')
    ax.tick_params(axis='x', rotation=45)

    for i, (bar, val) in enumerate(zip(bars, df['Min_RMSE'])):
        if val < 20:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Set y-axis limit to show detail for good models
    ax.set_ylim(0, min(5, df['Min_RMSE'].max() * 1.2))

plt.tight_layout()
plt.savefig(output_dir / 'gnn_architecture_comparison.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {output_dir / 'gnn_architecture_comparison.png'}")

# ============================================================================
# Figure 2: Stability Analysis (Box Plot style with error bars)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Create summary data across all datasets
stability_summary = {
    'Model': ['Graph', 'GCN', 'TAG', 'GIN', 'SGC', 'Transformer', 'GAT', 'SAGE'],
    'Half_Life': [0.839, 0.949, 0.959, 0.985, 1.065, 1.027, 17.22, 17.29],
    'Hepatocyte': [1.192, 1.221, 1.230, 1.339, 1.242, 1.277, 50.79, 49.69],
    'Microsome': [1.018, 1.198, 1.041, 1.075, 1.235, 1.150, 39.79, 37.43]
}

df_stability = pd.DataFrame(stability_summary)

# Calculate mean and std across datasets
df_stability['Mean'] = df_stability[['Half_Life', 'Hepatocyte', 'Microsome']].mean(axis=1)
df_stability['Std'] = df_stability[['Half_Life', 'Hepatocyte', 'Microsome']].std(axis=1)

# Sort by mean performance
df_stability = df_stability.sort_values('Mean')

# Create grouped bar chart
x = np.arange(len(df_stability))
width = 0.25

bars1 = ax.bar(x - width, df_stability['Half_Life'], width, label='Half_Life', color='steelblue', alpha=0.8)
bars2 = ax.bar(x, df_stability['Hepatocyte'], width, label='Hepatocyte', color='darkorange', alpha=0.8)
bars3 = ax.bar(x + width, df_stability['Microsome'], width, label='Microsome', color='forestgreen', alpha=0.8)

ax.set_xlabel('GNN Architecture')
ax.set_ylabel('Best Test RMSE (log-scale)')
ax.set_title('GNN Architecture Performance Across Datasets\n(Lower is better)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_stability['Model'], rotation=45, ha='right')
ax.legend(loc='upper right')

# Limit y-axis to show detail
ax.set_ylim(0, 5)

# Add annotation for outliers
for model in ['GAT', 'SAGE']:
    model_idx = df_stability[df_stability['Model'] == model].index[0]
    x_pos = list(df_stability['Model']).index(model)
    ax.annotate(f'{model}\noutlier', xy=(x_pos, 4.8), ha='center', fontsize=8, color='red')

plt.tight_layout()
plt.savefig(output_dir / 'gnn_architecture_stability.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {output_dir / 'gnn_architecture_stability.png'}")

# ============================================================================
# Figure 3: Hyperparameter Sensitivity Heatmap
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Layers sensitivity data
layers_data = {
    'Layers': [2, 3, 4, 5, 6],
    'Half_Life': [0.959, 0.949, 16.51, 0.839, 17.15],
    'Hepatocyte': [1.230, 1.221, 47.18, 1.192, np.nan],
    'Microsome': [1.180, 1.150, 36.49, 1.018, np.nan]
}

# Hidden channels sensitivity data
hidden_data = {
    'Hidden': [32, 64, 128, 256, 512],
    'Half_Life': [0.976, 0.949, 0.839, 15.82, 17.29],
    'Hepatocyte': [1.304, 1.221, 1.192, 48.78, np.nan],
    'Microsome': [1.198, 1.041, 1.018, 37.67, np.nan]
}

# Learning rate sensitivity data
lr_data = {
    'LR': ['5e-5', '1e-4', '5e-4', '1e-3', '2e-3', '5e-3'],
    'Half_Life': [17.35, 16.92, 15.82, 0.839, 18.29, 16.86],
    'Hepatocyte': [52.71, 49.90, 48.94, 1.192, np.nan, np.nan],
    'Microsome': [44.39, 38.95, 37.08, 1.018, np.nan, np.nan]
}

# Plot 1: Layers
df_layers = pd.DataFrame(layers_data).set_index('Layers')
df_layers_clipped = df_layers.clip(upper=5)  # Clip for visualization
sns.heatmap(df_layers_clipped.T, annot=df_layers.T.round(3), fmt='.3f', cmap='RdYlGn_r',
            ax=axes[0], cbar_kws={'label': 'RMSE (log)'})
axes[0].set_title('Effect of Number of Layers\n(Best RMSE per config)', fontweight='bold')
axes[0].set_xlabel('Number of Layers')

# Plot 2: Hidden channels
df_hidden = pd.DataFrame(hidden_data).set_index('Hidden')
df_hidden_clipped = df_hidden.clip(upper=5)
sns.heatmap(df_hidden_clipped.T, annot=df_hidden.T.round(3), fmt='.3f', cmap='RdYlGn_r',
            ax=axes[1], cbar_kws={'label': 'RMSE (log)'})
axes[1].set_title('Effect of Hidden Dimensions\n(Best RMSE per config)', fontweight='bold')
axes[1].set_xlabel('Hidden Dimensions')

# Plot 3: Learning rate
df_lr = pd.DataFrame(lr_data).set_index('LR')
df_lr_clipped = df_lr.clip(upper=5)
sns.heatmap(df_lr_clipped.T, annot=df_lr.T.round(3), fmt='.3f', cmap='RdYlGn_r',
            ax=axes[2], cbar_kws={'label': 'RMSE (log)'})
axes[2].set_title('Effect of Learning Rate\n(Best RMSE per config)', fontweight='bold')
axes[2].set_xlabel('Learning Rate')

plt.tight_layout()
plt.savefig(output_dir / 'hyperparameter_sensitivity_analysis.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {output_dir / 'hyperparameter_sensitivity_analysis.png'}")

# ============================================================================
# Figure 4: Architecture Selection Summary
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Summary data for architecture selection
summary_data = {
    'Architecture': ['Graph', 'GCN', 'TAG', 'GIN', 'SGC', 'Transformer', 'GAT', 'SAGE'],
    'Avg_Rank': [2.3, 2.0, 2.3, 6.0, 3.7, 6.7, 7.0, 6.0],
    'Stability': ['High', 'High', 'Medium', 'Medium', 'High', 'Medium', 'Low', 'Low'],
    'Training_Speed': ['Fast', 'Fast', 'Medium', 'Slow', 'Fast', 'Slow', 'Slow', 'Medium']
}

df_summary = pd.DataFrame(summary_data)
df_summary = df_summary.sort_values('Avg_Rank')

# Create color mapping for stability
stability_colors = {'High': 'forestgreen', 'Medium': 'orange', 'Low': 'red'}
colors = [stability_colors[s] for s in df_summary['Stability']]

bars = ax.barh(df_summary['Architecture'], df_summary['Avg_Rank'], color=colors,
               edgecolor='black', linewidth=0.5)

ax.set_xlabel('Average Rank Across Datasets (Lower is Better)')
ax.set_title('GNN Architecture Selection Summary\n(Color indicates training stability)', fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='forestgreen', label='High Stability'),
                   Patch(facecolor='orange', label='Medium Stability'),
                   Patch(facecolor='red', label='Low Stability')]
ax.legend(handles=legend_elements, loc='lower right')

# Add rank labels
for bar, rank in zip(bars, df_summary['Avg_Rank']):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{rank:.1f}', va='center', fontsize=10)

# Highlight selected architecture
for bar, arch in zip(bars, df_summary['Architecture']):
    if arch == 'GCN':
        bar.set_edgecolor('gold')
        bar.set_linewidth(3)
        ax.annotate('Selected for\nHPO Benchmark', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   xytext=(4, 0.5), fontsize=9, fontweight='bold', color='darkblue',
                   arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5))

ax.set_xlim(0, 9)
plt.tight_layout()
plt.savefig(output_dir / 'gnn_architecture_selection_summary.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {output_dir / 'gnn_architecture_selection_summary.png'}")

print("\n" + "="*60)
print("GNN Architecture Comparison Figures Generated Successfully!")
print("="*60)
print(f"\nOutput directory: {output_dir.absolute()}")
print("\nGenerated figures:")
print("  1. gnn_architecture_comparison.png - Bar chart of best RMSE per architecture")
print("  2. gnn_architecture_stability.png - Multi-dataset performance comparison")
print("  3. hyperparameter_sensitivity_analysis.png - Heatmaps for layers/hidden/LR")
print("  4. gnn_architecture_selection_summary.png - Summary with stability ratings")
