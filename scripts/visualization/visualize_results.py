import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load data
csv_files = glob.glob('GNN_test/**/*.csv', recursive=True) + glob.glob('results/foundation_benchmark/*.csv', recursive=True)
all_data = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        from pathlib import Path
        filename = Path(file).stem

        if 'tdc_excretion_plus' in filename:
            model_version = 'tdc_excretion_plus'
            rest = filename.replace('tdc_excretion_plus_', '')
        elif 'final_fixed_molecular_gnn' in filename:
            model_version = 'final_fixed_molecular_gnn'
            rest = filename.replace('final_fixed_molecular_gnn_', '')
        elif 'enhanced_molecular_gnn' in filename:
            model_version = 'enhanced_molecular_gnn'
            rest = filename.replace('enhanced_molecular_gnn_', '')
        elif 'fixed_molecular_gnn' in filename:
            model_version = 'fixed_molecular_gnn'
            rest = filename.replace('fixed_molecular_gnn_', '')
        elif 'benchmark_results' in filename:
            model_version = 'foundation_benchmark'
            rest = filename.replace('benchmark_results_', '')
        else:
            model_version = 'unknown'
            rest = filename

        if 'Half_Life_Obach' in rest:
            dataset_name = 'Half_Life_Obach'
        elif 'Clearance_Hepatocyte_AZ' in rest:
            dataset_name = 'Clearance_Hepatocyte_AZ'
        elif 'Clearance_Microsome_AZ' in rest:
            dataset_name = 'Clearance_Microsome_AZ'
        elif 'ALL' in rest:
            dataset_name = 'ALL'
        else:
            dataset_name = 'unknown'

        if 'benchmark_results' in filename:
             # Foundation benchmark has different structure
             df['model_version'] = df['model'] # Use model name as version
             df['dataset_name'] = df['dataset']
             # Rename columns to match GNN results
             if 'test_rmse' not in df.columns and 'rmse' in df.columns:
                 df['test_rmse'] = df['rmse']
        else:
            df['model_version'] = model_version
            df['dataset_name'] = dataset_name
            
        all_data.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")
        pass

combined_df = pd.concat(all_data, ignore_index=True)
datasets = ['Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']

# Create visualizations
print("Креирање на визуелизации...")

# 1. Model Version Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Споредба на Model Versions - Test RMSE', fontsize=16, fontweight='bold')

for i, dataset in enumerate(datasets):
    df = combined_df[combined_df['dataset_name'] == dataset]

    # Filter outliers for better visualization
    df_filtered = df[df['test_rmse'] < df['test_rmse'].quantile(0.95)]

    version_data = df_filtered.groupby('model_version')['test_rmse'].mean().sort_values()

    axes[i].barh(range(len(version_data)), version_data.values)
    axes[i].set_yticks(range(len(version_data)))
    axes[i].set_yticklabels(version_data.index)
    axes[i].set_xlabel('Mean Test RMSE')
    axes[i].set_title(dataset)
    axes[i].grid(axis='x', alpha=0.3)

    # Highlight best version
    best_idx = version_data.idxmin()
    for j, version in enumerate(version_data.index):
        if version == best_idx:
            axes[i].get_children()[j].set_color('green')
            axes[i].get_children()[j].set_alpha(0.7)

plt.tight_layout()
plt.savefig('model_versions_comparison.png', dpi=300, bbox_inches='tight')
print("Зачувано: model_versions_comparison.png")
plt.close()

# 2. Model Architecture Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Споредба на Model Architectures - Test RMSE', fontsize=16, fontweight='bold')

for i, dataset in enumerate(datasets):
    df = combined_df[combined_df['dataset_name'] == dataset]

    # Filter tdc_excretion_plus only for fair comparison
    df_tdc = df[df['model_version'] == 'tdc_excretion_plus']

    if len(df_tdc) > 0:
        model_data = df_tdc.groupby('model_name')['test_rmse'].mean().sort_values()

        axes[i].barh(range(len(model_data)), model_data.values, color='skyblue')
        axes[i].set_yticks(range(len(model_data)))
        axes[i].set_yticklabels(model_data.index)
        axes[i].set_xlabel('Mean Test RMSE')
        axes[i].set_title(f"{dataset}\n(tdc_excretion_plus only)")
        axes[i].grid(axis='x', alpha=0.3)

        # Highlight best model
        axes[i].get_children()[0].set_color('green')
        axes[i].get_children()[0].set_alpha(0.7)

plt.tight_layout()
plt.savefig('model_architectures_comparison.png', dpi=300, bbox_inches='tight')
print("Зачувано: model_architectures_comparison.png")
plt.close()

# 3. Hyperparameter Analysis - Graph Layers
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Влијание на Graph Layers на Test RMSE', fontsize=16, fontweight='bold')

for i, dataset in enumerate(datasets):
    df = combined_df[combined_df['dataset_name'] == dataset]

    layers_data = df.groupby('graph_layers')['test_rmse'].agg(['mean', 'min', 'count']).sort_index()

    x = layers_data.index
    y_mean = layers_data['mean']
    y_min = layers_data['min']

    axes[i].plot(x, y_mean, marker='o', label='Mean', linewidth=2)
    axes[i].plot(x, y_min, marker='s', label='Min', linewidth=2)
    axes[i].set_xlabel('Number of Layers')
    axes[i].set_ylabel('Test RMSE')
    axes[i].set_title(dataset)
    axes[i].legend()
    axes[i].grid(alpha=0.3)

    # Add count annotations
    for j, (layer, row) in enumerate(layers_data.iterrows()):
        axes[i].annotate(f"n={int(row['count'])}",
                        (layer, y_mean.iloc[j]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8)

plt.tight_layout()
plt.savefig('graph_layers_analysis.png', dpi=300, bbox_inches='tight')
print("Зачувано: graph_layers_analysis.png")
plt.close()

# 4. Hyperparameter Analysis - Hidden Channels
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Влијание на Hidden Channels на Test RMSE', fontsize=16, fontweight='bold')

for i, dataset in enumerate(datasets):
    df = combined_df[combined_df['dataset_name'] == dataset]

    hidden_data = df.groupby('graph_hidden_channels')['test_rmse'].agg(['mean', 'min', 'count']).sort_index()

    x = hidden_data.index
    y_mean = hidden_data['mean']
    y_min = hidden_data['min']

    axes[i].plot(x, y_mean, marker='o', label='Mean', linewidth=2)
    axes[i].plot(x, y_min, marker='s', label='Min', linewidth=2)
    axes[i].set_xlabel('Hidden Channels')
    axes[i].set_ylabel('Test RMSE')
    axes[i].set_title(dataset)
    axes[i].legend()
    axes[i].grid(alpha=0.3)
    axes[i].set_xscale('log')

plt.tight_layout()
plt.savefig('hidden_channels_analysis.png', dpi=300, bbox_inches='tight')
print("Зачувано: hidden_channels_analysis.png")
plt.close()

# 5. Edge Features Impact
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Влијание на Edge Features на Test RMSE', fontsize=16, fontweight='bold')

for i, dataset in enumerate(datasets):
    df = combined_df[combined_df['dataset_name'] == dataset]
    df_edge = df[df['use_edge_features'].notna()]

    if len(df_edge) > 0:
        edge_data = df_edge.groupby('use_edge_features')['test_rmse'].agg(['mean', 'std', 'count'])

        x = ['Without Edge Features', 'With Edge Features']
        y = [edge_data.loc[False, 'mean'] if False in edge_data.index else 0,
             edge_data.loc[True, 'mean'] if True in edge_data.index else 0]
        yerr = [edge_data.loc[False, 'std'] if False in edge_data.index else 0,
                edge_data.loc[True, 'std'] if True in edge_data.index else 0]

        bars = axes[i].bar(x, y, yerr=yerr, capsize=5, alpha=0.7)
        bars[0].set_color('green')
        bars[1].set_color('red')

        axes[i].set_ylabel('Mean Test RMSE')
        axes[i].set_title(dataset)
        axes[i].grid(axis='y', alpha=0.3)

        # Add value labels
        for j, v in enumerate(y):
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('edge_features_impact.png', dpi=300, bbox_inches='tight')
print("Зачувано: edge_features_impact.png")
plt.close()

# 6. Val vs Test RMSE (Generalization)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Val RMSE vs Test RMSE - Генерализација', fontsize=16, fontweight='bold')

for i, dataset in enumerate(datasets):
    df = combined_df[combined_df['dataset_name'] == dataset]

    # Filter tdc_excretion_plus for cleaner plot
    df_tdc = df[df['model_version'] == 'tdc_excretion_plus']

    if len(df_tdc) > 0:
        axes[i].scatter(df_tdc['val_rmse'], df_tdc['test_rmse'], alpha=0.6, s=100)

        # Add diagonal line (perfect generalization)
        lims = [
            np.min([axes[i].get_xlim(), axes[i].get_ylim()]),
            np.max([axes[i].get_xlim(), axes[i].get_ylim()]),
        ]
        axes[i].plot(lims, lims, 'r--', alpha=0.5, zorder=0, label='Perfect Generalization')

        axes[i].set_xlabel('Val RMSE')
        axes[i].set_ylabel('Test RMSE')
        axes[i].set_title(f"{dataset}\n(tdc_excretion_plus only)")
        axes[i].legend()
        axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('val_vs_test_generalization.png', dpi=300, bbox_inches='tight')
print("Зачувано: val_vs_test_generalization.png")
plt.close()

# 7. Test RMSE vs Test R² Trade-off
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Test RMSE vs Test R² Trade-off', fontsize=16, fontweight='bold')

for i, dataset in enumerate(datasets):
    df = combined_df[combined_df['dataset_name'] == dataset]

    # Color by model version
    for version in df['model_version'].unique():
        df_v = df[df['model_version'] == version]
        axes[i].scatter(df_v['test_rmse'], df_v['test_r2'],
                       label=version, alpha=0.6, s=80)

    axes[i].set_xlabel('Test RMSE (lower is better)')
    axes[i].set_ylabel('Test R² (higher is better)')
    axes[i].set_title(dataset)
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)

    # Highlight best models
    best_rmse_idx = df['test_rmse'].idxmin()
    best_r2_idx = df['test_r2'].idxmax()

    axes[i].scatter(df.loc[best_rmse_idx, 'test_rmse'],
                   df.loc[best_rmse_idx, 'test_r2'],
                   s=200, facecolors='none', edgecolors='red', linewidths=2,
                   label='Best RMSE')

plt.tight_layout()
plt.savefig('rmse_vs_r2_tradeoff.png', dpi=300, bbox_inches='tight')
print("Зачувано: rmse_vs_r2_tradeoff.png")
plt.close()

# 8. Learning Rate Analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Влијание на Learning Rate на Test RMSE', fontsize=16, fontweight='bold')

for i, dataset in enumerate(datasets):
    df = combined_df[combined_df['dataset_name'] == dataset]

    lr_data = df.groupby('learning_rate')['test_rmse'].agg(['mean', 'min', 'count']).sort_index()

    x = lr_data.index
    y_mean = lr_data['mean']
    y_min = lr_data['min']

    axes[i].semilogx(x, y_mean, marker='o', label='Mean', linewidth=2)
    axes[i].semilogx(x, y_min, marker='s', label='Min', linewidth=2)
    axes[i].set_xlabel('Learning Rate')
    axes[i].set_ylabel('Test RMSE')
    axes[i].set_title(dataset)
    axes[i].legend()
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('learning_rate_analysis.png', dpi=300, bbox_inches='tight')
print("Зачувано: learning_rate_analysis.png")
plt.close()

# 9. Model Stability (Coefficient of Variation)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Стабилност на модели (Coefficient of Variation)', fontsize=16, fontweight='bold')

for i, dataset in enumerate(datasets):
    df = combined_df[combined_df['dataset_name'] == dataset]

    stability = df.groupby('model_name')['test_rmse'].agg(['mean', 'std']).copy()
    stability['cv'] = stability['std'] / stability['mean']
    stability = stability.sort_values('cv')

    axes[i].barh(range(len(stability)), stability['cv'].values, color='orange', alpha=0.7)
    axes[i].set_yticks(range(len(stability)))
    axes[i].set_yticklabels(stability.index)
    axes[i].set_xlabel('Coefficient of Variation (lower = more stable)')
    axes[i].set_title(dataset)
    axes[i].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_stability.png', dpi=300, bbox_inches='tight')
print("Зачувано: model_stability.png")
plt.close()

# Summary
print("\n" + "="*80)
print("КОМПЛЕТИРАНО!")
print("="*80)
print("\nКреирани визуелизации:")
print("1. model_versions_comparison.png - Споредба на model versions")
print("2. model_architectures_comparison.png - Споредба на model architectures")
print("3. graph_layers_analysis.png - Влијание на graph layers")
print("4. hidden_channels_analysis.png - Влијание на hidden channels")
print("5. edge_features_impact.png - Влијание на edge features")
print("6. val_vs_test_generalization.png - Генерализација (val vs test)")
print("7. rmse_vs_r2_tradeoff.png - RMSE vs R² trade-off")
print("8. learning_rate_analysis.png - Влијание на learning rate")
print("9. model_stability.png - Стабилност на модели")
print("\nФајловите се зачувани во: C:\\Users\\Martin.DESKTOP-J36C0SU\\Desktop\\MANU-master\\")
