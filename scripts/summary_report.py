import pandas as pd
import numpy as np
import glob

# Вчитај ги сите податоци
csv_files = glob.glob('GNN_test/**/*.csv', recursive=True)
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

        df['model_version'] = model_version
        df['dataset_name'] = dataset_name
        all_data.append(df)
    except:
        pass

combined_df = pd.concat(all_data, ignore_index=True)

# Креирај резиме табели
print("\n" + "="*120)
print("EXECUTIVE SUMMARY - НАЈДОБРИ РЕЗУЛТАТИ ЗА СЕКОЈ DATASET")
print("="*120 + "\n")

datasets = ['Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']

summary_data = []

for dataset in datasets:
    df = combined_df[combined_df['dataset_name'] == dataset].copy()
    if len(df) == 0:
        continue

    # Најдобар модел
    best_idx = df['test_rmse'].idxmin()
    best = df.loc[best_idx]

    summary_data.append({
        'Dataset': dataset,
        'Best Model': best['model_name'],
        'Model Version': best['model_version'],
        'Test RMSE': f"{best['test_rmse']:.4f}",
        'Test R²': f"{best['test_r2']:.4f}",
        'Val RMSE': f"{best['val_rmse']:.4f}",
        'Layers': int(best['graph_layers']),
        'Hidden': int(best['graph_hidden_channels']),
        'LR': best['learning_rate'],
        'Dropout': best['dropout'] if pd.notna(best['dropout']) else 'N/A',
        'Edge Features': best['use_edge_features'] if pd.notna(best['use_edge_features']) else 'N/A'
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*120)
print("СПОРЕДБА НА MODEL ARCHITECTURES (просечни перформанси)")
print("="*120 + "\n")

for dataset in datasets:
    df = combined_df[combined_df['dataset_name'] == dataset].copy()
    if len(df) == 0:
        continue

    print(f"\n{dataset}:")
    print("-" * 100)

    model_comparison = df.groupby('model_name').agg({
        'test_rmse': ['mean', 'min', 'count'],
        'test_r2': ['mean', 'max']
    }).round(4)

    model_comparison.columns = ['Test RMSE (mean)', 'Test RMSE (min)', 'N experiments', 'Test R² (mean)', 'Test R² (max)']
    model_comparison = model_comparison.sort_values('Test RMSE (min)')

    print(model_comparison.to_string())

print("\n" + "="*120)
print("СПОРЕДБА НА MODEL VERSIONS")
print("="*120 + "\n")

for dataset in datasets:
    df = combined_df[combined_df['dataset_name'] == dataset].copy()
    if len(df) == 0:
        continue

    print(f"\n{dataset}:")
    print("-" * 100)

    version_comparison = df.groupby('model_version').agg({
        'test_rmse': ['mean', 'min', 'count'],
        'test_r2': ['mean', 'max']
    }).round(4)

    version_comparison.columns = ['Test RMSE (mean)', 'Test RMSE (min)', 'N experiments', 'Test R² (mean)', 'Test R² (max)']
    version_comparison = version_comparison.sort_values('Test RMSE (min)')

    print(version_comparison.to_string())

print("\n" + "="*120)
print("ВЛИЈАНИЕ НА ХИПЕРПАРАМЕТРИ - ОПТИМАЛНИ ВРЕДНОСТИ")
print("="*120 + "\n")

for dataset in datasets:
    df = combined_df[combined_df['dataset_name'] == dataset].copy()
    if len(df) == 0:
        continue

    print(f"\n{dataset}:")
    print("-" * 100)

    # Graph Layers
    if 'graph_layers' in df.columns:
        layers_stats = df.groupby('graph_layers')['test_rmse'].agg(['mean', 'min', 'count']).sort_values('mean')
        print("\nGraph Layers:")
        print(layers_stats.to_string())

    # Hidden Channels
    if 'graph_hidden_channels' in df.columns:
        hidden_stats = df.groupby('graph_hidden_channels')['test_rmse'].agg(['mean', 'min', 'count']).sort_values('mean')
        print("\nHidden Channels:")
        print(hidden_stats.to_string())

    # Learning Rate
    if 'learning_rate' in df.columns:
        lr_stats = df.groupby('learning_rate')['test_rmse'].agg(['mean', 'min', 'count']).sort_values('mean')
        print("\nLearning Rate:")
        print(lr_stats.to_string())

    # Dropout
    if 'dropout' in df.columns:
        dropout_stats = df[df['dropout'].notna()].groupby('dropout')['test_rmse'].agg(['mean', 'min', 'count']).sort_values('mean')
        print("\nDropout:")
        print(dropout_stats.to_string())

    # Edge Features
    if 'use_edge_features' in df.columns:
        edge_stats = df[df['use_edge_features'].notna()].groupby('use_edge_features')['test_rmse'].agg(['mean', 'min', 'count']).sort_values('mean')
        print("\nEdge Features:")
        print(edge_stats.to_string())

print("\n" + "="*120)
print("КЛУЧНИ ПРЕПОРАКИ ЗА СЕКОЈ DATASET")
print("="*120 + "\n")

for dataset in datasets:
    df = combined_df[combined_df['dataset_name'] == dataset].copy()
    if len(df) == 0:
        continue

    print(f"\n{dataset}:")
    print("-" * 100)

    # Најдобар модел
    best_idx = df['test_rmse'].idxmin()
    best = df.loc[best_idx]
    print(f"\n1. НАЈДОБАР МОДЕЛ: {best['model_name']} ({best['model_version']})")
    print(f"   - Test RMSE: {best['test_rmse']:.4f}")
    print(f"   - Test R²: {best['test_r2']:.4f}")
    print(f"   - Конфигурација: Layers={best['graph_layers']}, Hidden={best['graph_hidden_channels']}, LR={best['learning_rate']}, Dropout={best['dropout']}")

    # Најдобра верзија
    best_version = df.groupby('model_version')['test_rmse'].mean().idxmin()
    best_version_rmse = df.groupby('model_version')['test_rmse'].mean()[best_version]
    print(f"\n2. НАЈДОБРА ВЕРЗИЈА: {best_version}")
    print(f"   - Просечен Test RMSE: {best_version_rmse:.4f}")

    # Најдобар architecture
    best_arch = df.groupby('model_name')['test_rmse'].mean().idxmin()
    best_arch_rmse = df.groupby('model_name')['test_rmse'].mean()[best_arch]
    print(f"\n3. НАЈДОБАР ARCHITECTURE: {best_arch}")
    print(f"   - Просечен Test RMSE: {best_arch_rmse:.4f}")

    # Edge Features препорака
    if 'use_edge_features' in df.columns:
        edge_comparison = df[df['use_edge_features'].notna()].groupby('use_edge_features')['test_rmse'].mean()
        if len(edge_comparison) > 1:
            best_edge = edge_comparison.idxmin()
            print(f"\n4. EDGE FEATURES: {'Препорачано' if best_edge else 'Не препорачано'}")
            print(f"   - With Edge Features: {edge_comparison.get(True, 'N/A'):.4f}" if True in edge_comparison else "   - With Edge Features: N/A")
            print(f"   - Without Edge Features: {edge_comparison.get(False, 'N/A'):.4f}" if False in edge_comparison else "   - Without Edge Features: N/A")

    # Оптимални хиперпараметри
    best_layers = df.groupby('graph_layers')['test_rmse'].mean().idxmin()
    best_hidden = df.groupby('graph_hidden_channels')['test_rmse'].mean().idxmin()
    best_lr = df.groupby('learning_rate')['test_rmse'].mean().idxmin()

    dropout_vals = df[df['dropout'].notna()]
    if len(dropout_vals) > 0:
        best_dropout = dropout_vals.groupby('dropout')['test_rmse'].mean().idxmin()
    else:
        best_dropout = 'N/A'

    print(f"\n5. ОПТИМАЛНИ ХИПЕРПАРАМЕТРИ (базирано на просек):")
    print(f"   - Layers: {best_layers}")
    print(f"   - Hidden Channels: {best_hidden}")
    print(f"   - Learning Rate: {best_lr}")
    print(f"   - Dropout: {best_dropout}")

print("\n" + "="*120)
print("ЗАКЛУЧОК")
print("="*120 + "\n")

print("Врз основа на анализата на 565 експерименти од 4 различни верзии на моделот,")
print("следниве се клучните заклучоци:\n")

for dataset in datasets:
    df = combined_df[combined_df['dataset_name'] == dataset].copy()
    if len(df) == 0:
        continue

    best_idx = df['test_rmse'].idxmin()
    best = df.loc[best_idx]

    print(f"\n{dataset}:")
    print(f"  - Најдобар резултат: {best['model_name']} со Test RMSE={best['test_rmse']:.4f} и R²={best['test_r2']:.4f}")
    print(f"  - Препорачана конфигурација: {best['graph_layers']} layers, {best['graph_hidden_channels']} hidden channels, LR={best['learning_rate']}")

print("\n" + "="*120)
