import pandas as pd
import numpy as np
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

# Најди ги сите CSV фајлови
csv_files = glob.glob('GNN_test/**/*.csv', recursive=True)
print(f"Пронајдени {len(csv_files)} CSV фајлови\n")

# Вчитај ги сите податоци
all_data = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        # Извлечи информации од името на фајлот
        filename = Path(file).stem
        parts = filename.split('_')

        # Најди го model_version (се до dataset_name или датумот)
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

        # Извлечи dataset_name
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
        df['source_file'] = Path(file).name
        all_data.append(df)
        print(f"Вчитан: {Path(file).name} - {len(df)} записи")
    except Exception as e:
        print(f"Грешка при вчитување на {file}: {e}")

# Комбинирај ги сите податоци
combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n{'='*100}")
print(f"Вкупно вчитани {len(combined_df)} записи од {len(csv_files)} фајлови")
print(f"{'='*100}\n")

# Филтрирај ги само валидните податоци (без ALL dataset)
datasets = ['Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']

print("\n" + "="*100)
print("ДЕТАЛЕН ИЗВЕШТАЈ ЗА АНАЛИЗА НА GNN МОДЕЛИ")
print("="*100)

for dataset in datasets:
    print(f"\n{'#'*100}")
    print(f"DATASET: {dataset}")
    print(f"{'#'*100}\n")

    # Филтрирај податоци за овој dataset
    df = combined_df[combined_df['dataset_name'] == dataset].copy()

    if len(df) == 0:
        print(f"Нема податоци за {dataset}")
        continue

    print(f"Вкупно {len(df)} експерименти за {dataset}\n")

    # 1. ТОП 5 МОДЕЛИ СПОРЕД VAL_RMSE
    print(f"\n{'─'*100}")
    print("1. ТОП 5 МОДЕЛИ СПОРЕД VAL_RMSE (пониски вредности се подобри)")
    print(f"{'─'*100}")
    if 'val_rmse' in df.columns:
        top_val = df.nsmallest(5, 'val_rmse')[['model_name', 'model_version', 'val_rmse', 'test_rmse',
                                                 'test_r2', 'graph_layers', 'graph_hidden_channels',
                                                 'learning_rate', 'dropout', 'use_edge_features']]
        for idx, row in top_val.iterrows():
            print(f"\n{list(top_val.index).index(idx) + 1}. {row['model_name']} ({row['model_version']})")
            print(f"   Val RMSE: {row['val_rmse']:.4f} | Test RMSE: {row['test_rmse']:.4f} | Test R²: {row['test_r2']:.4f}")
            print(f"   Layers: {row['graph_layers']} | Hidden: {row['graph_hidden_channels']} | LR: {row['learning_rate']} | Dropout: {row['dropout']} | Edge Features: {row['use_edge_features']}")

    # 2. ТОП 5 МОДЕЛИ СПОРЕД TEST_RMSE
    print(f"\n{'─'*100}")
    print("2. ТОП 5 МОДЕЛИ СПОРЕД TEST_RMSE (пониски вредности се подобри)")
    print(f"{'─'*100}")
    if 'test_rmse' in df.columns:
        top_test = df.nsmallest(5, 'test_rmse')[['model_name', 'model_version', 'val_rmse', 'test_rmse',
                                                   'test_r2', 'graph_layers', 'graph_hidden_channels',
                                                   'learning_rate', 'dropout', 'use_edge_features']]
        for idx, row in top_test.iterrows():
            print(f"\n{list(top_test.index).index(idx) + 1}. {row['model_name']} ({row['model_version']})")
            print(f"   Val RMSE: {row['val_rmse']:.4f} | Test RMSE: {row['test_rmse']:.4f} | Test R²: {row['test_r2']:.4f}")
            print(f"   Layers: {row['graph_layers']} | Hidden: {row['graph_hidden_channels']} | LR: {row['learning_rate']} | Dropout: {row['dropout']} | Edge Features: {row['use_edge_features']}")

    # 3. ТОП 5 МОДЕЛИ СПОРЕД TEST_R2
    print(f"\n{'─'*100}")
    print("3. ТОП 5 МОДЕЛИ СПОРЕД TEST_R² (повисоки вредности се подобри)")
    print(f"{'─'*100}")
    if 'test_r2' in df.columns:
        top_r2 = df.nlargest(5, 'test_r2')[['model_name', 'model_version', 'val_rmse', 'test_rmse',
                                             'test_r2', 'graph_layers', 'graph_hidden_channels',
                                             'learning_rate', 'dropout', 'use_edge_features']]
        for idx, row in top_r2.iterrows():
            print(f"\n{list(top_r2.index).index(idx) + 1}. {row['model_name']} ({row['model_version']})")
            print(f"   Val RMSE: {row['val_rmse']:.4f} | Test RMSE: {row['test_rmse']:.4f} | Test R²: {row['test_r2']:.4f}")
            print(f"   Layers: {row['graph_layers']} | Hidden: {row['graph_hidden_channels']} | LR: {row['learning_rate']} | Dropout: {row['dropout']} | Edge Features: {row['use_edge_features']}")

    # 4. СПОРЕДБА НА ВЕРЗИИ НА МОДЕЛОТ
    print(f"\n{'─'*100}")
    print("4. СПОРЕДБА НА ВЕРЗИИ НА МОДЕЛОТ")
    print(f"{'─'*100}")
    version_stats = df.groupby('model_version').agg({
        'val_rmse': ['mean', 'std', 'min'],
        'test_rmse': ['mean', 'std', 'min'],
        'test_r2': ['mean', 'std', 'max']
    }).round(4)

    for version in df['model_version'].unique():
        print(f"\n{version}:")
        v_data = df[df['model_version'] == version]
        print(f"  Број експерименти: {len(v_data)}")
        print(f"  Val RMSE:  mean={v_data['val_rmse'].mean():.4f} ± {v_data['val_rmse'].std():.4f}, min={v_data['val_rmse'].min():.4f}")
        print(f"  Test RMSE: mean={v_data['test_rmse'].mean():.4f} ± {v_data['test_rmse'].std():.4f}, min={v_data['test_rmse'].min():.4f}")
        print(f"  Test R²:   mean={v_data['test_r2'].mean():.4f} ± {v_data['test_r2'].std():.4f}, max={v_data['test_r2'].max():.4f}")

    # 5. СТАТИСТИКИ ПО MODEL_NAME (GCN, SAGE, итн.)
    print(f"\n{'─'*100}")
    print("5. СТАТИСТИКИ ПО MODEL_NAME (GCN, SAGE, GIN, GAT, TAG, SGC, Transformer, Graph)")
    print(f"{'─'*100}")
    if 'model_name' in df.columns:
        for model in sorted(df['model_name'].unique()):
            m_data = df[df['model_name'] == model]
            print(f"\n{model}:")
            print(f"  Број експерименти: {len(m_data)}")
            print(f"  Val RMSE:  mean={m_data['val_rmse'].mean():.4f} ± {m_data['val_rmse'].std():.4f}, min={m_data['val_rmse'].min():.4f}")
            print(f"  Test RMSE: mean={m_data['test_rmse'].mean():.4f} ± {m_data['test_rmse'].std():.4f}, min={m_data['test_rmse'].min():.4f}")
            print(f"  Test R²:   mean={m_data['test_r2'].mean():.4f} ± {m_data['test_r2'].std():.4f}, max={m_data['test_r2'].max():.4f}")

            # Најдобри хиперпараметри за овој модел
            best_idx = m_data['test_rmse'].idxmin()
            best = m_data.loc[best_idx]
            print(f"  Најдобри хиперпараметри:")
            print(f"    Layers: {best['graph_layers']}, Hidden: {best['graph_hidden_channels']}, LR: {best['learning_rate']}, Dropout: {best['dropout']}, Edge Features: {best['use_edge_features']}")

    # 6. ВЛИЈАНИЕ НА EDGE FEATURES
    print(f"\n{'─'*100}")
    print("6. ВЛИЈАНИЕ НА EDGE FEATURES (use_edge_features)")
    print(f"{'─'*100}")
    if 'use_edge_features' in df.columns:
        for use_edge in [True, False]:
            edge_data = df[df['use_edge_features'] == use_edge]
            if len(edge_data) > 0:
                print(f"\nuse_edge_features = {use_edge}:")
                print(f"  Број експерименти: {len(edge_data)}")
                print(f"  Val RMSE:  mean={edge_data['val_rmse'].mean():.4f} ± {edge_data['val_rmse'].std():.4f}, min={edge_data['val_rmse'].min():.4f}")
                print(f"  Test RMSE: mean={edge_data['test_rmse'].mean():.4f} ± {edge_data['test_rmse'].std():.4f}, min={edge_data['test_rmse'].min():.4f}")
                print(f"  Test R²:   mean={edge_data['test_r2'].mean():.4f} ± {edge_data['test_r2'].std():.4f}, max={edge_data['test_r2'].max():.4f}")

    # 7. ОПТИМАЛНИ ХИПЕРПАРАМЕТРИ
    print(f"\n{'─'*100}")
    print("7. АНАЛИЗА НА ХИПЕРПАРАМЕТРИ")
    print(f"{'─'*100}")

    # Graph layers
    if 'graph_layers' in df.columns:
        print("\nВлијание на graph_layers:")
        for layers in sorted(df['graph_layers'].unique()):
            layer_data = df[df['graph_layers'] == layers]
            print(f"  Layers={layers}: n={len(layer_data)}, Test RMSE mean={layer_data['test_rmse'].mean():.4f}, min={layer_data['test_rmse'].min():.4f}")

    # Hidden channels
    if 'graph_hidden_channels' in df.columns:
        print("\nВлијание на graph_hidden_channels:")
        for hidden in sorted(df['graph_hidden_channels'].unique()):
            hidden_data = df[df['graph_hidden_channels'] == hidden]
            print(f"  Hidden={hidden}: n={len(hidden_data)}, Test RMSE mean={hidden_data['test_rmse'].mean():.4f}, min={hidden_data['test_rmse'].min():.4f}")

    # Learning rate
    if 'learning_rate' in df.columns:
        print("\nВлијание на learning_rate:")
        for lr in sorted(df['learning_rate'].unique()):
            lr_data = df[df['learning_rate'] == lr]
            print(f"  LR={lr}: n={len(lr_data)}, Test RMSE mean={lr_data['test_rmse'].mean():.4f}, min={lr_data['test_rmse'].min():.4f}")

    # Dropout
    if 'dropout' in df.columns:
        print("\nВлијание на dropout:")
        for dropout in sorted(df['dropout'].unique()):
            dropout_data = df[df['dropout'] == dropout]
            print(f"  Dropout={dropout}: n={len(dropout_data)}, Test RMSE mean={dropout_data['test_rmse'].mean():.4f}, min={dropout_data['test_rmse'].min():.4f}")

# ФИНАЛЕН РЕЗИМЕ
print(f"\n\n{'='*100}")
print("ФИНАЛЕН РЕЗИМЕ И ПРЕПОРАКИ")
print(f"{'='*100}\n")

for dataset in datasets:
    df = combined_df[combined_df['dataset_name'] == dataset].copy()
    if len(df) == 0:
        continue

    print(f"\n{dataset}:")
    print("-" * 80)

    # Најдобар модел вкупно
    best_idx = df['test_rmse'].idxmin()
    best = df.loc[best_idx]
    print(f"\nАПСОЛУТНО НАЈДОБАР МОДЕЛ:")
    print(f"  Модел: {best['model_name']} ({best['model_version']})")
    print(f"  Перформанси: Val RMSE={best['val_rmse']:.4f}, Test RMSE={best['test_rmse']:.4f}, Test R²={best['test_r2']:.4f}")
    print(f"  Хиперпараметри: Layers={best['graph_layers']}, Hidden={best['graph_hidden_channels']}, LR={best['learning_rate']}, Dropout={best['dropout']}, Edge Features={best['use_edge_features']}")

    # Која верзија е најдобра
    best_version = df.groupby('model_version')['test_rmse'].mean().idxmin()
    print(f"\nНАЈДОБРА ВЕРЗИЈА НА МОДЕЛОТ: {best_version}")

    # Кој model_name е најдобар
    best_model_name = df.groupby('model_name')['test_rmse'].mean().idxmin()
    print(f"НАЈДОБАР MODEL_NAME: {best_model_name}")

    # Дали edge features помагаат
    if 'use_edge_features' in df.columns:
        edge_true_rmse = df[df['use_edge_features'] == True]['test_rmse'].mean()
        edge_false_rmse = df[df['use_edge_features'] == False]['test_rmse'].mean()
        if edge_true_rmse < edge_false_rmse:
            print(f"EDGE FEATURES: Препорачано use_edge_features=True (RMSE: {edge_true_rmse:.4f} vs {edge_false_rmse:.4f})")
        else:
            print(f"EDGE FEATURES: Препорачано use_edge_features=False (RMSE: {edge_false_rmse:.4f} vs {edge_true_rmse:.4f})")

    # Оптимални хиперпараметри (базирано на просек)
    best_layers = df.groupby('graph_layers')['test_rmse'].mean().idxmin()
    best_hidden = df.groupby('graph_hidden_channels')['test_rmse'].mean().idxmin()
    best_lr = df.groupby('learning_rate')['test_rmse'].mean().idxmin()
    best_dropout = df.groupby('dropout')['test_rmse'].mean().idxmin()

    print(f"\nОПТИМАЛНИ ХИПЕРПАРАМЕТРИ (базирано на просечни вредности):")
    print(f"  graph_layers: {best_layers}")
    print(f"  graph_hidden_channels: {best_hidden}")
    print(f"  learning_rate: {best_lr}")
    print(f"  dropout: {best_dropout}")

print(f"\n{'='*100}")
print("КРАЈ НА ИЗВЕШТАЈОТ")
print(f"{'='*100}\n")
