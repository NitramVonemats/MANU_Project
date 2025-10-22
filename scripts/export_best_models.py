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

datasets = ['Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']

# Креирај топ 20 модели за секој dataset
for dataset in datasets:
    df = combined_df[combined_df['dataset_name'] == dataset].copy()
    if len(df) == 0:
        continue

    # Топ 20 според test_rmse
    top_models = df.nsmallest(20, 'test_rmse')

    # Селектирај релевантни колони
    columns_to_export = [
        'model_name', 'model_version', 'dataset_name',
        'val_rmse', 'test_rmse', 'test_r2',
        'graph_layers', 'graph_hidden_channels',
        'learning_rate', 'dropout', 'use_edge_features',
        'train_time', 'epochs'
    ]

    # Проверка кои колони постојат
    available_columns = [col for col in columns_to_export if col in top_models.columns]
    export_df = top_models[available_columns].copy()

    # Сортирај по test_rmse
    export_df = export_df.sort_values('test_rmse')

    # Зачувај во CSV
    filename = f'best_models_{dataset}.csv'
    export_df.to_csv(filename, index=False)
    print(f"Зачуван: {filename} ({len(export_df)} модели)")

# Креирај summary табела за сите datasets
summary_rows = []

for dataset in datasets:
    df = combined_df[combined_df['dataset_name'] == dataset].copy()
    if len(df) == 0:
        continue

    # Најдобар модел
    best_idx = df['test_rmse'].idxmin()
    best = df.loc[best_idx]

    summary_rows.append({
        'Dataset': dataset,
        'Best_Model': best['model_name'],
        'Model_Version': best['model_version'],
        'Val_RMSE': round(best['val_rmse'], 4),
        'Test_RMSE': round(best['test_rmse'], 4),
        'Test_R2': round(best['test_r2'], 4),
        'Graph_Layers': int(best['graph_layers']),
        'Hidden_Channels': int(best['graph_hidden_channels']),
        'Learning_Rate': best['learning_rate'],
        'Dropout': best['dropout'] if pd.notna(best['dropout']) else None,
        'Use_Edge_Features': best['use_edge_features'] if pd.notna(best['use_edge_features']) else None,
        'Total_Experiments': len(df),
        'Mean_Test_RMSE': round(df['test_rmse'].mean(), 4),
        'Std_Test_RMSE': round(df['test_rmse'].std(), 4)
    })

    # Додај и топ 5 модели за секој dataset
    top_5 = df.nsmallest(5, 'test_rmse')
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        summary_rows.append({
            'Dataset': f"{dataset}_Top{i}",
            'Best_Model': row['model_name'],
            'Model_Version': row['model_version'],
            'Val_RMSE': round(row['val_rmse'], 4),
            'Test_RMSE': round(row['test_rmse'], 4),
            'Test_R2': round(row['test_r2'], 4),
            'Graph_Layers': int(row['graph_layers']),
            'Hidden_Channels': int(row['graph_hidden_channels']),
            'Learning_Rate': row['learning_rate'],
            'Dropout': row['dropout'] if pd.notna(row['dropout']) else None,
            'Use_Edge_Features': row['use_edge_features'] if pd.notna(row['use_edge_features']) else None,
            'Total_Experiments': None,
            'Mean_Test_RMSE': None,
            'Std_Test_RMSE': None
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('SUMMARY_BEST_MODELS.csv', index=False)
print(f"\nЗачуван: SUMMARY_BEST_MODELS.csv")

# Креирај статистички преглед
stats_rows = []

for dataset in datasets:
    df = combined_df[combined_df['dataset_name'] == dataset].copy()
    if len(df) == 0:
        continue

    # Статистики по model_name
    for model_name in df['model_name'].unique():
        model_data = df[df['model_name'] == model_name]

        stats_rows.append({
            'Dataset': dataset,
            'Model_Name': model_name,
            'N_Experiments': len(model_data),
            'Mean_Test_RMSE': round(model_data['test_rmse'].mean(), 4),
            'Std_Test_RMSE': round(model_data['test_rmse'].std(), 4),
            'Min_Test_RMSE': round(model_data['test_rmse'].min(), 4),
            'Max_Test_RMSE': round(model_data['test_rmse'].max(), 4),
            'Mean_Test_R2': round(model_data['test_r2'].mean(), 4),
            'Max_Test_R2': round(model_data['test_r2'].max(), 4)
        })

stats_df = pd.DataFrame(stats_rows)
stats_df = stats_df.sort_values(['Dataset', 'Min_Test_RMSE'])
stats_df.to_csv('MODEL_STATISTICS.csv', index=False)
print(f"Зачуван: MODEL_STATISTICS.csv")

# Креирај хиперпараметар статистики
hyperparam_rows = []

for dataset in datasets:
    df = combined_df[combined_df['dataset_name'] == dataset].copy()
    if len(df) == 0:
        continue

    # Layers
    for layers in sorted(df['graph_layers'].unique()):
        layer_data = df[df['graph_layers'] == layers]
        hyperparam_rows.append({
            'Dataset': dataset,
            'Hyperparameter': 'graph_layers',
            'Value': layers,
            'N_Experiments': len(layer_data),
            'Mean_Test_RMSE': round(layer_data['test_rmse'].mean(), 4),
            'Min_Test_RMSE': round(layer_data['test_rmse'].min(), 4)
        })

    # Hidden channels
    for hidden in sorted(df['graph_hidden_channels'].unique()):
        hidden_data = df[df['graph_hidden_channels'] == hidden]
        hyperparam_rows.append({
            'Dataset': dataset,
            'Hyperparameter': 'graph_hidden_channels',
            'Value': hidden,
            'N_Experiments': len(hidden_data),
            'Mean_Test_RMSE': round(hidden_data['test_rmse'].mean(), 4),
            'Min_Test_RMSE': round(hidden_data['test_rmse'].min(), 4)
        })

    # Learning rate
    for lr in sorted(df['learning_rate'].unique()):
        lr_data = df[df['learning_rate'] == lr]
        hyperparam_rows.append({
            'Dataset': dataset,
            'Hyperparameter': 'learning_rate',
            'Value': lr,
            'N_Experiments': len(lr_data),
            'Mean_Test_RMSE': round(lr_data['test_rmse'].mean(), 4),
            'Min_Test_RMSE': round(lr_data['test_rmse'].min(), 4)
        })

    # Dropout
    dropout_data = df[df['dropout'].notna()]
    for dropout in sorted(dropout_data['dropout'].unique()):
        d_data = dropout_data[dropout_data['dropout'] == dropout]
        hyperparam_rows.append({
            'Dataset': dataset,
            'Hyperparameter': 'dropout',
            'Value': dropout,
            'N_Experiments': len(d_data),
            'Mean_Test_RMSE': round(d_data['test_rmse'].mean(), 4),
            'Min_Test_RMSE': round(d_data['test_rmse'].min(), 4)
        })

    # Edge features
    edge_data = df[df['use_edge_features'].notna()]
    for use_edge in [True, False]:
        e_data = edge_data[edge_data['use_edge_features'] == use_edge]
        if len(e_data) > 0:
            hyperparam_rows.append({
                'Dataset': dataset,
                'Hyperparameter': 'use_edge_features',
                'Value': use_edge,
                'N_Experiments': len(e_data),
                'Mean_Test_RMSE': round(e_data['test_rmse'].mean(), 4),
                'Min_Test_RMSE': round(e_data['test_rmse'].min(), 4)
            })

hyperparam_df = pd.DataFrame(hyperparam_rows)
hyperparam_df.to_csv('HYPERPARAMETER_ANALYSIS.csv', index=False)
print(f"Зачуван: HYPERPARAMETER_ANALYSIS.csv")

print("\n" + "="*100)
print("КОМПЛЕТИРАНО!")
print("="*100)
print("\nКреирани фајлови:")
print("1. best_models_Half_Life_Obach.csv - Топ 20 модели за Half_Life_Obach")
print("2. best_models_Clearance_Hepatocyte_AZ.csv - Топ 20 модели за Clearance_Hepatocyte_AZ")
print("3. best_models_Clearance_Microsome_AZ.csv - Топ 20 модели за Clearance_Microsome_AZ")
print("4. SUMMARY_BEST_MODELS.csv - Резиме на најдобри модели за сите datasets")
print("5. MODEL_STATISTICS.csv - Статистички преглед по модел и dataset")
print("6. HYPERPARAMETER_ANALYSIS.csv - Анализа на влијание на хиперпараметри")
print("\nФајловите се зачувани во: C:\\Users\\Martin.DESKTOP-J36C0SU\\Desktop\\MANU-master\\")
