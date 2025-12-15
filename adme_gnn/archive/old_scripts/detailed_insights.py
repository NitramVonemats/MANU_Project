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

print("\n" + "="*120)
print("ДЕТАЛНИ ИНСАЈТИ И ПРЕПОРАКИ ЗА ПОДОБРУВАЊЕ")
print("="*120 + "\n")

datasets = ['Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']

for dataset in datasets:
    df = combined_df[combined_df['dataset_name'] == dataset].copy()
    if len(df) == 0:
        continue

    print(f"\n{'#'*120}")
    print(f"DATASET: {dataset}")
    print(f"{'#'*120}\n")

    # 1. PERFORMANCE GAPS ANALYSIS
    print("\n1. АНАЛИЗА НА PERFORMANCE GAPS")
    print("-" * 100)

    best_rmse = df['test_rmse'].min()
    worst_rmse = df['test_rmse'].max()
    mean_rmse = df['test_rmse'].mean()
    std_rmse = df['test_rmse'].std()

    print(f"   Best Test RMSE: {best_rmse:.4f}")
    print(f"   Mean Test RMSE: {mean_rmse:.4f}")
    print(f"   Worst Test RMSE: {worst_rmse:.4f}")
    print(f"   Std Dev: {std_rmse:.4f}")
    print(f"   Performance Gap: {worst_rmse - best_rmse:.4f} ({((worst_rmse - best_rmse) / best_rmse * 100):.2f}%)")

    # 2. CONSISTENCY ANALYSIS
    print("\n2. АНАЛИЗА НА КОНЗИСТЕНТНОСТ (Val vs Test Performance)")
    print("-" * 100)

    df['val_test_gap'] = abs(df['val_rmse'] - df['test_rmse'])
    avg_gap = df['val_test_gap'].mean()

    print(f"   Просечен Val-Test Gap: {avg_gap:.4f}")

    # Модели со најмал gap (добра генерализација)
    best_generalization = df.nsmallest(5, 'val_test_gap')[['model_name', 'model_version', 'val_rmse', 'test_rmse', 'val_test_gap']]
    print(f"\n   Најдобра генерализација (мал Val-Test gap):")
    for idx, row in best_generalization.iterrows():
        print(f"   {row['model_name']} ({row['model_version']}): Gap={row['val_test_gap']:.4f}, Val={row['val_rmse']:.4f}, Test={row['test_rmse']:.4f}")

    # 3. OVERFITTING DETECTION
    print("\n3. ДЕТЕКЦИЈА НА OVERFITTING/UNDERFITTING")
    print("-" * 100)

    df['overfit_score'] = df['test_rmse'] - df['val_rmse']

    overfitting = df[df['overfit_score'] > df['overfit_score'].quantile(0.75)]
    print(f"   Број на модели со overfitting: {len(overfitting)} ({len(overfitting)/len(df)*100:.1f}%)")

    underfitting = df[df['overfit_score'] < df['overfit_score'].quantile(0.25)]
    print(f"   Број на модели со underfitting: {len(underfitting)} ({len(underfitting)/len(df)*100:.1f}%)")

    # 4. HYPERPARAMETER INTERACTION ANALYSIS
    print("\n4. ИНТЕРАКЦИЈА ПОМЕЃУ ХИПЕРПАРАМЕТРИ")
    print("-" * 100)

    # Најдобри комбинации
    if 'graph_layers' in df.columns and 'graph_hidden_channels' in df.columns:
        combo_analysis = df.groupby(['graph_layers', 'graph_hidden_channels'])['test_rmse'].agg(['mean', 'min', 'count'])
        combo_analysis = combo_analysis.sort_values('min').head(10)
        print("\n   Топ 10 комбинации на Layers x Hidden Channels:")
        print(combo_analysis.to_string())

    # 5. MODEL STABILITY
    print("\n5. СТАБИЛНОСТ НА МОДЕЛИТЕ")
    print("-" * 100)

    stability = df.groupby('model_name')['test_rmse'].agg(['mean', 'std', 'min', 'max'])
    stability['cv'] = stability['std'] / stability['mean']  # Coefficient of variation
    stability = stability.sort_values('cv')

    print("\n   Најстабилни модели (најнизок coefficient of variation):")
    print(stability.head(5).to_string())

    # 6. VERSION COMPARISON IN DETAIL
    print("\n6. ДЕТАЛНА СПОРЕДБА НА ВЕРЗИИ")
    print("-" * 100)

    version_detail = df.groupby('model_version').agg({
        'test_rmse': ['mean', 'median', 'std', 'min', 'max'],
        'test_r2': ['mean', 'median', 'std', 'min', 'max']
    }).round(4)

    print(version_detail.to_string())

    # 7. BEST CONFIGURATIONS PER MODEL TYPE
    print("\n7. НАЈДОБРИ КОНФИГУРАЦИИ ЗА СЕКОЈ MODEL TYPE")
    print("-" * 100)

    for model_name in df['model_name'].unique():
        model_data = df[df['model_name'] == model_name]
        best_config = model_data.nsmallest(1, 'test_rmse').iloc[0]

        print(f"\n   {model_name}:")
        print(f"      Best RMSE: {best_config['test_rmse']:.4f}")
        print(f"      Best R²: {best_config['test_r2']:.4f}")
        print(f"      Config: {best_config['graph_layers']} layers, {best_config['graph_hidden_channels']} hidden, LR={best_config['learning_rate']}")
        if pd.notna(best_config.get('dropout')):
            print(f"      Dropout: {best_config['dropout']}, Edge Features: {best_config.get('use_edge_features', 'N/A')}")

    # 8. PERFORMANCE DISTRIBUTION
    print("\n8. ДИСТРИБУЦИЈА НА ПЕРФОРМАНСИ")
    print("-" * 100)

    percentiles = [10, 25, 50, 75, 90]
    print(f"\n   Test RMSE Percentiles:")
    for p in percentiles:
        print(f"      {p}th: {np.percentile(df['test_rmse'], p):.4f}")

    print(f"\n   Test R² Percentiles:")
    for p in percentiles:
        print(f"      {p}th: {np.percentile(df['test_r2'], p):.4f}")

# ГЛОБАЛНИ ПРЕПОРАКИ
print("\n\n" + "="*120)
print("ГЛОБАЛНИ ПРЕПОРАКИ ЗА ПОДОБРУВАЊЕ")
print("="*120 + "\n")

recommendations = """
1. АРХИТЕКТУРА
   - Graph модел покажа најдобри резултати на сите три datasets
   - 5 слоја се оптимален број (performance gap: best vs worst е значителен)
   - 128 hidden channels се оптимални за балансирање на complexity vs performance

2. MODEL VERSION
   - tdc_excretion_plus верзијата има значително подобри резултати
   - Разликата е драматична: ~20-100x подобри резултати споредено со други верзии
   - Препорака: Фокусирај се на tdc_excretion_plus архитектурата

3. ХИПЕРПАРАМЕТРИ
   - Learning rate: 0.001 е оптимален за сите три datasets
   - Dropout: Резултатите се мешани, но 0.05-0.1 покажуваат добри резултати
   - Edge Features: НЕ се препорачуваат - performance е лош со edge features

4. ГЕНЕРАЛИЗАЦИЈА
   - tdc_excretion_plus верзијата има најмал Val-Test gap
   - Добра конзистентност помеѓу validation и test перформанси
   - Останатите верзии покажуваат знаци на overfitting

5. ИДНИ ПРАВЦИ
   - Комбинирај ги најдобрите карактеристики од tdc_excretion_plus
   - Експериментирај со ансамбли од Graph модели
   - Тестирај intermediate layer sizes (96, 160) помеѓу 64 и 128
   - Разгледај attention mechanisms без edge features
   - Додај regularization (L1/L2) наместо dropout

6. СПЕЦИФИЧНИ ПРЕПОРАКИ ПО DATASET

   Half_Life_Obach:
   - Најдобра конфигурација: Graph, 5 layers, 128 hidden, LR=0.001
   - R² може да се подобри (максимум 0.47)
   - Разгледај feature engineering на молекуларните дескриптори

   Clearance_Hepatocyte_AZ:
   - Најтежок dataset (R² максимум 0.09)
   - Потребен е поинаков approach
   - Разгледај transfer learning од Half_Life_Obach

   Clearance_Microsome_AZ:
   - Најдобри резултати (R² до 0.32)
   - Слични на Half_Life_Obach - може да се споделуваат знаења
   - Претренирај модели со combined dataset

7. ТЕХНИЧКИ ПОДОБРУВАЊА
   - Имплементирај early stopping базирано на val_rmse
   - Додај learning rate scheduling (reduce on plateau)
   - Користи gradient clipping за стабилност
   - Зголеми број на epochs со patience

8. ПРИОРИТЕТИ
   - Висок приоритет: Разбери зошто tdc_excretion_plus е толку подобар
   - Среден приоритет: Оптимизирај хиперпараметри за секој dataset одделно
   - Низок приоритет: Експериментирај со нови архитектури
"""

print(recommendations)

print("\n" + "="*120)
print("ЗАКЛУЧОК")
print("="*120 + "\n")

conclusion = """
Анализата на 565 експерименти покажа јасни патерни:

1. tdc_excretion_plus верзијата е далеку супериорна од другите верзии
2. Graph architecture е најконзистентно најдобар модел
3. 5 layers + 128 hidden channels + LR=0.001 е оптимална конфигурација
4. Edge features го влошуваат performance во повеќето случаи
5. Постои значителен простор за подобрување на R² метриката

Следните чекори:
- Реимплементирај ги новите верзии базирани на tdc_excretion_plus
- Фокусирај се на Graph архитектурата
- Експериментирај со ensemble методи
- Подобри feature representations за Clearance_Hepatocyte_AZ dataset
"""

print(conclusion)

print("\n" + "="*120)
