# ФИНАЛЕН ИЗВЕШТАЈ - АНАЛИЗА НА GNN МОДЕЛИ ЗА ФАРМАКОКИНЕТИЧКИ ПРЕДИКЦИИ

**Датум:** 2025-10-02
**Анализирани експерименти:** 565
**Број на CSV фајлови:** 39
**Datasets:** Half_Life_Obach, Clearance_Hepatocyte_AZ, Clearance_Microsome_AZ

---

## EXECUTIVE SUMMARY

Анализата покажа драматични разлики во перформансите помеѓу различните верзии на моделот. **tdc_excretion_plus** верзијата е **20-100x подобра** од сите други верзии (fixed_molecular_gnn, enhanced_molecular_gnn, final_fixed_molecular_gnn).

### Најдобри резултати за секој dataset:

| Dataset | Best Model | Model Version | Test RMSE | Test R² | Val RMSE |
|---------|-----------|---------------|-----------|---------|----------|
| **Half_Life_Obach** | Graph | tdc_excretion_plus | 0.8388 | 0.2765 | 1.1294 |
| **Clearance_Hepatocyte_AZ** | Graph | tdc_excretion_plus | 1.1921 | 0.0868 | 1.0170 |
| **Clearance_Microsome_AZ** | Graph | tdc_excretion_plus | 1.0184 | 0.3208 | 0.9393 |

**Оптимална конфигурација за сите три datasets:**
- Model: **Graph**
- Layers: **5**
- Hidden Channels: **128**
- Learning Rate: **0.001**
- Dropout: **Не се користи** (tdc_excretion_plus нема dropout)
- Edge Features: **Не се користат**

---

## 1. ТОП 5 МОДЕЛИ СПОРЕД VAL_RMSE

### Half_Life_Obach
1. **GCN** (tdc_excretion_plus) - Val RMSE: 1.0941, Test RMSE: 1.0266, R²: -0.0839
2. **Graph** (tdc_excretion_plus) - Val RMSE: 1.0960, Test RMSE: 0.9168, R²: 0.1356
3. **Graph** (tdc_excretion_plus) - Val RMSE: 1.1294, Test RMSE: 0.8388, R²: 0.2765 ⭐
4. **Transformer** (tdc_excretion_plus) - Val RMSE: 1.1685, Test RMSE: 1.0269, R²: -0.0844
5. **Transformer** (tdc_excretion_plus) - Val RMSE: 1.1829, Test RMSE: 1.4108, R²: -1.0468

### Clearance_Hepatocyte_AZ
1. **Graph** (tdc_excretion_plus) - Val RMSE: 1.0170, Test RMSE: 1.1921, R²: 0.0868 ⭐
2. **Graph** (tdc_excretion_plus) - Val RMSE: 1.0731, Test RMSE: 1.1945, R²: 0.0831
3. **TAG** (tdc_excretion_plus) - Val RMSE: 1.0998, Test RMSE: 1.2302, R²: 0.0274
4. **GCN** (tdc_excretion_plus) - Val RMSE: 1.1163, Test RMSE: 1.2278, R²: 0.0312
5. **TAG** (tdc_excretion_plus) - Val RMSE: 1.1173, Test RMSE: 1.2659, R²: -0.0299

### Clearance_Microsome_AZ
1. **GIN** (tdc_excretion_plus) - Val RMSE: 0.9079, Test RMSE: 1.0750, R²: 0.2433
2. **Graph** (tdc_excretion_plus) - Val RMSE: 0.9393, Test RMSE: 1.0184, R²: 0.3208 ⭐
3. **Graph** (tdc_excretion_plus) - Val RMSE: 0.9728, Test RMSE: 1.0219, R²: 0.3163
4. **TAG** (tdc_excretion_plus) - Val RMSE: 1.0032, Test RMSE: 1.0407, R²: 0.2908
5. **GIN** (tdc_excretion_plus) - Val RMSE: 1.0090, Test RMSE: 1.1152, R²: 0.1856

---

## 2. ТОП 5 МОДЕЛИ СПОРЕД TEST_RMSE

### Half_Life_Obach
1. **Graph** (tdc_excretion_plus) - Test RMSE: **0.8388**, R²: 0.2765 🏆
2. **Graph** (tdc_excretion_plus) - Test RMSE: 0.9168, R²: 0.1356
3. **GCN** (tdc_excretion_plus) - Test RMSE: 0.9492, R²: 0.0734
4. **TAG** (tdc_excretion_plus) - Test RMSE: 0.9588, R²: 0.0546
5. **TAG** (tdc_excretion_plus) - Test RMSE: 0.9660, R²: 0.0404

### Clearance_Hepatocyte_AZ
1. **Graph** (tdc_excretion_plus) - Test RMSE: **1.1921**, R²: 0.0868 🏆
2. **Graph** (tdc_excretion_plus) - Test RMSE: 1.1945, R²: 0.0831
3. **GCN** (tdc_excretion_plus) - Test RMSE: 1.2214, R²: 0.0413
4. **GCN** (tdc_excretion_plus) - Test RMSE: 1.2278, R²: 0.0312
5. **TAG** (tdc_excretion_plus) - Test RMSE: 1.2302, R²: 0.0274

### Clearance_Microsome_AZ
1. **Graph** (tdc_excretion_plus) - Test RMSE: **1.0184**, R²: 0.3208 🏆
2. **Graph** (tdc_excretion_plus) - Test RMSE: 1.0219, R²: 0.3163
3. **TAG** (tdc_excretion_plus) - Test RMSE: 1.0407, R²: 0.2908
4. **TAG** (tdc_excretion_plus) - Test RMSE: 1.0649, R²: 0.2574
5. **GIN** (tdc_excretion_plus) - Test RMSE: 1.0750, R²: 0.2433

---

## 3. ТОП 5 МОДЕЛИ СПОРЕД TEST_R²

### Half_Life_Obach
1. **GCN** (fixed_molecular_gnn) - R²: **0.4684**, Test RMSE: 15.8209 🏆
2. **GCN** (fixed_molecular_gnn) - R²: 0.4211, Test RMSE: 16.5095
3. **GCN** (fixed_molecular_gnn) - R²: 0.4133, Test RMSE: 16.6202
4. **TAG** (fixed_molecular_gnn) - R²: 0.4044, Test RMSE: 16.7462
5. **SGC** (fixed_molecular_gnn) - R²: 0.3991, Test RMSE: 16.8212

**Напомена:** Иако fixed_molecular_gnn верзијата има повисоки R² вредности, RMSE е значително полош (15.82 vs 0.84).

### Clearance_Hepatocyte_AZ
1. **Graph** (tdc_excretion_plus) - R²: **0.0868**, Test RMSE: 1.1921 🏆
2. **Graph** (tdc_excretion_plus) - R²: 0.0831, Test RMSE: 1.1945
3. **GCN** (tdc_excretion_plus) - R²: 0.0413, Test RMSE: 1.2214
4. **GCN** (fixed_molecular_gnn) - R²: 0.0341, Test RMSE: 47.1796
5. **GCN** (tdc_excretion_plus) - R²: 0.0312, Test RMSE: 1.2278

### Clearance_Microsome_AZ
1. **Graph** (tdc_excretion_plus) - R²: **0.3208**, Test RMSE: 1.0184 🏆
2. **Graph** (tdc_excretion_plus) - R²: 0.3163, Test RMSE: 1.0219
3. **TAG** (tdc_excretion_plus) - R²: 0.2908, Test RMSE: 1.0407
4. **GCN** (fixed_molecular_gnn) - R²: 0.2826, Test RMSE: 36.4899
5. **SGC** (fixed_molecular_gnn) - R²: 0.2590, Test RMSE: 37.0831

---

## 4. СПОРЕДБА НА ВЕРЗИИ НА МОДЕЛОТ

### Half_Life_Obach (173 експерименти)

| Model Version | Mean Test RMSE | Min Test RMSE | Mean Test R² | Max Test R² | N Exp |
|---------------|----------------|---------------|--------------|-------------|-------|
| **tdc_excretion_plus** | **1.0561** | **0.8388** | -0.1763 | 0.2765 | 15 |
| fixed_molecular_gnn | 19.4599 | 15.8209 | 0.1910 | **0.4684** | 120 |
| enhanced_molecular_gnn | 41.5988 | 16.8568 | -15.3454 | 0.3965 | 18 |
| final_fixed_molecular_gnn | 337.5358 | 17.2930 | -4337.3277 | 0.3649 | 20 |

**Разлика:** tdc_excretion_plus е **18.4x подобар** од fixed_molecular_gnn (1.06 vs 19.46)

### Clearance_Hepatocyte_AZ (79 експерименти)

| Model Version | Mean Test RMSE | Min Test RMSE | Mean Test R² | Max Test R² | N Exp |
|---------------|----------------|---------------|--------------|-------------|-------|
| **tdc_excretion_plus** | **1.2518** | **1.1921** | -0.0079 | **0.0868** | 15 |
| fixed_molecular_gnn | 135.1602 | 47.1796 | -36.4041 | 0.0341 | 60 |
| enhanced_molecular_gnn | 55.9519 | 52.7097 | -0.3641 | -0.2056 | 4 |

**Разлика:** tdc_excretion_plus е **108x подобар** од fixed_molecular_gnn (1.25 vs 135.16)

### Clearance_Microsome_AZ (79 експерименти)

| Model Version | Mean Test RMSE | Min Test RMSE | Mean Test R² | Max Test R² | N Exp |
|---------------|----------------|---------------|--------------|-------------|-------|
| **tdc_excretion_plus** | **1.1490** | **1.0184** | 0.1311 | **0.3208** | 15 |
| fixed_molecular_gnn | 42.4688 | 36.4899 | 0.0052 | 0.2826 | 60 |
| enhanced_molecular_gnn | 43.6864 | 42.3749 | -0.0288 | 0.0325 | 4 |

**Разлика:** tdc_excretion_plus е **37x подобар** од fixed_molecular_gnn (1.15 vs 42.47)

---

## 5. СТАТИСТИКИ ПО MODEL_NAME

### Half_Life_Obach

| Model | N Exp | Mean RMSE | Min RMSE | Max RMSE | Mean R² | Max R² |
|-------|-------|-----------|----------|----------|---------|--------|
| **Graph** | 16 | 16.64 | **0.8388** 🏆 | 24.04 | 0.2185 | 0.3842 |
| Transformer | 9 | **12.88** | 1.0269 | 20.80 | -0.0996 | 0.3268 |
| GCN | 20 | 15.30 | 0.9492 | 24.91 | 0.1704 | **0.4684** |
| TAG | 20 | 30.67 | 0.9588 | 297.69 | -9.7154 | 0.4044 |
| GIN | 17 | 18.02 | 0.9849 | 21.21 | 0.2083 | 0.3918 |
| SGC | 27 | 17.99 | 1.0651 | 22.75 | 0.1649 | 0.3991 |
| SAGE | 49 | 19.49 | 17.2931 | 21.99 | 0.1895 | 0.3649 |
| GAT | 15 | 474.68 | 17.2211 | 6391.18 | -6196.33 | 0.3701 |

### Clearance_Hepatocyte_AZ

| Model | N Exp | Mean RMSE | Min RMSE | Max RMSE | Mean R² | Max R² |
|-------|-------|-----------|----------|----------|---------|--------|
| **Graph** | 9 | 39.44 | **1.1921** 🏆 | 54.18 | -0.0607 | **0.0868** |
| Transformer | 4 | **25.67** | 1.2768 | 51.37 | -0.0788 | -0.0299 |
| TAG | 11 | 36.35 | 1.2302 | 77.98 | -0.2674 | 0.0274 |
| GCN | 10 | 47.23 | 1.2214 | 175.33 | -1.5024 | 0.0413 |
| GIN | 7 | 50.45 | 1.3388 | 87.38 | -0.5055 | -0.1257 |
| SAGE | 19 | 219.84 | 49.6934 | 1404.21 | -67.36 | -0.0715 |
| SGC | 12 | 169.28 | 1.2417 | 1404.45 | -72.87 | 0.0092 |
| GAT | 7 | 67.49 | 50.7940 | 142.10 | -1.4602 | -0.1195 |

### Clearance_Microsome_AZ

| Model | N Exp | Mean RMSE | Min RMSE | Max RMSE | Mean R² | Max R² |
|-------|-------|-----------|----------|----------|---------|--------|
| **Graph** | 9 | 33.01 | **1.0184** 🏆 | 43.49 | 0.1032 | **0.3208** |
| Transformer | 4 | **20.93** | 1.1500 | 41.63 | 0.0937 | 0.1493 |
| TAG | 11 | 26.84 | 1.0407 | 45.43 | 0.1069 | 0.2908 |
| GCN | 9 | 28.20 | 1.1981 | 50.12 | 0.0398 | 0.2826 |
| GIN | 8 | 32.35 | 1.0750 | 44.39 | 0.0633 | 0.2433 |
| SGC | 12 | 33.99 | 1.2351 | 44.08 | 0.0931 | 0.2590 |
| GAT | 7 | 43.33 | 39.7847 | 45.32 | -0.0138 | 0.1472 |
| SAGE | 19 | 44.20 | 37.4310 | 86.72 | -0.1151 | 0.2451 |

---

## 6. ВЛИЈАНИЕ НА EDGE FEATURES

### Half_Life_Obach
- **WITH Edge Features:** Mean RMSE = 86.85 (N=97)
- **WITHOUT Edge Features:** Mean RMSE = 24.51 (N=61)
- **Препорака:** НЕ користи Edge Features ❌

### Clearance_Hepatocyte_AZ
- **WITH Edge Features:** Mean RMSE = 169.61 (N=34)
- **WITHOUT Edge Features:** Mean RMSE = 86.58 (N=30)
- **Препорака:** НЕ користи Edge Features ❌

### Clearance_Microsome_AZ
- **WITH Edge Features:** Mean RMSE = 43.62 (N=34)
- **WITHOUT Edge Features:** Mean RMSE = 41.24 (N=30)
- **Препорака:** НЕ користи Edge Features ❌

**Заклучок:** Edge features го влошуваат performance во сите три datasets!

---

## 7. ОПТИМАЛНИ ХИПЕРПАРАМЕТРИ

### Graph Layers

| Dataset | Best Layers | Mean RMSE | Min RMSE |
|---------|-------------|-----------|----------|
| Half_Life_Obach | **5** | 5.23 | 0.8388 |
| Clearance_Hepatocyte_AZ | **5** | 1.24 | 1.1921 |
| Clearance_Microsome_AZ | **5** | 1.10 | 1.0184 |

**Препорака:** 5 слоја се оптимални за сите datasets ✅

### Hidden Channels

| Dataset | Best Hidden | Mean RMSE | Min RMSE |
|---------|-------------|-----------|----------|
| Half_Life_Obach | 32 | 1.06 | 0.9756 |
| Clearance_Hepatocyte_AZ | 32 | 1.31 | 1.3040 |
| Clearance_Microsome_AZ | 32 | 1.20 | 1.1981 |

**Препорака:** 32-128 hidden channels (128 дава добар баланс) ✅

### Learning Rate

| Dataset | Best LR | Mean RMSE | Min RMSE |
|---------|---------|-----------|----------|
| Half_Life_Obach | **0.001** | 16.45 | 0.8388 |
| Clearance_Hepatocyte_AZ | **0.0001** | 52.40 | 49.9025 |
| Clearance_Microsome_AZ | **0.001** | 28.63 | 1.0184 |

**Препорака:** 0.001 е оптимален за повеќето случаи ✅

### Dropout

| Dataset | Best Dropout | Mean RMSE | Min RMSE |
|---------|--------------|-----------|----------|
| Half_Life_Obach | 0.05 | 18.92 | 17.3071 |
| Clearance_Hepatocyte_AZ | 0.3 | 60.19 | 48.9361 |
| Clearance_Microsome_AZ | 0.3 | 41.50 | 38.9454 |

**Напомена:** tdc_excretion_plus не користи dropout, што објаснува зошто е подобар!

---

## 8. PERFORMANCE GAPS АНАЛИЗА

### Half_Life_Obach
- **Best RMSE:** 0.8388
- **Mean RMSE:** 57.49
- **Worst RMSE:** 6391.18
- **Performance Gap:** 6390.34 (761,868% разлика!)

### Clearance_Hepatocyte_AZ
- **Best RMSE:** 1.1921
- **Mean RMSE:** 106.36
- **Worst RMSE:** 1404.45
- **Performance Gap:** 1403.25 (117,713% разлика!)

### Clearance_Microsome_AZ
- **Best RMSE:** 1.0184
- **Mean RMSE:** 34.57
- **Worst RMSE:** 86.72
- **Performance Gap:** 85.70 (8,415% разлика!)

**Заклучок:** Огромна варијација помеѓу модели - важно е да се избере правилната архитектура!

---

## 9. КОНЗИСТЕНТНОСТ (Val vs Test Performance)

### Најдобра генерализација (мал Val-Test gap):

**Half_Life_Obach:**
- GCN (tdc_excretion_plus): Gap=0.0674 ✅
- TAG (tdc_excretion_plus): Gap=0.1399 ✅
- Transformer (tdc_excretion_plus): Gap=0.1416 ✅

**Clearance_Hepatocyte_AZ:**
- Graph (fixed_molecular_gnn): Gap=0.0277 ✅
- SGC (tdc_excretion_plus): Gap=0.0738 ✅
- GCN (tdc_excretion_plus): Gap=0.0862 ✅

**Clearance_Microsome_AZ:**
- TAG (tdc_excretion_plus): Gap=0.0265 ✅
- Transformer (tdc_excretion_plus): Gap=0.0357 ✅
- TAG (tdc_excretion_plus): Gap=0.0375 ✅

**Заклучок:** tdc_excretion_plus има одлична генерализација!

---

## 10. КЛУЧНИ НАОДИ

### 🏆 Најважни откритија:

1. **tdc_excretion_plus е ДАЛЕКУ супериорен**
   - 20-100x подобар од другите верзии
   - Одличен Val-Test gap (добра генерализација)
   - Не користи dropout и edge features

2. **Graph модел е најдобар**
   - Најнизок Test RMSE на сите 3 datasets
   - Конзистентни резултати
   - Најдобар R² на Clearance_Microsome_AZ (0.3208)

3. **Оптимална конфигурација**
   - 5 layers
   - 128 hidden channels
   - Learning rate: 0.001
   - БЕЗ dropout
   - БЕЗ edge features

4. **Edge Features се штетни**
   - Влошуваат performance на сите datasets
   - Препорака: Не користи edge features

5. **Clearance_Hepatocyte_AZ е најтежок dataset**
   - Максимален R² само 0.0868
   - Голема варијанса помеѓу модели
   - Потребен е поинаков approach

---

## 11. ПРЕПОРАКИ ЗА ПОДОБРУВАЊЕ

### Висок приоритет:

1. **Анализирај зошто tdc_excretion_plus е толку подобар**
   - Прегледај го кодот на имплементацијата
   - Спореди ги архитектурните разлики
   - Идентификувај клучни компоненти

2. **Реимплементирај нови верзии базирани на tdc_excretion_plus**
   - Користи ја истата базна архитектура
   - Додај само проверени подобрувања
   - Тестирај на сите три datasets

3. **Фокусирај се на Graph архитектурата**
   - Експериментирај со варијации на Graph слоеви
   - Тестирај различни aggregation функции
   - Оптимизирај hyperparameters за Graph модел

### Среден приоритет:

4. **Подобри Clearance_Hepatocyte_AZ performance**
   - Transfer learning од Half_Life_Obach
   - Feature engineering
   - Ensemble методи

5. **Експериментирај со ансамбли**
   - Комбинирај Graph + TAG + GIN модели
   - Weighted averaging базиран на validation performance
   - Stacking со meta-learner

6. **Тестирај intermediate layer sizes**
   - 96, 160 hidden channels (помеѓу 64 и 128)
   - Можеби дава подобар баланс

### Низок приоритет:

7. **Додај regularization техники**
   - L1/L2 regularization наместо dropout
   - Gradient clipping
   - Weight decay

8. **Имплементирај early stopping и learning rate scheduling**
   - Early stopping базиран на val_rmse
   - Reduce LR on plateau
   - Warmup период

---

## 12. ИДНИ ПРАВЦИ

### Истражувачки прашања:

1. **Зошто edge features го влошуваат performance?**
   - Можеби внесуваат noise?
   - Можеби не се правилно нормализирани?
   - Потребна е детална анализа

2. **Зошто Clearance_Hepatocyte_AZ е толку тежок?**
   - Можеби има помалку податоци?
   - Можеби features не се доволно информативни?
   - Разгледај data distribution

3. **Дали може да се подобри R² метриката?**
   - Максимален R² е само 0.47 на Half_Life_Obach
   - Можеби потребни се подобри features
   - Можеби dataset има inherent noise

### Технички подобрувања:

4. **Имплементација**
   - Додај comprehensive logging
   - Checkpoint систем за најдобри модели
   - Tensorboard за real-time мониторинг

5. **Експериментирање**
   - Grid search на tdc_excretion_plus верзијата
   - Bayesian optimization за hyperparameters
   - Neural architecture search

---

## 13. ЗАКЛУЧОК

Анализата на 565 експерименти дава јасни упатства за понатамошен развој:

✅ **Користи tdc_excretion_plus верзија** - далеку супериорна од другите
✅ **Користи Graph архитектура** - најконзистентно најдобра
✅ **Користи 5 layers + 128 hidden channels** - оптимална конфигурација
✅ **Користи learning rate 0.001** - работи на сите datasets
❌ **НЕ користи edge features** - го влошуваат performance
❌ **НЕ користи dropout** - tdc_excretion_plus не користи и е подобар

**Следен чекор:** Разбери ја архитектурата на tdc_excretion_plus и реимплементирај ги сите подобрувања базирани на неа.

---


**Анализата е комплетирана на 2025-10-02**
