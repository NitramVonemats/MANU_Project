# MODEL COMPARISON - Phase 1

Comprehensive model comparison script што тестира 5 GNN архитектури (GCN, GAT, SAGE, GIN, GINE) на 7 датасети со различни хиперпараметри.

## Што го прави скриптот?

1. **Тестира 5 GNN модели:**
   - GCN (Graph Convolutional Network)
   - GAT (Graph Attention Network)
   - GraphSAGE (Sampling and Aggregation)
   - GIN (Graph Isomorphism Network)
   - GINE (GIN with Edge features)

2. **На 7 датасети:**
   - **ADME (regression):** Caco2_Wang, Half_Life_Obach, Clearance_Hepatocyte_AZ, Clearance_Microsome_AZ
   - **Toxicity (classification):** tox21, herg, clintox

3. **Со grid search на хиперпараметри:**
   - Број на layers: 3, 5, 7
   - Hidden dimensions: 64, 128, 256
   - Learning rate: 0.001, 0.0001
   - Dropout: 0.0, 0.3, 0.5
   - Head dimensions: (256,128,64), (128,64), (64,32)

4. **Total experiments:** 5 × 7 × 3 × 3 × 2 × 3 × 3 = **5,670 runs**

## Како да се пушти скриптот?

### 1. Основно (со default параметри - 5,670 runs)

```bash
cd C:/Users/martinsta/PycharmProjects/MANU_Project
source .venv/Scripts/activate
python model_comparison.py
```

Ова ќе тестира сè што е спомнато погоре.

### 2. Брзо тестирање (помал број runs)

Ако сакаш да тестираш помал број комбинации за да видиш дали работи:

```python
# Едитирај го model_comparison.py на крајот и додај:
if __name__ == "__main__":
    results = run_model_comparison(
        # Ограничи на 2 модели
        models=["GCN", "SAGE"],
        # Ограничи на 2 датасети
        datasets=["Caco2_Wang", "tox21"],
        # Ограничи хиперпараметри
        num_layers_list=[5],           # само 5 layers
        hidden_dims_list=[128],        # само 128 hidden
        learning_rates=[1e-3],         # само 0.001 LR
        dropouts=[0.0, 0.3],          # 2 dropout вредности
        head_dims_list=[(256, 128, 64)],  # само еден head config
        # Друго
        epochs=50,  # помалку epochs
        patience=10,  # брз early stopping
        device="auto",
        seed=42,
    )
```

Ова би биле: 2 × 2 × 1 × 1 × 1 × 2 × 1 = **8 runs**

### 3. Тест само модели (без grid search на хиперпараметри)

```python
if __name__ == "__main__":
    results = run_model_comparison(
        models=["GCN", "GAT", "SAGE", "GIN", "GINE"],  # сите 5 модели
        datasets=[  # сите 7 датасети
            "Caco2_Wang",
            "Half_Life_Obach",
            "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ",
            "tox21",
            "herg",
            "clintox",
        ],
        # Фиксирај хиперпараметри (без grid search)
        num_layers_list=[5],            # само 5 layers
        hidden_dims_list=[128],         # само 128 hidden
        learning_rates=[1e-3],          # само 0.001 LR
        dropouts=[0.0],                 # без dropout
        head_dims_list=[(256, 128, 64)],  # еден head config
        epochs=100,
        patience=20,
        device="auto",
        seed=42,
    )
```

Ова би биле: 5 × 7 × 1 × 1 × 1 × 1 × 1 = **35 runs**

### 4. Ablation study - тест еден параметар одеднаш

#### a) Само layers (фиксирај сè друго)

```python
results = run_model_comparison(
    models=["SAGE"],  # користи најдобар модел од претходни тестови
    datasets=["Caco2_Wang"],  # еден датасет за брзина
    num_layers_list=[3, 5, 7, 9],  # ТЕСТИРАЈ ОВА
    hidden_dims_list=[128],  # фиксирано
    learning_rates=[1e-3],  # фиксирано
    dropouts=[0.0],  # фиксирано
    head_dims_list=[(256, 128, 64)],  # фиксирано
    epochs=100,
    patience=20,
)
```

Ова: 1 × 1 × 4 × 1 × 1 × 1 × 1 = **4 runs**

#### b) Само hidden dimensions

```python
results = run_model_comparison(
    models=["SAGE"],
    datasets=["Caco2_Wang"],
    num_layers_list=[5],  # фиксирано
    hidden_dims_list=[32, 64, 128, 256, 512],  # ТЕСТИРАЈ ОВА
    learning_rates=[1e-3],
    dropouts=[0.0],
    head_dims_list=[(256, 128, 64)],
    epochs=100,
    patience=20,
)
```

Ова: 1 × 1 × 1 × 5 × 1 × 1 × 1 = **5 runs**

#### c) Само learning rate

```python
results = run_model_comparison(
    models=["SAGE"],
    datasets=["Caco2_Wang"],
    num_layers_list=[5],
    hidden_dims_list=[128],
    learning_rates=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],  # ТЕСТИРАЈ ОВА
    dropouts=[0.0],
    head_dims_list=[(256, 128, 64)],
    epochs=100,
    patience=20,
)
```

Ова: 1 × 1 × 1 × 1 × 7 × 1 × 1 = **7 runs**

#### d) Само dropout

```python
results = run_model_comparison(
    models=["SAGE"],
    datasets=["Caco2_Wang"],
    num_layers_list=[5],
    hidden_dims_list=[128],
    learning_rates=[1e-3],
    dropouts=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],  # ТЕСТИРАЈ ОВА
    head_dims_list=[(256, 128, 64)],
    epochs=100,
    patience=20,
)
```

Ова: 1 × 1 × 1 × 1 × 1 × 6 × 1 = **6 runs**

## Резултати

Скриптот автоматски генерира 3 фајла во `reports/model_comparison/<timestamp>/`:

1. **results.json** - Комплетни резултати во JSON формат
2. **summary.csv** - Табела со сите резултати (model, dataset, hyperparams, metrics)
3. **best_models_per_dataset.csv** - Најдобар модел + хиперпараметри за секој датасет

### Колони во summary.csv:

- **Dataset**: Име на датасет
- **Model**: Тип на GNN модел
- **Task**: Classification или Regression
- **Layers**: Број на GNN layers
- **Hidden_Dim**: Hidden dimension
- **Learning_Rate**: Learning rate
- **Dropout**: Dropout rate
- **Head_Dims**: Prediction head архитектура
- **Train_Time_s**: Време за тренирање (секунди)
- **Best_Epoch**: Епоха каде што е најдобар validation метрик
- **N_Params**: Број на параметри во моделот
- **Test_RMSE** / **Test_AUC**: Test перформанса
- **Test_R2** / **Test_F1**: Дополнителни метрици
- **Val_RMSE** / **Val_AUC**: Validation перформанса

## Препораки

### Фаза 1: Model Selection (35 runs - неколку часа)
Тестирај сите 5 модели на сите 7 датасети со фиксирани хиперпараметри за да одбереш најдобар модел за секој датасет.

```python
results = run_model_comparison(
    num_layers_list=[5],
    hidden_dims_list=[128],
    learning_rates=[1e-3],
    dropouts=[0.0],
    head_dims_list=[(256, 128, 64)],
)
```

### Фаза 2: Hyperparameter Grid Search (многу runs - неколку дена)
По што ќе одбереш најдобар модел за секој датасет, тестирај grid search на хиперпараметрите.

```python
# Пример: ако SAGE е најдобар за Caco2_Wang
results = run_model_comparison(
    models=["SAGE"],
    datasets=["Caco2_Wang"],
    num_layers_list=[3, 5, 7],
    hidden_dims_list=[64, 128, 256],
    learning_rates=[1e-3, 1e-4],
    dropouts=[0.0, 0.3, 0.5],
    head_dims_list=[(256, 128, 64), (128, 64), (64, 32)],
)
```

### Фаза 3: Ablation Studies (малку runs - неколку часа)
Тестирај impact на секој хиперпараметар одделно (како горните примери).

## Checkpoint & Resume

Скриптот автоматски зачувува intermediate резултати во `results.json` по секој експеримент. Ако прекинеш, можеш да го продолжиш со:

```python
# TODO: Имплементирај resume функционалност
# results = resume_model_comparison("reports/model_comparison/20251201_123456")
```

## Мониторирање

За да го следиш прогресот во реално време:

```bash
# Во друг terminal
tail -f model_comparison_run.log
```

Или додај pbar (progress bar):

```python
from tqdm import tqdm
# Додај tqdm wrapper во main loop
```

## Време на извршување

Приближно времиња (зависи од hardware):

- **35 runs** (model selection): ~2-4 часа
- **5,670 runs** (full grid search): ~10-20 дена (без GPU)
- **5,670 runs** (full grid search): ~1-3 дена (со GPU)

За брзо тестирање користи `epochs=50, patience=10` наместо default `epochs=100, patience=20`.

## Troubleshooting

### Error: CUDA out of memory
Смали го batch size:
```python
config = OptimizedGNNConfig(batch_train=16, batch_eval=32)  # наместо 32, 64
```

### Error: Dataset loading failed
Провери дали TDC библиотеката е инсталирана:
```bash
pip install PyTDC
```

### Warning: Training time too long
Смали го бројот на epochs или зголеми го patience:
```python
epochs=50, patience=10  # наместо 100, 20
```

---

**Забелешка:** Ова е Phase 1 од целокупниот workflow. По завршување на model comparison, продолжи со Phase 2 (HPO со PSO/GA/ABC/etc) и Phase 3 (Ablation studies + Visualization).
