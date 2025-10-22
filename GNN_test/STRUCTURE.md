# Структура на проектот

## 📁 Организација

```
GNN_test/
├── 📦 configs/              # Конфигурации и hyperparameters
│   ├── base_config.py      # Phase 1 best configs, scalers
│   └── model_config.py     # Model configuration класи
│
├── 🔧 functional/           # Utility функции
│   ├── metrics.py          # RMSE, R², Spearman correlation
│   ├── transforms.py       # Log-scale трансформации
│   └── utils.py            # Random seed setting
│
├── 📊 graph/                # Графовски податоци и featurization
│   ├── featurizer.py       # Atom/bond/ADME features
│   └── loader.py           # TDC data loading и preprocessing
│
├── 🤖 models/               # Neural network архитектури
│   ├── gnn.py              # GNN backbone (SAGEConv, GINEConv)
│   ├── foundation.py       # Foundation models (ChemBERTa, MolFormer, итн.)
│   └── predictors.py       # Full models (GNN-only, Foundation-only, Hybrid)
│
├── 🚀 services/             # Training и benchmarking
│   ├── trainer.py          # Training loop, evaluation, early stopping
│   └── benchmark.py        # Benchmark runner за споредба на модели
│
├── 💾 data/                 # TDC dataset кеш
├── 🏆 best_models/          # Зачувани најдобри модели
├── 📈 results/              # CSV резултати од benchmarks
├── 🧪 tests/                # Unit тестови
├── 📉 visualizations/       # Графици и визуелизации
├── 📦 archive/              # Стари скрипти и документација
│   ├── old_structure/      # Пред-реорганизација фајлови
│   │   ├── old_scripts/
│   │   └── old_docs/
│   └── (Phase 1 експерименти)
│
├── 🎯 run_benchmark.py      # ГЛАВЕН СКРИПТ за benchmarking
├── ✅ test_structure.py     # Тест за верификација на структурата
├── 📖 README.md             # Главна документација
└── 📋 STRUCTURE.md          # Овој фајл
```

## 🎯 Главни фајлови

| Фајл | Опис |
|------|------|
| `run_benchmark.py` | CLI скрипт за running benchmarks |
| `test_structure.py` | Верификација дека сè работи |
| `__init__.py` | Package initialization |

## 🚀 Брз старт

### 1. Тестирај структурата
```bash
python test_structure.py
```

### 2. Покрени benchmark
```bash
# Еден dataset со еден seed (брзо)
python run_benchmark.py --dataset Half_Life_Obach --seeds 42

# Сите datasets со default seeds
python run_benchmark.py
```

### 3. Користи како модул
```python
from services.benchmark import benchmark_dataset

results = benchmark_dataset("Half_Life_Obach", seeds=[42, 123])
```

## 📊 Резултати

Резултатите се зачувуваат во `results/`:
- `phase2_benchmark_YYYYMMDD_HHMMSS.csv` - Детални резултати
- `phase2_summary_YYYYMMDD_HHMMSS.csv` - Агрегирани статистики

## 🔄 Разлики од стара структура

### Пред (монолитна)
```
phase2_foundation_benchmark.py  (~960 lines)
├── [configs внатре]
├── [utils внатре]
├── [features внатре]
├── [models внатре]
├── [training внатре]
└── [benchmarking внатре]
```

### Сега (модуларна)
```
configs/
functional/
graph/
models/
services/
run_benchmark.py
```

## ✅ Предности

1. **Модуларност** - Секој компонент е одвоен
2. **Повторна употреба** - Модулите се независни
3. **Лесно одржување** - Промените се локализирани
4. **Тестабилност** - Секој модул може да се тестира одделно
5. **Професионална организација** - Слично на GWEN-AI framework
6. **Проширливост** - Лесно додавање нови модели/features


## 🎓 Инспирирано од

GWEN-AI molecular modeling platform структура
