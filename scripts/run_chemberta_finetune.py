"""
ChemBERTa Fine-tuning for ADMET Prediction
Unfreezes last N layers for task-specific training
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ============== CONFIGURATION ==============

DATASETS = {
    'Caco2_Wang': 'regression',
    'Half_Life_Obach': 'regression',
    'Clearance_Hepatocyte_AZ': 'regression',
    'Clearance_Microsome_AZ': 'regression',
    'tox21': 'classification',
    'herg': 'classification',
}

SEED = 42
MAX_EPOCHS = 30
PATIENCE = 10
BATCH_SIZE = 32
UNFREEZE_LAYERS = 2  # Unfreeze last 2 transformer layers
LR_ENCODER = 1e-5    # Lower LR for pretrained layers
LR_HEAD = 1e-3       # Normal LR for prediction head
OUTPUT_DIR = os.path.join(project_root, 'results', 'chemberta_finetune')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset(name):
    """Load dataset from TDC"""
    if name.lower() in ['tox21', 'herg']:
        from tdc.single_pred import Tox
        if name.lower() == 'tox21':
            data = Tox(name='tox21', label_name='NR-AR')
        else:
            data = Tox(name='herg')
    else:
        from tdc.single_pred import ADME
        data = ADME(name=name)

    return data.get_split(method='scaffold', seed=SEED)


class ChemBERTaDataset(Dataset):
    """Dataset for ChemBERTa with SMILES tokenization"""

    def __init__(self, smiles_list, labels, tokenizer, max_length=128):
        self.smiles = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class ChemBERTaFineTuned(nn.Module):
    """ChemBERTa with fine-tuning support"""

    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1",
                 unfreeze_layers=2, dropout=0.2, task_type='regression'):
        super().__init__()

        print(f"  Loading ChemBERTa from {model_name}...")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.task_type = task_type
        self.hidden_size = self.encoder.config.hidden_size

        # Freeze all layers first
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer layers
        total_layers = len(self.encoder.encoder.layer)
        for i in range(total_layers - unfreeze_layers, total_layers):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = True

        # Also unfreeze pooler if present
        if hasattr(self.encoder, 'pooler') and self.encoder.pooler is not None:
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Count parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS token representation
        pooled = outputs.last_hidden_state[:, 0, :]

        logits = self.head(pooled)

        if self.task_type == 'classification':
            return torch.sigmoid(logits)
        return logits


def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item() * len(labels)
        n_samples += len(labels)

    return total_loss / n_samples


def evaluate(model, loader, device, task_type):
    """Evaluate model"""
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            preds.extend(outputs.cpu().numpy().flatten())
            labels.extend(batch['label'].numpy().flatten())

    preds = np.array(preds)
    labels = np.array(labels)

    if task_type == 'classification':
        return roc_auc_score(labels, preds)
    else:
        return np.sqrt(mean_squared_error(labels, preds))


def run_finetune_for_dataset(dataset_name, task_type, tokenizer, device):
    """Run fine-tuning for a single dataset"""

    # Load data
    print(f"  Loading {dataset_name}...")
    split = get_dataset(dataset_name)

    # Prepare datasets
    train_smiles = split['train']['Drug'].tolist()
    train_labels = split['train']['Y'].astype(float).tolist()
    valid_smiles = split['valid']['Drug'].tolist()
    valid_labels = split['valid']['Y'].astype(float).tolist()
    test_smiles = split['test']['Drug'].tolist()
    test_labels = split['test']['Y'].astype(float).tolist()

    # Filter NaN
    train_data = [(s, l) for s, l in zip(train_smiles, train_labels) if not np.isnan(l)]
    valid_data = [(s, l) for s, l in zip(valid_smiles, valid_labels) if not np.isnan(l)]
    test_data = [(s, l) for s, l in zip(test_smiles, test_labels) if not np.isnan(l)]

    train_smiles, train_labels = zip(*train_data)
    valid_smiles, valid_labels = zip(*valid_data)
    test_smiles, test_labels = zip(*test_data)

    print(f"  Train: {len(train_smiles)}, Valid: {len(valid_smiles)}, Test: {len(test_smiles)}")

    # Create datasets
    train_dataset = ChemBERTaDataset(train_smiles, train_labels, tokenizer)
    valid_dataset = ChemBERTaDataset(valid_smiles, valid_labels, tokenizer)
    test_dataset = ChemBERTaDataset(test_smiles, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE * 2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2)

    # Create model
    model = ChemBERTaFineTuned(
        unfreeze_layers=UNFREEZE_LAYERS,
        dropout=0.2,
        task_type=task_type
    ).to(device)

    # Separate parameter groups
    encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'head' in n]

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': LR_ENCODER, 'weight_decay': 0.01},
        {'params': head_params, 'lr': LR_HEAD, 'weight_decay': 0.01}
    ])

    # Scheduler
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Loss
    if task_type == 'classification':
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    # Training
    best_val = float('inf') if task_type == 'regression' else 0
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_metric': []}

    print("  Training...")
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_metric = evaluate(model, valid_loader, device, task_type)

        history['train_loss'].append(train_loss)
        history['val_metric'].append(val_metric)

        metric_name = 'AUC' if task_type == 'classification' else 'RMSE'
        print(f"    Epoch {epoch+1}: Loss={train_loss:.4f}, Val {metric_name}={val_metric:.4f}")

        # Early stopping
        if task_type == 'regression':
            improved = val_metric < best_val
        else:
            improved = val_metric > best_val

        if improved:
            best_val = val_metric
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test
    if best_model_state:
        model.load_state_dict(best_model_state)

    test_metric = evaluate(model, test_loader, device, task_type)
    print(f"  Test {metric_name}: {test_metric:.4f}")

    return {
        'dataset': dataset_name,
        'task_type': task_type,
        'best_val_metric': best_val,
        'test_metric': test_metric,
        'epochs_trained': len(history['train_loss']),
        'history': history
    }


def run_chemberta_finetune_benchmark():
    """Run ChemBERTa fine-tuning on all datasets"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print("CHEMBERTA FINE-TUNING BENCHMARK")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Unfreeze layers: {UNFREEZE_LAYERS}")
    print(f"LR (encoder): {LR_ENCODER}, LR (head): {LR_HEAD}")
    print(f"{'='*70}\n")

    set_seed(SEED)

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    all_results = []

    for dataset_name, task_type in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name} ({task_type})")
        print(f"{'='*60}")

        try:
            result = run_finetune_for_dataset(dataset_name, task_type, tokenizer, device)
            all_results.append(result)

            # Save individual result
            with open(f"{OUTPUT_DIR}/chemberta_ft_{dataset_name}_results.json", 'w') as f:
                json.dump({k: v for k, v in result.items() if k != 'history'}, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        # Clear cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save summary
    summary_df = pd.DataFrame([{
        'dataset': r['dataset'],
        'model': 'ChemBERTa-FT',
        'task_type': r['task_type'],
        'test_metric': r['test_metric'],
        'best_val_metric': r['best_val_metric'],
        'epochs': r['epochs_trained']
    } for r in all_results])

    summary_df.to_csv(f"{OUTPUT_DIR}/chemberta_finetune_summary.csv", index=False)

    # Print summary
    print(f"\n{'='*70}")
    print("CHEMBERTA FINE-TUNING SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<30} {'Task':<15} {'Test Metric':<15}")
    print("-"*60)

    for r in all_results:
        metric_name = 'AUC' if r['task_type'] == 'classification' else 'RMSE'
        print(f"{r['dataset']:<30} {r['task_type']:<15} {r['test_metric']:.4f} ({metric_name})")

    print(f"\n{'='*70}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    run_chemberta_finetune_benchmark()
