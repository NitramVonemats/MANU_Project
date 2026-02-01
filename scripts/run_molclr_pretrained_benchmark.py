"""
MolCLR PRETRAINED BENCHMARK
============================

Tests MolCLR with OFFICIAL pretrained weights on all 6 datasets.
Compares with old (random init) results.
"""

import os
import sys
import json
import time
import warnings
import traceback
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, f1_score, accuracy_score
)

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from adme_gnn.models.foundation import MolCLREncoder

# ============== CONFIGURATION ==============

DATASETS = [
    "Caco2_Wang",
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "tox21",
    "herg",
]

SEED = 42
OUTPUT_DIR = os.path.join(project_root, 'results', 'foundation_benchmark')

# Old results for comparison
OLD_MOLCLR_RESULTS = {
    'Caco2_Wang': {'test_rmse': 0.713, 'test_r2': -0.079},
    'Half_Life_Obach': {'test_rmse': 21.97, 'test_r2': -0.025},
    'Clearance_Hepatocyte_AZ': {'test_rmse': 48.71, 'test_r2': -0.030},
    'Clearance_Microsome_AZ': {'test_rmse': 43.33, 'test_r2': -0.012},
    'tox21': {'test_auc': 0.538, 'test_f1': 0.0},
    'herg': {'test_auc': 0.504, 'test_f1': 0.847},
}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset(name, split_method='scaffold'):
    if name.lower() in ['tox21', 'herg', 'clintox']:
        from tdc.single_pred import Tox
        tox_labels = {'tox21': 'NR-AR', 'herg': None, 'clintox': 'CT_TOX'}
        label = tox_labels.get(name.lower())
        if label:
            data = Tox(name=name, label_name=label)
        else:
            data = Tox(name=name)
    else:
        from tdc.single_pred import ADME
        data = ADME(name=name)

    split = data.get_split(method=split_method, seed=SEED)
    return split


def is_classification_task(dataset_name):
    return dataset_name.lower() in ['tox21', 'herg', 'clintox']


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def evaluate_classification(y_true, y_pred_proba):
    y_pred = (y_pred_proba > 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = 0.5
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {'auc_roc': auc, 'accuracy': acc, 'f1': f1}


def get_embeddings_batch(encoder, smiles_list, device, batch_size=32):
    embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        with torch.no_grad():
            try:
                emb = encoder(batch, device)
                embeddings.append(emb.cpu().numpy())
            except Exception as e:
                print(f"    Warning: Batch {i//batch_size} failed: {e}")
                embeddings.append(np.zeros((len(batch), encoder.proj_dim)))
    return np.vstack(embeddings)


def run_molclr_benchmark():
    """Run MolCLR pretrained benchmark on all datasets"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print("MolCLR PRETRAINED BENCHMARK")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Seed: {SEED}")
    print(f"{'='*70}\n")

    set_seed(SEED)

    # Initialize pretrained MolCLR
    print("Initializing MolCLR with PRETRAINED weights...")
    encoder = MolCLREncoder(proj_dim=256, model_type='gin')
    if hasattr(encoder, 'to'):
        encoder = encoder.to(device)
    encoder.eval()

    all_results = []
    comparison_results = []

    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*60}")

        is_classification = is_classification_task(dataset_name)
        task_type = "classification" if is_classification else "regression"
        print(f"Task type: {task_type}")

        try:
            # Load dataset
            print("Loading dataset...")
            split = get_dataset(dataset_name)
            train_df = split['train']
            valid_df = split['valid']
            test_df = split['test']

            print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

            # Prepare targets
            y_train = train_df['Y'].values.astype(float)
            y_valid = valid_df['Y'].values.astype(float)
            y_test = test_df['Y'].values.astype(float)

            # Handle NaN
            train_mask = ~np.isnan(y_train)
            valid_mask = ~np.isnan(y_valid)
            test_mask = ~np.isnan(y_test)

            train_smiles = train_df['Drug'].values[train_mask].tolist()
            valid_smiles = valid_df['Drug'].values[valid_mask].tolist()
            test_smiles = test_df['Drug'].values[test_mask].tolist()

            y_train = y_train[train_mask]
            y_valid = y_valid[valid_mask]
            y_test = y_test[test_mask]

            # Scale for regression
            scaler = None
            if not is_classification:
                scaler = StandardScaler()
                y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_valid_scaled = scaler.transform(y_valid.reshape(-1, 1)).flatten()
                y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
            else:
                y_train_scaled = y_train
                y_valid_scaled = y_valid
                y_test_scaled = y_test

            result = {
                'dataset': dataset_name,
                'model': 'MolCLR-Pretrained',
                'task_type': task_type,
                'train_size': len(train_smiles),
                'test_size': len(test_smiles),
            }

            # Extract embeddings
            print("Extracting embeddings with pretrained MolCLR...")
            t0 = time.time()
            X_train = get_embeddings_batch(encoder, train_smiles, device)
            X_valid = get_embeddings_batch(encoder, valid_smiles, device)
            X_test = get_embeddings_batch(encoder, test_smiles, device)
            embed_time = time.time() - t0
            print(f"Embedding time: {embed_time:.1f}s")

            # Train predictor
            print("Training predictor...")
            t1 = time.time()

            if is_classification:
                predictor = MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    max_iter=500,
                    early_stopping=True,
                    random_state=SEED,
                    verbose=False
                )
            else:
                predictor = MLPRegressor(
                    hidden_layer_sizes=(256, 128),
                    max_iter=500,
                    early_stopping=True,
                    random_state=SEED
                )

            predictor.fit(X_train, y_train_scaled)
            train_time = time.time() - t1

            # Evaluate
            if is_classification:
                try:
                    test_preds_proba = predictor.predict_proba(X_test)[:, 1]
                    valid_preds_proba = predictor.predict_proba(X_valid)[:, 1]
                except:
                    test_preds_proba = predictor.predict(X_test)
                    valid_preds_proba = predictor.predict(X_valid)

                test_metrics = evaluate_classification(y_test_scaled, test_preds_proba)
                valid_metrics = evaluate_classification(y_valid_scaled, valid_preds_proba)

                print(f"Test AUC: {test_metrics['auc_roc']:.4f}")
                print(f"Test F1:  {test_metrics['f1']:.4f}")
                print(f"Test Acc: {test_metrics['accuracy']:.4f}")

                result.update({
                    'test_auc': test_metrics['auc_roc'],
                    'test_f1': test_metrics['f1'],
                    'test_acc': test_metrics['accuracy'],
                    'valid_auc': valid_metrics['auc_roc'],
                })

                # Comparison
                old = OLD_MOLCLR_RESULTS.get(dataset_name, {})
                improvement = test_metrics['auc_roc'] - old.get('test_auc', 0.5)
                print(f"\n  OLD MolCLR (random init): AUC = {old.get('test_auc', 'N/A')}")
                print(f"  NEW MolCLR (pretrained):  AUC = {test_metrics['auc_roc']:.4f}")
                print(f"  IMPROVEMENT: +{improvement:.4f} ({improvement*100:.1f}%)")

                comparison_results.append({
                    'dataset': dataset_name,
                    'old_auc': old.get('test_auc'),
                    'new_auc': test_metrics['auc_roc'],
                    'improvement': improvement
                })

            else:
                test_preds = predictor.predict(X_test)
                valid_preds = predictor.predict(X_valid)

                if scaler:
                    test_preds_orig = scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
                    valid_preds_orig = scaler.inverse_transform(valid_preds.reshape(-1, 1)).flatten()
                else:
                    test_preds_orig = test_preds
                    valid_preds_orig = valid_preds

                test_metrics = evaluate_regression(y_test, test_preds_orig)
                valid_metrics = evaluate_regression(y_valid, valid_preds_orig)

                print(f"Test RMSE: {test_metrics['rmse']:.4f}")
                print(f"Test R2:   {test_metrics['r2']:.4f}")
                print(f"Test MAE:  {test_metrics['mae']:.4f}")

                result.update({
                    'test_rmse': test_metrics['rmse'],
                    'test_r2': test_metrics['r2'],
                    'test_mae': test_metrics['mae'],
                    'valid_rmse': valid_metrics['rmse'],
                    'valid_r2': valid_metrics['r2'],
                })

                # Comparison
                old = OLD_MOLCLR_RESULTS.get(dataset_name, {})
                old_rmse = old.get('test_rmse', float('inf'))
                improvement = old_rmse - test_metrics['rmse']
                print(f"\n  OLD MolCLR (random init): RMSE = {old_rmse}")
                print(f"  NEW MolCLR (pretrained):  RMSE = {test_metrics['rmse']:.4f}")
                print(f"  IMPROVEMENT: {improvement:.4f} ({(improvement/old_rmse)*100:.1f}% better)")

                comparison_results.append({
                    'dataset': dataset_name,
                    'old_rmse': old_rmse,
                    'new_rmse': test_metrics['rmse'],
                    'improvement_rmse': improvement,
                    'old_r2': old.get('test_r2'),
                    'new_r2': test_metrics['r2'],
                })

            result['embed_time'] = embed_time
            result['train_time'] = train_time
            result['total_time'] = embed_time + train_time
            result['status'] = 'success'

            all_results.append(result)

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            result['status'] = 'failed'
            result['error'] = str(e)
            all_results.append(result)

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_results = pd.DataFrame(all_results)
    results_file = os.path.join(OUTPUT_DIR, f'molclr_pretrained_results_{timestamp}.csv')
    df_results.to_csv(results_file, index=False)

    df_comparison = pd.DataFrame(comparison_results)
    comparison_file = os.path.join(OUTPUT_DIR, f'molclr_old_vs_new_comparison_{timestamp}.csv')
    df_comparison.to_csv(comparison_file, index=False)

    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY: OLD vs NEW MolCLR")
    print(f"{'='*70}")

    print("\n--- REGRESSION (RMSE - lower is better) ---")
    for row in comparison_results:
        if 'old_rmse' in row:
            print(f"{row['dataset']:<30} OLD: {row['old_rmse']:.4f}  NEW: {row['new_rmse']:.4f}  Δ: {row['improvement_rmse']:.4f}")

    print("\n--- CLASSIFICATION (AUC - higher is better) ---")
    for row in comparison_results:
        if 'old_auc' in row:
            print(f"{row['dataset']:<30} OLD: {row['old_auc']:.4f}  NEW: {row['new_auc']:.4f}  Δ: +{row['improvement']:.4f}")

    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"Comparison saved to: {comparison_file}")
    print(f"{'='*70}")

    return results_file, comparison_file


if __name__ == "__main__":
    run_molclr_benchmark()
