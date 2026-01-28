"""
COMPLETE FOUNDATION MODEL BENCHMARK
====================================

Tests all 5 foundation models on all 6 datasets:
- Models: Morgan-FP, ChemBERTa, BioMed-IBM, MolCLR, MolE-FP
- ADME (regression): Caco2_Wang, Half_Life_Obach, Clearance_Hepatocyte_AZ, Clearance_Microsome_AZ
- Toxicity (classification): tox21, herg

Compares with GNN best results from HPO runs.
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from adme_gnn.models.foundation import (
    ChemBERTaEncoder, BioMedEncoder, MolCLREncoder,
    MolEEncoder, MorganFingerprintEncoder
)


# ============== CONFIGURATION ==============

DATASETS = [
    # ADME (regression)
    "Caco2_Wang",
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    # Toxicity (classification)
    "tox21",
    "herg",
]

MODELS = {
    'Morgan-FP': (MorganFingerprintEncoder, {'n_bits': 2048}),
    'ChemBERTa': (ChemBERTaEncoder, {'model_name': 'seyonec/ChemBERTa-zinc-base-v1'}),
    'BioMed-IBM': (BioMedEncoder, {'model_name': 'ibm-research/biomed.sm.mv-te-84m'}),
    'MolCLR': (MolCLREncoder, {'proj_dim': 256}),
    'MolE-FP': (MolEEncoder, {'proj_dim': 256}),
}

SEED = 42
OUTPUT_DIR = os.path.join(project_root, 'results', 'foundation_benchmark')
FIGURES_DIR = os.path.join(project_root, 'figures', 'foundation')


# ============== HELPER FUNCTIONS ==============

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset(name, split_method='scaffold'):
    """Load dataset from TDC"""
    # Classification datasets
    if name.lower() in ['tox21', 'herg', 'clintox']:
        from tdc.single_pred import Tox

        # Handle multi-task datasets
        tox_labels = {
            'tox21': 'NR-AR',
            'herg': None,
            'clintox': 'CT_TOX',
        }
        label = tox_labels.get(name.lower())
        if label:
            data = Tox(name=name, label_name=label)
        else:
            data = Tox(name=name)
    else:
        # Regression datasets (ADME)
        from tdc.single_pred import ADME
        data = ADME(name=name)

    split = data.get_split(method=split_method, seed=SEED)
    return split


def is_classification_task(dataset_name):
    """Check if dataset is classification"""
    return dataset_name.lower() in ['tox21', 'herg', 'clintox']


def evaluate_regression(y_true, y_pred):
    """Evaluate regression model"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def evaluate_classification(y_true, y_pred_proba):
    """Evaluate classification model"""
    y_pred = (y_pred_proba > 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = 0.5  # Default for edge cases
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {'auc_roc': auc, 'accuracy': acc, 'f1': f1}


def load_gnn_best_results():
    """Load best GNN results from HPO runs"""
    runs_dir = os.path.join(project_root, 'runs')
    gnn_results = {}

    for dataset in DATASETS:
        dataset_dir = os.path.join(runs_dir, dataset)
        if not os.path.exists(dataset_dir):
            continue

        best_result = None
        best_metric = float('inf') if not is_classification_task(dataset) else 0

        for filename in os.listdir(dataset_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(dataset_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    if is_classification_task(dataset):
                        # For classification, maximize AUC
                        metric = data.get('best_test_auc', data.get('test_auc', 0))
                        if metric > best_metric:
                            best_metric = metric
                            best_result = data
                    else:
                        # For regression, minimize RMSE
                        metric = data.get('best_test_rmse', data.get('test_rmse', float('inf')))
                        if metric < best_metric:
                            best_metric = metric
                            best_result = data
                except:
                    continue

        if best_result:
            gnn_results[dataset] = best_result

    return gnn_results


def get_embeddings_batch(encoder, smiles_list, device, batch_size=32):
    """Extract embeddings in batches"""
    embeddings = []

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        with torch.no_grad():
            try:
                emb = encoder(batch, device)
                embeddings.append(emb.cpu().numpy())
            except Exception as e:
                # Fallback: return zeros for failed batches
                print(f"    Warning: Batch {i//batch_size} failed: {e}")
                embeddings.append(np.zeros((len(batch), encoder.proj_dim)))

    return np.vstack(embeddings)


# ============== MAIN BENCHMARK ==============

def run_foundation_benchmark():
    """Run complete foundation model benchmark"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print("COMPLETE FOUNDATION MODEL BENCHMARK")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Models: {len(MODELS)}")
    print(f"Seed: {SEED}")
    print(f"{'='*70}\n")

    set_seed(SEED)

    # Load GNN best results for comparison
    print("Loading GNN best results from HPO...")
    gnn_results = load_gnn_best_results()
    print(f"Found GNN results for {len(gnn_results)} datasets\n")

    all_results = []

    # Run benchmark for each dataset
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

            # Handle NaN values
            train_mask = ~np.isnan(y_train)
            valid_mask = ~np.isnan(y_valid)
            test_mask = ~np.isnan(y_test)

            train_smiles = train_df['Drug'].values[train_mask].tolist()
            valid_smiles = valid_df['Drug'].values[valid_mask].tolist()
            test_smiles = test_df['Drug'].values[test_mask].tolist()

            y_train = y_train[train_mask]
            y_valid = y_valid[valid_mask]
            y_test = y_test[test_mask]

            # Scale targets for regression
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

            # Test each model
            for model_name, (model_class, kwargs) in MODELS.items():
                print(f"\n  Model: {model_name}")
                print(f"  {'-'*40}")

                result = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'task_type': task_type,
                    'train_size': len(train_smiles),
                    'test_size': len(test_smiles),
                }

                try:
                    # Initialize encoder
                    t0 = time.time()
                    encoder = model_class(**kwargs)

                    if hasattr(encoder, 'to'):
                        encoder = encoder.to(device)

                    if not encoder.enabled:
                        print(f"    SKIPPED (not enabled)")
                        result['status'] = 'skipped'
                        all_results.append(result)
                        continue

                    encoder.eval()

                    # Extract embeddings
                    print("    Extracting embeddings...")
                    X_train = get_embeddings_batch(encoder, train_smiles, device)
                    X_valid = get_embeddings_batch(encoder, valid_smiles, device)
                    X_test = get_embeddings_batch(encoder, test_smiles, device)

                    embed_time = time.time() - t0
                    print(f"    Embedding time: {embed_time:.1f}s")

                    # Train downstream predictor
                    print("    Training predictor...")
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

                        print(f"    Test AUC: {test_metrics['auc_roc']:.4f}")
                        print(f"    Test F1:  {test_metrics['f1']:.4f}")
                        print(f"    Test Acc: {test_metrics['accuracy']:.4f}")

                        result.update({
                            'test_auc': test_metrics['auc_roc'],
                            'test_f1': test_metrics['f1'],
                            'test_acc': test_metrics['accuracy'],
                            'valid_auc': valid_metrics['auc_roc'],
                        })
                    else:
                        test_preds = predictor.predict(X_test)
                        valid_preds = predictor.predict(X_valid)

                        # Inverse transform predictions
                        if scaler:
                            test_preds_orig = scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
                            valid_preds_orig = scaler.inverse_transform(valid_preds.reshape(-1, 1)).flatten()
                        else:
                            test_preds_orig = test_preds
                            valid_preds_orig = valid_preds

                        test_metrics = evaluate_regression(y_test, test_preds_orig)
                        valid_metrics = evaluate_regression(y_valid, valid_preds_orig)

                        print(f"    Test RMSE: {test_metrics['rmse']:.4f}")
                        print(f"    Test R2:   {test_metrics['r2']:.4f}")
                        print(f"    Test MAE:  {test_metrics['mae']:.4f}")

                        result.update({
                            'test_rmse': test_metrics['rmse'],
                            'test_r2': test_metrics['r2'],
                            'test_mae': test_metrics['mae'],
                            'valid_rmse': valid_metrics['rmse'],
                            'valid_r2': valid_metrics['r2'],
                        })

                    result['embed_time'] = embed_time
                    result['train_time'] = train_time
                    result['total_time'] = embed_time + train_time
                    result['status'] = 'success'

                except Exception as e:
                    print(f"    ERROR: {e}")
                    traceback.print_exc()
                    result['status'] = 'failed'
                    result['error'] = str(e)

                all_results.append(result)

                # Cleanup
                if 'encoder' in locals():
                    del encoder
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            traceback.print_exc()

    # Save results
    df_results = pd.DataFrame(all_results)

    # Add GNN results for comparison
    gnn_comparison = []
    for dataset in DATASETS:
        if dataset in gnn_results:
            gnn_data = gnn_results[dataset]
            is_class = is_classification_task(dataset)

            gnn_row = {
                'dataset': dataset,
                'model': 'GNN-Best',
                'task_type': 'classification' if is_class else 'regression',
                'status': 'success',
            }

            if is_class:
                gnn_row['test_auc'] = gnn_data.get('best_test_auc', gnn_data.get('test_auc'))
                gnn_row['test_f1'] = gnn_data.get('best_test_f1', gnn_data.get('test_f1'))
            else:
                gnn_row['test_rmse'] = gnn_data.get('best_test_rmse', gnn_data.get('test_rmse'))
                gnn_row['test_r2'] = gnn_data.get('best_test_r2', gnn_data.get('test_r2'))

            gnn_comparison.append(gnn_row)

    if gnn_comparison:
        df_gnn = pd.DataFrame(gnn_comparison)
        df_results = pd.concat([df_results, df_gnn], ignore_index=True)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(OUTPUT_DIR, f'foundation_comparison_COMPLETE_{timestamp}.csv')
    df_results.to_csv(results_file, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")

    # Also save a "latest" version
    latest_file = os.path.join(OUTPUT_DIR, 'foundation_comparison_COMPLETE.csv')
    df_results.to_csv(latest_file, index=False)
    print(f"Latest saved to: {latest_file}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")

    print("\n--- REGRESSION DATASETS (ADME) ---")
    print(f"{'Dataset':<25} {'Model':<15} {'RMSE':<12} {'R2':<12}")
    print("-" * 64)

    for dataset in DATASETS:
        if is_classification_task(dataset):
            continue
        subset = df_results[(df_results['dataset'] == dataset) & (df_results['status'] == 'success')]
        for _, row in subset.iterrows():
            rmse = row.get('test_rmse', 'N/A')
            r2 = row.get('test_r2', 'N/A')
            if isinstance(rmse, float):
                rmse = f"{rmse:.4f}"
            if isinstance(r2, float):
                r2 = f"{r2:.4f}"
            print(f"{dataset:<25} {row['model']:<15} {rmse:<12} {r2:<12}")
        print("-" * 64)

    print("\n--- CLASSIFICATION DATASETS (Toxicity) ---")
    print(f"{'Dataset':<25} {'Model':<15} {'AUC-ROC':<12} {'F1':<12}")
    print("-" * 64)

    for dataset in DATASETS:
        if not is_classification_task(dataset):
            continue
        subset = df_results[(df_results['dataset'] == dataset) & (df_results['status'] == 'success')]
        for _, row in subset.iterrows():
            auc = row.get('test_auc', 'N/A')
            f1 = row.get('test_f1', 'N/A')
            if isinstance(auc, float):
                auc = f"{auc:.4f}"
            if isinstance(f1, float):
                f1 = f"{f1:.4f}"
            print(f"{dataset:<25} {row['model']:<15} {auc:<12} {f1:<12}")
        print("-" * 64)

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*70}")

    return results_file


if __name__ == "__main__":
    run_foundation_benchmark()
