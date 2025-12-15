import os
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tdc.single_pred import ADME
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from adme_gnn.models.foundation import (
    ChemBERTaEncoder, BioMedEncoder, MolCLREncoder, MolEEncoder, MorganFingerprintEncoder
)

def get_dataset(name, split_method='scaffold'):
    """Load dataset from TDC (ADME or Toxicity)"""
    # Classification datasets
    if name.lower() in ['tox21', 'herg', 'clintox']:
        from tdc.single_pred import Tox
        data = Tox(name=name)
    else:
        # Regression datasets (ADME)
        data = ADME(name=name)
    split = data.get_split(method=split_method)
    return split

def is_classification_task(dataset_name):
    """Check if dataset is classification"""
    return dataset_name.lower() in ['tox21', 'herg', 'clintox']

def evaluate_regression(y_true, y_pred, scaler=None):
    """Evaluate regression model"""
    if scaler:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def evaluate_classification(y_true, y_pred_proba):
    """Evaluate classification model"""
    y_pred = (y_pred_proba > 0.5).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {'auc_roc': auc, 'accuracy': acc, 'f1': f1}

def run_benchmark(datasets, models, output_dir='results/foundation_benchmark'):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    results = []
    
    for dataset_name in datasets:
        print(f"\n{'='*50}\nProcessing {dataset_name}\n{'='*50}")
        try:
            split = get_dataset(dataset_name)
            train_df, valid_df, test_df = split['train'], split['valid'], split['test']
            
            # Check if classification or regression
            is_classification = is_classification_task(dataset_name)

            # Prepare targets
            y_train = train_df['Y'].values
            y_valid = valid_df['Y'].values
            y_test = test_df['Y'].values

            # Scaling only for regression
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
            
            for model_class, kwargs in models.items():
                print(f"\nEvaluating {model_class.__name__}...")
                try:
                    encoder = model_class(**kwargs).to(device)
                    if not encoder.enabled:
                        print(f"Skipping {encoder.name} (not enabled)")
                        continue
                        
                    # Extract embeddings
                    print("Extracting embeddings...")
                    t0 = time.time()
                    
                    # Helper to batch process
                    def get_embeddings(smiles_list):
                        embeddings = []
                        batch_size = 32
                        for i in range(0, len(smiles_list), batch_size):
                            batch = smiles_list[i:i+batch_size]
                            with torch.no_grad():
                                emb = encoder(batch, device)
                                embeddings.append(emb.cpu().numpy())
                        return np.vstack(embeddings)

                    X_train = get_embeddings(train_df['Drug'].tolist())
                    X_valid = get_embeddings(valid_df['Drug'].tolist())
                    X_test = get_embeddings(test_df['Drug'].tolist())
                    
                    extract_time = time.time() - t0
                    print(f"Embedding extraction took {extract_time:.2f}s")
                    
                    # Train downstream predictor
                    print("Training downstream predictor...")
                    if is_classification:
                        predictor = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True, random_state=42)
                    else:
                        predictor = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True, random_state=42)

                    predictor.fit(X_train, y_train_scaled)

                    # Evaluate
                    if is_classification:
                        val_preds_proba = predictor.predict_proba(X_valid)[:, 1]
                        test_preds_proba = predictor.predict_proba(X_test)[:, 1]
                        val_metrics = evaluate_classification(y_valid_scaled, val_preds_proba)
                        test_metrics = evaluate_classification(y_test_scaled, test_preds_proba)
                        print(f"Test AUC: {test_metrics['auc_roc']:.4f}, Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

                        results.append({
                            'dataset': dataset_name,
                            'model': encoder.name,
                            'task': 'classification',
                            'val_auc': val_metrics['auc_roc'],
                            'val_acc': val_metrics['accuracy'],
                            'test_auc': test_metrics['auc_roc'],
                            'test_acc': test_metrics['accuracy'],
                            'test_f1': test_metrics['f1'],
                            'extract_time': extract_time
                        })
                    else:
                        val_preds = predictor.predict(X_valid)
                        test_preds = predictor.predict(X_test)
                        val_metrics = evaluate_regression(y_valid_scaled, val_preds, scaler)
                        test_metrics = evaluate_regression(y_test_scaled, test_preds, scaler)
                        print(f"Test RMSE: {test_metrics['rmse']:.4f}, R2: {test_metrics['r2']:.4f}")

                        results.append({
                            'dataset': dataset_name,
                            'model': encoder.name,
                            'task': 'regression',
                            'val_rmse': val_metrics['rmse'],
                            'val_r2': val_metrics['r2'],
                            'test_rmse': test_metrics['rmse'],
                            'test_mae': test_metrics['mae'],
                            'test_r2': test_metrics['r2'],
                            'extract_time': extract_time
                        })
                    
                except Exception as e:
                    print(f"Error evaluating {model_class.__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")

    # Save results
    if results:
        df_results = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/benchmark_results_{timestamp}.csv"
        df_results.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        return filename
    return None

if __name__ == "__main__":
    # All 7 datasets (4 ADME + 3 Tox)
    datasets = [
        # ADME (regression)
        "Caco2_Wang",
        "Half_Life_Obach",
        "Clearance_Hepatocyte_AZ",
        "Clearance_Microsome_AZ",
        # Toxicity (classification)
        "tox21",
        "herg",
        "clintox"
    ]

    models = {
        MorganFingerprintEncoder: {'n_bits': 2048},
        ChemBERTaEncoder: {'model_name': 'seyonec/ChemBERTa-zinc-base-v1'},
        # BioMedEncoder: {'model_name': 'ibm-research/biomed.sm.mv-te-84m'},  # Skip if slow/failing
        # MolCLREncoder: {}, # Requires graph input handling update
        # MolEEncoder: {}    # Requires weights
    }

    print("="*70)
    print("FOUNDATION MODELS BENCHMARK - All 7 Datasets")
    print("="*70)
    print(f"\nDatasets: {len(datasets)}")
    for i, ds in enumerate(datasets, 1):
        print(f"  {i}. {ds}")
    print(f"\nModels: {len(models)}")
    for model_class in models.keys():
        print(f"  - {model_class.__name__}")
    print("\n" + "="*70 + "\n")

    run_benchmark(datasets, models)
