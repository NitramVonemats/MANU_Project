"""
Training orchestrator for foundation models with optimized hyperparameters.

This module handles:
  1. Training with best hyperparameters discovered during HPO
  2. Evaluation on test set
  3. Saving results in JSON format compatible with GNN results
"""
import json
import os
import time
import numpy as np
import torch
from typing import Literal, Optional, Dict, Any
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, accuracy_score, f1_score

from adme_gnn.models.foundation import (
    MorganFingerprintEncoder,
    ChemBERTaEncoder,
    BioMedEncoder,
    MolCLREncoder,
    MolEEncoder,
)
from optimization.foundation_problem import get_dataset, is_classification_dataset

ModelType = Literal["morgan", "chemberta", "biomed", "molclr", "mole"]


def train_foundation_model_with_best(
    model_type: ModelType,
    best_params: Dict[str, Any],
    dataset: str,
    max_iter: int = 500,
    seed: int = 42,
    device: str = "auto",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train foundation model with best hyperparameters and evaluate on test set.
    
    Args:
        model_type: Type of foundation model
        best_params: Best hyperparameters from HPO
        dataset: Dataset name
        max_iter: Max iterations for sklearn predictor
        seed: Random seed
        device: Device for encoder
        verbose: Print progress
        
    Returns:
        Dictionary with training results
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    is_classification = is_classification_dataset(dataset)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()} on {dataset}")
        print(f"{'='*70}")
        print(f"Best hyperparameters: {best_params}")
    
    # Load dataset
    split = get_dataset(dataset)
    train_df = split['train']
    valid_df = split['valid']
    test_df = split['test']
    
    train_smiles = train_df['Drug'].tolist()
    valid_smiles = valid_df['Drug'].tolist()
    test_smiles = test_df['Drug'].tolist()
    
    y_train = train_df['Y'].values
    y_valid = valid_df['Y'].values
    y_test = test_df['Y'].values
    
    # Scaling (only for regression)
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
    
    # Create encoder
    proj_dim = best_params['proj_dim']
    
    if model_type == "morgan":
        encoder = MorganFingerprintEncoder(
            n_bits=best_params['n_bits'],
            radius=best_params['radius'],
            proj_dim=proj_dim
        )
    elif model_type == "chemberta":
        encoder = ChemBERTaEncoder(
            model_name="seyonec/ChemBERTa-zinc-base-v1",
            proj_dim=proj_dim
        )
    elif model_type == "biomed":
        encoder = BioMedEncoder(
            model_name="ibm-research/biomed.sm.mv-te-84m",
            proj_dim=proj_dim
        )
    elif model_type == "molclr":
        encoder = MolCLREncoder(proj_dim=proj_dim)
    elif model_type == "mole":
        encoder = MolEEncoder(proj_dim=proj_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    encoder = encoder.to(device)
    
    # Extract embeddings
    if verbose:
        print("\nExtracting embeddings...")
    
    start_time = time.time()
    
    def extract_embeddings(smiles_list):
        embeddings = []
        batch_size = 32
        encoder.eval()
        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i+batch_size]
                emb = encoder(batch, device)
                embeddings.append(emb.cpu().numpy())
        return np.vstack(embeddings)
    
    X_train = extract_embeddings(train_smiles)
    X_valid = extract_embeddings(valid_smiles)
    X_test = extract_embeddings(test_smiles)
    
    extract_time = time.time() - start_time
    if verbose:
        print(f"Embedding extraction took {extract_time:.2f}s")
    
    # Build and train predictor
    if verbose:
        print("\nTraining predictor...")
    
    train_start = time.time()
    
    hidden_dims = best_params['hidden_dims']
    alpha = best_params['weight_decay']
    lr = best_params['lr']
    
    if is_classification:
        predictor = MLPClassifier(
            hidden_layer_sizes=hidden_dims,
            max_iter=max_iter,
            early_stopping=True,
            random_state=seed,
            alpha=alpha,
            learning_rate_init=lr,
            verbose=verbose,
        )
    else:
        predictor = MLPRegressor(
            hidden_layer_sizes=hidden_dims,
            max_iter=max_iter,
            early_stopping=True,
            random_state=seed,
            alpha=alpha,
            learning_rate_init=lr,
            verbose=verbose,
        )
    
    predictor.fit(X_train, y_train_scaled)
    
    train_time = time.time() - train_start
    
    # Evaluate
    if verbose:
        print("\nEvaluating...")
    
    if is_classification:
        # Classification metrics
        val_pred_proba = predictor.predict_proba(X_valid)[:, 1]
        test_pred_proba = predictor.predict_proba(X_test)[:, 1]
        
        val_pred = (val_pred_proba > 0.5).astype(int)
        test_pred = (test_pred_proba > 0.5).astype(int)
        
        val_metrics = {
            "auc_roc": float(roc_auc_score(y_valid, val_pred_proba)),
            "accuracy": float(accuracy_score(y_valid, val_pred)),
            "f1": float(f1_score(y_valid, val_pred, zero_division=0)),
        }
        
        test_metrics = {
            "auc_roc": float(roc_auc_score(y_test, test_pred_proba)),
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        }
        
        if verbose:
            print(f"\nValidation - AUC: {val_metrics['auc_roc']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"Test - AUC: {test_metrics['auc_roc']:.4f}, Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
    else:
        # Regression metrics
        val_pred = predictor.predict(X_valid)
        test_pred = predictor.predict(X_test)
        
        # Inverse transform for real-scale metrics
        y_valid_real = scaler.inverse_transform(y_valid_scaled.reshape(-1, 1)).flatten()
        y_test_real = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        val_pred_real = scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
        test_pred_real = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
        
        val_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_valid_real, val_pred_real))),
            "mae": float(mean_absolute_error(y_valid_real, val_pred_real)),
            "r2": float(r2_score(y_valid_real, val_pred_real)),
        }
        
        test_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test_real, test_pred_real))),
            "mae": float(mean_absolute_error(y_test_real, test_pred_real)),
            "r2": float(r2_score(y_test_real, test_pred_real)),
        }
        
        if verbose:
            print(f"\nValidation - RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
            print(f"Test - RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}, MAE: {test_metrics['mae']:.4f}")
    
    return {
        "model_type": model_type,
        "dataset": dataset,
        "train_time": float(train_time),
        "extract_time": float(extract_time),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "n_train": len(train_smiles),
        "n_valid": len(valid_smiles),
        "n_test": len(test_smiles),
    }


def save_foundation_results(
    model_type: ModelType,
    dataset: str,
    algo_name: str,
    best_params: Dict[str, Any],
    best_val_loss: float,
    trials: int,
    result: Dict[str, Any],
    out_dir: str = "runs/foundation",
    seed: int = 42,
) -> str:
    """
    Save foundation model HPO results in JSON format (compatible with GNN results).
    
    Args:
        model_type: Foundation model type
        dataset: Dataset name
        algo_name: HPO algorithm name
        best_params: Best hyperparameters
        best_val_loss: Best validation loss from HPO search
        trials: Number of trials
        result: Training result dict
        out_dir: Output directory
        seed: Random seed
        
    Returns:
        Path to saved JSON file
    """
    out_dir_dataset = os.path.join(out_dir, dataset)
    os.makedirs(out_dir_dataset, exist_ok=True)
    out_path = os.path.join(out_dir_dataset, f"hpo_{dataset}_{model_type}_{algo_name}.json")
    
    payload = {
        "model_type": model_type,
        "algo": algo_name,
        "dataset": dataset,
        "seed": seed,
        "search": {
            "trials": int(trials),
            "best_val_loss": float(best_val_loss),
            "best_params": {k: (int(v) if isinstance(v, (np.integer, np.int64)) else 
                               (float(v) if isinstance(v, (np.floating, np.float64)) else 
                               (tuple(int(x) for x in v) if isinstance(v, tuple) else v)))
                           for k, v in best_params.items()},
        },
        "final_training": {
            "train_time": result["train_time"],
            "extract_time": result["extract_time"],
            "val_metrics": result["val_metrics"],
            "test_metrics": result["test_metrics"],
            "n_train": result["n_train"],
            "n_valid": result["n_valid"],
            "n_test": result["n_test"],
        },
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    
    return os.path.abspath(out_path)
