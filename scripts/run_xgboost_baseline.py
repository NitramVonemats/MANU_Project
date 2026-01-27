#!/usr/bin/env python3
"""
XGBoost Baseline with Morgan Fingerprints
==========================================
Classical ML baseline for comparison with GNN models.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[ERROR] XGBoost not installed")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[ERROR] RDKit not installed")

from optimized_gnn import is_classification_dataset

RESULTS_DIR = PROJECT_ROOT / "results" / "baselines"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def load_tdc_dataset(dataset_name: str, seed: int = 42):
    """Load dataset from TDC with same splits as GNN."""
    from tdc.single_pred import ADME, Tox

    # Map dataset names
    if dataset_name.lower() == "herg":
        data = Tox(name="hERG")
    elif dataset_name.lower() == "tox21":
        data = Tox(name="Tox21", label_name="NR-AR")
    elif dataset_name.lower() == "caco2_wang":
        data = ADME(name="Caco2_Wang")
    elif dataset_name.lower() == "half_life_obach":
        data = ADME(name="Half_Life_Obach")
    elif dataset_name.lower() == "clearance_hepatocyte_az":
        data = ADME(name="Clearance_Hepatocyte_AZ")
    elif dataset_name.lower() == "clearance_microsome_az":
        data = ADME(name="Clearance_Microsome_AZ")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Get scaffold split
    split = data.get_split(method="scaffold", seed=seed)

    return split["train"], split["test"]


def prepare_features(df: pd.DataFrame, radius: int = 2, n_bits: int = 2048):
    """Convert SMILES to Morgan fingerprints."""
    X = []
    y = []

    for idx, row in df.iterrows():
        smiles = row["Drug"]
        fp = smiles_to_morgan(smiles, radius, n_bits)
        if np.any(fp):  # Valid fingerprint
            X.append(fp)
            y.append(row["Y"])

    return np.array(X), np.array(y)


def run_xgboost_experiment(
    dataset_name: str,
    seed: int = 42,
    val_fraction: float = 0.1,
):
    """Run XGBoost baseline experiment."""
    if not XGBOOST_AVAILABLE or not RDKIT_AVAILABLE:
        raise ImportError("XGBoost and RDKit required")

    print(f"\n{'='*50}")
    print(f"XGBoost Baseline: {dataset_name}")
    print(f"{'='*50}")

    is_classification = is_classification_dataset(dataset_name)

    # Load data
    train_df, test_df = load_tdc_dataset(dataset_name, seed=seed)

    print(f"Raw data: train={len(train_df)}, test={len(test_df)}")

    # Split train into train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(train_df, test_size=val_fraction, random_state=seed)

    print(f"After split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    print(f"After filtering: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Build XGBoost model
    if is_classification:
        # Calculate class weight
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        print(f"Class balance: pos={n_pos}, neg={n_neg}, scale_pos_weight={scale_pos_weight:.2f}")

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
        )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    if is_classification:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_pred = model.predict(X_val)

        results = {
            "val_f1": f1_score(y_val, val_pred, zero_division=0),
            "val_auc": roc_auc_score(y_val, val_pred_proba) if len(np.unique(y_val)) > 1 else 0,
            "val_accuracy": accuracy_score(y_val, val_pred),
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
            "test_auc": roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
        }

        print(f"\nResults:")
        print(f"  Val  F1: {results['val_f1']:.4f}, AUC: {results['val_auc']:.4f}")
        print(f"  Test F1: {results['test_f1']:.4f}, AUC: {results['test_auc']:.4f}")

    else:
        y_pred = model.predict(X_test)
        val_pred = model.predict(X_val)

        results = {
            "val_rmse": np.sqrt(mean_squared_error(y_val, val_pred)),
            "val_mae": mean_absolute_error(y_val, val_pred),
            "val_r2": r2_score(y_val, val_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_r2": r2_score(y_test, y_pred),
        }

        print(f"\nResults:")
        print(f"  Val  RMSE: {results['val_rmse']:.4f}, R2: {results['val_r2']:.4f}")
        print(f"  Test RMSE: {results['test_rmse']:.4f}, R2: {results['test_r2']:.4f}")

    return results, model


def run_multiseed_validation(dataset_name: str, seeds: list = [42, 43, 44]):
    """Run XGBoost with multiple seeds."""
    all_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        results, _ = run_xgboost_experiment(dataset_name, seed=seed)
        all_results.append(results)

    # Compute summary
    is_classification = is_classification_dataset(dataset_name)

    if is_classification:
        summary = {
            "test_f1_mean": np.mean([r["test_f1"] for r in all_results]),
            "test_f1_std": np.std([r["test_f1"] for r in all_results]),
            "test_auc_mean": np.mean([r["test_auc"] for r in all_results]),
            "test_auc_std": np.std([r["test_auc"] for r in all_results]),
        }
    else:
        summary = {
            "test_rmse_mean": np.mean([r["test_rmse"] for r in all_results]),
            "test_rmse_std": np.std([r["test_rmse"] for r in all_results]),
            "test_r2_mean": np.mean([r["test_r2"] for r in all_results]),
            "test_r2_std": np.std([r["test_r2"] for r in all_results]),
        }

    summary["raw_results"] = all_results
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost Baseline")
    parser.add_argument("--dataset", type=str, default="herg")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    args = parser.parse_args()

    if args.all:
        datasets = ["herg", "Caco2_Wang", "tox21", "Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"]
    else:
        datasets = [args.dataset]

    all_results = {}

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")

        summary = run_multiseed_validation(ds)
        all_results[ds] = summary

        # Print summary
        is_cls = is_classification_dataset(ds)
        if is_cls:
            print(f"\n[SUMMARY] {ds}:")
            print(f"  Test F1:  {summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}")
            print(f"  Test AUC: {summary['test_auc_mean']:.4f} ± {summary['test_auc_std']:.4f}")
        else:
            print(f"\n[SUMMARY] {ds}:")
            print(f"  Test RMSE: {summary['test_rmse_mean']:.4f} ± {summary['test_rmse_std']:.4f}")
            print(f"  Test R²:   {summary['test_r2_mean']:.4f} ± {summary['test_r2_std']:.4f}")

    # Save results
    output_path = RESULTS_DIR / "xgboost_baseline_results.json"

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\n[OK] Results saved: {output_path}")


if __name__ == "__main__":
    main()
