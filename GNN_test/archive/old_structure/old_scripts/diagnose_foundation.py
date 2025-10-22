"""
FOUNDATION DIAGNOSTICS
======================
Diagnose why R² is negative and establish proper baselines.

This script will:
1. Check train/val/test distribution shifts
2. Implement simple baselines (mean, median, RF, XGBoost)
3. Implement molecular fingerprint baselines (ECFP)
4. Compare scaffold vs random splits
5. Analyze prediction errors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===================== IMPORTS =====================
try:
    from tdc.single_pred import ADME
    TDC_OK = True
except:
    TDC_OK = False
    print("WARNING: TDC not available")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKIT_OK = True
except:
    RDKIT_OK = False
    print("WARNING: RDKit not available")

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_OK = True
except:
    SKLEARN_OK = False
    print("WARNING: sklearn not available")

try:
    import xgboost as xgb
    XGBOOST_OK = True
except:
    XGBOOST_OK = False
    print("WARNING: XGBoost not available. Install: pip install xgboost")

# ===================== UTILS =====================
def compute_metrics(y_true, y_pred):
    """Compute comprehensive metrics"""
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Log-space metrics
    y_true_log = np.log1p(np.maximum(0, y_true))
    y_pred_log = np.log1p(np.maximum(0, y_pred))
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))

    # Correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'rmse_log': rmse_log,
        'correlation': corr,
        'mean_pred': y_pred.mean(),
        'std_pred': y_pred.std(),
        'mean_true': y_true.mean(),
        'std_true': y_true.std()
    }

def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Generate Morgan (ECFP) fingerprint"""
    if not RDKIT_OK:
        return np.zeros(n_bits)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def get_rdkit_descriptors(smiles):
    """Generate comprehensive RDKit descriptors"""
    if not RDKIT_OK:
        return np.zeros(30)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(30)

    try:
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            Descriptors.MolMR(mol),
            Descriptors.LabuteASA(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.BertzCT(mol),
            Descriptors.Chi0v(mol),
            Descriptors.Chi1v(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.NumHeteroatoms(mol),
            rdMolDescriptors.CalcNumHeterocycles(mol),
            rdMolDescriptors.CalcNumRings(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumValenceElectrons(mol),
            Descriptors.NumRadicalElectrons(mol),
            Descriptors.MaxPartialCharge(mol),
            Descriptors.MinPartialCharge(mol),
            rdMolDescriptors.CalcNumAmideBonds(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            rdMolDescriptors.CalcNumSpiroAtoms(mol),
            Descriptors.FractionCsp3(mol),
            rdMolDescriptors.CalcLabuteASA(mol),
        ]
        return np.array(descriptors[:30])
    except:
        return np.zeros(30)

# ===================== DIAGNOSTIC FUNCTIONS =====================
def diagnose_distributions(dataset_name, split_type='scaffold'):
    """Diagnose train/val/test distribution differences"""
    print(f"\n{'='*80}")
    print(f"DISTRIBUTION ANALYSIS: {dataset_name} ({split_type} split)")
    print(f"{'='*80}")

    if not TDC_OK:
        print("TDC not available, skipping...")
        return None

    data_api = ADME(name=dataset_name)
    try:
        split = data_api.get_split(method=split_type)
    except:
        split = data_api.get_split()

    train_df = split['train']
    val_df = split.get('valid', split.get('val'))
    test_df = split['test']

    # Extract Y values
    y_train = train_df['Y'].values
    y_val = val_df['Y'].values if val_df is not None else np.array([])
    y_test = test_df['Y'].values

    print(f"\nDataset sizes:")
    print(f"  Train: {len(y_train)}")
    print(f"  Val:   {len(y_val) if len(y_val) > 0 else 'N/A'}")
    print(f"  Test:  {len(y_test)}")

    print(f"\nTarget statistics (original space):")
    print(f"  Train: mean={y_train.mean():.3f}, std={y_train.std():.3f}, min={y_train.min():.3f}, max={y_train.max():.3f}")
    if len(y_val) > 0:
        print(f"  Val:   mean={y_val.mean():.3f}, std={y_val.std():.3f}, min={y_val.min():.3f}, max={y_val.max():.3f}")
    print(f"  Test:  mean={y_test.mean():.3f}, std={y_test.std():.3f}, min={y_test.min():.3f}, max={y_test.max():.3f}")

    print(f"\nTarget statistics (log space):")
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    print(f"  Train: mean={y_train_log.mean():.3f}, std={y_train_log.std():.3f}")
    print(f"  Test:  mean={y_test_log.mean():.3f}, std={y_test_log.std():.3f}")

    # Check for distribution shift (KS test)
    try:
        from scipy.stats import ks_2samp
        ks_stat, ks_pval = ks_2samp(y_train, y_test)
        print(f"\nKolmogorov-Smirnov test (train vs test):")
        print(f"  Statistic: {ks_stat:.4f}")
        print(f"  P-value: {ks_pval:.4f}")
        if ks_pval < 0.05:
            print(f"  ⚠️  WARNING: Significant distribution shift detected!")
        else:
            print(f"  ✓ No significant distribution shift")
    except:
        print("\n  (scipy not available for KS test)")

    # Overlap analysis
    train_min, train_max = y_train.min(), y_train.max()
    test_min, test_max = y_test.min(), y_test.max()
    test_outside_train = np.sum((y_test < train_min) | (y_test > train_max))
    print(f"\nRange overlap:")
    print(f"  Train range: [{train_min:.3f}, {train_max:.3f}]")
    print(f"  Test range:  [{test_min:.3f}, {test_max:.3f}]")
    print(f"  Test samples outside train range: {test_outside_train}/{len(y_test)} ({100*test_outside_train/len(y_test):.1f}%)")
    if test_outside_train > len(y_test) * 0.1:
        print(f"  ⚠️  WARNING: >10% of test samples outside training range!")

    return {
        'train': y_train,
        'val': y_val,
        'test': y_test,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }

def test_simple_baselines(data, dataset_name):
    """Test simple baselines: mean, median, linear"""
    print(f"\n{'='*80}")
    print(f"SIMPLE BASELINES: {dataset_name}")
    print(f"{'='*80}")

    y_train = data['train']
    y_test = data['test']

    results = []

    # Mean baseline
    y_pred_mean = np.full_like(y_test, y_train.mean())
    metrics_mean = compute_metrics(y_test, y_pred_mean)
    results.append({'model': 'Mean', **metrics_mean})
    print(f"\nMean Baseline:")
    print(f"  RMSE: {metrics_mean['rmse']:.3f}, MAE: {metrics_mean['mae']:.3f}, R2: {metrics_mean['r2']:.3f}")

    # Median baseline
    y_pred_median = np.full_like(y_test, np.median(y_train))
    metrics_median = compute_metrics(y_test, y_pred_median)
    results.append({'model': 'Median', **metrics_median})
    print(f"\nMedian Baseline:")
    print(f"  RMSE: {metrics_median['rmse']:.3f}, MAE: {metrics_median['mae']:.3f}, R2: {metrics_median['r2']:.3f}")

    return pd.DataFrame(results)

def test_ml_baselines(data, dataset_name):
    """Test ML baselines: RF, XGBoost with descriptors and fingerprints"""
    print(f"\n{'='*80}")
    print(f"ML BASELINES: {dataset_name}")
    print(f"{'='*80}")

    if not SKLEARN_OK or not RDKIT_OK:
        print("sklearn or RDKit not available, skipping...")
        return None

    train_df = data['train_df']
    test_df = data['test_df']
    y_train = data['train']
    y_test = data['test']

    # Generate features
    print("\nGenerating features...")
    print("  - RDKit descriptors (30 features)")
    X_train_desc = np.array([get_rdkit_descriptors(s) for s in train_df['Drug'].values])
    X_test_desc = np.array([get_rdkit_descriptors(s) for s in test_df['Drug'].values])

    print("  - Morgan fingerprints (2048 bits, radius=2)")
    X_train_fp = np.array([get_morgan_fingerprint(s, radius=2, n_bits=2048) for s in train_df['Drug'].values])
    X_test_fp = np.array([get_morgan_fingerprint(s, radius=2, n_bits=2048) for s in test_df['Drug'].values])

    print("  - Combined features (descriptors + fingerprints)")
    X_train_combined = np.hstack([X_train_desc, X_train_fp])
    X_test_combined = np.hstack([X_test_desc, X_test_fp])

    # Log-transform targets (common for ADME)
    y_train_log = np.log1p(np.maximum(0, y_train))

    results = []

    # Test different models and feature sets
    models = [
        ('RF', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)),
        ('RF_deep', RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42)),
        ('GBM', GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)),
    ]

    if XGBOOST_OK:
        models.append(('XGBoost', xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)))

    feature_sets = [
        ('Descriptors', X_train_desc, X_test_desc),
        ('Fingerprints', X_train_fp, X_test_fp),
        ('Combined', X_train_combined, X_test_combined),
    ]

    for model_name, model in models:
        for feat_name, X_tr, X_te in feature_sets:
            print(f"\nTraining {model_name} with {feat_name}...")

            # Train on log-space
            model.fit(X_tr, y_train_log)
            y_pred_log = model.predict(X_te)
            y_pred = np.expm1(y_pred_log)  # Back to original space

            metrics = compute_metrics(y_test, y_pred)
            results.append({
                'model': f"{model_name}_{feat_name}",
                **metrics
            })

            print(f"  RMSE: {metrics['rmse']:.3f}, MAE: {metrics['mae']:.3f}, R2: {metrics['r2']:.3f}, Corr: {metrics['correlation']:.3f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')

    print(f"\n{'='*60}")
    print("BEST BASELINES:")
    print(f"{'='*60}")
    for idx, row in results_df.head(3).iterrows():
        print(f"{row['model']:30} RMSE={row['rmse']:6.3f}  R2={row['r2']:6.3f}  Corr={row['correlation']:6.3f}")

    return results_df

def compare_split_methods(dataset_name):
    """Compare scaffold vs random split"""
    print(f"\n{'='*80}")
    print(f"SPLIT METHOD COMPARISON: {dataset_name}")
    print(f"{'='*80}")

    if not TDC_OK:
        print("TDC not available, skipping...")
        return None

    results = []

    for split_type in ['scaffold', 'random']:
        print(f"\n--- {split_type.upper()} SPLIT ---")
        data = diagnose_distributions(dataset_name, split_type)
        if data is None:
            continue

        # Test simple baseline on this split
        y_train = data['train']
        y_test = data['test']
        y_pred_mean = np.full_like(y_test, y_train.mean())
        metrics = compute_metrics(y_test, y_pred_mean)

        results.append({
            'split_type': split_type,
            'baseline': 'mean',
            **metrics
        })

    if results:
        results_df = pd.DataFrame(results)
        print(f"\n{'='*60}")
        print("SPLIT COMPARISON (Mean Baseline R2):")
        print(f"{'='*60}")
        for _, row in results_df.iterrows():
            print(f"{row['split_type']:15} R2={row['r2']:6.3f}  RMSE={row['rmse']:6.3f}")

    return results_df

# ===================== MAIN DIAGNOSTIC PIPELINE =====================
def run_full_diagnostics(datasets=None):
    """Run full diagnostic pipeline"""
    if datasets is None:
        datasets = ['Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    print(f"\n{'#'*80}")
    print(f"# FOUNDATION DIAGNOSTICS - {timestamp}")
    print(f"{'#'*80}")

    for dataset_name in datasets:
        print(f"\n\n{'='*80}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*80}")

        # 1. Distribution analysis
        data = diagnose_distributions(dataset_name, split_type='scaffold')
        if data is None:
            continue

        # 2. Simple baselines
        simple_results = test_simple_baselines(data, dataset_name)

        # 3. ML baselines
        ml_results = test_ml_baselines(data, dataset_name)

        # 4. Split comparison
        split_results = compare_split_methods(dataset_name)

        # Save results
        if ml_results is not None:
            ml_results['dataset'] = dataset_name
            all_results.append(ml_results)

    # Consolidate results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        output_file = f"diagnostic_results_{timestamp}.csv"
        final_results.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"RESULTS SAVED: {output_file}")
        print(f"{'='*80}")

        # Summary
        print("\n\nSUMMARY - BEST BASELINES PER DATASET:")
        print(f"{'='*80}")
        for dataset in datasets:
            subset = final_results[final_results['dataset'] == dataset]
            if len(subset) > 0:
                best = subset.loc[subset['rmse'].idxmin()]
                print(f"\n{dataset}:")
                print(f"  Best model: {best['model']}")
                print(f"  RMSE: {best['rmse']:.3f}")
                print(f"  R2: {best['r2']:.3f}")
                print(f"  Correlation: {best['correlation']:.3f}")

    print("\n\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nKEY INSIGHTS:")
    print("  1. Check if test R² is negative for simple baselines too")
    print("  2. If yes → distribution shift issue (scaffold split too harsh)")
    print("  3. If no → GNN overfitting issue (need stronger regularization)")
    print("  4. Compare GNN performance to best ML baseline")
    print("  5. Use best baseline as minimum target for GNN")

if __name__ == "__main__":
    run_full_diagnostics()
