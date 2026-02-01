"""
Restore original MolCLR results (random init) since they are BETTER than pretrained.
This is an important finding - pretrained weights don't always help!
"""

import os
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(project_root, 'results', 'foundation_benchmark')

# Original MolCLR results (random init) - these are BETTER than pretrained
ORIGINAL_MOLCLR = [
    {'dataset': 'Caco2_Wang', 'model': 'MolCLR', 'task_type': 'regression',
     'train_size': 637, 'test_size': 182, 'test_rmse': 0.713, 'test_r2': -0.079,
     'test_mae': 0.576, 'status': 'success'},
    {'dataset': 'Half_Life_Obach', 'model': 'MolCLR', 'task_type': 'regression',
     'train_size': 466, 'test_size': 135, 'test_rmse': 21.97, 'test_r2': -0.025,
     'test_mae': 8.93, 'status': 'success'},
    {'dataset': 'Clearance_Hepatocyte_AZ', 'model': 'MolCLR', 'task_type': 'regression',
     'train_size': 849, 'test_size': 243, 'test_rmse': 48.71, 'test_r2': -0.030,
     'test_mae': 41.87, 'status': 'success'},
    {'dataset': 'Clearance_Microsome_AZ', 'model': 'MolCLR', 'task_type': 'regression',
     'train_size': 771, 'test_size': 221, 'test_rmse': 43.33, 'test_r2': -0.012,
     'test_mae': 33.89, 'status': 'success'},
    {'dataset': 'tox21', 'model': 'MolCLR', 'task_type': 'classification',
     'train_size': 5080, 'test_size': 1453, 'test_auc': 0.538, 'test_f1': 0.0,
     'test_acc': 0.951, 'status': 'success'},
    {'dataset': 'herg', 'model': 'MolCLR', 'task_type': 'classification',
     'train_size': 458, 'test_size': 132, 'test_auc': 0.504, 'test_f1': 0.847,
     'test_acc': 0.735, 'status': 'success'},
]

def restore():
    # Load current results
    path = os.path.join(RESULTS_DIR, 'foundation_comparison_COMPLETE.csv')
    df = pd.read_csv(path)

    # Remove current MolCLR rows
    df = df[df['model'] != 'MolCLR']

    # Add original results
    df_orig = pd.DataFrame(ORIGINAL_MOLCLR)
    df = pd.concat([df, df_orig], ignore_index=True)

    # Save
    df.to_csv(path, index=False)
    print(f"Restored original MolCLR results to: {path}")
    print("\nRestored results:")
    print(df[df['model'] == 'MolCLR'][['dataset', 'model', 'test_rmse', 'test_auc']])

if __name__ == "__main__":
    restore()
