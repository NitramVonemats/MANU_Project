"""
Download and prepare Toxicity datasets from TDC.
"""
import os
import pandas as pd
from tdc.single_pred import Tox
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def compute_molecular_features(smiles):
    """Compute molecular descriptors from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
        'NumAromaticRings': Lipinski.NumAromaticRings(mol),
        'FractionCSP3': Lipinski.FractionCSP3(mol),
        'NumHeavyAtoms': Lipinski.HeavyAtomCount(mol),
        'NumRings': Lipinski.RingCount(mol)
    }


def download_and_prepare_tox_dataset(dataset_name, label_name=None, output_dir='datasets/toxicity'):
    """Download and prepare a toxicity dataset."""
    print(f"\nDownloading {dataset_name}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download dataset
    if label_name:
        data = Tox(name=dataset_name, label_name=label_name)
        print(f"  Using label: {label_name}")
    else:
        data = Tox(name=dataset_name)
    df = data.get_data()

    print(f"  Loaded {len(df)} molecules")
    print(f"  Columns: {df.columns.tolist()}")

    # Rename columns to standard format
    if 'Drug' in df.columns:
        df.rename(columns={'Drug': 'SMILES'}, inplace=True)

    # Compute molecular features
    print(f"  Computing molecular features...")
    features_list = []
    valid_indices = []

    for idx, smiles in enumerate(df['SMILES']):
        features = compute_molecular_features(smiles)
        if features is not None:
            features_list.append(features)
            valid_indices.append(idx)

        if (idx + 1) % 500 == 0:
            print(f"    Processed {idx + 1}/{len(df)} molecules...")

    # Filter valid molecules
    df = df.iloc[valid_indices].reset_index(drop=True)
    features_df = pd.DataFrame(features_list)

    # Combine
    result_df = pd.concat([df, features_df], axis=1)

    # Save
    output_path = os.path.join(output_dir, f'{dataset_name}.csv')
    result_df.to_csv(output_path, index=False)

    print(f"  Saved {len(result_df)} valid molecules to {output_path}")
    print(f"  Columns: {result_df.columns.tolist()}")

    # Print statistics
    if 'Y' in result_df.columns:
        print(f"\n  Label statistics:")
        print(f"    Unique values: {result_df['Y'].unique()}")
        print(f"    Value counts:\n{result_df['Y'].value_counts()}")
        if result_df['Y'].dtype in ['int64', 'float64']:
            balance = result_df['Y'].value_counts(normalize=True)
            print(f"    Class balance: {balance.to_dict()}")

    return result_df


def main():
    """Download all toxicity datasets."""
    print("="*80)
    print("DOWNLOADING TOXICITY DATASETS FROM TDC")
    print("="*80)

    # Define datasets with optional label names
    datasets_to_download = [
        ('Tox21', 'NR-AR'),  # Nuclear Receptor - Androgen Receptor
        ('hERG', None),      # hERG cardiac toxicity
        ('ClinTox', None),   # Clinical trial toxicity
    ]

    results = {}
    for dataset_config in datasets_to_download:
        if isinstance(dataset_config, tuple):
            dataset_name, label_name = dataset_config
        else:
            dataset_name, label_name = dataset_config, None

        try:
            df = download_and_prepare_tox_dataset(dataset_name, label_name=label_name)
            results[dataset_name] = df
        except Exception as e:
            print(f"\n[ERROR] Failed to download {dataset_name}: {e}")
            continue

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for dataset_name, df in results.items():
        print(f"{dataset_name}: {len(df)} molecules")

    total = sum(len(df) for df in results.values())
    print(f"\nTotal molecules across all Tox datasets: {total}")
    print("\nFiles saved to: datasets/toxicity/")


if __name__ == "__main__":
    main()
