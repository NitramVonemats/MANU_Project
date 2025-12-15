"""
Download and prepare ADME datasets from TDC.
"""
import os
import pandas as pd
from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


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


def download_and_prepare_adme_dataset(dataset_name, output_dir='datasets/adme'):
    """Download and prepare an ADME dataset."""
    print(f"\nDownloading {dataset_name}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download dataset
    data = ADME(name=dataset_name)
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
        print(f"    Mean: {result_df['Y'].mean():.4f}")
        print(f"    Std: {result_df['Y'].std():.4f}")
        print(f"    Min: {result_df['Y'].min():.4f}")
        print(f"    Max: {result_df['Y'].max():.4f}")

    return result_df


def main():
    """Download all ADME datasets."""
    print("="*80)
    print("DOWNLOADING ADME DATASETS FROM TDC")
    print("="*80)

    # Define ADME datasets
    datasets_to_download = [
        'Caco2_Wang',
        'Clearance_Hepatocyte_AZ',
        'Clearance_Microsome_AZ',
        'Half_Life_Obach'
    ]

    results = {}
    for dataset_name in datasets_to_download:
        try:
            df = download_and_prepare_adme_dataset(dataset_name)
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
    print(f"\nTotal molecules across all ADME datasets: {total}")
    print("\nFiles saved to: datasets/adme/")


if __name__ == "__main__":
    main()
