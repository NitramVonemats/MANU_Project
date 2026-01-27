"""
Verify class balance for classification datasets (Tox21, hERG)
Load datasets EXACTLY as training code does to ensure consistency.
Updated: 2026-01-20
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from optimized_gnn import prepare_dataset, is_classification_dataset


def verify_class_balance():
    """Load datasets using EXACT same pipeline as training and compute class balance."""

    print("=" * 80)
    print("CLASS BALANCE VERIFICATION (Using Training Pipeline)")
    print("Seed=42, val_fraction=0.1, scaffold split")
    print("=" * 80)

    results = {}

    # Classification datasets
    datasets = ['tox21', 'herg']

    for ds_name in datasets:
        print(f"\n{'=' * 70}")
        print(f"Loading {ds_name} via prepare_dataset()...")
        print("=" * 70)

        # Use EXACT same function as training
        cache = prepare_dataset(ds_name, val_fraction=0.1, seed=42, verbose=True)

        train_graphs = cache['train']
        val_graphs = cache['val']
        test_graphs = cache['test']

        # Extract labels
        train_labels = np.array([g.original_y for g in train_graphs])
        val_labels = np.array([g.original_y for g in val_graphs])
        test_labels = np.array([g.original_y for g in test_graphs])
        all_labels = np.concatenate([train_labels, val_labels, test_labels])

        # Compute statistics
        n_total = len(all_labels)
        n_train = len(train_labels)
        n_val = len(val_labels)
        n_test = len(test_labels)

        n_positive = int(np.sum(all_labels == 1))
        n_negative = int(np.sum(all_labels == 0))
        pos_rate = 100 * n_positive / n_total

        # Per-split statistics
        train_pos = int(np.sum(train_labels == 1))
        val_pos = int(np.sum(val_labels == 1))
        test_pos = int(np.sum(test_labels == 1))

        print(f"\nFINAL COUNTS (after SMILES filtering):")
        print(f"  Total: {n_total}")
        print(f"  Train: {n_train} (positive: {train_pos}, {100*train_pos/n_train:.1f}%)")
        print(f"  Val:   {n_val} (positive: {val_pos}, {100*val_pos/n_val:.1f}%)")
        print(f"  Test:  {n_test} (positive: {test_pos}, {100*test_pos/n_test:.1f}%)")
        print(f"\nOverall class balance:")
        print(f"  Positive: {n_positive} ({pos_rate:.1f}%)")
        print(f"  Negative: {n_negative} ({100-pos_rate:.1f}%)")

        results[ds_name] = {
            'total': n_total,
            'train': n_train,
            'val': n_val,
            'test': n_test,
            'positive': n_positive,
            'negative': n_negative,
            'pos_rate': pos_rate,
            'train_pos': train_pos,
            'val_pos': val_pos,
            'test_pos': test_pos,
        }

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (for paper)")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Total':<8} {'Train':<8} {'Val':<6} {'Test':<6} {'Pos%':<8}")
    print("-" * 60)
    for ds_name, r in results.items():
        print(f"{ds_name:<15} {r['total']:<8} {r['train']:<8} {r['val']:<6} {r['test']:<6} {r['pos_rate']:.1f}%")

    # LaTeX output
    print("\n" + "=" * 80)
    print("LATEX TABLE ROWS:")
    print("=" * 80)
    for ds_name, r in results.items():
        display_name = "Tox21 (NR-AR)" if ds_name == "tox21" else "hERG"
        print(f"{display_name} & Class. & {r['total']:,} & {r['train']:,} & {r['val']} & {r['test']:,} & {r['pos_rate']:.1f}\\% positive \\\\")

    return results


def verify_all_datasets():
    """Verify counts for ALL datasets (regression + classification)."""

    print("=" * 80)
    print("FULL DATASET VERIFICATION")
    print("=" * 80)

    datasets = [
        'Caco2_Wang',
        'Half_Life_Obach',
        'Clearance_Hepatocyte_AZ',
        'Clearance_Microsome_AZ',
        'tox21',
        'herg',
    ]

    results = []
    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        cache = prepare_dataset(ds_name, val_fraction=0.1, seed=42, verbose=False)

        n_train = len(cache['train'])
        n_val = len(cache['val'])
        n_test = len(cache['test'])
        n_total = n_train + n_val + n_test

        is_class = is_classification_dataset(ds_name)
        task = 'Classification' if is_class else 'Regression'

        results.append({
            'dataset': ds_name,
            'task': task,
            'total': n_total,
            'train': n_train,
            'val': n_val,
            'test': n_test,
        })

        print(f"  {task}: Total={n_total}, Train={n_train}, Val={n_val}, Test={n_test}")

    # CSV output
    print("\n" + "=" * 80)
    print("CSV FORMAT:")
    print("=" * 80)
    print("Dataset,Type,Task,Property,Size,Train,Val,Test")

    csv_info = [
        ('Caco2 Wang', 'ADME', 'Regression', 'Permeability'),
        ('Half-Life Obach', 'ADME', 'Regression', 'Half-life'),
        ('Clearance Hepatocyte', 'ADME', 'Regression', 'Clearance'),
        ('Clearance Microsome', 'ADME', 'Regression', 'Clearance'),
        ('Tox21', 'Toxicity', 'Classification', 'NR Toxicity'),
        ('hERG', 'Toxicity', 'Classification', 'Cardiac Toxicity'),
    ]

    for i, r in enumerate(results):
        info = csv_info[i]
        print(f"{info[0]},{info[1]},{info[2]},{info[3]},{r['total']},{r['train']},{r['val']},{r['test']}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Verify all datasets')
    args = parser.parse_args()

    if args.all:
        verify_all_datasets()
    else:
        verify_class_balance()
