"""
Quick test of Phase 2 foundation benchmark
Tests only 1 dataset, 1 seed, reduced epochs
"""

import sys
import os

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Import from phase2_foundation_benchmark
from phase2_foundation_benchmark import (
    ChemBERTaEncoder, MolFormerEncoder, RobertaLikeEncoder, MorganFingerprintEncoder,
    GNNBackbone, GNNOnlyRegressor, FoundationOnlyRegressor, HybridRegressor,
    build_loaders, train_model, PHASE1_BEST_CONFIGS, TRANSFORMERS_OK, RDKit_OK
)

print("="*80)
print("QUICK TEST: Phase 2 Foundation Benchmark")
print("="*80)
print(f"\nTransformers available: {TRANSFORMERS_OK}")
print(f"RDKit available: {RDKit_OK}")

# Test on single dataset with reduced settings
dataset_name = "Half_Life_Obach"
config = PHASE1_BEST_CONFIGS[dataset_name]
seed = 42
epochs = 10  # Reduced for testing
device = "cuda" if __import__('torch').cuda.is_available() else "cpu"

print(f"\nTesting on: {dataset_name}")
print(f"Device: {device}")
print(f"Epochs: {epochs} (reduced for testing)")

# Load data
print("\nLoading data...")
train_loader, val_loader, test_loader, adme_dim = build_loaders(
    dataset_name, split_type=config['split_type']
)
in_dim = int(train_loader.dataset[0].x.size(1))
edge_dim = int(train_loader.dataset[0].edge_attr.size(1)) if train_loader.dataset[0].edge_attr.numel() > 0 else 12

print(f"  Input dim: {in_dim}, Edge dim: {edge_dim}, ADME dim: {adme_dim}")

# Test 1: GNN-only
print("\n" + "="*60)
print("TEST 1: GNN-only")
print("="*60)
try:
    backbone = GNNBackbone(
        model_type=config['model_type'],
        layers=config['layers'],
        hidden=config['hidden'],
        dropout=config['dropout'],
        input_dim=in_dim,
        edge_dim=edge_dim
    )
    model_gnn = GNNOnlyRegressor(backbone, adme_dim=adme_dim)
    config_test = config.copy()
    result = train_model("GNN-only", dataset_name, model_gnn, config_test, seed=seed, epochs=epochs, device=device)
    print(f"[OK] GNN-only: R2={result['test_r2']:.3f}, RMSE={result['test_rmse']:.2f}")
except Exception as e:
    print(f"[FAIL] GNN-only failed: {e}")

# Test 2: ChemBERTa-only
if TRANSFORMERS_OK:
    print("\n" + "="*60)
    print("TEST 2: ChemBERTa-only")
    print("="*60)
    try:
        encoder = ChemBERTaEncoder(proj_dim=256)
        model = FoundationOnlyRegressor(encoder, adme_dim=adme_dim, text_dim=256)
        result = train_model("ChemBERTa-only", dataset_name, model, config, seed=seed, epochs=epochs, device=device)
        print(f"[OK] ChemBERTa-only: R2={result['test_r2']:.3f}, RMSE={result['test_rmse']:.2f}")
    except Exception as e:
        print(f"[FAIL] ChemBERTa-only failed: {e}")

# Test 3: MolFormer-only
if TRANSFORMERS_OK:
    print("\n" + "="*60)
    print("TEST 3: MolFormer-only")
    print("="*60)
    try:
        encoder = MolFormerEncoder(proj_dim=256)
        model = FoundationOnlyRegressor(encoder, adme_dim=adme_dim, text_dim=256)
        result = train_model("MolFormer-only", dataset_name, model, config, seed=seed, epochs=epochs, device=device)
        print(f"[OK] MolFormer-only: R2={result['test_r2']:.3f}, RMSE={result['test_rmse']:.2f}")
    except Exception as e:
        print(f"[FAIL] MolFormer-only failed: {e}")

# Test 4: Morgan-FP
print("\n" + "="*60)
print("TEST 4: Morgan-FP-only")
print("="*60)
try:
    encoder = MorganFingerprintEncoder(proj_dim=256)
    model = FoundationOnlyRegressor(encoder, adme_dim=adme_dim, text_dim=256)
    result = train_model("Morgan-FP-only", dataset_name, model, config, seed=seed, epochs=epochs, device=device)
    print(f"[OK] Morgan-FP-only: R2={result['test_r2']:.3f}, RMSE={result['test_rmse']:.2f}")
except Exception as e:
    print(f"[FAIL] Morgan-FP-only failed: {e}")

# Test 5: Hybrid (GNN + ChemBERTa)
if TRANSFORMERS_OK:
    print("\n" + "="*60)
    print("TEST 5: GNN + ChemBERTa Hybrid")
    print("="*60)
    try:
        backbone = GNNBackbone(
            model_type=config['model_type'],
            layers=config['layers'],
            hidden=config['hidden'],
            dropout=config['dropout'],
            input_dim=in_dim,
            edge_dim=edge_dim
        )
        encoder = ChemBERTaEncoder(proj_dim=256)
        model = HybridRegressor(backbone, encoder, adme_dim=adme_dim, text_dim=256)
        result = train_model("GNN+ChemBERTa", dataset_name, model, config, seed=seed, epochs=epochs, device=device)
        print(f"[OK] GNN+ChemBERTa: R2={result['test_r2']:.3f}, RMSE={result['test_rmse']:.2f}")
    except Exception as e:
        print(f"[FAIL] GNN+ChemBERTa failed: {e}")

print("\n" + "="*80)
print("QUICK TEST COMPLETE!")
print("="*80)
print("\nIf all tests passed, you can run the full benchmark:")
print("  python phase2_foundation_benchmark.py")
