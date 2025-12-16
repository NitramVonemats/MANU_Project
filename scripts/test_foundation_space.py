"""
Simple test to verify foundation model HPO components work without running full HPO.

This tests:
1. Search space bounds and decoding
2. Module imports
3. Basic component initialization
"""
import sys
import numpy as np

print("="*70)
print("FOUNDATION MODEL HPO - COMPONENT VERIFICATION")
print("="*70)

# Test 1: Search space module
print("\n[1/3] Testing search space module...")
try:
    from optimization.foundation_space import bounds, decode_vector, get_dimension
    
    for model_type in ["morgan", "chemberta", "biomed", "molclr", "mole"]:
        lb, ub = bounds(model_type)
        dim = get_dimension(model_type)
        x = np.random.uniform(lb, ub)
        params = decode_vector(x, model_type)
        print(f"  ✓ {model_type:12s}: dim={dim}, sample proj_dim={params['proj_dim']}")
    
    print("  ✓ Search space module works correctly")
except Exception as e:
    print(f"  ✗ Search space module failed: {e}")
    sys.exit(1)

# Test 2: Foundation model imports
print("\n[2/3] Testing foundation model imports...")
try:
    from adme_gnn.models.foundation import (
        MorganFingerprintEncoder,
        ChemBERTaEncoder,
        BioMedEncoder,
        MolCLREncoder,
        MolEEncoder,
    )
    print("  ✓ All foundation model classes imported successfully")
except Exception as e:
    print(f"  ✗ Foundation model import failed: {e}")
    sys.exit(1)

# Test 3: Create encoders
print("\n[3/3] Testing encoder initialization...")
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    # Test Morgan-FP (should always work)
    encoder = MorganFingerprintEncoder(n_bits=2048, radius=2, proj_dim=256)
    encoder = encoder.to(device)
    enabled = getattr(encoder, 'enabled', True)
    print(f"  ✓ Morgan-FP encoder created (enabled: {enabled})")
    
    # Test ChemBERTa (may not be enabled if transformers not installed)
    encoder = ChemBERTaEncoder(proj_dim=256)
    enabled = getattr(encoder, 'enabled', True)
    print(f"  ✓ ChemBERTa encoder created (enabled: {enabled})")
    
    print("  ✓ Encoder initialization works")
except Exception as e:
    print(f"  ✗ Encoder initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL COMPONENT TESTS PASSED!")
print("="*70)
print("\nNext Steps:")
print("1. Install TDC if not already: pip install PyTDC")
print("2. Run quick HPO test:")
print("   python scripts/run_foundation_hpo.py --model morgan --dataset Caco2_Wang --algo random --trials 5")
print("3. Run full batch:")
print("   python scripts/run_foundation_batch.py --config config_foundation_benchmark.yaml --dry-run")
print("="*70)
