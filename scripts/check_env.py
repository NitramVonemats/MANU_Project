"""Quick diagnostic to check what's available in tdc_env."""
import sys

print("Python version:", sys.version)
print("\nChecking imports...")

# Check core dependencies
try:
    import numpy as np
    print("✓ numpy:", np.__version__)
except ImportError as e:
    print("✗ numpy:", e)

try:
    import torch
    print("✓ torch:", torch.__version__)
except ImportError as e:
    print("✗ torch:", e)

try:
    from sklearn.neural_network import MLPRegressor
    print("✓ sklearn available")
except ImportError as e:
    print("✗ sklearn:", e)

try:
    from tdc.single_pred import ADME
    print("✓ TDC available")
except ImportError as e:
    print("✗ TDC:", e)

try:
    import niapy
    print("✓ NiaPy:", niapy.__version__)
except ImportError as e:
    print("✗ NiaPy:", e)

try:
    from rdkit import Chem
    print("✓ RDKit available")
except ImportError as e:
    print("✗ RDKit:", e)

print("\nTrying to import our modules...")
try:
    from optimization.foundation_space import bounds
    print("✓ foundation_space imports successfully")
except Exception as e:
    print("✗ foundation_space error:", e)

try:
    from adme_gnn.models.foundation import MorganFingerprintEncoder
    print("✓ foundation models import successfully")
except Exception as e:
    print("✗ foundation models error:", e)
