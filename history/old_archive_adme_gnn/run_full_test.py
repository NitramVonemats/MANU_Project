"""
Simple full test for all 3 datasets
Avoids complex test suite issues
"""
import time
from datetime import datetime

# Test primer_v2.py directly
print("="*80)
print("RUNNING primer_v2.py ON ALL DATASETS")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    import primer_v2

    result = primer_v2.run_enhanced_molecular_benchmark(
        max_combos_per_dataset=10,  # 10 configs per dataset
        epochs=80,                   # 80 epochs
        patience=15,                 # Patience = 15
        device='cpu',
        seed=42,
        out_prefix='full_test_v2'
    )

    print("\n" + "="*80)
    print("PRIMER_V2 TEST COMPLETED SUCCESSFULLY!")
    print("="*80)

except Exception as e:
    print(f"\nERROR in primer_v2: {e}")
    import traceback
    traceback.print_exc()

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
