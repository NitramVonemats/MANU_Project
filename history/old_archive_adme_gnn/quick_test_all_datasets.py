"""
Quick test to verify all 3 datasets work
Tests 3 configs per dataset with shortened training
"""
import time
from datetime import datetime
import primer

print("="*80)
print("QUICK TEST: All 3 Datasets")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    result = primer.run_enhanced_molecular_benchmark(
        max_combos_per_dataset=3,   # Only 3 configs per dataset
        epochs=20,                   # Only 20 epochs for speed
        patience=10,
        device='cpu',
        seed=42,
        out_prefix='quick_test_all'
    )

    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nTotal configs tested: {len(result)}")
    print(f"Successful configs: {result['success'].sum()}")
    print(f"Failed configs: {(~result['success']).sum()}")

    # Show summary by dataset
    for dataset in ["Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"]:
        ds_data = result[result['dataset'] == dataset]
        success_count = ds_data['success'].sum()
        print(f"\n{dataset}: {success_count}/{len(ds_data)} successful")

        if success_count > 0:
            successful = ds_data[ds_data['success'] == True]
            best = successful.nsmallest(1, 'val_rmse').iloc[0]
            print(f"  Best model: {best['model_name']}")
            print(f"  Test RMSE: {best['test_rmse']:.3f}")
            print(f"  Test R2: {best['test_r2']:.3f}")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
