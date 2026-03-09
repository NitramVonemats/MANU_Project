"""
Debug script to catch full traceback of the "Invalid argument" error
"""
import traceback
from datetime import datetime

# Import primer
import primer

# Test single config on Clearance_Hepatocyte_AZ
try:
    print("Testing single config on Clearance_Hepatocyte_AZ...")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = {
        "model_name": "SAGE",
        "graph_layers": 3,
        "graph_hidden_channels": 128,
        "graph_norm": "BatchNorm",
        "activation": "relu",
        "heads": 1,
        "aggr": "mean",
        "residual": True,
        "use_edge_features": False,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.2
    }

    meta, val_m, test_m, aux = primer.train_eval_single_enhanced(
        config,
        "Clearance_Hepatocyte_AZ",
        epochs=10,  # Just 10 epochs for quick test
        patience=5,
        batch_train=32,
        batch_eval=64,
        device='cpu',
        seed=42,
    )

    print(f"\nSUCCESS!")
    print(f"Val RMSE: {val_m['rmse']:.3f}")
    print(f"Test RMSE: {test_m['rmse']:.3f}")
    print(f"Test R2: {test_m['r2']:.3f}")

except Exception as e:
    print(f"\n{'='*80}")
    print("CAUGHT EXCEPTION!")
    print(f"{'='*80}")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print(f"\nFull traceback:")
    print(f"{'='*80}")
    traceback.print_exc()
    print(f"{'='*80}")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
