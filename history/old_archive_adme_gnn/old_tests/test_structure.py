"""
Test script to verify the new modular structure works correctly
"""
import sys
import traceback

def test_imports():
    """Test all module imports"""
    print("="*80)
    print("TESTING MODULE IMPORTS")
    print("="*80)

    tests = []

    # Test configs
    try:
        from configs.base_config import BaseConfig, PHASE1_BEST_CONFIGS
        from configs.model_config import GNNConfig, ModelConfig
        tests.append(("configs", True, None))
        print("[OK] configs module imported successfully")
    except Exception as e:
        tests.append(("configs", False, str(e)))
        print(f"[FAIL] configs module failed: {e}")

    # Test functional
    try:
        from functional.metrics import compute_metrics_np, spearman_metric
        from functional.transforms import set_target_scaler, transform_y, inverse_y
        from functional.utils import set_global_seed
        tests.append(("functional", True, None))
        print("[OK] functional module imported successfully")
    except Exception as e:
        tests.append(("functional", False, str(e)))
        print(f"[FAIL] functional module failed: {e}")

    # Test graph
    try:
        from graph.featurizer import enhanced_atom_features, enhanced_bond_features
        from graph.loader import build_loaders
        tests.append(("graph", True, None))
        print("[OK] graph module imported successfully")
    except Exception as e:
        tests.append(("graph", False, str(e)))
        print(f"[FAIL] graph module failed: {e}")

    # Test models
    try:
        from models.gnn import GNNBackbone
        from models.foundation import ChemBERTaEncoder
        from models.predictors import GNNOnlyRegressor, HybridRegressor
        tests.append(("models", True, None))
        print("[OK] models module imported successfully")
    except Exception as e:
        tests.append(("models", False, str(e)))
        print(f"[FAIL] models module failed: {e}")

    # Test services
    try:
        from services.trainer import train_model, evaluate
        from services.benchmark import benchmark_dataset
        tests.append(("services", True, None))
        print("[OK] services module imported successfully")
    except Exception as e:
        tests.append(("services", False, str(e)))
        print(f"[FAIL] services module failed: {e}")

    return tests


def test_basic_functionality():
    """Test basic functionality without running full training"""
    print("\n" + "="*80)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*80)

    try:
        import torch
        import numpy as np
        from configs.base_config import BaseConfig
        from models.gnn import GNNBackbone
        from functional.metrics import compute_metrics_np

        # Test config loading
        config = BaseConfig.get_dataset_config("Half_Life_Obach")
        print(f"[OK] Config loaded: {config['model_type']}, layers={config['layers']}")

        # Test GNN backbone creation
        backbone = GNNBackbone(
            model_type="SAGE",
            layers=3,
            hidden=64,
            input_dim=27,
            edge_dim=12
        )
        print(f"[OK] GNN backbone created: {backbone.model_type}, output_dim={backbone.output_dim}")

        # Test metrics
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        metrics = compute_metrics_np(y_true, y_pred)
        print(f"[OK] Metrics computed: RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}")

        return True
    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("\n" + "#"*80)
    print("# TESTING REORGANIZED STRUCTURE")
    print("#"*80 + "\n")

    # Test imports
    import_tests = test_imports()

    # Test basic functionality
    func_test = test_basic_functionality()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, status, _ in import_tests if status)
    total = len(import_tests)

    print(f"\nModule imports: {passed}/{total} passed")
    for module, status, error in import_tests:
        status_str = "[PASS]" if status else "[FAIL]"
        print(f"  {status_str:10} {module:15} {error if error else ''}")

    print(f"\nBasic functionality: {'[PASS]' if func_test else '[FAIL]'}")

    if passed == total and func_test:
        print("\n" + "="*80)
        print("ALL TESTS PASSED! The reorganized structure is working correctly.")
        print("="*80)
        print("\nYou can now run:")
        print("  python run_benchmark.py --dataset Half_Life_Obach --seeds 42")
        return 0
    else:
        print("\n" + "="*80)
        print("SOME TESTS FAILED. Please check the errors above.")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
