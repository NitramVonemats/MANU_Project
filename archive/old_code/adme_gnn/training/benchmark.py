"""
Benchmark runner for comparing foundation models
"""
import torch
from configs.base_config import BaseConfig
from adme_gnn.data.loader import build_loaders
from models.gnn import GNNBackbone
from models.foundation import ChemBERTaEncoder, MolFormerEncoder, RobertaLikeEncoder, MorganFingerprintEncoder
from models.predictors import GNNOnlyRegressor, FoundationOnlyRegressor, HybridRegressor
from .trainer import train_model

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False


def benchmark_dataset(dataset_name, seeds=None, device=None):
    """
    Benchmark all foundation models on a single dataset

    Args:
        dataset_name: TDC ADME dataset name
        seeds: List of random seeds (default: [42, 123, 456])
        device: Device (cpu or cuda)

    Returns:
        List of result dictionaries
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = seeds or [42, 123, 456]

    config = BaseConfig.get_dataset_config(dataset_name)

    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {dataset_name}")
    print(f"{'='*80}")

    results = []

    # Get dimensions (need to load data once)
    train_loader, _, _, adme_dim = build_loaders(dataset_name, split_type=config['split_type'])
    in_dim = int(train_loader.dataset[0].x.size(1))
    edge_dim = int(train_loader.dataset[0].edge_attr.size(1)) if train_loader.dataset[0].edge_attr.numel() > 0 else 12

    # Define foundation models to test
    foundation_models = []

    # 1. ChemBERTa
    if TRANSFORMERS_OK:
        foundation_models.append(("ChemBERTa", ChemBERTaEncoder(proj_dim=256)))

    # 2. Morgan Fingerprints (baseline)
    foundation_models.append(("Morgan-FP", MorganFingerprintEncoder(proj_dim=256)))

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        # 1. GNN-only (baseline from Phase 1)
        print(f"\n[1] Testing GNN-only...")
        backbone = GNNBackbone(
            model_type=config['model_type'],
            layers=config['layers'],
            hidden=config['hidden'],
            dropout=config['dropout'],
            input_dim=in_dim,
            edge_dim=edge_dim
        )
        model_gnn = GNNOnlyRegressor(backbone, adme_dim=adme_dim)
        result_gnn = train_model("GNN-only", dataset_name, model_gnn, config, seed=seed, device=device)
        results.append(result_gnn)

        # 2. Foundation models only (without GNN)
        for idx, (fm_name, fm_encoder) in enumerate(foundation_models, start=2):
            print(f"\n[{idx}] Testing {fm_name}-only...")
            try:
                model_fm = FoundationOnlyRegressor(fm_encoder, adme_dim=adme_dim, text_dim=256)
                result_fm = train_model(f"{fm_name}-only", dataset_name, model_fm, config, seed=seed, device=device)
                results.append(result_fm)
            except Exception as e:
                print(f"ERROR: Failed to train {fm_name}-only: {e}")

        # 3. Hybrid models (GNN + Foundation)
        hybrid_idx = len(foundation_models) + 2
        for idx, (fm_name, fm_encoder) in enumerate(foundation_models, start=hybrid_idx):
            print(f"\n[{idx}] Testing GNN+{fm_name} Hybrid...")
            try:
                backbone_hybrid = GNNBackbone(
                    model_type=config['model_type'],
                    layers=config['layers'],
                    hidden=config['hidden'],
                    dropout=config['dropout'],
                    input_dim=in_dim,
                    edge_dim=edge_dim
                )
                model_hybrid = HybridRegressor(backbone_hybrid, fm_encoder, adme_dim=adme_dim, text_dim=256)
                result_hybrid = train_model(f"GNN+{fm_name}", dataset_name, model_hybrid, config, seed=seed, device=device)
                results.append(result_hybrid)
            except Exception as e:
                print(f"ERROR: Failed to train GNN+{fm_name} hybrid: {e}")

    return results
