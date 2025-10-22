"""
Base configuration for Phase 2 Foundation Model Benchmarking
"""

# Global scalers for target transformation
TARGET_SCALER = {"mode": "log", "mu": 0.0, "sigma": 1.0}
ADME_SCALER = {"mu": None, "sigma": None}

# Best hyperparameters from Phase 1 for each dataset
PHASE1_BEST_CONFIGS = {
    "Half_Life_Obach": {
        "model_type": "SAGE",  # SAGEConv worked best
        "layers": 3,
        "hidden": 64,
        "dropout": 0.5,
        "weight_decay": 1e-3,
        "lr": 1e-3,
        "split_type": "scaffold",
    },
    "Clearance_Hepatocyte_AZ": {
        "model_type": "GINE",  # GINEConv for bond awareness
        "layers": 3,
        "hidden": 64,
        "dropout": 0.55,
        "weight_decay": 3e-3,
        "lr": 5e-4,
        "split_type": "random",
    },
    "Clearance_Microsome_AZ": {
        "model_type": "GINE",
        "layers": 3,
        "hidden": 64,
        "dropout": 0.55,
        "weight_decay": 3e-3,
        "lr": 5e-4,
        "split_type": "random",
    }
}

class BaseConfig:
    """Base configuration class"""

    # Training
    EPOCHS = 150
    PATIENCE = 30
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_EVAL = 64

    # Foundation models
    FOUNDATION_MODEL_DIM = 256

    # ADME descriptors
    ADME_DESC_DIM = 20

    # Seeds for reproducibility
    DEFAULT_SEEDS = [42, 123, 456, 789, 1011]

    @classmethod
    def get_dataset_config(cls, dataset_name: str):
        """Get best config for a dataset"""
        return PHASE1_BEST_CONFIGS.get(dataset_name, {})
