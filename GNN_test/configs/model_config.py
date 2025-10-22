"""
Model configuration classes
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class GNNConfig:
    """Configuration for GNN backbone"""
    model_type: str = "SAGE"  # SAGE or GINE
    layers: int = 3
    hidden: int = 64
    dropout: float = 0.5
    input_dim: int = 27
    edge_dim: int = 12

@dataclass
class FoundationModelConfig:
    """Configuration for foundation models"""
    model_name: str = "ChemBERTa"
    proj_dim: int = 256
    model_path: Optional[str] = None

@dataclass
class ModelConfig:
    """Main model configuration"""
    gnn: Optional[GNNConfig] = None
    foundation: Optional[FoundationModelConfig] = None
    adme_dim: int = 20

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-3
    split_type: str = "scaffold"
