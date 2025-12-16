"""
NiaPy problem wrapper for foundation model hyperparameter optimization.

This module defines the optimization problem for foundation models:
  1. Load pre-trained encoder (or feature extractor)
  2. Extract embeddings for train/val/test sets
  3. Train sklearn predictor with hyperparameters from HPO
  4. Return validation loss for optimization
"""
import numpy as np
import torch
from typing import Optional, Literal
from niapy.problems import Problem
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
from tdc.single_pred import ADME, Tox

from adme_gnn.models.foundation import (
    MorganFingerprintEncoder,
    ChemBERTaEncoder,
    BioMedEncoder,
    MolCLREncoder,
    MolEEncoder,
)
from .foundation_space import bounds, decode_vector

ModelType = Literal["morgan", "chemberta", "biomed", "molclr", "mole"]


def is_classification_dataset(dataset_name: str) -> bool:
    """Check if dataset is classification (Tox) or regression (ADME)."""
    return dataset_name.lower() in ['tox21', 'herg', 'clintox']


def get_dataset(name: str, split_method: str = 'scaffold'):
    """Load dataset from TDC."""
    if is_classification_dataset(name):
        data = Tox(name=name)
    else:
        data = ADME(name=name)
    return data.get_split(method=split_method)


class FoundationModelProblem(Problem):
    """NiaPy problem wrapper for foundation model HPO."""
    
    def __init__(
        self,
        model_type: ModelType,
        dataset_name: str,
        max_iter: int = 500,
        seed: int = 42,
        device: str = "auto",
        verbose: bool = False,
    ):
        """
        Initialize foundation model HPO problem.
        
        Args:
            model_type: Type of foundation model ("morgan", "chemberta", etc.)
            dataset_name: Name of ADME/Tox dataset
            max_iter: Maximum iterations for sklearn predictor
            seed: Random seed
            device: Device for encoder ("auto", "cuda", "cpu")
            verbose: Whether to print progress
        """
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.max_iter = max_iter
        self.seed = seed
        self.verbose = verbose
        self.is_classification = is_classification_dataset(dataset_name)
        
        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load dataset
        if self.verbose:
            print(f"Loading dataset: {dataset_name}...")
        split = get_dataset(dataset_name)
        self.train_df = split['train']
        self.valid_df = split['valid']
        self.test_df = split['test']
        
        # Extract SMILES and labels
        self.train_smiles = self.train_df['Drug'].tolist()
        self.valid_smiles = self.valid_df['Drug'].tolist()
        self.test_smiles = self.test_df['Drug'].tolist()
        
        self.y_train = self.train_df['Y'].values
        self.y_valid = self.valid_df['Y'].values
        self.y_test = self.test_df['Y'].values
        
        # Scaling (only for regression)
        self.scaler = None
        if not self.is_classification:
            self.scaler = StandardScaler()
            self.y_train_scaled = self.scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
            self.y_valid_scaled = self.scaler.transform(self.y_valid.reshape(-1, 1)).flatten()
        else:
            self.y_train_scaled = self.y_train
            self.y_valid_scaled = self.y_valid
        
        # Initialize NiaPy Problem
        lower, upper = bounds(model_type)
        super().__init__(dimension=len(lower), lower=lower, upper=upper)
        
        if self.verbose:
            print(f"Dataset loaded: {len(self.train_smiles)} train, {len(self.valid_smiles)} val")
            print(f"Model type: {model_type}")
            print(f"Task type: {'classification' if self.is_classification else 'regression'}")
            print(f"Device: {self.device}")
    
    def _get_encoder(self, params: dict):
        """Create encoder instance with given parameters."""
        proj_dim = params['proj_dim']
        
        if self.model_type == "morgan":
            return MorganFingerprintEncoder(
                n_bits=params['n_bits'],
                radius=params['radius'],
                proj_dim=proj_dim
            )
        elif self.model_type == "chemberta":
            return ChemBERTaEncoder(
                model_name="seyonec/ChemBERTa-zinc-base-v1",
                proj_dim=proj_dim
            )
        elif self.model_type == "biomed":
            return BioMedEncoder(
                model_name="ibm-research/biomed.sm.mv-te-84m",
                proj_dim=proj_dim
            )
        elif self.model_type == "molclr":
            return MolCLREncoder(proj_dim=proj_dim)
        elif self.model_type == "mole":
            return MolEEncoder(proj_dim=proj_dim)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _extract_embeddings(self, encoder, smiles_list):
        """Extract embeddings using the encoder."""
        embeddings = []
        batch_size = 32
        
        encoder.eval()
        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i+batch_size]
                try:
                    emb = encoder(batch, self.device)
                    embeddings.append(emb.cpu().numpy())
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Embedding extraction failed for batch {i}: {e}")
                    # Return zeros as fallback
                    embeddings.append(np.zeros((len(batch), encoder.proj_dim)))
        
        return np.vstack(embeddings)
    
    def _evaluate(self, x):
        """
        Evaluate hyperparameters by training predictor and returning validation loss.
        
        Args:
            x: Continuous hyperparameter vector
            
        Returns:
            Validation loss (RMSE for regression, 1-AUC for classification)
        """
        try:
            # Decode hyperparameters
            params = decode_vector(np.array(x, dtype=float), self.model_type)
            
            # Create encoder
            encoder = self._get_encoder(params).to(self.device)
            
            # Check if encoder is enabled
            if hasattr(encoder, 'enabled') and not encoder.enabled:
                if self.verbose:
                    print(f"Warning: Encoder {self.model_type} not enabled")
                return float('inf')
            
            # Extract embeddings
            X_train = self._extract_embeddings(encoder, self.train_smiles)
            X_valid = self._extract_embeddings(encoder, self.valid_smiles)
            
            # Build predictor
            hidden_dims = params['hidden_dims']
            alpha = params['weight_decay']
            
            if self.is_classification:
                predictor = MLPClassifier(
                    hidden_layer_sizes=hidden_dims,
                    max_iter=self.max_iter,
                    early_stopping=True,
                    random_state=self.seed,
                    alpha=alpha,
                    learning_rate_init=params['lr'],
                    verbose=False,
                )
            else:
                predictor = MLPRegressor(
                    hidden_layer_sizes=hidden_dims,
                    max_iter=self.max_iter,
                    early_stopping=True,
                    random_state=self.seed,
                    alpha=alpha,
                    learning_rate_init=params['lr'],
                    verbose=False,
                )
            
            # Train predictor
            predictor.fit(X_train, self.y_train_scaled)
            
            # Evaluate on validation set
            if self.is_classification:
                y_pred_proba = predictor.predict_proba(X_valid)[:, 1]
                auc = roc_auc_score(self.y_valid_scaled, y_pred_proba)
                # Return 1 - AUC so minimization works
                return float(1.0 - auc)
            else:
                y_pred = predictor.predict(X_valid)
                rmse = np.sqrt(mean_squared_error(self.y_valid_scaled, y_pred))
                return float(rmse)
            
        except Exception as exc:
            if self.verbose:
                print(f"[HPO] Evaluation failed: {exc}")
            return float('inf')


def build_foundation_problem(
    model_type: ModelType,
    dataset: str,
    max_iter: int = 500,
    seed: int = 42,
    device: str = "auto",
    verbose: bool = False,
) -> FoundationModelProblem:
    """
    Build a foundation model HPO problem.
    
    Args:
        model_type: Type of foundation model
        dataset: Dataset name
        max_iter: Max iterations for sklearn predictor
        seed: Random seed
        device: Device for encoder
        verbose: Print progress
        
    Returns:
        FoundationModelProblem instance
    """
    return FoundationModelProblem(
        model_type=model_type,
        dataset_name=dataset,
        max_iter=max_iter,
        seed=seed,
        device=device,
        verbose=verbose,
    )
