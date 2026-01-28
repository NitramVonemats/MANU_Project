"""
Foundation model encoders for molecular representation learning
"""
import torch
import torch.nn as nn
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKit_OK = True
except ImportError:
    RDKit_OK = False


class ChemBERTaEncoder(nn.Module):
    """
    ChemBERTa pretrained transformer for SMILES
    
    Args:
        model_name: HuggingFace model name (default: seyonec/ChemBERTa-zinc-base-v1)
        proj_dim: Output projection dimension
    """

    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", proj_dim=256):
        super().__init__()
        self.enabled = TRANSFORMERS_OK
        self.proj_dim = proj_dim
        self.name = "ChemBERTa"

        if self.enabled:
            print(f"Loading ChemBERTa from {model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                h_dim = self.model.config.hidden_size
            except Exception as e:
                print(f"WARNING: Failed to load ChemBERTa: {e}")
                self.enabled = False
                h_dim = proj_dim
        else:
            h_dim = proj_dim

        self.proj = nn.Linear(h_dim, proj_dim)

    @torch.no_grad()
    def _embed_smiles(self, smiles_list, device):
        """Embed SMILES strings using ChemBERTa"""
        if not self.enabled or len(smiles_list) == 0:
            return torch.zeros((len(smiles_list), self.proj_dim), device=device)

        tokens = self.tokenizer(
            smiles_list, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(device)

        output = self.model(**tokens).last_hidden_state  # [B, T, H]
        emb = output.mean(dim=1)  # Mean pooling
        return emb

    def forward(self, smiles_list, device):
        """Forward pass with SMILES list"""
        emb = self._embed_smiles(smiles_list, device)
        return self.proj(emb)


class BioMedEncoder(nn.Module):
    """
    BioMed Multi-view Foundation Model (IBM)
    
    Args:
        model_name: HuggingFace model name
        proj_dim: Output projection dimension
    """

    def __init__(self, model_name="ibm-research/biomed.sm.mv-te-84m", proj_dim=256):
        super().__init__()
        self.enabled = TRANSFORMERS_OK
        self.proj_dim = proj_dim
        self.name = "BioMed-IBM"

        if self.enabled:
            print(f"Loading BioMed model from {model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                h_dim = self.model.config.hidden_size
            except Exception as e:
                print(f"WARNING: Failed to load BioMed model: {e}")
                print("Falling back to placeholder...")
                self.enabled = False
                h_dim = proj_dim
        else:
            h_dim = proj_dim

        self.proj = nn.Linear(h_dim, proj_dim)

    @torch.no_grad()
    def _embed_smiles(self, smiles_list, device):
        """Embed SMILES strings using BioMed model"""
        if not self.enabled or len(smiles_list) == 0:
            return torch.zeros((len(smiles_list), self.proj_dim), device=device)

        try:
            # Tokenize
            tokens = self.tokenizer(
                smiles_list, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(device)

            # Forward pass
            output = self.model(**tokens).last_hidden_state
            emb = output.mean(dim=1)  # Mean pooling
            return emb
        except Exception as e:
            print(f"WARNING: BioMed embedding failed: {e}")
            return torch.zeros((len(smiles_list), self.proj_dim), device=device)

    def forward(self, smiles_list, device):
        """Forward pass with SMILES list"""
        emb = self._embed_smiles(smiles_list, device)
        return self.proj(emb)


class MolCLREncoder(nn.Module):
    """
    MolCLR - Molecular Contrastive Learning

    Uses a GCN backbone for graph-based molecular embeddings.
    Converts SMILES to graphs on-the-fly using PyTorch Geometric.
    """

    def __init__(self, proj_dim=256, atom_feat_dim=9):
        super().__init__()
        self.proj_dim = proj_dim
        self.atom_feat_dim = atom_feat_dim
        self.name = "MolCLR"
        self.enabled = True

        try:
            from torch_geometric.nn import GCNConv, global_mean_pool
            from torch_geometric.data import Data, Batch
            self.GCNConv = GCNConv
            self.global_mean_pool = global_mean_pool
            self.Data = Data
            self.Batch = Batch

            # Try to import from_smiles
            try:
                from torch_geometric.utils import from_smiles
                self.from_smiles = from_smiles
            except ImportError:
                try:
                    from torch_geometric.data import from_smiles
                    self.from_smiles = from_smiles
                except ImportError:
                    print("WARNING: from_smiles not available, MolCLR will use fallback")
                    self.from_smiles = None

            # GCN layers - 3 layer architecture
            self.conv1 = GCNConv(atom_feat_dim, 128)
            self.conv2 = GCNConv(128, 256)
            self.conv3 = GCNConv(256, 256)
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(256)

        except ImportError as e:
            print(f"WARNING: PyTorch Geometric not available: {e}")
            self.enabled = False

        self.proj = nn.Linear(256, proj_dim)

    def _smiles_to_graph(self, smiles):
        """Convert single SMILES to PyG Data object"""
        if self.from_smiles is None:
            return None
        try:
            data = self.from_smiles(smiles)
            if data is None or data.x is None:
                return None
            # Ensure x has correct dimensions (pad or truncate to atom_feat_dim)
            if data.x.shape[1] < self.atom_feat_dim:
                padding = torch.zeros(data.x.shape[0], self.atom_feat_dim - data.x.shape[1])
                data.x = torch.cat([data.x.float(), padding], dim=1)
            elif data.x.shape[1] > self.atom_feat_dim:
                data.x = data.x[:, :self.atom_feat_dim].float()
            else:
                data.x = data.x.float()
            return data
        except Exception as e:
            return None

    def _smiles_list_to_batch(self, smiles_list, device):
        """Convert list of SMILES to batched PyG Data"""
        graphs = []
        for smi in smiles_list:
            g = self._smiles_to_graph(smi)
            if g is not None:
                graphs.append(g)
            else:
                # Create dummy graph for failed conversions
                dummy = self.Data(
                    x=torch.zeros(1, self.atom_feat_dim),
                    edge_index=torch.zeros(2, 0, dtype=torch.long)
                )
                graphs.append(dummy)

        if len(graphs) == 0:
            return None

        batch = self.Batch.from_data_list(graphs)
        return batch.to(device)

    def forward(self, data, device):
        """
        Forward pass - handles both SMILES list and PyG Data/Batch

        Args:
            data: Either list of SMILES strings or PyG Batch object
            device: torch device
        """
        if not self.enabled:
            if isinstance(data, list):
                return torch.zeros((len(data), self.proj_dim), device=device)
            return torch.zeros((1, self.proj_dim), device=device)

        # Convert SMILES list to batch if needed
        if isinstance(data, list):
            batch_data = self._smiles_list_to_batch(data, device)
            if batch_data is None:
                return torch.zeros((len(data), self.proj_dim), device=device)
        else:
            batch_data = data

        x = batch_data.x.to(device)
        edge_index = batch_data.edge_index.to(device)
        batch = batch_data.batch.to(device)

        # Forward through GCN layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x)

        # Global pooling
        x = self.global_mean_pool(x, batch)

        return self.proj(x)


class MolEEncoder(nn.Module):
    """
    MolE - Molecular Embeddings (Recursion-style)

    This is a placeholder implementation using Morgan fingerprints + learned projection.
    Real MolE would require specific pretrained weights from Recursion.

    For benchmarking purposes, this serves as a "learned fingerprint" baseline
    that learns a projection from Morgan FPs to an embedding space.
    """

    def __init__(self, proj_dim=256, n_bits=2048, hidden_dim=512):
        super().__init__()
        self.proj_dim = proj_dim
        self.n_bits = n_bits
        self.name = "MolE-FP"  # Indicate it's fingerprint-based
        self.enabled = RDKit_OK  # Enable if RDKit is available

        # Learned encoder from fingerprints
        self.encoder = nn.Sequential(
            nn.Linear(n_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, proj_dim)
        )

    def _get_fingerprint(self, smiles):
        """Get Morgan fingerprint for a SMILES string"""
        if not RDKit_OK:
            return np.zeros(self.n_bits, dtype=np.float32)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.n_bits, dtype=np.float32)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.n_bits)
            return np.array(fp, dtype=np.float32)
        except:
            return np.zeros(self.n_bits, dtype=np.float32)

    def forward(self, smiles_list, device):
        """Forward pass with SMILES list"""
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        if not self.enabled:
            return torch.zeros((len(smiles_list), self.proj_dim), device=device)

        fps = np.array([self._get_fingerprint(s) for s in smiles_list])
        fps_tensor = torch.tensor(fps, dtype=torch.float32, device=device)
        return self.encoder(fps_tensor)


class MorganFingerprintEncoder(nn.Module):
    """
    Morgan Fingerprint (ECFP) baseline
    """

    def __init__(self, n_bits=2048, radius=2, proj_dim=256):
        super().__init__()
        self.n_bits = n_bits
        self.radius = radius
        self.proj_dim = proj_dim
        self.enabled = RDKit_OK
        self.name = "Morgan-FP"

        self.encoder = nn.Sequential(
            nn.Linear(n_bits, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, proj_dim)
        )

    def _get_fingerprint(self, smiles):
        if not self.enabled:
            return np.zeros(self.n_bits, dtype=np.float32)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.n_bits, dtype=np.float32)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            return np.array(fp, dtype=np.float32)
        except:
            return np.zeros(self.n_bits, dtype=np.float32)

    def forward(self, smiles_list, device):
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        fps = np.array([self._get_fingerprint(s) for s in smiles_list])
        fps_tensor = torch.tensor(fps, dtype=torch.float32, device=device)
        return self.encoder(fps_tensor)
