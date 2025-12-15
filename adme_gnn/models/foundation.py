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
    
    Note: This is a placeholder/adaptation since MolCLR usually requires
    specific GNN architecture and pretrained weights.
    We will use a GCN backbone initialized with random weights if pretrained not found,
    or try to load if available.
    """

    def __init__(self, proj_dim=256):
        super().__init__()
        self.proj_dim = proj_dim
        self.name = "MolCLR"
        self.enabled = True # Assuming we can at least run the architecture
        
        # Placeholder GNN for MolCLR style embedding
        # In a real scenario, we would load 'gin_supervised_contextpred.pth' etc.
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        self.conv1 = GCNConv(37, 128) # 37 is standard atom feature size
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 256)
        
        self.proj = nn.Linear(256, proj_dim)

    def forward(self, data, device):
        """
        Expects PyG Data object or list of SMILES (which need conversion)
        For simplicity in this unified interface, we might need to handle both.
        But for now, let's assume we pass PyG batch if possible, or handle conversion.
        """
        # Note: This is a simplified mock-up. Real MolCLR needs graph conversion.
        # If passed SMILES list, we return zeros or need on-the-fly conversion.
        if isinstance(data, list):
             # Placeholder for SMILES list input - would need graph conversion
             return torch.zeros((len(data), self.proj_dim), device=device)
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        
        x = global_mean_pool(x, batch)
        return self.proj(x)


class MolEEncoder(nn.Module):
    """
    MolE - Molecular Embeddings (Recursion)
    
    Note: This often requires specific model files. 
    We will implement a placeholder that can be swapped with real model loading.
    """

    def __init__(self, proj_dim=256):
        super().__init__()
        self.proj_dim = proj_dim
        self.name = "MolE"
        self.enabled = False # Default to false unless we have the weights
        
        # Placeholder
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.proj = nn.Linear(1024, proj_dim) # MolE usually 1024 dim

    def forward(self, smiles_list, device):
        if not self.enabled:
            return torch.zeros((len(smiles_list), self.proj_dim), device=device)
            
        # Implementation would go here if we had the library/weights
        return torch.zeros((len(smiles_list), self.proj_dim), device=device)


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
