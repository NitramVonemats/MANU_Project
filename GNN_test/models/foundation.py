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
        model_name: HuggingFace model name
        proj_dim: Output projection dimension
    """

    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MLM", proj_dim=256):
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


class MolFormerEncoder(nn.Module):
    """
    MolFormer (IBM) - Large-scale pretrained transformer

    Args:
        model_name: HuggingFace model name
        proj_dim: Output projection dimension
    """

    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct", proj_dim=256):
        super().__init__()
        self.enabled = TRANSFORMERS_OK
        self.proj_dim = proj_dim
        self.name = "MolFormer"

        if self.enabled:
            print(f"Loading MolFormer from {model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, deterministic_eval=True)
                h_dim = self.model.config.hidden_size
            except Exception as e:
                print(f"WARNING: Failed to load MolFormer: {e}")
                print("Falling back to ChemBERTa architecture...")
                self.enabled = False
                h_dim = proj_dim
        else:
            h_dim = proj_dim

        self.proj = nn.Linear(h_dim, proj_dim)

    @torch.no_grad()
    def _embed_smiles(self, smiles_list, device):
        """Embed SMILES strings using MolFormer"""
        if not self.enabled or len(smiles_list) == 0:
            return torch.zeros((len(smiles_list), self.proj_dim), device=device)

        try:
            tokens = self.tokenizer(
                smiles_list, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(device)

            output = self.model(**tokens).last_hidden_state  # [B, T, H]
            emb = output.mean(dim=1)  # Mean pooling
            return emb
        except Exception as e:
            print(f"WARNING: MolFormer embedding failed: {e}")
            return torch.zeros((len(smiles_list), self.proj_dim), device=device)

    def forward(self, smiles_list, device):
        """Forward pass with SMILES list"""
        emb = self._embed_smiles(smiles_list, device)
        return self.proj(emb)


class RobertaLikeEncoder(nn.Module):
    """
    Generic RoBERTa-based encoder for molecular SMILES

    Args:
        model_name: HuggingFace model name
        proj_dim: Output projection dimension
    """

    def __init__(self, model_name="seyonec/PubChem10M_SMILES_BPE_450k", proj_dim=256):
        super().__init__()
        self.enabled = TRANSFORMERS_OK
        self.proj_dim = proj_dim
        self.name = "RoBERTa-SMILES"

        if self.enabled:
            print(f"Loading RoBERTa-based model from {model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                h_dim = self.model.config.hidden_size
            except Exception as e:
                print(f"WARNING: Failed to load RoBERTa model: {e}")
                self.enabled = False
                h_dim = proj_dim
        else:
            h_dim = proj_dim

        self.proj = nn.Linear(h_dim, proj_dim)

    @torch.no_grad()
    def _embed_smiles(self, smiles_list, device):
        """Embed SMILES strings using RoBERTa"""
        if not self.enabled or len(smiles_list) == 0:
            return torch.zeros((len(smiles_list), self.proj_dim), device=device)

        tokens = self.tokenizer(
            smiles_list, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        ).to(device)

        output = self.model(**tokens).last_hidden_state  # [B, T, H]
        emb = output.mean(dim=1)  # Mean pooling
        return emb

    def forward(self, smiles_list, device):
        """Forward pass with SMILES list"""
        emb = self._embed_smiles(smiles_list, device)
        return self.proj(emb)


class MorganFingerprintEncoder(nn.Module):
    """
    Morgan Fingerprint (ECFP) baseline - not a foundation model but good baseline

    Args:
        n_bits: Fingerprint size
        radius: Morgan fingerprint radius
        proj_dim: Output projection dimension
    """

    def __init__(self, n_bits=2048, radius=2, proj_dim=256):
        super().__init__()
        self.n_bits = n_bits
        self.radius = radius
        self.proj_dim = proj_dim
        self.enabled = RDKit_OK
        self.name = "Morgan-FP"

        # Simple MLP encoder for fingerprints
        self.encoder = nn.Sequential(
            nn.Linear(n_bits, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, proj_dim)
        )

    def _get_fingerprint(self, smiles):
        """Generate Morgan fingerprint from SMILES"""
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
        """Forward pass with SMILES list"""
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        fps = np.array([self._get_fingerprint(s) for s in smiles_list])
        fps_tensor = torch.tensor(fps, dtype=torch.float32, device=device)
        return self.encoder(fps_tensor)
