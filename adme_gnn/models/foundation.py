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
    MolCLR - Molecular Contrastive Learning with PRETRAINED WEIGHTS

    Uses official MolCLR pretrained GIN model from:
    https://github.com/yuyangw/MolCLR

    The model is pretrained on ~10M molecules using contrastive learning
    and achieves state-of-the-art results on molecular property prediction.
    """

    def __init__(self, proj_dim=256, pretrained_path=None, model_type='gin'):
        super().__init__()
        self.proj_dim = proj_dim
        self.name = "MolCLR"
        self.enabled = True
        self.model_type = model_type

        # Default pretrained path
        if pretrained_path is None:
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            pretrained_path = os.path.join(base_dir, 'external', 'MolCLR', 'ckpt',
                                          f'pretrained_{model_type}', 'checkpoints', 'model.pth')
        self.pretrained_path = pretrained_path

        # MolCLR feature constants
        self.num_atom_type = 119
        self.num_chirality_tag = 3
        self.num_bond_type = 5
        self.num_bond_direction = 3
        self.emb_dim = 300
        self.num_layer = 5
        self.feat_dim = 512

        try:
            from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
            from torch_geometric.utils import add_self_loops
            from torch_geometric.data import Data, Batch
            from rdkit import Chem
            from rdkit.Chem.rdchem import BondType as BT

            self.global_mean_pool = global_mean_pool
            self.global_add_pool = global_add_pool
            self.add_self_loops = add_self_loops
            self.Data = Data
            self.Batch = Batch
            self.Chem = Chem
            self.BT = BT

            # Build the GIN encoder (matching MolCLR architecture)
            self._build_gin_encoder()

            # Load pretrained weights
            self._load_pretrained_weights()

        except ImportError as e:
            print(f"WARNING: Required packages not available: {e}")
            self.enabled = False
        except Exception as e:
            print(f"WARNING: MolCLR initialization failed: {e}")
            self.enabled = False

        # Output projection
        self.proj = nn.Linear(self.feat_dim, proj_dim)

    def _build_gin_encoder(self):
        """Build GIN encoder matching MolCLR architecture"""
        # Atom embeddings
        self.x_embedding1 = nn.Embedding(self.num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(self.num_chirality_tag, self.emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # GIN convolution layers
        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(self.num_layer):
            self.gnns.append(GINEConvLayer(self.emb_dim, self.num_bond_type, self.num_bond_direction))
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

        # Feature projection
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

    def _load_pretrained_weights(self):
        """Load official MolCLR pretrained weights"""
        import os
        if os.path.exists(self.pretrained_path):
            print(f"Loading MolCLR pretrained weights from {self.pretrained_path}")
            state_dict = torch.load(self.pretrained_path, map_location='cpu')

            # Filter and load only encoder weights (not prediction head)
            own_state = self.state_dict()
            loaded_count = 0
            for name, param in state_dict.items():
                # Skip prediction head weights
                if 'pred_head' in name or 'pred_lin' in name:
                    continue
                if name in own_state:
                    if isinstance(param, nn.parameter.Parameter):
                        param = param.data
                    try:
                        own_state[name].copy_(param)
                        loaded_count += 1
                    except Exception as e:
                        print(f"  Could not load {name}: {e}")
            print(f"  Loaded {loaded_count} pretrained parameters")
        else:
            print(f"WARNING: Pretrained weights not found at {self.pretrained_path}")
            print("  MolCLR will use random initialization (not recommended)")

    def _smiles_to_graph(self, smiles):
        """Convert SMILES to PyG Data using MolCLR featurization"""
        try:
            mol = self.Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = self.Chem.AddHs(mol)

            # Atom features
            ATOM_LIST = list(range(1, 119))
            CHIRALITY_LIST = [
                self.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                self.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                self.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                self.Chem.rdchem.ChiralType.CHI_OTHER
            ]
            BOND_LIST = [self.BT.SINGLE, self.BT.DOUBLE, self.BT.TRIPLE, self.BT.AROMATIC]
            BONDDIR_LIST = [
                self.Chem.rdchem.BondDir.NONE,
                self.Chem.rdchem.BondDir.ENDUPRIGHT,
                self.Chem.rdchem.BondDir.ENDDOWNRIGHT
            ]

            type_idx = []
            chirality_idx = []
            for atom in mol.GetAtoms():
                atom_num = atom.GetAtomicNum()
                type_idx.append(ATOM_LIST.index(atom_num) if atom_num in ATOM_LIST else 0)
                chiral = atom.GetChiralTag()
                chirality_idx.append(CHIRALITY_LIST.index(chiral) if chiral in CHIRALITY_LIST else 0)

            x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
            x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
            x = torch.cat([x1, x2], dim=-1)

            # Edge features
            row, col, edge_feat = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                bond_type = bond.GetBondType()
                bond_dir = bond.GetBondDir()
                edge_feat.append([
                    BOND_LIST.index(bond_type) if bond_type in BOND_LIST else 0,
                    BONDDIR_LIST.index(bond_dir) if bond_dir in BONDDIR_LIST else 0
                ])
                edge_feat.append([
                    BOND_LIST.index(bond_type) if bond_type in BOND_LIST else 0,
                    BONDDIR_LIST.index(bond_dir) if bond_dir in BONDDIR_LIST else 0
                ])

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = torch.tensor(edge_feat, dtype=torch.long) if edge_feat else torch.zeros((0, 2), dtype=torch.long)

            return self.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
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
                # Dummy graph for failed conversions
                dummy = self.Data(
                    x=torch.zeros(1, 2, dtype=torch.long),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    edge_attr=torch.zeros(0, 2, dtype=torch.long)
                )
                graphs.append(dummy)

        if len(graphs) == 0:
            return None

        batch = self.Batch.from_data_list(graphs)
        return batch.to(device)

    def forward(self, data, device):
        """Forward pass with SMILES list or PyG Batch"""
        if not self.enabled:
            if isinstance(data, list):
                return torch.zeros((len(data), self.proj_dim), device=device)
            return torch.zeros((1, self.proj_dim), device=device)

        # Convert SMILES to batch
        if isinstance(data, list):
            batch_data = self._smiles_list_to_batch(data, device)
            if batch_data is None:
                return torch.zeros((len(data), self.proj_dim), device=device)
        else:
            batch_data = data

        x = batch_data.x.to(device)
        edge_index = batch_data.edge_index.to(device)
        edge_attr = batch_data.edge_attr.to(device)
        batch = batch_data.batch.to(device)

        # Atom embeddings
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        # GIN layers
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = torch.dropout(h, p=0.0, train=self.training)
            else:
                h = torch.dropout(torch.relu(h), p=0.0, train=self.training)

        # Global pooling
        h = self.global_mean_pool(h, batch)
        h = self.feat_lin(h)

        return self.proj(h)


class GINEConvLayer(nn.Module):
    """GIN convolution layer with edge features (matching MolCLR)"""

    def __init__(self, emb_dim, num_bond_type=5, num_bond_direction=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.emb_dim = emb_dim

    def forward(self, x, edge_index, edge_attr):
        from torch_geometric.utils import add_self_loops

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Self-loop edge attributes
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4  # Self-loop bond type
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        # Edge embeddings
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # Message passing
        row, col = edge_index
        out = torch.zeros_like(x)

        # Aggregate: sum of (neighbor features + edge features)
        messages = x[col] + edge_embeddings
        out.index_add_(0, row, messages)

        return self.mlp(out)


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
