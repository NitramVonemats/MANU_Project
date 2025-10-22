"""
Data loading utilities for molecular graphs
Now includes an offline fallback that loads local `.tab` files from
`GNN_test/data` when TDC or network access is unavailable.
"""
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from .featurizer import enhanced_atom_features, enhanced_bond_features, adme_specific_descriptors
from functional.transforms import set_target_scaler, transform_y
from configs.base_config import ADME_SCALER
import os
import pandas as pd

try:
    from rdkit import Chem
    RDKit_OK = True
except ImportError:
    RDKit_OK = False


def _import_tdc_and_from_smiles():
    """Import TDC and PyG's from_smiles"""
    try:
        from tdc.single_pred import ADME
    except Exception as e:
        raise RuntimeError("TDC not installed") from e

    try:
        from torch_geometric.utils import from_smiles
    except:
        try:
            from torch_geometric.data import from_smiles
        except Exception as e:
            raise RuntimeError("PyG from_smiles unavailable") from e

    return ADME, from_smiles


def _dataset_local_path(dataset_name: str) -> str:
    """Map dataset name to a local .tab file path inside GNN_test/data."""
    name_map = {
        "Half_Life_Obach": "half_life_obach.tab",
        "Clearance_Microsome_AZ": "clearance_microsome_az.tab",
        "Clearance_Hepatocyte_AZ": "clearance_hepatocyte_az.tab",
    }
    fname = name_map.get(dataset_name)
    if not fname:
        raise RuntimeError(f"No local dataset mapping for '{dataset_name}'")
    here = os.path.dirname(os.path.abspath(__file__))
    # GNN_test/graph -> GNN_test/data
    data_dir = os.path.abspath(os.path.join(here, os.pardir, "data"))
    return os.path.join(data_dir, fname)


def _scaffold_key(smiles: str) -> str:
    """Compute Bemis-Murcko scaffold SMILES (empty if RDKit not available)."""
    if not RDKit_OK:
        return ""
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf is not None else ""
    except Exception:
        return ""


def _make_splits_local(df: pd.DataFrame, split_type: str):
    """Create train/valid/test splits locally.

    - If `split_type == 'scaffold'` and RDKit is available, perform a simple
      scaffold-based split by packing whole scaffolds into splits with approx
      80/10/10 ratios.
    - Otherwise perform a deterministic random split with the same ratios.
    """
    df = df.copy()

    if split_type == "scaffold" and RDKit_OK:
        # Compute scaffold per row
        df["_scaf"] = df["Drug"].map(_scaffold_key)
        # Group by scaffold and sort groups by size (desc)
        grp = df.groupby("_scaf", dropna=False)
        groups = sorted(((k, len(v)) for k, v in grp), key=lambda x: -x[1])

        n_total = len(df)
        n_train_target, n_val_target = int(0.8 * n_total), int(0.1 * n_total)

        train_idx, val_idx, test_idx = [], [], []
        c_train = c_val = c_test = 0

        for scaf, _ in groups:
            idx = grp.get_group(scaf).index.tolist()
            # Assign whole scaffold to the first split that hasn't reached target yet
            if c_train < n_train_target:
                train_idx.extend(idx)
                c_train += len(idx)
            elif c_val < n_val_target:
                val_idx.extend(idx)
                c_val += len(idx)
            else:
                test_idx.extend(idx)
                c_test += len(idx)

        train_df = df.loc[train_idx].drop(columns=["_scaf"])
        val_df = df.loc[val_idx].drop(columns=["_scaf"])
        test_df = df.loc[test_idx].drop(columns=["_scaf"])
        return {"train": train_df, "valid": val_df, "test": test_df}

    # Deterministic random split (fallback)
    rng = np.random.RandomState(42)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(idx)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return {
        "train": df.iloc[train_idx],
        "valid": df.iloc[val_idx],
        "test": df.iloc[test_idx],
    }


def _load_local_dataset(dataset_name: str, split_type: str):
    """Load local .tab dataset and build a split dict like TDC's output.

    Expected columns in .tab: ID, X, Y where X is SMILES. Renamed to
    Drug (SMILES) and Y for compatibility with the rest of the pipeline.
    """
    path = _dataset_local_path(dataset_name)
    if not os.path.exists(path):
        raise RuntimeError(f"Local dataset not found at: {path}")

    df = pd.read_csv(path, sep="\t")
    # Normalize columns
    rename = {}
    if "X" in df.columns:
        rename["X"] = "Drug"
    if "smiles" in df.columns:
        rename["smiles"] = "Drug"
    df = df.rename(columns=rename)

    if "Drug" not in df.columns or "Y" not in df.columns:
        raise RuntimeError("Local dataset must contain 'Drug' and 'Y' columns")

    split = _make_splits_local(df[["Drug", "Y"]], split_type=split_type)
    print("  Using LOCAL dataset fallback (no TDC/network)")
    print(f"  Local file: {os.path.basename(path)} | Split: {split_type}")
    return split


def row_to_graph(from_smiles_fn, smiles: str, y_value: float):
    """
    Convert SMILES + label to PyTorch Geometric Data object

    Args:
        from_smiles_fn: PyG's from_smiles function
        smiles: SMILES string
        y_value: Target value

    Returns:
        PyG Data object or None if invalid
    """
    g = from_smiles_fn(smiles)
    if g is None or getattr(g, "x", None) is None:
        return None

    if RDKit_OK:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Enhanced atom features
        atom_features = [enhanced_atom_features(atom) for atom in mol.GetAtoms()]
        g.x = torch.tensor(np.array(atom_features), dtype=torch.float32)

        # Enhanced bond features
        if mol.GetNumBonds() > 0:
            bond_features = []
            for bond in mol.GetBonds():
                bond_features.append(enhanced_bond_features(bond))
                bond_features.append(enhanced_bond_features(bond))  # Reverse edge
            g.edge_attr = torch.tensor(np.array(bond_features), dtype=torch.float32)
        else:
            g.edge_attr = torch.zeros((0, 12), dtype=torch.float32)

    g.x = g.x.float()
    g.edge_index = g.edge_index.long()
    g.original_y = float(y_value)
    g.y = torch.tensor([0.0], dtype=torch.float)  # Placeholder, will be transformed later
    g.adme_features = torch.tensor(adme_specific_descriptors(smiles), dtype=torch.float)
    g.smiles = smiles  # Important for text encoders!

    return g


def df_to_graph_list(from_smiles_fn, df):
    """
    Convert pandas DataFrame to list of graph objects

    Args:
        from_smiles_fn: PyG's from_smiles function
        df: DataFrame with 'Drug' (SMILES) and 'Y' (target) columns

    Returns:
        List of PyG Data objects
    """
    out = []
    for _, r in df.iterrows():
        g = row_to_graph(from_smiles_fn, r["Drug"], r["Y"])
        if g is not None:
            out.append(g)
    return out


def build_loaders(dataset_name, split_type="scaffold", batch_train=32, batch_eval=64):
    """
    Build train/val/test data loaders from TDC dataset

    Args:
        dataset_name: TDC ADME dataset name
        split_type: Splitting method (scaffold, random, etc.)
        batch_train: Training batch size
        batch_eval: Evaluation batch size

    Returns:
        Tuple of (train_loader, val_loader, test_loader, adme_dim)
    """
    # Try online TDC first; if unavailable, use local fallback
    try:
        ADME, from_smiles_fn = _import_tdc_and_from_smiles()
        data_api = ADME(name=dataset_name)
        try:
            split = data_api.get_split(method=split_type)
        except Exception:
            split = data_api.get_split()
    except Exception:
        # Offline/local path
        from torch_geometric.utils import from_smiles as from_smiles_fn  # type: ignore
        split = _load_local_dataset(dataset_name, split_type)

    print(f"  Dataset: {dataset_name} ({split_type} split)")
    print(f"  Train: {len(split['train'])}, Test: {len(split['test'])}")

    # Fit target scaler
    y_train = split['train']['Y'].values.astype(float)
    set_target_scaler(y_train)

    # Convert to graphs
    train_list = df_to_graph_list(from_smiles_fn, split["train"])
    valid_df = split.get("valid") if "valid" in split else split.get("val")
    valid_list = df_to_graph_list(from_smiles_fn, valid_df) if valid_df is not None else []
    test_list = df_to_graph_list(from_smiles_fn, split["test"])

    # Create validation split if not provided
    if not valid_list:
        n = len(train_list)
        k = max(1, int(0.15 * n))
        valid_list = train_list[:k]
        train_list = train_list[k:]

    # Transform targets
    for ds in [train_list, valid_list, test_list]:
        for g in ds:
            g.y = torch.tensor([transform_y(g.original_y)], dtype=torch.float)

    # Fit ADME scaler on training data
    X = np.stack([d.adme_features.numpy() for d in train_list], axis=0)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    ADME_SCALER["mu"] = torch.tensor(mu, dtype=torch.float32)
    ADME_SCALER["sigma"] = torch.tensor(sigma, dtype=torch.float32)

    # Standardize ADME features
    for ds in [train_list, valid_list, test_list]:
        for g in ds:
            g.adme_features = (g.adme_features - ADME_SCALER["mu"]) / (ADME_SCALER["sigma"] + 1e-8)

    # Create loaders
    train_loader = DataLoader(train_list, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(valid_list, batch_size=batch_eval, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_eval, shuffle=False)

    adme_dim = int(train_list[0].adme_features.numel())

    return train_loader, val_loader, test_loader, adme_dim
