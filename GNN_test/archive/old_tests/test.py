import math
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention

from test_tdc import UniversalGraphStack, GNNTester

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKit_OK = True
except Exception:
    RDKit_OK = False


def enhanced_atom_features(atom):
    """Extract comprehensive atom features for molecular graphs"""
    try:
        features = [
            # Basic atomic properties
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetTotalNumHs(),
            atom.GetMass(),

            # Ring membership
            int(atom.IsInRingSize(3)),
            int(atom.IsInRingSize(4)),
            int(atom.IsInRingSize(5)),
            int(atom.IsInRingSize(6)),
            int(atom.IsInRingSize(7)),
            int(atom.IsInRingSize(8)),

            # Chemical environment
            atom.GetTotalValence(),
            atom.GetImplicitValence(),
            atom.GetExplicitValence(),
            int(atom.GetChiralTag()),

            # Connectivity
            len(atom.GetNeighbors()),
            atom.GetTotalDegree(),

            # Electronic properties
            atom.GetAtomicNum() / 100.0,  # Normalized atomic number
            int(atom.IsInRing()),
            int(atom.GetIsAromatic()),

            # Chemical type indicators
            int(atom.GetSymbol() == 'C'),
            int(atom.GetSymbol() == 'N'),
            int(atom.GetSymbol() == 'O'),
            int(atom.GetSymbol() == 'S'),
            int(atom.GetSymbol() == 'P'),
            int(atom.GetSymbol() == 'F'),
            int(atom.GetSymbol() == 'Cl'),
            int(atom.GetSymbol() == 'Br'),
            int(atom.GetSymbol() == 'I'),

            # Hybridization states
            int(atom.GetHybridization() == Chem.HybridizationType.SP),
            int(atom.GetHybridization() == Chem.HybridizationType.SP2),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3D),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3D2),
        ]

        return np.array(features, dtype=np.float32)

    except Exception:
        # Fallback to basic features
        return np.array([
                            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
                            int(atom.GetHybridization()), int(atom.GetIsAromatic()),
                            int(atom.IsInRing()), atom.GetTotalNumHs(), atom.GetMass()
                        ] + [0.0] * 30, dtype=np.float32)  # Pad to consistent size

def _rdkit_feature_names() -> List[str]:
    return [
        "MolWt", "TPSA", "MolLogP",
        "NumHDonors", "NumHAcceptors",
        "NumRotatableBonds", "RingCount",
        "HeavyAtomCount", "NumAromaticRings",
        "FractionCSP3", "BertzCT",
        "NOCount", "NHOHCount",
    ]


def _rdkit_features_from_smiles(smiles: str) -> np.ndarray:
    if not RDKit_OK:
        return np.zeros(50, dtype=np.float32)
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(50, dtype=np.float32)

    try:
        descriptors = [
            # Основни молекуларни својства
            Descriptors.MolWt(m),
            rdMolDescriptors.CalcTPSA(m),
            Descriptors.MolLogP(m),
            rdMolDescriptors.CalcNumHBD(m),
            rdMolDescriptors.CalcNumHBA(m),
            Descriptors.NumRotatableBonds(m),
            rdMolDescriptors.CalcNumRings(m),
            Descriptors.HeavyAtomCount(m),
            rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcFractionCSP3(m),
            Descriptors.BertzCT(m),

            # Connectivity indices
            Descriptors.Chi0(m),
            Descriptors.Chi1(m),
            Descriptors.Chi0n(m),
            Descriptors.Chi1n(m),
            Descriptors.Chi2n(m),
            Descriptors.Chi3n(m),
            Descriptors.Chi4n(m),

            # Kappa shape indices
            Descriptors.Kappa1(m),
            Descriptors.Kappa2(m),
            Descriptors.Kappa3(m),

            # Ring-related descriptors
            rdMolDescriptors.CalcNumAliphaticRings(m),
            rdMolDescriptors.CalcNumSaturatedRings(m),
            rdMolDescriptors.CalcNumHeterocycles(m),
            rdMolDescriptors.CalcNumAromaticHeterocycles(m),
            rdMolDescriptors.CalcNumAliphaticHeterocycles(m),
            rdMolDescriptors.CalcNumSaturatedHeterocycles(m),

            # Pharmacophore-related
            Descriptors.NumHeteroatoms(m),
            Descriptors.NumSaturatedCarbocycles(m),
            Descriptors.NumSaturatedHeterocycles(m),
            Descriptors.NumAromaticCarbocycles(m),
            Descriptors.NumAromaticHeterocycles(m),

            # Electronic properties
            Descriptors.MaxEStateIndex(m),
            Descriptors.MinEStateIndex(m),
            Descriptors.MaxAbsEStateIndex(m),
            Descriptors.MinAbsEStateIndex(m),

            # Lipophilicity-related
            Descriptors.MolMR(m),  # Molar refractivity
            Descriptors.LabuteASA(m),  # Labute's Accessible Surface Area

            # Flexibility
            Descriptors.NumAliphaticCarbocycles(m),
            Descriptors.NumAliphaticHeterocycles(m),

            # Complexity measures
            Descriptors.HallKierAlpha(m),
            rdMolDescriptors.CalcNumAmideBonds(m),
            rdMolDescriptors.CalcNumBridgeheadAtoms(m),
            rdMolDescriptors.CalcNumSpiroAtoms(m),

            # Additional structural descriptors
            Descriptors.FractionCsp3(m),
            rdMolDescriptors.CalcNumHeteroatoms(m),
            rdMolDescriptors.CalcNumRadicalElectrons(m),
            rdMolDescriptors.CalcNumValenceElectrons(m),

            # Final descriptors to reach 50
            len(rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024).GetOnBits()),
            float(rdMolDescriptors.GetHashedMorganFingerprint(m, radius=2).GetTotalVal())
        ]

        # Ensure exactly 50 features
        descriptors = descriptors[:50]
        while len(descriptors) < 50:
            descriptors.append(0.0)

        return np.array(descriptors, dtype=np.float32)

    except Exception as e:
        # Fallback in case of calculation errors
        return np.zeros(50, dtype=np.float32)


def _import_tdc_and_from_smiles():
    try:
        from tdc.single_pred import ADME
    except Exception as e:
        raise RuntimeError("TDC not installed. Install: pip install tdc") from e
    try:
        from torch_geometric.utils import from_smiles
    except Exception:
        try:
            from torch_geometric.data import from_smiles
        except Exception as e:
            raise RuntimeError("PyG 'from_smiles' unavailable. Update torch-geometric.") from e
    return ADME, from_smiles


def row_to_graph(from_smiles_fn, smiles: str, y_value: float, rdkit_vec: np.ndarray):
    g = from_smiles_fn(smiles)
    if g is None or getattr(g, "x", None) is None:
        return None

    # Заменете ги default atom features со enhanced
    if RDKit_OK:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append(enhanced_atom_features(atom))
            g.x = torch.tensor(np.array(atom_features), dtype=torch.float32)

    g.x = g.x.float()
    g.edge_index = g.edge_index.long()
    g.y = torch.tensor([np.log1p(float(y_value))], dtype=torch.float)
    g.rdkit = torch.tensor(rdkit_vec, dtype=torch.float).unsqueeze(0)
    return g


def df_to_graph_list(from_smiles_fn, df: pd.DataFrame) -> List[Data]:
    out: List[Data] = []
    for _, r in df.iterrows():
        smi = r["Drug"]
        yv = r["Y"]
        rdv = _rdkit_features_from_smiles(smi)
        g = row_to_graph(from_smiles_fn, smi, yv, rdv)
        if g is not None:
            out.append(g)
    return out


def build_tdc_loaders(
    dataset_name: str,
    batch_train: int = 128,
    batch_eval: int = 256,
    split_type: str = "scaffold",
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    ADME, from_smiles_fn = _import_tdc_and_from_smiles()
    data_api = ADME(name=dataset_name)
    try:
        split = data_api.get_split(method=split_type)
    except Exception:
        split = data_api.get_split()
    train_list = df_to_graph_list(from_smiles_fn, split["train"])
    valid_df = split["valid"] if "valid" in split else split.get("val", None)
    valid_list = df_to_graph_list(from_smiles_fn, valid_df) if valid_df is not None else []
    test_list = df_to_graph_list(from_smiles_fn, split["test"])
    if not valid_list:
        n = len(train_list)
        k = max(1, int(0.1 * n))
        valid_list = train_list[:k]
        train_list = train_list[k:]
    rdkit_dim = train_list[0].rdkit.numel() if len(train_list) else 0
    train_loader = DataLoader(train_list, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(valid_list, batch_size=batch_eval, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_eval, shuffle=False)
    return train_loader, val_loader, test_loader, rdkit_dim


class CombinedReadout(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(in_dim, max(32, in_dim // 2)),
                nn.GELU(),
                nn.Linear(max(32, in_dim // 2), 1),
            )
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean_p = global_mean_pool(x, batch)
        max_p = global_max_pool(x, batch)
        att_p = self.attn(x, batch)
        return torch.cat([mean_p, max_p, att_p], dim=-1)


class GraphRegressorEnriched(nn.Module):
    def __init__(self, backbone: nn.Module, rdkit_dim: int, hidden_head: int = 128, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.readout = CombinedReadout(backbone.output_dim)
        readout_dim = backbone.output_dim * 3
        self.rdkit_dim = rdkit_dim
        in_head = readout_dim + rdkit_dim
        self.head = nn.Sequential(
            nn.Linear(in_head, hidden_head),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_head, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.backbone(data)
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        gpool = self.readout(x, batch)
        rdkit = getattr(data, "rdkit", None)
        if rdkit is None:
            rdkit = torch.zeros(gpool.size(0), self.rdkit_dim, device=gpool.device, dtype=gpool.dtype)
        else:
            if rdkit.dim() == 1:
                rdkit = rdkit.view(-1, self.rdkit_dim)
        out = torch.cat([gpool, rdkit], dim=-1)
        out = self.head(out).squeeze(-1)
        return out


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device, max_grad_norm: float = 2.0):
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        y = data.y.view_as(pred)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += float(loss.item()) * data.num_graphs
    return total / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, device, inverse: bool = False) -> Dict[str, float]:
    model.eval()
    preds, trues = [], []
    for data in loader:
        data = data.to(device)
        p = model(data)
        preds.append(p.detach().cpu())
        trues.append(data.y.view_as(p).detach().cpu())
    if not preds:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    if inverse:
        preds = torch.expm1(preds)
        trues = torch.expm1(trues)
    mse = torch.mean((preds - trues) ** 2).item()
    mae = torch.mean(torch.abs(preds - trues)).item()
    var = torch.var(trues, unbiased=False).item()
    r2 = 1.0 - (mse / (var + 1e-12))
    return {"rmse": math.sqrt(mse), "mae": mae, "r2": r2}


def build_model_from_config(config: Dict, rdkit_dim: int, device: str):
    backbone = UniversalGraphStack(**config).to(device)
    model = GraphRegressorEnriched(backbone, rdkit_dim=rdkit_dim, hidden_head=128, dropout=0.25).to(device)
    return model


def train_eval_single(
    config: Dict,
    dataset_name: str,
    epochs: int = 300,
    patience: int = 30,
    batch_train: int = 128,
    batch_eval: int = 256,
    device: str = None,
    seed: int = 42,
    wd_default: float = 1e-4,
) -> Tuple[Dict, Dict[str, float], Dict[str, float], Dict]:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, rdkit_dim = build_tdc_loaders(dataset_name, batch_train, batch_eval, split_type="scaffold")
    model = build_model_from_config(config, rdkit_dim, device)
    lr = float(config.get("learning_rate", 1e-3))
    wd = float(config.get("weight_decay", wd_default))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(5, patience // 3), min_lr=1e-6)
    best = {"val_rmse": float("inf"), "state": None, "epoch": 0}
    no_imp = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        _ = train_one_epoch(model, train_loader, optimizer, device, max_grad_norm=2.0)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["rmse"])
        if val_metrics["rmse"] + 1e-6 < best["val_rmse"]:
            best.update(val_rmse=val_metrics["rmse"], state=deepcopy(model.state_dict()), epoch=epoch)
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    meta = {
        **config,
        "dataset": dataset_name,
        "epochs_trained": int(best["epoch"]),
        "timestamp": datetime.now().isoformat(),
        "train_time_s": round(time.time() - t0, 2),
        "success": True,
        "seed": int(seed),
    }
    return meta, val_metrics, test_metrics, {"rdkit_dim": rdkit_dim}


@torch.no_grad()
def _predict_with_models(models: List[nn.Module], loader, device) -> torch.Tensor:
    preds = []
    for m in models:
        m.eval()
        cur = []
        for data in loader:
            data = data.to(device)
            cur.append(m(data).detach().cpu())
        preds.append(torch.cat(cur))
    return torch.stack(preds, dim=0).mean(dim=0)


def train_eval_ensemble(
    config: Dict,
    dataset_name: str,
    ensemble_size: int = 5,
    base_seed: int = 42,
    epochs: int = 300,
    patience: int = 30,
    batch_train: int = 128,
    batch_eval: int = 256,
    device: str = None,
) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, rdkit_dim = build_tdc_loaders(dataset_name, batch_train, batch_eval, split_type="scaffold")
    models = []
    members_meta = []
    for k in range(ensemble_size):
        seed = base_seed + k * 11
        meta, _, _, _ = train_eval_single(config, dataset_name, epochs, patience, batch_train, batch_eval, device, seed)
        members_meta.append(meta)
    models = []
    loaders_cache = build_tdc_loaders(dataset_name, batch_train, batch_eval, split_type="scaffold")
    train_loader, val_loader, test_loader, rdkit_dim = loaders_cache
    for k in range(ensemble_size):
        seed = base_seed + k * 11
        set_seed(seed)
        model = build_model_from_config(config, rdkit_dim, device)
        lr = float(config.get("learning_rate", 1e-3))
        wd = float(config.get("weight_decay", 1e-4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(5, patience // 3), min_lr=1e-6)
        best_val = float("inf")
        best_state = None
        no_imp = 0
        for epoch in range(1, epochs + 1):
            _ = train_one_epoch(model, train_loader, optimizer, device, max_grad_norm=2.0)
            vm = evaluate(model, val_loader, device)
            scheduler.step(vm["rmse"])
            if vm["rmse"] + 1e-6 < best_val:
                best_val = vm["rmse"]
                best_state = deepcopy(model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        models.append(model)
    with torch.no_grad():
        val_pred = _predict_with_models(models, val_loader, device)
        val_true = torch.cat([b.y.view(-1).cpu() for b in val_loader.dataset])
        test_pred = _predict_with_models(models, test_loader, device)
        test_true = torch.cat([b.y.view(-1).cpu() for b in test_loader.dataset])

    def _metrics(p, t):
        mse = torch.mean((p - t) ** 2).item()
        mae = torch.mean(torch.abs(p - t)).item()
        var = torch.var(t, unbiased=False).item()
        r2 = 1.0 - (mse / (var + 1e-12))
        return {"rmse": math.sqrt(mse), "mae": mae, "r2": r2}

    val_m = _metrics(val_pred, val_true)
    test_m = _metrics(test_pred, test_true)
    return {
        "meta": {**config, "dataset": dataset_name, "ensemble_size": ensemble_size, "success": True},
        "val": val_m,
        "test": test_m,
        "members": members_meta,
    }


def run_excretion_benchmark_plus(
    max_combos_per_dataset: int = 16,
    epochs: int = 300,
    patience: int = 30,
    device: str = None,
    seed: int = 42,
    ensemble_size: int = 3,
    out_prefix: str = "tdc_excretion_plus",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dummy = GNNTester(train_data=Data(x=torch.randn(4, 8), edge_index=torch.tensor([[0, 1], [1, 2]])))
    combos = dummy.create_smart_parameter_combinations(max_combinations=max_combos_per_dataset)
    datasets = ["Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"]
    all_rows = []
    print(f"\n🧪 EXCRETION+ (RDKit + Strong Readout + Ensemble) on {device}")
    print(f"Datasets: {datasets}")
    print(f"Combos per dataset: {len(combos)}, Ensemble size: {ensemble_size}")
    print(f"Training: {epochs} epochs, patience={patience}\n")
    for dname in datasets:
        print(f"\n====================== Dataset: {dname} ======================")
        rows = []
        for i, cfg in enumerate(combos, 1):
            try:
                cfg = {**cfg}
                cfg.setdefault("residual", True)
                cfg.setdefault("graph_dropouts", 0.2)
                cfg.setdefault("graph_norm", "BatchNorm")
                print(f"({i}/{len(combos)}) {cfg['model_name']} L={cfg['graph_layers']} H={cfg['graph_hidden_channels']}")
                meta, val_m, test_m, aux = train_eval_single(
                    cfg,
                    dname,
                    epochs=epochs // 2,
                    patience=max(10, patience // 2),
                    batch_train=128,
                    batch_eval=256,
                    device=device,
                    seed=seed,
                )
                row = {
                    **cfg,
                    "dataset": dname,
                    "val_rmse": val_m["rmse"],
                    "test_rmse": test_m["rmse"],
                    "val_mae": val_m["mae"],
                    "test_mae": test_m["mae"],
                    "val_r2": val_m["r2"],
                    "test_r2": test_m["r2"],
                    "epochs_trained": meta["epochs_trained"],
                    "seed": seed,
                    "success": True,
                }
                rows.append(row)
                all_rows.append(row)
                print(f"   -> single valRMSE={val_m['rmse']:.3f}  testRMSE={test_m['rmse']:.3f}")
            except Exception as e:
                print(f"   !! Failed: {e}")
                rows.append({**cfg, "dataset": dname, "success": False, "error": str(e), "timestamp": datetime.now().isoformat()})
        df = pd.DataFrame(rows)
        ok = df[df["success"] == True]
        topK = min(3, len(ok))
        refined = []
        if topK > 0 and ensemble_size > 1:
            shortlist = ok.nsmallest(topK, "val_rmse")
            print(f"\n🔁 Ensemble refinement for top-{topK} configs …")
            for _, r in shortlist.iterrows():
                cfg = {
                    k: r[k]
                    for k in [
                        "model_name",
                        "graph_layers",
                        "graph_hidden_channels",
                        "graph_dropouts",
                        "graph_norm",
                        "activation",
                        "heads",
                        "aggr",
                        "residual",
                        "learning_rate",
                        "weight_decay",
                        "K",
                    ]
                    if k in r and not pd.isna(r[k])
                }
                result = train_eval_ensemble(
                    cfg,
                    dname,
                    ensemble_size=ensemble_size,
                    base_seed=seed,
                    epochs=epochs,
                    patience=patience,
                    batch_train=128,
                    batch_eval=256,
                    device=device,
                )
                ref_row = {
                    **cfg,
                    "dataset": dname,
                    "ensemble_size": ensemble_size,
                    "val_rmse": result["val"]["rmse"],
                    "test_rmse": result["test"]["rmse"],
                    "val_mae": result["val"]["mae"],
                    "test_mae": result["test"]["mae"],
                    "val_r2": result["val"]["r2"],
                    "test_r2": result["test"]["r2"],
                    "success": True,
                    "refined": True,
                }
                refined.append(ref_row)
                all_rows.append(ref_row)
                print(f"   -> ensemble valRMSE={ref_row['val_rmse']:.3f}  testRMSE={ref_row['test_rmse']:.3f}")
        out_df = pd.concat([df, pd.DataFrame(refined)], ignore_index=True) if refined else df
        csv_path = f"{out_prefix}_{dname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out_df.to_csv(csv_path, index=False)
        print(f"💾 Saved: {csv_path}")
        ok2 = out_df[out_df["success"] == True]
        if len(ok2):
            top = ok2.nsmallest(5, "val_rmse")[
                [
                    "model_name",
                    "graph_layers",
                    "graph_hidden_channels",
                    "graph_norm",
                    "activation",
                    "val_rmse",
                    "test_rmse",
                    "test_mae",
                    "test_r2",
                    "ensemble_size",
                ]
            ]
            print("\n🏆 Top-5 by val RMSE")
            for _, r in top.iterrows():
                ens = int(r.get("ensemble_size", 1)) if not pd.isna(r.get("ensemble_size", 1)) else 1
                print(
                    f"   {r['model_name']:11} L={int(r['graph_layers'])} H={int(r['graph_hidden_channels'])} "
                    f"{r['graph_norm']:>9} {str(r['activation']):>6} "
                    f"| val {r['val_rmse']:.3f} | test {r['test_rmse']:.3f} | "
                    f"MAE {r['test_mae']:.2f} | R² {r['test_r2']:.3f} | ens x{ens}"
                )
    all_df = pd.DataFrame(all_rows)
    all_csv = f"{out_prefix}_ALL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"\n📦 Global results saved: {all_csv}")
    return all_df


if __name__ == "__main__":
    MAX_COMBOS_PER_DATASET = 100
    EPOCHS = 400
    PATIENCE = 50
    ENSEMBLE = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    run_excretion_benchmark_plus(
        max_combos_per_dataset=MAX_COMBOS_PER_DATASET,
        epochs=EPOCHS,
        patience=PATIENCE,
        device=DEVICE,
        seed=42,
        ensemble_size=ENSEMBLE,
        out_prefix="tdc_comprehensive_research",
    )