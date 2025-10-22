"""
Molecular featurization for graph neural networks
"""
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKit_OK = True
except ImportError:
    RDKit_OK = False


def enhanced_atom_features(atom):
    """
    Extract enhanced atomic features for GNN

    Args:
        atom: RDKit atom object

    Returns:
        np.array of atom features (27 dimensions)
    """
    try:
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetMass(),
            # Hybridization
            int(atom.GetHybridization() == Chem.HybridizationType.SP),
            int(atom.GetHybridization() == Chem.HybridizationType.SP2),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3),
            int(atom.GetHybridization() == Chem.HybridizationType.SP3D),
            # Aromaticity & rings
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            int(atom.IsInRingSize(3)),
            int(atom.IsInRingSize(4)),
            int(atom.IsInRingSize(5)),
            int(atom.IsInRingSize(6)),
            int(atom.IsInRingSize(7)),
            # Valence
            atom.GetTotalValence(),
            atom.GetImplicitValence(),
            atom.GetExplicitValence(),
            # Chirality
            int(atom.GetChiralTag()),
            # Common elements
            int(atom.GetSymbol() == 'C'),
            int(atom.GetSymbol() == 'N'),
            int(atom.GetSymbol() == 'O'),
            int(atom.GetSymbol() == 'S'),
            int(atom.GetSymbol() == 'F'),
            int(atom.GetSymbol() == 'Cl'),
            int(atom.GetSymbol() == 'Br'),
        ]
        return np.array(features, dtype=np.float32)
    except:
        return np.zeros(27, dtype=np.float32)


def enhanced_bond_features(bond):
    """
    Extract enhanced bond features for GNN

    Args:
        bond: RDKit bond object

    Returns:
        np.array of bond features (12 dimensions)
    """
    try:
        bt = bond.GetBondType()
        features = [
            # Bond type
            int(bt == Chem.BondType.SINGLE),
            int(bt == Chem.BondType.DOUBLE),
            int(bt == Chem.BondType.TRIPLE),
            int(bt == Chem.BondType.AROMATIC),
            float(bond.GetBondTypeAsDouble()),
            # Properties
            int(bond.GetIsAromatic()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            # Stereochemistry
            int(bond.GetStereo() == Chem.BondStereo.STEREONONE),
            int(bond.GetStereo() == Chem.BondStereo.STEREOZ),
            int(bond.GetStereo() == Chem.BondStereo.STEREOE),
            int(bond.GetStereo() == Chem.BondStereo.STEREOANY),
        ]
        return np.array(features, dtype=np.float32)
    except:
        # Default to single bond
        return np.array([1, 0, 0, 0, 1.0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)


def adme_specific_descriptors(smiles: str):
    """
    Compute ADME-relevant molecular descriptors

    Args:
        smiles: SMILES string

    Returns:
        np.array of ADME descriptors (20 dimensions)
    """
    if not RDKit_OK:
        return np.zeros(20, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(20, dtype=np.float32)

    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)

        descriptors = [
            mw,
            logp,
            hbd,
            hba,
            Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            # Lipinski rule of 5 violations
            int(mw > 500),
            int(logp > 5),
            int(hbd > 5),
            int(hba > 10),
            # Additional descriptors
            Descriptors.MolMR(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.BertzCT(mol),
            Descriptors.Chi0v(mol),
            Descriptors.NumHeteroatoms(mol),
            rdMolDescriptors.CalcNumHeterocycles(mol),
            int(mw < 200),
            int(mw > 800),
        ]
        return np.array(descriptors[:20], dtype=np.float32)
    except:
        return np.zeros(20, dtype=np.float32)
