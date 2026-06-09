!pip install /kaggle/input/neurips-open-polymer-prediction-2025-dataset/packages/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install --no-deps /kaggle/input/neurips-open-polymer-prediction-2025-dataset/packages/scikit_learn-1.7.1-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
!pip install /kaggle/input/neurips-open-polymer-prediction-2025-dataset/packages/mordredcommunity-2.0.6-py3-none-any.whl
!pip install /kaggle/input/neurips-open-polymer-prediction-2025-dataset/packages/deepchem-2.8.0-py3-none-any.whl


import os
import warnings
from pathlib import Path
import pickle
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdReducedGraphs, LayeredFingerprint, PatternFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem.rdFingerprintGenerator import (
    GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator, GetRDKitFPGenerator,
    GetMorganAtomInvGen, GetMorganFeatureAtomInvGen, GetRDKitAtomInvGen
)
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from mordred import Calculator, descriptors
from mordred.error import Missing, Error
import deepchem as dc


warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

competition_dataset_directory = Path('/kaggle/input/neurips-open-polymer-prediction-2025')
external_dataset_directory = Path('/kaggle/input/neurips-open-polymer-prediction-2025-dataset')

pd.set_option('display.float_format', '{:.6f}'.format)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


dataset_path = competition_dataset_directory / 'test.csv'

df = pd.read_csv(dataset_path)
print(f'Dataset is loaded from {dataset_path} Shape: {df.shape}')
display(df)


def make_smiles_canonical(smiles):

    """
    Convert a SMILES string to its canonical form.

    This function parses a SMILES string into an RDKit Mol object and returns the canonical SMILES representation.
    Canonical SMILES ensures consistent atom ordering and structure formatting.
    It is useful for deduplication and data standardization in cheminformatics workflows.

    Parameters
    ----------
    smiles: str
        A valid SMILES string representing a molecule.

    Returns
    -------
    canonical_smiles: str
        The canonical SMILES string.
    """

    mol = Chem.MolFromSmiles(smiles)
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    return canonical_smiles



df['SMILES'] = df['SMILES'].apply(make_smiles_canonical)


def compute_rdkit_descriptors(mol):

    """
    Compute all RDKit molecular descriptors for a given molecule.

    This function computes a comprehensive set of molecular descriptors provided by RDKit for a given molecule.
    If a descriptor function fails, its value is set to None.
    These descriptors include constitutional, topological, electronic, and geometrical properties.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    Returns
    -------
    descriptors: dict
        A dictionary mapping descriptor names to their computed values or None if failed.
    """

    descriptors = {}

    for name, function in Descriptors._descList:
        try:
            descriptors[name] = function(mol)
        except:
            descriptors[name] = None

    descriptors['NumAtoms'] = rdMolDescriptors.CalcNumAtoms(mol)

    return descriptors


def compute_estate_fingerprint(mol):

    """
    Compute the Electrotopological State (EState) fingerprint for a given RDKit molecule.

    This function returns the EState fingerprint as described by Hall and Kier (JCICS, 1995).
    It encodes both the topological and electronic environment of atoms in a molecule.

    The EState fingerprint consists of two numeric arrays:
        1. An integer array (length 79):
           Each element represents the number of atoms in the molecule that match one of the
           79 predefined EState atom types. This captures the count of specific atom types based
           on atomic number, bonding pattern, and hybridization.

        2. A float array (length 79):
           Each element is the sum of EState indices for all atoms matching that atom type.
           The EState index is a numeric value that reflects both the electronic and topological
           environment of each atom.

    Together, these arrays provide a chemically meaningful fingerprint useful for
    quantitative structure–activity relationship (QSAR) modeling and other cheminformatics tasks.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    Returns
    -------
    estate_fingerprint: np.ndarray of shape (158, )
        Array of EState fingerprint matrix.
    """

    estate_fingerprint = FingerprintMol(mol)
    estate_fingerprint = np.concatenate(estate_fingerprint, axis=0).astype(np.float32)

    return estate_fingerprint


def compute_erg_fingerprint(mol, atomTypes, fuzzIncrement, minPath, maxPath):

    """
    Compute the Extended Reduced Graph (ErG) fingerprint for a given RDKit molecule.

    The ErG fingerprint is a sparse float vector representing the weighted occurrence of pharmacophore-like features.
    It is based on a reduced graph abstraction of the molecule.
    Unlike binary or integer-count fingerprints, the ErG fingerprint values are floats due to fuzzy matching of similar graph paths.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    Returns
    -------
    erg_fingerprint: np.ndarray
        Array of float values representing the ErG fingerprint.
    """

    return rdReducedGraphs.GetErGFingerprint(
        mol,
        atomTypes=atomTypes,
        fuzzIncrement=fuzzIncrement,
        minPath=minPath,
        maxPath=maxPath
    ).astype(np.float32)


def compute_maccs_keys(mol):

    """
    Compute the 167-bit MACCS structural key fingerprint for a given RDKit molecule.

    MACCS keys are a fixed-length binary fingerprint.
    Each bit represents the presence or absence of a predefined substructure or functional group in the molecule.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    Returns
    -------
    maccs_fingerprint: np.ndarray of shape (167, )
        A binary NumPy array representing the MACCS structural keys.
    """

    return np.array(GenMACCSKeys(mol).ToList(), dtype=np.uint8)


def compute_morgan_fingerprint(mol, morgan_generator):

    """
    Compute a circular Morgan fingerprint (ECFP-like) for a molecule.

    This function uses a preconfigured RDKit `MorganFingerprintGenerator` object to compute the hashed bit vector representation of atom environments up to a specified radius.
    Variants (ECFP/FCFP, counts/bits, chirality, etc.) are controlled by the generator settings passed in.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    morgan_generator: rdkit.Chem.rdFingerprintGenerator.MorganFingerprintGenerator
        Preconfigured generator.

    Returns
    -------
    fingerprint: np.ndarray of shape (fpSize, )
        Binary or count array (depending on generator) representing the Morgan fingerprint.
    """

    return np.array(morgan_generator.GetCountFingerprint(mol).ToList(), dtype=np.uint16)


def compute_atom_pairs(mol, atom_pair_generator):

    """
    Compute the Atom Pair fingerprint for a molecule.

    Atom Pair fingerprints encode all unique pairs of atoms in a molecule along with their topological (bond) distance.
    Each feature corresponds to a specific atom-type pair at a specific separation.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    atom_pair_generator: rdkit.Chem.rdFingerprintGenerator.AtomPairFingerprintGenerator
        Preconfigured generator specifying distance limits, size, and count/binary mode.

    Returns
    -------
    fingerprint: np.ndarray of shape (fpSize, )
        Binary or count vector representing the atom pair fingerprint.
    """

    return np.array(atom_pair_generator.GetCountFingerprint(mol).ToList(), dtype=np.uint16)


def compute_rdkit_fingerprint(mol, rdkit_fingerprint_generator):

    """
    Compute the RDKit topological fingerprint for a molecule.

    This is the default RDKit path-based fingerprint, encoding subgraphs corresponding to simple paths between atoms of lengths within a specified range.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    rdkit_fingerprint_generator: rdkit.Chem.rdFingerprintGenerator.FingerprintGenerator64
        Preconfigured generator controlling min/max path, fingerprint size, and count/binary mode.

    Returns
    -------
    fingerprint: np.ndarray of shape (fpSize, )
        Binary or count vector representing the topological torsion fingerprint.
    """

    return np.array(rdkit_fingerprint_generator.GetCountFingerprint(mol).ToList(), dtype=np.uint16)


def compute_topological_torsion(mol, topological_torsion_generator):

    """
    Compute the Topological Torsion fingerprint for a molecule.

    The Topological Torsion fingerprint encodes atom sequences of a given torsion length (typically four atoms) along topological paths.
    Each feature represents the atom types and their connectivity.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    topological_torsion_generator: rdkit.Chem.rdFingerprintGenerator.TopologicalTorsionFingerprintGenerator
        Preconfigured generator controlling torsion length, fingerprint size, and count/binary mode.

    Returns
    -------
    fingerprint: np.ndarray of shape (fpSize, )
        Binary or count vector representing the topological torsion fingerprint.
    """

    return np.array(topological_torsion_generator.GetCountFingerprint(mol).ToList(), dtype=np.uint16)


def compute_layered_fingerprint(mol):

    """
    Compute the Layered fingerprint for a molecule.

    The Layered fingerprint encodes subgraph patterns based on atom and bond types along paths of varying lengths.
    It supports hierarchical feature layers for substructure searching.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    Returns
    -------
    fingerprint: np.ndarray of shape (4096, )
        Binary array representing the Layered fingerprint.
    """

    return np.array(LayeredFingerprint(mol, fpSize=4096, minPath=1, maxPath=8).ToList(), dtype=np.uint8)


def compute_pattern_fingerprint(mol):

    """
    Compute the Pattern fingerprint for a molecule.

    The Pattern fingerprint is an RDKit substructure fingerprint that encodes predefined SMARTS patterns and is optimized for substructure search.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    Returns
    -------
    fingerprint  np.ndarray of shape (4096, )
        Binary array representing the Pattern fingerprint.
    """

    return np.array(PatternFingerprint(mol, fpSize=4096, tautomerFingerprints=True).ToList(), dtype=np.uint8)


def compute_avalon_fingerprint(mol):

    """
    Compute the Avalon fingerprint for a molecule.

    The Avalon fingerprint is a hashed substructure fingerprint implemented in the Avalon toolkit.
    It encodes paths, environments, and functional groups using Avalon’s hashing scheme.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    Returns
    -------
    fingerprint: np.ndarray of shape (4096, )
        Binary array representing the Avalon fingerprint.
    """

    return np.array(GetAvalonCountFP(mol, nBits=4096).ToList(), dtype=np.uint16)


def compute_invariants(mol, bits=64):

    """
    Compute hashed feature and connectivity invariant counts for a molecule.

    RDKit invariant functions provide integer encodings of atomic features and connectivity patterns.
    These invariants can be used as fixed-length hashed descriptors.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    bits: int
        Length of each hashed vector (features and connectivity).
        The final vector has length 2 * bits.

    Returns
    -------
    invariants: np.ndarray of shape (2 * bits, )
        Concatenated float array of hashed feature invariants and connectivity invariants.
    """

    feature_vector = np.zeros(bits, dtype=np.uint16)
    for i in rdMolDescriptors.GetFeatureInvariants(mol):
        feature_vector[i % bits] += 1

    connectivity_vector = np.zeros(bits, dtype=np.uint16)
    for i in rdMolDescriptors.GetConnectivityInvariants(mol):
        connectivity_vector[i % bits] += 1

    invariants = np.concatenate([feature_vector, connectivity_vector]).astype(np.float32)

    return invariants


def compute_brics_counts(mol, bits=64):

    """
    Compute hashed BRICS fragment counts for a molecule.

    BRICS (Breaking of Retrosynthetically Interesting Chemical Substructures) decomposes a molecule into chemically meaningful fragments.
    Each unique fragment SMILES is hashed into a fixed-length count vector.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.

    bits: int
        Length of the hashed count vector.

    Returns
    -------
    brics_counts: np.ndarray of shape (bits, )
        Count array where each element is the frequency of a hashed BRICS fragment.
    """

    brics_counts = np.zeros(bits, dtype=np.uint16)
    bonds = list(BRICS.FindBRICSBonds(mol))
    if len(bonds) < 40:
        fragments = BRICS.BRICSDecompose(mol, silent=True)
        for fragment in fragments:
            brics_counts[hash(fragment) % bits] += 1

    return brics_counts


def compute_scaffold_counts(mol, bits=64):

    """
    Compute hashed Bemis–Murcko scaffold counts for a molecule.

    The Murcko scaffold represents the core framework of a molecule by stripping side chains and leaving the ring systems and linkers.
    Each unique scaffold SMILES is hashed into a fixed-length count vector.

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        An RDKit molecule object.
        
    bits: int, default=64
        Length of the hashed count vector.

    Returns
    -------
    scaffold_counts: np.ndarray of shape (bits, )
        Count array where each element is the frequency of a hashed scaffold.
    """

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = Chem.MolToSmiles(scaffold) if scaffold is not None else ''
    scaffold_counts = np.zeros(bits, dtype=np.uint16)
    if scaffold:
        scaffold_counts[hash(scaffold) % bits] += 1
    return scaffold_counts


def compute_mol_graph_conv_features(smiles, featurizer):

    """
    Compute mol graph conv features.

    Parameters
    ----------
    smiles: str
        SMILES string.

    Returns
    -------
    features: dict
        A dictionary of mol graph conv features.
    """

    outputs = featurizer(smiles)[0]
    features = {}

    if isinstance(outputs, dc.feat.graph_data.GraphData):

        node_features = pd.DataFrame(outputs.node_features)
        node_features_flat = pd.Series(outputs.node_features.flatten())
        edge_features = pd.DataFrame(outputs.edge_features)
        edge_features_flat = pd.Series(outputs.edge_features.flatten())

        features['node_count'] = node_features.shape[0]
        features['node_sum'] = node_features_flat.sum()
        features['node_mean'] = node_features_flat.mean()
        features['node_std'] = node_features_flat.std()
        features['node_skew'] = node_features_flat.skew()
        features['node_kurt'] = node_features_flat.kurt()

        for node_idx in range(32):
            features[f'node_{node_idx}_sum'] = node_features.iloc[:, node_idx].sum()
            features[f'node_{node_idx}_mean'] = node_features.iloc[:, node_idx].mean()
            features[f'node_{node_idx}_std'] = node_features.iloc[:, node_idx].std()
            features[f'node_{node_idx}_skew'] = node_features.iloc[:, node_idx].skew()
            features[f'node_{node_idx}_kurt'] = node_features.iloc[:, node_idx].kurt()

        features['edge_count'] = edge_features.shape[0]
        features['edge_sum'] = edge_features_flat.sum()
        features['edge_mean'] = edge_features_flat.mean()
        features['edge_std'] = edge_features_flat.std()
        features['edge_skew'] = edge_features_flat.skew()
        features['edge_kurt'] = edge_features_flat.kurt()

        for edge_idx in range(11):
            features[f'edge_{edge_idx}_sum'] = edge_features.iloc[:, edge_idx].sum()
            features[f'edge_{edge_idx}_mean'] = edge_features.iloc[:, edge_idx].mean()
            features[f'edge_{edge_idx}_std'] = edge_features.iloc[:, edge_idx].std()
            features[f'edge_{edge_idx}_skew'] = edge_features.iloc[:, edge_idx].skew()
            features[f'edge_{edge_idx}_kurt'] = edge_features.iloc[:, edge_idx].kurt()

    else:

        features['node_count'] = np.nan
        features['node_sum'] = np.nan
        features['node_mean'] = np.nan
        features['node_std'] = np.nan
        features['node_skew'] = np.nan
        features['node_kurt'] = np.nan

        for node_idx in range(32):
            features[f'node_{node_idx}_sum'] = np.nan
            features[f'node_{node_idx}_mean'] = np.nan
            features[f'node_{node_idx}_std'] = np.nan
            features[f'node_{node_idx}_skew'] = np.nan
            features[f'node_{node_idx}_kurt'] = np.nan

        features['edge_count'] = np.nan
        features['edge_sum'] = np.nan
        features['edge_mean'] = np.nan
        features['edge_std'] = np.nan
        features['edge_skew'] = np.nan
        features['edge_kurt'] = np.nan

        for edge_idx in range(11):
            features[f'edge_{edge_idx}_sum'] = np.nan
            features[f'edge_{edge_idx}_mean'] = np.nan
            features[f'edge_{edge_idx}_std'] = np.nan
            features[f'edge_{edge_idx}_skew'] = np.nan
            features[f'edge_{edge_idx}_kurt'] = np.nan

    return features



def coerce_mordred_df(x):

    """
    Convert Mordred descriptor outputs to numeric values.

    Parameters
    ----------
    x: object
        A single descriptor value from Mordred.
        This may be a float, int, string representation of a number, None, or a Mordred error/missing object.

    Returns
    -------
    x: float
        A float value if coercion is successful; otherwise numpy.nan.
    """
    
    if x is None or isinstance(x, (Missing, Error)):
        return np.nan

    try:
        return float(x)
    except Exception:
        return np.nan



morgan_generator_raw = GetMorganGenerator(
    radius=4,
    countSimulation=False,
    includeChirality=False,
    useBondTypes=True,
    onlyNonzeroInvariants=True,
    includeRingMembership=True,
    fpSize=4096
)
morgan_generator_ecfp = GetMorganGenerator(
    radius=2,
    countSimulation=False,
    includeChirality=False,
    useBondTypes=True,
    onlyNonzeroInvariants=True,
    includeRingMembership=True,
    fpSize=4096,
    atomInvariantsGenerator=GetMorganAtomInvGen()
)
morgan_generator_fcfp = GetMorganGenerator(
    radius=2,
    countSimulation=False,
    includeChirality=False,
    useBondTypes=True,
    onlyNonzeroInvariants=True,
    includeRingMembership=True,
    fpSize=4096,
    atomInvariantsGenerator=GetMorganFeatureAtomInvGen()
)
morgan_generator_rdkit = GetMorganGenerator(
    radius=2,
    countSimulation=False,
    includeChirality=False,
    useBondTypes=True,
    onlyNonzeroInvariants=True,
    includeRingMembership=True,
    fpSize=4096,
    atomInvariantsGenerator=GetRDKitAtomInvGen()
)
atom_pair_generator_raw = GetAtomPairGenerator(
    minDistance=1,
    maxDistance=60,
    use2D=True,
    countSimulation=False,
    includeChirality=False,
    fpSize=4096
)
atom_pair_generator_ecfp = GetAtomPairGenerator(
    minDistance=1,
    maxDistance=60,
    use2D=True,
    countSimulation=False,
    includeChirality=False,
    fpSize=4096,
    atomInvariantsGenerator=GetMorganAtomInvGen()
)
atom_pair_generator_fcfp = GetAtomPairGenerator(
    minDistance=1,
    maxDistance=60,
    use2D=True,
    countSimulation=False,
    includeChirality=False,
    fpSize=4096,
    atomInvariantsGenerator=GetMorganFeatureAtomInvGen()
)
atom_pair_generator_rdkit = GetAtomPairGenerator(
    minDistance=1,
    maxDistance=60,
    use2D=True,
    countSimulation=False,
    includeChirality=False,
    fpSize=4096,
    atomInvariantsGenerator=GetRDKitAtomInvGen()
)
topological_torsion_generator_raw = GetTopologicalTorsionGenerator(
    torsionAtomCount=5,
    countSimulation=True,
    includeChirality=True,
    fpSize=4096
)
topological_torsion_generator_ecfp = GetTopologicalTorsionGenerator(
    torsionAtomCount=5,
    countSimulation=True,
    includeChirality=True,
    fpSize=4096,
    atomInvariantsGenerator=GetMorganAtomInvGen()
)
topological_torsion_generator_fcfp = GetTopologicalTorsionGenerator(
    torsionAtomCount=5,
    countSimulation=True,
    includeChirality=True,
    fpSize=4096,
    atomInvariantsGenerator=GetMorganFeatureAtomInvGen()
)
topological_torsion_generator_rdkit = GetTopologicalTorsionGenerator(
    torsionAtomCount=5,
    countSimulation=True,
    includeChirality=True,
    fpSize=4096,
    atomInvariantsGenerator=GetRDKitAtomInvGen()
)
rdkit_fingerprint_generator_raw = GetRDKitFPGenerator(
    minPath=1,
    maxPath=5,
    countSimulation=True,
    fpSize=4096
)
rdkit_fingerprint_generator_ecfp = GetRDKitFPGenerator(
    minPath=1,
    maxPath=5,
    countSimulation=True,
    fpSize=4096,
    atomInvariantsGenerator=GetMorganAtomInvGen()
)
rdkit_fingerprint_generator_fcfp = GetRDKitFPGenerator(
    minPath=1,
    maxPath=7,
    countSimulation=True,
    fpSize=4096,
    atomInvariantsGenerator=GetMorganFeatureAtomInvGen()
)

mol_graph_conv_featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True)

mols = []

rdkit_descriptors = []
estate_fingerprints = []
erg_fingerprints = []
maccs_keys = []
morgan_fingerprints_raw = []
morgan_fingerprints_ecfp = []
morgan_fingerprints_fcfp = []
morgan_fingerprints_rdkit = []
atom_pairs_raw = []
atom_pairs_ecfp = []
atom_pairs_fcfp = []
atom_pairs_rdkit = []
topological_torsions_raw = []
topological_torsions_ecfp = []
topological_torsions_fcfp = []
topological_torsions_rdkit = []
rdkit_fingerprints_raw = []
rdkit_fingerprints_ecfp = []
rdkit_fingerprints_fcfp = []
layered_fingerprints = []
pattern_fingerprints = []
avalon_fingerprints = []
invariants = []
brics_counts = []
scaffold_counts = []
mol_graph_conv_features = []

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

    mol = Chem.MolFromSmiles(row['SMILES'])
    mols.append(mol)
    
    rdkit_descriptors.append(compute_rdkit_descriptors(mol))
    estate_fingerprints.append(compute_estate_fingerprint(mol))
    erg_fingerprints.append(compute_erg_fingerprint(mol, atomTypes=0, fuzzIncrement=0.3, minPath=0, maxPath=10))
    maccs_keys.append(compute_maccs_keys(mol))
    morgan_fingerprints_raw.append(compute_morgan_fingerprint(mol, morgan_generator_raw))
    morgan_fingerprints_ecfp.append(compute_morgan_fingerprint(mol, morgan_generator_ecfp))
    morgan_fingerprints_fcfp.append(compute_morgan_fingerprint(mol, morgan_generator_fcfp))
    morgan_fingerprints_rdkit.append(compute_morgan_fingerprint(mol, morgan_generator_rdkit))
    atom_pairs_raw.append(compute_atom_pairs(mol, atom_pair_generator_raw))
    atom_pairs_ecfp.append(compute_atom_pairs(mol, atom_pair_generator_ecfp))
    atom_pairs_fcfp.append(compute_atom_pairs(mol, atom_pair_generator_fcfp))
    atom_pairs_rdkit.append(compute_atom_pairs(mol, atom_pair_generator_rdkit))
    topological_torsions_raw.append((compute_topological_torsion(mol, topological_torsion_generator_raw)))
    topological_torsions_ecfp.append((compute_topological_torsion(mol, topological_torsion_generator_ecfp)))
    topological_torsions_fcfp.append((compute_topological_torsion(mol, topological_torsion_generator_fcfp)))
    topological_torsions_rdkit.append((compute_topological_torsion(mol, topological_torsion_generator_rdkit)))
    rdkit_fingerprints_raw.append((compute_rdkit_fingerprint(mol, rdkit_fingerprint_generator_raw)))
    rdkit_fingerprints_ecfp.append((compute_rdkit_fingerprint(mol, rdkit_fingerprint_generator_ecfp)))
    rdkit_fingerprints_fcfp.append((compute_rdkit_fingerprint(mol, rdkit_fingerprint_generator_fcfp)))
    layered_fingerprints.append(compute_layered_fingerprint(mol))
    pattern_fingerprints.append(compute_pattern_fingerprint(mol))
    avalon_fingerprints.append(compute_avalon_fingerprint(mol))
    invariants.append(compute_invariants(mol))
    brics_counts.append(compute_brics_counts(mol))
    scaffold_counts.append(compute_scaffold_counts(mol))
    mol_graph_conv_features.append(compute_mol_graph_conv_features(row['SMILES'], mol_graph_conv_featurizer))

df_rdkit_descriptors = pd.DataFrame(rdkit_descriptors)
print(f'RDKit Descriptors Shape: {df_rdkit_descriptors.shape}')

df_estate_fingerprints = pd.DataFrame(
    np.stack(estate_fingerprints, axis=0),
    columns=[f'estate_count_{i}' for i in range(1, 80)] + [f'estate_sum_{i}' for i in range(1, 80)]
)
print(f'EState Fingerprints Shape: {df_estate_fingerprints.shape}')

erg_fingerprints = np.stack(erg_fingerprints, axis=0)
erg_fingerprints_dimensions = erg_fingerprints.shape[1]
df_erg_fingerprints = pd.DataFrame(
    erg_fingerprints,
    columns=[f'erg_{i}' for i in range(1, erg_fingerprints_dimensions + 1)]
)
print(f'ERG Fingerprints Shape: {df_erg_fingerprints.shape}')

df_maccs_keys = pd.DataFrame(
    np.stack(maccs_keys, axis=0),
    columns=[f'maccs_{i}' for i in range(1, 168)]
)
print(f'MACCS Keys Shape: {df_maccs_keys.shape}')

df_morgan_fingerprints_raw = pd.DataFrame(
    np.stack(morgan_fingerprints_raw, axis=0),
    columns=[f'morgan_raw_{i}' for i in range(1, 4096 + 1)]
)
print(f'Morgan Fingerprints Raw Shape: {df_morgan_fingerprints_raw.shape}')

df_morgan_fingerprints_ecfp = pd.DataFrame(
    np.stack(morgan_fingerprints_ecfp, axis=0),
    columns=[f'morgan_ecfp_{i}' for i in range(1, 4096 + 1)]
)
print(f'Morgan Fingerprints ECFP Shape: {df_morgan_fingerprints_ecfp.shape}')

df_morgan_fingerprints_fcfp = pd.DataFrame(
    np.stack(morgan_fingerprints_fcfp, axis=0),
    columns=[f'morgan_fcfp_{i}' for i in range(1, 4096 + 1)]
)
print(f'Morgan Fingerprints FCFP Shape: {df_morgan_fingerprints_fcfp.shape}')

df_morgan_fingerprints_rdkit = pd.DataFrame(
    np.stack(morgan_fingerprints_rdkit, axis=0),
    columns=[f'morgan_rdkit_{i}' for i in range(1, 4096 + 1)]
)
print(f'Morgan Fingerprints RDKit Shape: {df_morgan_fingerprints_rdkit.shape}')

df_atom_pairs_raw = pd.DataFrame(
    np.stack(atom_pairs_raw, axis=0),
    columns=[f'atom_pair_raw_{i}' for i in range(1, 4096 + 1)]
)
print(f'Atom Pairs Raw Shape: {df_atom_pairs_raw.shape}')

df_atom_pairs_ecfp = pd.DataFrame(
    np.stack(atom_pairs_ecfp, axis=0),
    columns=[f'atom_pair_ecfp_{i}' for i in range(1, 4096 + 1)]
)
print(f'Atom Pairs ECFP Shape: {df_atom_pairs_ecfp.shape}')

df_atom_pairs_fcfp = pd.DataFrame(
    np.stack(atom_pairs_fcfp, axis=0),
    columns=[f'atom_pair_fcfp_{i}' for i in range(1, 4096 + 1)]
)
print(f'Atom Pairs FCFP Shape: {df_atom_pairs_fcfp.shape}')

df_atom_pairs_rdkit = pd.DataFrame(
    np.stack(atom_pairs_rdkit, axis=0),
    columns=[f'atom_pair_rdkit_{i}' for i in range(1, 4096 + 1)]
)
print(f'Atom Pairs RDKit Shape: {df_atom_pairs_rdkit.shape}')

df_topological_torsions_raw = pd.DataFrame(
    np.stack(topological_torsions_raw, axis=0),
    columns=[f'topological_torsion_raw_{i}' for i in range(1, 4096 + 1)]
)
print(f'Topological Torsions Raw Shape: {df_topological_torsions_raw.shape}')

df_topological_torsions_ecfp = pd.DataFrame(
    np.stack(topological_torsions_ecfp, axis=0),
    columns=[f'topological_torsion_ecfp_{i}' for i in range(1, 4096 + 1)]
)
print(f'Topological Torsions ECFP Shape: {df_topological_torsions_ecfp.shape}')

df_topological_torsions_fcfp = pd.DataFrame(
    np.stack(topological_torsions_fcfp, axis=0),
    columns=[f'topological_torsion_fcfp_{i}' for i in range(1, 4096 + 1)]
)
print(f'Topological Torsions FCFP Shape: {df_topological_torsions_fcfp.shape}')

df_topological_torsions_rdkit = pd.DataFrame(
    np.stack(topological_torsions_rdkit, axis=0),
    columns=[f'topological_torsion_rdkit_{i}' for i in range(1, 4096 + 1)]
)
print(f'Topological Torsions RDKit Shape: {df_topological_torsions_rdkit.shape}')

df_rdkit_fingerprints_raw = pd.DataFrame(
    np.stack(rdkit_fingerprints_raw, axis=0),
    columns=[f'rdkit_fingerprint_raw_{i}' for i in range(1, 4096 + 1)]
)
print(f'RDKit Fingerprints Raw Shape: {df_rdkit_fingerprints_raw.shape}')

df_rdkit_fingerprints_ecfp = pd.DataFrame(
    np.stack(rdkit_fingerprints_ecfp, axis=0),
    columns=[f'rdkit_fingerprint_ecfp_{i}' for i in range(1, 4096 + 1)]
)
print(f'RDKit Fingerprints ECFP Shape: {df_rdkit_fingerprints_ecfp.shape}')

df_rdkit_fingerprints_fcfp = pd.DataFrame(
    np.stack(rdkit_fingerprints_fcfp, axis=0),
    columns=[f'rdkit_fingerprint_fcfp_{i}' for i in range(1, 4096 + 1)]
)
print(f'RDKit Fingerprints FCFP Shape: {df_rdkit_fingerprints_fcfp.shape}')

df_layered_fingerprints = pd.DataFrame(
    np.stack(layered_fingerprints, axis=0),
    columns=[f'layered_fingerprints_{i}' for i in range(1, 4096 + 1)]
)
print(f'Layered Fingerprints Shape: {df_layered_fingerprints.shape}')

df_pattern_fingerprints = pd.DataFrame(
    np.stack(pattern_fingerprints, axis=0),
    columns=[f'pattern_fingerprints_{i}' for i in range(1, 4096 + 1)]
)
print(f'Pattern Fingerprints Shape: {df_pattern_fingerprints.shape}')

df_avalon_fingerprints = pd.DataFrame(
    np.stack(avalon_fingerprints, axis=0),
    columns=[f'avalon_fingerprints_{i}' for i in range(1, 4096 + 1)]
)
print(f'Avalon Fingerprints Shape: {df_avalon_fingerprints.shape}')

df_invariants = pd.DataFrame(
    np.stack(invariants, axis=0),
    columns=[f'invariants_{i}' for i in range(1, 128 + 1)]
)
print(f'Invariants Shape: {df_invariants.shape}')

df_brics_counts = pd.DataFrame(
    np.stack(brics_counts, axis=0),
    columns=[f'brics_{i}' for i in range(1, 64 + 1)]
)
print(f'BRICS Counts Shape: {df_brics_counts.shape}')

df_scaffold_counts = pd.DataFrame(
    np.stack(scaffold_counts, axis=0),
    columns=[f'scaffold_{i}' for i in range(1, 64 + 1)]
)
print(f'Scaffold Counts Shape: {df_scaffold_counts.shape}')

df_mol_graph_conv_features = pd.DataFrame(mol_graph_conv_features)
print(f'Mol Graph Conv Features Shape: {df_mol_graph_conv_features.shape}')


mordred_calculator = Calculator(descriptors, ignore_3D=False)

df_mordred_descriptors = mordred_calculator.pandas(mols, nproc=1, ipynb=True)
df_mordred_descriptors = df_mordred_descriptors.map(coerce_mordred_df)
print(f'Mordred Descriptors Shape: {df_mordred_descriptors.shape}')


def concat_rdkit_descriptors(df, df_rdkit_descriptors, transformer_directory, scaler='standard_scaler'):

    df_rdkit_descriptors = df_rdkit_descriptors.drop(columns=[
        # Columns with large number of missing values
        'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI',
        'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW',
        'BCUT2D_MRHI', 'BCUT2D_MRLOW',
        'MaxPartialCharge', 'MinPartialCharge',
        'MaxAbsPartialCharge', 'MinAbsPartialCharge',

        # Columns with low variance
        'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur',
        'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide',
        'fr_isothiocyan', 'fr_lactam', 'fr_nitroso', 'fr_prisulfonamd',
        'fr_thiocyan',

        # Duplicate columns
        'fr_COO2', 'fr_Ar_NH', 'fr_amide', 'fr_diazo', 'fr_phos_ester'
    ])
    df_rdkit_descriptors = df_rdkit_descriptors.astype(np.float64)
    df_rdkit_descriptors = df_rdkit_descriptors.replace([np.inf, -np.inf], np.nan)

    column_missing_counts = df_rdkit_descriptors.isnull().sum()
    columns_with_missing_values = column_missing_counts.loc[column_missing_counts > 0].index
    for column in columns_with_missing_values:
        df_rdkit_descriptors[column] = df_rdkit_descriptors[column].fillna(df_rdkit_descriptors[column].median())

    log_transform_columns = ['Ipc']
    df_rdkit_descriptors.loc[:, log_transform_columns] = np.log1p(df_rdkit_descriptors.loc[:, log_transform_columns])

    with (open(transformer_directory / f'rdkit_descriptors_{scaler}.pickle', mode='rb') as f):
        rdkit_descriptors_scaler = pickle.load(f)

    df_rdkit_descriptors.loc[:, :] = rdkit_descriptors_scaler.transform(df_rdkit_descriptors)

    features = df_rdkit_descriptors.columns.tolist()
    df = pd.concat((
        df,
        df_rdkit_descriptors,
    ), axis=1, ignore_index=False)

    return df, features


def concat_estate_fingerprints(df, df_estate_fingerprints, transformer_directory, scaler='standard_scaler'):

    df_estate_fingerprints = df_estate_fingerprints.fillna(0.)

    estate_count_columns = [f'estate_count_{i}' for i in range(1, 80)]
    estate_sum_columns = [f'estate_sum_{i}' for i in range(1, 80)]
    estate_mean_columns = [f'estate_mean_{i}' for i in range(1, 80)]
    df_estate_fingerprints[estate_mean_columns] = pd.DataFrame(df_estate_fingerprints.loc[:, estate_sum_columns].values / df_estate_fingerprints.loc[:, estate_count_columns].values).fillna(0)
    df_estate_fingerprints.iloc[:, :79] = np.log1p(df_estate_fingerprints.iloc[:, :79])
    
    df_estate_fingerprints['estate_sum_sum'] = df_estate_fingerprints[estate_sum_columns].sum(axis=1)
    df_estate_fingerprints['estate_sum_mean'] = df_estate_fingerprints[estate_sum_columns].mean(axis=1)
    df_estate_fingerprints['estate_sum_median'] = df_estate_fingerprints[estate_sum_columns].median(axis=1)
    df_estate_fingerprints['estate_sum_std'] = df_estate_fingerprints[estate_sum_columns].std(axis=1)
    df_estate_fingerprints['estate_sum_max'] = df_estate_fingerprints[estate_sum_columns].max(axis=1)
    df_estate_fingerprints['estate_sum_skew'] = df_estate_fingerprints[estate_sum_columns].skew(axis=1)
    df_estate_fingerprints['estate_sum_kurt'] = df_estate_fingerprints[estate_sum_columns].kurt(axis=1)
    df_estate_fingerprints['estate_sum_argmin'] = np.argmin(df_estate_fingerprints[estate_sum_columns], axis=1).astype(np.float32)
    df_estate_fingerprints['estate_sum_argmax'] = np.argmax(df_estate_fingerprints[estate_sum_columns], axis=1).astype(np.float32)

    df_estate_fingerprints['estate_count_sum'] = df_estate_fingerprints[estate_count_columns].sum(axis=1)
    df_estate_fingerprints['estate_count_mean'] = df_estate_fingerprints[estate_count_columns].mean(axis=1)
    df_estate_fingerprints['estate_count_median'] = df_estate_fingerprints[estate_count_columns].median(axis=1)
    df_estate_fingerprints['estate_count_std'] = df_estate_fingerprints[estate_count_columns].std(axis=1)
    df_estate_fingerprints['estate_count_max'] = df_estate_fingerprints[estate_count_columns].max(axis=1)
    df_estate_fingerprints['estate_count_skew'] = df_estate_fingerprints[estate_count_columns].skew(axis=1)
    df_estate_fingerprints['estate_count_kurt'] = df_estate_fingerprints[estate_count_columns].kurt(axis=1)
    df_estate_fingerprints['estate_count_argmin'] = np.argmin(df_estate_fingerprints[estate_count_columns], axis=1).astype(np.float32)
    df_estate_fingerprints['estate_count_argmax'] = np.argmax(df_estate_fingerprints[estate_count_columns], axis=1).astype(np.float32)

    df_estate_fingerprints['estate_mean_sum'] = df_estate_fingerprints[estate_mean_columns].sum(axis=1)
    df_estate_fingerprints['estate_mean_mean'] = df_estate_fingerprints[estate_mean_columns].mean(axis=1)
    df_estate_fingerprints['estate_mean_median'] = df_estate_fingerprints[estate_mean_columns].median(axis=1)
    df_estate_fingerprints['estate_mean_std'] = df_estate_fingerprints[estate_mean_columns].std(axis=1)
    df_estate_fingerprints['estate_mean_max'] = df_estate_fingerprints[estate_mean_columns].max(axis=1)
    df_estate_fingerprints['estate_mean_skew'] = df_estate_fingerprints[estate_mean_columns].skew(axis=1)
    df_estate_fingerprints['estate_mean_kurt'] = df_estate_fingerprints[estate_mean_columns].kurt(axis=1)
    df_estate_fingerprints['estate_mean_argmin'] = np.argmin(df_estate_fingerprints[estate_mean_columns], axis=1).astype(np.float32)
    df_estate_fingerprints['estate_mean_argmax'] = np.argmax(df_estate_fingerprints[estate_mean_columns], axis=1).astype(np.float32)

    with open(transformer_directory / f'estate_fingerprints_{scaler}.pickle', mode='rb') as f:
        estate_fingerprints_standard_scaler = pickle.load(f)

    df_estate_fingerprints.loc[:, :] = estate_fingerprints_standard_scaler.transform(df_estate_fingerprints)

    features = df_estate_fingerprints.columns.tolist()
    df = pd.concat((
        df,
        df_estate_fingerprints,
    ), axis=1, ignore_index=False)

    return df, features


def concat_maccs_keys(df, df_maccs_keys, transformer_directory, scaler='standard_scaler'):

    df_maccs_keys = df_maccs_keys.fillna(0)
    maccs_keys_columns = df_maccs_keys.columns.tolist()
    maccs_keys_normalize_columns = [
        'maccs_keys_sum', 'maccs_keys_mean', 'maccs_keys_std',
        'maccs_keys_skew', 'maccs_keys_kurt',
    ]
    df_maccs_keys['maccs_keys_sum'] = df_maccs_keys[maccs_keys_columns].sum(axis=1).astype(np.float32)
    df_maccs_keys['maccs_keys_mean'] = df_maccs_keys[maccs_keys_columns].mean(axis=1).astype(np.float32)
    df_maccs_keys['maccs_keys_std'] = df_maccs_keys[maccs_keys_columns].std(axis=1).astype(np.float32)
    df_maccs_keys['maccs_keys_skew'] = df_maccs_keys[maccs_keys_columns].skew(axis=1).astype(np.float32)
    df_maccs_keys['maccs_keys_kurt'] = df_maccs_keys[maccs_keys_columns].kurt(axis=1).astype(np.float32)

    with open(transformer_directory / f'maccs_keys_{scaler}.pickle', mode='rb') as f:
        maccs_keys_standard_scaler = pickle.load(f)

    df_maccs_keys.loc[:, maccs_keys_normalize_columns] = maccs_keys_standard_scaler.transform(df_maccs_keys.loc[:, maccs_keys_normalize_columns])

    features = df_maccs_keys.columns.tolist()
    df = pd.concat((
        df,
        df_maccs_keys,
    ), axis=1, ignore_index=False)

    return df, features


def concat_erg_fingerprints(df, df_erg_fingerprints, transformer_directory, scaler='standard_scaler'):

    df_erg_fingerprints = df_erg_fingerprints.fillna(0.)
    erg_fingerprints_columns = df_erg_fingerprints.columns.tolist()
    df_erg_fingerprints['erg_sum'] = df_erg_fingerprints[erg_fingerprints_columns].sum(axis=1)
    df_erg_fingerprints['erg_mean'] = df_erg_fingerprints[erg_fingerprints_columns].mean(axis=1)
    df_erg_fingerprints['erg_median'] = df_erg_fingerprints[erg_fingerprints_columns].median(axis=1)
    df_erg_fingerprints['erg_std'] = df_erg_fingerprints[erg_fingerprints_columns].std(axis=1)
    df_erg_fingerprints['erg_max'] = df_erg_fingerprints[erg_fingerprints_columns].max(axis=1)
    df_erg_fingerprints['erg_skew'] = df_erg_fingerprints[erg_fingerprints_columns].skew(axis=1)
    df_erg_fingerprints['erg_kurt'] = df_erg_fingerprints[erg_fingerprints_columns].kurt(axis=1)
    df_erg_fingerprints['erg_argmin'] = np.argmin(df_erg_fingerprints, axis=1).astype(np.float32)
    df_erg_fingerprints['erg_argmax'] = np.argmax(df_erg_fingerprints, axis=1).astype(np.float32)

    with open(transformer_directory / f'erg_fingerprints_{scaler}.pickle', mode='rb') as f:
        erg_fingerprints_standard_scaler = pickle.load(f)

    df_erg_fingerprints.loc[:, :] = erg_fingerprints_standard_scaler.transform(df_erg_fingerprints)

    features = df_erg_fingerprints.columns.tolist()
    df = pd.concat((
        df,
        df_erg_fingerprints,
    ), axis=1, ignore_index=False)

    return df, features


def concat_morgan_fingerprints_raw(df, df_morgan_fingerprints_raw, transformer_directory, scaler='standard_scaler'):

    df_morgan_fingerprints_raw = df_morgan_fingerprints_raw.fillna(0)
    morgan_fingerprints_raw_columns = df_morgan_fingerprints_raw.columns.tolist()
    morgan_fingerprints_raw_normalize_columns = [
        'morgan_fingerprints_raw_sum', 'morgan_fingerprints_raw_mean', 'morgan_fingerprints_raw_median',
        'morgan_fingerprints_raw_std', 'morgan_fingerprints_raw_max',
        'morgan_fingerprints_raw_skew', 'morgan_fingerprints_raw_kurt',
    ]
    df_morgan_fingerprints_raw['morgan_fingerprints_raw_sum'] = df_morgan_fingerprints_raw[morgan_fingerprints_raw_columns].sum(axis=1).astype(np.float32)
    df_morgan_fingerprints_raw['morgan_fingerprints_raw_mean'] = df_morgan_fingerprints_raw[morgan_fingerprints_raw_columns].mean(axis=1).astype(np.float32)
    df_morgan_fingerprints_raw['morgan_fingerprints_raw_median'] = df_morgan_fingerprints_raw[morgan_fingerprints_raw_columns].median(axis=1).astype(np.float32)
    df_morgan_fingerprints_raw['morgan_fingerprints_raw_std'] = df_morgan_fingerprints_raw[morgan_fingerprints_raw_columns].std(axis=1).astype(np.float32)
    df_morgan_fingerprints_raw['morgan_fingerprints_raw_max'] = df_morgan_fingerprints_raw[morgan_fingerprints_raw_columns].max(axis=1).astype(np.float32)
    df_morgan_fingerprints_raw['morgan_fingerprints_raw_skew'] = df_morgan_fingerprints_raw[morgan_fingerprints_raw_columns].skew(axis=1).astype(np.float32)
    df_morgan_fingerprints_raw['morgan_fingerprints_raw_kurt'] = df_morgan_fingerprints_raw[morgan_fingerprints_raw_columns].kurt(axis=1).astype(np.float32)

    with open(transformer_directory / f'morgan_fingerprints_raw_{scaler}.pickle', mode='rb') as f:
        morgan_fingerprints_raw_standard_scaler = pickle.load(f)

    df_morgan_fingerprints_raw.loc[:, morgan_fingerprints_raw_normalize_columns] = morgan_fingerprints_raw_standard_scaler.transform(df_morgan_fingerprints_raw.loc[:, morgan_fingerprints_raw_normalize_columns])

    features = df_morgan_fingerprints_raw.columns.tolist()
    df = pd.concat((
        df,
        df_morgan_fingerprints_raw,
    ), axis=1, ignore_index=False)

    return df, features


def concat_morgan_fingerprints_ecfp(df, df_morgan_fingerprints_ecfp, transformer_directory, scaler='standard_scaler'):

    df_morgan_fingerprints_ecfp = df_morgan_fingerprints_ecfp.fillna(0)
    morgan_fingerprints_ecfp_columns = df_morgan_fingerprints_ecfp.columns.tolist()
    morgan_fingerprints_ecfp_normalize_columns = [
        'morgan_fingerprints_ecfp_sum', 'morgan_fingerprints_ecfp_mean', 'morgan_fingerprints_ecfp_median',
        'morgan_fingerprints_ecfp_std', 'morgan_fingerprints_ecfp_max',
        'morgan_fingerprints_ecfp_skew', 'morgan_fingerprints_ecfp_kurt',
    ]
    df_morgan_fingerprints_ecfp['morgan_fingerprints_ecfp_sum'] = df_morgan_fingerprints_ecfp[morgan_fingerprints_ecfp_columns].sum(axis=1).astype(np.float32)
    df_morgan_fingerprints_ecfp['morgan_fingerprints_ecfp_mean'] = df_morgan_fingerprints_ecfp[morgan_fingerprints_ecfp_columns].mean(axis=1).astype(np.float32)
    df_morgan_fingerprints_ecfp['morgan_fingerprints_ecfp_median'] = df_morgan_fingerprints_ecfp[morgan_fingerprints_ecfp_columns].median(axis=1).astype(np.float32)
    df_morgan_fingerprints_ecfp['morgan_fingerprints_ecfp_std'] = df_morgan_fingerprints_ecfp[morgan_fingerprints_ecfp_columns].std(axis=1).astype(np.float32)
    df_morgan_fingerprints_ecfp['morgan_fingerprints_ecfp_max'] = df_morgan_fingerprints_ecfp[morgan_fingerprints_ecfp_columns].max(axis=1).astype(np.float32)
    df_morgan_fingerprints_ecfp['morgan_fingerprints_ecfp_skew'] = df_morgan_fingerprints_ecfp[morgan_fingerprints_ecfp_columns].skew(axis=1).astype(np.float32)
    df_morgan_fingerprints_ecfp['morgan_fingerprints_ecfp_kurt'] = df_morgan_fingerprints_ecfp[morgan_fingerprints_ecfp_columns].kurt(axis=1).astype(np.float32)

    with open(transformer_directory / f'morgan_fingerprints_ecfp_{scaler}.pickle', mode='rb') as f:
        morgan_fingerprints_ecfp_standard_scaler = pickle.load(f)

    df_morgan_fingerprints_ecfp.loc[:, morgan_fingerprints_ecfp_normalize_columns] = morgan_fingerprints_ecfp_standard_scaler.transform(df_morgan_fingerprints_ecfp.loc[:, morgan_fingerprints_ecfp_normalize_columns])

    features = df_morgan_fingerprints_ecfp.columns.tolist()
    df = pd.concat((
        df,
        df_morgan_fingerprints_ecfp,
    ), axis=1, ignore_index=False)

    return df, features


def concat_morgan_fingerprints_fcfp(df, df_morgan_fingerprints_fcfp, transformer_directory, scaler='standard_scaler'):

    df_morgan_fingerprints_fcfp = df_morgan_fingerprints_fcfp.fillna(0)
    morgan_fingerprints_fcfp_columns = df_morgan_fingerprints_fcfp.columns.tolist()
    morgan_fingerprints_fcfp_normalize_columns = [
        'morgan_fingerprints_fcfp_sum', 'morgan_fingerprints_fcfp_mean', 'morgan_fingerprints_fcfp_median',
        'morgan_fingerprints_fcfp_std', 'morgan_fingerprints_fcfp_max',
        'morgan_fingerprints_fcfp_skew', 'morgan_fingerprints_fcfp_kurt',
    ]
    df_morgan_fingerprints_fcfp['morgan_fingerprints_fcfp_sum'] = df_morgan_fingerprints_fcfp[morgan_fingerprints_fcfp_columns].sum(axis=1).astype(np.float32)
    df_morgan_fingerprints_fcfp['morgan_fingerprints_fcfp_mean'] = df_morgan_fingerprints_fcfp[morgan_fingerprints_fcfp_columns].mean(axis=1).astype(np.float32)
    df_morgan_fingerprints_fcfp['morgan_fingerprints_fcfp_median'] = df_morgan_fingerprints_fcfp[morgan_fingerprints_fcfp_columns].median(axis=1).astype(np.float32)
    df_morgan_fingerprints_fcfp['morgan_fingerprints_fcfp_std'] = df_morgan_fingerprints_fcfp[morgan_fingerprints_fcfp_columns].std(axis=1).astype(np.float32)
    df_morgan_fingerprints_fcfp['morgan_fingerprints_fcfp_max'] = df_morgan_fingerprints_fcfp[morgan_fingerprints_fcfp_columns].max(axis=1).astype(np.float32)
    df_morgan_fingerprints_fcfp['morgan_fingerprints_fcfp_skew'] = df_morgan_fingerprints_fcfp[morgan_fingerprints_fcfp_columns].skew(axis=1).astype(np.float32)
    df_morgan_fingerprints_fcfp['morgan_fingerprints_fcfp_kurt'] = df_morgan_fingerprints_fcfp[morgan_fingerprints_fcfp_columns].kurt(axis=1).astype(np.float32)

    with open(transformer_directory / f'morgan_fingerprints_fcfp_{scaler}.pickle', mode='rb') as f:
        morgan_fingerprints_fcfp_standard_scaler = pickle.load(f)

    df_morgan_fingerprints_fcfp.loc[:, morgan_fingerprints_fcfp_normalize_columns] = morgan_fingerprints_fcfp_standard_scaler.transform(df_morgan_fingerprints_fcfp.loc[:, morgan_fingerprints_fcfp_normalize_columns])

    features = df_morgan_fingerprints_fcfp.columns.tolist()
    df = pd.concat((
        df,
        df_morgan_fingerprints_fcfp,
    ), axis=1, ignore_index=False)

    return df, features


def concat_morgan_fingerprints_rdkit(df, df_morgan_fingerprints_rdkit, transformer_directory, scaler='standard_scaler'):

    df_morgan_fingerprints_rdkit = df_morgan_fingerprints_rdkit.fillna(0)
    morgan_fingerprints_rdkit_columns = df_morgan_fingerprints_rdkit.columns.tolist()
    morgan_fingerprints_rdkit_normalize_columns = [
        'morgan_fingerprints_rdkit_sum', 'morgan_fingerprints_rdkit_mean', 'morgan_fingerprints_rdkit_median',
        'morgan_fingerprints_rdkit_std', 'morgan_fingerprints_rdkit_max',
        'morgan_fingerprints_rdkit_skew', 'morgan_fingerprints_rdkit_kurt',
    ]
    df_morgan_fingerprints_rdkit['morgan_fingerprints_rdkit_sum'] = df_morgan_fingerprints_rdkit[morgan_fingerprints_rdkit_columns].sum(axis=1).astype(np.float32)
    df_morgan_fingerprints_rdkit['morgan_fingerprints_rdkit_mean'] = df_morgan_fingerprints_rdkit[morgan_fingerprints_rdkit_columns].mean(axis=1).astype(np.float32)
    df_morgan_fingerprints_rdkit['morgan_fingerprints_rdkit_median'] = df_morgan_fingerprints_rdkit[morgan_fingerprints_rdkit_columns].median(axis=1).astype(np.float32)
    df_morgan_fingerprints_rdkit['morgan_fingerprints_rdkit_std'] = df_morgan_fingerprints_rdkit[morgan_fingerprints_rdkit_columns].std(axis=1).astype(np.float32)
    df_morgan_fingerprints_rdkit['morgan_fingerprints_rdkit_max'] = df_morgan_fingerprints_rdkit[morgan_fingerprints_rdkit_columns].max(axis=1).astype(np.float32)
    df_morgan_fingerprints_rdkit['morgan_fingerprints_rdkit_skew'] = df_morgan_fingerprints_rdkit[morgan_fingerprints_rdkit_columns].skew(axis=1).astype(np.float32)
    df_morgan_fingerprints_rdkit['morgan_fingerprints_rdkit_kurt'] = df_morgan_fingerprints_rdkit[morgan_fingerprints_rdkit_columns].kurt(axis=1).astype(np.float32)

    with open(transformer_directory / f'morgan_fingerprints_rdkit_{scaler}.pickle', mode='rb') as f:
        morgan_fingerprints_rdkit_standard_scaler = pickle.load(f)

    df_morgan_fingerprints_rdkit.loc[:, morgan_fingerprints_rdkit_normalize_columns] = morgan_fingerprints_rdkit_standard_scaler.transform(df_morgan_fingerprints_rdkit.loc[:, morgan_fingerprints_rdkit_normalize_columns])

    features = df_morgan_fingerprints_rdkit.columns.tolist()
    df = pd.concat((
        df,
        df_morgan_fingerprints_rdkit,
    ), axis=1, ignore_index=False)

    return df, features


def concat_atom_pairs_raw(df, df_atom_pairs_raw, transformer_directory, scaler='standard_scaler'):

    df_atom_pairs_raw = df_atom_pairs_raw.fillna(0)

    atom_pairs_raw_columns = df_atom_pairs_raw.columns.tolist()
    atom_pairs_raw_normalize_columns = [
        'atom_pairs_raw_sum', 'atom_pairs_raw_mean', 'atom_pairs_raw_median',
        'atom_pairs_raw_std', 'atom_pairs_raw_max', 'atom_pairs_raw_skew', 'atom_pairs_raw_kurt',
    ]
    df_atom_pairs_raw['atom_pairs_raw_sum'] = df_atom_pairs_raw[atom_pairs_raw_columns].sum(axis=1).astype(np.float32)
    df_atom_pairs_raw['atom_pairs_raw_mean'] = df_atom_pairs_raw[atom_pairs_raw_columns].mean(axis=1).astype(np.float32)
    df_atom_pairs_raw['atom_pairs_raw_median'] = df_atom_pairs_raw[atom_pairs_raw_columns].median(axis=1).astype(np.float32)
    df_atom_pairs_raw['atom_pairs_raw_std'] = df_atom_pairs_raw[atom_pairs_raw_columns].std(axis=1).astype(np.float32)
    df_atom_pairs_raw['atom_pairs_raw_max'] = df_atom_pairs_raw[atom_pairs_raw_columns].max(axis=1).astype(np.float32)
    df_atom_pairs_raw['atom_pairs_raw_skew'] = df_atom_pairs_raw[atom_pairs_raw_columns].skew(axis=1).astype(np.float32)
    df_atom_pairs_raw['atom_pairs_raw_kurt'] = df_atom_pairs_raw[atom_pairs_raw_columns].kurt(axis=1).astype(np.float32)
    df_atom_pairs_raw.loc[:, atom_pairs_raw_columns] = (df_atom_pairs_raw.loc[:, atom_pairs_raw_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'atom_pairs_raw_{scaler}.pickle', mode='rb') as f:
        atom_pairs_raw_standard_scaler = pickle.load(f)

    df_atom_pairs_raw.loc[:, atom_pairs_raw_normalize_columns] = atom_pairs_raw_standard_scaler.transform(df_atom_pairs_raw.loc[:, atom_pairs_raw_normalize_columns])

    features = df_atom_pairs_raw.columns.tolist()
    df = pd.concat((
        df,
        df_atom_pairs_raw,
    ), axis=1, ignore_index=False)

    return df, features


def concat_atom_pairs_ecfp(df, df_atom_pairs_ecfp, transformer_directory, scaler='standard_scaler'):

    df_atom_pairs_ecfp = df_atom_pairs_ecfp.fillna(0)

    atom_pairs_ecfp_columns = df_atom_pairs_ecfp.columns.tolist()
    atom_pairs_ecfp_normalize_columns = [
        'atom_pairs_ecfp_sum', 'atom_pairs_ecfp_mean', 'atom_pairs_ecfp_median',
        'atom_pairs_ecfp_std', 'atom_pairs_ecfp_max', 'atom_pairs_ecfp_skew', 'atom_pairs_ecfp_kurt',
    ]
    df_atom_pairs_ecfp['atom_pairs_ecfp_sum'] = df_atom_pairs_ecfp[atom_pairs_ecfp_columns].sum(axis=1).astype(np.float32)
    df_atom_pairs_ecfp['atom_pairs_ecfp_mean'] = df_atom_pairs_ecfp[atom_pairs_ecfp_columns].mean(axis=1).astype(np.float32)
    df_atom_pairs_ecfp['atom_pairs_ecfp_median'] = df_atom_pairs_ecfp[atom_pairs_ecfp_columns].median(axis=1).astype(np.float32)
    df_atom_pairs_ecfp['atom_pairs_ecfp_std'] = df_atom_pairs_ecfp[atom_pairs_ecfp_columns].std(axis=1).astype(np.float32)
    df_atom_pairs_ecfp['atom_pairs_ecfp_max'] = df_atom_pairs_ecfp[atom_pairs_ecfp_columns].max(axis=1).astype(np.float32)
    df_atom_pairs_ecfp['atom_pairs_ecfp_skew'] = df_atom_pairs_ecfp[atom_pairs_ecfp_columns].skew(axis=1).astype(np.float32)
    df_atom_pairs_ecfp['atom_pairs_ecfp_kurt'] = df_atom_pairs_ecfp[atom_pairs_ecfp_columns].kurt(axis=1).astype(np.float32)
    df_atom_pairs_ecfp.loc[:, atom_pairs_ecfp_columns] = (df_atom_pairs_ecfp.loc[:, atom_pairs_ecfp_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'atom_pairs_ecfp_{scaler}.pickle', mode='rb') as f:
        atom_pairs_ecfp_standard_scaler = pickle.load(f)

    df_atom_pairs_ecfp.loc[:, atom_pairs_ecfp_normalize_columns] = atom_pairs_ecfp_standard_scaler.transform(df_atom_pairs_ecfp.loc[:, atom_pairs_ecfp_normalize_columns])

    features = df_atom_pairs_ecfp.columns.tolist()
    df = pd.concat((
        df,
        df_atom_pairs_ecfp,
    ), axis=1, ignore_index=False)

    return df, features


def concat_atom_pairs_fcfp(df, df_atom_pairs_fcfp, transformer_directory, scaler='standard_scaler'):

    df_atom_pairs_fcfp = df_atom_pairs_fcfp.fillna(0)

    atom_pairs_fcfp_columns = df_atom_pairs_fcfp.columns.tolist()
    atom_pairs_fcfp_normalize_columns = [
        'atom_pairs_fcfp_sum', 'atom_pairs_fcfp_mean', 'atom_pairs_fcfp_median',
        'atom_pairs_fcfp_std', 'atom_pairs_fcfp_max', 'atom_pairs_fcfp_skew', 'atom_pairs_fcfp_kurt',
    ]
    df_atom_pairs_fcfp['atom_pairs_fcfp_sum'] = df_atom_pairs_fcfp[atom_pairs_fcfp_columns].sum(axis=1).astype(np.float32)
    df_atom_pairs_fcfp['atom_pairs_fcfp_mean'] = df_atom_pairs_fcfp[atom_pairs_fcfp_columns].mean(axis=1).astype(np.float32)
    df_atom_pairs_fcfp['atom_pairs_fcfp_median'] = df_atom_pairs_fcfp[atom_pairs_fcfp_columns].median(axis=1).astype(np.float32)
    df_atom_pairs_fcfp['atom_pairs_fcfp_std'] = df_atom_pairs_fcfp[atom_pairs_fcfp_columns].std(axis=1).astype(np.float32)
    df_atom_pairs_fcfp['atom_pairs_fcfp_max'] = df_atom_pairs_fcfp[atom_pairs_fcfp_columns].max(axis=1).astype(np.float32)
    df_atom_pairs_fcfp['atom_pairs_fcfp_skew'] = df_atom_pairs_fcfp[atom_pairs_fcfp_columns].skew(axis=1).astype(np.float32)
    df_atom_pairs_fcfp['atom_pairs_fcfp_kurt'] = df_atom_pairs_fcfp[atom_pairs_fcfp_columns].kurt(axis=1).astype(np.float32)
    df_atom_pairs_fcfp.loc[:, atom_pairs_fcfp_columns] = (df_atom_pairs_fcfp.loc[:, atom_pairs_fcfp_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'atom_pairs_fcfp_{scaler}.pickle', mode='rb') as f:
        atom_pairs_fcfp_standard_scaler = pickle.load(f)

    df_atom_pairs_fcfp.loc[:, atom_pairs_fcfp_normalize_columns] = atom_pairs_fcfp_standard_scaler.transform(df_atom_pairs_fcfp.loc[:, atom_pairs_fcfp_normalize_columns])

    features = df_atom_pairs_fcfp.columns.tolist()
    df = pd.concat((
        df,
        df_atom_pairs_fcfp,
    ), axis=1, ignore_index=False)

    return df, features


def concat_atom_pairs_rdkit(df, df_atom_pairs_rdkit, transformer_directory, scaler='standard_scaler'):

    df_atom_pairs_rdkit = df_atom_pairs_rdkit.fillna(0)

    atom_pairs_rdkit_columns = df_atom_pairs_rdkit.columns.tolist()
    atom_pairs_rdkit_normalize_columns = [
        'atom_pairs_rdkit_sum', 'atom_pairs_rdkit_mean', 'atom_pairs_rdkit_median',
        'atom_pairs_rdkit_std', 'atom_pairs_rdkit_max', 'atom_pairs_rdkit_skew', 'atom_pairs_rdkit_kurt',
    ]
    df_atom_pairs_rdkit['atom_pairs_rdkit_sum'] = df_atom_pairs_rdkit[atom_pairs_rdkit_columns].sum(axis=1).astype(np.float32)
    df_atom_pairs_rdkit['atom_pairs_rdkit_mean'] = df_atom_pairs_rdkit[atom_pairs_rdkit_columns].mean(axis=1).astype(np.float32)
    df_atom_pairs_rdkit['atom_pairs_rdkit_median'] = df_atom_pairs_rdkit[atom_pairs_rdkit_columns].median(axis=1).astype(np.float32)
    df_atom_pairs_rdkit['atom_pairs_rdkit_std'] = df_atom_pairs_rdkit[atom_pairs_rdkit_columns].std(axis=1).astype(np.float32)
    df_atom_pairs_rdkit['atom_pairs_rdkit_max'] = df_atom_pairs_rdkit[atom_pairs_rdkit_columns].max(axis=1).astype(np.float32)
    df_atom_pairs_rdkit['atom_pairs_rdkit_skew'] = df_atom_pairs_rdkit[atom_pairs_rdkit_columns].skew(axis=1).astype(np.float32)
    df_atom_pairs_rdkit['atom_pairs_rdkit_kurt'] = df_atom_pairs_rdkit[atom_pairs_rdkit_columns].kurt(axis=1).astype(np.float32)
    df_atom_pairs_rdkit.loc[:, atom_pairs_rdkit_columns] = (df_atom_pairs_rdkit.loc[:, atom_pairs_rdkit_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'atom_pairs_rdkit_{scaler}.pickle', mode='rb') as f:
        atom_pairs_rdkit_standard_scaler = pickle.load(f)

    df_atom_pairs_rdkit.loc[:, atom_pairs_rdkit_normalize_columns] = atom_pairs_rdkit_standard_scaler.transform(df_atom_pairs_rdkit.loc[:, atom_pairs_rdkit_normalize_columns])

    features = df_atom_pairs_rdkit.columns.tolist()
    df = pd.concat((
        df,
        df_atom_pairs_rdkit,
    ), axis=1, ignore_index=False)

    return df, features


def concat_topological_torsions_raw(df, df_topological_torsions_raw, transformer_directory, scaler='standard_scaler'):

    df_topological_torsions_raw = df_topological_torsions_raw.fillna(0)
    topological_torsions_raw_columns = df_topological_torsions_raw.columns.tolist()
    topological_torsions_raw_normalize_columns = [
        'topological_torsions_raw_sum', 'topological_torsions_raw_mean', 'topological_torsions_raw_median',
        'topological_torsions_raw_std', 'topological_torsions_raw_max', 'topological_torsions_raw_skew', 'topological_torsions_raw_kurt',
    ]
    df_topological_torsions_raw['topological_torsions_raw_sum'] = df_topological_torsions_raw[topological_torsions_raw_columns].sum(axis=1).astype(np.float32)
    df_topological_torsions_raw['topological_torsions_raw_mean'] = df_topological_torsions_raw[topological_torsions_raw_columns].mean(axis=1).astype(np.float32)
    df_topological_torsions_raw['topological_torsions_raw_median'] = df_topological_torsions_raw[topological_torsions_raw_columns].median(axis=1).astype(np.float32)
    df_topological_torsions_raw['topological_torsions_raw_std'] = df_topological_torsions_raw[topological_torsions_raw_columns].std(axis=1).astype(np.float32)
    df_topological_torsions_raw['topological_torsions_raw_max'] = df_topological_torsions_raw[topological_torsions_raw_columns].max(axis=1).astype(np.float32)
    df_topological_torsions_raw['topological_torsions_raw_skew'] = df_topological_torsions_raw[topological_torsions_raw_columns].skew(axis=1).astype(np.float32)
    df_topological_torsions_raw['topological_torsions_raw_kurt'] = df_topological_torsions_raw[topological_torsions_raw_columns].kurt(axis=1).astype(np.float32)
    df_topological_torsions_raw.loc[:, topological_torsions_raw_columns] = (df_topological_torsions_raw.loc[:, topological_torsions_raw_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'topological_torsions_raw_{scaler}.pickle', mode='rb') as f:
        topological_torsions_raw_standard_scaler = pickle.load(f)

    df_topological_torsions_raw.loc[:, topological_torsions_raw_normalize_columns] = topological_torsions_raw_standard_scaler.transform(df_topological_torsions_raw.loc[:, topological_torsions_raw_normalize_columns])

    features = df_topological_torsions_raw.columns.tolist()
    df = pd.concat((
        df,
        df_topological_torsions_raw,
    ), axis=1, ignore_index=False)

    return df, features


def concat_topological_torsions_ecfp(df, df_topological_torsions_ecfp, transformer_directory, scaler='standard_scaler'):

    df_topological_torsions_ecfp = df_topological_torsions_ecfp.fillna(0)
    topological_torsions_ecfp_columns = df_topological_torsions_ecfp.columns.tolist()
    topological_torsions_ecfp_normalize_columns = [
        'topological_torsions_ecfp_sum', 'topological_torsions_ecfp_mean', 'topological_torsions_ecfp_median',
        'topological_torsions_ecfp_std', 'topological_torsions_ecfp_max', 'topological_torsions_ecfp_skew', 'topological_torsions_ecfp_kurt',
    ]
    df_topological_torsions_ecfp['topological_torsions_ecfp_sum'] = df_topological_torsions_ecfp[topological_torsions_ecfp_columns].sum(axis=1).astype(np.float32)
    df_topological_torsions_ecfp['topological_torsions_ecfp_mean'] = df_topological_torsions_ecfp[topological_torsions_ecfp_columns].mean(axis=1).astype(np.float32)
    df_topological_torsions_ecfp['topological_torsions_ecfp_median'] = df_topological_torsions_ecfp[topological_torsions_ecfp_columns].median(axis=1).astype(np.float32)
    df_topological_torsions_ecfp['topological_torsions_ecfp_std'] = df_topological_torsions_ecfp[topological_torsions_ecfp_columns].std(axis=1).astype(np.float32)
    df_topological_torsions_ecfp['topological_torsions_ecfp_max'] = df_topological_torsions_ecfp[topological_torsions_ecfp_columns].max(axis=1).astype(np.float32)
    df_topological_torsions_ecfp['topological_torsions_ecfp_skew'] = df_topological_torsions_ecfp[topological_torsions_ecfp_columns].skew(axis=1).astype(np.float32)
    df_topological_torsions_ecfp['topological_torsions_ecfp_kurt'] = df_topological_torsions_ecfp[topological_torsions_ecfp_columns].kurt(axis=1).astype(np.float32)
    df_topological_torsions_ecfp.loc[:, topological_torsions_ecfp_columns] = (df_topological_torsions_ecfp.loc[:, topological_torsions_ecfp_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'topological_torsions_ecfp_{scaler}.pickle', mode='rb') as f:
        topological_torsions_ecfp_standard_scaler = pickle.load(f)

    df_topological_torsions_ecfp.loc[:, topological_torsions_ecfp_normalize_columns] = topological_torsions_ecfp_standard_scaler.transform(df_topological_torsions_ecfp.loc[:, topological_torsions_ecfp_normalize_columns])

    features = df_topological_torsions_ecfp.columns.tolist()
    df = pd.concat((
        df,
        df_topological_torsions_ecfp,
    ), axis=1, ignore_index=False)

    return df, features


def concat_topological_torsions_fcfp(df, df_topological_torsions_fcfp, transformer_directory, scaler='standard_scaler'):

    df_topological_torsions_fcfp = df_topological_torsions_fcfp.fillna(0)
    topological_torsions_fcfp_columns = df_topological_torsions_fcfp.columns.tolist()
    topological_torsions_fcfp_normalize_columns = [
        'topological_torsions_fcfp_sum', 'topological_torsions_fcfp_mean', 'topological_torsions_fcfp_median',
        'topological_torsions_fcfp_std', 'topological_torsions_fcfp_max', 'topological_torsions_fcfp_skew', 'topological_torsions_fcfp_kurt',
    ]
    df_topological_torsions_fcfp['topological_torsions_fcfp_sum'] = df_topological_torsions_fcfp[topological_torsions_fcfp_columns].sum(axis=1).astype(np.float32)
    df_topological_torsions_fcfp['topological_torsions_fcfp_mean'] = df_topological_torsions_fcfp[topological_torsions_fcfp_columns].mean(axis=1).astype(np.float32)
    df_topological_torsions_fcfp['topological_torsions_fcfp_median'] = df_topological_torsions_fcfp[topological_torsions_fcfp_columns].median(axis=1).astype(np.float32)
    df_topological_torsions_fcfp['topological_torsions_fcfp_std'] = df_topological_torsions_fcfp[topological_torsions_fcfp_columns].std(axis=1).astype(np.float32)
    df_topological_torsions_fcfp['topological_torsions_fcfp_max'] = df_topological_torsions_fcfp[topological_torsions_fcfp_columns].max(axis=1).astype(np.float32)
    df_topological_torsions_fcfp['topological_torsions_fcfp_skew'] = df_topological_torsions_fcfp[topological_torsions_fcfp_columns].skew(axis=1).astype(np.float32)
    df_topological_torsions_fcfp['topological_torsions_fcfp_kurt'] = df_topological_torsions_fcfp[topological_torsions_fcfp_columns].kurt(axis=1).astype(np.float32)
    df_topological_torsions_fcfp.loc[:, topological_torsions_fcfp_columns] = (df_topological_torsions_fcfp.loc[:, topological_torsions_fcfp_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'topological_torsions_fcfp_{scaler}.pickle', mode='rb') as f:
        topological_torsions_fcfp_standard_scaler = pickle.load(f)

    df_topological_torsions_fcfp.loc[:, topological_torsions_fcfp_normalize_columns] = topological_torsions_fcfp_standard_scaler.transform(df_topological_torsions_fcfp.loc[:, topological_torsions_fcfp_normalize_columns])

    features = df_topological_torsions_fcfp.columns.tolist()
    df = pd.concat((
        df,
        df_topological_torsions_fcfp,
    ), axis=1, ignore_index=False)

    return df, features


def concat_topological_torsions_rdkit(df, df_topological_torsions_rdkit, transformer_directory, scaler='standard_scaler'):

    df_topological_torsions_rdkit = df_topological_torsions_rdkit.fillna(0)
    topological_torsions_rdkit_columns = df_topological_torsions_rdkit.columns.tolist()
    topological_torsions_rdkit_normalize_columns = [
        'topological_torsions_rdkit_sum', 'topological_torsions_rdkit_mean', 'topological_torsions_rdkit_median',
        'topological_torsions_rdkit_std', 'topological_torsions_rdkit_max', 'topological_torsions_rdkit_skew', 'topological_torsions_rdkit_kurt',
    ]
    df_topological_torsions_rdkit['topological_torsions_rdkit_sum'] = df_topological_torsions_rdkit[topological_torsions_rdkit_columns].sum(axis=1).astype(np.float32)
    df_topological_torsions_rdkit['topological_torsions_rdkit_mean'] = df_topological_torsions_rdkit[topological_torsions_rdkit_columns].mean(axis=1).astype(np.float32)
    df_topological_torsions_rdkit['topological_torsions_rdkit_median'] = df_topological_torsions_rdkit[topological_torsions_rdkit_columns].median(axis=1).astype(np.float32)
    df_topological_torsions_rdkit['topological_torsions_rdkit_std'] = df_topological_torsions_rdkit[topological_torsions_rdkit_columns].std(axis=1).astype(np.float32)
    df_topological_torsions_rdkit['topological_torsions_rdkit_max'] = df_topological_torsions_rdkit[topological_torsions_rdkit_columns].max(axis=1).astype(np.float32)
    df_topological_torsions_rdkit['topological_torsions_rdkit_skew'] = df_topological_torsions_rdkit[topological_torsions_rdkit_columns].skew(axis=1).astype(np.float32)
    df_topological_torsions_rdkit['topological_torsions_rdkit_kurt'] = df_topological_torsions_rdkit[topological_torsions_rdkit_columns].kurt(axis=1).astype(np.float32)
    df_topological_torsions_rdkit.loc[:, topological_torsions_rdkit_columns] = (df_topological_torsions_rdkit.loc[:, topological_torsions_rdkit_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'topological_torsions_rdkit_{scaler}.pickle', mode='rb') as f:
        topological_torsions_rdkit_standard_scaler = pickle.load(f)

    df_topological_torsions_rdkit.loc[:, topological_torsions_rdkit_normalize_columns] = topological_torsions_rdkit_standard_scaler.transform(df_topological_torsions_rdkit.loc[:, topological_torsions_rdkit_normalize_columns])

    features = df_topological_torsions_rdkit.columns.tolist()
    df = pd.concat((
        df,
        df_topological_torsions_rdkit,
    ), axis=1, ignore_index=False)

    return df, features


def concat_rdkit_fingerprints_raw(df, df_rdkit_fingerprints_raw, transformer_directory, scaler='standard_scaler'):

    df_rdkit_fingerprints_raw = df_rdkit_fingerprints_raw.fillna(0)
    rdkit_fingerprints_raw_columns = df_rdkit_fingerprints_raw.columns.tolist()
    rdkit_fingerprints_raw_normalize_columns = [
        'rdkit_fingerprints_raw_sum', 'rdkit_fingerprints_raw_mean', 'rdkit_fingerprints_raw_median',
        'rdkit_fingerprints_raw_std', 'rdkit_fingerprints_raw_max', 'rdkit_fingerprints_raw_skew', 'rdkit_fingerprints_raw_kurt',
    ]
    df_rdkit_fingerprints_raw['rdkit_fingerprints_raw_sum'] = df_rdkit_fingerprints_raw[rdkit_fingerprints_raw_columns].sum(axis=1).astype(np.float32)
    df_rdkit_fingerprints_raw['rdkit_fingerprints_raw_mean'] = df_rdkit_fingerprints_raw[rdkit_fingerprints_raw_columns].mean(axis=1).astype(np.float32)
    df_rdkit_fingerprints_raw['rdkit_fingerprints_raw_median'] = df_rdkit_fingerprints_raw[rdkit_fingerprints_raw_columns].median(axis=1).astype(np.float32)
    df_rdkit_fingerprints_raw['rdkit_fingerprints_raw_std'] = df_rdkit_fingerprints_raw[rdkit_fingerprints_raw_columns].std(axis=1).astype(np.float32)
    df_rdkit_fingerprints_raw['rdkit_fingerprints_raw_max'] = df_rdkit_fingerprints_raw[rdkit_fingerprints_raw_columns].max(axis=1).astype(np.float32)
    df_rdkit_fingerprints_raw['rdkit_fingerprints_raw_skew'] = df_rdkit_fingerprints_raw[rdkit_fingerprints_raw_columns].skew(axis=1).astype(np.float32)
    df_rdkit_fingerprints_raw['rdkit_fingerprints_raw_kurt'] = df_rdkit_fingerprints_raw[rdkit_fingerprints_raw_columns].kurt(axis=1).astype(np.float32)
    df_rdkit_fingerprints_raw.loc[:, rdkit_fingerprints_raw_columns] = (df_rdkit_fingerprints_raw.loc[:, rdkit_fingerprints_raw_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'rdkit_fingerprints_raw_{scaler}.pickle', mode='rb') as f:
        rdkit_fingerprints_raw_standard_scaler = pickle.load(f)

    df_rdkit_fingerprints_raw.loc[:, rdkit_fingerprints_raw_normalize_columns] = rdkit_fingerprints_raw_standard_scaler.transform(df_rdkit_fingerprints_raw.loc[:, rdkit_fingerprints_raw_normalize_columns])

    features = df_rdkit_fingerprints_raw.columns.tolist()
    df = pd.concat((
        df,
        df_rdkit_fingerprints_raw,
    ), axis=1, ignore_index=False)

    return df, features


def concat_rdkit_fingerprints_ecfp(df, df_rdkit_fingerprints_ecfp, transformer_directory, scaler='standard_scaler'):

    df_rdkit_fingerprints_ecfp = df_rdkit_fingerprints_ecfp.fillna(0)
    rdkit_fingerprints_ecfp_columns = df_rdkit_fingerprints_ecfp.columns.tolist()
    rdkit_fingerprints_ecfp_normalize_columns = [
        'rdkit_fingerprints_ecfp_sum', 'rdkit_fingerprints_ecfp_mean', 'rdkit_fingerprints_ecfp_median',
        'rdkit_fingerprints_ecfp_std', 'rdkit_fingerprints_ecfp_max', 'rdkit_fingerprints_ecfp_skew', 'rdkit_fingerprints_ecfp_kurt',
    ]
    df_rdkit_fingerprints_ecfp['rdkit_fingerprints_ecfp_sum'] = df_rdkit_fingerprints_ecfp[rdkit_fingerprints_ecfp_columns].sum(axis=1).astype(np.float32)
    df_rdkit_fingerprints_ecfp['rdkit_fingerprints_ecfp_mean'] = df_rdkit_fingerprints_ecfp[rdkit_fingerprints_ecfp_columns].mean(axis=1).astype(np.float32)
    df_rdkit_fingerprints_ecfp['rdkit_fingerprints_ecfp_median'] = df_rdkit_fingerprints_ecfp[rdkit_fingerprints_ecfp_columns].median(axis=1).astype(np.float32)
    df_rdkit_fingerprints_ecfp['rdkit_fingerprints_ecfp_std'] = df_rdkit_fingerprints_ecfp[rdkit_fingerprints_ecfp_columns].std(axis=1).astype(np.float32)
    df_rdkit_fingerprints_ecfp['rdkit_fingerprints_ecfp_max'] = df_rdkit_fingerprints_ecfp[rdkit_fingerprints_ecfp_columns].max(axis=1).astype(np.float32)
    df_rdkit_fingerprints_ecfp['rdkit_fingerprints_ecfp_skew'] = df_rdkit_fingerprints_ecfp[rdkit_fingerprints_ecfp_columns].skew(axis=1).astype(np.float32)
    df_rdkit_fingerprints_ecfp['rdkit_fingerprints_ecfp_kurt'] = df_rdkit_fingerprints_ecfp[rdkit_fingerprints_ecfp_columns].kurt(axis=1).astype(np.float32)
    df_rdkit_fingerprints_ecfp.loc[:, rdkit_fingerprints_ecfp_columns] = (df_rdkit_fingerprints_ecfp.loc[:, rdkit_fingerprints_ecfp_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'rdkit_fingerprints_ecfp_{scaler}.pickle', mode='rb') as f:
        rdkit_fingerprints_ecfp_standard_scaler = pickle.load(f)

    df_rdkit_fingerprints_ecfp.loc[:, rdkit_fingerprints_ecfp_normalize_columns] = rdkit_fingerprints_ecfp_standard_scaler.transform(df_rdkit_fingerprints_ecfp.loc[:, rdkit_fingerprints_ecfp_normalize_columns])

    features = df_rdkit_fingerprints_ecfp.columns.tolist()
    df = pd.concat((
        df,
        df_rdkit_fingerprints_ecfp,
    ), axis=1, ignore_index=False)

    return df, features


def concat_rdkit_fingerprints_fcfp(df, df_rdkit_fingerprints_fcfp, transformer_directory, scaler='standard_scaler'):

    df_rdkit_fingerprints_fcfp = df_rdkit_fingerprints_fcfp.fillna(0)
    rdkit_fingerprints_fcfp_columns = df_rdkit_fingerprints_fcfp.columns.tolist()
    rdkit_fingerprints_fcfp_normalize_columns = [
        'rdkit_fingerprints_fcfp_sum', 'rdkit_fingerprints_fcfp_mean', 'rdkit_fingerprints_fcfp_median',
        'rdkit_fingerprints_fcfp_std', 'rdkit_fingerprints_fcfp_max', 'rdkit_fingerprints_fcfp_skew', 'rdkit_fingerprints_fcfp_kurt',
    ]
    df_rdkit_fingerprints_fcfp['rdkit_fingerprints_fcfp_sum'] = df_rdkit_fingerprints_fcfp[rdkit_fingerprints_fcfp_columns].sum(axis=1).astype(np.float32)
    df_rdkit_fingerprints_fcfp['rdkit_fingerprints_fcfp_mean'] = df_rdkit_fingerprints_fcfp[rdkit_fingerprints_fcfp_columns].mean(axis=1).astype(np.float32)
    df_rdkit_fingerprints_fcfp['rdkit_fingerprints_fcfp_median'] = df_rdkit_fingerprints_fcfp[rdkit_fingerprints_fcfp_columns].median(axis=1).astype(np.float32)
    df_rdkit_fingerprints_fcfp['rdkit_fingerprints_fcfp_std'] = df_rdkit_fingerprints_fcfp[rdkit_fingerprints_fcfp_columns].std(axis=1).astype(np.float32)
    df_rdkit_fingerprints_fcfp['rdkit_fingerprints_fcfp_max'] = df_rdkit_fingerprints_fcfp[rdkit_fingerprints_fcfp_columns].max(axis=1).astype(np.float32)
    df_rdkit_fingerprints_fcfp['rdkit_fingerprints_fcfp_skew'] = df_rdkit_fingerprints_fcfp[rdkit_fingerprints_fcfp_columns].skew(axis=1).astype(np.float32)
    df_rdkit_fingerprints_fcfp['rdkit_fingerprints_fcfp_kurt'] = df_rdkit_fingerprints_fcfp[rdkit_fingerprints_fcfp_columns].kurt(axis=1).astype(np.float32)
    df_rdkit_fingerprints_fcfp.loc[:, rdkit_fingerprints_fcfp_columns] = (df_rdkit_fingerprints_fcfp.loc[:, rdkit_fingerprints_fcfp_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'rdkit_fingerprints_fcfp_{scaler}.pickle', mode='rb') as f:
        rdkit_fingerprints_fcfp_standard_scaler = pickle.load(f)

    df_rdkit_fingerprints_fcfp.loc[:, rdkit_fingerprints_fcfp_normalize_columns] = rdkit_fingerprints_fcfp_standard_scaler.transform(df_rdkit_fingerprints_fcfp.loc[:, rdkit_fingerprints_fcfp_normalize_columns])

    features = df_rdkit_fingerprints_fcfp.columns.tolist()
    df = pd.concat((
        df,
        df_rdkit_fingerprints_fcfp,
    ), axis=1, ignore_index=False)

    return df, features


def concat_layered_fingerprints(df, df_layered_fingerprints, transformer_directory, scaler='standard_scaler'):

    df_layered_fingerprints = df_layered_fingerprints.fillna(0)
    layered_fingerprints_columns = df_layered_fingerprints.columns.tolist()
    layered_fingerprints_normalize_columns = [
        'layered_fingerprints_sum', 'layered_fingerprints_mean', 'layered_fingerprints_std',
        'layered_fingerprints_skew', 'layered_fingerprints_kurt',
    ]
    df_layered_fingerprints['layered_fingerprints_sum'] = df_layered_fingerprints[layered_fingerprints_columns].sum(axis=1).astype(np.float32)
    df_layered_fingerprints['layered_fingerprints_mean'] = df_layered_fingerprints[layered_fingerprints_columns].mean(axis=1).astype(np.float32)
    df_layered_fingerprints['layered_fingerprints_std'] = df_layered_fingerprints[layered_fingerprints_columns].std(axis=1).astype(np.float32)
    df_layered_fingerprints['layered_fingerprints_skew'] = df_layered_fingerprints[layered_fingerprints_columns].skew(axis=1).astype(np.float32)
    df_layered_fingerprints['layered_fingerprints_kurt'] = df_layered_fingerprints[layered_fingerprints_columns].kurt(axis=1).astype(np.float32)

    with open(transformer_directory / f'layered_fingerprints_{scaler}.pickle', mode='rb') as f:
        layered_fingerprints_standard_scaler = pickle.load(f)

    df_layered_fingerprints.loc[:, layered_fingerprints_normalize_columns] = layered_fingerprints_standard_scaler.transform(df_layered_fingerprints.loc[:, layered_fingerprints_normalize_columns])

    features = df_layered_fingerprints.columns.tolist()
    df = pd.concat((
        df,
        df_layered_fingerprints,
    ), axis=1, ignore_index=False)

    return df, features


def concat_pattern_fingerprints(df, df_pattern_fingerprints, transformer_directory, scaler='standard_scaler'):

    df_pattern_fingerprints = df_pattern_fingerprints.fillna(0)
    pattern_fingerprints_columns = df_pattern_fingerprints.columns.tolist()
    pattern_fingerprints_normalize_columns = [
        'pattern_fingerprints_sum', 'pattern_fingerprints_mean', 'pattern_fingerprints_std',
        'pattern_fingerprints_skew', 'pattern_fingerprints_kurt',
    ]
    df_pattern_fingerprints['pattern_fingerprints_sum'] = df_pattern_fingerprints[pattern_fingerprints_columns].sum(axis=1).astype(np.float32)
    df_pattern_fingerprints['pattern_fingerprints_mean'] = df_pattern_fingerprints[pattern_fingerprints_columns].mean(axis=1).astype(np.float32)
    df_pattern_fingerprints['pattern_fingerprints_std'] = df_pattern_fingerprints[pattern_fingerprints_columns].std(axis=1).astype(np.float32)
    df_pattern_fingerprints['pattern_fingerprints_skew'] = df_pattern_fingerprints[pattern_fingerprints_columns].skew(axis=1).astype(np.float32)
    df_pattern_fingerprints['pattern_fingerprints_kurt'] = df_pattern_fingerprints[pattern_fingerprints_columns].kurt(axis=1).astype(np.float32)

    with open(transformer_directory / f'pattern_fingerprints_{scaler}.pickle', mode='rb') as f:
        pattern_fingerprints_standard_scaler = pickle.load(f)

    df_pattern_fingerprints.loc[:, pattern_fingerprints_normalize_columns] = pattern_fingerprints_standard_scaler.transform(df_pattern_fingerprints.loc[:, pattern_fingerprints_normalize_columns])

    features = df_pattern_fingerprints.columns.tolist()
    df = pd.concat((
        df,
        df_pattern_fingerprints,
    ), axis=1, ignore_index=False)

    return df, features


def concat_avalon_fingerprints(df, df_avalon_fingerprints, transformer_directory, scaler='standard_scaler'):

    df_avalon_fingerprints = df_avalon_fingerprints.fillna(0)
    avalon_fingerprints_columns = df_avalon_fingerprints.columns.tolist()
    avalon_fingerprints_normalize_columns = [
        'avalon_fingerprints_sum', 'avalon_fingerprints_mean', 'avalon_fingerprints_median',
        'avalon_fingerprints_std', 'avalon_fingerprints_max',
        'avalon_fingerprints_skew', 'avalon_fingerprints_kurt',
    ]
    df_avalon_fingerprints['avalon_fingerprints_sum'] = df_avalon_fingerprints[avalon_fingerprints_columns].sum(axis=1).astype(np.float32)
    df_avalon_fingerprints['avalon_fingerprints_mean'] = df_avalon_fingerprints[avalon_fingerprints_columns].mean(axis=1).astype(np.float32)
    df_avalon_fingerprints['avalon_fingerprints_median'] = df_avalon_fingerprints[avalon_fingerprints_columns].median(axis=1).astype(np.float32)
    df_avalon_fingerprints['avalon_fingerprints_std'] = df_avalon_fingerprints[avalon_fingerprints_columns].std(axis=1).astype(np.float32)
    df_avalon_fingerprints['avalon_fingerprints_max'] = df_avalon_fingerprints[avalon_fingerprints_columns].max(axis=1).astype(np.float32)
    df_avalon_fingerprints['avalon_fingerprints_skew'] = df_avalon_fingerprints[avalon_fingerprints_columns].skew(axis=1).astype(np.float32)
    df_avalon_fingerprints['avalon_fingerprints_kurt'] = df_avalon_fingerprints[avalon_fingerprints_columns].kurt(axis=1).astype(np.float32)
    df_avalon_fingerprints.loc[:, avalon_fingerprints_columns] = (df_avalon_fingerprints.loc[:, avalon_fingerprints_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'avalon_fingerprints_{scaler}.pickle', mode='rb') as f:
        avalon_fingerprints_standard_scaler = pickle.load(f)

    df_avalon_fingerprints.loc[:, avalon_fingerprints_normalize_columns] = avalon_fingerprints_standard_scaler.transform(df_avalon_fingerprints.loc[:, avalon_fingerprints_normalize_columns])

    features = df_avalon_fingerprints.columns.tolist()
    df = pd.concat((
        df,
        df_avalon_fingerprints,
    ), axis=1, ignore_index=False)

    return df, features


def concat_invariants(df, df_invariants, transformer_directory, scaler='standard_scaler'):

    df_invariants = df_invariants.fillna(0)
    df_invariants = np.log1p(df_invariants)
    invariants_columns = df_invariants.columns.tolist()
    df_invariants['invariant_sum'] = df_invariants[invariants_columns].sum(axis=1).astype(np.float32)
    df_invariants['invariant_mean'] = df_invariants[invariants_columns].mean(axis=1).astype(np.float32)
    df_invariants['invariant_median'] = df_invariants[invariants_columns].median(axis=1).astype(np.float32)
    df_invariants['invariant_std'] = df_invariants[invariants_columns].std(axis=1).astype(np.float32)
    df_invariants['invariant_max'] = df_invariants[invariants_columns].max(axis=1).astype(np.float32)
    df_invariants['invariant_skew'] = df_invariants[invariants_columns].skew(axis=1).astype(np.float32)
    df_invariants['invariant_kurt'] = df_invariants[invariants_columns].kurt(axis=1).astype(np.float32)

    with open(transformer_directory / f'invariants_{scaler}.pickle', mode='rb') as f:
        invariants_scaler = pickle.load(f)

    df_invariants.loc[:, :] = invariants_scaler.transform(df_invariants)

    features = df_invariants.columns.tolist()
    df = pd.concat((
        df,
        df_invariants,
    ), axis=1, ignore_index=False)

    return df, features


def concat_brics_counts(df, df_brics_counts, transformer_directory, scaler='standard_scaler'):

    df_brics_counts = df_brics_counts.fillna(0.)
    brics_counts_columns = df_brics_counts.columns.tolist()
    brics_counts_normalize_columns = [
        'brics_counts_sum', 'brics_counts_mean', 'brics_counts_median',
        'brics_counts_std', 'brics_counts_max',
        'brics_counts_skew', 'brics_counts_kurt',
    ]
    df_brics_counts['brics_counts_sum'] = df_brics_counts[brics_counts_columns].sum(axis=1).astype(np.float32)
    df_brics_counts['brics_counts_mean'] = df_brics_counts[brics_counts_columns].mean(axis=1).astype(np.float32)
    df_brics_counts['brics_counts_median'] = df_brics_counts[brics_counts_columns].median(axis=1).astype(np.float32)
    df_brics_counts['brics_counts_std'] = df_brics_counts[brics_counts_columns].std(axis=1).astype(np.float32)
    df_brics_counts['brics_counts_max'] = df_brics_counts[brics_counts_columns].max(axis=1).astype(np.float32)
    df_brics_counts['brics_counts_skew'] = df_brics_counts[brics_counts_columns].skew(axis=1).astype(np.float32)
    df_brics_counts['brics_counts_kurt'] = df_brics_counts[brics_counts_columns].kurt(axis=1).astype(np.float32)
    df_brics_counts.loc[:, brics_counts_columns] = (df_brics_counts.loc[:, brics_counts_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'brics_counts_{scaler}.pickle', mode='rb') as f:
        brics_counts_scaler = pickle.load(f)

    df_brics_counts.loc[:, brics_counts_normalize_columns] = brics_counts_scaler.transform(df_brics_counts.loc[:, brics_counts_normalize_columns])

    features = df_brics_counts.columns.tolist()
    df = pd.concat((
        df,
        df_brics_counts,
    ), axis=1, ignore_index=False)

    return df, features


def concat_scaffold_counts(df, df_scaffold_counts, transformer_directory, scaler='standard_scaler'):

    df_scaffold_counts = df_scaffold_counts.fillna(0.)
    scaffold_counts_columns = df_scaffold_counts.columns.tolist()
    scaffold_counts_normalize_columns = [
        'scaffold_counts_sum', 'scaffold_counts_mean', 'scaffold_counts_median',
        'scaffold_counts_std', 'scaffold_counts_max',
        'scaffold_counts_skew', 'scaffold_counts_kurt',
    ]
    df_scaffold_counts['scaffold_counts_sum'] = df_scaffold_counts[scaffold_counts_columns].sum(axis=1).astype(np.float32)
    df_scaffold_counts['scaffold_counts_mean'] = df_scaffold_counts[scaffold_counts_columns].mean(axis=1).astype(np.float32)
    df_scaffold_counts['scaffold_counts_median'] = df_scaffold_counts[scaffold_counts_columns].median(axis=1).astype(np.float32)
    df_scaffold_counts['scaffold_counts_std'] = df_scaffold_counts[scaffold_counts_columns].std(axis=1).astype(np.float32)
    df_scaffold_counts['scaffold_counts_max'] = df_scaffold_counts[scaffold_counts_columns].max(axis=1).astype(np.float32)
    df_scaffold_counts['scaffold_counts_skew'] = df_scaffold_counts[scaffold_counts_columns].skew(axis=1).astype(np.float32)
    df_scaffold_counts['scaffold_counts_kurt'] = df_scaffold_counts[scaffold_counts_columns].kurt(axis=1).astype(np.float32)
    df_scaffold_counts.loc[:, scaffold_counts_columns] = (df_scaffold_counts.loc[:, scaffold_counts_columns] > 0).astype(np.uint8)

    with open(transformer_directory / f'scaffold_counts_{scaler}.pickle', mode='rb') as f:
        scaffold_counts_scaler = pickle.load(f)

    df_scaffold_counts.loc[:, scaffold_counts_normalize_columns] = scaffold_counts_scaler.transform(df_scaffold_counts.loc[:, scaffold_counts_normalize_columns])

    features = df_scaffold_counts.columns.tolist()
    df = pd.concat((
        df,
        df_scaffold_counts,
    ), axis=1, ignore_index=False)

    return df, features


def concat_mordred_descriptors(df, df_mordred_descriptors, transformer_directory, scaler='standard_scaler'):

    df_mordred_descriptors = df_mordred_descriptors.astype(np.float64)
    df_mordred_descriptors = df_mordred_descriptors.replace([np.inf, -np.inf], np.nan)
    
    df_mordred_descriptors.columns = [f'mordred_{column}' for column in df_mordred_descriptors.columns]
    df_mordred_descriptors = df_mordred_descriptors.drop(columns=['mordred_MOMI-X', 'mordred_Mor32p', 'mordred_Mor31p', 'mordred_Mor30p', 'mordred_Mor29p', 'mordred_Mor28p', 'mordred_PNSA1', 'mordred_PNSA2', 'mordred_PNSA3', 'mordred_PNSA4', 'mordred_PNSA5', 'mordred_PPSA1', 'mordred_PPSA2', 'mordred_PPSA3', 'mordred_PPSA4', 'mordred_PPSA5', 'mordred_Mor27se', 'mordred_Mor28se', 'mordred_Mor29se', 'mordred_Mor30se', 'mordred_Mor31se', 'mordred_Mor32se', 'mordred_Mor01p', 'mordred_Mor02p', 'mordred_Mor03p', 'mordred_Mor04p', 'mordred_Mor05p', 'mordred_Mor06p', 'mordred_Mor07p', 'mordred_Mor08p', 'mordred_Mor09p', 'mordred_Mor10p', 'mordred_Mor11se', 'mordred_Mor12se', 'mordred_Mor13se', 'mordred_Mor14se', 'mordred_Mor15se', 'mordred_Mor16se', 'mordred_Mor17se', 'mordred_Mor18se', 'mordred_Mor19se', 'mordred_Mor20se', 'mordred_Mor21se', 'mordred_Mor22se', 'mordred_Mor23se', 'mordred_Mor24se', 'mordred_Mor25se', 'mordred_Mor26se', 'mordred_Mor28v', 'mordred_Mor29v', 'mordred_Mor30v', 'mordred_Mor31v', 'mordred_Mor32v', 'mordred_Mor01se', 'mordred_Mor02se', 'mordred_Mor03se', 'mordred_Mor04se', 'mordred_Mor05se', 'mordred_Mor06se', 'mordred_Mor07se', 'mordred_Mor08se', 'mordred_Mor09se', 'mordred_Mor10se', 'mordred_Mor13v', 'mordred_Mor14v', 'mordred_Mor15v', 'mordred_Mor16v', 'mordred_Mor17v', 'mordred_Mor18v', 'mordred_Mor19v', 'mordred_Mor20v', 'mordred_Mor21v', 'mordred_Mor22v', 'mordred_Mor23v', 'mordred_Mor24v', 'mordred_Mor25v', 'mordred_Mor26v', 'mordred_Mor27v', 'mordred_Mor31m', 'mordred_Mor32m', 'mordred_Mor01v', 'mordred_Mor02v', 'mordred_Mor03v', 'mordred_Mor04v', 'mordred_Mor05v', 'mordred_Mor06v', 'mordred_Mor07v', 'mordred_Mor08v', 'mordred_Mor09v', 'mordred_Mor10v', 'mordred_Mor11v', 'mordred_Mor12v', 'mordred_Mor19m', 'mordred_Mor20m', 'mordred_Mor21m', 'mordred_Mor22m', 'mordred_Mor23m', 'mordred_Mor24m', 'mordred_Mor25m', 'mordred_Mor26m', 'mordred_Mor27m', 'mordred_Mor28m', 'mordred_Mor29m', 'mordred_Mor30m', 'mordred_Mor06m', 'mordred_Mor07m', 'mordred_Mor08m', 'mordred_Mor09m', 'mordred_Mor10m', 'mordred_Mor11m', 'mordred_Mor12m', 'mordred_Mor13m', 'mordred_Mor14m', 'mordred_Mor15m', 'mordred_Mor16m', 'mordred_Mor17m', 'mordred_Mor18m', 'mordred_DPSA1', 'mordred_DPSA2', 'mordred_DPSA3', 'mordred_DPSA4', 'mordred_DPSA5', 'mordred_FNSA1', 'mordred_FNSA2', 'mordred_FNSA3', 'mordred_FNSA4', 'mordred_FNSA5', 'mordred_FPSA1', 'mordred_FPSA2', 'mordred_FPSA3', 'mordred_Mor27', 'mordred_Mor28', 'mordred_Mor29', 'mordred_Mor30', 'mordred_Mor31', 'mordred_Mor32', 'mordred_Mor01m', 'mordred_Mor02m', 'mordred_Mor03m', 'mordred_Mor04m', 'mordred_Mor05m', 'mordred_Mor11', 'mordred_Mor12', 'mordred_Mor13', 'mordred_Mor14', 'mordred_Mor15', 'mordred_Mor16', 'mordred_Mor17', 'mordred_Mor18', 'mordred_Mor19', 'mordred_Mor20', 'mordred_Mor21', 'mordred_Mor22', 'mordred_Mor23', 'mordred_Mor24', 'mordred_Mor25', 'mordred_Mor26', 'mordred_GeomShapeIndex', 'mordred_GeomPetitjeanIndex', 'mordred_GRAV', 'mordred_GRAVH', 'mordred_GRAVp', 'mordred_GRAVHp', 'mordred_Mor01', 'mordred_Mor02', 'mordred_Mor03', 'mordred_Mor04', 'mordred_Mor05', 'mordred_Mor06', 'mordred_Mor07', 'mordred_Mor08', 'mordred_Mor09', 'mordred_Mor10', 'mordred_MINdssSe', 'mordred_MINddssSe', 'mordred_MINsSnH3', 'mordred_MINssSnH2', 'mordred_MINsssSnH', 'mordred_MINsPbH3', 'mordred_MINssPbH2', 'mordred_MINsssPbH', 'mordred_MINssssPb', 'mordred_GeomDiameter', 'mordred_GeomRadius', 'mordred_MINsssssP', 'mordred_MINsGeH3', 'mordred_MINssGeH2', 'mordred_MINsssGeH', 'mordred_MINsAsH2', 'mordred_MINssAsH', 'mordred_MINsssAs', 'mordred_MINsssdAs', 'mordred_MINsssssAs', 'mordred_MINsSeH', 'mordred_MINdSe', 'mordred_FPSA4', 'mordred_FPSA5', 'mordred_WNSA1', 'mordred_WNSA2', 'mordred_WNSA3', 'mordred_WNSA4', 'mordred_WNSA5', 'mordred_WPSA1', 'mordred_WPSA2', 'mordred_WPSA3', 'mordred_WPSA4', 'mordred_WPSA5', 'mordred_RNCS', 'mordred_RPCS', 'mordred_MINsNH3', 'mordred_MINssNH2', 'mordred_MINsSiH3', 'mordred_MINsPH2', 'mordred_MINssPH', 'mordred_TASA', 'mordred_TPSA', 'mordred_RASA', 'mordred_RPSA', 'mordred_MAXssBe', 'mordred_MAXssssBe', 'mordred_MAXssBH', 'mordred_MAXssssB', 'mordred_MAXsssSnH', 'mordred_MAXsPbH3', 'mordred_MAXssPbH2', 'mordred_MAXsssPbH', 'mordred_MAXssssPb', 'mordred_MINssBe', 'mordred_MINssssBe', 'mordred_MINssBH', 'mordred_MINssssB', 'mordred_MAXsssGeH', 'mordred_MAXsAsH2', 'mordred_MAXssAsH', 'mordred_MAXsssAs', 'mordred_MAXsssdAs', 'mordred_MAXsssssAs', 'mordred_MAXsSeH', 'mordred_MAXdSe', 'mordred_MAXdssSe', 'mordred_MAXddssSe', 'mordred_MAXsSnH3', 'mordred_MAXssSnH2', 'mordred_MAXsNH3', 'mordred_MAXssNH2', 'mordred_MAXsSiH3', 'mordred_MAXsPH2', 'mordred_MAXssPH', 'mordred_MAXsssssP', 'mordred_MAXsGeH3', 'mordred_MAXssGeH2', 'mordred_Mor27p', 'mordred_MOMI-Y', 'mordred_MOMI-Z', 'mordred_PBF', 'mordred_Mor11p', 'mordred_Mor12p', 'mordred_Mor13p', 'mordred_Mor14p', 'mordred_Mor15p', 'mordred_Mor16p', 'mordred_Mor17p', 'mordred_Mor18p', 'mordred_Mor19p', 'mordred_Mor20p', 'mordred_Mor21p', 'mordred_Mor22p', 'mordred_Mor23p', 'mordred_Mor24p', 'mordred_Mor25p', 'mordred_Mor26p', 'mordred_MAXaaSe', 'mordred_MAXsssNH', 'mordred_MINaaSe', 'mordred_MINsssNH', 'mordred_MINsLi', 'mordred_MAXsLi', 'mordred_MAXssSiH2', 'mordred_MAXdNH', 'mordred_MINssSiH2', 'mordred_MINdNH', 'mordred_MAXsSH', 'mordred_MINsSH', 'mordred_MINtCH', 'mordred_MAXtCH', 'mordred_MAXssSe', 'mordred_MAXsssP', 'mordred_MINsssP', 'mordred_MINssSe', 'mordred_MINddC', 'mordred_MAXssssGe', 'mordred_MAXddC', 'mordred_MINssssGe', 'mordred_MINssssN', 'mordred_MAXssssN', 'mordred_MINsssSiH', 'mordred_MINssssSn', 'mordred_MAXsssSiH', 'mordred_MAXssssSn', 'mordred_MINsssB', 'mordred_MAXsssB', 'mordred_MAXsI', 'mordred_MINsI', 'mordred_MINdssS', 'mordred_MAXdssS', 'mordred_MINdCH2', 'mordred_MAXdCH2', 'mordred_MAXdS', 'mordred_MINdS', 'mordred_MINsBr', 'mordred_MAXsBr', 'mordred_MAXaaNH', 'mordred_MINaaNH', 'mordred_MDEN-13', 'mordred_MAXsNH2', 'mordred_MINsNH2', 'mordred_MATS8i', 'mordred_GATS8c', 'mordred_MATS8are', 'mordred_AATSC8c', 'mordred_AATSC8p', 'mordred_MATS8pe', 'mordred_MATS8p', 'mordred_MATS8se', 'mordred_AATSC8are', 'mordred_AATSC8pe', 'mordred_AATSC8se', 'mordred_GATS8v', 'mordred_GATS8s', 'mordred_MATS8m', 'mordred_MATS8v', 'mordred_GATS8se', 'mordred_GATS8m', 'mordred_GATS8pe', 'mordred_AATS8s', 'mordred_GATS8are', 'mordred_GATS8i', 'mordred_GATS8p', 'mordred_MATS8s', 'mordred_AATS8v', 'mordred_AATS8m', 'mordred_AATS8se', 'mordred_AATS8are', 'mordred_AATSC8m', 'mordred_AATSC8v', 'mordred_AATSC8s', 'mordred_AATSC8i', 'mordred_AATS8pe', 'mordred_AATS8i', 'mordred_AATS8p', 'mordred_MATS8c', 'mordred_AXp-7dv', 'mordred_MINsCl', 'mordred_MAXsCl', 'mordred_MINaaO', 'mordred_MAXaaO', 'mordred_AATSC7v', 'mordred_GATS7c', 'mordred_MATS7se', 'mordred_AATSC7p', 'mordred_AATSC7se', 'mordred_AATSC7c', 'mordred_AATSC7pe', 'mordred_AATSC7s', 'mordred_AATSC7m', 'mordred_MATS7pe', 'mordred_MATS7i', 'mordred_MATS7are', 'mordred_MATS7p', 'mordred_AATS7are', 'mordred_AATSC7i', 'mordred_AATS7p', 'mordred_AATSC7are', 'mordred_AATS7i', 'mordred_MATS7c', 'mordred_MATS7s', 'mordred_GATS7p', 'mordred_GATS7i', 'mordred_AATS7pe', 'mordred_AATS7se', 'mordred_AATS7v', 'mordred_AATS7m', 'mordred_AATS7s', 'mordred_GATS7v', 'mordred_GATS7se', 'mordred_GATS7are', 'mordred_GATS7pe', 'mordred_MATS7v', 'mordred_GATS7m', 'mordred_MATS7m', 'mordred_GATS7s', 'mordred_MINaaS', 'mordred_MAXaaS', 'mordred_AXp-6dv', 'mordred_MDEN-11', 'mordred_AXp-5dv', 'mordred_MATS6se', 'mordred_AATSC6m', 'mordred_AATSC6s', 'mordred_AATS6pe', 'mordred_AATS6p', 'mordred_AATSC6v', 'mordred_MATS6p', 'mordred_AATSC6c', 'mordred_AATS6i', 'mordred_AATS6v', 'mordred_MATS6s', 'mordred_GATS6v', 'mordred_GATS6se', 'mordred_GATS6are', 'mordred_GATS6m', 'mordred_MATS6pe', 'mordred_GATS6c', 'mordred_GATS6s', 'mordred_MATS6m', 'mordred_AATS6are', 'mordred_MATS6i', 'mordred_AATSC6se', 'mordred_AATSC6pe', 'mordred_MATS6are', 'mordred_AATS6se', 'mordred_AATSC6i', 'mordred_MATS6c', 'mordred_GATS6p', 'mordred_AATS6s', 'mordred_GATS6i', 'mordred_AATSC6p', 'mordred_GATS6pe', 'mordred_MATS6v', 'mordred_AATSC6are', 'mordred_AATS6m', 'mordred_AXp-4dv', 'mordred_MAXddsN', 'mordred_MINddsN', 'mordred_MATS5p', 'mordred_AATSC5c', 'mordred_MATS5pe', 'mordred_AATSC5m', 'mordred_AATSC5s', 'mordred_AATS5se', 'mordred_MATS5i', 'mordred_GATS5s', 'mordred_AATS5v', 'mordred_AATS5are', 'mordred_AATSC5pe', 'mordred_MATS5are', 'mordred_GATS5c', 'mordred_AATSC5v', 'mordred_AATSC5se', 'mordred_AATS5p', 'mordred_AATS5i', 'mordred_AATS5pe', 'mordred_MATS5se', 'mordred_GATS5i', 'mordred_AATS5s', 'mordred_MATS5c', 'mordred_AATS5m', 'mordred_AATSC5are', 'mordred_AATSC5p', 'mordred_AATSC5i', 'mordred_MATS5m', 'mordred_GATS5se', 'mordred_GATS5v', 'mordred_GATS5pe', 'mordred_GATS5p', 'mordred_GATS5are', 'mordred_MATS5v', 'mordred_MATS5s', 'mordred_GATS5m', 'mordred_AXp-3dv', 'mordred_AATSC4c', 'mordred_GATS4v', 'mordred_MATS4v', 'mordred_MATS4p', 'mordred_MATS4pe', 'mordred_MATS4s', 'mordred_GATS4se', 'mordred_MATS4m', 'mordred_GATS4are', 'mordred_AATSC4v', 'mordred_AATSC4se', 'mordred_AATS4i', 'mordred_MATS4i', 'mordred_AATSC4pe', 'mordred_GATS4s', 'mordred_MATS4c', 'mordred_AATSC4are', 'mordred_AATS4s', 'mordred_AATS4v', 'mordred_MATS4se', 'mordred_AATSC4i', 'mordred_AATS4are', 'mordred_AATSC4p', 'mordred_GATS4p', 'mordred_AATS4m', 'mordred_AATS4se', 'mordred_AATS4pe', 'mordred_AATSC4s', 'mordred_GATS4c', 'mordred_AATSC4m', 'mordred_GATS4m', 'mordred_MATS4are', 'mordred_AATS4p', 'mordred_GATS4pe', 'mordred_GATS4i', 'mordred_AATSC3se', 'mordred_GATS3v', 'mordred_GATS3p', 'mordred_MATS3m', 'mordred_AATS3s', 'mordred_MATS3c', 'mordred_MATS3s', 'mordred_MATS3se', 'mordred_AATSC3p', 'mordred_AATSC3i', 'mordred_AATS3are', 'mordred_AATS3pe', 'mordred_AATS3p', 'mordred_MATS3p', 'mordred_GATS3c', 'mordred_AATS3m', 'mordred_AATS3v', 'mordred_AATS3se', 'mordred_GATS3s', 'mordred_AATSC3m', 'mordred_GATS3m', 'mordred_MATS3pe', 'mordred_MATS3i', 'mordred_MATS3v', 'mordred_GATS3se', 'mordred_AATSC3c', 'mordred_AATSC3v', 'mordred_GATS3are', 'mordred_AATSC3are', 'mordred_AATSC3pe', 'mordred_GATS3pe', 'mordred_GATS3i', 'mordred_AATS3i', 'mordred_MATS3are', 'mordred_AATSC3s', 'mordred_AXp-2dv', 'mordred_Vabc', 'mordred_AATSC2s', 'mordred_ATSC5s', 'mordred_ATSC6s', 'mordred_AATSC1s', 'mordred_AATSC0s', 'mordred_ATSC8s', 'mordred_ATSC7s', 'mordred_ATS2s', 'mordred_ATS1s', 'mordred_ATS8s', 'mordred_ATS7s', 'mordred_ATS6s', 'mordred_ATS5s', 'mordred_ATS4s', 'mordred_ATS3s', 'mordred_ATSC4s', 'mordred_AXp-1dv', 'mordred_ATSC1s', 'mordred_ATSC2s', 'mordred_ATSC3s', 'mordred_BCUTs-1l', 'mordred_BCUTs-1h', 'mordred_ATSC0s', 'mordred_GATS1s', 'mordred_GATS2s', 'mordred_MATS1s', 'mordred_MATS2s', 'mordred_AATS0s', 'mordred_AATS1s', 'mordred_AATS2s', 'mordred_ATS0s', 'mordred_VR3_Dzp', 'mordred_VR3_Dzv', 'mordred_AETA_eta_FL', 'mordred_ETA_eta_FL', 'mordred_AETA_eta_F', 'mordred_ETA_eta_F', 'mordred_VR3_Dzpe', 'mordred_VR3_Dzare', 'mordred_AETA_eta', 'mordred_VR3_Dzi', 'mordred_AETA_eta_L', 'mordred_Xp-0dv', 'mordred_VR3_DzZ', 'mordred_ETA_eta', 'mordred_AXp-0dv', 'mordred_VR3_Dzm', 'mordred_VR3_Dzse', 'mordred_ETA_eta_L', 'mordred_AATSC1se', 'mordred_AATSC0se', 'mordred_GATS2p', 'mordred_MATS1m', 'mordred_AATSC2se', 'mordred_ATSC0m', 'mordred_AATS2i', 'mordred_ATSC2c', 'mordred_GATS1c', 'mordred_GATS2c', 'mordred_MATS1are', 'mordred_MATS2are', 'mordred_MATS1p', 'mordred_MATS2p', 'mordred_MATS2m', 'mordred_ATSC2m', 'mordred_ATSC3m', 'mordred_ATSC4m', 'mordred_ATSC5m', 'mordred_ATSC6m', 'mordred_ATSC7m', 'mordred_ATSC8m', 'mordred_ATSC0v', 'mordred_ATSC1v', 'mordred_ATSC0c', 'mordred_ATSC1c', 'mordred_ATS8i', 'mordred_VMcGowan', 'mordred_ATSC5i', 'mordred_ATSC6i', 'mordred_AATSC2v', 'mordred_GATS1are', 'mordred_GATS2are', 'mordred_GATS1p', 'mordred_ATS7i', 'mordred_ATSC3c', 'mordred_ATS2are', 'mordred_ATS3are', 'mordred_ATS4are', 'mordred_ATSC8se', 'mordred_BCUTpe-1h', 'mordred_BCUTpe-1l', 'mordred_BCUTare-1h', 'mordred_BCUTare-1l', 'mordred_BCUTp-1h', 'mordred_BCUTp-1l', 'mordred_BCUTi-1h', 'mordred_BCUTi-1l', 'mordred_SpAbs_DzZ', 'mordred_SpMax_DzZ', 'mordred_AATSC2m', 'mordred_AATSC0are', 'mordred_AATSC1are', 'mordred_AATSC2are', 'mordred_ATSC7c', 'mordred_Mp', 'mordred_Mi', 'mordred_Xp-1dv', 'mordred_Xp-2dv', 'mordred_AATS1pe', 'mordred_AATS2pe', 'mordred_AATS0are', 'mordred_SpDiam_DzZ', 'mordred_ATSC4c', 'mordred_ATSC5c', 'mordred_ATSC6c', 'mordred_AATSC0m', 'mordred_AATSC1m', 'mordred_ATSC1m', 'mordred_AATSC1v', 'mordred_ATSC8c', 'mordred_AATS0p', 'mordred_AATS1p', 'mordred_AATS2p', 'mordred_ATS6i', 'mordred_GATS2pe', 'mordred_GATS2i', 'mordred_BCUTc-1h', 'mordred_BCUTc-1l', 'mordred_BCUTm-1h', 'mordred_BCUTm-1l', 'mordred_BCUTv-1h', 'mordred_BCUTv-1l', 'mordred_AATSC0v', 'mordred_AATS0i', 'mordred_AATS1i', 'mordred_GATS1pe', 'mordred_ETA_shape_y', 'mordred_ETA_shape_x', 'mordred_GATS2m', 'mordred_GATS1v', 'mordred_ETA_dPsi_B', 'mordred_ATSC2v', 'mordred_ETA_alpha', 'mordred_ATSC1pe', 'mordred_ATSC2pe', 'mordred_ATSC3pe', 'mordred_GATS1se', 'mordred_ATSC5pe', 'mordred_ATSC6pe', 'mordred_ATSC7pe', 'mordred_ATSC8pe', 'mordred_ATSC0are', 'mordred_ATSC7are', 'mordred_ATSC8are', 'mordred_ETA_epsilon_4', 'mordred_ETA_epsilon_5', 'mordred_ATSC1p', 'mordred_MATS1i', 'mordred_ATSC4pe', 'mordred_ATSC5v', 'mordred_ATS1are', 'mordred_ATS4v', 'mordred_ATS5v', 'mordred_ATS6v', 'mordred_ATS7v', 'mordred_ATS8v', 'mordred_ATS0se', 'mordred_ATS1se', 'mordred_ATS2se', 'mordred_ETA_shape_p', 'mordred_ATS6pe', 'mordred_ATS7pe', 'mordred_ATS8pe', 'mordred_ATS8m', 'mordred_ATS7m', 'mordred_ETA_dEpsilon_B', 'mordred_ETA_dEpsilon_C', 'mordred_ETA_dEpsilon_D', 'mordred_ETA_psi_1', 'mordred_ETA_dPsi_A', 'mordred_AETA_alpha', 'mordred_ATS5pe', 'mordred_ATSC3se', 'mordred_ATSC4se', 'mordred_ATSC5se', 'mordred_ATSC6se', 'mordred_ATSC7se', 'mordred_ATSC0p', 'mordred_ATSC7i', 'mordred_ATSC8i', 'mordred_AATSC0c', 'mordred_AATSC1c', 'mordred_MATS2i', 'mordred_ATS1m', 'mordred_ATS2m', 'mordred_ATS3m', 'mordred_ATS4m', 'mordred_GATS2v', 'mordred_AATS1are', 'mordred_GATS2se', 'mordred_MATS1se', 'mordred_MATS2se', 'mordred_AATSC2c', 'mordred_ATSC5p', 'mordred_ATS0m', 'mordred_apol', 'mordred_bpol', 'mordred_MATS1pe', 'mordred_MATS2pe', 'mordred_ATS7se', 'mordred_ATSC2p', 'mordred_ATSC3p', 'mordred_ATSC4p', 'mordred_ATSC6v', 'mordred_ATSC7v', 'mordred_ATSC2se', 'mordred_ATSC4i', 'mordred_ATSC0pe', 'mordred_ATSC6p', 'mordred_ATSC7p', 'mordred_ATSC8p', 'mordred_ATSC0i', 'mordred_ATSC1i', 'mordred_ATSC2i', 'mordred_ATSC3i', 'mordred_ATSC0se', 'mordred_ATSC1se', 'mordred_ATSC8v', 'mordred_SpDiam_Dzare', 'mordred_SpAD_Dzare', 'mordred_SpMAD_Dzare', 'mordred_LogEE_Dzare', 'mordred_SpAbs_Dzse', 'mordred_VE1_Dzare', 'mordred_VE2_Dzare', 'mordred_VE3_Dzare', 'mordred_VR1_Dzare', 'mordred_VR2_Dzare', 'mordred_ATS4i', 'mordred_SpMax_Dzp', 'mordred_SpDiam_Dzp', 'mordred_SpAD_Dzp', 'mordred_VR2_Dzse', 'mordred_SpAbs_Dzpe', 'mordred_SpMax_Dzpe', 'mordred_SpDiam_Dzpe', 'mordred_VR1_Dzm', 'mordred_SpMAD_Dzpe', 'mordred_LogEE_Dzpe', 'mordred_SM1_Dzpe', 'mordred_SpAbs_Dzp', 'mordred_SpMax_Dzv', 'mordred_SpDiam_Dzv', 'mordred_SpAD_Dzv', 'mordred_SpMAD_Dzv', 'mordred_LogEE_Dzv', 'mordred_SM1_Dzv', 'mordred_VE1_Dzv', 'mordred_SpMAD_Dzp', 'mordred_LogEE_Dzp', 'mordred_SM1_Dzp', 'mordred_SpMax_Dzare', 'mordred_VE2_Dzp', 'mordred_SpAD_Dzpe', 'mordred_VR1_Dzp', 'mordred_VR2_Dzp', 'mordred_SpAbs_Dzi', 'mordred_SpMax_Dzi', 'mordred_SpDiam_Dzi', 'mordred_SpAD_Dzi', 'mordred_SpMAD_Dzi', 'mordred_LogEE_Dzi', 'mordred_SM1_Dzi', 'mordred_VE1_Dzp', 'mordred_ATS0v', 'mordred_ATS1v', 'mordred_ATS2v', 'mordred_ATS3v', 'mordred_ATS8se', 'mordred_ATS0pe', 'mordred_ATS1pe', 'mordred_ATS2pe', 'mordred_ATS6m', 'mordred_ATSC3v', 'mordred_VE1_Dzpe', 'mordred_ATS3pe', 'mordred_ETA_dEpsilon_A', 'mordred_ATSC6are', 'mordred_ATSC5are', 'mordred_ATSC4are', 'mordred_ATSC3are', 'mordred_ATSC2are', 'mordred_ATSC1are', 'mordred_ETA_epsilon_2', 'mordred_ETA_epsilon_1', 'mordred_ETA_dAlpha_B', 'mordred_ATSC4v', 'mordred_VE2_Dzpe', 'mordred_VE3_Dzpe', 'mordred_VR1_Dzpe', 'mordred_VR2_Dzpe', 'mordred_SpAbs_Dzare', 'mordred_ATS4pe', 'mordred_VE3_Dzv', 'mordred_VR1_Dzv', 'mordred_VR2_Dzv', 'mordred_VE1_DzZ', 'mordred_ATS5m', 'mordred_SpDiam_Dzse', 'mordred_SpAD_Dzse', 'mordred_SpMAD_Dzse', 'mordred_LogEE_Dzse', 'mordred_SM1_Dzse', 'mordred_VE1_Dzse', 'mordred_VE2_Dzse', 'mordred_VE3_Dzse', 'mordred_VR1_Dzse', 'mordred_ATS0are', 'mordred_ATS3se', 'mordred_SpMax_Dzse', 'mordred_SpDiam_Dzm', 'mordred_SpAD_Dzm', 'mordred_SpMAD_Dzm', 'mordred_LogEE_Dzm', 'mordred_AATSC2i', 'mordred_AATS0se', 'mordred_AATS1se', 'mordred_AATS2se', 'mordred_AATS0pe', 'mordred_AATS0m', 'mordred_SpAbs_Dzv', 'mordred_AATS2m', 'mordred_ATS8p', 'mordred_ATS4se', 'mordred_ATS5se', 'mordred_AATSC1p', 'mordred_AATSC2p', 'mordred_AATSC0i', 'mordred_AATSC1i', 'mordred_SpAD_DzZ', 'mordred_SpMAD_DzZ', 'mordred_LogEE_DzZ', 'mordred_AATS1m', 'mordred_AATSC0pe', 'mordred_AATSC1pe', 'mordred_AATSC2pe', 'mordred_BCUTse-1h', 'mordred_ATS3i', 'mordred_ETA_dAlpha_A', 'mordred_ATS5i', 'mordred_ATS6se', 'mordred_MATS1v', 'mordred_MATS2v', 'mordred_Mare', 'mordred_Sv', 'mordred_AATS2are', 'mordred_Sse', 'mordred_Spe', 'mordred_Sare', 'mordred_Sp', 'mordred_Si', 'mordred_Mm', 'mordred_Mv', 'mordred_Mse', 'mordred_Mpe', 'mordred_Sm', 'mordred_VE2_Dzv', 'mordred_VE2_Dzi', 'mordred_VE3_Dzi', 'mordred_VR1_Dzi', 'mordred_VR2_Dzi', 'mordred_SM1_Dzare', 'mordred_RNCG', 'mordred_RPCG', 'mordred_ATS0i', 'mordred_ATS1i', 'mordred_SM1_DzZ', 'mordred_VE3_Dzp', 'mordred_GATS1m', 'mordred_ATS5are', 'mordred_ATS6are', 'mordred_VE1_Dzi', 'mordred_SM1_Dzm', 'mordred_VE1_Dzm', 'mordred_VE2_Dzm', 'mordred_VE3_Dzm', 'mordred_BCUTse-1l', 'mordred_VR2_Dzm', 'mordred_ATS2i', 'mordred_GATS1i', 'mordred_VE2_DzZ', 'mordred_VE3_DzZ', 'mordred_VR1_DzZ', 'mordred_VR2_DzZ', 'mordred_SpAbs_Dzm', 'mordred_SpMax_Dzm', 'mordred_ATS7are', 'mordred_ATS8are', 'mordred_ATS0p', 'mordred_AATSC0p', 'mordred_ATS2p', 'mordred_ATS3p', 'mordred_ATS4p', 'mordred_ATS5p', 'mordred_ATS6p', 'mordred_ATS7p', 'mordred_MATS1c', 'mordred_MATS2c', 'mordred_AATS0v', 'mordred_AATS1v', 'mordred_AATS2v', 'mordred_ATS1p', 'mordred_Xp-3dv', 'mordred_MINssS', 'mordred_MAXssS', 'mordred_Xp-4dv', 'mordred_MAXdsssP', 'mordred_MINdsssP', 'mordred_MDEN-12', 'mordred_Xp-5dv', 'mordred_MINtN', 'mordred_MAXtN', 'mordred_MAXssssSi', 'mordred_MINssssSi', 'mordred_Xpc-5dv', 'mordred_Xpc-4dv', 'mordred_MINaasN', 'mordred_MAXaasN', 'mordred_MAXsOH', 'mordred_Xpc-6dv', 'mordred_MINsOH', 'mordred_MINtsC', 'mordred_MAXtsC', 'mordred_Xp-6dv', 'mordred_Xp-7dv', 'mordred_MAXddssS', 'mordred_MINddssS', 'mordred_MINdsN', 'mordred_MAXdsN', 'mordred_MINaaN', 'mordred_MAXaaN', 'mordred_MINsF', 'mordred_MAXsF', 'mordred_MDEN-23', 'mordred_MINdsCH', 'mordred_MAXdsCH', 'mordred_MDEC-44', 'mordred_MAXaaaC', 'mordred_MINaaaC', 'mordred_MDEC-14', 'mordred_Xc-3dv', 'mordred_MDEN-33', 'mordred_MINsssCH', 'mordred_MAXsssCH', 'mordred_MAXsssN', 'mordred_MINsssN'])

    log_transform_columns = ['mordred_ABC', 'mordred_ABCGG', 'mordred_nAcid', 'mordred_nBase', 'mordred_SpAbs_A', 'mordred_SpMax_A', 'mordred_SpDiam_A', 'mordred_SpAD_A', 'mordred_SpMAD_A', 'mordred_LogEE_A', 'mordred_VE1_A', 'mordred_VE2_A', 'mordred_VR1_A', 'mordred_VR2_A', 'mordred_nAromAtom', 'mordred_nAromBond', 'mordred_nAtom', 'mordred_nHeavyAtom', 'mordred_nSpiro', 'mordred_nBridgehead', 'mordred_nHetero', 'mordred_nH', 'mordred_nB', 'mordred_nC', 'mordred_nN', 'mordred_nO', 'mordred_nS', 'mordred_nP', 'mordred_nF', 'mordred_nCl', 'mordred_nBr', 'mordred_nI', 'mordred_nX', 'mordred_ATS0dv', 'mordred_ATS1dv', 'mordred_ATS2dv', 'mordred_ATS3dv', 'mordred_ATS4dv', 'mordred_ATS5dv', 'mordred_ATS6dv', 'mordred_ATS7dv', 'mordred_ATS8dv', 'mordred_ATS0d', 'mordred_ATS1d', 'mordred_ATS2d', 'mordred_ATS3d', 'mordred_ATS4d', 'mordred_ATS5d', 'mordred_ATS6d', 'mordred_ATS7d', 'mordred_ATS8d', 'mordred_ATS0Z', 'mordred_ATS1Z', 'mordred_ATS2Z', 'mordred_ATS3Z', 'mordred_ATS4Z', 'mordred_ATS5Z', 'mordred_ATS6Z', 'mordred_ATS7Z', 'mordred_ATS8Z', 'mordred_AATS0dv', 'mordred_AATS1dv', 'mordred_AATS2dv', 'mordred_AATS3dv', 'mordred_AATS4dv', 'mordred_AATS5dv', 'mordred_AATS6dv', 'mordred_AATS7dv', 'mordred_AATS8dv', 'mordred_AATS0d', 'mordred_AATS1d', 'mordred_AATS2d', 'mordred_AATS3d', 'mordred_AATS4d', 'mordred_AATS5d', 'mordred_AATS6d', 'mordred_AATS7d', 'mordred_AATS8d', 'mordred_AATS0Z', 'mordred_AATS1Z', 'mordred_AATS2Z', 'mordred_AATS3Z', 'mordred_AATS4Z', 'mordred_AATS5Z', 'mordred_AATS6Z', 'mordred_AATS7Z', 'mordred_AATS8Z', 'mordred_ATSC0dv', 'mordred_ATSC0d', 'mordred_ATSC0Z', 'mordred_AATSC0dv', 'mordred_AATSC0d', 'mordred_AATSC0Z', 'mordred_GATS1dv', 'mordred_GATS2dv', 'mordred_GATS3dv', 'mordred_GATS4dv', 'mordred_GATS5dv', 'mordred_GATS6dv', 'mordred_GATS7dv', 'mordred_GATS8dv', 'mordred_GATS1d', 'mordred_GATS2d', 'mordred_GATS3d', 'mordred_GATS4d', 'mordred_GATS5d', 'mordred_GATS6d', 'mordred_GATS7d', 'mordred_GATS8d', 'mordred_GATS1Z', 'mordred_GATS2Z', 'mordred_GATS3Z', 'mordred_GATS4Z', 'mordred_GATS5Z', 'mordred_GATS6Z', 'mordred_GATS7Z', 'mordred_GATS8Z', 'mordred_BCUTdv-1h', 'mordred_BCUTd-1h', 'mordred_BCUTZ-1h', 'mordred_BalabanJ', 'mordred_BertzCT', 'mordred_nBonds', 'mordred_nBondsO', 'mordred_nBondsS', 'mordred_nBondsD', 'mordred_nBondsT', 'mordred_nBondsA', 'mordred_nBondsM', 'mordred_nBondsKS', 'mordred_nBondsKD', 'mordred_C1SP1', 'mordred_C2SP1', 'mordred_C1SP2', 'mordred_C2SP2', 'mordred_C3SP2', 'mordred_C1SP3', 'mordred_C2SP3', 'mordred_C3SP3', 'mordred_C4SP3', 'mordred_HybRatio', 'mordred_FCSP3', 'mordred_Xch-3d', 'mordred_Xch-4d', 'mordred_Xch-5d', 'mordred_Xch-6d', 'mordred_Xch-7d', 'mordred_Xch-3dv', 'mordred_Xch-4dv', 'mordred_Xch-5dv', 'mordred_Xch-6dv', 'mordred_Xch-7dv', 'mordred_Xc-3d', 'mordred_Xc-4d', 'mordred_Xc-5d', 'mordred_Xc-6d', 'mordred_Xc-4dv', 'mordred_Xc-5dv', 'mordred_Xc-6dv', 'mordred_Xpc-4d', 'mordred_Xpc-5d', 'mordred_Xpc-6d', 'mordred_Xp-0d', 'mordred_Xp-1d', 'mordred_Xp-2d', 'mordred_Xp-3d', 'mordred_Xp-4d', 'mordred_Xp-5d', 'mordred_Xp-6d', 'mordred_Xp-7d', 'mordred_AXp-0d', 'mordred_AXp-1d', 'mordred_AXp-2d', 'mordred_AXp-3d', 'mordred_AXp-4d', 'mordred_AXp-5d', 'mordred_AXp-6d', 'mordred_AXp-7d', 'mordred_SZ', 'mordred_MZ', 'mordred_SpAbs_Dt', 'mordred_SpMax_Dt', 'mordred_SpDiam_Dt', 'mordred_SpAD_Dt', 'mordred_SpMAD_Dt', 'mordred_LogEE_Dt', 'mordred_SM1_Dt', 'mordred_VE1_Dt', 'mordred_VE2_Dt', 'mordred_VR1_Dt', 'mordred_VR2_Dt', 'mordred_DetourIndex', 'mordred_SpAbs_D', 'mordred_SpMax_D', 'mordred_SpDiam_D', 'mordred_SpAD_D', 'mordred_SpMAD_D', 'mordred_LogEE_D', 'mordred_VE1_D', 'mordred_VE2_D', 'mordred_VR1_D', 'mordred_VR2_D', 'mordred_NsLi', 'mordred_NssBe', 'mordred_NssssBe', 'mordred_NssBH', 'mordred_NsssB', 'mordred_NssssB', 'mordred_NsCH3', 'mordred_NdCH2', 'mordred_NssCH2', 'mordred_NtCH', 'mordred_NdsCH', 'mordred_NaaCH', 'mordred_NsssCH', 'mordred_NddC', 'mordred_NtsC', 'mordred_NdssC', 'mordred_NaasC', 'mordred_NaaaC', 'mordred_NssssC', 'mordred_NsNH3', 'mordred_NsNH2', 'mordred_NssNH2', 'mordred_NdNH', 'mordred_NssNH', 'mordred_NaaNH', 'mordred_NtN', 'mordred_NsssNH', 'mordred_NdsN', 'mordred_NaaN', 'mordred_NsssN', 'mordred_NddsN', 'mordred_NaasN', 'mordred_NssssN', 'mordred_NsOH', 'mordred_NdO', 'mordred_NssO', 'mordred_NaaO', 'mordred_NsF', 'mordred_NsSiH3', 'mordred_NssSiH2', 'mordred_NsssSiH', 'mordred_NssssSi', 'mordred_NsPH2', 'mordred_NssPH', 'mordred_NsssP', 'mordred_NdsssP', 'mordred_NsssssP', 'mordred_NsSH', 'mordred_NdS', 'mordred_NssS', 'mordred_NaaS', 'mordred_NdssS', 'mordred_NddssS', 'mordred_NsCl', 'mordred_NsGeH3', 'mordred_NssGeH2', 'mordred_NsssGeH', 'mordred_NssssGe', 'mordred_NsAsH2', 'mordred_NssAsH', 'mordred_NsssAs', 'mordred_NsssdAs', 'mordred_NsssssAs', 'mordred_NsSeH', 'mordred_NdSe', 'mordred_NssSe', 'mordred_NaaSe', 'mordred_NdssSe', 'mordred_NddssSe', 'mordred_NsBr', 'mordred_NsSnH3', 'mordred_NssSnH2', 'mordred_NsssSnH', 'mordred_NssssSn', 'mordred_NsI', 'mordred_NsPbH3', 'mordred_NssPbH2', 'mordred_NsssPbH', 'mordred_NssssPb', 'mordred_SssBe', 'mordred_SssssBe', 'mordred_SssBH', 'mordred_SssssB', 'mordred_SdCH2', 'mordred_StCH', 'mordred_SdsCH', 'mordred_SddC', 'mordred_StsC', 'mordred_SsNH3', 'mordred_SsNH2', 'mordred_SssNH2', 'mordred_SdNH', 'mordred_SssNH', 'mordred_SaaNH', 'mordred_StN', 'mordred_SsssNH', 'mordred_SdsN', 'mordred_SaaN', 'mordred_SssssN', 'mordred_SsOH', 'mordred_SdO', 'mordred_SssO', 'mordred_SaaO', 'mordred_SsF', 'mordred_SsSiH3', 'mordred_SsPH2', 'mordred_SssPH', 'mordred_SsssssP', 'mordred_SsSH', 'mordred_SdS', 'mordred_SsCl', 'mordred_SsGeH3', 'mordred_SssGeH2', 'mordred_SsssGeH', 'mordred_SsAsH2', 'mordred_SssAsH', 'mordred_SsssAs', 'mordred_SsssdAs', 'mordred_SsssssAs', 'mordred_SsSeH', 'mordred_SdSe', 'mordred_SssSe', 'mordred_SaaSe', 'mordred_SdssSe', 'mordred_SddssSe', 'mordred_SsBr', 'mordred_SsSnH3', 'mordred_SssSnH2', 'mordred_SsssSnH', 'mordred_SsI', 'mordred_SsPbH3', 'mordred_SssPbH2', 'mordred_SsssPbH', 'mordred_SssssPb', 'mordred_MAXssNH', 'mordred_MAXdO', 'mordred_MAXssO', 'mordred_MINssNH', 'mordred_MINdO', 'mordred_MINssO', 'mordred_ECIndex', 'mordred_ETA_beta', 'mordred_AETA_beta', 'mordred_ETA_beta_s', 'mordred_AETA_beta_s', 'mordred_ETA_beta_ns', 'mordred_AETA_beta_ns', 'mordred_ETA_beta_ns_d', 'mordred_AETA_beta_ns_d', 'mordred_ETA_eta_R', 'mordred_AETA_eta_R', 'mordred_ETA_eta_RL', 'mordred_AETA_eta_RL', 'mordred_ETA_epsilon_3', 'mordred_fragCpx', 'mordred_fMF', 'mordred_nHBAcc', 'mordred_nHBDon', 'mordred_IC0', 'mordred_IC1', 'mordred_IC2', 'mordred_IC3', 'mordred_IC4', 'mordred_IC5', 'mordred_TIC0', 'mordred_TIC1', 'mordred_TIC2', 'mordred_TIC3', 'mordred_TIC4', 'mordred_TIC5', 'mordred_SIC0', 'mordred_SIC1', 'mordred_SIC2', 'mordred_SIC3', 'mordred_SIC4', 'mordred_SIC5', 'mordred_BIC0', 'mordred_BIC1', 'mordred_BIC2', 'mordred_BIC3', 'mordred_BIC4', 'mordred_BIC5', 'mordred_CIC0', 'mordred_CIC1', 'mordred_CIC2', 'mordred_MIC0', 'mordred_MIC1', 'mordred_MIC2', 'mordred_MIC3', 'mordred_MIC4', 'mordred_MIC5', 'mordred_ZMIC0', 'mordred_ZMIC1', 'mordred_ZMIC2', 'mordred_ZMIC3', 'mordred_ZMIC4', 'mordred_ZMIC5', 'mordred_Kier1', 'mordred_Kier2', 'mordred_Kier3', 'mordred_Lipinski', 'mordred_GhoseFilter', 'mordred_LabuteASA', 'mordred_PEOE_VSA1', 'mordred_PEOE_VSA2', 'mordred_PEOE_VSA3', 'mordred_PEOE_VSA4', 'mordred_PEOE_VSA5', 'mordred_PEOE_VSA6', 'mordred_PEOE_VSA7', 'mordred_PEOE_VSA8', 'mordred_PEOE_VSA9', 'mordred_PEOE_VSA10', 'mordred_PEOE_VSA11', 'mordred_PEOE_VSA12', 'mordred_PEOE_VSA13', 'mordred_SMR_VSA1', 'mordred_SMR_VSA2', 'mordred_SMR_VSA3', 'mordred_SMR_VSA4', 'mordred_SMR_VSA5', 'mordred_SMR_VSA6', 'mordred_SMR_VSA7', 'mordred_SMR_VSA8', 'mordred_SMR_VSA9', 'mordred_SlogP_VSA1', 'mordred_SlogP_VSA2', 'mordred_SlogP_VSA3', 'mordred_SlogP_VSA4', 'mordred_SlogP_VSA5', 'mordred_SlogP_VSA6', 'mordred_SlogP_VSA7', 'mordred_SlogP_VSA8', 'mordred_SlogP_VSA9', 'mordred_SlogP_VSA10', 'mordred_SlogP_VSA11', 'mordred_EState_VSA1', 'mordred_EState_VSA2', 'mordred_EState_VSA3', 'mordred_EState_VSA4', 'mordred_EState_VSA5', 'mordred_EState_VSA6', 'mordred_EState_VSA7', 'mordred_EState_VSA8', 'mordred_EState_VSA9', 'mordred_EState_VSA10', 'mordred_MDEC-11', 'mordred_MDEC-12', 'mordred_MDEC-13', 'mordred_MDEC-22', 'mordred_MDEC-23', 'mordred_MDEC-24', 'mordred_MDEC-33', 'mordred_MDEC-34', 'mordred_MDEO-11', 'mordred_MDEO-12', 'mordred_MDEO-22', 'mordred_MDEN-22', 'mordred_MID', 'mordred_AMID', 'mordred_MID_h', 'mordred_AMID_h', 'mordred_MID_C', 'mordred_AMID_C', 'mordred_MID_N', 'mordred_AMID_N', 'mordred_MID_O', 'mordred_AMID_O', 'mordred_MID_X', 'mordred_AMID_X', 'mordred_MPC2', 'mordred_MPC3', 'mordred_MPC4', 'mordred_MPC5', 'mordred_MPC6', 'mordred_MPC7', 'mordred_MPC8', 'mordred_MPC9', 'mordred_MPC10', 'mordred_TMPC10', 'mordred_piPC1', 'mordred_piPC2', 'mordred_piPC3', 'mordred_piPC4', 'mordred_piPC5', 'mordred_piPC6', 'mordred_piPC7', 'mordred_piPC8', 'mordred_piPC9', 'mordred_piPC10', 'mordred_TpiPC10', 'mordred_nRing', 'mordred_n3Ring', 'mordred_n4Ring', 'mordred_n5Ring', 'mordred_n6Ring', 'mordred_n7Ring', 'mordred_n8Ring', 'mordred_n9Ring', 'mordred_n10Ring', 'mordred_n11Ring', 'mordred_n12Ring', 'mordred_nG12Ring', 'mordred_nHRing', 'mordred_n3HRing', 'mordred_n4HRing', 'mordred_n5HRing', 'mordred_n6HRing', 'mordred_n7HRing', 'mordred_n8HRing', 'mordred_n9HRing', 'mordred_n10HRing', 'mordred_n11HRing', 'mordred_n12HRing', 'mordred_nG12HRing', 'mordred_naRing', 'mordred_n3aRing', 'mordred_n4aRing', 'mordred_n5aRing', 'mordred_n6aRing', 'mordred_n7aRing', 'mordred_n8aRing', 'mordred_n9aRing', 'mordred_n10aRing', 'mordred_n11aRing', 'mordred_n12aRing', 'mordred_nG12aRing', 'mordred_naHRing', 'mordred_n3aHRing', 'mordred_n4aHRing', 'mordred_n5aHRing', 'mordred_n6aHRing', 'mordred_n7aHRing', 'mordred_n8aHRing', 'mordred_n9aHRing', 'mordred_n10aHRing', 'mordred_n11aHRing', 'mordred_n12aHRing', 'mordred_nG12aHRing', 'mordred_nARing', 'mordred_n3ARing', 'mordred_n4ARing', 'mordred_n5ARing', 'mordred_n6ARing', 'mordred_n7ARing', 'mordred_n8ARing', 'mordred_n9ARing', 'mordred_n10ARing', 'mordred_n11ARing', 'mordred_n12ARing', 'mordred_nG12ARing', 'mordred_nAHRing', 'mordred_n3AHRing', 'mordred_n4AHRing', 'mordred_n5AHRing', 'mordred_n6AHRing', 'mordred_n7AHRing', 'mordred_n8AHRing', 'mordred_n9AHRing', 'mordred_n10AHRing', 'mordred_n11AHRing', 'mordred_n12AHRing', 'mordred_nG12AHRing', 'mordred_nFRing', 'mordred_n4FRing', 'mordred_n5FRing', 'mordred_n6FRing', 'mordred_n7FRing', 'mordred_n8FRing', 'mordred_n9FRing', 'mordred_n10FRing', 'mordred_n11FRing', 'mordred_n12FRing', 'mordred_nG12FRing', 'mordred_nFHRing', 'mordred_n4FHRing', 'mordred_n5FHRing', 'mordred_n6FHRing', 'mordred_n7FHRing', 'mordred_n8FHRing', 'mordred_n9FHRing', 'mordred_n10FHRing', 'mordred_n11FHRing', 'mordred_n12FHRing', 'mordred_nG12FHRing', 'mordred_nFaRing', 'mordred_n4FaRing', 'mordred_n5FaRing', 'mordred_n6FaRing', 'mordred_n7FaRing', 'mordred_n8FaRing', 'mordred_n9FaRing', 'mordred_n10FaRing', 'mordred_n11FaRing', 'mordred_n12FaRing', 'mordred_nG12FaRing', 'mordred_nFaHRing', 'mordred_n4FaHRing', 'mordred_n5FaHRing', 'mordred_n6FaHRing', 'mordred_n7FaHRing', 'mordred_n8FaHRing', 'mordred_n9FaHRing', 'mordred_n10FaHRing', 'mordred_n11FaHRing', 'mordred_n12FaHRing', 'mordred_nG12FaHRing', 'mordred_nFARing', 'mordred_n4FARing', 'mordred_n5FARing', 'mordred_n6FARing', 'mordred_n7FARing', 'mordred_n8FARing', 'mordred_n9FARing', 'mordred_n10FARing', 'mordred_n11FARing', 'mordred_n12FARing', 'mordred_nG12FARing', 'mordred_nFAHRing', 'mordred_n4FAHRing', 'mordred_n5FAHRing', 'mordred_n6FAHRing', 'mordred_n7FAHRing', 'mordred_n8FAHRing', 'mordred_n9FAHRing', 'mordred_n10FAHRing', 'mordred_n11FAHRing', 'mordred_n12FAHRing', 'mordred_nG12FAHRing', 'mordred_nRot', 'mordred_RotRatio', 'mordred_SMR', 'mordred_TopoPSA(NO)', 'mordred_TopoPSA', 'mordred_GGI1', 'mordred_GGI2', 'mordred_GGI3', 'mordred_GGI4', 'mordred_GGI5', 'mordred_GGI6', 'mordred_GGI7', 'mordred_GGI8', 'mordred_GGI9', 'mordred_GGI10', 'mordred_JGI1', 'mordred_JGI2', 'mordred_JGI3', 'mordred_JGI4', 'mordred_JGI5', 'mordred_JGI6', 'mordred_JGI7', 'mordred_JGI8', 'mordred_JGI9', 'mordred_JGI10', 'mordred_JGT10', 'mordred_Diameter', 'mordred_Radius', 'mordred_TopoShapeIndex', 'mordred_PetitjeanIndex', 'mordred_VAdjMat', 'mordred_MWC01', 'mordred_MWC02', 'mordred_MWC03', 'mordred_MWC04', 'mordred_MWC05', 'mordred_MWC06', 'mordred_MWC07', 'mordred_MWC08', 'mordred_MWC09', 'mordred_MWC10', 'mordred_TMWC10', 'mordred_SRW02', 'mordred_SRW03', 'mordred_SRW04', 'mordred_SRW05', 'mordred_SRW06', 'mordred_SRW07', 'mordred_SRW08', 'mordred_SRW09', 'mordred_SRW10', 'mordred_TSRW10', 'mordred_MW', 'mordred_AMW', 'mordred_WPath', 'mordred_WPol', 'mordred_Zagreb1', 'mordred_Zagreb2', 'mordred_mZagreb1', 'mordred_mZagreb2']
    df_mordred_descriptors.loc[:, log_transform_columns] = np.log1p(df_mordred_descriptors.loc[:, log_transform_columns])

    column_missing_counts = df_mordred_descriptors.isnull().sum()
    columns_with_missing_values = column_missing_counts.loc[column_missing_counts > 0].index
    for column in columns_with_missing_values:
        df_mordred_descriptors[column] = df_mordred_descriptors[column].fillna(df_mordred_descriptors[column].median()).fillna(0.)

    with (open(transformer_directory / f'mordred_descriptors_{scaler}.pickle', mode='rb') as f):
        mordred_descriptors_standard_scaler = pickle.load(f)

    df_mordred_descriptors.loc[:, :] = mordred_descriptors_standard_scaler.transform(df_mordred_descriptors)

    features = df_mordred_descriptors.columns.tolist()
    df = pd.concat((
        df,
        df_mordred_descriptors,
    ), axis=1, ignore_index=False)

    return df, features


def concat_mol_graph_conv_features(df, df_mol_graph_conv_features, transformer_directory, scaler='standard_scaler'):

    df_mol_graph_conv_features = df_mol_graph_conv_features.astype(np.float32).fillna(0.)
    df_mol_graph_conv_features.columns = [f'mol_graph_conv_{column}' for column in df_mol_graph_conv_features.columns]

    with (open(transformer_directory / f'mol_graph_conv_features_{scaler}.pickle', mode='rb') as f):
        mol_graph_conv_features_standard_scaler = pickle.load(f)

    df_mol_graph_conv_features.loc[:, :] = mol_graph_conv_features_standard_scaler.transform(df_mol_graph_conv_features)

    features = df_mol_graph_conv_features.columns.tolist()
    df = pd.concat((
        df,
        df_mol_graph_conv_features,
    ), axis=1, ignore_index=False)

    return df, features


def concat_features(df, feature_groups_to_concat, transformer_directory, feature_group_scalers):

    feature_groups = {}

    if 'rdkit_descriptors' in feature_groups_to_concat:
        df, rdkit_descriptors = concat_rdkit_descriptors(df, df_rdkit_descriptors, transformer_directory, feature_group_scalers['rdkit_descriptors'])
        feature_groups['rdkit_descriptors'] = rdkit_descriptors

    if 'estate_fingerprints' in feature_groups_to_concat:
        df, estate_fingerprints = concat_estate_fingerprints(df, df_estate_fingerprints, transformer_directory, feature_group_scalers['estate_fingerprints'])
        feature_groups['estate_fingerprints'] = estate_fingerprints

    if 'maccs_keys' in feature_groups_to_concat:
        df, maccs_keys = concat_maccs_keys(df, df_maccs_keys, transformer_directory, feature_group_scalers['maccs_keys'])
        feature_groups['maccs_keys'] = maccs_keys

    if 'erg_fingerprints' in feature_groups_to_concat:
        df, erg_fingerprints = concat_erg_fingerprints(df, df_erg_fingerprints, transformer_directory, feature_group_scalers['erg_fingerprints'])
        feature_groups['erg_fingerprints'] = erg_fingerprints

    if 'morgan_fingerprints_raw' in feature_groups_to_concat:
        df, morgan_fingerprints_raw = concat_morgan_fingerprints_raw(df, df_morgan_fingerprints_raw, transformer_directory, feature_group_scalers['morgan_fingerprints_raw'])
        feature_groups['morgan_fingerprints_raw'] = morgan_fingerprints_raw

    if 'morgan_fingerprints_ecfp' in feature_groups_to_concat:
        df, morgan_fingerprints_ecfp = concat_morgan_fingerprints_ecfp(df, df_morgan_fingerprints_ecfp, transformer_directory, feature_group_scalers['morgan_fingerprints_ecfp'])
        feature_groups['morgan_fingerprints_ecfp'] = morgan_fingerprints_ecfp

    if 'morgan_fingerprints_fcfp' in feature_groups_to_concat:
        df, morgan_fingerprints_fcfp = concat_morgan_fingerprints_fcfp(df, df_morgan_fingerprints_fcfp, transformer_directory, feature_group_scalers['morgan_fingerprints_fcfp'])
        feature_groups['morgan_fingerprints_fcfp'] = morgan_fingerprints_fcfp

    if 'morgan_fingerprints_rdkit' in feature_groups_to_concat:
        df, morgan_fingerprints_rdkit = concat_morgan_fingerprints_rdkit(df, df_morgan_fingerprints_rdkit, transformer_directory, feature_group_scalers['morgan_fingerprints_rdkit'])
        feature_groups['morgan_fingerprints_rdkit'] = morgan_fingerprints_rdkit

    if 'atom_pairs_raw' in feature_groups_to_concat:
        df, atom_pairs_raw = concat_atom_pairs_raw(df, df_atom_pairs_raw, transformer_directory, feature_group_scalers['atom_pairs_raw'])
        feature_groups['atom_pairs_raw'] = atom_pairs_raw

    if 'atom_pairs_ecfp' in feature_groups_to_concat:
        df, atom_pairs_ecfp = concat_atom_pairs_ecfp(df, df_atom_pairs_ecfp, transformer_directory, feature_group_scalers['atom_pairs_ecfp'])
        feature_groups['atom_pairs_ecfp'] = atom_pairs_ecfp

    if 'atom_pairs_fcfp' in feature_groups_to_concat:
        df, atom_pairs_fcfp = concat_atom_pairs_fcfp(df, df_atom_pairs_fcfp, transformer_directory, feature_group_scalers['atom_pairs_fcfp'])
        feature_groups['atom_pairs_fcfp'] = atom_pairs_fcfp

    if 'atom_pairs_rdkit' in feature_groups_to_concat:
        df, atom_pairs_rdkit = concat_atom_pairs_rdkit(df, df_atom_pairs_rdkit, transformer_directory, feature_group_scalers['atom_pairs_rdkit'])
        feature_groups['atom_pairs_rdkit'] = atom_pairs_rdkit

    if 'topological_torsions_raw' in feature_groups_to_concat:
        df, topological_torsions_raw = concat_topological_torsions_raw(df, df_topological_torsions_raw, transformer_directory, feature_group_scalers['topological_torsions_raw'])
        feature_groups['topological_torsions_raw'] = topological_torsions_raw

    if 'topological_torsions_ecfp' in feature_groups_to_concat:
        df, topological_torsions_ecfp = concat_topological_torsions_ecfp(df, df_topological_torsions_ecfp, transformer_directory, feature_group_scalers['topological_torsions_ecfp'])
        feature_groups['topological_torsions_ecfp'] = topological_torsions_ecfp

    if 'topological_torsions_fcfp' in feature_groups_to_concat:
        df, topological_torsions_fcfp = concat_topological_torsions_fcfp(df, df_topological_torsions_fcfp, transformer_directory, feature_group_scalers['topological_torsions_fcfp'])
        feature_groups['topological_torsions_fcfp'] = topological_torsions_fcfp

    if 'topological_torsions_rdkit' in feature_groups_to_concat:
        df, topological_torsions_rdkit = concat_topological_torsions_rdkit(df, df_topological_torsions_rdkit, transformer_directory, feature_group_scalers['topological_torsions_rdkit'])
        feature_groups['topological_torsions_rdkit'] = topological_torsions_rdkit

    if 'rdkit_fingerprints_raw' in feature_groups_to_concat:
        df, rdkit_fingerprints_raw = concat_rdkit_fingerprints_raw(df, df_rdkit_fingerprints_raw, transformer_directory, feature_group_scalers['rdkit_fingerprints_raw'])
        feature_groups['rdkit_fingerprints_raw'] = rdkit_fingerprints_raw

    if 'rdkit_fingerprints_ecfp' in feature_groups_to_concat:
        df, rdkit_fingerprints_ecfp = concat_rdkit_fingerprints_ecfp(df, df_rdkit_fingerprints_ecfp, transformer_directory, feature_group_scalers['rdkit_fingerprints_ecfp'])
        feature_groups['rdkit_fingerprints_ecfp'] = rdkit_fingerprints_ecfp

    if 'rdkit_fingerprints_fcfp' in feature_groups_to_concat:
        df, rdkit_fingerprints_fcfp = concat_rdkit_fingerprints_fcfp(df, df_rdkit_fingerprints_fcfp, transformer_directory, feature_group_scalers['rdkit_fingerprints_fcfp'])
        feature_groups['rdkit_fingerprints_fcfp'] = rdkit_fingerprints_fcfp

    if 'layered_fingerprints' in feature_groups_to_concat:
        df, layered_fingerprints = concat_layered_fingerprints(df, df_layered_fingerprints, transformer_directory, feature_group_scalers['layered_fingerprints'])
        feature_groups['layered_fingerprints'] = layered_fingerprints

    if 'pattern_fingerprints' in feature_groups_to_concat:
        df, pattern_fingerprints = concat_pattern_fingerprints(df, df_pattern_fingerprints, transformer_directory, feature_group_scalers['pattern_fingerprints'])
        feature_groups['pattern_fingerprints'] = pattern_fingerprints

    if 'avalon_fingerprints' in feature_groups_to_concat:
        df, avalon_fingerprints = concat_avalon_fingerprints(df, df_avalon_fingerprints, transformer_directory, feature_group_scalers['avalon_fingerprints'])
        feature_groups['avalon_fingerprints'] = avalon_fingerprints

    if 'invariants' in feature_groups_to_concat:
        df, invariants = concat_invariants(df, df_invariants, transformer_directory, feature_group_scalers['invariants'])
        feature_groups['invariants'] = invariants

    if 'brics_counts' in feature_groups_to_concat:
        df, brics_counts = concat_brics_counts(df, df_brics_counts, transformer_directory, feature_group_scalers['brics_counts'])
        feature_groups['brics_counts'] = brics_counts

    if 'scaffold_counts' in feature_groups_to_concat:
        df, scaffold_counts = concat_scaffold_counts(df, df_scaffold_counts, transformer_directory, feature_group_scalers['scaffold_counts'])
        feature_groups['scaffold_counts'] = scaffold_counts

    if 'mordred_descriptors' in feature_groups_to_concat:
        df, mordred_descriptors = concat_mordred_descriptors(df, df_mordred_descriptors, transformer_directory, feature_group_scalers['mordred_descriptors'])
        feature_groups['mordred_descriptors'] = mordred_descriptors

    if 'mol_graph_conv_features' in feature_groups_to_concat:
        df, mol_graph_conv_features = concat_mol_graph_conv_features(df, df_mol_graph_conv_features, transformer_directory, feature_group_scalers['mol_graph_conv_features'])
        feature_groups['mol_graph_conv_features'] = mol_graph_conv_features


    return df, feature_groups



def load_sklearn_models(model_directory):

    """
    Load trained sklearn models from given model directory

    Parameters
    ----------
    model_directory: str or pathlib.Path
        Path-like string of the model directory

    Returns
    -------
    config: dict
        Dictionary of model configurations

    models: dict {model_file_name: model}
        Dictionary of model file names as keys and model objects as values
    """

    config_path = model_directory / 'config.yaml'
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    models = {}

    for model_path in tqdm(sorted(list(model_directory.glob('model*')))):
        model_path = str(model_path)
        with open(model_path, mode='rb') as f:
            model = pickle.load(f)
        model_file_name = model_path.split('/')[-1].split('.')[0]
        models[model_file_name] = model
        print(f'Loaded {model.__class__.__name__} from {model_path}')

    return config, models


def sklearn_predict(df, feature_groups, model_name, config, models):

    target_log_transform = config['training']['target_log_transform']
    prediction_min, prediction_max = config['training']['prediction_range']
    features = []
    for feature_group in config['training']['feature_groups']:
        features.extend(feature_groups[feature_group])
        
    prediction_column = f'{model_name}_prediction'
    df[prediction_column] = 0.
    predictions = []

    for model_file_name, model in tqdm(models.items()):

        model_predictions = model.predict(df[features])
        print(f'{model.__class__.__name__} Model {model_file_name} Predictions - Mean: {np.mean(model_predictions):.4f} Std: {np.std(model_predictions):.4f} Min: {np.min(model_predictions):.4f} Max: {np.max(model_predictions):.4f}')

        predictions.append(model_predictions)

    predictions = np.median(np.stack(predictions, axis=1), axis=1)
    df[prediction_column] = predictions

    if target_log_transform:
        df[prediction_column] = np.expm1(df[prediction_column])

    return df



density_ridge_config, density_ridge_models = load_sklearn_models(model_directory=external_dataset_directory / 'Density_ridge')
ffv_ridge_config, ffv_ridge_models = load_sklearn_models(model_directory=external_dataset_directory / 'FFV_ridge')
rg_ridge_config, rg_ridge_models = load_sklearn_models(model_directory=external_dataset_directory / 'Rg_ridge')
tc_ridge_config, tc_ridge_models = load_sklearn_models(model_directory=external_dataset_directory / 'Tc_ridge')
tg_ridge_config, tg_ridge_models = load_sklearn_models(model_directory=external_dataset_directory / 'Tg_ridge')


df_density_linear, feature_groups_density_linear = concat_features(
    df,
    feature_groups_to_concat=density_ridge_config['training']['feature_groups'],
    transformer_directory=external_dataset_directory / 'scalers',
    feature_group_scalers=density_ridge_config['training']['feature_group_scalers']
)
print(f'Density Linear Dataset Shape: {df_density_linear.shape}')

df_density_linear = sklearn_predict(
    df=df_density_linear,
    feature_groups=feature_groups_density_linear,
    model_name='Density_ridge',
    config=density_ridge_config,
    models=density_ridge_models
)
density_ridge_predictions = df_density_linear['Density_ridge_prediction']
del df_density_linear, feature_groups_density_linear

df_ffv_linear, feature_groups_ffv_linear = concat_features(
    df,
    feature_groups_to_concat=ffv_ridge_config['training']['feature_groups'],
    transformer_directory=external_dataset_directory / 'scalers',
    feature_group_scalers=ffv_ridge_config['training']['feature_group_scalers']
)
print(f'FFV Linear Dataset Shape: {df_ffv_linear.shape}')

df_ffv_linear = sklearn_predict(
    df=df_ffv_linear,
    feature_groups=feature_groups_ffv_linear,
    model_name='FFV_ridge',
    config=ffv_ridge_config,
    models=ffv_ridge_models
)
ffv_ridge_predictions = df_ffv_linear['FFV_ridge_prediction']
del df_ffv_linear, feature_groups_ffv_linear

df_rg_linear, feature_groups_rg_linear = concat_features(
    df,
    feature_groups_to_concat=rg_ridge_config['training']['feature_groups'],
    transformer_directory=external_dataset_directory / 'scalers',
    feature_group_scalers=rg_ridge_config['training']['feature_group_scalers']
)
print(f'Rg Linear Dataset Shape: {df_rg_linear.shape}')

df_rg_linear = sklearn_predict(
    df=df_rg_linear,
    feature_groups=feature_groups_rg_linear,
    model_name='Rg_ridge',
    config=rg_ridge_config,
    models=rg_ridge_models
)
rg_ridge_predictions = df_rg_linear['Rg_ridge_prediction']
del df_rg_linear, feature_groups_rg_linear

df_tc_linear, feature_groups_tc_linear = concat_features(
    df,
    feature_groups_to_concat=tc_ridge_config['training']['feature_groups'],
    transformer_directory=external_dataset_directory / 'scalers',
    feature_group_scalers=tc_ridge_config['training']['feature_group_scalers']
)
print(f'Tc Linear Dataset Shape: {df_tc_linear.shape}')

df_tc_linear = sklearn_predict(
    df=df_tc_linear,
    feature_groups=feature_groups_tc_linear,
    model_name='Tc_ridge',
    config=tc_ridge_config,
    models=tc_ridge_models
)
tc_ridge_predictions = df_tc_linear['Tc_ridge_prediction']
del df_tc_linear, feature_groups_tc_linear

df_tg_linear, feature_groups_tg_linear = concat_features(
    df,
    feature_groups_to_concat=tg_ridge_config['training']['feature_groups'],
    transformer_directory=external_dataset_directory / 'scalers',
    feature_group_scalers=tg_ridge_config['training']['feature_group_scalers']
)
print(f'Tg Linear Dataset Shape: {df_tg_linear.shape}')

df_tg_linear = sklearn_predict(
    df=df_tg_linear,
    feature_groups=feature_groups_tg_linear,
    model_name='Tg_ridge',
    config=tg_ridge_config,
    models=tg_ridge_models
)
tg_ridge_predictions = df_tg_linear['Tg_ridge_prediction']
del df_tg_linear, feature_groups_tg_linear


df['Tg_ridge_prediction'] = (tg_ridge_predictions * 9 / 5) + 32
df['FFV_ridge_prediction'] = ffv_ridge_predictions
df['Tc_ridge_prediction'] = tc_ridge_predictions
df['Density_ridge_prediction'] = density_ridge_predictions
df['Rg_ridge_prediction'] = rg_ridge_predictions


prediction_column_mapping = {
    'Tg_ridge_prediction': 'Tg',
    'FFV_ridge_prediction': 'FFV',
    'Tc_ridge_prediction': 'Tc',
    'Density_ridge_prediction': 'Density',
    'Rg_ridge_prediction': 'Rg',
}

df = df.rename(columns=prediction_column_mapping)
display(df)


df_submission = df.loc[:, ['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
df_submission.to_csv('submission.csv')
display(df_submission)




