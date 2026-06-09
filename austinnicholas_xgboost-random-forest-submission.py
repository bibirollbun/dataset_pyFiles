import pandas as pd
input_path = "/kaggle/input/neurips-open-polymer-prediction-2025/test.csv"
def load_polymer_data(csv_path=input_path):
    """
    Loads polymer dataset and returns a cleaned DataFrame with relevant columns.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with selected columns.
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()  # Remove accidental whitespace

        selected_columns = ['id', 'SMILES']
        missing = [col for col in selected_columns if col not in df.columns]

        if missing:
            raise KeyError(f"Missing expected columns in the dataset: {missing}")

        # Drop only rows missing SMILES or Tc if you want more flexibility
        return df[selected_columns].dropna(subset=['SMILES'])

    except FileNotFoundError:
        print(f"â�Œ File not found at: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"â�Œ An error occurred while loading the dataset: {e}")
        return pd.DataFrame()
df = load_polymer_data()



from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors

def kappa3(mol):
    if mol is None:
        return None
    try:
        return rdMolDescriptors.CalcKappa3(mol)
    except Exception:
        return None

def count_branch_points(mol):
    if mol is None:
        return None
    branch_points = 0
    for atom in mol.GetAtoms():
        heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_neighbors) > 2:
            branch_points += 1
    return branch_points

def count_atoms_including_hydrogens(mol):
    if mol is None:
        return None
    mol_with_H = Chem.AddHs(mol)
    return mol_with_H.GetNumAtoms()

def calculate_molar_mass(mol):
    if mol is None:
        return None
    return Descriptors.MolWt(mol)

def molar_mass_per_atom(mol):
    if mol is None:
        return None
    molar_mass = Descriptors.MolWt(mol)
    atom_count = count_atoms_including_hydrogens(mol)
    if atom_count and atom_count > 0:
        return molar_mass / atom_count
    else:
        return None

def count_heavy_atoms(mol):
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()

def molar_mass_per_heavy_atom(mol):
    if mol is None:
        return None
    molar_mass = Descriptors.MolWt(mol)
    heavy_atom_count = mol.GetNumHeavyAtoms()
    if heavy_atom_count == 0:
        return None
    return molar_mass / heavy_atom_count

def percent_hydrogen(mol):
    if mol is None:
        return None
    mol_with_H = Chem.AddHs(mol)
    total_atoms = mol_with_H.GetNumAtoms()
    num_hydrogens = sum(1 for atom in mol_with_H.GetAtoms() if atom.GetAtomicNum() == 1)
    if total_atoms == 0:
        return None
    return (num_hydrogens / total_atoms) * 100

def count_hydrogen_donors(mol):
    if mol is None:
        raise ValueError("Invalid molecule")
    mol = Chem.AddHs(mol)
    donor_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 1 and neighbors[0].GetAtomicNum() in [7, 8, 16]:
                donor_count += 1
    return donor_count

def count_hydrogen_acceptors_with_fluorine(mol):
    if mol is None:
        raise ValueError("Invalid molecule")
    acceptor_atoms = {7, 8, 9, 16}
    acceptor_count = 0
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num in acceptor_atoms:
            num_h = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)
            if num_h < 2:
                acceptor_count += 1
    return acceptor_count

def count_rotatable_bonds(mol):
    if mol is None:
        raise ValueError("Invalid molecule")
    return Descriptors.NumRotatableBonds(mol)

def count_double_bonds(mol):
    if mol is None:
        raise ValueError("Invalid molecule")
    double_bond_count = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            double_bond_count += 1
    return double_bond_count

def count_rings(mol):
    if mol is None:
        raise ValueError("Invalid molecule")
    ring_info = mol.GetRingInfo()
    return ring_info.NumRings()

def count_aromatic_rings(mol):
    if mol is None:
        raise ValueError("Invalid molecule")
    ring_info = mol.GetRingInfo()
    aromatic_rings = []
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings.append(set(ring))
    return len(aromatic_rings)

def average_side_chain_mw(mol):
    if mol is None:
        raise ValueError("Invalid molecule")
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    if len(dummy_indices) != 2:
        return 0.0
    backbone_path = Chem.rdmolops.GetShortestPath(mol, dummy_indices[0], dummy_indices[1])
    backbone_atoms = set(backbone_path)

    def get_fragment_atoms(start_idx, exclude_atoms):
        to_visit = [start_idx]
        fragment = set()
        while to_visit:
            current = to_visit.pop()
            fragment.add(current)
            for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in fragment and nbr_idx not in exclude_atoms:
                    to_visit.append(nbr_idx)
        return fragment

    side_chain_mws = []
    visited_atoms = set(backbone_atoms)

    for idx in backbone_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in visited_atoms:
                fragment_atoms = get_fragment_atoms(nbr_idx, backbone_atoms)
                visited_atoms.update(fragment_atoms)
                submol = Chem.PathToSubmol(mol, list(fragment_atoms))
                side_chain_mws.append(Descriptors.MolWt(submol))

    if not side_chain_mws:
        return 0.0
    return sum(side_chain_mws) / len(side_chain_mws)

def number_of_side_chains(mol):
    if mol is None:
        return 0
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    if len(dummy_indices) != 2:
        return 0
    backbone_path = Chem.rdmolops.GetShortestPath(mol, dummy_indices[0], dummy_indices[1])
    backbone_atoms = set(backbone_path)

    def get_fragment_atoms(start_idx, exclude_atoms):
        to_visit = [start_idx]
        fragment = set()
        while to_visit:
            current = to_visit.pop()
            fragment.add(current)
            for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in fragment and nbr_idx not in exclude_atoms:
                    to_visit.append(nbr_idx)
        return fragment

    visited = set(backbone_atoms)
    count = 0
    for idx in backbone_atoms:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in visited:
                fragment_atoms = get_fragment_atoms(nbr_idx, backbone_atoms)
                visited.update(fragment_atoms)
                count += 1
    return count

def max_side_chain_mw(mol):
    if mol is None:
        return 0.0
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    if len(dummy_indices) != 2:
        return 0.0
    backbone_path = Chem.rdmolops.GetShortestPath(mol, dummy_indices[0], dummy_indices[1])
    backbone_atoms = set(backbone_path)

    def get_fragment_atoms(start_idx, exclude_atoms):
        to_visit = [start_idx]
        fragment = set()
        while to_visit:
            current = to_visit.pop()
            fragment.add(current)
            for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in fragment and nbr_idx not in exclude_atoms:
                    to_visit.append(nbr_idx)
        return fragment

    visited = set(backbone_atoms)
    max_mw = 0.0
    for idx in backbone_atoms:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in visited:
                fragment_atoms = get_fragment_atoms(nbr_idx, backbone_atoms)
                visited.update(fragment_atoms)
                submol = Chem.PathToSubmol(mol, list(fragment_atoms))
                mw = Descriptors.MolWt(submol)
                if mw > max_mw:
                    max_mw = mw
    return max_mw

def backbone_length(mol):
    if mol is None:
        return 0
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    if len(dummy_indices) != 2:
        return 0
    backbone_path = Chem.rdmolops.GetShortestPath(mol, dummy_indices[0], dummy_indices[1])
    return len(backbone_path)



headers_list =['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg', 'RotRatio', 'Percent_Hydrogen', 'SssCH2', 'VSA_EState7', 'NssCH2', 'AATS0d', 'CIC3', 'BIC2', 'SIC1', 
                    'GATS2Z', 'SIC2', 'AATS1dv', 'C2SP3', 'SMR_VSA5', 'AATSC0d', 'SlogP_VSA5', 'CIC2', 'AATS1d', 
                    'BIC1', 'AMW', 'MassPerAtom', 'CIC1', 'AATS2d', 'MZ', 'IC1', 'CIC4', 'MIC1', 'IC0', 'AMID_h', 'AETA_beta_s', 'AATSC0dv', 
                    'JGT10', 'AATS1Z', 'nH', 'AMID_C', 'AATS0dv', 'EState_VSA5', 'GATS1Z', 'AATS2dv', 'RotatableBonds', 'nRot', 
                    'ATSC2Z', 'FilterItLogS', 'JGI2', 'AXp-7d']


import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, EState, QED, Crippen, Lipinski, rdMolDescriptors,
    GraphDescriptors, Fragments, MolSurf
)
import inspect
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler

# NEW: Mordred imports
from mordred import Calculator, descriptors as mordred_descriptors


# STEP 3: Your original custom + RDKit descriptors
custom_feature_functions = {
    'Percent_Hydrogen': percent_hydrogen,
    'MassPerAtom': molar_mass_per_atom,
    'RotatableBonds': count_rotatable_bonds,
}




descriptor_list = [
    'SMR_VSA5', 'SlogP_VSA5', 'EState_VSA5', 'RotatableBonds'
]




# --- Lists Provided ---
first_order_list = ['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg', 'RotRatio', 'Percent_Hydrogen', 'SssCH2', 'VSA_EState7', 'NssCH2', 'AATS0d', 'CIC3', 'BIC2', 'SIC1', 
                    'GATS2Z', 'SIC2', 'AATS1dv', 'C2SP3', 'SMR_VSA5', 'AATSC0d', 'SlogP_VSA5', 'CIC2', 'AATS1d', 
                    'BIC1', 'AMW', 'MassPerAtom', 'CIC1', 'AATS2d', 'MZ', 'IC1', 'CIC4', 'MIC1', 'IC0', 'AMID_h', 'AETA_beta_s', 'AATSC0dv', 
                    'JGT10', 'AATS1Z', 'nH', 'AMID_C', 'AATS0dv', 'EState_VSA5', 'GATS1Z', 'AATS2dv', 'RotatableBonds', 'nRot', 
                    'ATSC2Z', 'FilterItLogS', 'JGI2', 'AXp-7d']

second_order_list = ['AATS0d_x_BIC2', 'BIC2_x_AATS1d', 'AATS0d_x_SIC2', 'SIC2_x_AATS1d', 'BIC2_x_AATS2d', 'AATS0d_x_BIC1', 'AATS1d_x_BIC1', 'Percent_Hydrogen_div_SIC2', 'SIC1_x_AATS1d', 'Percent_Hydrogen_div_BIC2', 'AATS0d_x_SIC1', 'SIC2_x_AETA_beta_s', 'SIC2_div_AXp-2d', 'GATS2Z_div_SIC2', 'SIC2_x_SpDiam_A', 'AATS3d_x_BIC0', 'SIC2_div_AXp-1d', 'BIC2_div_AXp-2d', 'SIC2_x_SpMax_A', 'BIC1_x_AATS2d', 'SIC2_x_AATS2d', 'Percent_Hydrogen_div_SIC1', 'Percent_Hydrogen_div_BIC1', 'AATS3d_x_SIC0', 'RotRatio_div_MolarMass_per_HeavyAtom', 'AATS2d_x_BIC0', 'BIC1_div_AXp-2d', 'Percent_Hydrogen_div_AATSC0d', 'AATS2d_x_SIC0', 'SIC2_x_MZ', 'SIC1_div_AXp-2d', 'RotRatio_div_AATSC0Z', 'CIC1_div_AATS2d', 'SIC1_x_SpDiam_A', 'SIC1_x_AATS2d', 'JGT10_x_AETA_beta', 'SIC1_x_SpMax_A', 'BIC2_div_AXp-1d', 'Percent_Hydrogen_x_CIC1', 'IC0_div_AXp-2d', 'SIC2_x_AMW', 'GATS2Z_x_CIC1', 'SIC1_x_AETA_beta_s', 'GATS2Z_div_BIC1', 'SIC2_x_MassPerAtom', 'BIC2_x_AATS3d', 'BIC2_x_SpMax_A', 'BIC2_x_SpDiam_A', 'SIC1_div_AXp-1d', 'CIC2_div_AATS2d', 'RotRatio_div_AATS1Z', 'SIC2_x_AATS1dv', 'SIC1_x_AATS1dv', 'BIC2_x_MZ', 'RotRatio_x_SpMAD_A', 'RotRatio_div_AXp-0d', 'Percent_Hydrogen_x_CIC2', 'CIC1_x_GATS1Z', 'SIC2_x_AATSC0d', 'BIC1_x_SpDiam_A', 'AATS0d_x_IC0', 'NssCH2_div_AtomCount', 'NssCH2_div_nAtom', 'AATS1dv_x_BIC1', 'BIC2_x_AETA_beta_s', 'BIC1_x_SpMax_A', 'NssCH2_div_nBondsKS', 'BIC2_x_AATS1dv', 'GATS2Z_div_SIC0', 'JGT10_x_AATS3d', 'CIC2_div_AATS1d', 'NssCH2_div_nBondsS', 'Percent_Hydrogen_x_GATS2Z', 'BIC1_x_AATS3d', 'RotRatio_x_AMID', 'NssCH2_div_mZagreb2', 'GATS2Z_x_CIC2', 'RotRatio_x_SRW02', 'RotRatio_x_AETA_eta_RL', 'NssCH2_div_nBonds', 'NssCH2_div_MWC01', 'NssCH2_div_nBondsO', 'RotRatio_x_VAdjMat', 'Percent_Hydrogen_div_SIC0', 'RotRatio_div_MZ', 'RotRatio_div_AATS0Z', 'BIC1_div_AXp-1d', 'Percent_Hydrogen_div_IC1', 'IC0_x_SpDiam_A', 'BIC2_x_AMW', 'CIC1_div_MZ', 'SssCH2_div_mZagreb2', 'SIC1_x_AATSC0d', 'CIC2_x_GATS1Z', 'NssCH2_div_CIC0', 'RotatableBonds_div_mZagreb2', 'nRot_div_mZagreb2', 'RotRatio_x_ETA_epsilon_3', 'BIC2_x_MassPerAtom', 'Percent_Hydrogen_div_IC0']





import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, EState, QED, Crippen, Lipinski, rdMolDescriptors, GraphDescriptors, Fragments, MolSurf
from mordred import Calculator, descriptors as mordred_descriptors
import inspect
from sklearn.preprocessing import normalize

def safe_apply_mol(func):
    def wrapped(mol):
        try:
            if mol is None:
                return np.nan
            return func(mol)
        except Exception:
            return np.nan
    return wrapped
def parse_molecules(df_raw):
    df = df_raw.copy()
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    print("ğŸ”� After molecule parsing:")
    print(df.head())
    return df
def get_descriptor_functions():
    descriptor_sources = [Descriptors, EState, QED, Crippen, Lipinski, rdMolDescriptors, GraphDescriptors, Fragments, MolSurf]
    descriptor_funcs = {}
    for mod in descriptor_sources:
        funcs = dict(inspect.getmembers(mod, inspect.isfunction))
        descriptor_funcs.update(funcs)
    return descriptor_funcs
def compute_mordred_descriptors(df):
    print("ğŸ§ª Calculating Mordred descriptors...")
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    mordred_df = calc.pandas(df['mol'])
    return mordred_df
def drop_duplicate_columns(df):
    """
    Drops duplicate columns from a DataFrame based on column names,
    keeping the first occurrence.
    """
    print("ğŸ“› Checking for duplicate columns...")
    before = df.shape[1]
    df = df.loc[:, ~df.columns.duplicated()]
    after = df.shape[1]

    if before != after:
        print(f"âœ… Removed {before - after} duplicate column(s).")
    else:
        print("ğŸ‘Œ No duplicate columns found.")
    
    return df
def normalize_columns(df, cols_to_process):
    for col in cols_to_process:
        series = df[col]

        # Ensure it's a Series
        if isinstance(series, pd.DataFrame):
            raise ValueError(f"Expected a Series for column '{col}', got DataFrame. Check for duplicate column names.")

        col_min = series.min()
        col_max = series.max()

        if pd.isna(col_min) or pd.isna(col_max):
            df[col] = 0.0
        elif col_max != col_min:
            df[col] = ((series - col_min) / (col_max - col_min)) * 2 - 1
        else:
            df[col] = 0.0

    return df
def finalize_l2_normalized_df(df, excluded_cols, cols_to_process):
    df_features = pd.DataFrame(
        normalize(df[cols_to_process], norm='l2'),
        columns=cols_to_process,
        index=df.index
    )
    df_final = pd.concat([df[excluded_cols].reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)
    print("âœ… Final dataset after L2 normalization:")
    print(df_final.head())
    return df_final
def clean_numeric_columns(df, excluded_cols):
    cols_to_process = [col for col in df.columns if col not in excluded_cols]
    for col in cols_to_process:
        if col in df.columns and df[col].ndim == 1:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].mean())
        else:
            print(f"Skipping invalid column: {col}")
    print("ğŸ§¼ After filling NaNs and coercing to numeric:")
    print(df.head())
    return df, cols_to_process
def select_features(df, df_raw, headers_list):
    df = df[[col for col in headers_list if col in df.columns]]

    print("ğŸ“� After selecting only headers_list features:")
    print(df.head())

    excluded_cols = [col for col in ['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg'] if col in df_raw.columns]
    for col in excluded_cols:
        df[col] = df_raw[col]
    return df, excluded_cols
def compute_descriptors(df, mordred_df, headers_list, descriptor_funcs, custom_feature_functions):
    custom_set = set(custom_feature_functions)
    rdkit_set = set(descriptor_funcs)
    mordred_set = set(mordred_df.columns)

    custom_to_compute = [f for f in headers_list if f in custom_set]
    rdkit_to_compute = [f for f in headers_list if f in rdkit_set]
    mordred_to_keep = [f for f in headers_list if f in mordred_set]

    for feat in custom_to_compute:
        df[feat] = df['mol'].apply(safe_apply_mol(custom_feature_functions[feat]))
    for feat in rdkit_to_compute:
        df[feat] = df['mol'].apply(safe_apply_mol(descriptor_funcs[feat]))
    
    df = pd.concat([df.reset_index(drop=True), mordred_df[mordred_to_keep].reset_index(drop=True)], axis=1)
    print("ğŸ“Š After descriptor computation:")
    print(df.head())
    return df



# Load your raw CSV
df_raw = pd.read_csv(input_path)

# Your headers list and custom descriptors dictionary must be defined somewhere above
# Example:
# headers_list = [...]  
# custom_feature_functions = {...}

# Step 1: Parse molecules
df = parse_molecules(df_raw)

# Step 2: Get RDKit descriptor functions
descriptor_funcs = get_descriptor_functions()

# Step 3: Compute Mordred descriptors
mordred_df = compute_mordred_descriptors(df)

# Step 4: Compute all descriptors (custom + RDKit + Mordred)
df = compute_descriptors(df, mordred_df, headers_list, descriptor_funcs, custom_feature_functions)

# Step 5: Filter only headers_list + restore original metadata columns
df, excluded_cols = select_features(df, df_raw, headers_list)

# Step 6: Fill NaNs, coerce to numeric
df, cols_to_process = clean_numeric_columns(df, excluded_cols)

df = drop_duplicate_columns(df)

# Step 7: Normalize to [-1, 1]
df = normalize_columns(df, cols_to_process)

# Step 8: Final L2 normalization
df_final = finalize_l2_normalized_df(df, excluded_cols, cols_to_process)




# Step 4: Save to CSV
output_path = "descriptors.csv"
df.to_csv(output_path, index=False)



import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class BoostedXGBoostRFModel:
    def __init__(self, feature_order, params=None):
        self.feature_order = feature_order
        self.model = None
        self.target_column = None

        # Default parameters for GPU-accelerated Random Forest + regularization
        # --- Parameters ---
        default_params = {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'verbosity': 1,
            'max_depth': 7,
            'min_child_weight': 5,
            'subsample': 1.0,
            'colsample_bynode': 0.4,
            'eta': 0.01,
            'gamma': 0.0,
            'alpha': 0.373,
            'lambda': 10.0,
            'grow_policy': 'depthwise',
            'num_parallel_tree': 13,
            'seed': 42
        }

        # Use user params if provided
        self.params = params if params is not None else default_params

    def fit(self, df, target_column, validation_split=0.2, early_stopping_rounds=10):
        self.target_column = target_column

        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[target_column])
        df = df.reset_index(drop=True)

        # Split train/val
        n_total = len(df)
        n_val = int(n_total * validation_split)
        train_df = df.iloc[:n_total - n_val]
        val_df = df.iloc[n_total - n_val:]

        X_train = train_df[self.feature_order].astype(np.float32)
        y_train = train_df[target_column].astype(np.float32)
        X_val = val_df[self.feature_order].astype(np.float32)
        y_val = val_df[target_column].astype(np.float32)

        dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
        dval = xgb.DMatrix(X_val, label=y_val, missing=np.nan)

        evals = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True
        )

        best_iter = self.model.best_iteration
        best_score = self.model.best_score
        val_preds = self.model.predict(dval, iteration_range=(0, best_iter))
        val_rmse = mean_squared_error(y_val, val_preds, squared=False)

        print(f"\nâœ… Training complete for target: '{target_column}'")
        print(f"ğŸ”� Best iteration: {best_iter}")
        print(f"ğŸ�¯ Best validation RMSE (from eval metric): {best_score:.5f}")
        print(f"ğŸ§ª Final validation RMSE (manual calc): {val_rmse:.5f}")

        return self.model

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_order].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0)
        dmatrix = xgb.DMatrix(X)
        preds = self.model.predict(dmatrix, iteration_range=(0, self.model.best_iteration))
        return pd.DataFrame({f"{self.target_column}_predictions": preds}, index=df.index)



import os
import xgboost as xgb

# Folder where models are stored
load_dir = "/kaggle/input/7.2.25/other/default/1"
timestamp = "2025-07-02_16-09"

# Targets to load
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Assuming feature_functions is defined elsewhere and is a dict of your features

# Ensure all columns in headers_list are valid and exist in df
original_features = [col for col in df.columns[2:] if col in headers_list]

# Dictionary to hold loaded models
models = {}

for target in targets:
    # Initialize your custom wrapper model (you may want to pass params if needed)
    model_wrapper = BoostedXGBoostRFModel(feature_order=original_features)
    model_wrapper.target_column = target  # Keep track of which target it's for

    # Construct the full path for the model file
    model_path = os.path.join(load_dir, f"{target}_model_{timestamp}.json")

    # Load the booster model from JSON
    booster = xgb.Booster()
    booster.load_model(model_path)

    # Assign the booster inside your wrapper
    model_wrapper.model = booster

    # Store the wrapped model in dictionary
    models[target] = model_wrapper

    print(f"âœ… Loaded model for target '{target}' from {model_path}")





# Predict for all targets, keep original prediction DataFrames as is,
# but also create new DataFrames with columns renamed to "Predicted_<target>"
predictions = {}

for target in targets:
    pred_df = models[target].predict(df)
    # Save the original prediction DataFrame (unchanged)
    predictions[target] = pred_df
    print(f"\nPredictions for {target}:")
    print(pred_df.head())

# Now create renamed prediction DataFrames for evaluation with your function
renamed_predictions = {}

for target in targets:
    renamed_predictions[target] = predictions[target].rename(
        columns={f"{target}_predictions": f"Predicted_{target}"}
    )




# Unpack renamed predictions for all targets
Tg_df = renamed_predictions['Tg']
FFV_df = renamed_predictions['FFV']
Tc_df = renamed_predictions['Tc']
Density_df = renamed_predictions['Density']
Rg_df = renamed_predictions['Rg']

# Print to verify
print("Tg_df:")
print(Tg_df.head(), end="\n\n")

print("FFV_df:")
print(FFV_df.head(), end="\n\n")

print("Tc_df:")
print(Tc_df.head(), end="\n\n")

print("Density_df:")
print(Density_df.head(), end="\n\n")

print("Rg_df:")
print(Rg_df.head())





def combine_predictions_with_id(df, Tg_df, FFV_df, Tc_df, Density_df, Rg_df):
    """
    Combine prediction columns from separate DataFrames into one DataFrame
    along with the 'id' column from the original df.

    Parameters:
        df (pd.DataFrame): Original DataFrame with an 'id' column.
        Tg_df, FFV_df, Tc_df, Density_df, Rg_df (pd.DataFrame): DataFrames with 'Predicted_<target>' columns.

    Returns:
        pd.DataFrame: Combined DataFrame with columns ['Tg'].
    """
    combined_df = pd.DataFrame({
        'id': df['id'].values,
        'Tg': Tg_df['Predicted_Tg'].values,
        'FFV': FFV_df['Predicted_FFV'].values,
        'Tc': Tc_df['Predicted_Tc'].values,
        'Density': Density_df['Predicted_Density'].values,
        'Rg': Rg_df['Predicted_Rg'].values
    })
   
    return combined_df





# Combine predictions
combined_predictions_df = combine_predictions_with_id(
    df,
    Tg_df,
    FFV_df,
    Tc_df,
    Density_df,
    Rg_df
)

# Save to CSV
#combined_predictions_df.to_csv("/mnt/c/Users/Smack/Documents/machine learning practice/chemistry/Polymer Modeling/submission15.csv", index=False)

# Hard-coded normalization stats (mean, std) for each target
normalization_stats = {
    'Tg': (96.4523136840, 111.2282791080),
    'FFV': (0.3672119955, 0.0296087791),
    'Tc': (0.2563340925, 0.0895378496),
    'Density': (0.9854843785, 0.1461891539),
    'Rg': (16.4197867095, 4.6086400865),
}

# Example combined predictions DataFrame (replace with your actual combined_df)
# combined_df = combine_predictions_with_id(df, Tg_predictions_df, FFV_predictions_df, Tc_predictions_df, Density_predictions_df, Rg_predictions_df)

# Undo normalization for each target column
for target in normalization_stats:
    mean, std = normalization_stats[target]
    combined_predictions_df[target] = combined_predictions_df[target] * std + mean

print(combined_predictions_df.head())




# Save to CSV
combined_predictions_df.to_csv("submission.csv", index=False)



