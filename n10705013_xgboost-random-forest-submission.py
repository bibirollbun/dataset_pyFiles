#MODE = "original"  
MODE = "scaler_fixed"  
# ADJUST_TG = None
#ADJUST_TG = "model_descriptors"
ADJUST_TG = "mordred"



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
        print(f"❌ File not found at: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ An error occurred while loading the dataset: {e}")
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

# Load your base polymer dataset
df = load_polymer_data()

# ========== STEP 1: Define Original Custom Feature Functions ==========
custom_feature_functions = {
    'NumberOfSideChains': number_of_side_chains,
    'MaxSideChainMW': max_side_chain_mw,
    'BackboneLength': backbone_length,
    'AverageSideChainMW': average_side_chain_mw,
    'BranchPoints': count_branch_points,
    'AtomCount': count_atoms_including_hydrogens,
    'MassPerAtom': molar_mass_per_atom,
    'MolarMass_per_HeavyAtom': molar_mass_per_heavy_atom,
    'Percent_Hydrogen': percent_hydrogen,
    'HydrogenDonors': count_hydrogen_donors,
    'HydrogenAcceptorsWithF': count_hydrogen_acceptors_with_fluorine,
    'RotatableBonds': count_rotatable_bonds,
    'DoubleBonds': count_double_bonds,
    'AromaticRingCount': count_aromatic_rings,
    'MolarMass': calculate_molar_mass,
}

def safe_apply_mol(func):
    def wrapped(mol):
        try:
            if mol is None:
                return np.nan
            return func(mol)
        except Exception:
            return np.nan
    return wrapped

df = df.copy()
df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)

# ---- STEP 2: Prepare descriptor functions that accept Mol directly ----
descriptor_sources = [
    Descriptors, EState, QED, Crippen, Lipinski, rdMolDescriptors,
    GraphDescriptors, Fragments, MolSurf
]

descriptor_funcs = {}
for mod in descriptor_sources:
    funcs = dict(inspect.getmembers(mod, inspect.isfunction))
    descriptor_funcs.update(funcs)

descriptor_list = '''
MaxAbsEStateIndex MaxEStateIndex MinAbsEStateIndex MinEStateIndex qed SPS HeavyAtomMolWt ExactMolWt
NumValenceElectrons FpDensityMorgan1 FpDensityMorgan2 FpDensityMorgan3 
Kappa1 Kappa2 Kappa3 LabuteASA PEOE_VSA1 PEOE_VSA10 PEOE_VSA12 PEOE_VSA13 PEOE_VSA14 PEOE_VSA2 PEOE_VSA3
PEOE_VSA4 PEOE_VSA5 PEOE_VSA6 PEOE_VSA7 PEOE_VSA8 PEOE_VSA9 SMR_VSA1 SMR_VSA10 SMR_VSA2 SMR_VSA3 SMR_VSA4
SMR_VSA5 SMR_VSA6 SMR_VSA7 SMR_VSA9 SlogP_VSA1 SlogP_VSA10 SlogP_VSA11 SlogP_VSA12 SlogP_VSA2 SlogP_VSA3
SlogP_VSA4 SlogP_VSA5 SlogP_VSA6 SlogP_VSA7 SlogP_VSA8 SlogP_VSA9 TPSA EState_VSA1 EState_VSA10
EState_VSA11 EState_VSA2 EState_VSA3 EState_VSA4 EState_VSA5 EState_VSA6 EState_VSA7 EState_VSA8
EState_VSA9 VSA_EState1 VSA_EState10 VSA_EState2 VSA_EState3 VSA_EState4 VSA_EState5 VSA_EState6
VSA_EState7 VSA_EState8 VSA_EState9 FractionCSP3 HeavyAtomCount NHOHCount NOCount NumAliphaticCarbocycles
NumAliphaticHeterocycles NumAliphaticRings NumAromaticCarbocycles NumAromaticHeterocycles NumAromaticRings
NumHAcceptors NumHDonors NumHeteroatoms NumRotatableBonds NumSaturatedCarbocycles NumSaturatedHeterocycles
NumSaturatedRings RingCount MolLogP MolMR fr_C_O fr_C_O_noCOO fr_NH0 fr_NH1 fr_amide fr_amidine fr_aniline
fr_benzene fr_bicyclic fr_ester fr_ether fr_unbrch_alkane
'''.split()

descriptor_feature_functions = {
    desc: (lambda mol, f=descriptor_funcs[desc]: f(mol) if mol is not None else np.nan)
    for desc in descriptor_list if desc in descriptor_funcs
}

# ---- STEP 3: Combine feature functions ----
feature_functions = {**custom_feature_functions, **descriptor_feature_functions}
feature_order = list(feature_functions.keys())

# ---- STEP 4: Compute descriptors on Mol in parallel ----
def compute_all_descriptors(mol):
    if mol is None:
        return {desc: np.nan for desc in descriptor_list}
    result = {}
    for desc in descriptor_list:
        try:
            result[desc] = descriptor_feature_functions[desc](mol)
        except Exception:
            result[desc] = np.nan
    return result

with ProcessPoolExecutor(max_workers=6) as executor:
    results = list(executor.map(compute_all_descriptors, df['mol']))

desc_df = pd.DataFrame(results)
df = pd.concat([df.reset_index(drop=True), desc_df], axis=1)

# ---- STEP 5: Apply custom feature functions safely on Mol ----
for name, func in custom_feature_functions.items():
    df[name] = df['mol'].apply(safe_apply_mol(func))



# Cleanup if you want:
df.drop(columns=['mol'], inplace=True)


train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
train.head()


from pathlib import Path
def calc_descriptors(df):
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    with ProcessPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(compute_all_descriptors, df['mol']))
    desc_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), desc_df], axis=1)        
    for name, func in custom_feature_functions.items():
        df[name] = df['mol'].apply(safe_apply_mol(func))
    df.drop(columns=['mol'], inplace=True)        
    return df

cache_train_desc = "train_desc.csv"
if Path(cache_train_desc).exists():
    # just for convenience    
    train_desc_df = pd.read_csv(cache_train_desc)
else:
    train_desc_df = calc_descriptors(train[["id", "SMILES"]])
    train_desc_df.to_csv(cache_train_desc, index=False)

len(train_desc_df.columns)


df.columns, len(df.columns), train_desc_df.columns, len(train_desc_df.columns)


if MODE == "original":
    # ---- STEP 6: Scale features with test data ----
    scaler.fit(df[feature_order])
else:
    # ---- STEP 6: Scale features with train data (as expected) ----
    scaler = StandardScaler()    
    scaler.fit(train_desc_df[feature_order])
    
scaled_features = scaler.transform(df[feature_order])
df.loc[:, feature_order] = scaled_features



train_desc_scaled = train_desc_df.copy()
# the numeric cols will become float and pandas complains
train_desc_scaled[feature_order] = train_desc_scaled[feature_order].astype(float)
train_scaled_features = scaler.transform(train_desc_scaled[feature_order])
train_desc_scaled.loc[:, feature_order] = train_scaled_features   


import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class BoostedXGBoostRFModel:
    def __init__(self, feature_order, params=None):
        self.feature_order = feature_order
        self.model = None
        self.target_column = None
        
        # Default parameters for GPU-accelerated Random Forest style
        default_params = {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'verbosity': 1,
            'num_parallel_tree': 100,
            'subsample': 0.8,
            'colsample_bynode': 0.8,
            'max_depth': 6,
            'eta': 0.1,
            'gamma': 0,
            'min_child_weight': 1,
            'grow_policy': 'depthwise',
        }
        
        # Override defaults if params provided
        self.params = params if params is not None else default_params

    def fit(self, df, target_column, validation_split=0.2, early_stopping_rounds=10):
        self.target_column = target_column

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[target_column])  # Drop rows with NaN target

        df = df.reset_index(drop=True)
        n_total = len(df)
        n_val = int(n_total * validation_split)
        n_train = n_total - n_val

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:]

        X_train = train_df[self.feature_order].astype(np.float32)
        y_train = train_df[target_column].astype(np.float32)

        X_val = val_df[self.feature_order].astype(np.float32)
        y_val = val_df[target_column].astype(np.float32)

        dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
        dval = xgb.DMatrix(X_val, label=y_val, missing=np.nan)

        evals = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(self.params,
                            dtrain,
                            num_boost_round=1000,
                            evals=evals,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=True)

        print(f"Training for target '{target_column}' complete.")
        print(f"Best iteration: {self.model.best_iteration}")
        print(f"Best validation score: {self.model.best_score}")

        return self.model

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_order].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0)
        dmatrix = xgb.DMatrix(X)
        preds = self.model.predict(dmatrix, iteration_range=(0, self.model.best_iteration))
        return pd.DataFrame({f"{self.target_column}_predictions": preds}, index=df.index)





import os
import xgboost as xgb

# Folder where models are stored
load_dir = "/kaggle/input/new-models/other/default/1"

# List of targets
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Timestamp used in filenames
timestamp = "2025-06-25_16-18"

# Reconstruct models dictionary
models = {}
original_features = list(feature_functions.keys())  # Ensure this is defined elsewhere

for target in targets:
    # Initialize your wrapper
    model_wrapper = BoostedXGBoostRFModel(feature_order=original_features)
    model_wrapper.target_column = target

    # Load the model file with timestamp
    model_path = os.path.join(load_dir, f"{target}_model_{timestamp}.json")
    booster = xgb.Booster()
    booster.load_model(model_path)

    # Assign booster to wrapper
    model_wrapper.model = booster
    models[target] = model_wrapper

    print(f"✅ Loaded model for target '{target}' from {model_path}")




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




# Assign each renamed prediction DataFrame
Tg_df = renamed_predictions['Tg']
FFV_df = renamed_predictions['FFV']
Tc_df = renamed_predictions['Tc']
Density_df = renamed_predictions['Density']
Rg_df = renamed_predictions['Rg']

# Combine all predictions into a single DataFrame
xgboost_predictions_df = combine_predictions_with_id(
    df,
    Tg_df,
    FFV_df,
    Tc_df,
    Density_df,
    Rg_df
)

# Display the result
xgboost_predictions_df.head()



xgboost_predictions_df.to_csv("submission.csv", index=False)
xgboost_predictions_df.to_csv(f"submission_{MODE}.csv", index=False)




test_pred = xgboost_predictions_df.copy()


from rdkit import Chem
from mordred import Calculator, descriptors

import sys
sys.path.append('/kaggle/input/adjusting-dataset-shift-by-model-residuals')
from dataset_shift.adjust_residuals import adjust_predictions_with_knn_residuals, apply_shift


print("Adjust TG shift", ADJUST_TG)
if ADJUST_TG is not None:
    target = "Tg"
    test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
    train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")

    train_pred = models[target].predict(train_desc_df)
    train_pred = train_pred.rename(columns={"Tg_predictions": "Tg"})

    if ADJUST_TG == "modred":
        # one can calculate Mordred descriptors on train instead of reading , but it takes time
        train_desc = pd.read_csv("/kaggle/input/adjusting-dataset-shift-by-model-residuals/dataset_shift/mordred_decr_train.csv")
        train_desc = train_desc.dropna(axis=1, how='any')
        mols_test = [Chem.MolFromSmiles(s) for s in test.SMILES]
        calc = Calculator(descriptors, ignore_3D=True) 
        test_desc = calc.pandas(mols_test)
        # mordred stores error messages in cells if it fails
        test_desc = test_desc.apply(pd.to_numeric, errors='coerce')
        # mordred desc can have nans
        test_desc = test_desc.dropna(axis=1, how='any')
        # and we just want common set of descriptors
        common_cols = train_desc.columns.intersection(test_desc.columns)
        test_desc = test_desc[common_cols]
        train_desc = train_desc[common_cols]

    else:  # use model descriptors
        train_desc = train_desc_scaled.drop(columns=["id", "SMILES"])
        test_desc = df.drop(columns=["id", "SMILES"])
        #print(test_desc.describe())
        #print(train_desc.describe())

    y_corrected, prediction_intervals = apply_shift(
        target, train, train_pred,train_desc, 
        test_desc, test_pred, n_neighbors=5, quantiles=[0.1, 0.9], plot=True)
    
    print(y_corrected, prediction_intervals)
    #display(test.head())
    xgboost_predictions_df["Tg"] = y_corrected
    xgboost_predictions_df.to_csv('submission.csv',index=False)
    xgboost_predictions_df.to_csv(f'submission_{MODE}_adjusted_{ADJUST_TG}.csv',index=False)
    display(xgboost_predictions_df.head())

    
    

