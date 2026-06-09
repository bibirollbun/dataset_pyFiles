# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


# === Imports ===
import pandas as pd
import numpy as np
from rdkit import Chem

# === Config ===
BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
BAD_PATTERNS = ['[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]',
                "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
                '([R])', '([R1])', '([R2])']

# === SMILES Cleaner ===
def clean_and_validate_smiles(smiles):
    if not isinstance(smiles, str) or not smiles:
        return None
    for pattern in BAD_PATTERNS:
        if pattern in smiles:
            return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
    return None

# === Load Train/Test ===
train = pd.read_csv(BASE_PATH + 'train.csv')
test = pd.read_csv(BASE_PATH + 'test.csv')

train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)

train.dropna(subset=['SMILES'], inplace=True)
test.dropna(subset=['SMILES'], inplace=True)

# === Load External Datasets (excluding dataset2) ===
external_datasets = []

def load_external(path, target, rename_map=None):
    try:
        df = pd.read_csv(path)
        if rename_map:
            df = df.rename(columns=rename_map)
        if 'SMILES' in df.columns and target in df.columns:
            df = df[['SMILES', target]].dropna()
            external_datasets.append((target, df))
            print(f"âœ… Loaded {path} ({len(df)} entries for {target})")
        else:
            print(f"âš ï¸� Skipped {path}: required columns missing")
    except Exception as e:
        print(f"âš ï¸� Failed to load {path}: {e}")

load_external(BASE_PATH + 'train_supplement/dataset1.csv', 'Tc', rename_map={'TC_mean': 'Tc'})
load_external(BASE_PATH + 'train_supplement/dataset3.csv', 'Tg')
load_external(BASE_PATH + 'train_supplement/dataset4.csv', 'FFV')

# === Merge External Data ===
def merge_external(train_df, ext_df, target):
    ext_df['SMILES'] = ext_df['SMILES'].apply(clean_and_validate_smiles)
    ext_df = ext_df.dropna(subset=['SMILES', target])
    ext_df = ext_df.groupby('SMILES', as_index=False)[target].mean()

    # Fill missing target values in existing rows
    existing_smiles = set(train_df['SMILES'])
    to_fill = ext_df[ext_df['SMILES'].isin(existing_smiles)]
    for _, row in to_fill.iterrows():
        mask = (train_df['SMILES'] == row['SMILES']) & (train_df[target].isna())
        train_df.loc[mask, target] = row[target]

    # Add new rows
    new_smiles = set(ext_df['SMILES']) - existing_smiles
    new_rows = ext_df[ext_df['SMILES'].isin(new_smiles)].copy()
    for col in TARGETS:
        if col not in new_rows.columns:
            new_rows[col] = np.nan
    return pd.concat([train_df, new_rows[['SMILES'] + TARGETS]], ignore_index=True)

# === Apply Merges ===
train_extended = train[['SMILES'] + TARGETS].copy()
for target, ext in external_datasets:
    train_extended = merge_external(train_extended, ext, target)

# === Final Clean-Up ===
train_extended = train_extended.replace([np.inf, -np.inf], np.nan)
train_extended = train_extended.dropna(subset=TARGETS, how='all')
train_extended = train_extended.drop_duplicates(subset=['SMILES']).reset_index(drop=True)

# === Summary ===
print("\nğŸ“Š Final Summary:")
print(f"Train: {len(train)} | Extended: {len(train_extended)}")
for t in TARGETS:
    base = train[t].notna().sum()
    ext = train_extended[t].notna().sum()
    print(f"â€¢ {t:<8}: {ext} total ({ext - base:+} from supplements)")

print("\nâœ… Data loading and preprocessing complete.")



smiles_list = train_extended['SMILES'].tolist()
# Clean SMILES column robustly
train_extended['SMILES'] = train_extended['SMILES'].apply(clean_and_validate_smiles)
train_extended.shape
train_extended


import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, rdMolDescriptors
from rdkit.Chem import Descriptors
import networkx as nx
from tqdm import tqdm

# === Feature category config ===
FEATURE_CATEGORIES = {
    "rdkit": "all", 
    "graph": [
        'graph_diameter', 'avg_shortest_path', 'num_cycles',
        'betweenness_mean', 'betweenness_std', 'eigenvector_mean',
        'ring_4', 'max_degree', 'closeness_mean', 'katz_centrality_std',
        'heteroatom_ratio'
    ],
    "maccs": ['MACCS_Key130', 'MACCS_Key142', 'MACCS_Key066', 'MACCS_Key153'],
    "topo_torsion": ['TopologicalTorsion_Bit0512', 'TopologicalTorsion_Bit1296'],
    "atom_pair": ['AtomPair_B512_Bit0138', 'AtomPair_B512_Bit0448', 'AtomPair_B512_Bit0408'],
    "morgan": "all",
}

USELESS_COLS = set([
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
    'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur',
    'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide', 'fr_isothiocyan',
    'fr_lactam', 'fr_nitroso', 'fr_prisulfonamd', 'fr_thiocyan',
    'LabuteASA', 'HeavyAtomCount', 'Chi4v', 'MinAbsPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'fr_nitrile',
    'NumAromaticCarbocycles', 'NumAromaticRings', 'fr_amide',
    'fr_Nhpyrrole', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_COO2',
    'fr_diazo', 'fr_nitro_arom', 'fr_phos_ester'
])

# === Core Functions ===

def compute_rdkit_descriptors(mol, allowed=None):
    descs = {}
    for name, func in Descriptors.descList:
        if name in USELESS_COLS:
            continue
        if allowed is not None and name not in allowed:
            continue
        try:
            descs[name] = func(mol)
        except:
            descs[name] = np.nan
        
    try:
        descs['LogP'] = MolLogP(mol)
    except:
        descs['LogP'] = np.nan

    try:
        descs['NumAtoms'] = mol.GetNumAtoms() if mol else np.nan
    except:
        descs['NumAtoms'] = np.nan

    try:
        descs['RotatableBonds'] = CalcNumRotatableBonds(mol)
    except:
        descs['RotatableBonds'] = np.nan
        
    return descs

def compute_graph_descriptors(mol):
    g = nx.Graph()
    g.add_nodes_from(range(mol.GetNumAtoms()))
    g.add_edges_from([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()])
    
    descriptors = {}
    try:
        descriptors['graph_diameter'] = nx.diameter(g)
        descriptors['avg_shortest_path'] = nx.average_shortest_path_length(g)
    except:
        descriptors['graph_diameter'] = 0
        descriptors['avg_shortest_path'] = 0

    descriptors['num_cycles'] = len(nx.cycle_basis(g))
    descriptors['betweenness_mean'] = np.mean(list(nx.betweenness_centrality(g).values()))
    descriptors['betweenness_std'] = np.std(list(nx.betweenness_centrality(g).values()))
    
    try:
        ec = nx.eigenvector_centrality_numpy(g)
        descriptors['eigenvector_mean'] = np.mean(list(ec.values()))
    except Exception:
        descriptors['eigenvector_mean'] = np.nan  # <-- fix here
        
    descriptors['closeness_mean'] = np.mean(list(nx.closeness_centrality(g).values()))
    descriptors['max_degree'] = max(dict(g.degree()).values())
    
    try:
        katz = nx.katz_centrality_numpy(g)
        descriptors['katz_centrality_std'] = np.std(list(katz.values()))
    except Exception:
        descriptors['katz_centrality_std'] = np.nan  # <-- fix here
        
    descriptors['ring_4'] = sum(1 for r in mol.GetRingInfo().AtomRings() if len(r) == 4)
    descriptors['heteroatom_ratio'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [1, 6]) / mol.GetNumAtoms()
    
    return descriptors

def compute_maccs(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    return {f'MACCS_Key{str(i).zfill(3)}': int(fp[i]) for i in range(len(fp))}

def compute_topo_torsion(mol):
    torsion_bits = ['TopologicalTorsion_Bit0512', 'TopologicalTorsion_Bit1296']
    torsion_defaults = {bit: 0 for bit in torsion_bits}

    if mol:
        try:
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
            bits = [int(x) for x in fp.ToBitString()]
            return {
                'TopologicalTorsion_Bit0512': bits[512],
                'TopologicalTorsion_Bit1296': bits[1296]
            }
        except Exception as e:
            print(f"TT fingerprint failed: {e}")
            return torsion_defaults
    else:
        return torsion_defaults

def compute_atom_pair(mol):
    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
    arr = np.zeros((2048,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return {
        'AtomPair_B512_Bit0138': arr[138],
        'AtomPair_B512_Bit0448': arr[448],
        'AtomPair_B512_Bit0408': arr[408],
    }

def compute_morgan(mol, nBits=2048, radius=2):
    results = {}
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        bit_array = [int(b) for b in fp.ToBitString()]
        for i in range(nBits):
            results[f'Morgan_Bit{i:04d}'] = bit_array[i]
    except Exception:
        for i in range(nBits):
            results[f'Morgan_Bit{i:04d}'] = np.nan  # Use NaN to indicate failure
    return results

def compute_all_features(smiles_list, verbose=True):
    feature_dict = {}
    valid_idx = []
    failed_idx = []

    for idx, smi in enumerate(tqdm(smiles_list, desc="Computing Features")):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed_idx.append(idx)
            continue
        valid_idx.append(idx)

        feats = {}

        # 1. RDKit descriptors (filtered)
        feats.update(compute_rdkit_descriptors(mol))

        # 2. Graph features
        feats.update(compute_graph_descriptors(mol))

        # 3. MACCS
        feats.update(compute_maccs(mol))

        # 4. Topological Torsion
        feats.update(compute_topo_torsion(mol))

        # 5. Atom Pair
        feats.update(compute_atom_pair(mol))

        # 6. Morgan fingerprint bits
        feats.update(compute_morgan(mol)) 

        for k, v in feats.items():
            if k not in feature_dict:
                feature_dict[k] = []
            feature_dict[k].append(v)

    # Fill missing values with None for failed molecules
    for k in feature_dict:
        while len(feature_dict[k]) < len(valid_idx):
            feature_dict[k].append(None)

    # Summary
    if verbose:
        print("\n--- Feature Engineering Summary ---")
        print(f"Total SMILES: {len(smiles_list)}")
        print(f"Valid molecules: {len(valid_idx)}")
        print(f"Invalid molecules: {len(failed_idx)}")
        print(f"Number of computed features: {len(feature_dict)}")
        sample_key = next(iter(feature_dict))
        print(f"Feature vector length per molecule: {len(feature_dict[sample_key])}")
        print("-----------------------------------")

    return feature_dict, valid_idx



from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  # Suppress all RDKit warnings

# For train set (you already have this)
feature_dict_train, valid_idx_train = compute_all_features(train_extended["SMILES"], verbose=True)
features_train = pd.DataFrame(feature_dict_train)
features_train = features_train.reset_index(drop=True)

# For test set (do the same)
feature_dict_test, valid_idx_test = compute_all_features(test["SMILES"], verbose=True)
features_test = pd.DataFrame(feature_dict_test)
features_test = features_test.reset_index(drop=True)

print("Train features shape:", features_train.shape)
print("Test features shape:", features_test.shape)
print("Training dataframe Shape:",train_extended.shape)
print("Test dataframe Shape:",test.shape)


!pip install lightgbm scikit-learn catboost

import warnings
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

# Suppress warnings
warnings.filterwarnings("ignore")

# List of targets and model choices
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
catboost_targets = ['Tg', 'FFV', 'Tc', 'Density']
svr_targets = ['Rg']

# Dictionaries for models and predictions
models = {}
test_preds = pd.DataFrame(index=test.index)

# =====================
# --- CatBoost Block ---
# =====================
for target in catboost_targets:
    print(f"\nğŸš§ Training CatBoost model for {target}...")

    mask = ~train_extended[target].isna()
    X_train_target = features_train.loc[mask].reset_index(drop=True)
    y_train_target = train_extended.loc[mask, target].reset_index(drop=True)

    train_pool = Pool(data=X_train_target, label=y_train_target)

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=42,
        early_stopping_rounds=100,
        verbose=100,
        task_type='CPU'
    )

    model.fit(train_pool)

    preds = model.predict(features_test)
    test_preds[target] = preds
    models[target] = model

    print("ğŸ“Š Top 5 important features:")
    print(model.get_feature_importance(prettified=True).head())

# =================
# --- SVR Block ---
# =================
for target in svr_targets:
    print(f"\nğŸ”� Training SVR model for {target}...")

    mask = ~train_extended[target].isna()
    X_train_target = features_train.loc[mask].reset_index(drop=True)
    y_train_target = train_extended.loc[mask, target].reset_index(drop=True)
    X_test_target = features_test.reset_index(drop=True)

    X_train_target.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_target.replace([np.inf, -np.inf], np.nan, inplace=True)

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train_target)
    X_test_imputed = imputer.transform(X_test_target)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train_target)

    # Predictions
    preds_test = model.predict(X_test_scaled)
    preds_train = model.predict(X_train_scaled)

    # Save test predictions
    test_preds[target] = preds_test
    models[target] = model

    # Evaluation
    mae = mean_absolute_error(y_train_target, preds_train)
    r2 = r2_score(y_train_target, preds_train)

    print(f"ğŸ“ˆ Training MAE for {target}: {mae:.6f}")
    print(f"ğŸ“‰ Training RÂ² score for {target}: {r2:.6f}")
    print(f"âœ… SVR model for {target} completed.")

# ============================
# --- Save Final Submission ---
# ============================
submission = test[['id']].copy()
submission = pd.concat([submission, test_preds], axis=1)
submission.to_csv("submission.csv", index=False)
print("\nğŸ�‰ Submission file 'submission.csv' created successfully!")


# Display the final predicted table
print("\nğŸ“‹ Preview of final test predictions:")
display_columns = ['id'] + target_cols
display(submission[display_columns].head(10))  # Change 10 to see more rows

