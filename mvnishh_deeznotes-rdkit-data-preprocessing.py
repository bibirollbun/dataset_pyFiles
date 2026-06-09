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

# === Load Additional External Datasets ===
try:
    extra_data_tg3 = pd.read_excel("/kaggle/input/smiles-extra-data/data_tg3.xlsx")
    extra_data_dnst1 = pd.read_excel("/kaggle/input/smiles-extra-data/data_dnst1.xlsx")
    jcim_sup_bigsmiles = pd.read_csv("/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv")
    tc_smiles_df = pd.read_csv("/kaggle/input/tc-smiles/Tc_SMILES.csv")
except Exception as e:
    print(f"âš ï¸� Error loading extra data: {e}")

# Helper to standardize and append
def process_and_append_external(df, target, source_name):
    if 'SMILES' in df.columns and target in df.columns:
        df = df[['SMILES', target]].copy()
        df['SMILES'] = df['SMILES'].apply(clean_and_validate_smiles)
        df = df.dropna(subset=['SMILES'])

        # Ensure the target column is numeric
        df[target] = pd.to_numeric(df[target], errors='coerce')
        df = df.dropna(subset=[target])

        df = df.groupby('SMILES', as_index=False)[target].mean()
        external_datasets.append((target, df))
        print(f"âœ… Integrated {source_name}: {len(df)} entries for {target}")
    else:
        print(f"âš ï¸� Skipped {source_name}: missing columns")

# Process each extra dataset (with correct column names)
process_and_append_external(extra_data_tg3.rename(columns={"Tg [K]": "Tg"}), "Tg", "data_tg3.xlsx")
process_and_append_external(extra_data_dnst1.rename(columns={"density(g/cm3)": "Density"}), "Density", "data_dnst1.xlsx")
process_and_append_external(tc_smiles_df.rename(columns={"TC_mean": "Tc"}), "Tc", "Tc_SMILES.csv")

# JCIM SMILES only (for future feature engineering)
jcim_smiles_only = jcim_sup_bigsmiles[['SMILES']].dropna()
jcim_smiles_only['SMILES'] = jcim_smiles_only['SMILES'].apply(clean_and_validate_smiles)
jcim_smiles_only = jcim_smiles_only.dropna().drop_duplicates()
print(f"âœ… Loaded JCIM SMILES-only dataset: {len(jcim_smiles_only)} unique SMILES (no targets)")

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
# === Final Clean-Up ===
train_extended = train_extended.replace([np.inf, -np.inf], np.nan)
train_extended = train_extended.dropna(subset=TARGETS, how='all')
train_extended = train_extended.drop_duplicates(subset=['SMILES']).reset_index(drop=True)

# === Drop constant columns ===
constant_cols = [col for col in train_extended.columns if train_extended[col].nunique() == 1]
train_extended.drop(columns=constant_cols, inplace=True)
print(f"Dropped {len(constant_cols)} constant columns from train_extended")


train_extended.shape
train_extended



import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import networkx as nx
from tqdm import tqdm

# === Canonicalize SMILES ===
def canonicalize_smiles(smiles_list):
    canonical = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canonical.append(Chem.MolToSmiles(mol, canonical=True))
        else:
            print(f"Warning: invalid SMILES at index {idx}: '{smi}'")
            canonical.append(None)
    return canonical

# === All RDKit Descriptors ===
def compute_rdkit_descriptors(mol):
    descs = {}
    for name, func in Descriptors.descList:
        try:
            descs[name] = func(mol)
        except Exception as e:
            descs[name] = np.nan
    return descs

# === Graph Features ===
def compute_graph_descriptors(mol):
    descriptors = {}
    g = nx.Graph()
    g.add_edges_from([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()])

    try:
        descriptors['graph_diameter'] = nx.diameter(g) if nx.is_connected(g) else 0
        descriptors['avg_shortest_path'] = nx.average_shortest_path_length(g) if nx.is_connected(g) else 0
    except Exception as e:
        descriptors['graph_diameter'] = 0
        descriptors['avg_shortest_path'] = 0

    descriptors['num_cycles'] = len(nx.cycle_basis(g))

    try:
        bc = nx.betweenness_centrality(g)
        descriptors['betweenness_mean'] = np.mean(list(bc.values()))
        descriptors['betweenness_std'] = np.std(list(bc.values()))
        descriptors['closeness_mean'] = np.mean(list(nx.closeness_centrality(g).values()))
        descriptors['max_degree'] = max(dict(g.degree()).values())
    except Exception as e:
        descriptors['betweenness_mean'] = np.nan
        descriptors['betweenness_std'] = np.nan
        descriptors['closeness_mean'] = np.nan
        descriptors['max_degree'] = np.nan

    try:
        ec = nx.eigenvector_centrality_numpy(g)
        descriptors['eigenvector_mean'] = np.mean(list(ec.values()))
    except Exception as e:
        descriptors['eigenvector_mean'] = np.nan

    try:
        ring_info = mol.GetRingInfo().AtomRings()
        descriptors['ring_4'] = sum(1 for r in ring_info if len(r) == 4)
    except Exception as e:
        descriptors['ring_4'] = 0

    try:
        descriptors['heteroatom_ratio'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [1, 6]) / mol.GetNumAtoms()
    except Exception as e:
        descriptors['heteroatom_ratio'] = np.nan

    return descriptors

# === Final Combined Feature Computation ===
def compute_all_features(smiles_list, verbose=True):
    smiles_list = canonicalize_smiles(smiles_list)

    feature_dict = {}
    valid_idx = []
    failed_idx = []

    for idx, smi in enumerate(tqdm(smiles_list, desc="Computing Features")):
        if smi is None:
            print(f"Skipping invalid molecule at index {idx}")
            failed_idx.append(idx)
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Failed to create mol from SMILES at index {idx}: '{smi}'")
            failed_idx.append(idx)
            continue

        valid_idx.append(idx)
        feats = {}

        # Compute all descriptors from RDKit
        feats.update(compute_rdkit_descriptors(mol))
        # Compute graph descriptors
        feats.update(compute_graph_descriptors(mol))

        for k, v in feats.items():
            if k not in feature_dict:
                feature_dict[k] = []
            feature_dict[k].append(v)

    total = len(smiles_list)
    # Make sure all feature lists are aligned to total length by padding None for failed indices
    for k in feature_dict:
        current_len = len(feature_dict[k])
        if current_len < total:
            feature_dict[k].extend([None] * (total - current_len))

    if verbose:
        print("\n--- Feature Engineering Summary ---")
        print(f"Total SMILES: {total}")
        print(f"Valid molecules: {len(valid_idx)}")
        print(f"Invalid molecules: {len(failed_idx)}")
        print(f"Number of computed features: {len(feature_dict)}")
        if len(feature_dict) > 0:
            sample_key = next(iter(feature_dict))
            print(f"Feature vector length per molecule: {len(feature_dict[sample_key])}")
        print("-----------------------------------")

    return feature_dict, valid_idx

from rdkit import RDLogger
import pandas as pd

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

useless_cols = [   
    'MaxPartialCharge', 
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
    'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur',
    'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide', 'fr_isothiocyan',
    'fr_lactam', 'fr_nitroso', 'fr_prisulfonamd', 'fr_thiocyan',
    'MaxEStateIndex', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
    'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Kappa1',
    'LabuteASA', 'HeavyAtomCount', 'MolMR', 'Chi3n', 'BertzCT', 'Chi2v',
    'Chi4n', 'HallKierAlpha', 'Chi3v', 'Chi4v', 'MinAbsPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'Phi', 'Kappa3', 'fr_nitrile', 'SlogP_VSA6',
    'NumAromaticCarbocycles', 'NumAromaticRings', 'fr_benzene', 'VSA_EState6',
    'NOCount', 'fr_C_O', 'fr_C_O_noCOO', 'NumHDonors', 'fr_amide',
    'fr_Nhpyrrole', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_COO2',
    'fr_halogen', 'fr_diazo', 'fr_nitro_arom', 'fr_phos_ester'
]

# === Compute Train Features ===
feature_dict_train, valid_idx_train = compute_all_features(train_extended["SMILES"], verbose=True)
features_train = pd.DataFrame(feature_dict_train)

# Add back SMILES column corresponding to valid_idx_train to keep alignment
features_train['SMILES'] = train_extended.loc[valid_idx_train, 'SMILES'].values

# Drop useless columns if present
features_train = features_train.drop(columns=[col for col in useless_cols if col in features_train.columns])

# === Compute Test Features ===
feature_dict_test, valid_idx_test = compute_all_features(test["SMILES"], verbose=True)
features_test = pd.DataFrame(feature_dict_test)

# Add back SMILES column corresponding to valid_idx_test
features_test['SMILES'] = test.loc[valid_idx_test, 'SMILES'].values

features_test = features_test.drop(columns=[col for col in useless_cols if col in features_test.columns])

# === Output Summary ===
print("Train features shape:", features_train.shape)
print("Test features shape:", features_test.shape)
print("Training dataframe Shape:", train_extended.shape)
print("Test dataframe Shape:", test.shape)



import numpy as np
from sklearn.impute import SimpleImputer

def clean_features(df):
    # Replace inf and -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Cap extremely large values at a threshold (e.g., 1e6 or 1e9) to avoid overflow
    max_val = 1e6
    df = df.clip(upper=max_val)
    
    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df_imputed

X_train_clean = clean_features(X_train)
X_test_clean = clean_features(X_test)

predictions_dict = {}

for target_col in y_train.columns:
    print(f"Training model for target: {target_col}")
    y = y_train[target_col]

    # Combine X_train_clean and y to drop rows where y is NaN
    df_train = X_train_clean.copy()
    df_train[target_col] = y.values

    # Drop rows with NaN in the target
    df_train_clean = df_train.dropna(subset=[target_col])

    X_clean = df_train_clean.drop(columns=[target_col])
    y_clean = df_train_clean[target_col]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_clean, y_clean)

    preds = model.predict(X_test_clean)
    predictions_dict[target_col] = preds


predictions_df = pd.DataFrame(predictions_dict)
predictions_df['SMILES'] = features_test['SMILES'].values
predictions_df.to_csv('submission.csv', index=False)

print("Predictions saved to 'predictions.csv'")



predictions_df

