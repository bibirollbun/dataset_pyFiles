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
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canonical.append(Chem.MolToSmiles(mol, canonical=True))
        else:
            canonical.append(None)
    return canonical

# === All RDKit Descriptors ===
def compute_rdkit_descriptors(mol):
    descs = {}
    for name, func in Descriptors.descList:
        try:
            descs[name] = func(mol)
        except:
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
    except:
        descriptors['graph_diameter'] = 0
        descriptors['avg_shortest_path'] = 0

    descriptors['num_cycles'] = len(nx.cycle_basis(g))

    try:
        descriptors['betweenness_mean'] = np.mean(list(nx.betweenness_centrality(g).values()))
        descriptors['betweenness_std'] = np.std(list(nx.betweenness_centrality(g).values()))
        descriptors['closeness_mean'] = np.mean(list(nx.closeness_centrality(g).values()))
        descriptors['max_degree'] = max(dict(g.degree()).values())
    except:
        descriptors['betweenness_mean'] = np.nan
        descriptors['betweenness_std'] = np.nan
        descriptors['closeness_mean'] = np.nan
        descriptors['max_degree'] = np.nan

    try:
        ec = nx.eigenvector_centrality_numpy(g)
        descriptors['eigenvector_mean'] = np.mean(list(ec.values()))
    except:
        descriptors['eigenvector_mean'] = np.nan

    try:
        ring_info = mol.GetRingInfo().AtomRings()
        descriptors['ring_4'] = sum(1 for r in ring_info if len(r) == 4)
    except:
        descriptors['ring_4'] = 0

    try:
        descriptors['heteroatom_ratio'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [1, 6]) / mol.GetNumAtoms()
    except:
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
            failed_idx.append(idx)
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
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
    for k in feature_dict:
        if len(feature_dict[k]) < total:
            feature_dict[k].extend([None] * (total - len(feature_dict[k])))

    if verbose:
        print("\n--- Feature Engineering Summary ---")
        print(f"Total SMILES: {total}")
        print(f"Valid molecules: {len(valid_idx)}")
        print(f"Invalid molecules: {len(failed_idx)}")
        print(f"Number of computed features: {len(feature_dict)}")
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
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Save SMILES before dropping it for numeric processing
smiles_train = features_train['SMILES'].copy()

# === Preprocessing Utilities ===

def replace_inf_and_clip(df, df_name="", lower=-1e6, upper=1e6):
    print(f"\nğŸ§¯ [{df_name}] Replacing Â±inf and clipping to [{lower}, {upper}]...")
    inf_count = np.isinf(df.values).sum()
    print(f"  - Replacing {inf_count} Â±inf values with NaN...")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Clip before filling NaNs
    max_vals = df.max()
    min_vals = df.min()
    too_large = max_vals[max_vals > upper]
    too_small = min_vals[min_vals < lower]
    if not too_large.empty or not too_small.empty:
        print(f"  - Clipping {len(too_large)} overly large and {len(too_small)} overly small features.")
        df = df.clip(lower, upper)
    else:
        print("  - No extreme outliers found.")

    return df


def fill_nans(df, df_name=""):
    print(f"\nğŸ”§ [{df_name}] Handling NaNs...")
    all_nan_cols = df.columns[df.isna().all()].tolist()
    print(f"  - Dropping {len(all_nan_cols)} all-NaN columns...")
    df = df.dropna(axis=1, how='all')

    nan_count = df.isna().sum().sum()
    print(f"  - Filling {nan_count} remaining NaNs with column means...")
    df = df.fillna(df.mean())

    return df


def remove_low_variance(train_df, test_df, threshold=1e-5):
    print(f"\nğŸ§¹ Applying VarianceThreshold (threshold={threshold}) on TRAIN, transforming TEST...")
    selector = VarianceThreshold(threshold=threshold)
    train_reduced = selector.fit_transform(train_df)
    test_reduced = selector.transform(test_df)

    kept_cols = train_df.columns[selector.get_support()]
    removed_count = train_df.shape[1] - len(kept_cols)
    print(f"  - Removed {removed_count} low-variance features.")
    
    train_df = pd.DataFrame(train_reduced, columns=kept_cols)
    test_df = pd.DataFrame(test_reduced, columns=kept_cols)
    return train_df, test_df


# === Apply Preprocessing ===

# Drop non-numeric columns (like 'SMILES') if present before preprocessing
features_train_numeric = features_train.drop(columns=['SMILES'], errors='ignore')
features_test_numeric = features_test.drop(columns=['SMILES'], errors='ignore')

# Step 1: Replace Â±inf and clip first
features_train_clean = replace_inf_and_clip(features_train_numeric, df_name="Train")
features_test_clean = replace_inf_and_clip(features_test_numeric, df_name="Test")

# Step 2: Fill NaNs
features_train_clean = fill_nans(features_train_clean, df_name="Train")
features_test_clean = fill_nans(features_test_clean, df_name="Test")

# Step 3: Align columns
common_cols = features_train_clean.columns.intersection(features_test_clean.columns)
features_train_clean = features_train_clean[common_cols].copy()
features_test_clean = features_test_clean[common_cols].copy()

# Step 4: Remove low-variance features (fit on train, apply to both)
features_train_clean, features_test_clean = remove_low_variance(features_train_clean, features_test_clean)

# === Summary ===
print("\nâœ… Final Preprocessing Summary:")
print(f"  - Train shape: {features_train_clean.shape}")
print(f"  - Test shape:  {features_test_clean.shape}")
print(f"  - Common features retained: {features_train_clean.shape[1]}")



# === Add SMILES back to the cleaned train features ===
features_train_clean['SMILES'] = smiles_train.values

# === Set SMILES as index to align features and targets ===
features_train_clean = features_train_clean.set_index('SMILES')
train_extended_indexed = train_extended.set_index('SMILES')

# === Extract target columns aligned by SMILES index ===
targets_df = train_extended_indexed.loc[features_train_clean.index, TARGETS]

# === Reset index for modeling ===
X = features_train_clean.reset_index(drop=True)
y = targets_df.reset_index(drop=True)

# === Optionally drop rows where all targets are missing ===
mask = y.notnull().any(axis=1)
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

print(f"Final feature shape: {X.shape}")
print(f"Final target shape: {y.shape}")



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd

# Split before scaling to avoid data leakage
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd


# Reset indices to keep alignment
X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

# --- Choose scaler here ---
use_robust_scaling = True  # ğŸ”� Set to False to use StandardScaler

if use_robust_scaling:
    scaler = RobustScaler()
    print("\nğŸ”§ Using RobustScaler to reduce effect of outliers.")
else:
    scaler = StandardScaler()
    print("\nğŸ”§ Using StandardScaler.")

# Scale features
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns)

print("\nâœ… Feature scaling complete.")
print(f"Scaled train shape: {X_train_scaled.shape}")
print(f"Scaled val shape:   {X_val_scaled.shape}")




from scipy.stats import skew
import numpy as np

def analyze_targets(y_df, log_transform_targets=None):
    log_transform_targets = log_transform_targets or []

    for col in y_df.columns:
        # Replace inf with nan, drop nan
        target_data = y_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        skew_val = skew(target_data)

        # Outlier detection
        Q1 = target_data.quantile(0.25)
        Q3 = target_data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((target_data < (Q1 - 1.5 * IQR)) | (target_data > (Q3 + 1.5 * IQR))).sum()

        print(f"\nğŸ“Œ Target: {col}")
        print(f"  - Skewness: {skew_val:.3f}")
        print(f"  - Outliers (IQR method): {outlier_count} of {len(target_data)}")

        # Optional log-transform safely: handle inf/nan again before transform
        if col in log_transform_targets:
            print(f"  - Applying log1p transformation to '{col}'...")
            clean_col = y_df[col].replace([np.inf, -np.inf], np.nan)
            y_df[col] = np.log1p(clean_col)

# Example call:
analyze_targets(y, log_transform_targets=['Tg', 'FFV', 'Tc','Density'])




from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}

print("\nğŸ”� Training baseline models...\n")

for target in y_train.columns:
    print(f"ğŸ“Œ Target: {target}")
    
    # Filter rows where target is NOT NaN
    not_null_train = y_train[target].notnull()
    not_null_val = y_val[target].notnull()
    
    # Filter features for those rows and drop any remaining NaNs in features
    X_t_train = X_train_scaled.loc[not_null_train].dropna()
    y_t_train = y_train.loc[X_t_train.index, target]
    
    X_t_val = X_val_scaled.loc[not_null_val].dropna()
    y_t_val = y_val.loc[X_t_val.index, target]
    
    # Check again for any NaNs in targets after aligning indices
    y_t_train = y_t_train.loc[X_t_train.index].dropna()
    X_t_train = X_t_train.loc[y_t_train.index]
    
    y_t_val = y_t_val.loc[X_t_val.index].dropna()
    X_t_val = X_t_val.loc[y_t_val.index]
    
    print(f"  - Train target NaNs after filtering: {y_t_train.isna().sum()}")
    print(f"  - Train features NaNs after filtering: {X_t_train.isna().sum().sum()}")
    print(f"  - Val target NaNs after filtering: {y_t_val.isna().sum()}")
    print(f"  - Val features NaNs after filtering: {X_t_val.isna().sum().sum()}")
    
    target_results = {}
    
    for name, model in models.items():
        model.fit(X_t_train, y_t_train)
        preds = model.predict(X_t_val)
        mae = mean_absolute_error(y_t_val, preds)
        print(f"  - {name}: MAE = {mae:.4f}")
        target_results[name] = mae
    
    results[target] = target_results
    print()

# Summary of MAE
import pandas as pd
results_df = pd.DataFrame(results).T
print("ğŸ“Š MAE Summary (lower is better):\n")
print(results_df)



best_models = {}
for target in results_df.index:
    best_model_name = results_df.loc[target].idxmin()
    best_models[target] = models[best_model_name]



from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from IPython.display import display

# Re-initialize models dictionary to ensure clean state
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

# âœ… Select best model for each target from results_df
best_models = {}
for target in results_df.index:
    best_model_name = results_df.loc[target].idxmin()  # get model with lowest MAE
    best_models[target] = models[best_model_name]
    print(f"âœ… Best model for {target}: {best_model_name}")

# âœ… Scale test features using existing fitted scaler
X_test_final = pd.DataFrame(
    scaler.transform(features_test_clean), 
    columns=features_test_clean.columns
)

# Align test features to training features
X_test_final = X_test_final[features_train_clean.columns]

# Prepare final predictions DataFrame
final_predictions = pd.DataFrame()

# Add ID column if available
if 'id' in test.columns:
    final_predictions['id'] = test['id'].values
else:
    raise KeyError("Test dataframe does not contain 'id' column.")

# ğŸš€ Make predictions using best model per target
print("\nğŸš€ Generating final test predictions using best models per target...\n")

for target in TARGETS:
    print(f"ğŸ“Œ Target: {target}")

    best_model = best_models[target]
    print(f"  - Using model: {type(best_model).__name__}")

    # Use full training data where target is not null
    not_null_mask = y_train[target].notnull()
    X_target_train = X_train_scaled.loc[not_null_mask]
    y_target_train = y_train.loc[not_null_mask, target]

    # Retrain best model on full data
    best_model.fit(X_target_train, y_target_train)

    # Predict on test set
    preds = best_model.predict(X_test_final)
    final_predictions[target] = preds

print("\nâœ… Prediction complete.")

# Save to both local and Kaggle output paths
submission_path = "/kaggle/working/submission.csv"
final_predictions.to_csv(submission_path, index=False)
final_predictions.to_csv("submission.csv", index=False)
print(f"\nğŸ“� Submission saved to: {submission_path} and 'submission.csv'")

# Display preview
print("\nğŸ“‹ Preview of final test predictions:")
display(final_predictions.head(10))


