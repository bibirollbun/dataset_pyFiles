# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
# Install RDKit


# Install six (required dependency for Mordred)
!pip install --no-index --find-links=file:///kaggle/input/six-1-17-0-py2-py3-none-any/ six==1.17.0 

# Install Mordred directly from wheel (avoids dependency resolution issues)
!pip install --no-deps /kaggle/input/mordred-1-2-0-py3-none-any/mordred-1.2.0-py3-none-any.whl 

# Install compatible version of networkx for Mordred
!pip install --no-deps /kaggle/input/mordred-1-2-0-py3-none-any/networkx-2.8.8-py3-none-any.whl 





import warnings
warnings.filterwarnings("ignore")
import mordred
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, MACCSkeys, rdmolops
import networkx as nx
from collections import Counter
from mordred import Calculator, descriptors
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from tqdm import tqdm
import warnings
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt


# kaggle Data Import
train_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
new_data = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
tc_smiles = pd.read_csv("/kaggle/input/tc-smiles/Tc_SMILES.csv")
tg_smiles = pd.read_csv("/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv")
ktg_smiles = pd.read_excel("/kaggle/input/smiles-extra-data/data_tg3.xlsx")
de_smiles = pd.read_excel("/kaggle/input/smiles-extra-data/data_dnst1.xlsx")
# feature_df = pd.read_csv("/kaggle/input/curatedfeatures/feature_df.csv")

train_df.head()


def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan

train_df['SMILES'] = train_df['SMILES'].apply(lambda s: make_smile_canonical(s))
new_data['SMILES'] = new_data['SMILES'].apply(lambda s: make_smile_canonical(s))


ktg_smiles.rename(columns={'Tg [K]': 'Tg'}, inplace=True)
tg_smiles.rename(columns={'Tg (C)': 'Tg'}, inplace=True)
tc_smiles.rename(columns={'TC_mean': 'Tc'}, inplace=True)
de_smiles.rename(columns={'density(g/cm3)': 'Density'}, inplace=True)


# Disable RDKit SMILES parse warnings
RDLogger.DisableLog('rdApp.error')  # suppress parsing warnings

def is_valid_smiles(s):
    try:
        mol = Chem.MolFromSmiles(s)
        return mol is not None
    except:
        return False

# Filter valid SMILES
de_smiles = de_smiles[de_smiles['SMILES'].apply(is_valid_smiles)]

# Canonicalize
de_smiles['SMILES'] = de_smiles['SMILES'].apply(lambda s: make_smile_canonical(s))

# Clean and convert
de_smiles = de_smiles[
    (de_smiles['SMILES'].notnull()) &
    (de_smiles['Density'].notnull()) &
    (de_smiles['Density'] != 'nylon')
]
de_smiles['Density'] = de_smiles['Density'].astype('float64')
de_smiles['Density'] -= 0.118

# Tg conversion (K â†’ Â°C)
ktg_smiles['Tg'] = ktg_smiles['Tg'] - 273.15



def add_extra_data(df_train, df_extra, target):
    n_samples_before = len(df_train[df_train[target].notnull()])

    df_extra['SMILES'] = df_extra['SMILES'].apply(lambda s: make_smile_canonical(s))
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # Make priority target value from competition's df
    for smile in df_train[df_train[target].notnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            cross_smiles.remove(smile)

    # Imput missing values for competition's SMILES
    for smile in cross_smiles:
        df_train.loc[df_train['SMILES']==smile, target] = df_extra[df_extra['SMILES']==smile][target].values[0]

    df_train = pd.concat([df_train, df_extra[df_extra['SMILES'].isin(unique_smiles_extra)]], axis=0).reset_index(drop=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'\nFor target "{target}" added {n_samples_after-n_samples_before} new samples!')
    print(f'New unique SMILES: {len(unique_smiles_extra)}')
    return df_train

train_df = add_extra_data(train_df, tc_smiles, 'Tc')
train_df = add_extra_data(train_df, tg_smiles, 'Tg')
train_df = add_extra_data(train_df, ktg_smiles, 'Tg')
train_df = add_extra_data(train_df, de_smiles, 'Density')




# Suppress RDKit warnings
warnings.filterwarnings("ignore")

# === Feature Category Definitions ===
FEATURE_CATEGORIES = {
    "rdkit": [],
    "graph": [
        'graph_diameter', 'avg_shortest_path', 'num_cycles',
        'betweenness_mean', 'betweenness_std', 'eigenvector_mean',
        'ring_4', 'max_degree', 'closeness_mean', 'katz_centrality_std',
        'heteroatom_ratio'
    ],
    "mordred": ['AMW', 'TIC2', 'naRing', 'MPC3'],
    "maccs": ['MACCS_Key130', 'MACCS_Key142', 'MACCS_Key066', 'MACCS_Key153'],
    "topo_torsion": ['TopologicalTorsion_Bit0512', 'TopologicalTorsion_Bit1296'],
    "atom_pair": ['AtomPair_B512_Bit0138', 'AtomPair_B512_Bit0448', 'AtomPair_B512_Bit0408'],
}

# === Useless RDKit Descriptors to Exclude ===
USELESS_COLS = [
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

# === RDKit descriptor initialization ===
def initialize_rdkit_features(useless_cols):
    return [d[0] for d in Descriptors.descList if d[0] not in useless_cols]

# === Feature Extraction Function ===
def compute_molecular_features_for_smiles(smiles, useless_cols):
    try:
        results = {}
        mol = Chem.MolFromSmiles(smiles)
        rdkit_names = FEATURE_CATEGORIES["rdkit"]

        # RDKit
        if mol:
            rdkit_vals = [d[1](mol) for d in Descriptors.descList if d[0] in rdkit_names]
            results.update(dict(zip(rdkit_names, rdkit_vals)))
        else:
            results.update({name: None for name in rdkit_names})

        # Graph
        graph_defaults = {k: 0 for k in FEATURE_CATEGORIES['graph']}
        if mol:
            try:
                adj = rdmolops.GetAdjacencyMatrix(mol)
                G = nx.from_numpy_array(adj)
                if nx.is_connected(G):
                    results['graph_diameter'] = nx.diameter(G)
                    results['avg_shortest_path'] = nx.average_shortest_path_length(G)
                    closeness = list(nx.closeness_centrality(G).values())
                    results['closeness_mean'] = np.mean(closeness)
                else:
                    results.update({k: 0 for k in ['graph_diameter', 'avg_shortest_path', 'closeness_mean']})

                results['max_degree'] = max([d for _, d in G.degree()]) if G else 0
                cycles = list(nx.cycle_basis(G))
                results['num_cycles'] = len(cycles)
                results['ring_4'] = sum(1 for c in cycles if len(c) == 4)

                try:
                    b = list(nx.betweenness_centrality(G).values())
                    results['betweenness_mean'] = np.mean(b)
                    results['betweenness_std'] = np.std(b)
                except:
                    results['betweenness_mean'] = results['betweenness_std'] = 0

                try:
                    e = list(nx.eigenvector_centrality(G).values())
                    results['eigenvector_mean'] = np.mean(e)
                except:
                    results['eigenvector_mean'] = 0

                try:
                    k = list(nx.katz_centrality(G, max_iter=1000).values())
                    results['katz_centrality_std'] = np.std(k)
                except:
                    results['katz_centrality_std'] = 0

                atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
                total = len(atom_types)
                hetero = total - atom_types.count('C')
                results['heteroatom_ratio'] = hetero / total if total > 0 else 0

            except:
                results.update(graph_defaults)
        else:
            results.update(graph_defaults)

        # Mordred
        mordred_defaults = {k: 0 for k in FEATURE_CATEGORIES['mordred']}
        if mol:
            try:
                calc = Calculator([
                    descriptors.Weight,
                    descriptors.InformationContent,
                    descriptors.RingCount,
                    descriptors.PathCount
                ])
                m = calc(mol)
                results['AMW'] = float(m.get('Weight.AMW', 0))
                results['TIC2'] = float(m.get('InformationContent.TIC2', 0))
                results['naRing'] = float(m.get('RingCount.naRing', 0))
                results['MPC3'] = float(m.get('PathCount.MPC3', 0))
            except:
                results.update(mordred_defaults)
        else:
            results.update(mordred_defaults)

        # MACCS
        maccs_defaults = {k: 0 for k in FEATURE_CATEGORIES['maccs']}
        if mol:
            try:
                fp = MACCSkeys.GenMACCSKeys(mol)
                bits = [int(x) for x in fp.ToBitString()]
                results['MACCS_Key130'] = bits[129]
                results['MACCS_Key142'] = bits[141]
                results['MACCS_Key066'] = bits[65]
                results['MACCS_Key153'] = bits[152]
            except:
                results.update(maccs_defaults)
        else:
            results.update(maccs_defaults)

        # Topological Torsion
        torsion_defaults = {k: 0 for k in FEATURE_CATEGORIES['topo_torsion']}
        if mol:
            try:
                fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
                bits = [int(x) for x in fp.ToBitString()]
                results['TopologicalTorsion_Bit0512'] = bits[512]
                results['TopologicalTorsion_Bit1296'] = bits[1296]
            except:
                results.update(torsion_defaults)
        else:
            results.update(torsion_defaults)

        # Atom Pair
        atom_pair_defaults = {k: 0 for k in FEATURE_CATEGORIES['atom_pair']}
        if mol:
            try:
                fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=512)
                bits = [int(x) for x in fp.ToBitString()]
                results['AtomPair_B512_Bit0138'] = bits[138]
                results['AtomPair_B512_Bit0448'] = bits[448]
                results['AtomPair_B512_Bit0408'] = bits[408]
            except:
                results.update(atom_pair_defaults)
        else:
            results.update(atom_pair_defaults)

        return results

    except Exception as e:
        print(f"Critical error on {smiles}: {e}")
        return None

# === Parallel Feature Processing ===
def process_molecular_features_parallel(df, useless_cols, n_jobs=4):
    if n_jobs == -1:
        n_jobs = cpu_count()

    print(f"\nğŸ”„ Using joblib with {n_jobs} parallel workers...")

    # âœ… Proper RDKit init before parallel
    FEATURE_CATEGORIES["rdkit"] = initialize_rdkit_features(useless_cols)

    smiles_list = df['SMILES'].tolist()

    results = Parallel(n_jobs=n_jobs, verbose=0)(
    delayed(lambda s: {"SMILES": s, **compute_molecular_features_for_smiles(s, useless_cols)})(smiles)
    for smiles in tqdm(smiles_list, desc="ğŸ”¬ Computing Features")
)

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("âš ï¸� No valid molecular features computed")
        return pd.DataFrame()

    df_result = pd.DataFrame(valid_results)
    df_result = df_result.replace([np.inf, -np.inf], np.nan)

    print("\nâœ… Feature extraction complete.")
    summarize_features(df_result)

    return df_result

# === Summary Reporter ===
def summarize_features(df):
    print("\nğŸ“Š Feature Summary by Category:")
    for cat, features in FEATURE_CATEGORIES.items():
        found = [f for f in features if f in df.columns]
        print(f" - {cat.upper():<12}: {len(found)} features")
    print(f"\nğŸ§¬ Total features: {df.shape[1]}")



if __name__ == "__main__":

    data = train_df

    feature_df = process_molecular_features_parallel(data, USELESS_COLS, n_jobs=4)



if __name__ == "__main__":

    data = new_data

    test_feature_df = process_molecular_features_parallel(data, USELESS_COLS, n_jobs=4)



# Assume feature_cols contains all your descriptor/fingerprint columns (exclude non-feature cols like 'SMILES' or targets)
feature_cols = [col for col in test_feature_df.columns if col not in ['SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]

# Convert all feature columns to numeric (in case some are strings or objects)
test_feature_df[feature_cols] = test_feature_df[feature_cols].apply(pd.to_numeric, errors='coerce')

# Replace infinities with NaN
test_feature_df[feature_cols] = test_feature_df[feature_cols].replace([np.inf, -np.inf], np.nan)

# Fill NaNs with column medians
test_feature_df[feature_cols] = test_feature_df[feature_cols].fillna(test_feature_df[feature_cols].median())

# **Extra fix: specifically fill any remaining NaNs in MaxPartialCharge with the median of that column**
if 'MaxPartialCharge' in test_feature_df.columns:
    median_val = test_feature_df['MaxPartialCharge'].median()
    test_feature_df['MaxPartialCharge'] = test_feature_df['MaxPartialCharge'].fillna(median_val)

# Clip extreme values to a reasonable range
max_threshold = 1e6
min_threshold = -1e6
test_feature_df[feature_cols] = test_feature_df[feature_cols].clip(lower=min_threshold, upper=max_threshold)

# Optional: log-transform columns known to have large ranges, e.g. 'Ipc', 'WPath', if present
for col in ['Ipc', 'WPath']:
    if col in test_feature_df.columns:
        # Clip negatives to zero before log1p to avoid errors
        test_feature_df[col] = np.log1p(test_feature_df[col].clip(lower=0))




feature_df
# feature_df.to_csv("feature_df.csv", index=False)
# from google.colab import files
# files.download("feature_df.csv")


# Keep only SMILES and target columns from train_df
target_columns = ['SMILES', 'Density', 'Tc', 'Tg', 'Rg', 'FFV']
targets_df = train_df[target_columns]

# Merge features with targets
final_df = pd.merge(feature_df, targets_df, on='SMILES')
final_df


# Assume feature_cols contains all your descriptor/fingerprint columns (exclude targets and SMILES)
feature_cols = [col for col in final_df.columns if col not in ['SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]

# Convert all feature columns to numeric (in case of object/string types)
final_df[feature_cols] = final_df[feature_cols].apply(pd.to_numeric, errors='coerce')

# Replace infinities with NaN
final_df[feature_cols] = final_df[feature_cols].replace([np.inf, -np.inf], np.nan)

# Fill NaNs with column medians
final_df[feature_cols] = final_df[feature_cols].fillna(final_df[feature_cols].median())

# Clip extreme values (choose a reasonable threshold)
max_threshold = 1e6
min_threshold = -1e6

final_df[feature_cols] = final_df[feature_cols].clip(lower=min_threshold, upper=max_threshold)

# Optional: log-transform columns known to have large ranges, e.g. 'Ipc', 'WPath', if present
for col in ['Ipc', 'WPath']:
    if col in final_df.columns:
        # Clip negatives to zero to avoid log issues, then log1p transform
        final_df[col] = np.log1p(final_df[col].clip(lower=0))




X = final_df.drop(columns=['SMILES', 'Density', 'Tc', 'Tg', 'Rg', 'FFV']) #features
y = final_df[['Density', 'Tc', 'Tg', 'Rg', 'FFV']] #targets

# Replace with your actual target column names
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Features are all columns except SMILES and targets
feature_cols = [col for col in final_df.columns if col not in ['SMILES'] + targets]
target_feature_map = {target: feature_cols for target in targets}



def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

def scaffold_split(df, smiles_col, test_size=0.2, random_state=42):
    df = df.reset_index(drop=True)
    scaffolds = df[smiles_col].apply(get_scaffold)
    scaffold_to_indices = {}
    for idx, scaffold in enumerate(scaffolds):
        if scaffold not in scaffold_to_indices:
            scaffold_to_indices[scaffold] = []
        scaffold_to_indices[scaffold].append(idx)

    scaffold_sets = sorted(scaffold_to_indices.items(), key=lambda x: len(x[1]), reverse=True)
    train_idx, test_idx = [], []
    n_total = len(df)
    n_test = int(n_total * test_size)

    for scaffold, indices in scaffold_sets:
        if len(test_idx) + len(indices) <= n_test:
            test_idx.extend(indices)
        else:
            train_idx.extend(indices)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df

def train_model_for_target(target, final_df, target_feature_map, param_dist, smiles_col='SMILES'):
    print(f"\nğŸ”§ Training model for target: {target}")
    data = final_df.dropna(subset=[target]).reset_index(drop=True)
    train_df, val_df = scaffold_split(data, smiles_col=smiles_col, test_size=0.2)

    X_train = train_df[target_feature_map[target]]
    y_train = train_df[target]
    X_val = val_df[target_feature_map[target]]
    y_val = val_df[target]

    base_model = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist, n_iter=20, cv=3,
        scoring='r2', random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    y_pred = model.predict(X_val)

    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    range_y = y_val.max() - y_val.min()
    normalized_mae = mae / range_y if range_y != 0 else np.nan

    print(f"âœ… RÂ² Score: {r2:.4f}")
    print(f"ğŸ“‰ MAE: {mae:.4f}")
    print(f"ğŸ“Š Range: {range_y:.4f} | Samples: {len(y_val)}")
    print(f"ğŸ“� Normalized MAE: {normalized_mae:.4f}")

    X_full = data[target_feature_map[target]]
    y_full = data[target]
    cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='r2')
    print(f"ğŸ“ˆ CV RÂ²: {cv_scores.mean():.4f} Â± {cv_scores.std():.4f}")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = np.array(target_feature_map[target])[indices[:20]]

    # plt.figure(figsize=(10, 5))
    # plt.bar(range(20), importances[indices[:20]], align="center")
    # plt.xticks(range(20), top_features, rotation=90)
    # plt.title(f"Top 20 Features for {target}")
    # plt.tight_layout()
    # plt.show()

    return model, y_val, y_pred, mae

# === USAGE ===

param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 30, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ['sqrt']
}

# --- Setup ---
predictions = {}
all_y_true = {}
trained_models = {}
maes = {}

# --- Define weights for each target ---
target_weights = {
    "Tg": 0.0006,
    "FFV": 0.1535,
    "Tc": 0.5787,
    "Density": 0.2561,
    "Rg": 0.0112
}

# --- Train each target ---
for target in targets:
    model, y_val, y_pred, mae = train_model_for_target(
        target, final_df, target_feature_map, param_dist, smiles_col='SMILES'
    )
    predictions[target] = pd.Series(y_pred, index=y_val.index)
    all_y_true[target] = y_val
    trained_models[target] = model
    maes[target] = mae

# --- Weighted MAE Calculation ---
wmae_total = 0.0
print("\nğŸ“Š Weighted MAE (wMAE) Breakdown:")
for target in targets:
    mae = maes[target]
    weight = target_weights.get(target, 0.0)
    weighted = mae * weight
    wmae_total += weighted
    print(f"{target}: MAE={mae:.4f}, Weight={weight:.4f}, Weighted={weighted:.4f}")
print(f"\nâœ… Final Weighted MAE (wMAE): {wmae_total:.6f}")



# Make predictions on test data
# Merge on 'SMILES' (adjust column if needed)
test_df_merged = new_data.merge(test_feature_df, on="SMILES", how="left")

# Confirm merged shape and check for missing values
print(f"âœ… Merged test_df with features: {test_df_merged.shape}")
print(test_df_merged.isnull().sum().sort_values(ascending=False).head(10))  # optional debug

test_predictions = {}

for target in targets:
    if target not in trained_models:
        print(f"âš ï¸� Skipping {target} (no trained model found)")
        continue

    model = trained_models[target]
    features = target_feature_map[target]

    if not set(features).issubset(test_df_merged.columns):
        missing = set(features) - set(test_df_merged.columns)
        print(f"âš ï¸� Skipping {target} (missing features: {missing})")
        continue

    X_test = test_df_merged[features].copy()
    
    # Handle missing and infinite values
    medians = X_test.median()
    X_test = X_test.fillna(medians)       # Fill NaNs with median
    X_test = X_test.fillna(0)             # Fill any remaining NaNs with 0
    X_test = X_test.replace([np.inf, -np.inf], 0)  # Replace inf/-inf with 0

    nan_count = X_test.isnull().sum().sum()
    print(f"NaNs after fill for {target}: {nan_count}")
    if nan_count > 0:
        raise ValueError(f"Still have NaNs after fill for {target}")

    y_test_pred = model.predict(X_test)
    test_predictions[target] = y_test_pred
    print(f"âœ… Predicted {target} on test data.")

# Create a DataFrame for submission
submission_df = new_data[['SMILES']].copy()

# Add predictions
for target in test_predictions:
    submission_df[target] = test_predictions[target]

# Save to CSV (no index)
submission_df.to_csv("submission.csv", index=False)

print("âœ… Saved submission to submission.csv")
submission_df.head()

from IPython.display import display
from ipywidgets import Widget

for widget in Widget.widgets.values():
    widget.close()




import nbformat
from IPython import get_ipython
import os
import glob

def clean_current_notebook():
    ipython = get_ipython()
    if not ipython:
        raise RuntimeError("Not running inside IPython/Jupyter")

    # Try to find the current notebook filename
    candidates = glob.glob("*.ipynb")
    if not candidates:
        print("â�Œ No .ipynb file found in the current directory.")
        return
    notebook_name = candidates[0]
    print(f"ğŸ§¹ Cleaning notebook: {notebook_name}")

    # Load the notebook
    with open(notebook_name, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Clean top-level widgets metadata
    if 'widgets' in nb.get('metadata', {}):
        print("  - Removing top-level widgets metadata")
        del nb['metadata']['widgets']

    # Clean individual cells
    for cell in nb.get('cells', []):
        cell['metadata'] = {}

        # Remove widget-related outputs
        if 'outputs' in cell:
            new_outputs = []
            for output in cell['outputs']:
                data = output.get('data', {})
                if 'application/vnd.jupyter.widget-view+json' in data:
                    continue  # skip widget output
                new_outputs.append(output)
            cell['outputs'] = new_outputs

    # Save cleaned notebook
    with open(notebook_name, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("âœ… Notebook cleaned. Try saving a version again.")

# Run it
clean_current_notebook()


