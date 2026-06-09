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


# Cell 1: Imports & File Listing
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

import networkx as nx
from rdkit.Chem import AllChem, Descriptors, rdmolops
from rdkit import Chem

pd.set_option('display.max_columns', None)

print("âœ… All libraries imported successfully!")



# Cell 2: Configuration
class CFG:
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    SEED = 42
    FOLDS = 5

useless_cols = [
    'MaxPartialCharge', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',  # NaN/constant/correlated cols etc...
    'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur', 'fr_benzodiazepine', 'fr_dihydropyridine',
    # ... add further as needed for your project
]
print(f"ğŸ�¯ Target properties: {CFG.TARGETS}")



# Cell 3: Data Loading (FIXED)
def load_all_datasets():
    print("ğŸ“� Loading datasets...")
    train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
    ss = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
    external_data = {}
    try:
        # FIXED: Remove the "..." typo in the path
        external_data['tc_smiles'] = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
        external_data['tg_smiles'] = pd.read_csv('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv')
        external_data['ktg_smiles'] = pd.read_excel('/kaggle/input/smiles-extra-data/data_tg3.xlsx')
        external_data['de_smiles'] = pd.read_excel('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
        print("âœ… External datasets loaded successfully")
    except Exception as e:
        print(f"âš ï¸� External datasets not found: {e}")
        external_data = {}
    return train, test, ss, external_data

train_df, test_df, submission_template, external_datasets = load_all_datasets()



# Cell 4: SMILES Canonicalization & Ext. Data Augmentation Setup
def make_smile_canonical(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return np.nan
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return np.nan

print("ğŸ”„ Canonicalizing SMILES...")
train_df['SMILES'] = train_df['SMILES'].apply(make_smile_canonical)
test_df['SMILES'] = test_df['SMILES'].apply(make_smile_canonical)

def preprocess_external_datasets(external_data):
    processed = {}
    if 'tc_smiles' in external_data:
        df = external_data['tc_smiles'].copy()
        df.rename(columns={'TC_mean': 'Tc'}, inplace=True)
        processed['tc_smiles'] = df
    if 'tg_smiles' in external_data:
        df = external_data['tg_smiles'].copy()
        df.rename(columns={'Tg (C)': 'Tg'}, inplace=True)
        processed['tg_smiles'] = df
    if 'ktg_smiles' in external_data:
        df = external_data['ktg_smiles'].copy()
        df.rename(columns={'Tg [K]': 'Tg'}, inplace=True)
        df['Tg'] = df['Tg'] - 273.15
        processed['ktg_smiles'] = df
    if 'de_smiles' in external_data:
        df = external_data['de_smiles'].copy()
        df.rename(columns={'density(g/cm3)': 'Density'}, inplace=True)
        df['SMILES'] = df['SMILES'].apply(make_smile_canonical)
        df = df[(df['SMILES'].notnull()) & (df['Density'].notnull()) & (df['Density'] != 'nylon')]
        df['Density'] = df['Density'].astype(float) - 0.118
        processed['de_smiles'] = df
    return processed

processed_external = preprocess_external_datasets(external_datasets)



# Cell 5: External Data Augmentation
def add_extra_data(df_train, df_extra, target):
    df_extra['SMILES'] = df_extra['SMILES'].apply(make_smile_canonical)
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()

    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    for smile in df_train[df_train[target].notnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            cross_smiles.remove(smile)
    for smile in cross_smiles:
        external_value = df_extra[df_extra['SMILES'] == smile][target].values[0]
        df_train.loc[df_train['SMILES'] == smile, target] = external_value

    new_samples = df_extra[df_extra['SMILES'].isin(unique_smiles_extra)]
    df_train = pd.concat([df_train, new_samples], axis=0, ignore_index=True)
    return df_train

print("ğŸ“ˆ Augmenting training data with external datasets...")
if processed_external:
    if 'tc_smiles' in processed_external:
        train_df = add_extra_data(train_df, processed_external['tc_smiles'], 'Tc')
    if 'tg_smiles' in processed_external:
        train_df = add_extra_data(train_df, processed_external['tg_smiles'], 'Tg')
    if 'ktg_smiles' in processed_external:
        train_df = add_extra_data(train_df, processed_external['ktg_smiles'], 'Tg')
    if 'de_smiles' in processed_external:
        train_df = add_extra_data(train_df, processed_external['de_smiles'], 'Density')
print(f"ğŸ�¯ Final training data size: {len(train_df)} samples")



# Cell 6: Feature Engineering (MEMORY OPTIMIZED)
def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]
        return [None] * len(desc_names)
    descriptors = []
    for desc_name, desc_func in Descriptors.descList:
        if desc_name not in useless_cols:
            try:
                value = desc_func(mol)
                descriptors.append(value)
            except:
                descriptors.append(None)
    return descriptors

def compute_graph_features(smiles, graph_feats):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            graph_feats['graph_diameter'].append(0)
            graph_feats['avg_shortest_path'].append(0)
            graph_feats['num_cycles'].append(0)
            return
        adj = rdmolops.GetAdjacencyMatrix(mol)
        G = nx.from_numpy_array(adj)
        if nx.is_connected(G):
            diameter = nx.diameter(G)
            avg_path = nx.average_shortest_path_length(G)
        else:
            diameter = 0
            avg_path = 0
        num_cycles = len(list(nx.cycle_basis(G)))
        graph_feats['graph_diameter'].append(diameter)
        graph_feats['avg_shortest_path'].append(avg_path)
        graph_feats['num_cycles'].append(num_cycles)
    except:
        graph_feats['graph_diameter'].append(0)
        graph_feats['avg_shortest_path'].append(0)
        graph_feats['num_cycles'].append(0)

def preprocessing(df):
    print(f"ğŸ§ª Computing features for {len(df)} molecules...")
    desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]
    
    # Process in smaller batches to avoid memory issues
    batch_size = 1000
    all_descriptors = []
    all_graph_feats = {'graph_diameter': [], 'avg_shortest_path': [], 'num_cycles': []}
    
    for i in range(0, len(df), batch_size):
        batch_smiles = df['SMILES'].iloc[i:i+batch_size].tolist()
        batch_descriptors = [compute_all_descriptors(smi) for smi in batch_smiles]
        all_descriptors.extend(batch_descriptors)
        
        for smile in batch_smiles:
            compute_graph_features(smile, all_graph_feats)
    
    result = pd.concat([
        pd.DataFrame(all_descriptors, columns=desc_names),
        pd.DataFrame(all_graph_feats)
    ], axis=1)
    result = result.replace([-np.inf, np.inf], np.nan)
    return result

print("ğŸ”¬ Feature Engineering Phase")
train_features = preprocessing(train_df)
test_features = preprocessing(test_df)
train_df = pd.concat([train_df.reset_index(drop=True), train_features], axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), test_features], axis=1)
print(f"âœ… Feature engineering complete! Training: {train_df.shape} Test: {test_df.shape}")



# Cell 7: Feature Transformations
print("ğŸ”§ Applying feature transformations...")
if 'Ipc' in train_df.columns:
    train_df['Ipc'] = np.log10(train_df['Ipc'].clip(lower=1e-10))
    test_df['Ipc'] = np.log10(test_df['Ipc'].clip(lower=1e-10))

feature_cols = train_df.columns.difference(['id', 'SMILES'] + CFG.TARGETS)
for col in feature_cols:
    train_df[col] = train_df[col].replace([-np.inf, np.inf], np.nan)
    test_df[col] = test_df[col].replace([-np.inf, np.inf], np.nan)
    col_mean = train_df[col].mean()
    train_df[col] = train_df[col].fillna(col_mean)
    test_df[col] = test_df[col].fillna(col_mean)
print("âœ… Data preprocessing completed")



# Cell 8: Final NaN Fix
print("ğŸ©¹ Filling residual NaNs in numeric features...")
num_cols = train_df.columns.difference(['id', 'SMILES'] + CFG.TARGETS)
for c in num_cols:
    if train_df[c].isna().any():
        grp_mean = train_df.groupby('SMILES')[c].transform('mean')
        train_df[c] = train_df[c].fillna(grp_mean).fillna(0)
        test_df[c] = test_df[c].fillna(grp_mean.mean()).fillna(0)
print("âœ… NaN filling complete")
print(f"Remaining NaNs in train: {train_df[num_cols].isna().sum().sum()}")
print(f"Remaining NaNs in test: {test_df[num_cols].isna().sum().sum()}")



# Cell 9: Remove Duplicate Columns (Fixes CatBoost Error)
def remove_duplicate_columns(df):
    # Only keep the first occurrence of each column name
    _, idx = np.unique(df.columns, return_index=True)
    if len(df.columns) != len(idx):
        print("Duplicate columns removed!")
    return df.iloc[:, np.sort(idx)]

train_df = remove_duplicate_columns(train_df)
test_df = remove_duplicate_columns(test_df)




# Cell 9A: Comprehensive Duplicate Removal & Validation
def remove_duplicate_columns_comprehensive(df):
    print("ğŸ”� Checking for duplicate columns...")
    original_cols = len(df.columns)
    
    # Find duplicates
    duplicates = df.columns[df.columns.duplicated()].tolist()
    if duplicates:
        print(f"âš ï¸� Found duplicate columns: {duplicates}")
    
    # Remove duplicates - keep first occurrence
    _, idx = np.unique(df.columns, return_index=True)
    df_clean = df.iloc[:, np.sort(idx)]
    
    print(f"âœ… Columns: {original_cols} â†’ {len(df_clean.columns)} (removed {original_cols - len(df_clean.columns)} duplicates)")
    return df_clean

# Apply comprehensive cleaning
train_df = remove_duplicate_columns_comprehensive(train_df)
test_df = remove_duplicate_columns_comprehensive(test_df)

# Validate column consistency
train_feature_cols = set(train_df.columns) - {'id', 'SMILES'} - set(CFG.TARGETS)
test_feature_cols = set(test_df.columns) - {'id', 'SMILES'}

print(f"ğŸ“Š Train features: {len(train_feature_cols)}")
print(f"ğŸ“Š Test features: {len(test_feature_cols)}")

if train_feature_cols != test_feature_cols:
    print("âš ï¸� Feature mismatch detected - aligning datasets...")
    common_features = train_feature_cols & test_feature_cols
    print(f"ğŸ”§ Using {len(common_features)} common features")
else:
    print("âœ… Feature sets aligned")



# Cell 10: Target Dataset Preparation
def prepare_target_datasets(train_df, test_df):
    print("ğŸ�¯ Preparing target-specific datasets...")
    target_datasets = {}
    feature_cols = [col for col in train_df.columns if col not in ['id', 'SMILES'] + CFG.TARGETS]
    for target in CFG.TARGETS:
        target_data = train_df[['SMILES', target]].copy().dropna()
        features_df = train_df[['SMILES'] + feature_cols]
        target_with_features = target_data.merge(features_df, on='SMILES', how='left')
        target_with_features = target_with_features.drop('SMILES', axis=1).dropna(axis=1, how='all')
        target_datasets[target] = target_with_features
        print(f"  âœ… {target}: {len(target_with_features)} samples")
    test_features = test_df[feature_cols].fillna(0)
    return target_datasets, test_features
target_datasets, test_features = prepare_target_datasets(train_df, test_df)



# Cell 11: Tuned Model Definitions
def get_model_for_target(target):
    model_configs = {
        'Tg': HistGradientBoostingRegressor(learning_rate=0.05, max_depth=6, max_iter=500, random_state=CFG.SEED),
        'FFV': ExtraTreesRegressor(n_estimators=200, max_features=0.8, min_samples_leaf=2, random_state=CFG.SEED),
        'Tc': CatBoostRegressor(learning_rate=0.03, depth=6, iterations=1200, verbose=False, random_state=CFG.SEED),
        'Density': ExtraTreesRegressor(n_estimators=200, max_features=0.8, min_samples_leaf=2, random_state=CFG.SEED),
        'Rg': ExtraTreesRegressor(n_estimators=200, max_features=0.8, min_samples_leaf=2, random_state=CFG.SEED)
    }
    return model_configs[target]

def train_model(train_data, test_data, model, target, submission=False):
    X = train_data.drop(target, axis=1)
    y = train_data[target].copy()
    if not submission:
        # StratifiedKFold for better validation
        bin_edges = np.histogram(y, bins=5)[1]
        bins = np.digitize(y, bin_edges[1:-1])
        skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
        maes = []
        for tr_idx, va_idx in skf.split(X, bins):
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[va_idx])
            maes.append(mean_absolute_error(y.iloc[va_idx], pred))
        return np.mean(maes)
    else:
        model.fit(X, y)
        predictions = model.predict(test_data)
        return predictions



# Cell 12: Model Evaluation
print("ğŸ“Š Model Evaluation Phase")
print("=" * 50)

evaluation_results = {}
for target in CFG.TARGETS:
    if target in target_datasets:
        print(f"\nğŸ�¯ Evaluating {target}...")
        target_data = target_datasets[target]
        model = get_model_for_target(target)
        mae_score = train_model(target_data, test_features, model, target, submission=False)
        evaluation_results[target] = mae_score
        print(f"  âœ… {target} MAE: {mae_score:.6f}")

print("\nğŸ“Š EVALUATION SUMMARY")
print("=" * 30)
for target, score in evaluation_results.items():
    print(f"{target:8}: {score:.6f} MAE")



# Cell 13: 3-Model Simple Ensemble for Submission
def train_ensemble(train_data, test_data, target, n_models=3):
    predictions = []
    for seed in range(CFG.SEED, CFG.SEED + n_models):
        model = get_model_for_target(target)
        model.set_params(random_state=seed)
        model.fit(train_data.drop(target, axis=1), train_data[target])
        preds = model.predict(test_data)
        predictions.append(preds)
    return np.mean(predictions, axis=0)

print("\nğŸš€ Training Final Models for Submission")
print("=" * 50)
final_predictions = {}

for target in CFG.TARGETS:
    if target in target_datasets:
        print(f"\nğŸ�¯ Training final {target} ensemble...")
        target_data = target_datasets[target]
        predictions = train_ensemble(target_data, test_features, target, n_models=3)
        final_predictions[target] = predictions
        print(f"  âœ… {target} predictions: {predictions.min():.4f} - {predictions.max():.4f}")
    else:
        mean_val = train_df[target].mean()
        final_predictions[target] = np.full(len(test_df), mean_val)
        print(f"  âš ï¸� {target} using mean value: {mean_val:.4f}")

print("âœ… All models trained successfully!")



# Cell 14: Submission Creation
def create_submission(test_df, predictions, filename='submission.csv'):
    print("ğŸ“� Creating submission file...")
    submission = pd.DataFrame({'id': test_df['id']})
    for target in CFG.TARGETS:
        if target in predictions:
            submission[target] = predictions[target]
        else:
            mean_val = train_df[target].mean()
            submission[target] = mean_val
            print(f"  âš ï¸� Using mean value for {target}: {mean_val:.4f}")
    print("ğŸ”§ Applying physical constraints...")
    if 'FFV' in submission.columns:
        submission['FFV'] = np.clip(submission['FFV'], 0.0, 1.0)
    if 'Density' in submission.columns:
        submission['Density'] = np.maximum(submission['Density'], 0.1)
    if 'Tc' in submission.columns:
        submission['Tc'] = np.maximum(submission['Tc'], 0.001)
    if 'Rg' in submission.columns:
        submission['Rg'] = np.maximum(submission['Rg'], 0.1)
    submission.to_csv(filename, index=False)
    print(f"âœ… Submission saved as '{filename}'")
    return submission

final_submission = create_submission(test_df, final_predictions)

print("\nğŸ�‰ SUBMISSION SUMMARY")
print("=" * 50)
print(f"ğŸ“Š Test samples: {len(final_submission)}")
print(f"ğŸ�¯ Targets: {len(CFG.TARGETS)}\n")
print("ğŸ“ˆ Prediction Ranges:")
for target in CFG.TARGETS:
    if target in final_submission.columns:
        values = final_submission[target]
        print(f"  {target:8}: {values.min():.4f} - {values.max():.4f} (mean: {values.mean():.4f})")
print(f"\nâœ… Solution Complete!")
print(f"ğŸ“Š Training samples (after augmentation): {len(train_df):,}")
print(f"ğŸ§ª Features generated: {len([col for col in train_df.columns if col not in ['id', 'SMILES'] + CFG.TARGETS]):,}")
print(f"ğŸ�¯ Models trained: {len([t for t in CFG.TARGETS if t in target_datasets])}")
print("\nğŸ�† Final submission ready!")
print(final_submission.head())





