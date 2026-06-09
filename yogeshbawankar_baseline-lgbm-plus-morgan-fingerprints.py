# The notebook installs a specific version of RDKit from a local wheel file.
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
import os

# RDKit imports for cheminformatics
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger

# Suppress RDKit logging messages to keep the output clean
RDLogger.DisableLog('rdApp.*')

# Avoids a warning from the transformers library if it were used
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Load the datasets
try:
    train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
except FileNotFoundError:
    print("Ensure the dataset is in the correct path. Using placeholder paths for now.")
    # Add placeholder paths if needed, e.g., 'train.csv'
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')


# Define the target columns to predict
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


# --- 3.1 RDKit Molecular Descriptors ---
print("Generating RDKit molecular descriptors...")
def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList]

desc_names = [desc[0] for desc in Descriptors.descList]
train_descriptors_df = pd.DataFrame([compute_all_descriptors(smi) for smi in train['SMILES']], columns=desc_names)
test_descriptors_df = pd.DataFrame([compute_all_descriptors(smi) for smi in test['SMILES']], columns=desc_names)


# --- 3.2 Morgan Fingerprints ---
print("Generating Morgan Fingerprints...")
def compute_morgan_fingerprints(df, n_bits=2048, radius=2, n_components_desired=100):
    mols = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits) for m in mols]
    fps_np = np.array(fps, dtype=np.float32)
    
    # *** FIX: Dynamically adjust n_components for TruncatedSVD ***
    # The number of components must be less than the number of samples.
    n_samples = fps_np.shape[0]
    n_components = min(n_components_desired, n_samples - 1)
    
    print(f"  - Using {n_components} components for SVD from {n_samples} samples.")

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    fps_svd = svd.fit_transform(fps_np)
    
    fp_df = pd.DataFrame(fps_svd, columns=[f'MFP_{i}' for i in range(n_components)])
    return fp_df

train_fp_df = compute_morgan_fingerprints(train)
test_fp_df = compute_morgan_fingerprints(test)


# --- 3.3 Combine All Features ---
print("Combining all features...")
# Align columns for test set in case SVD produced fewer components
if test_fp_df.shape[1] < train_fp_df.shape[1]:
    missing_cols = set(train_fp_df.columns) - set(test_fp_df.columns)
    for c in missing_cols:
        test_fp_df[c] = 0
test_fp_df = test_fp_df[train_fp_df.columns] # Ensure order is the same

descriptor_feats = list(train_descriptors_df.columns)
fp_feats = list(train_fp_df.columns)
feats = descriptor_feats + fp_feats

train_features = pd.concat([train_descriptors_df, train_fp_df], axis=1)
test_features = pd.concat([test_descriptors_df, test_fp_df], axis=1)

# Add original features back into main dataframes
train = pd.concat([train, train_features], axis=1)
test = pd.concat([test, test_features], axis=1)



print("Saving features to parquet files...")
train_features.to_parquet('train_features.parquet')
test_features.to_parquet('test_features.parquet')


def lgb_kfold(train_df, test_df, target, feats, folds):
    params = {
         'objective' : 'mae', 'metric' : 'mae', 'num_leaves': 31,
         'min_data_in_leaf': 30, 'learning_rate': 0.01, 'max_depth': -1,
         'max_bin': 256, 'boosting': 'gbdt', 'feature_fraction': 0.7,
         'bagging_freq': 1, 'bagging_fraction': 0.7, 'bagging_seed': 42,
         "lambda_l1": 1, "lambda_l2": 1, 'verbosity': -1,
         'num_boost_round' : 20000, 'device_type' : 'cpu'
    }
    sub_preds = np.zeros(test_df.shape[0])

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df[target])):
        print(f'--- Fold: {n_fold} ---')
        train_x, train_y = train_df[feats].iloc[train_idx], train_df[target].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[target].iloc[valid_idx]

        dtrain = lgb.Dataset(train_x, label=train_y)
        dval = lgb.Dataset(valid_x, label=valid_y, reference=dtrain)
        callbacks = [lgb.log_evaluation(period=100), lgb.early_stopping(200, verbose=False)]

        bst = lgb.train(params, dtrain, valid_sets=[dval], callbacks=callbacks)
        sub_preds += bst.predict(test_df[feats], num_iteration=bst.best_iteration) / folds.n_splits

    return sub_preds

# --- Setup for training ---
n_splits = 5
seed = 817
folds = KFold(n_splits=n_splits, random_state=seed, shuffle=True)

# --- Loop through targets and train a model for each ---
for t in targets:
    print(f'\n========== Training model for target: {t} ==========')
    train_subset = train[train[t].notnull()].reset_index(drop=True)
    sub_preds = lgb_kfold(train_subset, test, t, feats, folds)
    test[t] = sub_preds


print("\nGenerating submission file...")
submission_df = test[['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")

