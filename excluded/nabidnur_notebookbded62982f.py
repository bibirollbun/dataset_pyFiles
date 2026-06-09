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


pip install numpy pandas scikit-learn lightgbm rdkit-pypi



import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# ----------------------------
# 1. Load data
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')  # Replace with your path if needed
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']



# ----------------------------
# 2. Feature extraction with RDKit

def compute_morgan_fp(mol, radius=2, n_bits=2048):
    """Return Morgan fingerprint as numpy array"""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)  # fixed size
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def compute_physchem_descriptors(mol):
    """Return list of common phys-chem descriptors"""
    desc = []
    desc.append(Descriptors.MolWt(mol))       # Molecular weight
    desc.append(Descriptors.MolLogP(mol))     # LogP
    desc.append(Descriptors.TPSA(mol))        # Topological polar surface area
    desc.append(Descriptors.NumRotatableBonds(mol))
    desc.append(Descriptors.NumHAcceptors(mol))
    desc.append(Descriptors.NumHDonors(mol))
    return np.array(desc)

def extract_features(smiles_list):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Return zeros if invalid SMILES
            fp = np.zeros(2048, dtype=np.int8)
            desc = np.zeros(6, dtype=np.float32)
        else:
            fp = compute_morgan_fp(mol)
            desc = compute_physchem_descriptors(mol)
        combined = np.concatenate([fp, desc])
        features.append(combined)
    return np.array(features)

print("Extracting features from training data...")
X_train = extract_features(train_df['SMILES'])

print("Extracting features from test data...")
X_test = extract_features(test_df['SMILES'])



# ----------------------------
# 3. Prepare targets array (labels)
y_train = train_df[TARGETS].values

# ----------------------------
# 4. Train LightGBM models per target with 5-fold cross-validation

NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

test_preds = np.zeros((X_test.shape[0], len(TARGETS)))
oof_preds = np.zeros_like(y_train)



from lightgbm import early_stopping, log_evaluation

for i, target in enumerate(TARGETS):
    print(f"\nTraining for target: {target}")
    fold_maes = []
    test_fold_preds = np.zeros((X_test.shape[0], NFOLDS))
    
    # Use only samples with non-missing target values
    idx_notnull = ~np.isnan(y_train[:, i])
    X_tr_full = X_train[idx_notnull]
    y_tr_full = y_train[idx_notnull, i]
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr_full)):
        print(f" Fold {fold + 1}")
        
        X_tr, X_val = X_tr_full[train_idx], X_tr_full[val_idx]
        y_tr, y_val = y_tr_full[train_idx], y_tr_full[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': 42 + fold
        }
        
        model = lgb.train(params,
                          train_data,
                          num_boost_round=1000,
                          valid_sets=[train_data, val_data],
                          callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=100)],
                         )
        
        # Out-of-fold predictions
        oof_preds[idx_notnull, i][val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Test predictions for this fold
        test_fold_preds[:, fold] = model.predict(X_test, num_iteration=model.best_iteration)
        
        fold_maes.append(mean_absolute_error(y_val, oof_preds[idx_notnull, i][val_idx]))
    
    print(f" Average MAE for {target}: {np.mean(fold_maes):.4f}")
    
    # Average test predictions across folds
    test_preds[:, i] = test_fold_preds.mean(axis=1)



# ----------------------------
# 5. Evaluate overall MAE on out-of-fold predictions (only on non-null targets)
mask = ~np.isnan(y_train)
overall_mae = mean_absolute_error(y_train[mask], oof_preds[mask])
print(f"\nOverall OOF MAE: {overall_mae:.4f}")

# ----------------------------
# 6. Prepare submission file
submission = test_df[['id']].copy()
for i, target in enumerate(TARGETS):
    submission[target] = test_preds[:, i]

submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")


