!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import numpy as np
import optuna
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import DataStructs

from typing import Dict, List, Tuple


# load data from csv files
train_full = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')

# external data
tc_smiles = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')


def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList]

desc_names = [desc[0] for desc in Descriptors.descList]

descriptors = [compute_all_descriptors(smi) for smi in train_full['SMILES'].to_list()]
descriptors = pd.DataFrame(descriptors, columns=desc_names)
train_full = pd.concat([train_full, descriptors],axis=1)

# drop SMILES columns
train_full = train_full.drop(columns=['SMILES'])

# drop RDKit descriptors that are missing for some reason
cols = ['Ipc', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW']
train_full = train_full.drop(columns=cols)
desc_names = sorted(set(desc_names) - set(cols))

# are there even any such values?
def replace_nan_inf_with_stat(df, cols, method='median'):
    df_cleaned = df.copy()

    for col in cols:
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            # Replace inf/-inf with NaN
            df_cleaned[col] = df_cleaned[col].replace([np.inf, -np.inf], np.nan)

            # Compute fill value
            if method == 'median':
                fill_value = df_cleaned[col].median(skipna=True)
            elif method == 'mean':
                fill_value = df_cleaned[col].mean(skipna=True)
            else:
                raise ValueError("Method must be either 'mean' or 'median'.")

            # Fill NaNs
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)

    return df_cleaned

train_full = replace_nan_inf_with_stat(train_full, desc_names, method='median')


TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
N_CV_SPLITS = 5

kf = KFold(n_splits=N_CV_SPLITS, random_state=42, shuffle=True)


def objective(trial):
    """ Objective function for Optuna. """
    params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'random_state': 77,
        'verbosity': 0,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'n_estimators': 10_000,
        'max_delta_step': trial.suggest_float('max_delta_step', 1e-3, 10, log=True),
        'gamma': trial.suggest_float('gamma', 1e-3, 10, log=True),
        'use_label_encoder': False,
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 25, 25),
        'enable_categorical': False,
        'n_jobs': 1
    }

    kf = KFold(n_splits=N_CV_SPLITS, random_state=42, shuffle=True)
    oof_preds = np.zeros(train.shape[0])

    for train_idx, val_idx in kf.split(train):
        X_train_fold = train[desc_names].iloc[train_idx].values
        y_train_fold = train[TARGET_COL].iloc[train_idx].values
        X_val_fold = train[desc_names].iloc[val_idx].values
        y_val_fold = train[TARGET_COL].iloc[val_idx].values

        model = xgb.XGBRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  verbose=False
                  )

        oof_preds[val_idx] = model.predict(X_val_fold)

    return mean_absolute_error(train[TARGET_COL].values, oof_preds)


import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

for TARGET_COL in TARGETS:
    train = train_full[train_full[TARGET_COL].notnull()]  # filter rows
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10_000, timeout=60*60*0.5)

    worst_mae = max(trial.value for trial in study.trials if trial.value is not None)

    print(f"Best trial: {study.best_trial.number}/{len(study.trials)}")
    print(f"  MAE: {study.best_value}")
    print(f"  worst MAE: {worst_mae}")
    print('  Best hyperparameters:')
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

