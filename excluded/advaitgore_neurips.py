import numpy as np
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/


import numpy as np
import pandas as pd 

DATA_DIR = "/kaggle/input/neurips-open-polymer-prediction-2025"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

SMILES_COL = "SMILES"
TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]


import os
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import random
import warnings
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from mordred import Calculator, descriptors 

SEED = 42
random.seed(SEED); np.random.seed(SEED)
pd.set_option("display.max_columns", 200)
warnings.filterwarnings("ignore", category=UserWarning)


import torch
print("GPU available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


MORGAN_BITS = 1024
mg2 = GetMorganGenerator(radius=2, fpSize=MORGAN_BITS)
mg3 = GetMorganGenerator(radius=3, fpSize=MORGAN_BITS)
calc = Calculator(descriptors, ignore_3D=True)

def safe_mol(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m: Chem.SanitizeMol(m)
        return m
    except:
        return None

def extract_rdkit_features(m):
    if not m: return {}
    feats = {
        'MolWt': Descriptors.MolWt(m), 'HeavyAtom': Descriptors.HeavyAtomCount(m),
        'MR': Crippen.MolMR(m), 'TPSA': rdMolDescriptors.CalcTPSA(m),
        'LogP': Crippen.MolLogP(m), 'HBD': Lipinski.NumHDonors(m),
        'HBA': Lipinski.NumHAcceptors(m), 'RotBonds': Lipinski.NumRotatableBonds(m),
        'RingCount': rdMolDescriptors.CalcNumRings(m), 'AromRings': rdMolDescriptors.CalcNumAromaticRings(m),
        'AliphRings': rdMolDescriptors.CalcNumAliphaticRings(m), 'FracCSP3': rdMolDescriptors.CalcFractionCSP3(m),
    }
    atoms = [a.GetAtomicNum() for a in m.GetAtoms()]
    tot = len(atoms) or 1
    for elem in [6,1,7,8,9,17,16,14]:
        feats[f'Frac_{elem}'] = atoms.count(elem) / tot
    feats['HeteroFrac'] = sum(1 for a in atoms if a not in [1,6]) / tot
    feats['AromFrac'] = sum(1 for a in m.GetAtoms() if a.GetIsAromatic()) / tot
    feats['Arom_x_Hetero'] = feats['AromRings'] * feats['HeteroFrac']
    feats['TPSA_div_LogP'] = feats['TPSA'] / (abs(feats['LogP']) + 1e-6)
    feats['MolWt_per_Rot'] = feats['MolWt'] / (feats['RotBonds'] + 1)
    feats['PackIdx'] = feats['AromRings'] - feats['RotBonds']
    feats['MobilityIdx'] = feats['RotBonds'] + 5 * feats['FracCSP3']
    return feats

def build_features(df):
    mols = [safe_mol(s) for s in df[SMILES_COL]]
    # RDKit
    rdkit_list = [extract_rdkit_features(m) for m in mols]
    rdkit_df = pd.DataFrame(rdkit_list).fillna(0)
    # Morgan
    fp2_list = []; fp3_list = []
    for m in mols:
        if m:
            fp2 = mg2.GetFingerprint(m); fp3 = mg3.GetFingerprint(m)
            arr2 = np.zeros(MORGAN_BITS, dtype=np.uint8); arr3 = np.zeros(MORGAN_BITS, dtype=np.uint8)
            AllChem.DataStructs.ConvertToNumpyArray(fp2, arr2)
            AllChem.DataStructs.ConvertToNumpyArray(fp3, arr3)
            fp2_list.append(arr2); fp3_list.append(arr3)
        else:
            fp2_list.append(np.zeros(MORGAN_BITS, dtype=np.uint8))
            fp3_list.append(np.zeros(MORGAN_BITS, dtype=np.uint8))
    fp_df = pd.DataFrame(np.hstack([fp2_list, fp3_list]), 
                         columns=[f'FP2_{i}' for i in range(MORGAN_BITS)] + [f'FP3_{i}' for i in range(MORGAN_BITS)])
    # Mordred (limited to numeric, fill NaN)
    mordred_df = calc.pandas([m for m in mols if m is not None])
    mordred_df = mordred_df.select_dtypes(include=[np.number]).fillna(0)
    return pd.concat([rdkit_df, fp_df, mordred_df], axis=1)

X_train = build_features(train)
X_test = build_features(test)

# Align columns (add missing in test with 0s)
common_cols = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]
missing_cols = list(set(X_train.columns) - set(X_test.columns))
if missing_cols:
    X_test[missing_cols] = 0
X_test = X_test[X_train.columns]  # Order match

print(f"Features generated: {X_train.shape[1]}")


# import optuna
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_absolute_error
# import lightgbm as lgb
# from catboost import CatBoostRegressor
# import xgboost as xgb
# import torch  # For GPU check

# def objective(trial, X, y, model_type):
#     has_gpu = torch.cuda.is_available()
#     print(f"Using GPU for {model_type}: {has_gpu}")  # Verify per trial
    
#     if model_type == 'lgb':
#         params = {
#             'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#             'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
#             'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
#             'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
#             'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#             'device': 'gpu' if has_gpu else 'cpu'  # Enable GPU
#         }
#         model = lgb.LGBMRegressor(**params, random_state=SEED, verbosity=-1)
#     elif model_type == 'cat':
#         params = {
#             'depth': trial.suggest_int('depth', 4, 10),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#             'iterations': trial.suggest_int('iterations', 1000, 3000),
#             'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
#             'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#             'task_type': 'GPU' if has_gpu else 'CPU',  # Enable GPU
#             'bootstrap_type': 'Bernoulli'  # Required for subsample
#         }
#         # Fix for GPU error: Set colsample_bylevel=1.0 if GPU (avoids rsm restriction)
#         if has_gpu:
#             params['colsample_bylevel'] = 1.0
#         else:
#             params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.6, 1.0)
#         model = CatBoostRegressor(**params, random_state=SEED, verbose=0)
#     elif model_type == 'xgb':
#         params = {
#             'max_depth': trial.suggest_int('max_depth', 4, 10),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#             'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
#             'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
#             'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
#             'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#             'tree_method': 'gpu_hist' if has_gpu else 'auto'  # Enable GPU
#         }
#         model = xgb.XGBRegressor(**params, random_state=SEED, verbosity=0)
#     else:
#         raise ValueError("Invalid model_type")

#     kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
#     maes = []
#     for tr, val in kf.split(X):
#         X_tr, X_val = X.iloc[tr], X.iloc[val]
#         y_tr, y_val = y.iloc[tr], y.iloc[val]
        
#         # Filter non-NaN for stability
#         valid_tr = ~y_tr.isna()
#         valid_val = ~y_val.isna()
#         if valid_tr.sum() == 0 or valid_val.sum() == 0:
#             continue
        
#         model.fit(X_tr[valid_tr], y_tr[valid_tr])
#         pred = model.predict(X_val[valid_val])
#         maes.append(mean_absolute_error(y_val[valid_val], pred))
    
#     return np.mean(maes) if maes else float('inf')  # Handle empty cases

# # Tune for each target and model type
# best_params = {tgt: {} for tgt in TARGETS}
# for tgt in TARGETS:
#     y_tgt = y_train[tgt]
#     for model_type in ['lgb', 'cat', 'xgb']:
#         print(f"Tuning {model_type} for {tgt}")
#         study = optuna.create_study(direction='minimize')
#         study.optimize(lambda trial: objective(trial, X_train, y_tgt, model_type), n_trials=20)
#         best_params[tgt][model_type] = study.best_params
#         print(f"Best params for {tgt} {model_type}: {study.best_params}")

# # Now update your per_target_params dictionary with these
# # Example: per_target_params['Tg']['lgb'] = best_params['Tg']['lgb']
# # Then rerun your training cell
# print("Best params:", best_params)



has_gpu = torch.cuda.is_available()
def train_model_with_oof(X, y, params, model_type, tgt):
    valid_idx = ~y.isna()
    X = X.loc[valid_idx]; y = y.loc[valid_idx]
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    test_preds = np.zeros((len(X_test),))
    oof_preds = np.zeros(len(y))
    for fold, (tr, val) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y.iloc[tr], y.iloc[val]
        full_params = params.copy()
        if model_type == 'lgb' and has_gpu: full_params['device'] = 'gpu'
        elif model_type == 'cat' and has_gpu: full_params.update({'task_type': 'GPU', 'bootstrap_type': 'Bernoulli'})
        elif model_type == 'xgb' and has_gpu: full_params['device'] = 'cuda'
        
        if model_type == 'lgb':
            model = LGBMRegressor(**full_params, verbosity=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        elif model_type == 'cat':
            model = CatBoostRegressor(**full_params, verbose=0)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        elif model_type == 'xgb':
            model = XGBRegressor(**full_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        oof_preds[val] = model.predict(X_val)
        test_preds += model.predict(X_test) / 5
    mae = mean_absolute_error(y, oof_preds)
    print(f"{tgt} OOF MAE for {model_type}: {mae}")
    return test_preds, mae

per_target_params = {
    "Tg": {
        "lgb": {
            "num_leaves": 36, "learning_rate": 0.011, "n_estimators": 1100,
            "reg_alpha": 2.0, "reg_lambda": 0.15, "subsample": 0.60, "colsample_bytree": 0.70
        },
        "cat": {
            "depth": 8, "learning_rate": 0.016, "iterations": 1500,
            "l2_leaf_reg": 7, "subsample": 0.82
        },
        "xgb": {
            "max_depth": 8, "learning_rate": 0.015, "n_estimators": 1400,
            "reg_alpha": 1.2, "reg_lambda": 0.2, "subsample": 0.72, "colsample_bytree": 0.7
        }
    },

    "FFV": {
        "lgb": {
            "num_leaves": 27, "learning_rate": 0.022, "n_estimators": 500,
            "reg_alpha": 1.5, "reg_lambda": 0.18, "subsample": 0.67, "colsample_bytree": 0.82
        },
        "cat": {
            "depth": 6, "learning_rate": 0.03, "iterations": 1150,
            "l2_leaf_reg": 6, "subsample": 0.88
        },
        "xgb": {
            "max_depth": 6, "learning_rate": 0.016, "n_estimators": 1000,
            "reg_alpha": 1.0, "reg_lambda": 0.18, "subsample": 0.67, "colsample_bytree": 0.8
        }
    },

    "Tc": {
        "lgb": {
            "num_leaves": 32, "learning_rate": 0.030, "n_estimators": 1050,
            "reg_alpha": 0.7, "reg_lambda": 0.8, "subsample": 0.68, "colsample_bytree": 0.85
        },
        "cat": {
            "depth": 7, "learning_rate": 0.021, "iterations": 1200,
            "l2_leaf_reg": 5, "subsample": 0.82
        },
        "xgb": {
            "max_depth": 8, "learning_rate": 0.019, "n_estimators": 1200,
            "reg_alpha": 1.4, "reg_lambda": 0.77, "subsample": 0.7, "colsample_bytree": 0.85
        }
    },

    "Density": {
        "lgb": {
            "num_leaves": 44, "learning_rate": 0.017, "n_estimators": 1300,
            "reg_alpha": 0.9, "reg_lambda": 1.1, "subsample": 0.75, "colsample_bytree": 0.74
        },
        "cat": {
            "depth": 8, "learning_rate": 0.015, "iterations": 1200,
            "l2_leaf_reg": 4.5, "subsample": 0.79
        },
        "xgb": {
            "max_depth": 7, "learning_rate": 0.018, "n_estimators": 1300,
            "reg_alpha": 1.0, "reg_lambda": 1.0, "subsample": 0.74, "colsample_bytree": 0.74
        }
    },

    "Rg": {
        "lgb": {
            "num_leaves": 22, "learning_rate": 0.012, "n_estimators": 1000,
            "reg_alpha": 2.0, "reg_lambda": 0.2, "subsample": 0.62, "colsample_bytree": 0.72
        },
        "cat": {
            "depth": 6, "learning_rate": 0.013, "iterations": 1050,
            "l2_leaf_reg": 8, "subsample": 0.83
        },
        "xgb": {
            "max_depth": 6, "learning_rate": 0.013, "n_estimators": 900,
            "reg_alpha": 1.7, "reg_lambda": 0.35, "subsample": 0.64, "colsample_bytree": 0.75
        }
    }
}


final_preds = np.zeros((len(test), len(TARGETS)))
for i, tgt in enumerate(TARGETS):
    y_tgt = train[tgt]
    params_set = per_target_params[tgt]
    models = ['lgb', 'cat', 'xgb']
    test_preds_list = []; maes = []
    for model_type in models:
        test_p, mae = train_model_with_oof(X_train, y_tgt, params_set[model_type], model_type, tgt)
        test_preds_list.append(test_p); maes.append(mae)
    weights = 1 / np.array(maes); weights /= weights.sum()
    blend = np.average(test_preds_list, axis=0, weights=weights)
    lo, hi = np.percentile(y_tgt.dropna(), [0.5, 99.5])
    blend = np.clip(blend, lo, hi)
    final_preds[:, i] = blend
    print(f'{tgt} done with weighted blending')


sub = pd.DataFrame(final_preds, columns=TARGETS)
sub['id'] = test['id']
sub = sub[['id'] + TARGETS]
sub.to_csv('submission.csv', index=False)
print(sub.head())

