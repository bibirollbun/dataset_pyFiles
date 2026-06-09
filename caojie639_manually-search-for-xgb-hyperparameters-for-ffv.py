!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss

import lightgbm as lgb
import xgboost as xgb

import os
from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
print(train.shape, test.shape)


train.isnull().sum()


def get_molecular_descriptors(max_autocorr=10):
    """Get molecular descriptors - either hardcoded list or auto-discovered"""

    descriptor_list_all = []
    test_mol = Chem.MolFromSmiles('CCO')

    # Collect all valid descriptors first
    for name in dir(Descriptors):
        if not name.startswith('_'):
            try:
                func = getattr(Descriptors, name)
                if callable(func):
                    result = func(test_mol)
                    if isinstance(result, (int, float)) and not np.isnan(result):
                        descriptor_list_all.append((name, func))
            except:
                pass

    print(f"ğŸ”� Total discovered descriptors before filtering: {len(descriptor_list_all)}")

    # Sort AUTOCORR2D descriptors by their numeric suffix
    autocorr_descriptors = [
        (name, func)
        for name, func in descriptor_list_all
        if name.startswith('AUTOCORR2D_')
    ]
    autocorr_descriptors.sort(key=lambda x: int(x[0].split('_')[-1]))

    # Select only the lowest-numbered ones
    limited_autocorr = autocorr_descriptors[:max_autocorr]

    # Include all other descriptors
    other_descriptors = [
        (name, func)
        for name, func in descriptor_list_all
        if not name.startswith('AUTOCORR2D_')
    ]

    # Final descriptor list
    descriptor_list = limited_autocorr + other_descriptors

    print(f"âœ… Auto-discovered {len(descriptor_list)} descriptors (limited to {max_autocorr} AUTOCORR2D):")
    names = [name for name, _ in descriptor_list]
    print("  " + ", ".join(names))

    feature_names = [name for name, _ in descriptor_list]
    return descriptor_list, feature_names

molecular_descriptors =  get_molecular_descriptors(max_autocorr=10)


def smiles_to_features(smiles_list, descriptor_functions):
   """ Convert SMILES strings to raw feature matrix
   https://www.kaggle.com/code/richolson/smiles-rdkit-lgbm-ftw """

   features = []
   total = len(smiles_list)

   print(f"Processing {total} SMILES...", end="", flush=True)

   for i, smiles in enumerate(smiles_list):
       # Progress indicator every 1000 molecules or at milestones
       if i > 0 and (i % 1000 == 0 or i == total - 1):
           print(f" {i+1}/{total}", end="", flush=True)

       mol_features = []
       try:
           mol = Chem.MolFromSmiles(smiles)
           if mol is None:
               # Invalid SMILES - fill with NaN
               mol_features = [np.nan] * len(descriptor_functions)
           else:
               # Calculate each descriptor
               for name, func in descriptor_functions:
                   try:
                       value = func(mol)
                       # Handle problematic values
                       if np.isinf(value) or abs(value) > 1e10:
                           value = np.nan
                       mol_features.append(value)
                   except:
                       # Descriptor calculation failed
                       mol_features.append(np.nan)
       except:
           # Complete failure - fill entire row with NaN
           mol_features = [np.nan] * len(descriptor_functions)

       features.append(mol_features)

   print(" âœ…", flush=True)
   return np.array(features, dtype=float)

descriptor_functions, feature_names = molecular_descriptors
X_raw = smiles_to_features(train['SMILES'].values, descriptor_functions)


def clean_features(X):
    """Handle NaN/inf values and impute missing data"""
    # Create a copy to avoid modifying the original
    X_clean = X.copy()

    X_clean[np.isinf(X_clean)] = np.nan

    # Count and report missing values
    missing = np.isnan(X_clean).sum()
    print(f"ğŸ§¹ Cleaned {missing:,} missing values ({missing/X_clean.size*100:.1f}%)")

    # Median imputation
    for i in range(X_clean.shape[1]):
        col = X_clean[:, i]
        if np.isnan(col).any():
            X_clean[np.isnan(col), i] = np.nanmedian(col) if not np.isnan(np.nanmedian(col)) else 0

    return X_clean


mask = train['FFV'].notna()
X_FFV = X_raw[mask]
train_FFV = pd.DataFrame(X_FFV, columns=feature_names)
train_FFV.head()


X_FFV.shape


remove_idx = []
for i in range(X_FFV.shape[-1]):
    col = X_FFV[:, i]
    nan_ratio = np.isnan(col).sum() / len(col)
    if nan_ratio > 0.8:
        remove_idx.append(i)
X_FFV = np.delete(X_FFV, remove_idx, axis=1)
X_FFV


X_FFV = clean_features(X_FFV)  # np.array


y_FFV = train['FFV'][mask]


y_FFV.describe()


y_FFV = y_FFV.values
if np.isnan(X_FFV).any():
    print("There are still NaN values in the data")
data_xgb = xgb.DMatrix(X_FFV, y_FFV)
params = {"max_depth":6,
          "seed":123,
         }
result = xgb.cv(params, data_xgb,
                num_boost_round=100,
                nfold=5,
                seed=123
                )
result.iloc[-1,:]


def overfitcheck(result):
    return (result.iloc[-1,2] - result.iloc[-1,0]).min()


train = []
test = []
gamma = np.arange(0,0.2,0.01)
overfit = []
for i in gamma:
    params = {"max_depth":6,
              "seed":123,
              "eta":0.1,
              "gamma":float(i),
             }
    result = xgb.cv(params,data_xgb,
                    num_boost_round=100,
                    nfold=5,
                    seed=123
               )
    overfit.append(overfitcheck(result))
    train.append(result.iloc[-1,0])
    test.append(result.iloc[-1,2])
plt.plot(gamma,overfit)


plt.plot(gamma,test,color="red")


train = []
test = []
lambda_ = np.arange(0,5,0.2)
overfit = []
for i in lambda_:
    params = {"max_depth":6,
              "seed":123,
              "eta":0.1,
              "lambda":float(i),
             }
    result = xgb.cv(params,data_xgb,
                    num_boost_round=100,
                    nfold=5,
                    seed=123
               )
    overfit.append(overfitcheck(result))
    train.append(result.iloc[-1,0])
    test.append(result.iloc[-1,2])
plt.plot(lambda_, overfit, color="blue");


plt.plot(lambda_, test, color="red");


train = []
test = []
option = np.arange(20,200,10)
overfit = []
for i in option:
    params = {"max_depth":6,"seed":123,"eta":0.1,
             }
    result = xgb.cv(params,data_xgb,
                    num_boost_round=i,
                    nfold=5, #è¡¥å……äº¤å�‰éªŒè¯�ä¸­æ‰€éœ€çš„å�‚æ•°ï¼Œnfold=5è¡¨ç¤º5æŠ˜äº¤å�‰éªŒè¯�
                    seed=123 #äº¤å�‰éªŒè¯�çš„éš�æœºæ•°ç§�å­�ï¼Œparamsä¸­çš„æ˜¯ç®¡ç�†boostingè¿‡ç¨‹çš„éš�æœºæ•°ç§�å­�
               )
    overfit.append(overfitcheck(result))
    train.append(result.iloc[-1,0])
    test.append(result.iloc[-1,2])

plt.plot(option,test)


plt.plot(option,overfit)


train = []
test = []
option = np.arange(0,30,1)
overfit = []
for i in option:
    params = {"max_depth":6,
              "seed":123,
              "eta":0.1,
              "min_child_weight":i,
             }
    result = xgb.cv(params,
                    data_xgb,
                    num_boost_round=100,
                    nfold=5,
                    seed=123
                    )
    overfit.append(overfitcheck(result))
    train.append(result.iloc[-1,0])
    test.append(result.iloc[-1,2])
plt.plot(option,test)


plt.plot(option,overfit)


def hyperopt_objective(params):
    paramsforxgb = {"eta":params["eta"],
                    "booster":params["booster"],
                    "colsample_bytree":params["colsample_bytree"],
                    "colsample_bynode":params["colsample_bynode"],
                    "gamma":params["gamma"],
                    "lambda":params["lambda"],
                    "min_child_weight":params["min_child_weight"],
                    "max_depth":int(params["max_depth"]),
                    "subsample":params["subsample"],
                    "objective":params["objective"],
                    "rate_drop":params["rate_drop"],
                    "verbosity":0,
                    "seed":123,
                    }
    result = xgb.cv(paramsforxgb, data_xgb, seed=123, metrics=("mae"), num_boost_round=int(params["num_boost_round"]))
    return result.iloc[-1,2]


param_grid_simple = {'num_boost_round': hp.quniform("num_boost_round",20,200,10),
                     "eta": hp.quniform("eta",0.05,2.05,0.05),
                     "booster":hp.choice("booster",["gbtree","dart"]),
                     "colsample_bytree":hp.quniform("colsample_bytree",0.3,1,0.1),
                     "colsample_bynode":hp.quniform("colsample_bynode",0.1,1,0.1),
                     "gamma":hp.quniform("gamma", 0, 0.2, 0.01),
                     "lambda":hp.quniform("lambda", 0, 5, 0.2),
                     "min_child_weight":hp.quniform("min_child_weight",0,30,1),
                     "max_depth":hp.quniform("max_depth", 2, 20, 2),
                     "subsample":hp.quniform("subsample",0.1,1,0.1),
                     "objective":hp.choice("objective", ["reg:squarederror", "reg:absoluteerror"]),
                     "rate_drop":hp.quniform("rate_drop",0.1,1,0.1),
                    }


def param_hyperopt(max_evals=100):
    trials = Trials()

    early_stop_fn = no_progress_loss(50)

    params_best = fmin(hyperopt_objective
                       , space = param_grid_simple
                       , algo = tpe.suggest
                       , max_evals = max_evals
                       , verbose=True
                       , trials = trials
                       , early_stop_fn = early_stop_fn
                      )

    print("\n","\n","best params: ", params_best, "\n")
    return params_best, trials


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(150)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(150)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(150)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(150)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(150)


param_grid_simple = {'num_boost_round': hp.quniform("num_boost_round", 90, 250, 5),
                     "eta": hp.quniform("eta", 0, 0.3, 0.01),
                     "colsample_bytree":hp.quniform("colsample_bytree", 0.6, 1, 0.05),
                     "colsample_bynode":hp.quniform("colsample_bynode", 0.1, 1, 0.05),
                     "gamma":hp.quniform("gamma", 0.04, 0.2, 0.005),
                     "lambda":hp.quniform("lambda", 0, 6, 0.05),
                     "min_child_weight":hp.quniform("min_child_weight", 3, 17, 1),
                     "max_depth":hp.quniform("max_depth",10, 30, 1),
                     "subsample":hp.quniform("subsample", 0.6, 1, 0.05),
                    }


def hyperopt_objective(params):
    paramsforxgb = {"booster":"gbtree",
                    "objective":"reg:absoluteerror",
                    "eta":params["eta"],
                    "colsample_bytree":params["colsample_bytree"],
                    "colsample_bynode":params["colsample_bynode"],
                    "gamma":params["gamma"],
                    "lambda":params["lambda"],
                    "min_child_weight":params["min_child_weight"],
                    "max_depth":int(params["max_depth"]),
                    "subsample":params["subsample"],
                    "verbosity":0,
                    "seed":123,
                    }
    result = xgb.cv(paramsforxgb, data_xgb, seed=123, metrics=("mae")
                    ,num_boost_round=int(params["num_boost_round"]))
    return result.iloc[-1,2]


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(150)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(150)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(150)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(150)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(150)


param_grid_simple = {'num_boost_round': hp.quniform("num_boost_round", 190, 280, 1),
                     "eta": hp.quniform("eta", 0.02, 0.1, 0.005),
                     "colsample_bytree":hp.quniform("colsample_bytree", 0.5, 1, 0.01),
                     "colsample_bynode":hp.quniform("colsample_bynode", 0.6, 1, 0.01),
                     "gamma":hp.quniform("gamma", 0.07, 0.2, 0.002),
                     "lambda":hp.quniform("lambda", 0.5, 7, 0.02),
                     "min_child_weight":hp.quniform("min_child_weight", 2, 13, 1),
                     "max_depth":hp.quniform("max_depth",12, 36, 1),
                     "subsample":hp.quniform("subsample", 0.7, 1, 0.01),
                    }


def hyperopt_objective(params):
    paramsforxgb = {"booster":"gbtree",
                    "objective":"reg:absoluteerror",
                    "eta":params["eta"],
                    "colsample_bytree":params["colsample_bytree"],
                    "colsample_bynode":params["colsample_bynode"],
                    "gamma":params["gamma"],
                    "lambda":params["lambda"],
                    "min_child_weight":params["min_child_weight"],
                    "max_depth":int(params["max_depth"]),
                    "subsample":params["subsample"],
                    "verbosity":0,
                    "seed":123,
                    }
    result = xgb.cv(paramsforxgb, data_xgb, seed=123, metrics=("mae")
                    ,num_boost_round=int(params["num_boost_round"]))
    return result.iloc[-1,2]


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(200)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(200)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(200)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(200)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    params_best, trials = param_hyperopt(200)


bestparams = {"objective":"reg:absoluteerror",
              "booster":"gbtree",
              'colsample_bynode': 0.42,
              'colsample_bytree': 0.92,
              'eta': 0.045,
              'gamma': 0.052,
              'lambda': 0.8,
              'max_depth': 20,
              'min_child_weight': 5.0,
              'num_boost_round': 268,
              'subsample': 0.94
              }


def hyperopt_validation(params):
    paramsforxgb = {'objective':params["objective"],
                    "eta":params["eta"],
                    "booster":params["booster"],
                    "colsample_bytree":params["colsample_bytree"],
                    "colsample_bynode":params["colsample_bynode"],
                    "gamma":params["gamma"],
                    "lambda":params["lambda"],
                    "min_child_weight":params["min_child_weight"],
                    "max_depth":int(params["max_depth"]),
                    "subsample":params["subsample"],
                    "verbosity":0,
                    "seed":123,
                    }
    result = xgb.cv(paramsforxgb, data_xgb, seed=123, metrics=("mae")
                    ,num_boost_round=int(params["num_boost_round"]))
    return result.iloc[-1,2]


%%time
hyperopt_validation(bestparams)

