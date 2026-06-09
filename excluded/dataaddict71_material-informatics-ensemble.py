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


import os
import gc
import numpy as np
import pandas as pd
import random
import warnings
import sys
import subprocess
import time
from tqdm.notebook import tqdm

# --- Install Required Libraries ---
def install_package(package_name):
    """Checks and installs package via pip if missing."""
    try: __import__(package_name.split('-')[0].split('.')[0])
    except ImportError:
        print(f"Package '{package_name}' not found. Attempting installation...")
        try:
            process = subprocess.run( [sys.executable, "-m", "pip", "install", package_name, "--no-cache"], capture_output=True, text=True, check=True, timeout=300 )
            print(f"pip install {package_name} successful.")
            try: __import__(package_name.split('-')[0].split('.')[0]); print(f"{package_name} imported.")
            except ImportError: print(f"ERROR: Failed import {package_name} after install.")
        except Exception as e: print(f"ERROR during {package_name} install: {e}")

# --- Setup Basic Output ---

required_packages = ["lightgbm", "tqdm", "scikit-learn", "pandas", "numpy"]
print(f"Checking/installing required packages: {required_packages}")
for pkg in required_packages: install_package(pkg)
print("Package check/installation complete.")

# --- Standard Imports ---
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C, ExpSineSquared, RationalQuadratic
import lightgbm as lgb
from scipy.stats import uniform, loguniform

warnings.filterwarnings("ignore")
tqdm.pandas()

# --- Configuration ---
CONFIG = {
    "seed": 42,
    "n_loocv_splits": None, 
    "n_tuning_cv_folds": 5,
    "n_tuning_trials": 30,
    "models_to_tune": ["svr_rbf", "gp", "lgbm"],
    "models": {
        "ridge": {"alpha": 1.0, "solver": "auto"},
        "svr_rbf": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "epsilon": 0.1},
        "gp": { "kernel": C(1.0) * RBF(1.0) + WhiteKernel(0.1), "n_restarts_optimizer": 15, "normalize_y": True, "random_state": 42, },
        "lgbm": { "objective": "regression_l2", "metric": "rmse", "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 5, "max_depth": 3, "feature_fraction": 0.8, "bagging_fraction": 0.8,"bagging_freq": 1, "lambda_l1": 0.2, "lambda_l2": 0.2, "min_child_samples": 5, "verbose": -1, "n_jobs": -1, "seed": 42, "boosting_type": "gbdt", },
    },
    "output_dir": "./",
    "ensemble_weights": "rmse",
}
PARAM_GRIDS = {
    "svr_rbf": { 'C': loguniform(1e-1, 1e3), 'gamma': loguniform(1e-4, 1e1), 'epsilon': uniform(0.01, 0.5) },
    "gp": { 'alpha': loguniform(1e-10, 1e-1) }, 
    "lgbm": { 'n_estimators': [50, 100, 150, 200], 'learning_rate': loguniform(0.01, 0.2), 'num_leaves': [3, 5, 7, 10], 'max_depth': [2, 3, 4], 'min_child_samples': [3, 5, 7], 'reg_alpha': uniform(0, 1), 'reg_lambda': uniform(0, 1), 'feature_fraction': uniform(0.6, 0.4), 'bagging_fraction': uniform(0.6, 0.4), }
}
# --- Seed Everything ---
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    try: import torch; torch.manual_seed(seed); 
    except ImportError: pass
    print(f"Seed set globally to {seed}")
seed_everything(CONFIG["seed"])


# --- Load Data & Prepare ---
print("--- Starting Data Loading and Processing ---")
train_df = None
test_df = None
sample_sub = None
composition_col = 'Composition (X)'
property_col = 'Property (Y)'
try:
    train_path = '/kaggle/input/spring-2025-regression-challenge/Training_Dataset.csv'
    test_path = '/kaggle/input/spring-2025-regression-challenge/Test_Dataset.csv'
    sample_sub_path = '/kaggle/input/spring-2025-regression-challenge/Sample_submission.csv'
    if not os.path.exists(train_path): raise FileNotFoundError(train_path)
    if not os.path.exists(test_path): raise FileNotFoundError(test_path)
    if not os.path.exists(sample_sub_path): raise FileNotFoundError(sample_sub_path)

    train_df = pd.read_csv(train_path); test_df = pd.read_csv(test_path); sample_sub = pd.read_csv(sample_sub_path)
    if composition_col not in train_df.columns or property_col not in train_df.columns: raise ValueError("Missing req train cols")
    if composition_col not in test_df.columns: raise ValueError("Missing req test col")
    CONFIG["n_loocv_splits"] = len(train_df)
    print(f"Train={train_df.shape}, Test={test_df.shape}, LOOCV Splits={CONFIG['n_loocv_splits']}")

    print(f"Processing '{composition_col}' as numeric...")
    # Keep original test compositions safe
    test_compositions_original = test_df[composition_col].copy()

    train_df['X_numeric'] = pd.to_numeric(train_df[composition_col], errors='coerce')
    test_df['X_numeric'] = pd.to_numeric(test_df[composition_col], errors='coerce')
    train_nans = train_df['X_numeric'].isnull().sum(); test_nans = test_df['X_numeric'].isnull().sum()
    train_mean = train_df['X_numeric'].mean(); train_mean = 0 if pd.isna(train_mean) else train_mean
    if train_nans > 0: print(f"Imputing {train_nans} train NaNs."); train_df['X_numeric'] = train_df['X_numeric'].fillna(train_mean)
    if test_nans > 0: print(f"Imputing {test_nans} test NaNs."); test_df['X_numeric'] = test_df['X_numeric'].fillna(train_mean) 
    if not pd.api.types.is_numeric_dtype(train_df['X_numeric']): raise TypeError(f"'{composition_col}' not numeric.")

    X_train_np = train_df[['X_numeric']].values
    X_test_np = test_df[['X_numeric']].values
    y_train = train_df[property_col].values
    print(f"Data shapes: X_train={X_train_np.shape}, y_train={y_train.shape}, X_test={X_test_np.shape}")
    if X_train_np.shape[1] == 0: raise ValueError("X_train has zero features.")

except Exception as e: print(f"Error loading/processing data: {e}"); raise
# --- Hyperparameter Tuning (Initial - Outside LOOCV
print("\n--- Starting Initial Hyperparameter Tuning ---")
scaler_tune = StandardScaler(); X_train_scaled_tune = scaler_tune.fit_transform(X_train_np)
tuned_params = {}
for model_name in CONFIG["models_to_tune"]:
    if model_name not in CONFIG["models"] or model_name not in PARAM_GRIDS: continue
    print(f"--- Tuning Model: {model_name} ---")
    start_tune_time = time.time(); model_instance = None; param_dist = PARAM_GRIDS[model_name]
    try:
        if model_name == "svr_rbf": model_instance = SVR(kernel='rbf')
        elif model_name == "gp": model_instance = GaussianProcessRegressor(**CONFIG["models"]["gp"])
        elif model_name == "lgbm": model_instance = lgb.LGBMRegressor(**CONFIG["models"]["lgbm"])
        else: print(f"Tuning not impl for {model_name}"); continue
        tuning_cv = KFold(n_splits=CONFIG["n_tuning_cv_folds"], shuffle=True, random_state=CONFIG["seed"])
        random_search = RandomizedSearchCV( estimator=model_instance, param_distributions=param_dist, n_iter=CONFIG["n_tuning_trials"], scoring='neg_root_mean_squared_error', n_jobs=-1, cv=tuning_cv, random_state=CONFIG["seed"], verbose=0 )
        random_search.fit(X_train_scaled_tune, y_train)
        best_params = random_search.best_params_; best_score = -random_search.best_score_
        tuned_params[model_name] = best_params; CONFIG["models"][model_name].update(best_params)
        print(f"Tuning Complete [{model_name}] ({time.time() - start_tune_time:.2f}s). Best Params: {best_params}. Best RMSE: {best_score:.6f}")
    except Exception as e: print(f"Error tuning {model_name}: {e}")
print("--- Initial Hyperparameter Tuning Finished ---"); print(f"Final model configurations: {CONFIG['models']}")


# --- Modeling with LOOCV ---
print("\n--- Starting Modeling with Leave-One-Out Cross-Validation ---")
os.makedirs(CONFIG["output_dir"], exist_ok=True)
model_oof_predictions = {}; model_test_predictions = {}; model_oof_rmses = {}
scaler = StandardScaler(); loo = LeaveOneOut()

for model_name, model_params in CONFIG["models"].items():
    print(f"\n===== Training Model: {model_name} ====="); print(f"Using Params: {model_params}")
    oof_preds = np.zeros_like(y_train, dtype=float)
    test_preds_loo_folds = np.zeros((CONFIG["n_loocv_splits"], len(test_df)), dtype=float)
    fold_errors = 0; model = None

    for i, (train_index, val_index) in enumerate(tqdm(loo.split(X_train_np), total=CONFIG["n_loocv_splits"], desc=f"{model_name} LOOCV")):
        try:
            X_train_loo, X_val_loo = X_train_np[train_index], X_train_np[val_index]
            y_train_loo, y_val_loo = y_train[train_index], y_train[val_index]
            if X_train_loo.shape[1] == 0: print(f"Fold {i}: 0 features!"); fold_errors += 1; continue

            scaler.fit(X_train_loo); X_train_loo_scaled = scaler.transform(X_train_loo); X_val_loo_scaled = scaler.transform(X_val_loo)
            X_test_scaled = scaler.transform(X_test_np)

            current_seed = CONFIG["seed"] + i; model_params_fold = model_params.copy()
            if 'random_state' in model_params_fold: model_params_fold['random_state'] = current_seed
            if 'seed' in model_params_fold: model_params_fold['seed'] = current_seed

            if model_name == "ridge": model = Ridge(**model_params_fold)
            elif model_name == "svr_rbf": model = SVR(**model_params_fold)
            elif model_name == "gp": model = GaussianProcessRegressor(**model_params_fold)
            elif model_name == "lgbm": model = lgb.LGBMRegressor(**model_params_fold)
            else: print(f"Model '{model_name}' unknown. Skip."); fold_errors+=CONFIG["n_loocv_splits"]; break

            model.fit(X_train_loo_scaled, y_train_loo)
            oof_pred = model.predict(X_val_loo_scaled)[0]
            if not np.isfinite(oof_pred): print(f"WARN: {model_name} split {i} OOF NaN/Inf"); oof_pred = np.mean(y_train_loo)
            oof_preds[val_index[0]] = oof_pred
            test_pred = model.predict(X_test_scaled)
            if not np.all(np.isfinite(test_pred)): print(f"WARN: {model_name} split {i} test NaN/Inf"); test_pred = np.nan_to_num(test_pred, nan=np.mean(y_train_loo))
            test_preds_loo_folds[i, :] = test_pred

        except Exception as e:
            print(f"ERROR model {model_name} split {i}: {e}")
            fold_errors += 1; split_train_mean = np.mean(y_train_loo) if len(y_train_loo) > 0 else np.mean(y_train)
            oof_preds[val_index[0]] = split_train_mean; test_preds_loo_folds[i, :] = split_train_mean
        finally:
            if model is not None: del model

    if fold_errors == CONFIG["n_loocv_splits"]: print(f"Model {model_name} failed ALL splits."); continue

    model_oof_predictions[model_name] = oof_preds
    model_test_predictions[model_name] = np.mean(test_preds_loo_folds, axis=0)
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    model_oof_rmses[model_name] = oof_rmse
    print(f"Model: {model_name} | OOF RMSE: {oof_rmse:.6f} | Errors: {fold_errors}/{CONFIG['n_loocv_splits']}")
    del oof_preds, test_preds_loo_folds; gc.collect()


# --- Ensemble Predictions ---
print("\n--- Ensembling Model Predictions ---")
print(f"Models available: {list(model_test_predictions.keys())}"); print(f"OOF RMSEs: {model_oof_rmses}")
all_test_preds_dict = model_test_predictions; all_oof_preds_dict = model_oof_predictions
final_test_predictions = None; final_oof_predictions_ensemble = None; ensemble_oof_rmse = np.inf

if len(all_test_preds_dict) == 0:
    print("ERROR: No models available! Submission with train mean.")
    final_test_predictions = np.full(len(test_df), y_train.mean())
    final_oof_predictions_ensemble = np.full_like(y_train, y_train.mean(), dtype=float)
elif len(all_test_preds_dict) == 1:
    model_name = list(all_test_preds_dict.keys())[0]; print(f"Only one model '{model_name}'. Using its predictions.")
    final_test_predictions = all_test_preds_dict[model_name]; final_oof_predictions_ensemble = all_oof_preds_dict[model_name]
    ensemble_oof_rmse = model_oof_rmses[model_name]
else:
    if CONFIG["ensemble_weights"] == "simple":
        print(f"Using Simple Averaging ensemble for {len(all_test_preds_dict)} models.")
        final_test_predictions = np.mean(np.array(list(all_test_preds_dict.values())), axis=0)
        final_oof_predictions_ensemble = np.mean(np.array(list(all_oof_preds_dict.values())), axis=0)
    elif CONFIG["ensemble_weights"] == "rmse":
        print(f"Using Inverse RMSE Weighted Averaging for {len(all_test_preds_dict)} models.")
        weights = []; valid_model_names = list(all_oof_preds_dict.keys())
        for name in valid_model_names: rmse = model_oof_rmses.get(name, np.inf); weights.append(1.0 / (rmse + 1e-9))
        weights = np.array(weights)
        if np.sum(weights) == 0 or not np.all(np.isfinite(weights)): print("WARN: Invalid weights, using simple avg."); weights = np.ones(len(valid_model_names))
        weights /= np.sum(weights); print(f"Ensemble Weights ({valid_model_names}): {weights}")
        oof_preds_array = np.array([all_oof_preds_dict[name] for name in valid_model_names])
        test_preds_array = np.array([all_test_preds_dict[name] for name in valid_model_names])
        final_oof_predictions_ensemble = np.sum(oof_preds_array * weights[:, np.newaxis], axis=0)
        final_test_predictions = np.sum(test_preds_array * weights[:, np.newaxis], axis=0)
    else:
        print(f"WARN: Unknown ensemble strategy. Using simple avg.")
        final_test_predictions = np.mean(np.array(list(all_test_preds_dict.values())), axis=0)
        final_oof_predictions_ensemble = np.mean(np.array(list(all_oof_preds_dict.values())), axis=0)

    
    if final_oof_predictions_ensemble is not None:
        ensemble_oof_rmse = np.sqrt(mean_squared_error(y_train, final_oof_predictions_ensemble))
        print(f"Ensemble OOF RMSE ({CONFIG['ensemble_weights']} method): {ensemble_oof_rmse:.6f}")

print(f"Final Test Predictions shape: {final_test_predictions.shape}, sample: {final_test_predictions[:5]}")


# --- Create Submission File --

print("--- Creating Submission File ---")
try:
    
    print("Reloading original test data and forcing 'Composition (X)' as string for submission keys...")
    test_path = '/kaggle/input/spring-2025-regression-challenge/Test_Dataset.csv'
    if not os.path.exists(test_path): raise FileNotFoundError(test_path)

    
    test_df_orig_for_sub = pd.read_csv(
        test_path,
        dtype={composition_col: str} 
    )
    print(f"Reloaded test_df_orig_for_sub. '{composition_col}' dtype: {test_df_orig_for_sub[composition_col].dtype}")

    if composition_col not in test_df_orig_for_sub.columns:
        raise ValueError(f"Original test data missing '{composition_col}' on reload!")
    if len(test_df_orig_for_sub) != len(final_test_predictions):
         raise ValueError(f"Length mismatch! Reloaded test ({len(test_df_orig_for_sub)}) vs predictions ({len(final_test_predictions)}).")

    print("Creating submission DataFrame using original string compositions...")
    submission_df = pd.DataFrame({
        composition_col: test_df_orig_for_sub[composition_col], 
        property_col: final_test_predictions
    })
    # --- End Reload & String Forcing ---

    if len(submission_df) != len(sample_sub): print(f"WARN: Submission length ({len(submission_df)}) != sample ({len(sample_sub)}).")
    submission_path = os.path.join(CONFIG["output_dir"], "submission.csv")
    abs_submission_path = os.path.abspath(submission_path); print(f"Attempting to save submission to: {abs_submission_path}")
    submission_df.to_csv(submission_path, index=False)
    if os.path.exists(submission_path): print(f"Submission file created: {submission_path}. Size: {os.path.getsize(submission_path)} bytes"); print("\nPreview:"); print(submission_df.head())
    else: print(f"ERROR: Submission file NOT FOUND after saving!")
except Exception as e: print(f"FATAL: Error creating/saving submission file: {e}"); print(f"\nFinal Preds:\n{final_test_predictions}")
# --- Save OOF Predictions --

print("--- Saving OOF Predictions ---")
oof_save_df = train_df[[composition_col, property_col]].copy()
for model_name, preds in model_oof_predictions.items(): oof_save_df[f'oof_{model_name}'] = preds
if final_oof_predictions_ensemble is not None and len(model_oof_predictions) > 1: oof_save_df['oof_ensemble'] = final_oof_predictions_ensemble
oof_path = os.path.join(CONFIG["output_dir"], "oof_predictions.csv")
try: oof_save_df.to_csv(oof_path, index=False); print(f"OOF predictions saved: {oof_path}")
except Exception as e: print(f"Failed to save OOF predictions file: {e}")


# --- Save Test Predictions ---

print("--- Saving Test Predictions ---")
test_pred_df = test_df[[composition_col]].copy() 
for model_name, preds in model_test_predictions.items(): test_pred_df[f'pred_{model_name}'] = preds
if final_test_predictions is not None and len(model_test_predictions) > 0 : test_pred_df['pred_final_ensemble'] = final_test_predictions
test_pred_path = os.path.join(CONFIG["output_dir"], "test_predictions.csv")
try: test_pred_df.to_csv(test_pred_path, index=False); print(f"Test predictions saved: {test_pred_path}")
except Exception as e: print(f"Failed to save Test predictions file: {e}")



