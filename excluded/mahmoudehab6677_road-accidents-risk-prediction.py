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
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from scipy.stats import norm
import scipy.stats as stats 
import optuna # <-- Import Optuna

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 2025]
np.random.seed(SEEDS[0])


INPUT_DIR = "/kaggle/input/playground-series-s5e10"
SYN_DIR = "/kaggle/input/simulated-roads-accident-data"
TRAIN_P = os.path.join(INPUT_DIR, "train.csv")
TEST_P = os.path.join(INPUT_DIR, "test.csv")
SAMP_P = os.path.join(INPUT_DIR, "sample_submission.csv")

df_train = pd.read_csv(TRAIN_P)
df_test = pd.read_csv(TEST_P)
df_sample = pd.read_csv(SAMP_P)

syn_paths = []
for s in [2, 10, 100]:
    p = os.path.join(SYN_DIR, f"synthetic_road_accidents_{s}k.csv")
    if os.path.exists(p):
        syn_paths.append(p)

if syn_paths:
    df_syn = pd.concat([pd.read_csv(p) for p in syn_paths], axis=0, ignore_index=True)
else:
    df_syn = pd.DataFrame()

target_col = "accident_risk"
if target_col not in df_test.columns:
    df_test[target_col] = 0.5

n_train = len(df_train)
n_test = len(df_test)

if not df_syn.empty:
    if "id" not in df_syn.columns:
        if "id" in df_test.columns:
            start_id = int(df_test["id"].max()) + 1
        else:
            start_id = int(df_train["id"].max()) + 1
        df_syn.insert(0, "id", np.arange(start_id, start_id + len(df_syn)))
    for c in df_train.columns:
        if c not in df_syn.columns:
            df_syn[c] = np.nan
    df_syn = df_syn[df_train.columns]

df_all = pd.concat([df_train, df_test, df_syn], axis=0, ignore_index=True)
print("Combined shape:", df_all.shape)


for c in df_all.select_dtypes(include="bool").columns:
    df_all[c] = df_all[c].astype(int)
for c in df_all.select_dtypes(include="object").columns:
    df_all[c] = df_all[c].astype(str).str.strip()


def road_risk(X):
    return (
        0.3 * X["curvature"] +
        0.2 * (X["lighting"] == "night").astype(int) +
        0.1 * (X["weather"] != "clear").astype(int) +
        0.2 * (X["speed_limit"] >= 60).astype(int) +
        0.1 * (X["num_reported_accidents"] > 2).astype(int)
    )

def clipped(func):
    def clip_f(X):
        mu = func(X)
        sigma = 0.05 
        a, b = -mu / sigma, (1 - mu / sigma)
        Phi_a, Phi_b = stats.norm.cdf(a), stats.norm.cdf(b)
        phi_a, phi_b = stats.norm.pdf(a), stats.norm.pdf(b)
        return mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + 1 - Phi_b
    return clip_f

df_all["y"] = clipped(road_risk)(df_all)


original_cols = [col for col in df_train.columns if col not in ['id', 'accident_risk']]
CATS, NUMS = [], []
for col in original_cols:
    if df_all[col].dtype == "object":
        CATS.append(col)
    else:
        NUMS.append(col)

print(f"Factorizing {len(CATS)} categorical columns: {CATS}")
for col in CATS:
    df_all[col], _ = df_all[col].factorize()

FEATURES = CATS + NUMS + ["y"]


df_train_p = df_all.iloc[:n_train].reset_index(drop=True)
df_test_p = df_all.iloc[n_train:n_train + n_test].reset_index(drop=True)
df_syn_p = df_all.iloc[n_train + n_test:].reset_index(drop=True) if not df_syn.empty else pd.DataFrame()
print("Sizes -> train:", len(df_train_p), "test:", len(df_test_p), "synthetic:", len(df_syn_p))


TE_features = []
target_col = "accident_risk" 
te_source = df_syn_p 

features_for_te = FEATURES

print(f"Using {len(features_for_te)} features for Target Encoding...")
for col in features_for_te:
    te_map = te_source.groupby(col)[target_col].mean()
    te_name = f"TE_{col}"
    
    df_train_p[te_name] = df_train_p[col].map(te_map)
    df_test_p[te_name] = df_test_p[col].map(te_map)
    
    TE_features.append(te_name)
print("Target Encoding complete.")


FINAL_FEATURES = FEATURES + TE_features
print("Feature count:", len(FINAL_FEATURES))


def objective(trial):
    
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 10000,
        "tree_method": "hist",
        "device": "cuda",
        "seed": 42,
        "nthread": -1,
        
        'eta': trial.suggest_float('eta', 0.005, 0.02),
        'max_depth': trial.suggest_int('max_depth', 5, 8),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
        'lambda': trial.suggest_float('lambda', 1.0, 4.0),
        'alpha': trial.suggest_float('alpha', 0.1, 3.0),
    }
    
    FOLDS = 7
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    fold_rmses = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(df_train_p), 1):
        
        X_tr = df_train_p.iloc[tr_idx][FINAL_FEATURES]
        X_val = df_train_p.iloc[val_idx][FINAL_FEATURES]
        y_tr = df_train_p.iloc[tr_idx][target_col].values - df_train_p.iloc[tr_idx]["y"].values
        y_val = df_train_p.iloc[val_idx][target_col].values - df_train_p.iloc[val_idx]["y"].values
        
        # --- NEW: Get the *true* target for this fold for validation ---
        y_val_true = df_train_p.iloc[val_idx][target_col].values

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        evallist = [(dval, "valid")]
        
        model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            evals=evallist,
            early_stopping_rounds=200,
            verbose_eval=False 
        )
        
        val_pred = model.predict(dval) + df_train_p.iloc[val_idx]["y"].values
        
        fold_rmse = np.sqrt(mean_squared_error(y_val_true, val_pred))
        fold_rmses.append(fold_rmse)

    return np.mean(fold_rmses)

print("Starting Optuna study (v2, corrected)... This will take a while.")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True) # 50 trials

print("\n--- Best Trial Results ---")
print(f"Best OOF RMSE: {study.best_value:.5f}") 
print("Best params found:")
print(study.best_params)


all_oof_preds = np.zeros((len(SEEDS), len(df_train_p)))
all_test_preds = np.zeros((len(SEEDS), len(df_test_p)))
FOLDS = 7

best_params_from_optuna = study.best_params

base_xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 10000,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": -1,
}

final_tuned_params = {**base_xgb_params, **best_params_from_optuna}
print("\nUsing these tuned parameters for final training:")
print(final_tuned_params)


# Loop over each seed
for seed_idx, seed in enumerate(SEEDS):
    print(f"\n--- Training with SEED: {seed} ({seed_idx + 1}/{len(SEEDS)}) ---")
    np.random.seed(seed) # Set seed for numpy
    
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=seed) # Use seed for KFold
    
    oof_preds_seed = np.zeros(len(df_train_p))
    test_preds_seed = np.zeros(len(df_test_p))

    final_tuned_params['seed'] = seed 

    print(f"Training residual XGBoost with {FOLDS}-fold CV...")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(df_train_p), 1):
        print(f"  Fold {fold}/{FOLDS}", end=" ... ")
        
        X_tr = df_train_p.iloc[tr_idx][FINAL_FEATURES]
        X_val = df_train_p.iloc[val_idx][FINAL_FEATURES]
        y_tr = df_train_p.iloc[tr_idx][target_col].values - df_train_p.iloc[tr_idx]["y"].values
        y_val = df_train_p.iloc[val_idx][target_col].values - df_train_p.iloc[val_idx]["y"].values

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(df_test_p[FINAL_FEATURES]) 
        evallist = [(dtrain, "train"), (dval, "valid")]
        
        model = xgb.train(
            params=final_tuned_params, # <-- Use the new tuned params
            dtrain=dtrain,
            num_boost_round=100_000, # Kept from original
            evals=evallist,
            early_stopping_rounds=200,
            verbose_eval=False 
        )
        print(f"best_iter: {model.best_iteration}", end=" ... ")

        val_pred = model.predict(dval) + df_train_p.iloc[val_idx]["y"].values
        oof_preds_seed[val_idx] = val_pred
        test_preds_seed += (model.predict(dtest) + df_test_p["y"].values) / FOLDS
        print("done")
    
    all_oof_preds[seed_idx] = oof_preds_seed
    all_test_preds[seed_idx] = test_preds_seed
    
    rmse_oof_seed = np.sqrt(mean_squared_error(df_train_p[target_col], oof_preds_seed))
    print(f"SEED {seed} OOF RMSE: {rmse_oof_seed:.5f}")

# Average the OOF and test predictions across all seeds
final_oof_preds = np.mean(all_oof_preds, axis=0)
final_test_preds = np.mean(all_test_preds, axis=0)


rmse_oof_final = np.sqrt(mean_squared_error(df_train_p[target_col], final_oof_preds))
rmse_prior = np.sqrt(mean_squared_error(df_train_p[target_col], df_train_p["y"]))

print(f"\n--- Final Results ---")
print(f"Baseline prior RMSE: {rmse_prior:.5f}")
print(f"Final Tuned OOF RMSE: {rmse_oof_final:.5f}") # <-- This is the number to watch!

# Clip and export submission
df_sample[target_col] = np.clip(final_test_preds, 0.0, 1.0)
df_sample.to_csv("submission.csv", index=False)
print("\nWrote submission.csv (preview):")
print(df_sample.head())

# Save OOF predictions for analysis
df_train_p["oof_pred"] = final_oof_preds
df_train_p[["id", "oof_pred"]].to_csv("oof_predictions.csv", index=False)
print("\nDone.")

