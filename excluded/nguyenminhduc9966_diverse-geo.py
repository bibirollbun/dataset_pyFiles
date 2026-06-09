# ============================================================
#  Kaggle Playground S5E9 - Version 15: Diverse Baseline
#  Strategy: Diverse Param Seeds + Geometric Mean + Soft Rounding
#  Target: Beat 26.40467
#  + Save OOF & Test predictions for ensemble
# ============================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
N_SPLITS = 12  # Increased slightly for stability
DATA_DIR = "/kaggle/input/playground-series-s5e9"
ORIG_DATA_DIR = "/kaggle/input/bpm-prediction-challenge" 
TARGET_COL = "BeatsPerMinute"

# DIVERSITY STRATEGY:
# Instead of identical params, we create 3 variants.
# 1. Conservative: Fewer leaves, learns broader patterns.
# 2. Balanced: Your current best settings.
# 3. Aggressive: More leaves, learns finer details.
MODEL_VARIANTS = [
    {"seed": 42,   "leaves": 31, "lr": 0.009,  "name": "Conservative"},
    {"seed": 2024, "leaves": 34, "lr": 0.009,  "name": "Balanced (Ref)"},
    {"seed": 777,  "leaves": 38, "lr": 0.0085, "name": "Aggressive"} 
]

# ---------------- LOAD DATA ----------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

try:
    orig_train = pd.read_csv(os.path.join(ORIG_DATA_DIR, "Train.csv"))
    common_cols = [c for c in train.columns if c in orig_train.columns]
    orig_train = orig_train[common_cols]
    
    train['is_generated'] = 1
    test['is_generated'] = 1
    orig_train['is_generated'] = 0
    
    train = pd.concat([train, orig_train], axis=0).reset_index(drop=True)
    print(f"Data Loaded with Original. Shape: {train.shape}")
except:
    train['is_generated'] = 1
    test['is_generated'] = 1

y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL, "id"]) 
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# ---------------- FEATURE ENGINEERING ----------------
numeric_cols = X.select_dtypes(include=[np.number]).columns
for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
            freq_enc = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq_enc)
        else:
            df[col] = df[col].fillna(df[col].median())
    
    for col in numeric_cols:
        if col != 'is_generated':
            df[f'{col}_frac'] = df[col] % 1
            df[f'{col}_is_int'] = (df[col] % 1 == 0).astype(int)

X_test = X_test.reindex(columns=X.columns, fill_value=0)
X_vals = X.values
X_test_vals = X_test.values

# ---------------- TRAINING LOOP ----------------
# We store predictions in lists to apply Geometric Mean later
oof_preds_list = []
test_preds_list = []

# Base Params (Shared)
base_params = {
    "n_estimators": 3500, # Increased slightly to ensure convergence
    "max_depth": -1,
    "subsample": 0.75,          
    "colsample_bytree": 0.55,   
    "min_child_samples": 100,   
    "reg_alpha": 0.6,           
    "reg_lambda": 0.5,
    "n_jobs": -1,
    "metric": "rmse",
    "verbosity": -1
}

print(f"--- Starting Diverse Training ---")

for variant in MODEL_VARIANTS:
    v_name = variant['name']
    print(f"\nTraining {v_name} (Leaves={variant['leaves']})...")
    
    # Specific Params
    params = base_params.copy()
    params['random_state'] = variant['seed']
    params['num_leaves'] = variant['leaves']
    params['learning_rate'] = variant['lr']
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=variant['seed'])
    
    oof_seed = np.zeros(X_vals.shape[0])
    test_seed = np.zeros(X_test_vals.shape[0])
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(300, verbose=False)]
        )
        
        oof_seed[va_idx] = model.predict(X_va)
        test_seed += model.predict(X_test_vals) / N_SPLITS
    
    rmse_seed = mean_squared_error(y, oof_seed, squared=False)
    print(f"{v_name} RMSE: {rmse_seed:.5f}")
    
    oof_preds_list.append(oof_seed)
    test_preds_list.append(test_seed)

# ---------------- GEOMETRIC MEAN AGGREGATION ----------------
print("\n--- Aggregating with Geometric Mean ---")

# Stack predictions
oof_stack = np.column_stack(oof_preds_list)
test_stack = np.column_stack(test_preds_list)

# Geometric Mean Formula: exp(mean(log(x)))
geo_oof = np.exp(np.mean(np.log(oof_stack), axis=1))
geo_test = np.exp(np.mean(np.log(test_stack), axis=1))

raw_geo_rmse = mean_squared_error(y, geo_oof, squared=False)
print(f"Geometric Mean OOF RMSE: {raw_geo_rmse:.5f}")

# ---------------- RIDGE CALIBRATION ----------------
print("\n--- Ridge Calibration ---")
mask_generated = train['is_generated'] == 1

ridge = Ridge(alpha=10.0)
ridge.fit(geo_oof[mask_generated].reshape(-1, 1), y[mask_generated])

print(f"Ridge Slope: {ridge.coef_[0]:.5f}")
calibrated_oof = ridge.predict(geo_oof.reshape(-1, 1))
calibrated_test = ridge.predict(geo_test.reshape(-1, 1))

calibrated_rmse = mean_squared_error(y, calibrated_oof, squared=False)
print(f"Calibrated RMSE: {calibrated_rmse:.5f}")

# ---------------- SOFT INTEGER BLENDING (The Finisher) ----------------
print("\n--- Soft Integer Blending Optimization ---")
best_ratio = 0.0
best_rmse = calibrated_rmse

for r in np.linspace(0, 0.2, 50):
    temp_pred = (1 - r) * calibrated_oof + r * np.round(calibrated_oof)
    temp_rmse = mean_squared_error(y, temp_pred, squared=False)
    
    if temp_rmse < best_rmse:
        best_rmse = temp_rmse
        best_ratio = r

print(f"Best Rounding Ratio: {best_ratio:.4f}")
print(f"Final Optimized OOF RMSE: {best_rmse:.5f}")

# Apply best ratio to train & test
oof_final = (1 - best_ratio) * calibrated_oof + best_ratio * np.round(calibrated_oof)
final_pred = (1 - best_ratio) * calibrated_test + best_ratio * np.round(calibrated_test)

# ---------------- SAVE OOF + TEST FOR ENSEMBLE ----------------
np.save("oof_v15.npy", oof_final)
np.save("test_v15.npy", final_pred)
print("\nSaved OOF + Test predictions for V15 (diverse_geo).")

# ---------------- SUBMISSION ----------------
submission = pd.DataFrame({"id": test_ids, "BeatsPerMinute": final_pred})
submission.to_csv("submission_v15_diverse_geo.csv", index=False)
print("\nSaved submission_v15_diverse_geo.csv")


