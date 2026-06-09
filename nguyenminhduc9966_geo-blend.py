# ============================================================
#  Kaggle Playground S5E9 - Version 16: Geometric Baseline (Revised)
#  Strategy: Parameter Diversity + Geometric Mean Ensemble
#  + Save OOF & Test for Ensemble
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
N_SPLITS = 12 
DATA_DIR = "/kaggle/input/playground-series-s5e9"
ORIG_DATA_DIR = "/kaggle/input/bpm-prediction-challenge" 
TARGET_COL = "BeatsPerMinute"

MODEL_VARIANTS = [
    {"seed": 42,   "leaves": 31, "lr": 0.006,  "name": "Conservative"},
    {"seed": 2024, "leaves": 34, "lr": 0.006,  "name": "Balanced"},
    {"seed": 777,  "leaves": 37, "lr": 0.0055, "name": "Aggressive"}
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
    print(f"Loaded original dataset: {train.shape}")
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
            freq = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq)
        else:
            df[col] = df[col].fillna(df[col].median())
    
    for col in numeric_cols:
        if col != "is_generated":
            df[f"{col}_frac"] = df[col] % 1
            df[f"{col}_is_int"] = (df[col] % 1 == 0).astype(int)

X_test = X_test.reindex(columns=X.columns, fill_value=0)
X_vals = X.values
X_test_vals = X_test.values

# ---------------- TRAINING & BLENDING ----------------
oof_dict = {}
test_dict = {}

base_params = {
    "n_estimators": 6000,
    "max_depth": -1,
    "subsample": 0.75,
    "colsample_bytree": 0.55,
    "min_child_samples": 100,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "n_jobs": -1,
    "metric": "rmse",
    "verbosity": -1
}

print("\n--- Training All Model Variants ---")

for variant in MODEL_VARIANTS:
    name = variant["name"]
    print(f"\nTraining {name}  (Leaves={variant['leaves']}, LR={variant['lr']})")

    params = base_params.copy()
    params["random_state"] = variant["seed"]
    params["num_leaves"] = variant["leaves"]
    params["learning_rate"] = variant["lr"]
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=variant["seed"])
    
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_vals)):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(300, verbose=False)]
        )

        oof[va_idx] = model.predict(X_va)
        test_pred += model.predict(X_test_vals) / N_SPLITS
    
    print(f"{name} RMSE = {mean_squared_error(y, oof, squared=False):.5f}")
    
    oof_dict[name] = oof
    test_dict[name] = test_pred

# ---------------- GEOMETRIC BLENDING ----------------
print("\n--- Geometric Mean Blending ---")

oof_stack = np.column_stack(list(oof_dict.values()))
test_stack = np.column_stack(list(test_dict.values()))

geo_oof = np.exp(np.mean(np.log(oof_stack), axis=1))
geo_test = np.exp(np.mean(np.log(test_stack), axis=1))

geo_rmse = mean_squared_error(y, geo_oof, squared=False)
print(f"Geometric Mean RMSE: {geo_rmse:.5f}")

# ---------------- RIDGE CALIBRATION ----------------
print("\n--- Ridge Calibration ---")

mask = train["is_generated"] == 1
ridge = Ridge(alpha=10.0)
ridge.fit(geo_oof[mask].reshape(-1, 1), y[mask])

cal_oof = ridge.predict(geo_oof.reshape(-1, 1))
cal_rmse = mean_squared_error(y, cal_oof, squared=False)
print(f"Calibrated RMSE: {cal_rmse:.5f}")

cal_test = ridge.predict(geo_test.reshape(-1, 1))

# ---------------- SAVE OOF + TEST ----------------
np.save("oof_v16.npy", cal_oof)
np.save("test_v16.npy", cal_test)
print("\nSaved oof_v16.npy and test_v16.npy")

# ---------------- SUBMISSION ----------------
submission = pd.DataFrame({
    "id": test_ids,
    "BeatsPerMinute": cal_test
})
submission.to_csv("submission_v16_geo_blend.csv", index=False)

print("\nSaved submission_v16_geo_blend.csv")


