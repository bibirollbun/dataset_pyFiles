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
SEEDS = [42, 2024, 777]
N_SPLITS = 10

DATA_DIR = "/kaggle/input/playground-series-s5e9"
ORIG_DATA_DIR = "/kaggle/input/bpm-prediction-challenge"
TARGET_COL = "BeatsPerMinute"


train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

try:
    orig_train = pd.read_csv(os.path.join(ORIG_DATA_DIR, "Train.csv"))
    common_cols = [c for c in train.columns if c in orig_train.columns]
    orig_train = orig_train[common_cols]

    train["is_generated"] = 1
    test["is_generated"] = 1
    orig_train["is_generated"] = 0

    train = pd.concat([train, orig_train], axis=0).reset_index(drop=True)
    print(f"Data Loaded with Original. Shape: {train.shape}")
except:
    print("Original data not found. Using synthetic only.")
    train["is_generated"] = 1
    test["is_generated"] = 1



y = train[TARGET_COL]
X = train.drop(columns=[TARGET_COL, "id"])
X_test = test.drop(columns=["id"])
test_ids = test["id"]



numeric_cols = X.select_dtypes(include=[np.number]).columns

for df in [X, X_test]:
    # Fill NA + encode categorical
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing")
            freq_enc = df[col].value_counts().to_dict()
            df[col] = df[col].map(freq_enc)
        else:
            df[col] = df[col].fillna(df[col].median())

    # Add frac / is_int
    for col in numeric_cols:
        if col != "is_generated":
            df[f"{col}_frac"] = df[col] % 1
            df[f"{col}_is_int"] = (df[col] % 1 == 0).astype(int)

# Align columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)

X_vals = X.values
X_test_vals = X_test.values



base_params = {
    "n_estimators": 8000,
    "learning_rate": 0.002,
    "num_leaves": 34,
    "max_depth": -1,
    "subsample": 0.78,
    "colsample_bytree": 0.58,
    "min_child_samples": 100,
    "reg_alpha": 0.55,
    "reg_lambda": 0.45,
    "n_jobs": -1,
    "metric": "rmse",
    "verbosity": -1
}



oof_preds_total = np.zeros(X_vals.shape[0])
test_preds_total = np.zeros(X_test_vals.shape[0])

print(f"--- Starting Seed Averaging (Seeds: {SEEDS}) ---")

for i, seed in enumerate(SEEDS):
    print(f"\nTraining Seed {seed} ({i+1}/{len(SEEDS)})...")

    params = base_params.copy()
    params["random_state"] = seed
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

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

    oof_preds_total += oof_seed / len(SEEDS)
    test_preds_total += test_seed / len(SEEDS)

    rmse_seed = mean_squared_error(y, oof_seed, squared=False)
    print(f"Seed {seed} Raw RMSE: {rmse_seed:.5f}")

print("\n--- All Seeds Completed ---")
raw_oof_rmse = mean_squared_error(y, oof_preds_total, squared=False)
print(f"Combined Raw OOF RMSE: {raw_oof_rmse:.5f}")



print("\n--- Final Calibration (Ridge) ---")

mask_generated = train["is_generated"] == 1

ridge = Ridge(alpha=4.5)
ridge.fit(oof_preds_total[mask_generated].reshape(-1, 1), y[mask_generated])

print(f"Ridge Slope: {ridge.coef_[0]:.5f}")
print(f"Ridge Intercept: {ridge.intercept_:.5f}")

calibrated_oof = ridge.predict(oof_preds_total.reshape(-1, 1))
calibrated_rmse = mean_squared_error(y, calibrated_oof, squared=False)
print(f"Final Calibrated RMSE: {calibrated_rmse:.5f}")



final_pred = ridge.predict(test_preds_total.reshape(-1, 1))

submission = pd.DataFrame({"id": test_ids, "BeatsPerMinute": final_pred})
submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv")


