# === Step 1: Install & Imports ===

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from autofeat import AutoFeatRegressor
from cuml.ensemble import RandomForestRegressor as cuRF
import time

# === Step 2: Globals ===
le = LabelEncoder()
scaler = None
autofeat_model = None

# === Step 3: Data Preparation with AutoFeat ===
def prepare_data(df, fit=False):
    global scaler, autofeat_model
    df = df.copy()
    df["Sex"] = le.fit_transform(df["Sex"]) if fit else le.transform(df["Sex"])

    X_raw = df.drop(columns=["Calories", "id"], errors="ignore").astype(np.float32).fillna(0)

    if fit:
        y_full = np.log1p(df["Calories"])
        autofeat_model = AutoFeatRegressor(verbose=1, feateng_steps=2)
        X_transformed = autofeat_model.fit_transform(X_raw, y_full)
    else:
        X_transformed = autofeat_model.transform(X_raw)

    scaler = scaler or StandardScaler()
    return scaler.fit_transform(X_transformed).astype(np.float32) if fit else scaler.transform(X_transformed).astype(np.float32)

# === Step 4: Load and Prepare Data ===
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

X = prepare_data(train, fit=True)
X_test = prepare_data(test, fit=False)
y = np.log1p(train["Calories"].clip(lower=0)).astype(np.float32)

# === Step 5: Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 6: Train RAPIDS Random Forest Regressor ===
start = time.time()
print("â�³ Starting RAPIDS Random Forest training...")
rf = cuRF(
    n_estimators=300,
    max_depth=20,
    max_features='sqrt',
    bootstrap=True,
    split_criterion='mse',
    random_state=42
)
rf.fit(X_train, y_train)
print(f"â�± Training completed in {time.time() - start:.2f} seconds")
y_pred = rf.predict(X_val)

# === Step 7: Evaluate ===
rmsle = mean_squared_log_error(y_val, y_pred) ** 0.5
print(f"ğŸ“Š RAPIDS RF Validation RMSLE: {rmsle:.4f}")

# === Step 8: Predict and Save Submission ===
y_test_pred = rf.predict(X_test)
test_preds = np.expm1(np.clip(y_test_pred, 0, None))
submission = pd.DataFrame({"id": test["id"], "Calories": test_preds})
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved.")


# === Step 6: Train RAPIDS Random Forest Regressor ===
start = time.time()
print("â�³ Starting RAPIDS Random Forest training...")
rf = cuRF(
    n_estimators=600,
    max_depth=15,
    max_features='sqrt',
    bootstrap=True,
    split_criterion='mse',
    random_state=42
)
rf.fit(X_train, y_train)
print(f"â�± Training completed in {time.time() - start:.2f} seconds")
y_pred = rf.predict(X_val)

# === Step 7: Evaluate ===
rmsle = mean_squared_log_error(y_val, y_pred) ** 0.5
print(f"ğŸ“Š RAPIDS RF Validation RMSLE: {rmsle:.4f}")

# === Step 8: Predict and Save Submission ===
y_test_pred = rf.predict(X_test)
test_preds = np.expm1(np.clip(y_test_pred, 0, None))
submission = pd.DataFrame({"id": test["id"], "Calories": test_preds})
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved.")

