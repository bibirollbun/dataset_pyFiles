# =============================
# SONG BPM PREDICTION PIPELINE
# =============================

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# ======================
# STEP 1: LOAD DATA
# ======================

train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ======================
# STEP 2: PREPROCESSING
# ======================

# Separate target
y = train["BeatsPerMinute"]
X = train.drop(columns=["id", "BeatsPerMinute"])
X_test = test.drop(columns=["id"])

# Encode categorical features
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# Optional: Scale numerical features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)
X_test[X_test.columns] = scaler.transform(X_test)

# ======================
# STEP 3: CROSS-VALIDATION
# ======================
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(X.shape[0])
test_preds_lgb = np.zeros(X_test.shape[0])
test_preds_xgb = np.zeros(X_test.shape[0])
test_preds_cat = np.zeros(X_test.shape[0])

# ======================
# STEP 4: TRAIN MODELS
# ======================
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n===== Fold {fold+1} =====")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # ---- LightGBM ----
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    val_pred = lgb_model.predict(X_val)
    oof_preds[val_idx] += val_pred
    test_preds_lgb += lgb_model.predict(X_test) / N_FOLDS
    
    # ---- XGBoost ----
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        objective="reg:squarederror"
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=200
    )
    val_pred = xgb_model.predict(X_val)
    oof_preds[val_idx] += val_pred
    test_preds_xgb += xgb_model.predict(X_test) / N_FOLDS
    
    # ---- CatBoost ----
    cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        eval_metric="RMSE",
        random_seed=42,
        verbose=200,
        early_stopping_rounds=100
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    val_pred = cat_model.predict(X_val)
    oof_preds[val_idx] += val_pred
    test_preds_cat += cat_model.predict(X_test) / N_FOLDS

# ======================
# STEP 5: EVALUATION
# ======================
rmse = np.sqrt(mean_squared_error(y, oof_preds / 3))  # averaged models
print("\nOOF RMSE:", rmse)

# ======================
# STEP 6: SUBMISSION
# ======================
final_preds = (test_preds_lgb + test_preds_xgb + test_preds_cat) / 3
submission = sample_sub.copy()
submission["bpm"] = final_preds
submission.to_csv("submission.csv", index=False)

print("Submission saved!")



# ======================
# STEP 6: SUBMISSION
# ======================
final_preds = (test_preds_lgb + test_preds_xgb + test_preds_cat) / 3

submission = pd.DataFrame({
    "id": test["id"],                  # keep the test IDs
    "BeatsPerMinute": final_preds      # match required column name
})

# Save to CSV
submission.to_csv("submission.csv", index=False)

print("Submission file saved:", submission.shape)
print(submission.head())




