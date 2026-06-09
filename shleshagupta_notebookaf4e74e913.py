# =========================================================
# MERCEDES-BENZ GREENER MANUFACTURING
# CLEAN PIPELINE: OHE → FEATURE SELECTION → XGB → LGB → ENSEMBLE
# =========================================================

import pandas as pd
import numpy as np

from xgboost import XGBRegressor
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# =========================================================
# 1. LOAD DATA
# =========================================================
train_df = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip")
test_df  = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip")

TARGET_COL = "y"
ID_COL = "ID"

y = train_df[TARGET_COL]

train_features = train_df.drop(columns=[TARGET_COL])
test_features  = test_df.copy()

# =========================================================
# 2. ONE-HOT ENCODING (SAFE)
# =========================================================
combined = pd.concat([train_features, test_features], axis=0)
combined_encoded = pd.get_dummies(combined, drop_first=True)

X_full = combined_encoded.iloc[:len(train_df)]
X_test_full = combined_encoded.iloc[len(train_df):]

print("Full encoded shape:", X_full.shape)

# =========================================================
# 3. FEATURE SELECTION (STABLE)
# =========================================================
fs_model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
    tree_method="hist"
)

fs_model.fit(X_full, y)

imp_df = pd.DataFrame({
    "feature": X_full.columns,
    "importance": fs_model.feature_importances_
}).sort_values("importance", ascending=False)

TOP_K = 300
top_features = imp_df.head(TOP_K)["feature"].values

X = X_full[top_features]
X_test = X_test_full[top_features]

print("Reduced feature count:", X.shape[1])

# =========================================================
# 4. XGBOOST WITH CV
# =========================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_xgb = []
test_preds_xgb = np.zeros(len(X_test))

print("\nRunning XGBoost CV...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"Fold {fold}")

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.025,
        max_depth=4,
        min_child_weight=7,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.6,
        reg_lambda=1.8,
        gamma=0.1,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist"
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )

    val_preds = model.predict(X_val)
    score = r2_score(y_val, val_preds)

    cv_scores_xgb.append(score)
    test_preds_xgb += model.predict(X_test) / 5

    print("Fold R2:", round(score, 5))

print("\nXGB Mean CV R2:", round(np.mean(cv_scores_xgb), 5))
print("XGB Std  CV R2:", round(np.std(cv_scores_xgb), 5))

# =========================================================
# 5. LIGHTGBM WITH CV
# =========================================================
cv_scores_lgb = []
test_preds_lgb = np.zeros(len(X_test))

print("\nRunning LightGBM CV...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"Fold {fold}")

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=1.0,
        random_state=42,
        objective="regression"
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="r2",
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    val_preds = model.predict(X_val)
    score = r2_score(y_val, val_preds)

    cv_scores_lgb.append(score)
    test_preds_lgb += model.predict(X_test) / 5

    print("Fold R2:", round(score, 5))

print("\nLGB Mean CV R2:", round(np.mean(cv_scores_lgb), 5))
print("LGB Std  CV R2:", round(np.std(cv_scores_lgb), 5))

# =========================================================
# 6. FINAL ENSEMBLE (SAFE)
# =========================================================
final_preds = 0.7 * test_preds_xgb + 0.3 * test_preds_lgb

# =========================================================
# 7. SUBMISSION FILE
# =========================================================
submission = pd.DataFrame({
    "ID": test_df[ID_COL],
    "y": final_preds
})

submission.to_csv("submission.csv", index=False)
print("\nsubmission.csv created successfully ✅")


