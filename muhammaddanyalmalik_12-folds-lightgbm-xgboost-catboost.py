# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")

# --- Load Data ---
train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# --- Separate Features and Target ---
X_train = train_df.drop("diagnosed_diabetes", axis=1)
y_train = train_df["diagnosed_diabetes"]
X_test = test_df.copy()

# --- Categorical Encoding ---
cat_columns = X_train.select_dtypes(include="object").columns.tolist()
for col in cat_columns:
    X_train[col] = X_train[col].astype("category").cat.codes
    X_test[col] = X_test[col].astype("category").cat.codes

# --- 12-Fold Stratified CV Setup ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Prepare OOF and Test Prediction Arrays ---
oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))

pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test))

print("Starting 5-Fold Cross-Validation...\n")

# --- LightGBM ---
lgb_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.08,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    n_estimators=5000,
    random_state=42
)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"LightGBM - Fold {fold}/12")
    X_tr, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_tr, y_tr)

    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:,1]
    pred_lgb += lgb_model.predict_proba(X_test)[:,1] / 5

    print(f"  Fold {fold} OOF AUC: {roc_auc_score(y_val, oof_lgb[val_idx]):.5f}\n")

print("LightGBM finished. Overall OOF AUC:", roc_auc_score(y_train, oof_lgb), "\n")

# --- XGBoost ---
xgb_model = xgb.XGBClassifier(
    n_estimators=5000,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    use_label_encoder=False,
    random_state=42
)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"XGBoost - Fold {fold}/12")
    X_tr, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    xgb_model.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:,1]
    pred_xgb += xgb_model.predict_proba(X_test)[:,1] / 5

    print(f"  Fold {fold} OOF AUC: {roc_auc_score(y_val, oof_xgb[val_idx]):.5f}\n")

print("XGBoost finished. Overall OOF AUC:", roc_auc_score(y_train, oof_xgb), "\n")

# --- CatBoost ---
cat_model = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.08,
    depth=5,
    eval_metric="AUC",
    random_seed=42,
    verbose=1
)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"CatBoost - Fold {fold}/12")
    X_tr, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:,1]
    pred_cat += cat_model.predict_proba(X_test)[:,1] / 5

    print(f"  Fold {fold} OOF AUC: {roc_auc_score(y_val, oof_cat[val_idx]):.5f}\n")

print("CatBoost finished. Overall OOF AUC:", roc_auc_score(y_train, oof_cat), "\n")

# --- Final Ensemble (Equal Weights) ---
# final_pred = (pred_lgb + pred_xgb + pred_cat) / 3
# print("Final ensemble predictions ready.\n")

# --- Other Ensemble Techniques (in comments) ---
# Weighted Average (example weights: 0.4 LGB, 0.3 XGB, 0.3 Cat)
# final_pred = 0.4*pred_lgb + 0.3*pred_xgb + 0.3*pred_cat

# Rank Average Ensemble
# final_pred = (pd.Series(pred_lgb).rank() + pd.Series(pred_xgb).rank() + pd.Series(pred_cat).rank()) / 3
# final_pred = final_pred / final_pred.max()  # normalize to [0,1]

# Stacking (using Logistic Regression as meta-model)
from sklearn.linear_model import LogisticRegression
meta_X = np.column_stack([oof_lgb, oof_xgb, oof_cat])
meta_model = LogisticRegression()
meta_model.fit(meta_X, y_train)
final_pred = meta_model.predict_proba(np.column_stack([pred_lgb, pred_xgb, pred_cat]))[:,1]

# --- Save Submission ---
submission = pd.DataFrame({
    "id": X_test["id"],
    "diagnosed_diabetes": final_pred
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

