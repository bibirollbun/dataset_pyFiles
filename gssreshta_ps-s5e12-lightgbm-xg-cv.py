# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

X = train.drop("diagnosed_diabetes", axis=1)
y = train["diagnosed_diabetes"]


# --- Simple categorical encoding ---
cat_cols = X.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category").cat.codes
    test[col] = test[col].astype("category").cat.codes


# --- CV Setup ---
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
pred_lgb = np.zeros(len(test))
pred_xgb = np.zeros(len(test))


# --- LightGBM ---
lgb_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    n_estimators=2000,
)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr)

    oof_lgb[val_idx] = model.predict_proba(X_val)[:,1]
    pred_lgb += model.predict_proba(test)[:,1] / 10

print("LGB OOF AUC:", roc_auc_score(y, oof_lgb))


# --- XGBoost ---
xgb_model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42
)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    xgb_model.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:,1]
    pred_xgb += xgb_model.predict_proba(test)[:,1] / 10

print("XGB OOF AUC:", roc_auc_score(y, oof_xgb))


# --- Final Ensemble ---
test_pred = 0.5 * pred_lgb + 0.5 * pred_xgb

submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_pred
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


