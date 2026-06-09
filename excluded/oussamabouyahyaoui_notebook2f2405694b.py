# ===============================
# IMPORTS
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

# ===============================
# LOAD DATA
# ===============================
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

TARGET = "diagnosed_diabetes"
FEATURES = [c for c in train.columns if c != TARGET]

# ===============================
# CATEGORICAL ENCODING
# ===============================
cat_cols = train.select_dtypes(include=["object"]).columns

for col in cat_cols:
    full = pd.concat([train[col], test[col]])
    mapping = {k: i for i, k in enumerate(full.unique())}
    train[col] = train[col].map(mapping)
    test[col] = test[col].map(mapping)

# ===============================
# MISSING VALUES
# ===============================
for col in FEATURES:
    if train[col].isnull().sum() > 0:
        med = train[col].median()
        train[col].fillna(med, inplace=True)
        test[col].fillna(med, inplace=True)

# ===============================
# SCALING (for Logistic Regression)
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train[FEATURES])
X_test_scaled = scaler.transform(test[FEATURES])

# ===============================
# CROSS-VALIDATION SETUP
# ===============================
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
oof_lr  = np.zeros(len(train))

test_lgb = np.zeros(len(test))
test_xgb = np.zeros(len(test))
test_lr  = np.zeros(len(test))

# ===============================
# MODEL PARAMETERS
# ===============================
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "seed": 42
}

# ===============================
# TRAINING LOOP
# ===============================
for fold, (tr_idx, va_idx) in enumerate(skf.split(train[FEATURES], train[TARGET])):
    print(f"ðŸš€ Fold {fold+1}")

    X_tr, X_va = train.loc[tr_idx, FEATURES], train.loc[va_idx, FEATURES]
    y_tr, y_va = train.loc[tr_idx, TARGET], train.loc[va_idx, TARGET]

    # ---- LightGBM ----
    lgb_tr = lgb.Dataset(X_tr, y_tr)
    lgb_va = lgb.Dataset(X_va, y_va)

    lgb_model = lgb.train(
        lgb_params,
        lgb_tr,
        num_boost_round=4000,
        valid_sets=[lgb_va],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
    )

    oof_lgb[va_idx] = lgb_model.predict(X_va)
    test_lgb += lgb_model.predict(test[FEATURES]) / N_FOLDS

    # ---- XGBoost ----
    xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=3000)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=200,
        verbose=False
    )

    oof_xgb[va_idx] = xgb_model.predict_proba(X_va)[:, 1]
    test_xgb += xgb_model.predict_proba(test[FEATURES])[:, 1] / N_FOLDS

    # ---- Logistic Regression ----
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_scaled[tr_idx], y_tr)

    oof_lr[va_idx] = lr.predict_proba(X_scaled[va_idx])[:, 1]
    test_lr += lr.predict_proba(X_test_scaled)[:, 1] / N_FOLDS

# ===============================
# ENSEMBLE
# ===============================
oof_ensemble = (
    0.45 * oof_lgb +
    0.40 * oof_xgb +
    0.15 * oof_lr
)

auc = roc_auc_score(train[TARGET], oof_ensemble)
print("ðŸ”¥ Ensemble CV AUC:", auc)

# ===============================
# SUBMISSION
# ===============================
test_preds = (
    0.45 * test_lgb +
    0.40 * test_xgb +
    0.15 * test_lr
)

submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()


