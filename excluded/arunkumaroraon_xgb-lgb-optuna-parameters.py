# ============================================================
# 1. Imports & Settings
# ============================================================
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

train_id = train["id"]
test_id  = test["id"]

y = train["diagnosed_diabetes"]
X = train.drop(["id", "diagnosed_diabetes"], axis=1)
X_test = test.drop("id", axis=1)



num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object","category"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ],
    remainder="drop"
)

Xp = preprocessor.fit_transform(X)
Xt = preprocessor.transform(X_test)


xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.008559367757686604,
    'max_depth': 5,
    'subsample': 0.93,
    'colsample_bytree': 0.19,
    'seed': 2025,
    'device': 'cuda',
    'grow_policy': 'lossguide',
    'reg_alpha': 2.0,
    'reg_lambda': 0.73,
    'min_child_weight': 5,
    'max_bin': 512,
    'n_estimators': 20000,
}

lgb_params = {
    'random_state': SEED,
    'verbose': -1,
    'n_estimators': 10000,
    'metric': 'AUC',
    'objective': 'binary',
    'learning_rate': 0.0002975707557336301,
    'max_depth': 6,
    'min_child_samples': 14,
    'subsample': 0.88,
    'colsample_bytree': 0.72,
    'num_leaves': 575,
    'reg_alpha': 0.79,
    'reg_lambda': 9.96,
    'max_bin': 157,
}


oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))

pred_xgb = np.zeros(len(X_test))
pred_lgb = np.zeros(len(X_test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


for fold, (tr_idx, val_idx) in enumerate(skf.split(Xp, y)):
    print(f"XGB Fold {fold+1}")

    X_tr, X_val = Xp[tr_idx], Xp[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model_xgb = XGBClassifier(**xgb_params)

    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=300
    )

    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    pred_xgb += model_xgb.predict_proba(Xt)[:, 1] / skf.n_splits


for fold, (tr_idx, val_idx) in enumerate(skf.split(Xp, y)):
    print(f"LGB Fold {fold+1}")

    X_tr, X_val = Xp[tr_idx], Xp[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model_lgb = lgb.LGBMClassifier(**lgb_params)

    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=300),
            log_evaluation(period=100),   # optional: controls logging frequency
        ]
    )

    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    pred_lgb += model_lgb.predict_proba(Xt)[:, 1] / skf.n_splits


oof_ensemble = 0.5 * oof_xgb + 0.5 * oof_lgb
test_preds    = 0.5 * pred_xgb + 0.5 * pred_lgb

auc = roc_auc_score(y, oof_ensemble)
print("OOF AUC:", auc)


submission = pd.DataFrame({
    "id": test_id,
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()

