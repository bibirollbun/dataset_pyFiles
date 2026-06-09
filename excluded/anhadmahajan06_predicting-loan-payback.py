import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

y = train["loan_paid_back"]
X = train.drop(["loan_paid_back", "id"], axis=1)
X_test = test.drop("id", axis=1)

# ========================================
# ðŸ”¥ FIX: LABEL ENCODE ALL STRING FEATURES
# ========================================

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))


# fill missing values
for c in X.columns:
    if X[c].dtype != "object":
        X[c] = X[c].fillna(X[c].median())
        X_test[c] = X_test[c].fillna(X[c].median())


# scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.95)
X = pca.fit_transform(X)
X_test = pca.transform(X_test)



lgb_params = {
    "learning_rate": 0.01,
    "max_depth": -1,
    "n_estimators": 4500,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "objective": "binary",
    "metric": "auc",
}

xgb_params = {
    "learning_rate": 0.01,
    "max_depth": 7,
    "n_estimators": 3000,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
}

cat_params = {
    "depth": 8,
    "learning_rate": 0.01,
    "iterations": 3000,
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": 0
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = []
preds = []

for train_idx, valid_idx in skf.split(X, y):

    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_train, y_train)
    
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train)
    
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_train, y_train)

    valid_pred = (
        0.45 * model_lgb.predict_proba(X_valid)[:, 1] +
        0.35 * model_xgb.predict_proba(X_valid)[:, 1] +
        0.20 * model_cat.predict_proba(X_valid)[:, 1]
    )

    oof.extend(valid_pred)

    fold_test_pred = (
        0.45 * model_lgb.predict_proba(X_test)[:, 1] +
        0.35 * model_xgb.predict_proba(X_test)[:, 1] +
        0.20 * model_cat.predict_proba(X_test)[:, 1]
    )

    preds.append(fold_test_pred)


print("CV AUC:", roc_auc_score(y, np.array(oof)))

final_preds = np.mean(preds, axis=0)

sub = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": final_preds
})

sub.to_csv("submission.csv", index=False)
print("DONE")


