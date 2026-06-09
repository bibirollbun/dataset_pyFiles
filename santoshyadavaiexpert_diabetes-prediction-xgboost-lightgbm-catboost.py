import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


train_path = "/kaggle/input/playground-series-s5e12/train.csv"
test_path = "/kaggle/input/playground-series-s5e12/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


TARGET = "diagnosed_diabetes"

y = train[TARGET]
X = train.drop(columns=[TARGET])
X_test = test.copy()

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nTrain info:")
print(train.info())

print("\nMissing values (train):")
print(train.isna().sum())

print("\nTarget distribution (proportion):")
print(y.value_counts(normalize=True))


numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Remove id from numeric features if present
if "id" in numeric_features:
    numeric_features.remove("id")

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

preprocessor


xgb_model = XGBClassifier(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="auc",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

lgb_params = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": -1,
    "num_leaves": 63,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "binary",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

cat_model = CatBoostClassifier(
    depth=6,
    learning_rate=0.03,
    n_estimators=700,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=False,
    random_state=RANDOM_STATE,
)

xgb_model, lgb_params, cat_model


from sklearn.pipeline import Pipeline

def make_xgb_pipeline():
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", clone(xgb_model)),
    ])

def make_lgb_pipeline():
    # LightGBM works on numpy arrays; we wrap it in a simple sklearn-compatible class
    class LGBMWrapper(lgb.LGBMClassifier):
        pass
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LGBMWrapper(**lgb_params)),
    ])

def make_cat_pipeline():
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", clone(cat_model)),
    ])


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_pred_xgb = np.zeros(len(X_test))
test_pred_lgb = np.zeros(len(X_test))
test_pred_cat = np.zeros(len(X_test))

fold_scores_xgb = []
fold_scores_lgb = []
fold_scores_cat = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\nFold {fold}/{n_splits}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    # XGBoost
    xgb_clf = make_xgb_pipeline()
    xgb_clf.fit(X_train, y_train)
    oof_xgb[valid_idx] = xgb_clf.predict_proba(X_valid)[:, 1]
    fold_xgb = roc_auc_score(y_valid, oof_xgb[valid_idx])
    fold_scores_xgb.append(fold_xgb)
    test_pred_xgb += xgb_clf.predict_proba(X_test)[:, 1] / n_splits
    print(f"  XGB  Fold ROC-AUC: {fold_xgb:.5f}")
    
    # LightGBM
    lgb_clf = make_lgb_pipeline()
    lgb_clf.fit(X_train, y_train)
    oof_lgb[valid_idx] = lgb_clf.predict_proba(X_valid)[:, 1]
    fold_lgb = roc_auc_score(y_valid, oof_lgb[valid_idx])
    fold_scores_lgb.append(fold_lgb)
    test_pred_lgb += lgb_clf.predict_proba(X_test)[:, 1] / n_splits
    print(f"  LGBM Fold ROC-AUC: {fold_lgb:.5f}")
    
    # CatBoost
    cat_clf = make_cat_pipeline()
    cat_clf.fit(X_train, y_train)
    oof_cat[valid_idx] = cat_clf.predict_proba(X_valid)[:, 1]
    fold_cat = roc_auc_score(y_valid, oof_cat[valid_idx])
    fold_scores_cat.append(fold_cat)
    test_pred_cat += cat_clf.predict_proba(X_test)[:, 1] / n_splits
    print(f"  CAT  Fold ROC-AUC: {fold_cat:.5f}")

print("\nXGB  OOF ROC-AUC:", roc_auc_score(y, oof_xgb))
print("LGBM OOF ROC-AUC:", roc_auc_score(y, oof_lgb))
print("CAT  OOF ROC-AUC:", roc_auc_score(y, oof_cat))

print("\nXGB  Fold scores:", fold_scores_xgb)
print("LGBM Fold scores:", fold_scores_lgb)
print("CAT  Fold scores:", fold_scores_cat)


# Blend weights can be tuned; start with equal or slightly favor best model
w_xgb = 0.4
w_lgb = 0.3
w_cat = 0.3

oof_blend = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cat * oof_cat
blend_oof_score = roc_auc_score(y, oof_blend)
print("Blended OOF ROC-AUC:", blend_oof_score)


# Create meta-features from OOF predictions
meta_X = np.vstack([oof_xgb, oof_lgb, oof_cat]).T
meta_y = y.values

meta_model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_STATE
)
meta_model.fit(meta_X, meta_y)

meta_oof_pred = meta_model.predict_proba(meta_X)[:, 1]
meta_oof_score = roc_auc_score(meta_y, meta_oof_pred)
print("Stacked meta-model OOF ROC-AUC:", meta_oof_score)


# Meta-features for test set
meta_test_X = np.vstack([test_pred_xgb, test_pred_lgb, test_pred_cat]).T
meta_test_pred = meta_model.predict_proba(meta_test_X)[:, 1]

# You can also compare with simple blend:
blend_test_pred = w_xgb * test_pred_xgb + w_lgb * test_pred_lgb + w_cat * test_pred_cat


# Choose which prediction to submit: meta_test_pred (stacked) or blend_test_pred (simple blend)
# Here we submit the stacked predictions:
submission = pd.DataFrame({
    "id": X_test["id"],
    "diagnosed_diabetes": meta_test_pred
})

submission.head()


submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

