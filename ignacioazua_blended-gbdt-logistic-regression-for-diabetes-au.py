# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
DATA_DIR = Path("/kaggle/input/playground-series-s5e12")

# Load
train = pd.read_csv(DATA_DIR / "train.csv", index_col="id")
test  = pd.read_csv(DATA_DIR / "test.csv",  index_col="id")

TARGET_COL = "diagnosed_diabetes"

y = train[TARGET_COL].astype(int)
X = train.drop(columns=[TARGET_COL])
X_test = test.copy()

print(train.shape, test.shape)


# Basic info
display(train.head())
display(train.describe())

print("\nMissing values per column:")
print(train.isna().sum().sort_values(ascending=False))

# Target distribution
print("\nTarget distribution:")
print(y.value_counts(normalize=True).rename("fraction"))

# Simple Pearson correlation with target (numeric only)
corr_with_target = train.corr(numeric_only=True)[TARGET_COL].sort_values(ascending=False)
display(corr_with_target)



TARGET_COL = "diagnosed_diabetes"

# Target
y = train[TARGET_COL].astype(int)

# Raw features (with strings like 'Female', etc.)
X_raw = train.drop(columns=[TARGET_COL])
X_test_raw = test.copy()

# 1) One-hot encode all categorical/string features
X = pd.get_dummies(X_raw, drop_first=True)
X_test = pd.get_dummies(X_test_raw, drop_first=True)

# 2) Make sure train and test have exactly the same columns
X, X_test = X.align(X_test, join="left", axis=1)

# 3) Any columns that exist in train but not in test will be NaN in X_test → fill with 0
X_test = X_test.fillna(0)

print("X shape:", X.shape)
print("X_test shape:", X_test.shape)



import matplotlib.pyplot as plt

y.value_counts(normalize=True).plot(kind="bar")
plt.title("Diagnosed Diabetes (target) distribution")
plt.xlabel("diagnosed_diabetes")
plt.ylabel("fraction")
plt.show()



log_reg = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=500,
    class_weight="balanced",  # helps if target is slightly imbalanced
    random_state=RANDOM_STATE,
)

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", log_reg),
])

cv = StratifiedKFold(
    n_splits=15,
    shuffle=True,
    random_state=RANDOM_STATE
)

# CV performance
lr_auc_scores = cross_val_score(
    lr_pipe, X, y,
    cv=cv,
    scoring="roc_auc"
)
print(f"LogReg CV AUC: {lr_auc_scores.mean():.5f} ± {lr_auc_scores.std():.5f}")

# Fit on full train to extract coefficients
lr_pipe.fit(X, y)

coefs = lr_pipe.named_steps["log_reg"].coef_[0]
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coef": coefs,
})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)

display(coef_df.head(20))



TOP_N = 7  # adjust if you want more/less

top_features = coef_df.head(TOP_N)["feature"].tolist()
print("Top features:", top_features)

X_top = X[top_features].copy()
X_test_top = X_test[top_features].copy()



# 1) Logistic Regression (already defined as lr_pipe, we just pass X_top later)

# 2) XGBoost
xgb_clf = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    tree_method="hist",
)

# 3) LightGBM
lgb_clf = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=-8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=RANDOM_STATE,
)



def cv_model(clf, X, y, X_test, cv, model_name):
    """
    Returns:
        oof_preds: np.array of shape (len(X),)
        test_preds: np.array of shape (len(X_test),)
        scores: list of AUC per fold
    """
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = clone(clf)

        model.fit(X_tr, y_tr)

        oof_fold = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = oof_fold

        test_preds += model.predict_proba(X_test)[:, 1] / cv.get_n_splits()

        fold_auc = roc_auc_score(y_val, oof_fold)
        scores.append(fold_auc)
        print(f"{model_name} | Fold {fold} AUC: {fold_auc:.5f}")

    print(f"{model_name} | CV AUC: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    return oof_preds, test_preds, scores



# Reuse the same CV object as before
cv = StratifiedKFold(
    n_splits=15,
    shuffle=True,
    random_state=RANDOM_STATE
)

# 1) Logistic Regression (with scaling)
oof_lr, test_lr, scores_lr = cv_model(lr_pipe, X_top, y, X_test_top, cv, "LogReg")

# 2) XGBoost
oof_xgb, test_xgb, scores_xgb = cv_model(xgb_clf, X_top, y, X_test_top, cv, "XGBoost")

# 3) LightGBM
oof_lgb, test_lgb, scores_lgb = cv_model(lgb_clf, X_top, y, X_test_top, cv, "LightGBM")



mean_lr  = np.mean(scores_lr)
mean_xgb = np.mean(scores_xgb)
mean_lgb = np.mean(scores_lgb)

model_means = np.array([mean_lr, mean_xgb, mean_lgb])
weights = model_means / model_means.sum()

print("Model mean AUCs:", model_means)
print("Blend weights (LR, XGB, LGB):", weights)

# OOF blend (for honest CV AUC of the blend)
oof_blend = (
    weights[0] * oof_lr +
    weights[1] * oof_xgb +
    weights[2] * oof_lgb
)

blend_auc = roc_auc_score(y, oof_blend)
print(f"Blended OOF AUC: {blend_auc:.5f}")

# Test blend
test_blend = (
    weights[0] * test_lr +
    weights[1] * test_xgb +
    weights[2] * test_lgb
)



sub = pd.read_csv(DATA_DIR / "sample_submission.csv", index_col="id")
sub[TARGET_COL] = test_blend
sub.to_csv("submission_blend_lr_xgb_lgb.csv")
sub.head()

