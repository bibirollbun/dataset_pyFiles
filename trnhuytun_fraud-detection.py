!pip install -q imbalanced-learn


import os, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest


train = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/train.csv")
test  = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/test.csv")
sub   = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/sample_submission.csv")


# Anything in train not in test is a strong target candidate
candidates = [c for c in train.columns if c not in test.columns]
id_like = {"id","ID","index","Index","row_id","RowId","rowid"}
candidates = [c for c in candidates if c not in id_like]
TARGET_COL = candidates[0] if len(candidates)==1 else [c for c in candidates if c.lower() in {"class","target","label","is_fraud","fraud"}][0]

ID_COL = sub.columns[0]
PRED_COL = sub.columns[1]


feature_cols = [c for c in train.columns if c not in {TARGET_COL, ID_COL}]
X = train[feature_cols].copy()
y = train[TARGET_COL].astype(int).copy()
X_test = test[feature_cols].copy()

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", make_ohe())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop"
)

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


#Train IsolationForest
X_tr_t = preprocessor.fit_transform(X_tr)
X_va_t = preprocessor.transform(X_va)

iso = IsolationForest(
    n_estimators=3000,
    contamination=y.mean(),
    max_samples='auto',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_tr_t)


#Validation
val_scores = -iso.decision_function(X_va_t)
roc = roc_auc_score(y_va, val_scores)
pr  = average_precision_score(y_va, val_scores)

precision, recall, thresholds = precision_recall_curve(y_va, val_scores)
f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
best_idx = f1_scores.argmax()
best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

y_pred = (val_scores >= best_thresh).astype(int)
print(f"Validation ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")


#Train on full data
X_all_t  = preprocessor.fit_transform(X)
X_test_t = preprocessor.transform(X_test)

iso_full = IsolationForest(
    n_estimators=3000,
    contamination=y.mean(),
    max_samples='auto',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
iso_full.fit(X_all_t)

test_scores = -iso_full.decision_function(X_test_t)

sub_out = sub.copy()
sub_out[PRED_COL] = test_scores
sub_out.to_csv("submission.csv", index=False)

