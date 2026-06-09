!pip install -q flaml lightgbm xgboost imbalanced-learn


# --- Light setup. On Kaggle most packages exist; otherwise uncomment:
# !pip install -q flaml lightgbm xgboost imbalanced-learn

import os, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix
)
from flaml import AutoML



train = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/train.csv")
test  = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/test.csv")
sub   = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/sample_submission.csv")

print("train:", train.shape, "test:", test.shape)
display(train.head(3))
display(sub.head())



# Anything in train not in test is a strong target candidate
candidates = [c for c in train.columns if c not in test.columns]

# Common id-like names to exclude from accidental target pick
id_like = {"id","ID","index","Index","row_id","RowId","rowid"}

candidates = [c for c in candidates if c not in id_like]
if len(candidates) == 1:
    TARGET_COL = candidates[0]
else:
    guess = [c for c in candidates if c.lower() in {"class","target","label","is_fraud","fraud"}]
    TARGET_COL = guess[0] if len(guess) >= 1 else (candidates[0] if candidates else "Class")

# Use sample_submission to infer the ID column name
ID_COL = sub.columns[0]
PRED_COL = sub.columns[1]
print("ğŸ�¯ TARGET_COL:", TARGET_COL, "| ğŸ†” ID_COL:", ID_COL, "| ğŸ“¤ PRED_COL:", PRED_COL)




print("Dtypes:", train.dtypes.value_counts())
missing = train.isna().mean().sort_values(ascending=False).head(10)
print("\nTop-10 missing rate:\n", missing)

vc = train[TARGET_COL].value_counts(dropna=False)
print("\nClass counts:\n", vc)
print("\nClass ratios:\n", (vc/len(train)).round(6))

plt.figure()
vc.sort_index().plot(kind="bar")
plt.title("Target Class Distribution")
plt.xlabel("Class"); plt.ylabel("Count")
plt.show()




# Drop target & ID from features (avoid leakage!)
feature_cols = [c for c in train.columns if c not in {TARGET_COL, ID_COL}]
X = train[feature_cols].copy()
y = train[TARGET_COL].astype(int).copy()
X_test = test[feature_cols].copy()

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print(f"ğŸ”¢ Numeric: {len(num_cols)} | ğŸ”¤ Categorical: {len(cat_cols)}")

# sklearn compatibility for OneHotEncoder (sparse_output introduced in >=1.2)
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # fallback for older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    # with_mean=False is safer if the final matrix is sparse
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

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)




automl = AutoML()
settings = {
    "task": "classification",
    "metric": "roc_auc",
    "time_budget": 180,            # adjust as you like
    "estimator_list": ["lgbm"],    # LightGBM only for speed
    "eval_method": "holdout",
    "split_ratio": 0.2,
    "early_stop": True,
    "seed": 42,
    "verbose": 0,
}

# Fit preprocessing on train and transform both train/valid
X_tr_t = preprocessor.fit_transform(X_tr)
X_va_t = preprocessor.transform(X_va)

automl.fit(X_tr_t, y_tr, **settings)




valid_proba = automl.predict_proba(X_va_t)[:, 1]
roc = roc_auc_score(y_va, valid_proba)
pr  = average_precision_score(y_va, valid_proba)
print(f"âœ… Valid ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")

precision, recall, thresholds = precision_recall_curve(y_va, valid_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
best_idx = f1_scores.argmax()
best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
print(f"Suggested threshold (max F1): {best_thresh:.4f}")

y_pred = (valid_proba >= best_thresh).astype(int)
print("\nClassification report:\n", classification_report(y_va, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_va, y_pred))




X_all_t  = preprocessor.fit_transform(X)   # refit on ALL training data
X_test_t = preprocessor.transform(X_test)

automl_full = AutoML()
automl_full.fit(X_all_t, y, **settings)

test_proba = automl_full.predict_proba(X_test_t)[:, 1]

# Build submission EXACTLY like sample_submission
sub_out = sub.copy()
# If sample rows equal test rows, assign by order; else align by ID join
if len(sub_out) == len(test):
    sub_out[PRED_COL] = test_proba
else:
    tmp = test[[ID_COL]].copy()
    tmp[PRED_COL] = test_proba
    sub_out = sub[[ID_COL]].merge(tmp, on=ID_COL, how="left")

out_path = "submission.csv"
sub_out.to_csv(out_path, index=False)
print("âœ… Submission saved ->", os.path.abspath(out_path))
display(sub_out.head())


import pickle

# Save the full preprocessing + automl model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump((preprocessor, automl_full), f)





