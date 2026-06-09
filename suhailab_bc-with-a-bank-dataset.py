# Step 1: Imports & global config
import os, sys, glob, gc, warnings, json, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Try XGBoost if available; otherwise we'll fall back automatically.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def infer_comp_dir():
    """Find the competition input folder robustly."""
    default = "/kaggle/input/playground-series-s5e8"
    if os.path.exists(default):
        return default
    # Fallback: scan /kaggle/input for a folder with 'playground-series-s5e8' or 'bank'
    candidates = glob.glob("/kaggle/input/*")
    for c in candidates:
        name = os.path.basename(c).lower()
        if "playground-series-s5e8" in name or ("bank" in name and os.path.exists(os.path.join(c, "train.csv"))):
            return c
    # Last resort: search recursively for train.csv
    trains = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    return os.path.dirname(trains[0]) if trains else "."

COMP_DIR = infer_comp_dir()
print("COMP_DIR =", COMP_DIR)



# Step 2: Load data & infer id/target from sample_submission
train_path  = os.path.join(COMP_DIR, "train.csv")
test_path   = os.path.join(COMP_DIR, "test.csv")
sub_path    = os.path.join(COMP_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sub   = pd.read_csv(sub_path)

print("Shapes -> train:", train.shape, "| test:", test.shape, "| sample_submission:", sub.shape)
display(train.head(3))
display(sub.head(3))

# Infer columns from sample_submission
sub_cols = sub.columns.tolist()
assert len(sub_cols) == 2, f"Unexpected sample_submission columns: {sub_cols}"
ID_COL, TARGET_COL = sub_cols[0], sub_cols[1]
print("ID_COL:", ID_COL, "| TARGET_COL:", TARGET_COL)

# Sanity checks
assert ID_COL in train.columns and ID_COL in test.columns, "ID column not found in train/test!"
assert TARGET_COL in train.columns, "Target column not found in train! Check columns."

# Separate features/target
y_raw = train[TARGET_COL].copy()
X = train.drop(columns=[TARGET_COL])
X_test = test.copy()

print("Feature columns:", len(X.columns))
print("Target dtype:", y_raw.dtype, "| Unique:", pd.Series(y_raw).unique()[:10])



# Step 3: Target cleaning & feature typing

def to_binary(y):
    """Map target to {0,1} robustly."""
    if pd.api.types.is_numeric_dtype(y):
        # Assume already 0/1; if not, map min->0, max->1
        uniq = np.sort(pd.Series(y).dropna().unique())
        if set(uniq).issubset({0,1}):
            return y.astype(int), {0:0, 1:1}
        # Map smallest to 0, largest to 1
        m = {uniq[0]:0, uniq[-1]:1}
        return y.map(m), m
    # Object/string labels
    y_str = y.astype(str).str.lower().str.strip()
    # Common positive tokens
    pos_tokens = {"yes","y","true","t","positive","pos","1","subscribed"}
    mapped = y_str.apply(lambda v: 1 if any(tok in v for tok in pos_tokens) else 0)
    return mapped.astype(int), None

y, mapping = to_binary(y_raw)
print("Target mapped. Pos rate:", y.mean().round(4), "| Count pos/neg:", y.value_counts().to_dict())

# Feature types
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print(f"Numeric cols: {len(num_cols)} | Categorical cols: {len(cat_cols)}")
print("Missing values in train (top 10):")
display(train.isna().sum().sort_values(ascending=False).head(10))



# Step 4: Preprocess pipeline & CV
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
scaler = StandardScaler(with_mean=False)  # keeps sparse matrix compatible with linear models

preprocess = ColumnTransformer(
    transformers=[
        ("num", scaler, num_cols),
        ("cat", ohe, cat_cols)
    ],
    remainder="drop",
    sparse_threshold=1.0
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
auc = make_scorer(roc_auc_score, needs_proba=True)



# Step 5: Logistic Regression baseline (fast)
logreg = LogisticRegression(
    penalty="l2",
    solver="saga",          # supports large sparse OHE matrices
    max_iter=2000,
    n_jobs=-1,
    class_weight=None,      # set to 'balanced' if your pos rate is very low
    random_state=RANDOM_STATE
)

pipe_logreg = Pipeline(steps=[
    ("prep", preprocess),
    ("model", logreg)
])

scores_lr = cross_val_score(pipe_logreg, X, y, cv=cv, scoring=auc, n_jobs=-1)
print("LogReg AUC (CV):", np.mean(scores_lr).round(5), "+/-", np.std(scores_lr).round(5))



# Step 6: XGBoost (preferred) or RandomForest
models = {}

if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe_xgb = Pipeline(steps=[("prep", preprocess), ("model", xgb)])
    models["xgb"] = pipe_xgb
else:
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    pipe_rf = Pipeline(steps=[("prep", preprocess), ("model", rf)])
    models["rf"] = pipe_rf

cv_scores = {}
for name, model in models.items():
    s = cross_val_score(model, X, y, cv=cv, scoring=auc, n_jobs=-1)
    cv_scores[name] = (s.mean(), s.std())
    print(f"{name.upper()} AUC (CV): {s.mean():.5f} +/- {s.std():.5f}")

best_name = max(cv_scores, key=lambda k: cv_scores[k][0]) if cv_scores else "logreg"
print("Best so far:", best_name.upper())



# Step 7: Train best on full data & predict test

# Choose the pipeline
if best_name == "xgb" and HAS_XGB:
    final_pipe = models["xgb"]
elif best_name == "rf" and not HAS_XGB:
    final_pipe = models["rf"]
else:
    # fallback to logistic regression if above didn't run
    final_pipe = pipe_logreg

final_pipe.fit(X, y)

# Predict proba for submission (positive class)
test_proba = final_pipe.predict_proba(X_test)[:, 1]

# Build submission using sample_submission's column names
submission = sub.copy()
# Ensure alignment by ID column:
if ID_COL in test.columns:
    # merge to keep original order if needed
    # Here we assume sample_submission already in correct test order; otherwise, align by ID:
    sub_idx = submission[ID_COL].values
    # Create a mapping from test id to probability
    proba_map = pd.Series(test_proba, index=test[ID_COL].values).to_dict()
    submission[TARGET_COL] = [proba_map[i] for i in sub_idx]
else:
    # fallback: assume order matches
    submission[TARGET_COL] = test_proba

submission.to_csv("submission.csv", index=False)
submission.head()


