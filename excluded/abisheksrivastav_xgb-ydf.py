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


!pip install -q ydf




import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import ydf

# ================= PATHS =================
TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"
ORIG_PATH  = "/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"

TARGET = "diagnosed_diabetes"
IDCOL = "id"
NFOLDS = 5
SEED = 42

# ================= LOAD =================
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
orig  = pd.read_csv(ORIG_PATH)

# ---- FORCE TARGET TO INT (CRITICAL FOR YDF) ----
train[TARGET] = train[TARGET].astype(int)
y = train[TARGET].values

# ================= FEATURE ENGINEERING =================

# ---- Base features ----
BASE = [c for c in train.columns if c not in [IDCOL, TARGET]]

# ---- Original dataset: COUNT features ONLY ----
ORIG_FEATS = []
for col in BASE:
    cnt = orig.groupby(col).size().reset_index(name=f"orig_count_{col}")
    train = train.merge(cnt, on=col, how="left")
    test  = test.merge(cnt, on=col, how="left")
    ORIG_FEATS.append(f"orig_count_{col}")

for c in ORIG_FEATS:
    train[c] = train[c].fillna(0)
    test[c]  = test[c].fillna(0)

# ---- Simple numeric interactions (safe) ----
if {"bmi", "age"}.issubset(train.columns):
    train["bmi_age"] = train["bmi"] * train["age"]
    test["bmi_age"]  = test["bmi"] * test["age"]

if {"systolic_bp", "diastolic_bp"}.issubset(train.columns):
    train["pulse_pressure"] = train["systolic_bp"] - train["diastolic_bp"]
    test["pulse_pressure"]  = test["systolic_bp"] - test["diastolic_bp"]

# ---- Final features ----
FEATURES = [c for c in train.columns if c not in [IDCOL, TARGET]]
X = train[FEATURES].copy()
X_test = test[FEATURES].copy()

# ================= ENCODING =================
cat_cols = X.select_dtypes("object").columns
for c in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([X[c], X_test[c]]).astype(str))
    X[c] = le.transform(X[c].astype(str))
    X_test[c] = le.transform(X_test[c].astype(str))

# ================= CV =================
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

oof_xgb = np.zeros(len(X))
oof_ydf = np.zeros(len(X))
pred_xgb = np.zeros(len(X_test))
pred_ydf = np.zeros(len(X_test))

# =====================================================
# XGBOOST (SAFE CONFIG)
# =====================================================
for fold, (tr, val) in enumerate(skf.split(X, y), 1):
    model = xgb.XGBClassifier(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        tree_method="hist",   # SAFE everywhere
        device="cuda",        # GPU if available, else CPU
        random_state=SEED + fold
    )
    model.fit(
        X.iloc[tr], y[tr],
        eval_set=[(X.iloc[val], y[val])],
        verbose=False
    )
    oof_xgb[val] = model.predict_proba(X.iloc[val])[:, 1]
    pred_xgb += model.predict_proba(X_test)[:, 1] / NFOLDS

print("XGBoost CV AUC:", roc_auc_score(y, oof_xgb))

# =====================================================
# YDF (EXPLICIT CLASSIFICATION, INT LABEL)
# =====================================================
for fold, (tr, val) in enumerate(skf.split(X, y), 1):
    df_tr = X.iloc[tr].copy()
    df_tr[TARGET] = y[tr].astype(int)  # MUST be int

    learner = ydf.GradientBoostedTreesLearner(
        label=TARGET,
        task=ydf.Task.CLASSIFICATION,   # CRITICAL
        num_trees=1000,
        max_depth=5,
        random_seed=SEED + fold
    )
    model = learner.train(df_tr)

    oof_ydf[val] = model.predict(X.iloc[val])
    pred_ydf += model.predict(X_test) / NFOLDS

print("YDF CV AUC:", roc_auc_score(y, oof_ydf))

# ================= FINAL BLEND =================
final_pred = 0.5 * pred_xgb + 0.5 * pred_ydf

submission = pd.DataFrame({
    "id": test[IDCOL],
    TARGET: final_pred
})

submission.to_csv("/kaggle/working/submission_xgb_ydf.csv", index=False)
print("Saved: submission_xgb_ydf.csv")


