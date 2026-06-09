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
import os 
import random
SEED = 42
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load & Preprocess Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


train.head()


test.head()


train.info()


TARGET = "diagnosed_diabetes"
ID_COL = "id"

# 1) Columns
print("Train columns:", train.columns.tolist())
print(f"*"*80)
print("Test  columns:", test.columns.tolist())

# 2) Target exists only in train
assert TARGET in train.columns, "Target not found in train!"
assert TARGET not in test.columns, "Target should not be in test!"
# 3) Missing values
missing_train = train.isna().sum().sort_values(ascending=False)
missing_test  = test.isna().sum().sort_values(ascending=False)

print(f"*"*80)
print("\nTop missing (train):")
print(missing_train.head(10))

print(f"*"*80)
print("\nTop missing (test):")
print(missing_test.head(10))

print(f"*"*80)
# 4) Duplicates in id (should be unique)
print("\nUnique train ids:", train[ID_COL].nunique(), " / rows:", len(train))
print("Unique test  ids:", test[ID_COL].nunique(),  " / rows:", len(test))


train[TARGET].value_counts(normalize=True)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,4))
train["diagnosed_diabetes"].value_counts(normalize=True).plot(
    kind="bar", rot=0
)
plt.title("Target Distribution (Diabetes)")
plt.ylabel("Proportion")
plt.show()


num_cols = train.select_dtypes(include=["int64","float64"]).columns
num_cols = [c for c in num_cols if c not in [ID_COL, TARGET]]
train[num_cols].hist(bins=40, figsize=(16,12))
plt.suptitle("Numerical Feature Distributions", y=1.02)
plt.show()


important_num = [
    "age",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "cholesterol_total"
]

plt.figure(figsize=(14,8))
for i, col in enumerate(important_num, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(
        x="diagnosed_diabetes",
        y=col,
        data=train
    )
    plt.title(col)

plt.tight_layout()
plt.show()


cat_cols = train.select_dtypes(include="object").columns

for col in cat_cols:
    plt.figure(figsize=(6,4))
    train[col].value_counts().plot(kind="bar")
    plt.title(col)
    plt.xticks(rotation=45)
    plt.show()


for col in cat_cols:
    plt.figure(figsize=(6,4))
    train.groupby(col)["diagnosed_diabetes"].mean().sort_values().plot(
        kind="bar"
    )
    plt.title(f"Diabetes Rate by {col}")
    plt.ylabel("P(Diabetes)")
    plt.xticks(rotation=45)
    plt.show()


plt.figure(figsize=(12,10))
corr = train[num_cols].corr()
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
plt.title("Numerical Feature Correlation")
plt.show()


TARGET = "diagnosed_diabetes"
ID_COL = "id"

X = train.drop(columns = [TARGET] )
y = train[TARGET].astype(int)

# identify the columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

if ID_COL in num_cols:
    num_cols.remove(ID_COL)

print(f"Num cols:", len(num_cols))
print(f"Cat cols:", len(cat_cols))
print(f"*" * 80)
print(f"Num cols:", num_cols)
print(f"*" * 80)
print(f"cat cols:", cat_cols)


X_proc = X.copy()
test_proc = test.copy()

# Fill missing categoricals and cast to category dtype
for c in cat_cols:
    X_proc[c] = X_proc[c].fillna("Unknown").astype("category")
    test_proc[c] = test_proc[c].fillna("Unknown").astype("category")

# Fill missing numericals (simple baseline)
for c in num_cols:
    med = X_proc[c].median()
    X_proc[c] = X_proc[c].fillna(med)
    test_proc[c] = test_proc[c].fillna(med)

X_proc.head()


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


c
import numpy as np

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
}

oof = np.zeros(len(X_proc))
test_pred = np.zeros(len(test_proc))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_proc, y), 1):
    X_tr, X_va = X_proc.iloc[tr_idx], X_proc.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMClassifier(**params, n_estimators=5000)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )

    va_prob = model.predict_proba(X_va)[:, 1]
    oof[va_idx] = va_prob

    fold_auc = roc_auc_score(y_va, va_prob)
    print(f"Fold {fold} AUC: {fold_auc:.6f} | Best iter: {model.best_iteration_}")

    test_pred += model.predict_proba(test_proc)[:, 1] / N_SPLITS

oof_auc = roc_auc_score(y, oof)
print(f"\nOOF AUC: {oof_auc:.6f}")


submission = pd.DataFrame({
    "id": test[ID_COL],
    "diagnosed_diabetes": test_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()


print("Submission shape:", submission.shape)
print("Min/Max prob:", submission["diagnosed_diabetes"].min(), submission["diagnosed_diabetes"].max())
print("Nulls:", submission.isna().sum().to_dict())

# Should be [0,1] probabilities with no nulls


train[TARGET].value_counts(normalize=True)


!pip -q install category_encoders


# Base features
X = train.drop(columns=[TARGET])
y = train[TARGET].astype(int)

# Identify categorical columns (needed for target encoding)
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Create feature-engineered copies
X_fe = X.copy()
test_fe = test.copy()

# High-ROI interactions
if "age" in X_fe.columns and "bmi" in X_fe.columns:
    X_fe["age_bmi"] = X_fe["age"] * X_fe["bmi"]
    test_fe["age_bmi"] = test_fe["age"] * test_fe["bmi"]

if "age" in X_fe.columns:
    X_fe["age_sq"] = X_fe["age"] ** 2
    test_fe["age_sq"] = test_fe["age"] ** 2

if "glucose" in X_fe.columns and "bmi" in X_fe.columns:
    X_fe["glucose_bmi"] = X_fe["glucose"] * X_fe["bmi"]
    test_fe["glucose_bmi"] = test_fe["glucose"] * test_fe["bmi"]

print("X_fe shape:", X_fe.shape)
print("test_fe shape:", test_fe.shape)
print("Categorical cols:", len(cat_cols))


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
N_SPLITS = 5

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)


params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "verbosity": -1,
    "seed": SEED,
}


import category_encoders as ce
import numpy as np

te_cols = cat_cols  # categorical columns only

oof_te = np.zeros(len(X_fe))
test_pred_te = np.zeros(len(test_fe))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_fe, y), 1):
    X_tr, X_va = X_fe.iloc[tr_idx].copy(), X_fe.iloc[va_idx].copy()
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    enc = ce.TargetEncoder(cols=te_cols, smoothing=10)
    X_tr_enc = enc.fit_transform(X_tr, y_tr)
    X_va_enc = enc.transform(X_va)
    test_enc = enc.transform(test_fe)

    model = lgb.LGBMClassifier(**params, n_estimators=5000)
    model.fit(
        X_tr_enc, y_tr,
        eval_set=[(X_va_enc, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )

    va_prob = model.predict_proba(X_va_enc)[:, 1]
    oof_te[va_idx] = va_prob

    print(f"Fold {fold} AUC (TE+LGB): {roc_auc_score(y_va, va_prob):.6f}")

    test_pred_te += model.predict_proba(test_enc)[:, 1] / N_SPLITS

print("OOF AUC (Target Encoding + LGB):", roc_auc_score(y, oof_te))


import xgboost as xgb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# Build preprocessor on X_fe schema
cat_cols_xgb = X_fe.select_dtypes(include=["object"]).columns.tolist()
num_cols_xgb = X_fe.select_dtypes(include=[np.number]).columns.tolist()
if ID_COL in num_cols_xgb:
    num_cols_xgb.remove(ID_COL)

preprocess_ohe = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols_xgb),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols_xgb),
    ],
    remainder="drop"
)


xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "n_estimators": 5000,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}

oof_xgb = np.zeros(len(X_fe))
test_pred_xgb = np.zeros(len(test_fe))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_fe, y), 1):
    X_tr, X_va = X_fe.iloc[tr_idx], X_fe.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # IMPORTANT: fit transformer on training fold only
    X_tr_m = preprocess_ohe.fit_transform(X_tr, y_tr)
    X_va_m = preprocess_ohe.transform(X_va)
    T_m    = preprocess_ohe.transform(test_fe)

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_tr_m, y_tr,
        eval_set=[(X_va_m, y_va)],
        early_stopping_rounds=200,
        verbose=False
    )

    va_prob = model.predict_proba(X_va_m)[:, 1]
    oof_xgb[va_idx] = va_prob

    fold_auc = roc_auc_score(y_va, va_prob)
    print(f"Fold {fold} AUC (XGB): {fold_auc:.6f} | Best iter: {model.best_iteration}")

    test_pred_xgb += model.predict_proba(T_m)[:, 1] / N_SPLITS

print("OOF AUC (XGB):", roc_auc_score(y, oof_xgb))


submission = pd.DataFrame({
    "id": test[ID_COL],
    "diagnosed_diabetes": test_pred_xgb
})

submission.to_csv("submission.csv", index=False)
submission.head()

print("Submission shape:", submission.shape)
print("Min/Max prob:", submission["diagnosed_diabetes"].min(), submission["diagnosed_diabetes"].max())
print("Nulls:", submission.isna().sum().to_dict())

# Should be [0,1] probabilities with no nulls




