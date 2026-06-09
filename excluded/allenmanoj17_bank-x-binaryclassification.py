import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import os


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


DATA_DIR = "/kaggle/input/playground-series-s5e8"
WORK_DIR = "/kaggle/working"


train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")


print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print("\nTarget distribution (y):")
print(train['y'].value_counts(normalize=True).rename('ratio'))


train.head()


# FEATURE ENGINEERING 

RANDOM_STATE = 42       # for reproducibility
N_TE_FOLDS   = 5        # folds for Target Encoding
TE_SMOOTH    = 20.0     # smoothing strength for Target Encoding

# Helper 1:  log transform for skewed numeric features
def log1p_shift(series):
    """
    Apply log(1 + x) to reduce skew.
    If values are <= 0, shift them up before log transform.
    """
    min_val = series.min()
    shift = 1 - min_val if min_val <= 0 else 0
    return np.log1p(series + shift)

# Helper 2: Target Encoding (mean encoding with KFold)
def target_encode(train_col, test_col, target,
                  n_splits=5, smoothing=20.0, seed=42):
    """
    For each category, replace with the mean target value.
    Uses KFold to avoid data leakage.
    Smoothing blends category mean with overall mean.
    """
    train_col = train_col.reset_index(drop=True)
    test_col  = test_col.reset_index(drop=True)
    target    = target.reset_index(drop=True).astype(int)

    # Overall mean of target
    global_mean = target.mean()
    encoded_train = pd.Series(np.nan, index=train_col.index, dtype="float32")

    # Create folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(train_col):
        # Stats from training part of the fold
        stats = (
            pd.DataFrame({"cat": train_col.iloc[train_idx], "y": target.iloc[train_idx]})
            .groupby("cat")["y"].agg(["mean", "count"])
        )
        # Smoothed mean
        stats["enc"] = (stats["mean"] * stats["count"] + smoothing * global_mean) / (stats["count"] + smoothing)

        # Map validation part
        encoded_train.iloc[val_idx] = train_col.iloc[val_idx].map(stats["enc"]).astype("float32")

    # For test data: use full training data mapping
    full_stats = (
        pd.DataFrame({"cat": train_col, "y": target})
        .groupby("cat")["y"].agg(["mean", "count"])
    )
    full_stats["enc"] = (full_stats["mean"] * full_stats["count"] + smoothing * global_mean) / (full_stats["count"] + smoothing)
    encoded_test = test_col.map(full_stats["enc"]).astype("float32")

    # Fill missing with overall mean
    encoded_train = encoded_train.fillna(global_mean)
    encoded_test  = encoded_test.fillna(global_mean)

    return encoded_train, encoded_test

# initial data info
print("BEFORE FEATURE ENGINEERING")
print(f"Train shape: {train.shape} | Test shape: {test.shape}")
if "y" in train.columns:
    print(f"Target mean (positive rate): {train['y'].mean():.4f}")
print("-" * 50)

# Was customer contacted before? (pdays != 999)
train["contacted_before"] = (train["pdays"] != 999).astype(int)
test["contacted_before"]  = (test["pdays"] != 999).astype(int)

# Replace sentinel value 999 with -1 in pdays
train["pdays"] = train["pdays"].replace(999, -1)
test["pdays"]  = test["pdays"].replace(999, -1)

# Debt flag: loan == yes OR default == yes
train["any_debt"] = ((train["loan"] == "yes") | (train["default"] == "yes")).astype(int)
test["any_debt"]  = ((test["loan"] == "yes") | (test["default"] == "yes")).astype(int)

# Month to number mapping
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
             'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
train["month_num"] = train["month"].map(month_map)
test["month_num"]  = test["month"].map(month_map)

# Fix skewness in some numeric columns
skewed_columns = ["previous", "balance", "campaign", "pdays", "duration"]
for col in skewed_columns:
    train[f"{col}_log"] = log1p_shift(train[col])
    test[f"{col}_log"]  = log1p_shift(test[col])

# Encode categorical columns
categorical_cols = train.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns to encode:", categorical_cols)

for col in categorical_colsyes
    # Target encoding
    train[f"{col}_te"], test[f"{col}_te"] = target_encode(
        train[col], test[col], train["y"], n_splits=N_TE_FOLDS, smoothing=TE_SMOOTH, seed=RANDOM_STATE
    )

    # Frequency encoding (how many times each category appears)
    freq_map = train[col].value_counts()
    train[f"{col}_fe"] = train[col].map(freq_map)
    test[f"{col}_fe"]  = test[col].map(freq_map).fillna(0)

# drop raw string columns
train_encoded = train.drop(columns=categorical_cols).copy()
test_encoded  = test.drop(columns=categorical_cols).copy()

print("\nAFTER FEATURE ENGINEERING")
print(f"Train_encoded shape: {train_encoded.shape}")
print(f"Test_encoded shape : {test_encoded.shape}")
print("Remaining object columns:", train_encoded.select_dtypes(include=["object"]).columns.tolist())

# Peek at first few rows
print(train_encoded.head())


import xgboost as xgb

RANDOM_STATE = 42
N_SPLITS = 3
N_ESTIMATORS = 10000
EARLY_STOP = 200


X = train_encoded.drop(columns=["y"])
y = train_encoded["y"].astype(int)
X_test = test_encoded.copy()

# Handle imbalance
neg, pos = (y == 0).sum(), (y == 1).sum()
spw = neg / pos
print(f"Positives={pos}, Negatives={neg}, scale_pos_weight={spw:.2f}")

# Model parameters
params = dict(
    n_estimators=N_ESTIMATORS,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    eval_metric="auc",
    random_state=RANDOM_STATE,
    early_stopping_rounds=EARLY_STOP,
    tree_method="hist"  
)

# Training
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    oof[val_idx] = model.predict_proba(X_val)[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    fold_auc = roc_auc_score(y_val, oof[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.5f}")

# Overall CV AUC
cv_auc = roc_auc_score(y, oof)
print(f"\nCV AUC: {cv_auc:.5f}")




