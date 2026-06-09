import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


X = train.drop(columns=["id", "y"])
y = train["y"]

X_test = test.drop(columns=["id"])


for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# train_data = lgb.Dataset(X_train, label=y_train, categorical_feature="auto")
# val_data = lgb.Dataset(X_val, label=y_val, categorical_feature="auto")
model = LGBMClassifier(
    objective="binary",
    learning_rate=0.05,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    n_estimators=1000
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
)


val_preds = model.predict_proba(X_val)[:, 1]   # probability of class 1

print(val_preds[:10])  # e.g. [0.12, 0.87, 0.45, ...]


test_preds = model.predict_proba(X_test)
submission["y"] = test_preds
submission.to_csv("submission.csv", index=False)


submission


# CatBoost Stratified K-Fold template for binary classification (Kaggle-ready)
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

# --- 1) Load data (tries Kaggle path then local)
try:
    train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
    test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
    sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
except Exception:
    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

# --- 2) Basic setup
TARGET = "y"
ID = "id"

X = train.drop(columns=[ID, TARGET])
y = train[TARGET].values
X_test = test.drop(columns=[ID]).copy()

# --- 3) Detect categorical features automatically (object / category dtype)
cat_features = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
print("Detected categorical features:", cat_features)

# If you prefer to pass indices
cat_feature_indices = [X.columns.get_loc(c) for c in cat_features]

# --- 4) (Optional) Quick preprocessing: convert strings to category dtype (CatBoost accepts names or indices)
for c in cat_features:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")

# --- 5) CV & model params
NFOLDS = 5
SEED = 42

params = {
    "iterations": 10000,           # large, early stopping will stop sooner
    "learning_rate": 0.03,
    "depth": 6,
    "eval_metric": "AUC",
    "random_seed": SEED,
    "use_best_model": True,
    "verbose": 200,
    "task_type": "GPU",         # uncomment if you have GPU and CatBoost built with GPU support
    "devices": "0"              # optional GPU device id
}

# --- 6) Prepare arrays for out-of-fold preds and test predictions
oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []

skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n---- Fold {fold} ----")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    train_pool = Pool(X_tr, label=y_tr, cat_features=cat_features)
    val_pool   = Pool(X_val, label=y_val, cat_features=cat_features)
    test_pool  = Pool(X_test, cat_features=cat_features)

    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=300,   # stops if no improvement
        use_best_model=True,
    )

    # OOF predictions (prob of class 1)
    val_pred = model.predict_proba(X_val)[:, 1]
    oof[val_idx] = val_pred
    fold_auc = roc_auc_score(y_val, val_pred)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} ROC AUC: {fold_auc:.5f}")

    # test preds (average later)
    test_preds += model.predict_proba(X_test)[:, 1] / NFOLDS

# --- 7) CV results
print("\nCV AUC scores:", np.round(fold_scores, 5))
print("Mean CV AUC:", np.mean(fold_scores))
print("OOF ROC AUC:", roc_auc_score(y, oof))

# --- 8) Prepare submission (probabilities between 0 and 1)
sample[TARGET] = test_preds
out_name = "catboost_submission.csv"
sample.to_csv(out_name, index=False)
print(f"\nSaved {out_name}")


