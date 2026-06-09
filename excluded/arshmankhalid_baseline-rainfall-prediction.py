%%time

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore
from IPython.display import clear_output
import seaborn as sns

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import *
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
from lightgbm import LGBMRegressor
import lightgbm as lgb
from tqdm import tqdm

def print_heading(title):
    print("#" * 50)
    print(f" {title} ")
    print("#" * 50)


%%time

SEED = 42
n_splits = 5

train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')
sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

train.head()


%%time

print_heading("Train Shape")
print(train.shape)
print_heading("Test Shape")
print(test.shape)


%%time

print_heading("Train Null Values")
print(train.isnull().sum())

print_heading("Test Null Values")
print(test.isnull().sum())


%%time

cat_cols = train.select_dtypes(include='object').columns
print_heading('CAT_COLS')
print(f"{cat_cols}\n")

num_cols = train.select_dtypes(include='float').columns
print_heading('NUM_COLS')
print(f"{num_cols}")


%%time

def update(df):

    for col in cat_cols:
        df[col] = df[col].astype('category')

    return df

train = update(train)
test = update(test)


%%time

train_data = train.drop('rainfall',axis=1)
train_labels = train['rainfall']

def train_lgbm(train_data, train_labels, test_data, n_splits=10, seed=42):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    test_preds = np.zeros(len(test_data))
    oof_preds = np.zeros(len(train_data))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, train_labels)):
        X_train, X_val = train_data[train_idx], train_data[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]

        model = lgb.LGBMClassifier(random_state=seed,verbose=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc")

        # Out-of-Fold Predictions (Validation set)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        # Test Predictions (Averaging over folds)
        test_preds += model.predict_proba(test_data)[:, 1] / n_splits

        # AUC for Fold
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold+1}: Validation AUC = {fold_auc:.5f}")

    # **Final Metrics Calculation**
    train_auc = roc_auc_score(train_labels, oof_preds)  # OOF AUC = Full Train AUC
    print(f"\nâœ… Overall Train AUC (OOF): {train_auc:.5f}")

    return test_preds


%%time

test_preds = train_lgbm(train_data.values, train_labels.values, test.values)


%%time

sample["rainfall"] = test_preds
sample.to_csv("submission.csv", index=False)
print_heading("Sub shape:")
print(sample.shape)
sample.head()

