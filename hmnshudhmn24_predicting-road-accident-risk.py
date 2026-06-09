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


# %%
"""
Predicting Road Accident Risk
Kaggle Playground Series - Season 5, Episode 10

Updated version:
- Fixed LightGBM `fit()` API (removed unsupported args in some Kaggle envs).
- Uses callbacks for early stopping and logging.
- Works even if categorical_features are integer encoded.

End-to-end notebook-style script.
"""

# %%
# Imports and settings
import os
import sys
import gc
import math
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

# %%
# Helper: locate files
def find_data_files():
    kaggle_path = Path('/kaggle/input/playground-series-s5e10')
    candidates = {}
    if kaggle_path.exists():
        for fname in ['train.csv', 'test.csv', 'sample_submission.csv']:
            p = kaggle_path / fname
            if p.exists():
                candidates[fname] = str(p)
    cwd = Path('.')
    for fname in ['train.csv', 'test.csv', 'sample_submission.csv']:
        p = cwd / fname
        if p.exists():
            candidates.setdefault(fname, str(p))
    return candidates

files = find_data_files()
if 'train.csv' not in files or 'test.csv' not in files:
    print('Error: Could not find train.csv and/or test.csv.')
    raise SystemExit()

print('Using files:')
pprint(files)

# %%
# Load data
train = pd.read_csv(files['train.csv'])
test = pd.read_csv(files['test.csv'])
if 'sample_submission.csv' in files:
    sample_submission = pd.read_csv(files['sample_submission.csv'])
else:
    sample_submission = None

print('\nTrain shape:', train.shape)
print('Test shape:', test.shape)

# %%
# Quick EDA
if 'accident_risk' not in train.columns:
    raise KeyError('Train does not contain target column "accident_risk"')

print('\nTarget summary:')
print(train['accident_risk'].describe())

# %%
# Preprocessing
def preprocess(train_df, test_df, target_col='accident_risk', id_col='id'):
    train = train_df.copy()
    test = test_df.copy()
    test_ids = test[id_col].values if id_col in test.columns else None
    y = train[target_col].values
    train = train.drop([target_col], axis=1)
    if id_col in train.columns:
        train = train.drop([id_col], axis=1)
    if id_col in test.columns:
        test = test.drop([id_col], axis=1)
    numeric_feats = train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = [c for c in train.columns if c not in numeric_feats]
    if numeric_feats:
        medians = train[numeric_feats].median()
        train[numeric_feats] = train[numeric_feats].fillna(medians)
        test[numeric_feats] = test[numeric_feats].fillna(medians)
    for col in categorical_feats:
        train[col] = train[col].fillna('MISSING').astype(str)
        test[col] = test[col].fillna('MISSING').astype(str)
        union_cats = sorted(set(train[col].unique()).union(set(test[col].unique())))
        train[col] = pd.Categorical(train[col], categories=union_cats).codes
        test[col] = pd.Categorical(test[col], categories=union_cats).codes
    return train, test, y, test_ids, numeric_feats, categorical_feats

X_train, X_test, y_train, test_ids, numeric_feats, categorical_feats = preprocess(train, test)

print('\nAfter preprocessing:')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# %%
# Modeling
def train_cv_predict(X, y, X_test=None, categorical_features=None, n_splits=5, seed=SEED):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(X_test.shape[0]) if X_test is not None else None
    feature_importances = []
    fold_scores = []

    try:
        import lightgbm as lgb
        use_lgb = True
    except Exception:
        print('LightGBM not available, using RandomForest fallback.')
        use_lgb = False

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if use_lgb:
            model = lgb.LGBMRegressor(
                n_estimators=10000,
                learning_rate=0.05,
                num_leaves=128,
                colsample_bytree=0.6,
                subsample=0.7,
                random_state=seed + fold,
                n_jobs=-1
            )
            callbacks = [
                lgb.early_stopping(150),
                lgb.log_evaluation(200)
            ]
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=callbacks)
            val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
            oof_preds[val_idx] = val_pred
            if X_test is not None:
                test_preds += model.predict(X_test, num_iteration=model.best_iteration_) / n_splits
            fi = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_, 'fold': fold+1})
            feature_importances.append(fi)
        else:
            model = RandomForestRegressor(n_estimators=300, random_state=seed+fold, n_jobs=-1)
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_val)
            oof_preds[val_idx] = val_pred
            if X_test is not None:
                test_preds += model.predict(X_test) / n_splits
            fi = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_, 'fold': fold+1})
            feature_importances.append(fi)

        fold_rmse = math.sqrt(mean_squared_error(y_val, val_pred))
        print(f'Fold {fold+1} RMSE: {fold_rmse:.6f}')
        fold_scores.append(fold_rmse)

        del X_tr, X_val, y_tr, y_val, model
        gc.collect()

    cv_rmse = math.sqrt(mean_squared_error(y, oof_preds))
    print('\nCV RMSE:', cv_rmse)
    fi_df = pd.concat(feature_importances, axis=0).reset_index(drop=True)
    fi_mean = fi_df.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
    return oof_preds, test_preds, fold_scores, cv_rmse, fi_mean

# %%
# Train
X_train_df = X_train.reset_index(drop=True)
X_test_df = X_test.reset_index(drop=True)

oof, test_preds, fold_scores, cv_rmse, feature_importance = train_cv_predict(
    X_train_df, y_train, X_test_df, n_splits=5, seed=SEED
)

# %%
# Plot feature importance
plt.figure(figsize=(8, 10))
plt.title('Top feature importances')
plt.barh(feature_importance['feature'].head(30)[::-1], feature_importance['importance'].head(30)[::-1])
plt.tight_layout()
plt.show()

# %%
# Submission
test_preds = np.clip(test_preds, 0, 1)
if test_ids is not None:
    submission = pd.DataFrame({'id': test_ids, 'accident_risk': test_preds})
else:
    submission = pd.DataFrame({'id': np.arange(len(test_preds)), 'accident_risk': test_preds})
submission.to_csv('submission.csv', index=False)
print('\nSaved submission.csv:')
print(submission.head())

# %%
print('Done.')

