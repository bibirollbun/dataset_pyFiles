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


"""
Final LightGBM Pipeline for 'Binary Classification with a Bank Dataset'
- Modular functions
- Stratified CV
- Works with LightGBM >= 4.0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

SEED = 42

# ------------------------
# 1. Load Data
# ------------------------
def load_data(train_path, test_path):
    train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
    return train, test

# ------------------------
# 2. Feature Engineering
# ------------------------
def feature_engineering(df):
    df = df.copy()
    # Example numeric interaction
    if {'age','balance'}.issubset(df.columns):
        df['age_x_balance'] = df['age'] * (df['balance'] + 1)
    return df

# ------------------------
# 3. Prepare Features
# ------------------------
def prepare_features(train, test, target_col='y', exclude_cols=['id']):
    X = train.drop(columns=[target_col] + [c for c in exclude_cols if c in train.columns])
    y = train[target_col].values
    X_test = test.drop(columns=[c for c in exclude_cols if c in test.columns])
    return X, y, X_test

# ------------------------
# 4. Train LightGBM with CV
# ------------------------
def train_lgb_cv(X, y, X_test, n_splits=5, params=None):
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        clf = lgb.train(
            params,
            lgb_train,
            num_boost_round=10000,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=200)]
        )

        oof[val_idx] = clf.predict(X_val, num_iteration=clf.best_iteration)
        preds += clf.predict(X_test, num_iteration=clf.best_iteration) / n_splits

    score = roc_auc_score(y, oof)
    return oof, preds, score

# ------------------------
# 5. Save Submission
# ------------------------
def save_submission(test_ids, preds, out_path='submission.csv'):
    sub = pd.DataFrame({'id': test_ids, 'target': preds})
    sub.to_csv(out_path, index=False)
    print(f"âœ… Saved submission to {out_path}")

# ------------------------
# 6. Main Script
# ------------------------
if __name__ == '__main__':
    train_path = 'train.csv'
    test_path = 'test.csv'

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 128,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': SEED,
        'verbosity': -1,
        'n_jobs': -1
    }

    # Load
    train, test = load_data(train_path, test_path)

    # Convert target from yes/no â†’ 1/0
    if train['y'].dtype == 'object':
        train['y'] = (train['y'] == 'yes').astype(int)

    # Feature engineering
    train = feature_engineering(train)
    test = feature_engineering(test)

    # Prepare X, y, X_test
    X, y, X_test = prepare_features(train, test, target_col='y', exclude_cols=['id'])

    # Convert categorical columns
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    # Train & predict
    oof, preds, cv_score = train_lgb_cv(X, y, X_test, n_splits=5, params=params)
    print('ğŸ�† CV AUC:', cv_score)

    # Save submission
    save_submission(test['id'] if 'id' in test.columns else test.index, preds, out_path='submission.csv')

