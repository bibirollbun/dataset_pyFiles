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


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression # <--- New Meta-Model
import warnings

warnings.filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/predicting-euphoria-in-the-streets/train.csv")
test_df = pd.read_csv("/kaggle/input/predicting-euphoria-in-the-streets/test.csv")


print(train_df.info())


print(train_df.describe().T)


test_ids = test_df['id']
train_labels = train_df['Y'].astype(int)
train_len = len(train_df)
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)
all_data = pd.concat([train_df.drop('Y', axis=1), test_df], ignore_index=True)
all_data = all_data.reset_index(drop=True)


numerical_cols = [col for col in all_data.columns if col.startswith('x_') and col not in ['x_13', 'x_15']]
all_data = all_data.replace([np.inf, -np.inf], np.nan)
for col in ['x_3', 'x_5']:
    all_data[f'{col}_missing'] = all_data[col].isnull().astype(int)
all_data['nan_count'] = all_data.isnull().sum(axis=1)
all_data['row_mean'] = all_data[numerical_cols].mean(axis=1)
for col in ['x_1', 'x_2']:
    all_data[f'{col}_rank'] = all_data[col].rank(method='dense')
all_data['x6_x7_prod'] = all_data['x_6'] * all_data['x_7']

imputer = SimpleImputer(strategy='median')
all_data_imputed = imputer.fit_transform(all_data)
all_data_columns_unique = list(pd.Series(all_data.columns).drop_duplicates())

X_imputed = pd.DataFrame(all_data_imputed[:, :len(all_data_columns_unique)], columns=all_data_columns_unique)
X = X_imputed.iloc[:train_len]
X_test = X_imputed.iloc[train_len:]
y = train_labels


NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
lgbm_sub_preds = np.zeros(X_test.shape[0])
xgb_sub_preds = np.zeros(X_test.shape[0])
oof_lgbm_preds = np.zeros(X.shape[0])
oof_xgb_preds = np.zeros(X.shape[0])

# Hyperparameters (Optimized for AUC)
lgbm_params = {
    'objective': 'binary', 'metric': 'auc', 'n_estimators': 3000,
    'learning_rate': 0.01, 'num_leaves': 20, 'max_depth': 5,
    'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1,
    'reg_lambda': 0.1, 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
}

xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_estimators': 3000,
    'learning_rate': 0.01, 'max_depth': 5, 'subsample': 0.7,
    'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': 42, 'n_jobs': -1, 'use_label_encoder': False,
}


print("\nStarting Stacking Base Model Training...")

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    
    lgb_model = lgb.LGBMClassifier(**lgbm_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)])
    oof_lgbm_preds[valid_idx] = lgb_model.predict_proba(X_valid)[:, 1]
    lgbm_sub_preds += lgb_model.predict_proba(X_test)[:, 1] / folds.n_splits

    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    oof_xgb_preds[valid_idx] = xgb_model.predict_proba(X_valid)[:, 1]
    xgb_sub_preds += xgb_model.predict_proba(X_test)[:, 1] / folds.n_splits



X_meta = np.column_stack((oof_lgbm_preds, oof_xgb_preds))

X_meta_test = np.column_stack((lgbm_sub_preds, xgb_sub_preds))

# (Logistic Regression)
meta_model = LogisticRegression(solver='liblinear', C=0.1, random_state=42)
meta_model.fit(X_meta, y)

stacked_sub_preds = meta_model.predict_proba(X_meta_test)[:, 1]
stacked_oof_preds = meta_model.predict_proba(X_meta)[:, 1]
overall_auc_stacked = roc_auc_score(y, stacked_oof_preds)


print(f"**Overall STACKED AUC Score (LGBM + XGB + LogReg): {overall_auc_stacked:.4f}**")
print(f"Meta-Model Weights (LGBM, XGBoost): {meta_model.coef_[0]}")

submission_df = pd.DataFrame({'id': test_ids, 'Y': stacked_sub_preds})
submission_file_name = 'sub_9.csv'
submission_df.to_csv(submission_file_name, index=False)
print(f"\nSubmission file '{submission_file_name}' created successfully.")


import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import RidgeClassifierCV 
import warnings

warnings.filterwarnings('ignore')


test_ids = test_df['id']
train_labels = train_df['Y'].astype(int)

train_len = len(train_df)
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)
all_data = pd.concat([train_df.drop('Y', axis=1), test_df], ignore_index=True)
all_data = all_data.reset_index(drop=True)


# ---Feature Engineering---

all_data = all_data.replace([np.inf, -np.inf], np.nan)

numerical_cols = [col for col in all_data.columns if col.startswith('x_') and col not in ['x_13', 'x_15']]

missing_cols = ['x_3', 'x_5', 'x_9', 'x_14']
for col in missing_cols:
    all_data[f'{col}_missing'] = all_data[col].isnull().astype(int)

all_data['nan_count'] = all_data.isnull().sum(axis=1)


all_data['row_mean'] = all_data[numerical_cols].mean(axis=1)
all_data['row_std'] = all_data[numerical_cols].std(axis=1)
all_data['row_min'] = all_data[numerical_cols].min(axis=1)
all_data['row_max'] = all_data[numerical_cols].max(axis=1)
all_data['row_skew'] = all_data[numerical_cols].skew(axis=1)


for col in numerical_cols:
    all_data[f'{col}_rank'] = all_data[col].rank(method='dense')


all_data['x6_x7_prod'] = all_data['x_6'] * all_data['x_7']
all_data['x10_div_x11'] = all_data['x_10'] / (all_data['x_11'] + 1e-6)
all_data['x12_log'] = np.log1p(all_data['x_12'])
all_data['x1_x2_sum'] = all_data['x_1'] + all_data['x_2']
all_data['x1_div_x4'] = all_data['x_1'] / (all_data['x_4'] + 1e-6)

all_data['Y'] = pd.concat([train_labels, pd.Series([np.nan] * len(test_df))]).reset_index(drop=True)

key = 'x_15'
for col in ['x_2', 'x_6', 'x_12']:
    map_dict = all_data.iloc[:train_len].groupby(key)[col].mean()
    all_data[f'{key}_mean_{col}'] = all_data[key].map(map_dict)

map_dict = all_data.iloc[:train_len].groupby(key)['Y'].mean()
all_data[f'{key}_mean_target'] = all_data[key].map(map_dict)

key = 'x_13'
for col in ['x_1', 'x_12']:
    map_dict = all_data.iloc[:train_len].groupby(key)[col].mean()
    all_data[f'{key}_mean_{col}'] = all_data[key].map(map_dict)

all_data = all_data.drop('Y', axis=1)


imputer = SimpleImputer(strategy='median')
all_data_imputed = imputer.fit_transform(all_data)

all_data_columns_unique = []
for col in all_data.columns:
    if col not in all_data_columns_unique:
        all_data_columns_unique.append(col)

X_imputed = pd.DataFrame(all_data_imputed[:, :len(all_data_columns_unique)], columns=all_data_columns_unique)

X = X_imputed.iloc[:train_len]
X_test = X_imputed.iloc[train_len:]
y = train_labels

print(f"Final feature matrix shape: {X.shape}")


NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
lgbm_sub_preds = np.zeros(X_test.shape[0])
xgb_sub_preds = np.zeros(X_test.shape[0])
oof_lgbm_preds = np.zeros(X.shape[0])
oof_xgb_preds = np.zeros(X.shape[0])

lgbm_params = {
    'objective': 'binary', 'metric': 'auc', 'n_estimators': 3000,
    'learning_rate': 0.01, 'num_leaves': 20, 'max_depth': 5,
    'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1,
    'reg_lambda': 0.1, 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
}

xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_estimators': 3000,
    'learning_rate': 0.01, 'max_depth': 5, 'subsample': 0.7,
    'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': 42, 'n_jobs': -1, 'use_label_encoder': False,
}


print("\nStarting 5-Fold Cross-Validation ...")

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # --- Train LightGBM ---
    lgb_model = lgb.LGBMClassifier(**lgbm_params)
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)])

    oof_lgbm_preds[valid_idx] = lgb_model.predict_proba(X_valid)[:, 1]
    lgbm_sub_preds += lgb_model.predict_proba(X_test)[:, 1] / folds.n_splits

    # --- Train XGBoost (No early stopping args for compatibility) ---
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  verbose=False)

    oof_xgb_preds[valid_idx] = xgb_model.predict_proba(X_valid)[:, 1]
    xgb_sub_preds += xgb_model.predict_proba(X_test)[:, 1] / folds.n_splits


X_meta = np.column_stack((oof_lgbm_preds, oof_xgb_preds))

X_meta_test = np.column_stack((lgbm_sub_preds, xgb_sub_preds))

meta_model = LogisticRegression(solver='liblinear', C=0.1, random_state=42)
meta_model.fit(X_meta, y)

stacked_sub_preds = meta_model.predict_proba(X_meta_test)[:, 1]
stacked_oof_preds = meta_model.predict_proba(X_meta)[:, 1]
overall_auc_stacked = roc_auc_score(y, stacked_oof_preds)


print(f"**Overall STACKED AUC Score (LGBM + XGB + LogReg): {overall_auc_stacked:.4f}**")
print(f"Meta-Model Weights (LGBM, XGBoost): {meta_model.coef_[0]}")

submission_df = pd.DataFrame({'id': test_ids, 'Y': stacked_sub_preds})
submission_file_name = 'sub_10.csv'
submission_df.to_csv(submission_file_name, index=False)
print(f"\nSubmission file '{submission_file_name}' created successfully.")




