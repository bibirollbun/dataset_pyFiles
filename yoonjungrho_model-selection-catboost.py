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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
from scipy.stats import ttest_ind, ks_2samp

# ë�°ì�´í„° ì¤€ë¹„
train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')

# ê³µë°± ì œê±°
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()


display(train.head()) # 58 columns
display(test.head()) # 57 columns (- Class column)


display(train.info())
display(test.info())


# Feature ì„¤ì •
cat_cols = ['EJ']  # ë²”ì£¼í˜• ë³€ìˆ˜
num_cols = train.columns.tolist()[1:-1]  # ì²« ë²ˆì§¸ ì—´(ID) ì œì™¸, ë§ˆì§€ë§‰ ì—´(Class) ì œì™¸
num_cols = [col for col in num_cols if col not in cat_cols]  # ë²”ì£¼í˜• ë³€ìˆ˜ ì œê±°

# ë²”ì£¼í˜• ë³€ìˆ˜ ì�¸ì½”ë”©
encoder = LabelEncoder()
train[cat_cols[0]] = encoder.fit_transform(train[cat_cols[0]].astype(str))
test[cat_cols[0]] = encoder.transform(test[cat_cols[0]].astype(str))


import sys
sys.path.append('/kaggle/input/iterativestratification')


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------
# Custom Balanced Log Loss
# ---------------------------
def balanced_log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    y_pred = y_pred / np.sum(y_pred, axis=1)[:, None]
    nc = np.bincount(y_true)
    w0, w1 = 1/(nc[0]/y_true.shape[0]), 1/(nc[1]/y_true.shape[0])
    logloss = (-w0/nc[0]*(np.sum(np.where(y_true==0,1,0)*np.log(y_pred[:,0]))) - w1/nc[1]*(np.sum(np.where(y_true!=0,1,0)*np.log(y_pred[:,1])))) / (w0+w1)
    return logloss

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def load_and_preprocess_data():
    train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
    test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
    greeks = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/greeks.csv')
    
    # Clean column names
    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()
    greeks.columns = greeks.columns.str.strip()
    
    # Merge greek features with train only
    train = train.merge(greeks, on='Id', how='left')
    
    # Define feature columns
    num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in ['Class', 'Id']:
        if col in num_cols:
            num_cols.remove(col)
    
    cat_cols = ['EJ']
    if 'EJ' in num_cols:
        num_cols.remove('EJ')
    
    greek_features = ['Alpha']  # Only use Alpha for stratification
    feature_cols = num_cols + cat_cols

    # Feature Engineering: Log-transform skewed numerical features
    for col in num_cols:
        if (train[col] > 0).all() and (test[col] > 0).all():
            train[f'log_{col}'] = np.log1p(train[col])
            test[f'log_{col}'] = np.log1p(test[col])
            num_cols.append(f'log_{col}')

    # Feature Engineering: Interaction Features
    train['AB_EL'] = train['AB'] / (train['EL'] + 1e-6)
    test['AB_EL'] = test['AB'] / (test['EL'] + 1e-6)
    train['BQ_EL'] = train['BQ'] / (train['EL'] + 1e-6)
    test['BQ_EL'] = test['BQ'] / (test['EL'] + 1e-6)
    num_cols.extend(['AB_EL', 'BQ_EL'])

    # Impute missing values
    train[num_cols] = train[num_cols].fillna(train[num_cols].median())
    test[num_cols] = test[num_cols].fillna(train[num_cols].median())
    
    # Encode categorical features
    for col in cat_cols:
        encoder = LabelEncoder()
        train[col] = encoder.fit_transform(train[col].astype(str))
        test[col] = encoder.transform(test[col].astype(str))
    
    # Encode Alpha for stratification
    encoder = LabelEncoder()
    train['Alpha'] = encoder.fit_transform(train['Alpha'].astype(str))
    
    return train, test, feature_cols

train, test, feature_cols = load_and_preprocess_data()

# ---------------------------
# Adversarial Validation
# ---------------------------
adv_train = train[feature_cols].copy()
adv_test = test[feature_cols].copy()
adv_train['is_test'] = 0
adv_test['is_test'] = 1
adv_data = pd.concat([adv_train, adv_test], axis=0)

X_adv = adv_data.drop('is_test', axis=1)
y_adv = adv_data['is_test']

X_adv_train, X_adv_val, y_adv_train, y_adv_val = train_test_split(X_adv, y_adv, test_size=0.2, random_state=42)
adv_model = cb.CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0)
adv_model.fit(X_adv_train, y_adv_train, eval_set=[(X_adv_val, y_adv_val)])
adv_preds = adv_model.predict_proba(X_adv_val)[:, 1]
auc = roc_auc_score(y_adv_val, adv_preds)
print("Adversarial Validation AUC:", auc)

if auc > 0.7:
    adv_scores = adv_model.predict_proba(adv_train.drop('is_test', axis=1))[:, 1]
    train['adv_score'] = adv_scores
    mask = train['adv_score'] > train['adv_score'].quantile(0.5)  # Less aggressive threshold
    train = train[mask].copy()
    train = train.reset_index(drop=True)

# ---------------------------
# Feature Selection
# ---------------------------
prelim_model = cb.CatBoostClassifier(iterations=1000, learning_rate=0.01, depth=6, verbose=0)
prelim_model.fit(train[feature_cols], train['Class'])
importances = prelim_model.get_feature_importance()
feature_importance = pd.Series(importances, index=feature_cols)
selected_features = feature_importance[feature_importance > feature_importance.quantile(0.3)].index.tolist()
print("Selected Features:", selected_features)
feature_cols = selected_features

# ---------------------------
# Model Training with Seed Ensembling (CatBoost + XGBoost + LightGBM)
# ---------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
seeds = [42, 2021, 7]

# CatBoost Parameters
catboost_params = {
    'iterations': 10000,
    'learning_rate': 0.05,
    'early_stopping_rounds': 1000,
    'depth': 4,
    'l2_leaf_reg': 0.5,
    'subsample': 0.7,
    'verbose': 1000
}

# XGBoost Parameters
xgboost_params = {
    'n_estimators': 10000,
    'learning_rate': 0.05,
    'early_stopping_rounds': 1000,
    'eval_metric': 'logloss',
    'max_depth': 4,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_lambda': 1,
}

# LightGBM Parameters
lightgbm_params = {
    'n_estimators': 10000,
    'learning_rate': 0.05,
    'early_stopping_rounds': 1000,
    'max_depth': 4,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_lambda': 1,
    'verbosity': -1
}

oof_catboost_all = np.zeros((len(train), 2))
oof_xgboost_all = np.zeros((len(train), 2))
oof_lightgbm_all = np.zeros((len(train), 2))
test_preds_catboost_all = []
test_preds_xgboost_all = []
test_preds_lightgbm_all = []

for seed in seeds:
    print(f"\nTraining with seed: {seed}")
    catboost_params['random_seed'] = seed
    xgboost_params['random_state'] = seed
    lightgbm_params['random_state'] = seed
    
    oof_catboost = np.zeros((len(train), 2))
    oof_xgboost = np.zeros((len(train), 2))
    oof_lightgbm = np.zeros((len(train), 2))
    test_preds_catboost = []
    test_preds_xgboost = []
    test_preds_lightgbm = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['Alpha'])):
        X_train = train.loc[train_idx, feature_cols]
        X_val = train.loc[val_idx, feature_cols]
        y_train = train.loc[train_idx, 'Class']
        y_val = train.loc[val_idx, 'Class']

        # CatBoost
        model_cb = cb.CatBoostClassifier(**catboost_params)
        print(f"Training CatBoost fold {fold + 1}")
        model_cb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        oof_catboost[val_idx, :] = model_cb.predict_proba(X_val)
        test_preds_catboost.append(model_cb.predict_proba(test[feature_cols]))

        # XGBoost
        model_xgb = xgb.XGBClassifier(**xgboost_params)
        print(f"Training XGBoost fold {fold + 1}")
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1000)
        oof_xgboost[val_idx, :] = model_xgb.predict_proba(X_val)
        test_preds_xgboost.append(model_xgb.predict_proba(test[feature_cols]))

        # LightGBM
        model_lgb = lgb.LGBMClassifier(**lightgbm_params)
        print(f"Training LightGBM fold {fold + 1}")
        model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        oof_lightgbm[val_idx, :] = model_lgb.predict_proba(X_val)
        test_preds_lightgbm.append(model_lgb.predict_proba(test[feature_cols]))
    
    oof_catboost_all += oof_catboost
    oof_xgboost_all += oof_xgboost
    oof_lightgbm_all += oof_lightgbm
    test_preds_catboost_all.append(np.mean(test_preds_catboost, axis=0))
    test_preds_xgboost_all.append(np.mean(test_preds_xgboost, axis=0))
    test_preds_lightgbm_all.append(np.mean(test_preds_lightgbm, axis=0))

# Average predictions over seeds
final_oof_catboost = oof_catboost_all / len(seeds)
final_oof_xgboost = oof_xgboost_all / len(seeds)
final_oof_lightgbm = oof_lightgbm_all / len(seeds)
final_test_preds_catboost = np.mean(test_preds_catboost_all, axis=0)
final_test_preds_xgboost = np.mean(test_preds_xgboost_all, axis=0)
final_test_preds_lightgbm = np.mean(test_preds_lightgbm_all, axis=0)

# Ensemble: Weighted average (60% CatBoost + 20% XGBoost + 20% LightGBM)
final_oof = 0.6 * final_oof_catboost + 0.2 * final_oof_xgboost + 0.2 * final_oof_lightgbm
final_test_preds = 0.6 * final_test_preds_catboost + 0.2 * final_test_preds_xgboost + 0.2 * final_test_preds_lightgbm

# Post-Processing: Clip and normalize predictions
final_oof = np.clip(final_oof, 1e-15, 1-1e-15)
final_oof = final_oof / np.sum(final_oof, axis=1)[:, None]
final_test_preds = np.clip(final_test_preds, 1e-15, 1-1e-15)
final_test_preds = final_test_preds / np.sum(final_test_preds, axis=1)[:, None]

print("Final CatBoost OOF Balanced Log Loss:", balanced_log_loss(train['Class'], final_oof_catboost))
print("Final XGBoost OOF Balanced Log Loss:", balanced_log_loss(train['Class'], final_oof_xgboost))
print("Final LightGBM OOF Balanced Log Loss:", balanced_log_loss(train['Class'], final_oof_lightgbm))
print("Final Ensemble OOF Balanced Log Loss:", balanced_log_loss(train['Class'], final_oof))

# ---------------------------
# Prepare Submission
# ---------------------------
sample_submission = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')
sample_submission[['class_0', 'class_1']] = final_test_preds
sample_submission.to_csv('submission.csv', index=False)
sample_submission




