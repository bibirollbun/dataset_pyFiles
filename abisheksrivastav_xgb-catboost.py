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




import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
NFOLD = 5
USE_GPU = True

TRAIN_PATH = '/kaggle/input/playground-series-s5e11/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e11/test.csv'
ORIG_PATH = '/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv'
OUTPUT_SUB = 'submission.csv'

np.random.seed(SEED)

def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)
    return df

print('Loading datasets...')
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
orig = pd.read_csv(ORIG_PATH)

print('Train shape:', train.shape, '| Test shape:', test.shape, '| Orig shape:', orig.shape)

if 'loan_paid_back' not in orig.columns:
    for cand in ['Loan_Status', 'target', 'loan_status']:
        if cand in orig.columns:
            orig.rename(columns={cand: 'loan_paid_back'}, inplace=True)

train_cols = set(train.columns)
orig_cols = set(orig.columns)
common_cols = list(train_cols.intersection(orig_cols))
orig = orig[common_cols]
train_full = pd.concat([train, orig], axis=0, ignore_index=True)
train_full = train_full.drop_duplicates(subset=[c for c in train.columns if c != 'loan_paid_back'])

ID_COL = 'id' if 'id' in train.columns else train.columns[0]
TARGET = 'loan_paid_back'

all_df = pd.concat([train_full.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
num_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in all_df.columns if c not in num_cols and c != ID_COL]

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')

train_num = pd.DataFrame(num_imputer.fit_transform(train_full[num_cols]), columns=num_cols)
test_num = pd.DataFrame(num_imputer.transform(test[num_cols]), columns=num_cols)

train_cat = train_full[cat_cols].astype(str).fillna('missing')
test_cat = test[cat_cols].astype(str).fillna('missing')

for col in cat_cols:
    le = LabelEncoder()
    le.fit(list(train_cat[col]) + list(test_cat[col]))
    train_cat[col] = le.transform(train_cat[col])
    test_cat[col] = le.transform(test_cat[col])

X_train = pd.concat([train_num, train_cat], axis=1)
X_test = pd.concat([test_num, test_cat], axis=1)
y = train_full[TARGET].astype(int)

X_train = reduce_mem_usage(X_train)
X_test = reduce_mem_usage(X_test)

skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
sub_lgb = np.zeros(len(X_test))
sub_xgb = np.zeros(len(X_test))
sub_cat = np.zeros(len(X_test))

fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f'Fold {fold+1}/{NFOLD}')
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'seed': SEED + fold,
        'n_jobs': -1,
        'verbose': -1
    }
    if USE_GPU:
        lgb_params['device_type'] = 'gpu'

    clf = lgb.LGBMClassifier(**lgb_params, n_estimators=1000)
    # lightgbm scikit-learn API may not accept `early_stopping_rounds` depending on version.
    # Use callbacks for compatibility.
    clf.fit(X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)])
    val_pred = clf.predict_proba(X_val)[:, 1]
    oof_lgb[val_idx] = val_pred
    sub_lgb += clf.predict_proba(X_test)[:, 1] / NFOLD
    print(' LGB AUC:', roc_auc_score(y_val, val_pred))

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.05,
        'max_depth': 6,
        'seed': SEED + fold,
        'tree_method': 'gpu_hist' if USE_GPU else 'hist'
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    bst = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'valid')], early_stopping_rounds=50, verbose_eval=100)
    val_pred = bst.predict(dval)
    oof_xgb[val_idx] = val_pred
    sub_xgb += bst.predict(xgb.DMatrix(X_test)) / NFOLD
    print(' XGB AUC:', roc_auc_score(y_val, val_pred))

    cat_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'eval_metric': 'AUC',
        'random_seed': SEED + fold,
        'verbose': 100,
        'early_stopping_rounds': 50,
        'task_type': 'GPU' if USE_GPU else 'CPU'
    }
    cat = CatBoostClassifier(**cat_params)
    cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    val_pred = cat.predict_proba(X_val)[:, 1]
    oof_cat[val_idx] = val_pred
    sub_cat += cat.predict_proba(X_test)[:, 1] / NFOLD
    print(' CAT AUC:', roc_auc_score(y_val, val_pred))

    avg_auc = roc_auc_score(y_val, (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_cat[val_idx]) / 3)
    fold_scores.append(avg_auc)
    print(f'Fold {fold+1} stacked AUC: {avg_auc:.5f}\n')

print('Mean CV AUC:', np.mean(fold_scores))

meta_train = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
meta_test = np.vstack([sub_lgb, sub_xgb, sub_cat]).T

meta_clf = LogisticRegression(max_iter=1000)
meta_clf.fit(meta_train, y)
meta_oof = meta_clf.predict_proba(meta_train)[:, 1]
meta_preds = meta_clf.predict_proba(meta_test)[:, 1]

print('Meta AUC:', roc_auc_score(y, meta_oof))

sub = pd.DataFrame({
    'id': test['id'] if 'id' in test.columns else range(len(test)),
    'loan_paid_back': meta_preds
})
sub.to_csv(OUTPUT_SUB, index=False)
print('Saved submission.csv with shape', sub.shape)


