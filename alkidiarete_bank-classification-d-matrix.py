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
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder
from itertools import combinations
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')

TARGET = 'y'
NUMS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATS = ['job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'poutcome']


train[CATS] = train[CATS].astype('category')
test[CATS] = test[CATS].astype('category')


%%time

columns = NUMS + CATS
for cols in tqdm(list(combinations(columns, 2))):
    name = '-'.join(cols)
    tmp = pd.concat([
        train[list(cols)].astype(str).agg('_'.join, axis=1),
        test[list(cols)].astype(str).agg('_'.join, axis=1)
    ], ignore_index=True)
    tmp, _ = tmp.factorize()
    train[name] = tmp[:len(train)]
    test[name] = tmp[len(train):]


def target_encode(train, valid, test, col, target=TARGET, kfold=5, smooth=20, agg='mean'):
    train = train.copy()
    train['kfold'] = train.index % kfold
    col_name = '_'.join(col)
    new_col = f'TE_{agg.upper()}_{col_name}'
    train[new_col] = 0.

    for i in range(kfold):
        df_tr = train[train['kfold'] != i]
        mn = getattr(train[target], agg)() if agg != 'nunique' else 0
        df_tmp = df_tr[col + [target]].groupby(col).agg([agg, 'count']).reset_index()
        df_tmp.columns = col + [agg, 'count']
        if agg == 'nunique':
            df_tmp['TE_tmp'] = df_tmp[agg] / df_tmp['count']
        else:
            df_tmp['TE_tmp'] = ((df_tmp[agg] * df_tmp['count']) + (mn * smooth)) / (df_tmp['count'] + smooth)
        idx = train['kfold'] == i
        train.loc[idx, new_col] = train.loc[idx, col].merge(df_tmp, on=col, how='left')['TE_tmp'].fillna(mn).values

    df_tmp = train[col + [target]].groupby(col).agg([agg, 'count']).reset_index()
    df_tmp.columns = col + [agg, 'count']
    mn = getattr(train[target], agg)() if agg != 'nunique' else 0
    if agg == 'nunique':
        df_tmp['TE_tmp'] = df_tmp[agg] / df_tmp['count']
    else:
        df_tmp['TE_tmp'] = ((df_tmp[agg] * df_tmp['count']) + (mn * smooth)) / (df_tmp['count'] + smooth)

    valid[new_col] = valid[col].merge(df_tmp, on=col, how='left')['TE_tmp'].fillna(mn).astype('float32').values
    test[new_col] = test[col].merge(df_tmp, on=col, how='left')['TE_tmp'].fillna(mn).astype('float32').values

    return train.drop('kfold', axis=1), valid, test


X = train.drop("y", axis=1).copy()
y = train["y"].copy()
X_test = test.copy()

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")


kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))


for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n===== Fold {fold+1} =====")

    X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    te = TargetEncoder(cols=cat_cols)
    X_train[cat_cols] = te.fit_transform(X_train[cat_cols], y_train)
    X_valid[cat_cols] = te.transform(X_valid[cat_cols])
    X_test_enc = X_test.copy()
    X_test_enc[cat_cols] = te.transform(X_test[cat_cols])

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test_enc, enable_categorical=True)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "gpu_hist",  
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }

    evals = [(dtrain, "train"), (dvalid, "valid")]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=200,
        verbose_eval=100
    )

    oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration)) / kf.n_splits

    print(f"AUC fold {fold+1}: {roc_auc_score(y_valid, oof_preds[valid_idx]):.4f}")

print("\nOverall AUC:", roc_auc_score(y, oof_preds))


submission = pd.DataFrame({
    "id": test.index,
    "y": test_preds
})
submission.to_csv("submission_xgb.csv", index=False)
submission.head()

