%load_ext cudf.pandas

import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import optuna
import gc
from sklearn.model_selection import RepeatedStratifiedKFold
from pandas.errors import PerformanceWarning
from sklearn.metrics import roc_auc_score
from itertools import combinations
from xgboost import XGBClassifier
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=PerformanceWarning)
TARGET = 'y'
NUMS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')
orig = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', delimiter=';')
orig['y'] = orig['y'].replace({'yes': 1, 'no': 0})

train[CATS] = train[CATS].astype('category')
test[CATS] = test[CATS].astype('category')
orig[CATS] = orig[CATS].astype('category')

TE_columns = []

columns = NUMS + CATS

for r in [2]:
    for cols in tqdm(list(combinations(columns, r))):
        name = '-'.join(cols)

        train[name] = train[cols[0]].astype(str)
        for col in cols[1:]:
            train[name] = train[name] + '_' + train[col].astype(str)

        test[name] = test[cols[0]].astype(str)
        for col in cols[1:]:
            test[name] = test[name] + '_' + test[col].astype(str)

        orig[name] = orig[cols[0]].astype(str)
        for col in cols[1:]:
            orig[name] = orig[name] + '_' + orig[col].astype(str)
        
        combined = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]
        orig[name] = combined[len(train) + len(test):]

        TE_columns.append(name)

FEATURES = train.columns.tolist()
FEATURES.remove(TARGET)


def target_encode(train, valid, test, col, target=TARGET, kfold=5, smooth=3):
    train['kfold'] = ((train.index) % kfold)
    col_name = '_'.join(col)
    train[f'TE_MEAN_' + col_name] = 0.

    np.random.seed(42)
    
    for i in range(kfold):
        df_tmp = train[train['kfold'] != i]
        mn = train[target].mean()
        df_tmp = df_tmp[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
        df_tmp.columns = col + ['mean', 'count']
        df_tmp['TE_tmp'] = ((df_tmp['mean'] * df_tmp['count']) + (mn * smooth)) / (df_tmp['count'] + smooth)
        df_tmp_m = train[col + ['kfold', f'TE_MEAN_' + col_name]].merge(df_tmp, how='left', left_on=col, right_on=col)
        df_tmp_m.loc[df_tmp_m['kfold'] == i, f'TE_MEAN_' + col_name] = df_tmp_m.loc[df_tmp_m['kfold'] == i, 'TE_tmp']
        train[f'TE_MEAN_' + col_name] = df_tmp_m[f'TE_MEAN_' + col_name].fillna(mn).values

    df_tmp = train[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
    mn = train[target].mean()
    df_tmp.columns = col + ['mean', 'count']
    df_tmp['TE_tmp'] = ((df_tmp['mean'] * df_tmp['count']) + (mn * smooth)) / (df_tmp['count'] + smooth)
    
    df_tmp_m = valid[col].merge(df_tmp, how='left', left_on=col, right_on=col)
    valid[f'TE_MEAN_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    valid[f'TE_MEAN_' + col_name] = valid[f'TE_MEAN_' + col_name].astype('float32')

    df_tmp_m = test[col].merge(df_tmp, how='left', left_on=col, right_on=col)
    test[f'TE_MEAN_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    test[f'TE_MEAN_' + col_name] = test[f'TE_MEAN_' + col_name].astype('float32')

    train = train.drop('kfold', axis=1)
    train[f'TE_MEAN_' + col_name] = train[f'TE_MEAN_' + col_name].astype('float32')

    return (train, valid, test)

def count_encode(train, valid, test, col):
    counts = train[col].value_counts()

    train[f'CE_{col}'] = train[col].map(counts)
    valid[f'CE_{col}'] = valid[col].map(counts).fillna(0)
    test[f'CE_{col}'] = test[col].map(counts).fillna(0)
    return (train, valid, test)


oof = np.zeros(len(train))
pred = np.zeros(len(test))

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

for idx, (train_idx, val_idx) in enumerate(rskf.split(train, train[TARGET])):
    X_train, X_val = train.loc[train_idx, FEATURES], train.loc[val_idx, FEATURES]
    y_train, y_val = train.loc[train_idx, TARGET], train.loc[val_idx, TARGET]
    X_test = test.copy()

    X_train = pd.concat([X_train, orig[FEATURES]])
    y_train = pd.concat([y_train, orig[TARGET]])

    for col in tqdm(TE_columns):
        X_train, X_val, X_test = target_encode(pd.concat([X_train, y_train], axis=1), X_val, X_test, [col], smooth=2)
        X_train = X_train.drop(TARGET, axis=1)
        X_train, X_val, X_test = count_encode(X_train, X_val, X_test, col)
    
        X_train = X_train.drop(col, axis=1)
        X_val = X_val.drop(col, axis=1)
        X_test = X_test.drop(col, axis=1)

    model = XGBClassifier(
        n_estimators=10000,
        learning_rate=0.008,
        max_leaves=127,
        min_child_weight=1.5,
        max_depth=0,
        grow_policy='lossguide',
        tree_method='hist',
        subsample=0.85,
        colsample_bylevel=0.7,
        colsample_bytree=0.75,
        colsample_bynode=0.8,
        sampling_method='gradient_based',
        reg_alpha=2.5,
        reg_lambda=0.8,
        objective='binary:logistic',
        eval_metric='auc',
        early_stopping_rounds=300,
        random_state=42+idx,
        enable_categorical=True,
        device='cuda',
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=300
    )
    
    oof[val_idx] = model.predict_proba(X_val)[:, 1]
    pred += model.predict_proba(X_test)[:, 1] / 5

    print(f'Fold {idx + 1}: {roc_auc_score(y_val, oof[val_idx])}')

    del model, X_train, X_val, y_train, y_val, X_test
    gc.collect()

print(f'CV AUC: {roc_auc_score(train[TARGET], oof)}')


submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission['y'] = pred
submission.to_csv('submission.csv', index=False)

