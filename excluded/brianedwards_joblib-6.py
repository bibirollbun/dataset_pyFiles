#!/usr/bin/env python

import warnings; warnings.simplefilter('ignore')
import os
import sys
from glob import glob
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from lifelines import NelsonAalenFitter
import joblib

def read_csv(p):
    df = pd.read_csv(p).set_index('ID')
    df.index = df.index.astype('int32')
    print(f'read {p}')
    return df

def preprocess(name):
    print('preprocess')
    p = '../input/equity-post-HCT-survival-predictions'
    train = read_csv(f'{p}/train.csv')
    test = read_csv(f'{p}/test.csv')
    X = train.drop(columns=['efs', 'efs_time'])
    X = pd.concat([X, test])
    Xf = X.select_dtypes('float').astype('float32')
    Xc = X.select_dtypes('object')

    for col in Xc:
        Xc[col], _ = Xc[col].factorize(use_na_sentinel=False)
        Xc[col] = Xc[col].astype('int32').astype('category')

    Xc = pd.get_dummies(Xc, drop_first=True, dtype='int32')
    Xc.columns = Xc.columns.str.replace(r'[\[\]<]', '_', regex=True)
    X = pd.concat([Xf, Xc], axis=1)
    naf = NelsonAalenFitter(label='y')
    naf.fit(train['efs_time'], event_observed=train['efs'])
    y = -train[['efs_time']].join(naf.cumulative_hazard_, on='efs_time')['y']
    path = f'./{name}'
    os.makedirs(path, exist_ok=True)
    filenames = []

    for n, data in [('X_train', X.iloc[:len(train)]),
                    ('X_test', X.iloc[len(train):]),
                    ('y', y)]:
        filenames.append(f'{path}/{n}.csv')
        data.to_csv(filenames[-1])
        print(f'wrote {filenames[-1]}')

def fit_fold(name, X, y, split):
    fold_n, (i_fold, i_oof) = split
    m = xgb.XGBRegressor(enable_categorical=True)
    m.fit(X.iloc[i_fold], y.iloc[i_fold])
    m_path = f'../input/models'
    os.makedirs(m_path, exist_ok=True)
    filename = f'{m_path}/{name}-{fold_n}.joblib'
    joblib.dump(m, filename)
    print(f'fold {fold_n} wrote {filename}')

def fit(name):
    print('\nfold')
    path = f'./{name}'
    X = read_csv(f'{path}/X_train.csv')
    y = read_csv(f'{path}/y.csv')
    n_splits = 5
    kfold = KFold(n_splits=5, shuffle=True, random_state=1729)
    fit_fold_ = partial(fit_fold, name, X, y)
    with Pool(n_splits) as pool:
        pool.map(fit_fold_, enumerate(kfold.split(X)))

def predict(name):
    print('\npredict')
    X = read_csv(f'./{name}/X_test.csv')
    y_pred = np.zeros(len(X))
    for filename in glob(f'../input/models/{name}-*.joblib'):
        m = joblib.load(filename)
        print(f'read {filename}')
        y_pred += rankdata(m.predict(X))
    s = pd.DataFrame(y_pred, index=X.index, columns=['prediction'])
    s.to_csv('submission.csv')
    print('wrote submission.csv')

NAME = 'joblib-6'

def main():
    preprocess(NAME)
    fit(NAME)
    predict(NAME)

if __name__ == '__main__':
    try:
        get_ipython
        preprocess(NAME)
        predict(NAME)
    except NameError:
        main()


