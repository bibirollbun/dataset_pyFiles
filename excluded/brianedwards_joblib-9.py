#!/usr/bin/env python

import warnings; warnings.simplefilter('ignore')
import os
import sys
from glob import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from lifelines import NelsonAalenFitter
from lifelines.utils import concordance_index
import joblib

def read_csv(p):
    df = pd.read_csv(p).set_index('ID')
    df.index = df.index.astype('int32')
    fn = p.split('/')[-1].split('.')[0]
    if fn.startswith('X_') and fn.endswith('_raw'):
        df = pd.concat([df.select_dtypes('float').astype('float32'),
                        df.select_dtypes('object').astype('category')],
                        axis=1)
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
    Xc_raw = X.select_dtypes('object').astype('category')
    for col in Xc_raw:
        if Xc_raw[col].isna().any():
            Xc_raw[col] = Xc_raw[col].cat.add_categories('Missing').fillna('Missing')
    X_raw = pd.concat([Xf, Xc_raw], axis=1)
    Xc = X.select_dtypes('object').astype('category')
    for col in Xc:
        Xc[col], _ = Xc[col].factorize(use_na_sentinel=False)
        Xc[col] = Xc[col].astype('int32').astype('category')
    X_label = pd.concat([Xf, Xc], axis=1)
    Xc = pd.get_dummies(Xc, drop_first=True, dtype='int32')
    Xc.columns = Xc.columns.str.replace(r'[\[\]<]', '_', regex=True)
    X_onehot = pd.concat([Xf, Xc], axis=1)
    naf = NelsonAalenFitter(label='y')
    naf.fit(train['efs_time'], event_observed=train['efs'])
    y = -train[['efs_time']].join(naf.cumulative_hazard_, on='efs_time')['y']
    path = f'./{name}'
    os.makedirs(path, exist_ok=True)
    filenames = []
    for n, data in [('X_train_raw', X_raw.iloc[:len(train)]),
                    ('X_train_label', X_label.iloc[:len(train)]),
                    ('X_train_onehot', X_onehot.iloc[:len(train)]),
                    ('X_test_raw', X_raw.iloc[:len(test)]),
                    ('X_test_label', X_label.iloc[len(train):]),
                    ('X_test_onehot', X_onehot.iloc[len(train):]),
                    ('y', y)]:
        filenames.append(f'{path}/{n}.csv')
        data.to_csv(filenames[-1])
        print(f'wrote {filenames[-1]}')

def get_Xs(name, train_or_test):
    path = f'./{name}'
    return [('raw', read_csv(f'{path}/X_{train_or_test}_raw.csv')),
            ('label', read_csv(f'{path}/X_{train_or_test}_label.csv')),
            ('onehot', read_csv(f'{path}/X_{train_or_test}_onehot.csv'))]

def create_kfold():
    return KFold(shuffle=True, random_state=1729)

def fit_fold_model(name, X_name, X, y, m_path, fold_n, i_fold, i_oof, m_name, m):
    m.fit(X.iloc[i_fold], y.iloc[i_fold])
    filename = f'{m_path}/{name}-{X_name}-{m_name}-{fold_n}.joblib'
    joblib.dump(m, filename)
    print(f'fold {fold_n} wrote {filename}')

def fit(name):
    print('\nfit')
    Xs = get_Xs(name, 'train')
    y = read_csv(f'./{name}/y.csv')
    m_path = f'../input/models'
    os.makedirs(m_path, exist_ok=True)
    kfold = create_kfold()
    models = []
    for m_name, m in [('xgb', xgb.XGBRegressor(enable_categorical=True)),
              ('lgb', lgb.LGBMRegressor(verbose=-1))]:
        for X_name, X in Xs:
            models.append((m_name, m, X_name, X))
    for X_name, X in Xs:
        kwargs = {}
        if 'raw' in X_name or 'label' in X_name:
            kwargs = dict(cat_features=X.select_dtypes('category').columns.to_list())
        m = cb.CatBoostRegressor(silent=True, **kwargs)
        models.append(('cb', m, X_name, X))
    args = []
    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(Xs[0][1])):
        for m_name, m, X_name, X in models:
            args.append((name, X_name, X, y, m_path, fold_n, i_fold, i_oof, m_name, m))
    with Pool(min(os.cpu_count(), kfold.get_n_splits() * len(models))) as pool:
        pool.starmap(fit_fold_model, args)

def calc_score(y_pred_oof):
    p = '../input/equity-post-HCT-survival-predictions'
    train = read_csv(f'{p}/train.csv')
    merged_df = train[['race_group', 'efs_time', 'efs']].assign(prediction=y_pred_oof)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        c_index_race = concordance_index(merged_df_race['efs_time'], -merged_df_race['prediction'], merged_df_race['efs'])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))

def cv_score(name):
    print('\nCV score')
    Xs = get_Xs(name, 'train')
    kfold = create_kfold()
    y_pred = np.zeros(len(Xs[0][1]))
    for X_name, X in Xs:
        for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X)):
            for filename in glob(f'../input/models/{name}-{X_name}-*-{fold_n}.joblib'):
                m = joblib.load(filename)
                print(f'read {filename}')
                y_pred[i_oof] += rankdata(m.predict(X.iloc[i_oof]))
    score = calc_score(y_pred)
    print(f'{score:.4f}')

def predict(name):
    print('\npredict')
    Xs = get_Xs(name, 'test')
    y_pred = np.zeros(len(Xs[0][1]))
    for X_name, X in Xs:
        for filename in glob(f'../input/models/{name}-{X_name}-*.joblib'):
            m = joblib.load(filename)
            print(f'read {filename}')
            y_pred += rankdata(m.predict(X))
    s = pd.DataFrame(y_pred, index=X.index, columns=['prediction'])
    s.to_csv('submission.csv')
    print('wrote submission.csv')

NAME = 'joblib-9'

if __name__ == '__main__':
    try:
        get_ipython
    except NameError:
        pass
    else:
        preprocess(NAME)  # kaggle notebook
        predict(NAME)
        sys.exit()

    preprocess(NAME)  # terminal
    fit(NAME)
    cv_score(NAME)
    predict(NAME)


