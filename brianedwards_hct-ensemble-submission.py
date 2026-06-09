%reset -f

import warnings
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from lifelines import NelsonAalenFitter

warnings.simplefilter('ignore')

train = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv'
                   ).set_index('ID')
train.index = train.index.astype('int32')
test = pd.read_csv('../input/equity-post-HCT-survival-predictions/test.csv'
                   ).set_index('ID')
test.index = test.index.astype('int32')

X = pd.concat([train.drop(columns=['efs', 'efs_time']), test])
Xf = X.select_dtypes('float').astype('float32')
Xc = X.select_dtypes('object')
for c in Xc.columns:
    Xc[c], _ = Xc[c].factorize(use_na_sentinel=False)
    Xc[c] = Xc[c].astype('int32').astype('category')
X = pd.concat([Xf, Xc], axis=1)

Y = train[['efs', 'efs_time']]
naf = NelsonAalenFitter(label='estimate')
naf.fit(Y['efs_time'], event_observed=Y['efs'])
Y = Y.join(naf.cumulative_hazard_, on='efs_time')
Y['y'] = RobustScaler().fit_transform(-Y['estimate'].to_frame())
Y['y_cox'] = Y['efs_time']
Y.loc[Y['efs'] == 0, 'y_cox'] *= -1
y = Y['y']
y_cox = Y['y_cox']

kwargs_xgb = dict(max_depth=3, colsample_bytree=0.5, subsample=0.8, n_estimators=2000, learning_rate=0.02, enable_categorical=True, min_child_weight=80)
kwargs_cb = dict(learning_rate=0.1, grow_policy='Lossguide')

models = [
    (
        'lgb',
        lgb.LGBMRegressor(max_depth=3, colsample_bytree=0.4, n_estimators=2500, learning_rate=0.02, num_leaves=8, verbose=-1),
        y,
        {'eval_set': None},
    ), (
        'xgb',
        xgb.XGBRegressor(**kwargs_xgb),
        y,
        {},
    ), (
        'xgb_cox',
        xgb.XGBRegressor(**kwargs_xgb, objective='survival:cox', eval_metric='cox-nloglik'),
        y_cox,
        {},
    ), (
        'cb',
        cb.CatBoostRegressor(**kwargs_cb),
        y,
        {'eval_set': None, 'cat_features': Xc.columns.to_list(), 'silent': True},
    ), (
        'cb_cox',
        cb.CatBoostRegressor(**kwargs_cb, iterations=400, use_best_model=False),
        y_cox,
        {'eval_set': None, 'cat_features': Xc.columns.to_list(), 'silent': True},
    )
]

for i in range(len(models)):
    models[i] += (np.zeros(len(test)),)

n_splits = 10

for i, (fit_i, pred_i) in enumerate(KFold(n_splits=n_splits, shuffle=True).split(train)):
    for j, (name, m, y_, fit_kwargs, pred) in enumerate(models):
        if 'eval_set' in fit_kwargs:
            fit_kwargs['eval_set'] = [(X.iloc[pred_i], y_.iloc[pred_i])]
        m.fit(X.iloc[fit_i], y_.iloc[fit_i], **fit_kwargs)
        pred += m.predict(X.iloc[len(y_):])

submission = pd.DataFrame(np.sum([rankdata(pred) for _, _, _, _, pred in models], axis=0),
                          index=test.index,
                          columns=['prediction'])

submission.to_csv("submission.csv")




