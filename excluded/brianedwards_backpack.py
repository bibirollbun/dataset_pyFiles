import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn import set_config
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import catboost as cb


train = pd.read_csv('../input/playground-series-s5e2/train.csv')
train = train.set_index('id')
train.index = train.index.astype('int32')
train.columns = train.columns.str.replace(' ', '_')


X = train.drop(columns=['Price'])
Xf = X.select_dtypes('float').astype('float32')
Xc = X.select_dtypes('object')
for c in Xc.columns:
    Xc[c], _ = Xc[c].factorize(use_na_sentinel=False)
    Xc[c] = Xc[c].astype('int32').astype('category')
X = pd.concat([Xf, Xc], axis=1)


y = train['Price']


models = {
    'lgb': lgb.LGBMRegressor(verbose=-1),
    'cb': cb.CatBoostRegressor(silent=True, cat_features=Xc.columns.to_list()),
}

oofs = {
    'lgb': pd.Series(np.nan, index=y.index),
    'cb': pd.Series(np.nan, index=y.index),
}

for i, (fit_i, pred_i) in enumerate(KFold(shuffle=True, random_state=42).split(y.index)):
    for name, m in models.items():
        m.fit(X.iloc[fit_i],
              y.iloc[fit_i],
              eval_set=[(X.iloc[pred_i],
                         y.iloc[pred_i])])
        
        oofs[name].iloc[pred_i] = m.predict(X.iloc[pred_i])


print(f"{np.sqrt(mean_squared_error(y, oofs['lgb'])):.3f}")


print(f"{np.sqrt(mean_squared_error(y, oofs['cb'])):.3f}")


print(f"{np.sqrt(mean_squared_error(y, (oofs['lgb'] + oofs['cb']) / 2)):.3f}")


def weighted_sum(weights, preds):
    return sum(w * p for w, p in zip(weights, preds))

def weighted_rmse(weights, preds, y_true):
    return np.sqrt(((y_true - weighted_sum(weights, preds)) ** 2).mean())

both_oofs = oofs.values()

weights = minimize(weighted_rmse,
                   x0=[0.5, 0.5],
                   args=(both_oofs, y),
                   bounds=[(0, 1), (0, 1)],
                   constraints={'type': 'eq',  # weights sum to 100%
                                'fun': lambda weights: sum(weights) - 1})

weights = weights.x
print(f"{np.sqrt(mean_squared_error(y, weighted_sum(weights, both_oofs))):.3f}")


stacking_oofs = {
    'lgb,cb': pd.Series(np.nan, index=y.index),
    'cb,lgb': pd.Series(np.nan, index=y.index),
}

for i, (fit_i, pred_i) in enumerate(KFold(shuffle=True, random_state=42).split(y.index)):
    for name1, m1 in models.items():
        oof1 = oofs[name1]
        name2 = 'cb' if name1 == 'lgb' else 'lgb'
        m2 = models[name2]
        oof2 = stacking_oofs[f'{name1},{name2}']
        
        m2.fit(X.iloc[fit_i],
               y.iloc[fit_i] - m1.predict(X.iloc[fit_i]),
               eval_set=[(X.iloc[pred_i],
                          y.iloc[pred_i] - oof1.iloc[pred_i])])
        
        oof2.iloc[pred_i] = m2.predict(X.iloc[pred_i])


for name1 in models.keys():
    name2 = 'cb' if name1 == 'lgb' else 'lgb'
    stacking_key = f'{name1},{name2}'
    print(stacking_key)
    print(f"{np.sqrt(mean_squared_error(y, oofs[name1] + stacking_oofs[stacking_key])):.3f}\n")


set_config(enable_metadata_routing=True)
voting_m = VotingRegressor([('lgb', models['lgb']), ('cb', models['cb'])])
models['lgb'].set_fit_request(eval_set=True)
voting_oof = pd.Series(np.nan, index=y.index)

for i, (fit_i, pred_i) in enumerate(KFold(shuffle=True, random_state=42).split(y.index)):
   voting_m.fit(X.iloc[fit_i],
                y.iloc[fit_i], 
                eval_set=[(X.iloc[pred_i], y.iloc[pred_i])])
   voting_oof.iloc[pred_i] = voting_m.predict(X.iloc[pred_i])


print(f"{np.sqrt(mean_squared_error(y, voting_oof)):.3f}")




