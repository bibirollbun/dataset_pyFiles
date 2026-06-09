import polars as pl

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize,basinhopping
import optuna
from IPython.display import clear_output
import logging
logging.getLogger("optuna").setLevel(logging.WARNING)


class cfg:
    shift = 1
    history = False  #只使用登录率来进行预测  
    analyze = False #只是用模型值预测来预测
    ensemble = history and analyze #组合两种方式来进行预测
    optuna_optimize = True
    xgb_w = 0.0
    lgb_w = 1
    cat_w = 0.0


def smape(y_true, y_pred):
    scores = []
    for t, p in zip(y_true, y_pred):
        if t == 0 and p == 0:
            scores.append(0)
        else:
            scores.append(2 * abs(t - p) / (abs(t) + abs(p)))
    return 100 * np.mean(scores) 


sub = pd.read_csv('/kaggle/input/user-retention-prediction/submit_sample.csv')
oof_xgb = np.load('/kaggle/input/user-retention-prediction-re-interfence/user_retention_oof_xgb.npy')
test_xgb = np.load('/kaggle/input/user-retention-prediction-re-interfence/user_retention_test_xgb.npy')
df_valid = pd.read_parquet('/kaggle/input/user-retention-prediction-create-data/user_retention_df_valid.parquet')
fans_user_ids = np.load('/kaggle/input/user-retention-prediction-create-data/user_retention_fans_user_id.npy')
fans_user = pd.read_parquet('/kaggle/input/user-retention-prediction-create-data/user_retention_fans_user.parquet')
oof_lgb = np.load('/kaggle/input/user-retention-prediction-lgb-interfence/user_retention_oof_lgb.npy')
test_lgb = np.load('/kaggle/input/user-retention-prediction-lgb-interfence/user_retention_test_lgb.npy')
oof_cat = np.load('/kaggle/input/user-retention-prediction-cat-interfence/user_retention_oof_cat.npy')
test_cat = np.load('/kaggle/input/user-retention-prediction-cat-interfence/user_retention_test_cat.npy')


if cfg.optuna_optimize:
    def objective(trial):
        ratios = np.array([0.165077, 0.071544, 0.061548, 0.055787, 0.053056, 0.058299, 0.083344, 0.451345])
        add_list = [trial.suggest_float(f'{i}',0,0.2) if i in [0,5,6,7] else trial.suggest_float(f'{i}',0,0)
                    for i in range(8)]
        ratios = ratios + add_list
        ratios = ratios / sum(ratios)
        oof_xgb[df_valid.ID.isin(fans_user_ids)] += cfg.shift
        oof_lgb[df_valid.ID.isin(fans_user_ids)] += cfg.shift
        oof_cat[df_valid.ID.isin(fans_user_ids)] += cfg.shift
        oof = oof_xgb*cfg.xgb_w + oof_lgb*cfg.lgb_w + oof_cat*cfg.cat_w
        cumulative_ratios = np.cumsum(ratios[:-1])
        quantiles = np.quantile(oof, cumulative_ratios)
        bins = np.concatenate([[oof.min()], quantiles, [oof.max()]])
        oof_pred_pp = np.digitize(oof, bins, right=True) - 1
        df_valid['pred'] = np.clip(oof_pred_pp,0,7)
        clear_output()
        return smape(df_valid['future_7d_login'], df_valid['pred'])
        
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective,n_trials=1000,show_progress_bar=False)
    print('best_trials',study.best_trial)
    print('best_params',study.best_params)


add_list = []
if cfg.optuna_optimize:
    for k,v in study.best_params.items():
        add_list.append(v)
else:
    best_params = {'0': 0.019873672837908793, '1': 0.0001119785631350257, '2': 4.560208382018101e-05, '3': 0.007857259070707622, '4': 0.014328918085886641, '5': 0.024833689733851857, '6': 0.1349775808681694, '7': 0.19020660680542345}
    for k,v in best_params.items():
        add_list.append(v)


print(add_list)


test_xgb[sub.ID.isin(fans_user_ids)] += cfg.shift
test_lgb[sub.ID.isin(fans_user_ids)] += cfg.shift
test_cat[sub.ID.isin(fans_user_ids)] += cfg.shift
test = test_xgb*cfg.xgb_w + test_lgb*cfg.lgb_w + test_cat*cfg.cat_w
test_ratios = np.array([0.11310252, 0.06577263, 0.04719162, 0.05515901, 0.05475995,0.07644968, 0.12103906, 0.46647806])
ratios = test_ratios + np.array(add_list)
ratios = ratios / sum(ratios)
cumulative_ratios = np.cumsum(ratios[:-1])
quantiles = np.quantile(test, cumulative_ratios)
bins = np.concatenate([[test.min()], quantiles, [test.max()]])
pred_test_pp = np.digitize(test, bins, right=True) - 1
    
sub['pred'] = np.clip(pred_test_pp,0,7)
sub[['ID', 'pred']].to_csv('submission.csv', index=False)
print(sub['pred'].value_counts(normalize=True))
print(sub.loc[sub.ID.isin(fans_user_ids),'pred'].value_counts())

