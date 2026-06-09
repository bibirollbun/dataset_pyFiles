import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
import matplotlib.pyplot as plt
from learntools.time_series.style import *
from learntools.time_series.utils import plot_periodogram, seasonal_plot
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import optuna

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor

from sklearn.model_selection import KFold, train_test_split, cross_validate
import re


TRAIN_DF = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
TEST_DF = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
GDP_PC = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv')
GDP_PPP_PC = pd.read_csv('/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_ppp_per_capita.csv')
TRAIN_DF['date'] = pd.to_datetime(TRAIN_DF['date'])
# TRAIN_DF.drop('id', axis=1, inplace=True)

TEST_DF['date'] = pd.to_datetime(TEST_DF['date'])
# TEST_DF.drop('id', axis=1, inplace=True)
TEST_DF['num_sold'] = np.nan
TRAIN_DF


countries = TRAIN_DF['country'].unique()
YEARS = [
    '2010', '2011', '2012', 
    '2013', '2014', '2015', 
    '2016', '2017', '2018', '2019'
]


GDP_PC = GDP_PC.loc[GDP_PC['Country Name'].isin(countries), ['Country Name'] + YEARS]
GDP_PPP_PC = GDP_PPP_PC.loc[GDP_PPP_PC['Country Name'].isin(countries), ['Country Name'] + YEARS]


GDP_PPP_PC = GDP_PPP_PC.set_index('Country Name').stack().rename('GDP_PPP_PC')
GDP_PC = GDP_PC.set_index('Country Name').stack().rename('GDP_PC')


Xy = TRAIN_DF[['id', 'date', 'country', 'store', 'product', 'num_sold']].copy()
# Xy = Xy.dropna()
Xy = Xy.set_index(['date', 'country', 'store', 'product'])
Xy['num_sold'] = np.log1p(Xy['num_sold'])

Xy = Xy.unstack(['country', 'store', 'product'])
Xy = Xy.drop([
                ('num_sold', 'Kenya', 'Discount Stickers', 'Holographic Goose'), 
                ('num_sold', 'Canada', 'Discount Stickers', 'Holographic Goose'),
                ('id', 'Kenya', 'Discount Stickers', 'Holographic Goose'), 
                ('id', 'Canada', 'Discount Stickers', 'Holographic Goose')
            ], axis=1)

Xy = Xy.interpolate(method='linear', limit_direction='both', axis=0)
Xy


fourier_M = CalendarFourier(freq='M', order=24)
fourier_Q = CalendarFourier(freq='Q', order=8)
fourier_Y = CalendarFourier(freq='Y', order=6)
fourier_2Y = CalendarFourier(freq='2Y', order=4)
Xy = pd.concat([
    Xy,
    TEST_DF
    .set_index(['date', 'country', 'store', 'product'])
    .unstack(['country', 'store', 'product'])
    .drop([
        ('num_sold', 'Kenya', 'Discount Stickers', 'Holographic Goose'), 
        ('num_sold', 'Canada', 'Discount Stickers', 'Holographic Goose'),
        ('id', 'Kenya', 'Discount Stickers', 'Holographic Goose'), 
        ('id', 'Canada', 'Discount Stickers', 'Holographic Goose')
    ], axis=1)
])
# Xy
dp = DeterministicProcess(
    index=Xy.index,
    constant=False,
    order=0,
    seasonal=True,
    additional_terms=[fourier_M, fourier_Q, fourier_Y, fourier_2Y],
    drop=True
)


Xy_id       = (
                Xy.unstack().reset_index()
                .loc[Xy.unstack().reset_index()['level_0'] == 'id']
                .drop('level_0', axis=1).rename(columns={0 : 'id'})
              )
Xy_num_sold = (
                Xy.unstack().reset_index()
                .loc[Xy.unstack().reset_index()['level_0'] == 'num_sold']
                .drop('level_0', axis=1).rename(columns={0 : 'num_sold'})
              )
Xy_id = Xy_id.merge(Xy_num_sold, how='left', on=['country', 'product', 'store', 'date'])#.dropna()
# print(Xy_id)
Xy = (
    # Xy.unstack().reset_index()
    # .drop('level_0', axis=1)
    # .rename(columns={
    #                     0: 'num_sold'
    #                 })
    Xy_id
    .join(
        dp.in_sample(), 
        on='date'
    ).sort_values(by=['id'])
)
# print(Xy.columns)
Xy = Xy.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# print(Xy.shape)
Xy['year'] = Xy['date'].dt.year.apply(str)
Xy = Xy.join(GDP_PC, on=['country', 'year'], how='left')
Xy = Xy.join(GDP_PPP_PC, on=['country', 'year'], how='left')
Xy = Xy.drop('year', axis=1)
# print(Xy.shape)
# Xy
for col in ['country', 'store', 'product']:
    # Xy = (
    #     pd.concat(
    #         [Xy, pd.get_dummies(Xy[col])],
    #         axis=1
    #     )
    #     .drop(col, axis=1)
    # )
    Xy[col] = LabelEncoder().fit_transform(Xy[col])

# Xy['trend'] = MinMaxScaler().fit_transform(pd.DataFrame(data={'trend':Xy['trend']}))
# Xy['trend_squared'] = MinMaxScaler().fit_transform(pd.DataFrame(data={'trend_squared':Xy['trend_squared']}))
# Xy['trend_cubed'] = MinMaxScaler().fit_transform(pd.DataFrame(data={'trend_cubed':Xy['trend_cubed']}))
Xy = Xy.reset_index(drop=True)
Xy


idx_test =  Xy[Xy['date'] > pd.to_datetime('2016-12-31')].index
Xy_test = Xy.iloc[idx_test]
Xy = Xy.drop(idx_test)

idx_of_2013 = Xy[Xy['date'] == pd.to_datetime('2013-01-01')].index[0]
idx_of_2014 = Xy[Xy['date'] == pd.to_datetime('2014-01-01')].index[0]
idx_of_2015 = Xy[Xy['date'] == pd.to_datetime('2015-01-01')].index[0]
idx_of_2016 = Xy[Xy['date'] == pd.to_datetime('2016-01-01')].index[0]
cv_borders = [ idx_of_2015, idx_of_2016]
# idx_of_2013, idx_of_2014,
cv_borders


idx_test


dates = Xy['date']
Xy = Xy.drop('date', axis=1)
Xy_test = Xy_test.drop('date', axis=1)
Xy


def objective_xgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 10000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'objective': 'reg:squarederror',
        'tree_method':"hist",
        'device':"cuda"
    }
    mape = []
    for border_idx in cv_borders:
        X_train = Xy.loc[Xy.index <  border_idx].drop(['num_sold', 'id'], axis=1)
        X_val   = Xy.loc[Xy.index >= border_idx].drop(['num_sold', 'id'], axis=1)
        y_train = Xy.loc[Xy.index <  border_idx, ['num_sold']] 
        y_val  =  Xy.loc[Xy.index >= border_idx, ['num_sold']] 

        SS_ppp_pc = StandardScaler()
        SS_pc = StandardScaler()
        X_train['GDP_PPP_PC'] = SS_ppp_pc.fit_transform(pd.DataFrame(X_train['GDP_PPP_PC']))
        X_train['GDP_PC'] = SS_pc.fit_transform(pd.DataFrame(X_train['GDP_PC']))
        X_val['GDP_PPP_PC'] = SS_ppp_pc.transform(pd.DataFrame(X_val['GDP_PPP_PC']))
        X_val['GDP_PC'] = SS_pc.transform(pd.DataFrame(X_val['GDP_PC']))
        
        model = XGBRegressor(random_state=42, **param)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        mape.append(mean_absolute_percentage_error(y_val, y_val_pred))
        
    return np.mean(mape)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)

best_params_xgb = study_xgb.best_params
print("Best XGBoost Parameters:", best_params_xgb)
print(f"Best MAPE: {study_xgb.best_value:.4f}")


def objective_cat(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 500, 10000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'MAPE',
        'eval_metric': 'MAPE',
        'task_type': "GPU",
        'devices': '0-1'
    }
    mape = []
    
    for border_idx in cv_borders:
        X_train = Xy.loc[Xy.index <  border_idx].drop(['num_sold', 'id'], axis=1)
        X_val   = Xy.loc[Xy.index >= border_idx].drop(['num_sold', 'id'], axis=1)
        y_train = Xy.loc[Xy.index <  border_idx, ['num_sold']] 
        y_val  =  Xy.loc[Xy.index >= border_idx, ['num_sold']] 

        SS_ppp_pc = StandardScaler()
        SS_pc = StandardScaler()
        X_train['GDP_PPP_PC'] = SS_ppp_pc.fit_transform(pd.DataFrame(X_train['GDP_PPP_PC']))
        X_train['GDP_PC'] = SS_pc.fit_transform(pd.DataFrame(X_train['GDP_PC']))
        X_val['GDP_PPP_PC'] = SS_ppp_pc.transform(pd.DataFrame(X_val['GDP_PPP_PC']))
        X_val['GDP_PC'] = SS_pc.transform(pd.DataFrame(X_val['GDP_PC']))
        
        model = CatBoostRegressor(random_state=42, silent=True, **param)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        mape.append(mean_absolute_percentage_error(y_val, y_val_pred))
        
    return np.mean(mape)

study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(objective_cat, n_trials=50, show_progress_bar=True)

best_params_cat = study_cat.best_params
print("Best CatBoost Parameters:", best_params_cat)
print(f"Best MAPE: {study_cat.best_value:.4f}")


def objective_lgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', -1, 15), 
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        "verbosity": -1,
        'objective':'regression',  
        'metric':'mape',
        'device': 'gpu'
    }
    mape = []
    
    for border_idx in cv_borders:
        X_train = Xy.loc[Xy.index <  border_idx].drop(['num_sold', 'id'], axis=1)
        X_val   = Xy.loc[Xy.index >= border_idx].drop(['num_sold', 'id'], axis=1)
        y_train = Xy.loc[Xy.index <  border_idx, ['num_sold']] 
        y_val  =  Xy.loc[Xy.index >= border_idx, ['num_sold']] 

        SS_ppp_pc = StandardScaler()
        SS_pc = StandardScaler()
        X_train['GDP_PPP_PC'] = SS_ppp_pc.fit_transform(pd.DataFrame(X_train['GDP_PPP_PC']))
        X_train['GDP_PC'] = SS_pc.fit_transform(pd.DataFrame(X_train['GDP_PC']))
        X_val['GDP_PPP_PC'] = SS_ppp_pc.transform(pd.DataFrame(X_val['GDP_PPP_PC']))
        X_val['GDP_PC'] = SS_pc.transform(pd.DataFrame(X_val['GDP_PC']))
        
        model = LGBMRegressor(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        y_val_pred = model.predict(X_val)
        mape.append(mean_absolute_percentage_error(y_val, y_val_pred))
            
    return np.mean(mape)

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

best_params_lgb = study_lgb.best_params
print("Best LightGBM Parameters:", best_params_lgb)
print(f"Best MAPE: {study_lgb.best_value:.4f}")


print("Best XGBoost Parameters:", best_params_xgb)
print("Best CatBoost Parameters:", best_params_cat)
print("Best LightGBM Parameters:", best_params_lgb)


xgb_params = 1{
    'n_estimators': 550, 'learning_rate': 0.49354069588464095, 'max_depth': 7,
    'min_child_weight': 4, 'subsample': 0.6971983828205991, 'colsample_bytree': 0.6791282877743667,
    'gamma': 0.04110550367061416, 'reg_alpha': 0.000945297222960338, 'reg_lambda': 0.010378360920537146,
    'objective': 'reg:squarederror',
    'tree_method':"hist",
    'device':"cuda"
}

cb_params = {
    'iterations': 9901, 'learning_rate': 0.15716127086036244, 'depth': 3,
    'l2_leaf_reg': 0.3910740661892778, 'bagging_temperature': 0.9985942594002873, 'border_count': 110,
    'loss_function': 'MAPE',
    'eval_metric': 'MAPE',
    'task_type': "GPU",
    'devices': '0-1'
}

lgb_params = {
    'n_estimators': 8980, 'learning_rate': 0.12936320623465541, 'max_depth': 10,
    'num_leaves': 21, 'min_child_samples': 49, 'subsample': 0.6557014577866733,
    'colsample_bytree': 0.6182670233163058, 'reg_alpha': 0.002599984818444811, 'reg_lambda': 0.00022143744769188796,
    'objective':'regression',  
    'metric': "mape",
    'device': 'gpu'
}


mape_metrcis = {
    'xgb' : {'loged': [], 'basic': []},
    'cb'  : {'loged': [], 'basic': []},
    'lgb' : {'loged': [], 'basic': []},
    'vr'  : {'loged': [], 'basic': []},
    'sr'  : {'loged': [], 'basic': []}
}
    
for border_idx in cv_borders:
    print('cv', border_idx)
    X_train = Xy.loc[Xy.index <  border_idx].drop('num_sold', axis=1)
    X_val   = Xy.loc[Xy.index >= border_idx].drop('num_sold', axis=1)
    y_train = Xy.loc[Xy.index <  border_idx, 'num_sold'] 
    y_val  =  Xy.loc[Xy.index >= border_idx, 'num_sold'] 

    SS_ppp_pc = StandardScaler()
    SS_pc = StandardScaler()
    X_train['GDP_PPP_PC'] = SS_ppp_pc.fit_transform(pd.DataFrame(X_train['GDP_PPP_PC']))
    X_train['GDP_PC'] = SS_pc.fit_transform(pd.DataFrame(X_train['GDP_PC']))
    X_val['GDP_PPP_PC'] = SS_ppp_pc.transform(pd.DataFrame(X_val['GDP_PPP_PC']))
    X_val['GDP_PC'] = SS_pc.transform(pd.DataFrame(X_val['GDP_PC']))
    
    meta_model = LinearRegression()
    xgb_reg = XGBRegressor(**xgb_params, verbose=0)
    lgb_reg = LGBMRegressor(**lgb_params, verbosity=-1)
    cb_reg = CatBoostRegressor(**cb_params, verbose=0)
    stacking_model = StackingRegressor(
        estimators=[
            ('xgb', xgb_reg),
            ('lgb', lgb_reg),
            ('cb', cb_reg)
        ],
        final_estimator=meta_model
    )
    voting_model = VotingRegressor(
        estimators=[
            ('xgb', xgb_reg),
            ('lgb', lgb_reg),
            ('cb', cb_reg)
        ]
    )
    
    lgb_reg.fit(X_train, y_train)
    cb_reg.fit(X_train, y_train)
    xgb_reg.fit(X_train, y_train)
    stacking_model.fit(X_train, y_train)
    voting_model.fit(X_train, y_train)

    y_lgb_pred = lgb_reg.predict(X_val)
    y_cb_pred  = cb_reg.predict(X_val)
    y_xgb_pred = xgb_reg.predict(X_val)
    y_sr_pred  = stacking_model.predict(X_val)
    y_vr_pred  = voting_model.predict(X_val)
    
    mape_metrcis['lgb']['loged'].append(mean_absolute_percentage_error(y_val, y_lgb_pred))
    mape_metrcis['cb']['loged'].append(mean_absolute_percentage_error(y_val,  y_cb_pred))
    mape_metrcis['xgb']['loged'].append(mean_absolute_percentage_error(y_val, y_xgb_pred))
    mape_metrcis['sr']['loged'].append(mean_absolute_percentage_error(y_val,  y_sr_pred))
    mape_metrcis['vr']['loged'].append(mean_absolute_percentage_error(y_val,  y_vr_pred))

    mape_metrcis['lgb']['basic'].append(mean_absolute_percentage_error(np.expm1(y_val), np.expm1(y_lgb_pred)))
    mape_metrcis['cb']['basic'].append(mean_absolute_percentage_error(np.expm1(y_val),  np.expm1(y_cb_pred)))
    mape_metrcis['xgb']['basic'].append(mean_absolute_percentage_error(np.expm1(y_val), np.expm1(y_xgb_pred)))
    mape_metrcis['sr']['basic'].append(mean_absolute_percentage_error(np.expm1(y_val),  np.expm1(y_sr_pred)))
    mape_metrcis['vr']['basic'].append(mean_absolute_percentage_error(np.expm1(y_val),  np.expm1(y_vr_pred)))

print('CatBoost Models mean MAPE:')
print(f"\tLog:   {np.mean(mape_metrcis['cb']['loged'])}")
print(f"\tBasic: {np.mean(mape_metrcis['cb']['basic'])}")

print('XGBoost Models mean MAPE:')
print(f"\tLog:   {np.mean(mape_metrcis['xgb']['loged'])}")
print(f"\tBasic: {np.mean(mape_metrcis['xgb']['basic'])}")

print('LGBoost Models mean MAPE:')
print(f"\tLog:   {np.mean(mape_metrcis['lgb']['loged'])}")
print(f"\tBasic: {np.mean(mape_metrcis['lgb']['basic'])}")

print('Stacking Models mean MAPE:')
print(f"\tLog:   {np.mean(mape_metrcis['sr']['loged'])}")
print(f"\tBasic: {np.mean(mape_metrcis['sr']['basic'])}")

print('Voting Models mean MAPE:')
print(f"\tLog:   {np.mean(mape_metrcis['vr']['loged'])}")
print(f"\tBasic: {np.mean(mape_metrcis['vr']['basic'])}")

print('All Metrics')
from pprint import pprint
pprint(mape_metrcis)


Xy_test


xgb_reg = XGBRegressor(**xgb_params, verbose=0)
lgb_reg = LGBMRegressor(**lgb_params, verbosity=-1)
cb_reg = CatBoostRegressor(**cb_params, verbose=0)

voting_model = VotingRegressor(
    estimators=[
        ('xgb', xgb_reg),
        ('lgb', lgb_reg),
        ('cb', cb_reg)
    ]
)

# SS_ppp_pc = StandardScaler()
# SS_pc = StandardScaler()
# Xy['GDP_PPP_PC'] = SS_ppp_pc.fit_transform(pd.DataFrame(Xy['GDP_PPP_PC']))
# Xy['GDP_PC'] = SS_pc.fit_transform(pd.DataFrame(Xy['GDP_PC']))
# Xy_test['GDP_PPP_PC'] = SS_ppp_pc.transform(pd.DataFrame(Xy_test['GDP_PPP_PC']))
# Xy_test['GDP_PC'] = SS_pc.transform(pd.DataFrame(Xy_test['GDP_PC']))
lgb_reg.fit(Xy.drop(['num_sold', 'id'], axis=1), Xy['num_sold'])
predicts = lgb_reg.predict(Xy_test.drop(['num_sold', 'id'], axis=1))
# voting_model.fit(Xy.drop(['num_sold', 'id'], axis=1), Xy['num_sold'])
# predicts = voting_model.predict(Xy_test.drop(['num_sold', 'id'], axis=1))

# stacking_model = StackingRegressor(
#         estimators=[
#             ('xgb', xgb_reg),
#             ('lgb', lgb_reg),
#             ('cb', cb_reg)
#         ],
#         final_estimator=meta_model
#     )

# stacking_model.fit(Xy.drop(['num_sold', 'id'], axis=1), Xy['num_sold'])
# predicts = stacking_model.predict(Xy_test.drop(['num_sold', 'id'], axis=1))


submission_pattern = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
submission_pattern['num_sold'] = np.nan
submission_pattern


Xy_test['num_sold'] = predicts
Xy_test['id'] = Xy_test['id'].apply(round)
# Xy_test
y_submit = submission_pattern.merge(Xy_test[['id', 'num_sold']], on='id', how='left').drop('num_sold_x', axis=1).rename(columns={'num_sold_y':'num_sold'})
y_submit.loc[y_submit['num_sold'].isna()]
y_submit = y_submit.fillna(0)
y_submit.loc[y_submit['num_sold'].isna()]
y_submit['num_sold'] = np.expm1(y_submit['num_sold'])
y_submit


y_submit.to_csv('29_01_v.csv', index=False)

