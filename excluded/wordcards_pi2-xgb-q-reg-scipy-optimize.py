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


import warnings
warnings.simplefilter('ignore')

from scipy.optimize import minimize

import polars as pl
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_pinball_loss, mean_squared_error

from xgboost import XGBRegressor


PATH = '../input/prediction-interval-competition-ii-house-price/'
train = pl.read_csv(PATH + 'dataset.csv', try_parse_dates=True).with_columns(
    pl.col('sale_date').dt.year().alias('sale_year')
)

test = pl.read_csv(PATH + 'test.csv', try_parse_dates=True).with_columns(
    pl.col('sale_date').dt.year().alias('sale_year')
)

MAX_PRICE = train['sale_price'].max()
MIN_PRICE = train['sale_price'].min()


zoning_vc = train['zoning'].value_counts()
top_zoning = zoning_vc.sort(
    'count', descending=True)[:32]['zoning'].to_list()
zoning_vc = zoning_vc.rename({'count': 'CE_zoning'})

subdivision_vc = train['subdivision'].value_counts().rename(
    {'count': 'CE_subdivision'})

columns_to_use = [c for c in test.columns if c != 'id']
positive_cols = ['view_rainier', 'view_olympics', 'view_cascades',
                 'view_territorial', 'view_skyline', 'view_sound',
                 'view_lakewash', 'view_lakesamm', 'view_otherwater',
                 'view_other']
# 'area' seems like categorical rather than numeric
cat_cols = ['join_status', 'city', 'zoning', 'present_use', 'golf',
            'greenbelt', 'noise_traffic', 'view_rainier', 'view_olympics',
            'view_cascades', 'view_territorial', 'view_skyline', 'view_sound',
            'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other',
            'submarket', 'has_warning', 'has_renovated', 'sale_month', 'area']
cat_str_cols =  ['join_status', 'city', 'zoning', 'submarket']
drop_cols = ['sale_warning', 'sale_date', 'latitude', 'longitude', 'year_reno']

# sum of positive 'view' scores
positive_expr = pl.col('view_rainier')
for c in positive_cols[1:]:
    positive_expr += pl.col(c)

def base_encoder(input_df):
    out_df = input_df[columns_to_use].join(        
        zoning_vc, how='left', on='zoning', maintain_order='left'
    ).join(
        subdivision_vc, how='left', on='subdivision', maintain_order='left'
    ).with_columns(
        pl.when(pl.col('sqft')==0).then(np.nan).otherwise(pl.col('sqft')).alias('sqft')
    ).with_columns(
        (pl.col('sqft')+pl.col('sqft_fbsmt')).alias('total_living_sqft'),
        (pl.col('sqft_lot') / pl.col('sqft')).alias('lot_to_living_ratio'),
        (pl.col('sqft_fbsmt') / pl.col('sqft')).alias('fbsmt_to_sqft_ratio'),
        pl.col('sale_date').dt.year().alias('sale_year'),
        pl.col('sale_date').dt.month().alias('sale_month'),
        (pl.col('sale_warning') != "   ").cast(pl.Int8).alias('has_warning'),
        pl.when(pl.col('zoning').is_in(top_zoning)).then(pl.col('zoning')
                                                        ).otherwise(pl.lit('other')),
        pl.when(pl.col('year_reno')==0).then(pl.col('year_built')
                                            ).otherwise(pl.col('year_reno'),                                                                            ).alias('year_current_state'),
        (pl.col('year_reno') != 0).cast(pl.Int8).alias('has_renovated'),
        pl.col('submarket').fill_null(pl.lit('NA')),
        # sum of positive 'view' scores
        positive_expr.alias('positive_view'),
        (pl.col('land_val')+pl.col('imp_val')).alias('total_val')
    ).with_columns(
        [(pl.col('sale_year') - pl.col('year_built')).alias('age'),
        (pl.col('sale_year') - pl.col('year_current_state')
        ).alias('age_current_state')]
        + [pl.col(c).cast(pl.Categorical) for c in cat_str_cols]
        + [pl.col(c).log1p() for c in ['land_val', 'imp_val', 'total_val', 'sqft', 'sqft_fbsmt',
                                       'sqft_lot', 'sqft_1', 'total_living_sqft']]
    ).drop(drop_cols)

    return out_df

x0 = base_encoder(train)
test_x0 = base_encoder(test)


# create aggregation features
agg_df = pl.concat([x0, test_x0], how='vertical'
    ).select(
        ['subdivision', 'sale_year', 'submarket', 'area', 'imp_val', 'land_val', 'total_val',
         'grade', 'sqft', 'year_built', 'total_living_sqft'])

def aggregation(key):
    return agg_df.group_by(key
                ).agg(
                    pl.col('total_living_sqft').mean().alias('agg_'+ key + '_living_sqft'),
                    pl.col('imp_val').mean().alias('agg_'+ key + '_imp_val'),
                    pl.col('land_val').mean().alias('agg_'+ key + '_land_val'),
                    pl.col('grade').median().alias('agg_'+ key + '_grade'),
                    pl.col('sqft').mean().alias('agg_'+ key + '_sqft'),
                    pl.col('year_built').mean().alias('agg_'+ key + '_year_built'),
                    pl.col('total_val').mean().alias('agg_'+ key + '_total_val')
                ).select([key, 'agg_'+ key + '_living_sqft', 'agg_'+ key + '_imp_val',
                          'agg_'+ key + '_land_val', 'agg_'+ key + '_grade',
                          'agg_'+ key + '_sqft', 'agg_'+ key + '_year_built',
                          'agg_'+ key + '_total_val'])

subdiv_df = aggregation('subdivision')
submarket_df = aggregation('submarket')
sale_year_df = aggregation('sale_year')
area_df = aggregation('area')


x0 = x0.join(subdiv_df, how='left', on='subdivision', maintain_order='left'
    ).join(sale_year_df, how='left', on='sale_year', maintain_order='left'
    ).join(submarket_df, how='left', on='submarket', maintain_order='left'
    ).join(area_df, how='left', on='area', maintain_order='left'
          ).drop('subdivision')

test_x0 = test_x0.join(subdiv_df, how='left', on='subdivision', maintain_order='left'
    ).join(sale_year_df, how='left', on='sale_year', maintain_order='left'
    ).join(submarket_df, how='left', on='submarket', maintain_order='left'
    ).join(area_df, how='left', on='area', maintain_order='left'
          ).drop('subdivision')

feature_names = x0.columns
cat_columns = [name for name, dtype in x0.schema.items() if dtype == pl.Categorical]

# transform categorical to int32 by 'to_physical(）'
allx = pl.concat([x0, test_x0], how='vertical').with_columns(
    [pl.col(c).to_physical() for c in cat_columns]
)

# transform polars to numpy
x = allx[:len(x0)].to_numpy()
test_x = allx[len(x0):].to_numpy()
y = train['sale_price'].to_numpy()


def run_log_xgb(alpha, params, folds, stratify, seed):
    N_FOLDS = folds
    lny = np.log1p(y)
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for i, (train_idx, valid_idx) in enumerate(skf.split(train, stratify)):
        x_train, y_train = x[train_idx], lny[train_idx]
        x_valid, y_valid = x[valid_idx], lny[valid_idx]

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=alpha,
            n_estimators=10000,
            random_state=seed,
            enable_categorical=True,
            tree_method='gpu_hist',
            early_stopping_rounds=100,
            verbose=False,
            **params
        )

        model.fit(x_train, y_train,
                    eval_set=[(x_valid, y_valid)],
                    verbose=False)

        oof[valid_idx] = np.expm1(model.predict(x_valid))
        pred += model.predict(test_x) / N_FOLDS

    tot_pinball = mean_pinball_loss(y, oof, alpha=alpha)
    print(f'Pinball Loss={tot_pinball.astype(int):,}')

    return oof, np.expm1(pred)


def winkler_breakdown(y_true, lower, upper, alpha=0.1):
    '''
    Utility function to break down the Winkler Score into its components:
    interval width, lower penalty, and upper penalty.
    '''
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    penalty_lower = np.where(y_true < lower, penalty_lower, 0)
    penalty_upper = np.where(y_true > upper, penalty_upper, 0)
    score += penalty_lower
    score += penalty_upper

    return np.mean(score), np.mean(width), np.mean(penalty_lower), np.mean(penalty_upper)


alphas = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

oofs = []
preds = []
XGB_params = {'max_depth':6, 'colsample_bytree': 0.7, 'learning_rate': 0.06}

for seed, alpha in enumerate(alphas):
    print(f"{alpha=}")
    oof, pred = run_log_xgb(alpha, XGB_params, 5, train['grade'], seed)
    oofs.append(oof)
    preds.append(pred)
    print()


q_keys = ['q05', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95']
q_oof = dict(zip(q_keys, oofs))
q_pred = dict(zip(q_keys, preds))
n_q = len(q_keys)

oof_df = pl.DataFrame(q_oof)
oof_lower = np.zeros(len(train))
oof_upper = np.zeros(len(train))
pred_upper = np.zeros(len(test))
pred_lower = np.zeros(len(test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for i, (train_idx, valid_idx) in enumerate(skf.split(oof_df, train['sale_year'])):
    train_oof, y_train = oof_df[train_idx].to_dict(), y[train_idx]
    valid_oof, y_valid = oof_df[valid_idx].to_dict(), y[valid_idx]

    def compute_bounds(weights, q_preds):
        w_lower = dict(zip(q_keys, weights[:n_q]))
        w_upper = dict(zip(q_keys, weights[n_q:]))

        lower = sum(w_lower[k] * q_preds[k] for k in q_keys)
        upper = sum(w_upper[k] * q_preds[k] for k in q_keys)

        return lower, upper

    def objective(weights):
        # Clip weights for numerical stability
        weights = np.clip(weights, 0, 1)
        lower, upper = compute_bounds(weights, train_oof)
        winkler, _, _, _ = winkler_breakdown(y_train, lower, upper, alpha=0.1)
        return winkler

    init_weights = np.ones(n_q * 2) / n_q
    bounds = [(0.0, 1.0)] * (n_q * 2)

    res = minimize(objective, init_weights, method='L-BFGS-B', bounds=bounds)

    opt_weights = np.clip(res.x, 0, 1)

    # weight of lower estimates and upper estimates
    w_lower = dict(zip(q_keys, opt_weights[:n_q]))
    w_upper = dict(zip(q_keys, opt_weights[n_q:]))

    # train data
    oof_lower[valid_idx] = sum(w_lower[k] * valid_oof[k] for k in q_keys)
    oof_upper[valid_idx] = sum(w_upper[k] * valid_oof[k] for k in q_keys)

    winkler, _, _, _ = winkler_breakdown(
        y_valid, oof_lower[valid_idx], oof_upper[valid_idx], alpha=0.1)

    # test data 
    pred_upper += sum(w_upper[k] * q_pred[k] for k in q_keys) / 5
    pred_lower += sum(w_lower[k] * q_pred[k] for k in q_keys) / 5

    print(f"Fold_{i}: Winkler = {winkler.astype(int):,}")

winkler, width, penalty_lower, penalty_upper = winkler_breakdown(
    y, oof_lower, oof_upper, alpha=0.1)

print(f"Winkler={winkler.astype(int):,}")
print(f"\twidth={width.astype(int):,}")
print(f"\tpenalty_lower={penalty_lower.astype(int):,}, penalty_upper={penalty_upper.astype(int):,}")


# clip estimates
pred_lower = np.clip(pred_lower, MIN_PRICE, MAX_PRICE)
pred_upper = np.clip(pred_upper, MIN_PRICE, MAX_PRICE)

pl.DataFrame({'id': test['id'],
              'pi_lower': pred_lower,
              'pi_upper': pred_upper}).write_csv('submission.csv')


# alpha = [0.1, 0.9]
winkler, width, penalty_lower, penalty_upper = winkler_breakdown(y, q_oof['q10'], q_oof['q90'], alpha=0.1)
print(f"Winkler={winkler.astype(int):,}")
print(f"\twidth={width.astype(int):,}")
print(f"\tpenalty_lower={penalty_lower.astype(int):,}, penalty_upper={penalty_upper.astype(int):,}")


# alpha = [0.05, 0.95]
winkler, width, penalty_lower, penalty_upper = winkler_breakdown(y, q_oof['q05'], q_oof['q95'], alpha=0.1)
print(f"Winkler={winkler.astype(int):,}")
print(f"\twidth={width.astype(int):,}")
print(f"\tpenalty_lower={penalty_lower.astype(int):,}, penalty_upper={penalty_upper.astype(int):,}")

