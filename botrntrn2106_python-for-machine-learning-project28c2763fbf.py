!pip install -q xgboost lightgbm calplot category_encoders


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import os
import datetime as dt
import calplot
from copy import deepcopy
from category_encoders import TargetEncoder, OneHotEncoder

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, RepeatedKFold, GridSearchCV, train_test_split, cross_val_score, RandomizedSearchCV 
from sklearn.metrics import mean_absolute_error, PredictionErrorDisplay
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler ,SplineTransformer, PolynomialFeatures, FunctionTransformer, LabelEncoder
from sklearn.base import clone, BaseEstimator, TransformerMixin

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


train_path = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv'
test_path = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv'
inventory_path = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv'
calendar_path = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv'
test_weights_path = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv'


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
inventory = pd.read_csv(inventory_path)
calendar = pd.read_csv(calendar_path)
weights = pd.read_csv(test_weights_path)


def merge_calen_and_inv_data(df):
    merged = df.merge(calendar, on=['date', 'warehouse'], how='left')
    merged =  merged.merge(inventory, on=['unique_id', 'warehouse'], how='left')
    return merged


train = train.dropna()
full_train_data = merge_calen_and_inv_data(train)
test = merge_calen_and_inv_data(test)


full_train_data = pd.concat([full_train_data, test])


full_train_data['log_orders'] = np.log1p(full_train_data['total_orders'])
full_train_data['sell_price_main_log'] = np.log1p(full_train_data['sell_price_main'])

# cat nguong
upper_clip = full_train_data['sell_price_main_log'].quantile(0.99)
full_train_data['sell_price_main_log'] = full_train_data['sell_price_main_log'].clip(upper=upper_clip)


# du lieu bat thuong
def fe_other(df):
    discount_cols = ['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount']
    df[discount_cols] = df[discount_cols].clip(0)
    return df


full_train_data = fe_other(full_train_data)


closed_days = calendar[calendar['shops_closed'] == 1]
closed_days.to_csv('closed_days.csv', index=False)


full_train_data['date'] = pd.to_datetime(full_train_data['date'])
full_train_data['day_of_week'] = full_train_data['date'].dt.dayofweek
full_train_data['day_of_year'] = full_train_data['date'].dt.day_of_year
full_train_data['week_of_year'] = full_train_data['date'].dt.isocalendar().week
full_train_data['month'] = full_train_data['date'].dt.month
full_train_data['day'] = full_train_data['date'].dt.day
full_train_data['month'] = full_train_data['date'].dt.month
full_train_data['year'] = full_train_data['date'].dt.year
full_train_data['cos_day'] = np.cos(full_train_data['day_of_year']*2*np.pi/365)
full_train_data['sin_day'] = np.sin(full_train_data['day_of_year']*2*np.pi/365)
full_train_data['is_weekend'] = full_train_data['day_of_week'].isin([5,6]).astype('int8')
full_train_data['long_weekend'] = ((full_train_data['shops_closed'] == 1) & (full_train_data['shops_closed'].shift(1) == 1)).astype(int)


def get_season(month):
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'        


full_train_data['season'] = full_train_data['month'].apply(get_season)
full_train_data['max_discount'] = full_train_data[['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount']].max(axis=1)
full_train_data['common_name'] = full_train_data['name'].apply(lambda x: x[:x.find('_')])

cols = full_train_data.columns.tolist()
cols.remove('common_name')
target_col_index = cols.index('name')
cols.insert(target_col_index + 1, 'common_name')
full_train_data = full_train_data[cols]

full_train_data['total_products'] = full_train_data.groupby(['date','warehouse','common_name'])['unique_id'].transform('nunique')
full_train_data['discount_avg'] = full_train_data.groupby(['date','warehouse','common_name'])['max_discount'].transform('mean')


def load_calendar(calendar):
    calendar = calendar.sort_values('date')
    calendar.reset_index(drop=True, inplace=True)

    calendar.loc[calendar['holiday_name'].isna(), 'holiday'] = 0

    calendar['last_holiday_date'] = calendar['date']
    calendar['next_holiday_date'] = calendar['date']

    calendar.loc[calendar['holiday'] == 0, ['last_holiday_date','next_holiday_date']] = np.nan

    calendar['last_holiday_date'] = calendar.sort_values('date').groupby('warehouse')['last_holiday_date'].ffill()
    calendar['next_holiday_date'] = calendar.sort_values('date').groupby('warehouse')['next_holiday_date'].bfill()

    calendar['days_since_last_holiday'] = ((calendar['date'] - calendar['last_holiday_date']).dt.days)
    calendar['days_to_next_holiday'] = ((calendar['next_holiday_date'] - calendar['date']).dt.days)

    calendar['day_before_holiday'] = calendar['days_to_next_holiday'] == 1
    calendar['day_after_holiday'] = calendar['days_since_last_holiday'] == 1

    calendar.drop(['last_holiday_date','next_holiday_date'],axis=1,inplace=True)

    calendar.drop(['shops_closed','winter_school_holidays','school_holidays','holiday_name'],axis=1,inplace=True)
    return calendar


full_train_data = load_calendar(full_train_data)


PERIODS = [14, 16, 18, 21, 30, 60, 90, 120, 180]


def add_lagged_product_sales(df):
    df = df.sort_values(['warehouse', 'name', 'date'])
    for shift in PERIODS:
        df[f'product_sales_{shift}']=df.groupby(['warehouse','name'])['sales'].shift(periods=shift)
    return df


full_train_data = add_lagged_product_sales(full_train_data)


full_train_data['days_since_2020'] = (full_train_data['date'] - pd.to_datetime('2020-01-01')).dt.days.astype('int')

mean_prices = full_train_data.groupby(full_train_data['unique_id'])['sell_price_main'].mean()
std_prices = full_train_data.groupby(full_train_data['unique_id'])['sell_price_main'].std()
full_train_data['price_scaled'] = np.where(
    full_train_data['unique_id'].map(std_prices) == 0, 0,
    (full_train_data['sell_price_main'] - full_train_data['unique_id'].map(mean_prices))/full_train_data['unique_id'].map(std_prices))
full_train_data['price_detrended'] = full_train_data['price_scaled'] - full_train_data.groupby(['days_since_2020','warehouse'])['price_scaled'].transform('mean')
full_train_data.drop('price_scaled',axis=1,inplace=True)


le = LabelEncoder()

label_cols = ['name', 'common_name','warehouse', 'L1_category_name_en',
              'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en','season']

for col in label_cols:
    full_train_data[col] = le.fit_transform(full_train_data[col])


def drop_discount_features(df):
    discount_cols = [col for col in df.columns if col.startswith('type_') and col.endswith('_discount')]
    df = df.drop(columns=discount_cols)
    df = df.drop(columns=['total_orders', 'sell_price_main'])
    return df


full_train_data = drop_discount_features(full_train_data)


train = full_train_data.loc[full_train_data['date'] < '2024-06-03']
test = full_train_data.loc[full_train_data['date'] >= '2024-06-03'].drop(['sales','availability'],axis=1)

train['id'] = train['unique_id'].astype('str') + '_' + train['date'].astype('str')
train.set_index('id',inplace=True)
test['id'] = test['unique_id'].astype('str') + '_' + test['date'].astype('str')
test.set_index('id',inplace=True)


train.head(5)


# thử bỏ mấy ngày trước 2022
train = train[train['date'] >= '2022-01-01']


y_train = train['sales']
train_availability = train['availability']
X_train = train.drop(columns=['sales', 'date', 'availability'])
X_train_weights = train['unique_id'].map(weights['weight'])


cat_cols = ['unique_id'] + list(X_train.columns[X_train.dtypes == 'object'])
all_data = pd.concat([X_train, test])
add_cols = ['last_sales_ema005','CN_sales_sum','last_sales_zs']


train_cp = train.groupby('unique_id')['date'].apply(lambda s: pd.date_range(s.min(), test.date.max())).explode().reset_index()
train_cp = train_cp.merge(
    pd.concat([train[['unique_id','date','sales','warehouse',]],
               test[['unique_id','date','warehouse']]]),
    on=['unique_id','date'],how='left')
train_cp = train_cp.merge(inventory, left_on='unique_id', right_index=True)
train_cp['common_name'] = train_cp['name'].apply(lambda x: x[:x.find('_')])
train_cp.sort_values('date',inplace=True)
train_cp['last_sales_ema005'] = train_cp.groupby(['unique_id'])['sales'].transform(lambda x: x.shift(1).ewm(alpha=.005).mean()).fillna(0)

train_cp = train_cp.drop(columns='warehouse_y')
train_cp.rename(columns={'warehouse_x': 'warehouse'}, inplace=True)

train_cp['CN_sales_sum'] = train_cp.groupby(['common_name','warehouse','date'])['last_sales_ema005'].transform('sum')


all_data = all_data.merge(train_cp.set_index(['unique_id','date'])[[
    'last_sales_ema005','CN_sales_sum'
]], left_on=['unique_id','date'],right_index=True,how='left')
sales_stats = train_cp.groupby(['common_name','warehouse'])['sales'].agg(['mean','std'])
all_data['last_sales_zs'] = (all_data['last_sales_ema005'] - pd.MultiIndex.from_frame(all_data[['common_name','warehouse']]).map(
    sales_stats['mean']))/ pd.MultiIndex.from_frame(all_data[['common_name','warehouse']]).map(sales_stats['std'])


X_train[add_cols] = all_data[add_cols]
test[add_cols] = all_data[add_cols]
all_data[cat_cols] = all_data[cat_cols].astype('str').astype('category')


drop_cols = ['name','L1_category_name_en', 'days_since_2020','product_sales_180']


X,y = deepcopy(X_train),deepcopy(y_train)
X[cat_cols] = all_data[cat_cols]
X.drop(drop_cols,axis=1,inplace=True)
test_copy = deepcopy(test)
test_copy[cat_cols] = all_data[cat_cols]
test_copy.drop(drop_cols,axis=1,inplace=True)
test_copy = test_copy.drop(columns='date')


X_train.sample(5)


# kt
test_copy.sample(10)


lr = 0.05
es = 10
n_est = 10000
seed = 42


xgb_model = XGBRegressor(enable_categorical=True, eval_metric='rmse')


param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 1],
    'reg_lambda': [0, 1],
    'min_child_weight': [1, 3, 5]
}


search_xgb = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions = param_grid,
    n_iter=50,
    cv=2,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42
)


search_xgb.fit(X,y)


print("Best parameters:", search_xgb.best_params_)


# base_params = {
#     'n_estimators':n_est
#     ,'learning_rate':lr
#     ,'verbosity':0
#     ,'enable_categorical':True
#     ,'early_stopping_rounds':es
#     ,'random_state':seed
#     ,'objective':'reg:squarederror'
#     ,'eval_metric':'rmse'
#     ,'device':'cuda'
#     ,'reg_lambda':0
#     , 'reg_alpha' : 1
#     ,'min_child_weight':3
#     ,'subsample': 0.8
#     ,'colsample_bytree': 0.6
#     ,'max_depth': 10
# }
# kf_params = {
#     'n_splits':5
#     ,'n_repeats':1
#     ,'random_state':seed
# }


# kf = RepeatedKFold(**kf_params)
# test_preds_xgb = []
# pow_trans = True
# pow_degree = .5


# # use Kfold to train model xgb
# for i, (idx_t, idx_v) in enumerate(kf.split(X)):
#     X_t, X_v = X.iloc[idx_t], X.iloc[idx_v]
#     y_t, y_v = y.loc[X_t.index], y.loc[X_v.index]

#     if pow_trans:
#         y_t, y_v = np.power(y_t, pow_degree), np.power(y_v, pow_degree)

#     xgb_model = XGBRegressor(**base_params)
#     xgb_model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=100*es)
#     preds = np.power(xgb_model.predict(test_copy).clip(0), 1/pow_degree) if pow_trans else xgb_model.predict(test_copy).clip(0)
#     test_preds_xgb.append(preds)


param_grid_xgb = {
    'max_depth': Integer(4, 8),
    'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
    'n_estimators': Integer(100, 300),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'reg_alpha': Real(0, 1),
    'reg_lambda': Real(0, 1),
    'min_child_weight': Integer(1, 5)
}


search_xgb = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=param_grid_xgb,
    n_iter=50,
    cv=2,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42
)


search_xgb.fit(X, y)


print("Best parameters for XGBoost:", search_xgb.best_params_)


lgb_params = {
    'objective'        : 'l2',
    'verbosity'        : -1,
    'n_iter'           : 800,
    'lambda_l1'        : 0.8942112689465215, 
    'lambda_l2'        : 6.4122663335284305, 
    'cat_l2'           : 7.696733748814076, 
    'learning_rate'    : 0.054754587005265434, 
    'max_depth'        : 11, 
    'num_leaves'       : 273, 
    'colsample_bytree' : 0.5488343345823052, 
    'colsample_bynode' : 0.7200134511260632, 
    'min_data_in_leaf' : 6, 
    'max_cat_threshold': 952,
}


lgb_model = LGBMRegressor(**lgb_params, random_state=seed)


param_grid_lgb = {
    'max_depth': [7, 9, 11, 13],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300, 400],
    'num_leaves': [31, 127, 255, 511],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'subsample': [0.6, 0.8, 1.0],
    'lambda_l1': [0, 0.1, 1.0],
    'lambda_l2': [0, 1.0, 10.0],
    'min_data_in_leaf': [5, 10, 20]
}


search_lgb = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid_lgb,
    n_iter=50,
    cv=2,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42
)


search_lgb.fit(X, y)


print("Best parameters for LightGBM:", search_lgb.best_params_)


param_grid_lgb = {
    'max_depth': Integer(7, 13),
    'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
    'n_estimators': Integer(100, 400),
    'num_leaves': Integer(31, 511),
    'colsample_bytree': Real(0.6, 1.0),
    'subsample': Real(0.6, 1.0),
    'lambda_l1': Real(0, 1.0),
    'lambda_l2': Real(0, 10.0),
    'min_data_in_leaf': Integer(5, 20)
}


search_lgb = BayesSearchCV(
    estimator=lgb_model,
    search_spaces=param_grid_lgb,
    n_iter=50,
    cv=2,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42
)


search_lgb.fit(X, y)


print("Best parameters for LightGBM:", search_lgb.best_params_)


cb_params = {
    'grow_policy'        : 'Lossguide',
    # 'task_type'          : 'GPU',
    'iterations'         : 800,
    'bagging_temperature': 0.5,
    'learning_rate'      : 0.1,
    'max_leaves'         : 128,
    'max_depth'          : 12,
    'l2_leaf_reg'        : 1.25,
    'min_data_in_leaf'   : 24,
    'verbose'            : 0,
    'border_count'       : 256,
    'cat_features'       : cat_cols,
}


cb_model = CatBoostRegressor(**cb_params, random_state=seed)


param_grid_cb = {
    'max_depth': [6, 8, 10, 12],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200, 300, 400],
    'l2_leaf_reg': [1, 3, 5, 7],
    'bagging_temperature': [0.0, 0.5, 1.0],
    'min_data_in_leaf': [5, 10, 20],
    'border_count': [32, 64, 128, 256]
}


search_cb = RandomizedSearchCV(
    estimator=cb_model,
    param_distributions=param_grid_cb,
    n_iter=50,
    cv=2,
    scoring='neg_root_mean_squared_error',
    n_jobs=1,
    random_state=42
)


# try:
#     torch.cuda.empty_cache()
# except:
#     pass


search_cb.fit(X, y)
print("Best parameters for CatBoost:", search_cb.best_params_)


cb_params = {
    'grow_policy'        : 'Lossguide',
    'task_type'          : 'GPU',
    'iterations'         : 800,
    'bagging_temperature': 0.5,
    'learning_rate'      : 0.1,
    'max_leaves'         : 128,
    'max_depth'          : 12,
    'l2_leaf_reg'        : 1.25,
    'min_data_in_leaf'   : 24,
    'verbose'            : 0,
    'border_count'       : 256,
    'cat_features'       : cat_cols,
}


cb_model = CatBoostRegressor(**cb_params, random_state=seed)


param_grid_cb = {
    'max_depth': Integer(6, 12),
    'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
    'iterations': Integer(100, 400),
    'l2_leaf_reg': Real(1, 7),
    'bagging_temperature': Real(0.0, 1.0),
    'min_data_in_leaf': Integer(5, 20),
    'border_count': Integer(32, 256)
}


search_cb = BayesSearchCV(
    estimator=cb_model,
    search_spaces=param_grid_cb,
    n_iter=50,
    cv=2,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42
)


search_cb.fit(X, y)


print("Best parameters for CatBoost:", search_cb.best_params_)


# test_copy['predicted_sales'] = test_preds_xgb[(len(test_preds_xgb)-1)]


# solution = test_copy.reset_index()[['id', 'predicted_sales']].copy()

# solution.columns = ['id', 'sales_hat']
    
# solution.to_csv('submission.csv', index=False)


# print(test_copy.columns)



# print(solution.tail(20))

