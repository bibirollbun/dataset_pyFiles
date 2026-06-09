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
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import statsmodels.api as sm

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
from sklearn.linear_model import Lasso
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.api import SimpleExpSmoothing

import lightgbm as lgb


# ignoring pandas performance warnings for now.
import warnings
warnings.filterwarnings('ignore')

import time

def record_time(text, fn):
    # start_time = time.time()
    ret = fn()
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"{text}, Elapsed time: {elapsed_time} seconds")
    return ret


df = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')
df_test = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv')


df['date'] = pd.to_datetime(df['date'])
df_test['date'] = pd.to_datetime(df_test['date'])

df['store'] = df['store'].astype('category')
df['item'] = df['item'].astype('category')
df_test['store'] = df_test['store'].astype('category')
df_test['item'] = df_test['item'].astype('category')


def encode_period(df_t, column, period):
        df_t[f'{column}_sin'] = np.sin(df_t[column] * 2 * np.pi / period)
        df_t[f'{column}_cos'] = np.cos(df_t[column] * 2 * np.pi / period)

def create_datetime_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['days_in_month'] = df['date'].dt.days_in_month
    df['day_of_week'] = df['date'].dt.day_of_week
    df['is_weekend'] = df['day_of_week'] >= 5

    encode_period(df, 'month', 12)
    encode_period(df, 'day', df['days_in_month'])
    encode_period(df, 'day_of_week', 7)

    return df.copy()


df = create_datetime_features(df)
df_test = create_datetime_features(df_test)


test_limit_dates = [
    '2015-07-01',
    '2015-10-01',
    '2016-01-01',
    '2016-04-01',
    '2016-07-01',
    '2016-10-01',
    '2017-01-01',
    '2017-04-01',
    '2017-07-01',
    '2017-10-01',
    '2018-01-01'
]


def create_lagged_features(df, df_w_grp):

    df_copy = pd.DataFrame(index=df.index)

    for i in [1,2,3,4,5,6]:
        df_copy[f"sales_lag{i}"] = df_w_grp.shift(i)
    
    for w in [1,2,3,4,5,6,7,8]:
        i = w * 7
        df_copy[f"sales_lag{i}"] = df_w_grp.shift(i)
    
    return df_copy

def create_rolling_features(df, df_w_grp):

    df_copy = pd.DataFrame(index=df.index)

    for w in range(7, 500, 7):
        i = w
        df_copy.loc[:, f'sales_rolling_mean{i}'] = df_w_grp.transform(lambda x : x.shift(1).rolling(window=i).mean())

    # df = df.copy()

    for w in range(7, 500, 7):
        i = w
        df_copy.loc[:, f'sales_rolling_stdev{i}'] = df_w_grp.transform(lambda x : x.shift(1).rolling(window=i).std())

    # df = df.copy()

    for w in range(7, 500, 7):
        i = w
        df_copy.loc[:, f'sales_rolling_mean_diff{7}-{i}'] = df_copy[f'sales_rolling_mean7'] - df_copy[f'sales_rolling_mean{i}']
    
    return df_copy

def create_features(df):
    
    df_w = df[['store', 'item', 'sales']]
    group_col = ['store', 'item']

    df_w_grp = df_w.groupby(group_col)

    # groupby can be reused as long as underlying dataframe is unchanged and only read-only operations are executed. 
    # Who would've thought!
    df_lagged = create_lagged_features(df, df_w_grp)
    df_rolled = create_rolling_features(df, df_w_grp)

    return pd.concat([df, df_lagged, df_rolled], axis=1)





df = create_features(df)


df_no_na = df.dropna()


def smape(forecast, actual):
    return 100/len(forecast) * np.sum(2*np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))

def smape_one(forecast, actual):
    return np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))

def smape_loss(y, data):
    t = data.get_label()

    abs_sum = np.abs(t) + np.abs(y)

    grad = 200 * (-np.sign(t - y) * abs_sum - np.abs(t - y) * np.sign(y)) / (abs_sum) ** 2
    hess = 400 * (np.sign(t - y) * np.sign(y) * abs_sum + np.abs(t - y)) / abs_sum ** 3

    return grad, hess

def smape_eval(y, data):
    return 'smape', smape(y, data.get_label()), False


def train(X, y, train_cols, test_limit_dates = test_limit_dates, objective='regression', params = {}):
    
    models = []
    cv_scores = []
    train_cv_scores = []

    X_vals = []
    evals_list = []

    print(f"process {test_limit_dates}")

    param = {
        'objective': objective,
        'metric': 'None',
        **params
    }

    fold = 1
    fold_len = len(test_limit_dates)
    while fold < fold_len:
        
        test_date_begin = test_limit_dates[fold-1]
        test_date_end = test_limit_dates[fold]

        print(f"process test period {test_date_begin} - {test_date_end}")
        
        train_idx = np.where(X['date'] < test_date_begin)
        val_idx = np.where((X['date'] >= test_date_begin) & (X['date'] < test_date_end))

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        lgb_train = lgb.Dataset(X_train[train_cols], y_train)
        lgb_val = lgb.Dataset(X_val[train_cols], y_val, reference=lgb_train)

        evals={}
        model = lgb.train(param, 
                          lgb_train, 
                          valid_sets=[lgb_train, lgb_val], 
                          feval=smape_eval, 
                          callbacks = [lgb.record_evaluation(evals)])

        y_train_pred = model.predict(X_train[train_cols])
        y_pred = model.predict(X_val[train_cols])

        train_mse = smape(y_train, y_train_pred)
        train_cv_scores.append(train_mse)

        mse = smape(y_val, y_pred)
        cv_scores.append(mse)

        models.append(model)
        evals_list.append(evals)
        
        X_val_t = X_val[['date', 'store', 'item', 'sales']].copy()
        X_val_t['sales_pred'] = y_pred

        X_vals.append(X_val_t)

        print(f"Fold {fold}, val date: {test_date_begin}: SMAPE TRAIN = {train_mse:.4f}, TEST = {mse:.4f}\n")

        fold += 1
        
    print(f"Average CV SMAPE TRAIN: {np.mean(train_cv_scores):.4f}, TEST: {np.mean(cv_scores):.4f}\n")
    print(f"STDEV CV SMAPE TRAIN: {np.std(train_cv_scores):.4f}, TEST: {np.std(cv_scores):.4f}\n")

    X_vals = pd.concat(X_vals, axis=0).reset_index(drop=True)

    return cv_scores, models, X_vals, evals_list

    


def train_eps(df, test_limit_dates, objective='regression', params = {}):
    train_cols = set(df.columns.tolist())
    train_cols.remove('date')
    train_cols.remove('year')
    target_col = 'sales'
    train_cols.remove(target_col)
    train_cols = list(train_cols)

    X = df
    y = df[target_col]

    return train(X, y, train_cols, test_limit_dates=test_limit_dates, objective=objective, params=params)


def fill_rolling_features(df : pd.DataFrame, X_test : pd.DataFrame) -> pd.DataFrame:

    X_test = X_test.set_index(['store', 'item'])

    # grouped by objects can be reused.
    grouped_by = df[['store', 'item', 'sales']].groupby(['store', 'item'])

    # combine both mean and std calculation and reusing group by makes this run much faster.
    for w in range(7, 500, 7):
        i = w

        df_stores_items_grouped_core = grouped_by.tail(i).groupby(['store', 'item'])
        mean_grp = df_stores_items_grouped_core.mean()
        std_grp = df_stores_items_grouped_core.std()

        X_test = X_test.join(mean_grp)
        X_test = X_test.drop(columns=[f'sales_rolling_mean{i}'])
        X_test = X_test.rename(columns={"sales" : f'sales_rolling_mean{i}'})

        X_test = X_test.join(std_grp)
        X_test = X_test.drop(columns=[f'sales_rolling_stdev{i}'])
        X_test = X_test.rename(columns={"sales" : f'sales_rolling_stdev{i}'})

    for w in range(7, 500, 7):
        i = w
        X_test[f'sales_rolling_mean_diff{7}-{i}'] = X_test[f'sales_rolling_mean7'] - X_test[f'sales_rolling_mean{i}']

    return X_test.reset_index()



def fill_lagged_features(df, X_test) -> pd.DataFrame :

    X_test = X_test.set_index(['store', 'item'])
    grouped_by = df[['store', 'item', 'sales']].groupby(['store', 'item'])

    # setting manually search by store, item is slow ~2.4s
    # this one works the best. ~0.8sec
    for i in [1,2,3,4,5,6]:
        df_stores_items_grouped = grouped_by.nth(-i).set_index(['store', 'item'])

        X_test = X_test.join(df_stores_items_grouped)
        X_test = X_test.drop(columns=[f"sales_lag{i}"])
        X_test = X_test.rename(columns={"sales" : f"sales_lag{i}"})

    for w in [1,2,3,4,5,6,7,8]:
        
        i = w * 7
        df_stores_items_grouped = grouped_by.nth(-i).set_index(['store', 'item'])

        X_test = X_test.join(df_stores_items_grouped)
        X_test = X_test.drop(columns=[f"sales_lag{i}"])
        X_test = X_test.rename(columns={"sales" : f"sales_lag{i}"})
    
    return X_test.reset_index()
            



def train_entire(df):
    df = df.copy()

    train_cols = set(df.columns.tolist())
    train_cols.remove('date')
    train_cols.remove('year')
    target_col = 'sales'
    train_cols.remove(target_col)
    train_cols = list(train_cols)

    X = df[train_cols]
    y = df[target_col]

    # tweedie is the best objective after some test.
    param = {
        'objective': 'tweedie',
        'metric': 'None',
        'num_iterations': 380
    }

    lgb_train = lgb.Dataset(X, y)

    model = lgb.train(param, lgb_train, feval=smape_eval)

    return model


def daterange(start_date, end_date):
    """
    Generates a sequence of dates from start_date to end_date (inclusive).
    """
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + datetime.timedelta(n)


def predict(model, df, df_test, max_iter = None):

    df = df.copy()

    train_cols = set(df.columns.tolist())
    train_cols.remove('date')
    train_cols.remove('year')
    target_col = 'sales'
    train_cols.remove(target_col)

    start_test_date = df_test['date'].min()
    end_test_date = df_test['date'].max()

    test_cols_from_date = ['date', 'store', 'item', 'year', 'month', 'day', 'day_of_week',
       'is_weekend', 'days_in_month', 'month_sin', 'month_cos', 'day_sin',
       'day_cos', 'day_of_week_sin', 'day_of_week_cos']
    test_cols_without_date_cols = train_cols.copy()
    for tc in test_cols_from_date:
        if tc in test_cols_without_date_cols:
            test_cols_without_date_cols.remove(tc)

    df_test[list(test_cols_without_date_cols)] = np.nan

    train_cols = list(train_cols)

    df['future'] = False

    i = 0

    for dt in daterange(start_test_date, end_test_date):
        
        i += 1

        print(f"processing dt={dt}")

        # append the row to df
        X_test = df_test[df_test['date'] == dt][train_cols].copy()

        X_test = record_time("lagged", lambda : fill_lagged_features(df, X_test))

        X_test = record_time("rolling", lambda : fill_rolling_features(df, X_test))

        y_pred = record_time("pred", lambda : model.predict(X_test[train_cols]))

        X_test['sales'] = y_pred
        X_test['date'] = dt
        X_test['year'] = dt.year
        X_test['future'] = True

        df = record_time("concat", lambda : pd.concat([df, X_test], axis=0).reset_index(drop=True))

        if max_iter is not None and i >= max_iter:
            return df, start_test_date


    return df, start_test_date



model = train_entire(df_no_na)


df_pred_1, std = predict(model, df_no_na, df_test, max_iter=None)


df_pred_simpl = df_pred_1[df_pred_1['future'] == True][['date', 'store', 'item', 'sales']]
df_pred_simpl['sales_rounded'] = np.round(df_pred_simpl['sales'])
df_pred_simpl


df_test_cp = df_test.copy()
df_test_cp_sales = df_test_cp.merge(right=df_pred_simpl, on=['date', 'store', 'item'], how='left')


df_test_cp_sales[['id', 'sales_rounded']].to_csv("submission.csv", header=['id', 'sales'], index=False)

