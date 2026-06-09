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


df = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')
df_test = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv')


df


df['date'] = pd.to_datetime(df['date'])
df_test['date'] = pd.to_datetime(df_test['date'])

df['store'].astype('category')
df['item'].astype('category')
df_test['store'].astype('category')
df_test['item'].astype('category')


df['store'].unique(), df['item'].unique(), df_test['store'].unique(), df_test['item'].unique()


def plot_sales(df, store_id, item_id, gap=100):
    df_1_1 = df[(df['store'] == store_id) & (df['item'] == item_id)]
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    sns.lineplot(data=df_1_1, x='date', y='sales', ax=ax)

    plt.xlabel("date")
    plt.ylabel("sales")
    plt.xticks(rotation="vertical")
    plt.xticks(df_1_1['date'][::gap])    # set here, ticks at step of 50
    plt.show()


plot_sales(df, 6, 31)


def plot_seasonal(df, store_id, item_id, period, model='additive'):
    df_1_1 = df[(df['store'] == store_id) & (df['item'] == item_id)]

    decomp = seasonal_decompose(df_1_1['sales'], model=model, period=period)
    fig = plt.figure()
    fig = decomp.plot()
    fig.set_size_inches(15,8)


plot_seasonal(df, 5, 24, 365)


plot_seasonal(df, 5, 24, 30)


plot_seasonal(df, 5, 24, 7)


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




create_datetime_features(df)
create_datetime_features(df_test)


df


df_sales_monthly = df[['sales', 'month']].groupby('month').mean()
df_sales_monthly.plot(kind='bar')


df_sales_day = df[['sales', 'day']].groupby('day').mean()
df_sales_day.plot(kind='bar')


df_sales_day_of_week = df[['sales', 'day_of_week']].groupby('day_of_week').mean()
df_sales_day_of_week.plot(kind='bar')


dfy = df[['sales', 'store']].groupby('store').mean()
dfy.plot(kind='bar')


dfy = df[['sales', 'item']].groupby('item').mean()
dfy.plot(kind='bar')


df['sales'].plot(kind='kde')


def plot_acf_pacf_series(series, lags):
    fig, ax = plt.subplots(ncols=2, figsize=[15, 5])
    plot_acf(series, lags=lags, ax=ax[0])
    ax[0].set_title("Autocorrelation of Sales")
    ax[0].set_xlabel("Lag")
    ax[0].set_ylabel("Autocorrelation")

    plot_pacf(series, lags=lags, ax=ax[1], method="ywmle")
    ax[1].set_title("Partial autocorrelation of Sales")
    ax[1].set_xlabel("Lag")
    ax[1].set_ylabel("Partial autocorrelation")

def plot_acf_pacf(df, store_id, item_id, lags):
    df_1_1 = df[(df['store'] == store_id) & (df['item'] == item_id)]

    plot_acf_pacf_series(df_1_1['sales'], lags)
    


plot_acf_pacf(df, 1, 1, 900)


plot_acf_pacf(df, 1, 1, 365)


plot_acf_pacf(df, 1, 1, 30)


df_5_24 = df[(df['store'] == 5) & (df['item'] == 24)]
fig, ax = plt.subplots(3, 2, figsize=(15,13))
df_5_24['sales'].plot(ax=ax[0][0], title='original')
df_5_24['sales'].diff().dropna().plot(ax=ax[0][1], title='daily')
df_5_24['sales'].diff(7).dropna().plot(ax=ax[1][0], title='weekly')
df_5_24['sales'].diff(30).dropna().plot(ax=ax[1][1], title='monthly')
df_5_24['sales'].diff(365).dropna().plot(ax=ax[2][0], title='yearly')
df_5_24['sales'].diff(153).dropna().plot(ax=ax[2][1], title='random')


plot_acf_pacf_series(df_5_24['sales'].diff().dropna(), 900)


plot_acf_pacf_series(df_5_24['sales'].diff().dropna(), 365)


plot_acf_pacf_series(df_5_24['sales'].diff().dropna(), 30)


plot_acf_pacf_series(df_5_24['sales'].diff().dropna(), 180)


def create_lagged_features(df):
    df_w = df[['store', 'item', 'sales']]
    group_col = ['store', 'item']

    for i in [1,2,3,4,5,6]:
        df[f"sales_lag{i}"] = df_w.groupby(group_col).shift(i)
    
    for w in [1,2,3,4,5,6,7,8]:
        i = w * 7
        df[f"sales_lag{i}"] = df_w.groupby(group_col).shift(i)
    
    return df.copy()

def create_rolling_features(df):
    df_w = df[['store', 'item', 'sales']]
    group_col = ['store', 'item']

    for w in range(7, 500, 7):
        i = w
        df.loc[:, f'sales_rolling_mean{i}'] = df_w.groupby(group_col).transform(lambda x : x.shift(1).rolling(window=i).mean())

    # df = df.copy()

    for w in range(7, 500, 7):
        i = w
        df.loc[:, f'sales_rolling_mean_diff{7}-{i}'] = df[f'sales_rolling_mean7'] - df[f'sales_rolling_mean{i}']

    # df = df.copy()

    for w in range(7, 500, 7):
        i = w
        df.loc[:, f'sales_rolling_stdev{i}'] = df_w.groupby(group_col).transform(lambda x : x.shift(1).rolling(window=i).std())
    
    return df.copy()


df_1 = create_lagged_features(df)
df = df_1
df_1 = create_rolling_features(df)
df = df_1
df


df


df_no_na = df.dropna()


df_no_na.columns


def smape(forecast, actual):
    return 100/len(forecast) * np.sum(2*np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))

def smape_eval(y, data):
    return 'smape', smape(y, data.get_label()), False


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


params = {
    'num_iterations': 1000
}
cv_scores_tweedie, models_tweedie, X_valid_tweedie, evals_list_tweedie = train_eps(
    df_no_na, test_limit_dates=test_limit_dates, objective='tweedie', params=params)


fig, ax = plt.subplots(1, 1, figsize=(15,5))
lgb.plot_metric(evals_list_tweedie[6], ax=ax)
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(15,5))
lgb.plot_metric(evals_list_tweedie[6], ax=ax)
ax.set_ylim(top=13.6, bottom=13.4)
plt.show()


# these are very slow. Probably because every iteration, pandas need to recalculate everything.
# after some optimization, it is completed in 14s. Entire prediction can be completed in ~1400 seconds.
def fill_rolling_features(df : pd.DataFrame, X_test : pd.DataFrame) -> pd.DataFrame:

    X_test = X_test.set_index(['store', 'item'])

    for w in range(7, 500, 7):
        i = w
        df_stores_items_grouped = df[['store', 'item', 'sales']].groupby(['store', 'item']).tail(i)
        df_stores_items_grouped = df_stores_items_grouped.groupby(['store', 'item']).mean()

        X_test = X_test.join(df_stores_items_grouped)
        X_test = X_test.drop(columns=[f'sales_rolling_mean{i}'])
        X_test = X_test.rename(columns={"sales" : f'sales_rolling_mean{i}'})

    for w in range(7, 500, 7):
        i = w

        df_stores_items_grouped = df[['store', 'item', 'sales']].groupby(['store', 'item']).tail(i)
        df_stores_items_grouped = df_stores_items_grouped.groupby(['store', 'item']).std()

        X_test = X_test.join(df_stores_items_grouped)
        X_test = X_test.drop(columns=[f'sales_rolling_stdev{i}'])
        X_test = X_test.rename(columns={"sales" : f'sales_rolling_stdev{i}'})

    for w in range(7, 500, 7):
        i = w
        X_test[f'sales_rolling_mean_diff{7}-{i}'] = X_test[f'sales_rolling_mean7'] - X_test[f'sales_rolling_mean{i}']

    return X_test.reset_index()


def fill_lagged_features(df, X_test) -> pd.DataFrame :

    X_test = X_test.set_index(['store', 'item'])

    # setting manually search by store, item is slow ~2.4s
    # this one works the best. ~0.8sec
    for i in [1,2,3,4,5,6]:
        df_stores_items_grouped = df[['store', 'item', 'sales']].groupby(['store', 'item']).nth(-i).set_index(['store', 'item'])

        X_test = X_test.join(df_stores_items_grouped)
        X_test = X_test.drop(columns=[f"sales_lag{i}"])
        X_test = X_test.rename(columns={"sales" : f"sales_lag{i}"})

    for w in [1,2,3,4,5,6,7,8]:
        
        i = w * 7
        df_stores_items_grouped = df[['store', 'item', 'sales']].groupby(['store', 'item']).nth(-i).set_index(['store', 'item'])

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
    # best num_iterations = ~470-480
    param = {
        'objective': 'tweedie',
        'metric': 'None',
        'num_iterations': 475
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

        #print(f"lagged")
        # fill the NA features
        X_test = fill_lagged_features(df, X_test)

        #print(f"rolling")
        X_test = fill_rolling_features(df, X_test)

        # print(len(train_cols))

        #print(f"predict")
        # predict the row.
        y_pred = model.predict(X_test[train_cols])

        X_test['sales'] = y_pred
        X_test['date'] = dt
        X_test['year'] = dt.year
        X_test['future'] = True

        df = pd.concat([df, X_test], axis=0).reset_index(drop=True)

        if max_iter is not None and i >= max_iter:
            return df, start_test_date


    return df, start_test_date



model = train_entire(df_no_na)


df_pred_1, std = predict(model, df_no_na, df_test, max_iter=None)


df_pred_simpl = df_pred_1[df_pred_1['future'] == True][['date', 'store', 'item', 'sales']]
df_pred_simpl['sales_rounded'] = np.round(df_pred_simpl['sales'])
df_pred_simpl


df_submission = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv')


df_test_cp = df_test.copy()
df_test_cp_sales = df_test_cp.merge(right=df_pred_simpl, on=['date', 'store', 'item'], how='left')


df_test_cp_sales[['id', 'sales_rounded']].to_csv("submission.csv", header=['id', 'sales'], index=False)

