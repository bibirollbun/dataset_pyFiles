# basic library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc
import holidays
import warnings

from statsmodels.graphics.tsaplots import plot_acf

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# model
import xgboost as xgb
import catboost as cb
import lightgbm as lgbm
import optuna

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


df_train.info()


df_test.info()


# change date to be datetime format
df_train['date'] = pd.to_datetime(df_train['date'], format='%Y-%m-%d')
df_test['date'] = pd.to_datetime(df_test['date'], format='%Y-%m-%d')


df_train


obj_col = ['country', 'store', 'product']
for i in obj_col:
    print(f'{i}: {df_train[i].unique()}')


plt.figure(figsize=(15,3))
df_train.groupby('date')['num_sold'].sum().plot(xlabel='date',
                                               ylabel='product sold',
                                               title='total sale over time')
plt.show()


plt.figure(figsize=(15,3))
sns.lineplot(x=df_train['date'].dt.year, y=df_train['num_sold'], hue=df_train['country'])
plt.show()


fig, axs = plt.subplots(2,2)
fig.set_size_inches(20,5)

# plot acf of original data
plot_acf(df_train.groupby('date')['num_sold'].sum().dropna(), lags=20, ax=axs[0,0])
axs[0,0].set_title('original acf plot')

# plot acf with drop nan and log tranformation
df_train_dropna_log = df_train.copy()
df_train_dropna_log = df_train_dropna_log.dropna().reset_index(drop=True)
df_train_dropna_log['num_sold'] = np.log(df_train_dropna_log['num_sold'])
plot_acf(df_train_dropna_log.groupby('date')['num_sold'].sum().dropna(), lags=20, ax=axs[0,1])
axs[0,1].set_title('drop nan and transfrom with log')

# plot acf with bfill and log tranformation
df_train_bfill_log = df_train.copy()
df_train_bfill_log['num_sold'] = df_train_bfill_log['num_sold'].bfill() # bfill
df_train_bfill_log['num_sold'] = np.log(df_train_bfill_log['num_sold']) # log transform
plot_acf(df_train_bfill_log.groupby('date')['num_sold'].sum().dropna(), lags=20, ax=axs[1,0])
axs[1,0].set_title('bfill and transfrom with log')

plt.show()


# do log transformation
X = df_train_bfill_log.drop(columns='num_sold')
y = df_train_bfill_log['num_sold']

print(f'X: {X.shape}')
print(f'y: {y.shape}')


class AddDays(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        us_holidays = holidays.country_holidays('US')
        X['weekday'] = np.nan
        for i in range(len(X)):
            if X['date'][i] in us_holidays:
                X['weekday'][i] = True
            else:
                X['weekday'][i] = False
        X['weekday'] = X['weekday'].astype('bool')
        X['day'] = X.date.dt.day
        X['month'] = X.date.dt.month
        X['year'] = X.date.dt.year
        return X
        

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.drop(self.cols, axis=1)


# Column Transformer --------------------------------------------------------------------------------
ohe_list = ['country', 'store', 'product']
ohe_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer([
    ('ohe', ohe_pipeline, ohe_list)
], remainder='passthrough')


# pipeline --------------------------------------------------------------------------------
pipeline = Pipeline([
    ('add_days', AddDays()),
    ('drop_id', DropColumns(cols=['id', 'date'])),
    ('preprocess', preprocess)
])

pipeline


xgb_pipeline = Pipeline([
    ('pipeline', pipeline),
    ('xgb', xgb.XGBRegressor(random_state=42))
])

xgb_pipeline


xgb_pipeline.fit(X, y)
y_preds = xgb_pipeline.predict(df_test)
y_preds


# inverse transform to original values
y_preds = np.exp(y_preds)
y_preds


df_sub = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', usecols=['id'])
df_sub['num_sold'] = y_preds
df_sub


df_sub.to_csv('submission.csv', index=False)

