import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import optuna
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv(r'/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e1/test.csv')


train.head()


train.info()


train = train.dropna()


COLS = train.columns.to_list()[1:]


grouped_data = train.groupby("date")["num_sold"].sum()
# Visualize the time series
plt.figure(figsize=(15, 8))
plt.plot(grouped_data, label="Sales")
plt.xticks(ticks=['2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', 
                  '2015-01-01', '2016-01-01', '2016-12-31'],
          labels = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017])
plt.title("Time Series - Sticker Sales")
plt.xlabel("Date")
plt.ylabel("Number of Sales")
plt.legend()
plt.grid(False)
plt.show()


WINDOW_SIZE=30
df_filtered = train.copy()
df_filtered.set_index('date', inplace=True)
df_filtered = df_filtered[['num_sold']]
df_filtered = df_filtered.groupby(df_filtered.index).mean()
df_filtered['moving_avg'] = df_filtered['num_sold'].rolling(window=WINDOW_SIZE).mean()
df_filtered['moving_avg'] = df_filtered['moving_avg'].dropna()


plt.figure(figsize=(15, 8))
plt.plot(df_filtered['num_sold'], label="Original Series", alpha=0.5)
plt.plot(df_filtered['moving_avg'], label=f"Moving Average ({WINDOW_SIZE} days)", color='red')
plt.xticks(ticks=['2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', 
                  '2015-01-01', '2016-01-01', '2016-12-31'],
          labels = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017])
plt.title("Time Series with Applied Moving Average")
plt.xlabel("Date")
plt.ylabel("Number of Sales")
plt.legend()
plt.grid(False)
plt.show()


def transform_date(df, col):
    df[col] = pd.to_datetime(df[col])
    
    df['year'] = df[col].dt.year.astype('float64')
    df['quarter'] = df[col].dt.quarter.astype('float64')
    df['month'] = df[col].dt.month.astype('float64')
    df['day'] = df[col].dt.day.astype('float64')
    df['day_of_week'] = df[col].dt.dayofweek.astype('float64')
    df['week_of_year'] = df[col].dt.isocalendar().week.astype('float64')
    df['hour'] = df[col].dt.hour.astype('float64')
    df['minute'] = df[col].dt.minute.astype('float64')
    
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7.0)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7.0)
    
    df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7
    
    return df

train = transform_date(train, 'date')
test = transform_date(test, 'date')

train = train.drop(columns=['date'], axis=1)
test = test.drop(columns=['date'], axis=1)


categorical_cols = ['country', 'store', 'product']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le


train['num_sold'] = np.log1p(train['num_sold'])
train.head()


X = train.drop(columns=['num_sold'])
y = train['num_sold']
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = xgb.XGBRegressor(
    colsample_bytree=0.45920559358578084,
    eta=0.0477816891488785,
    min_child_weight=8,
    subsample=0.8304989594139964,
    max_depth=7,
    gamma=0.005523299726962971,
    reg_alpha=1.2067712927214142e-06,
    reg_lambda=9.990274895374969e-05,
    n_estimators=1000,
    objective='reg:squarederror',
    random_state=42, 
    n_jobs=-1,
    device='cuda',
    tree_method='hist', 
    eval_metric='mape')

lgb_model = lgb.LGBMRegressor(
    learning_rate=0.13347923822343877,
    max_depth=8,
    reg_alpha=0.00012063539906841849,
    reg_lambda=5.938827979753753e-08,
    min_child_samples=22,
    colsample_bytree=0.7327897270985785,
    subsample=0.6049388744468239,
    n_estimators=1000,     
    objective='regression',  
    metric='mape',  
    n_jobs=-1,
    device='gpu',
    verbosity= -1
 )

catboost_model = CatBoostRegressor(
    learning_rate=0.1107362263432963,
    depth=5,
    min_data_in_leaf=82,
    l2_leaf_reg=0.004335491460394296,
    bagging_temperature=0.28679765376044786,
    random_strength=0.6412661294895914, 
    n_estimators=1000,
    loss_function='MAPE',
    eval_metric='MAPE',
    random_state=42,
    early_stopping_rounds=50,
    # task_type='GPU'
)


meta_model = LinearRegression()
stacking_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('catboost', catboost_model)],
    final_estimator=meta_model,
    n_jobs=-1
)
stacking_model.fit(X, y)


for col in categorical_cols:
    test[col] = label_encoders[col].transform(test[col])


predictions = stacking_model.predict(test)
predictions = np.expm1(predictions)





train = pd.read_csv(r'/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e1/test.csv')
test['num_sold'] = predictions

train = train[COLS]
test = test[COLS]
train = train.dropna()


for c in train['country'].unique():
    for s in train['store'].unique():
        for p in train['product'].unique():
            analysis_t = train.loc[(train['country'] == c) & (train['store'] == s) & 
                                   (train['product'] == p)]
            analysis_t.set_index('date', inplace=True)
            analysis_t.index = pd.to_datetime(analysis_t.index)
            plt.figure(figsize=(20, 6))
            plt.plot(analysis_t['num_sold'], label="Historical Data")

            analysis_p = test.loc[(test['country'] == c) & (test['store'] == s) & 
                                  (test['product'] == p)]
            analysis_p.set_index('date', inplace=True)
            analysis_p.index = pd.to_datetime(analysis_p.index)
            plt.plot(analysis_p['num_sold'], label="Future Predictions")

            plt.title(f"{c}_{s}_{p}")
            plt.xlabel("Date")
            plt.ylabel("Number of Sales")
            plt.legend()
            plt.grid(False)
            # plt.show()
            
            # break
        # break
    # break
        


