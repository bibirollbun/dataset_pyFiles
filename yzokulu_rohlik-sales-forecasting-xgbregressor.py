# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from math import exp
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
test_weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
sales_train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
sales_test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
solution = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')


#taken from https://www.kaggle.com/code/macarrony00/not-a-winner-but-maybe-some-inspiration
from datetime import datetime
czech_holiday = [ 
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),#loss
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"), #loss
]
brno_holiday = [
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),#loss
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"), #loss
]

budapest_holidays = []
munich_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),#loss
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),#loss
]

frank_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),#loss
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),#loss
]

def fill_loss_holidays(df_fill, warehouses, holidays):
    df = df_fill.copy()
    for item in holidays:
        dates, holiday_name = item
        generated_dates = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        for generated_date in generated_dates:
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday_name'] = holiday_name
    return df

df_calendar = fill_loss_holidays(df_fill=df_calendar, warehouses=['Prague_1', 'Prague_2', 'Prague_3'], holidays=czech_holiday)
df_calendar = fill_loss_holidays(df_fill=df_calendar, warehouses=['Brno_1'], holidays=brno_holiday)
df_calendar = fill_loss_holidays(df_fill=df_calendar, warehouses=['Munich_1'], holidays=munich_holidays)
df_calendar = fill_loss_holidays(df_fill=df_calendar, warehouses=['Frankfurt_1'], holidays=frank_holidays)
df_calendar = fill_loss_holidays(df_fill=df_calendar, warehouses=['Budapest_1'], holidays=budapest_holidays)



#sales_train = sales_train[sales_train['availability'] == 1.00]
sales_train.drop(columns=['availability'], inplace=True)


def combine_sales_data(target_df, inventory_df, calendar_df):
    target_df = target_df.merge(inventory_df[['unique_id', 'product_unique_id', 'name', 
                                  'L1_category_name_en', 'L2_category_name_en', 
                                  'L3_category_name_en', 'L4_category_name_en']], 
                    on='unique_id', how='left')
    
    target_df = target_df.merge(calendar_df[['date', 'warehouse', 'holiday', 'holiday_name', 
                                 'shops_closed', 'winter_school_holidays', 'school_holidays']], 
                    on=['date', 'warehouse'], how='left')

    return target_df


sales_train = combine_sales_data(sales_train, inventory, df_calendar)
sales_test = combine_sales_data(sales_test, inventory, df_calendar)
sales_train['date'] = pd.to_datetime(sales_train['date'])
sales_test['date'] = pd.to_datetime(sales_test['date'])




sales_train['is_test'] = 0
sales_test['is_test'] = 1


merged_df = pd.concat([sales_train, sales_test], axis=0)


discount_columns = ['type_0_discount', 'type_1_discount', 'type_2_discount', 
                    'type_3_discount', 'type_4_discount', 'type_5_discount', 'type_6_discount']
merged_df['max_discount'] = merged_df[discount_columns].max(axis=1)



merged_df = merged_df.sort_values(by=['product_unique_id', 'warehouse', 'date'])

periods = [3,7,10,14]
for p in periods: 
    
    merged_df['total_orders_mean_' + str(p)] = merged_df.groupby(['product_unique_id', 'warehouse'])['total_orders'] \
    .rolling(window=p, min_periods=1) \
    .mean() \
    .reset_index(level=[0, 1], drop=True)

    merged_df['total_orders_std_' + str(p)] = merged_df.groupby(['product_unique_id', 'warehouse'])['total_orders'] \
    .rolling(window=p, min_periods=1) \
    .std() \
    .reset_index(level=[0, 1], drop=True)

    merged_df['total_orders_max_' + str(p)] = merged_df.groupby(['product_unique_id', 'warehouse'])['total_orders'] \
    .rolling(window=p, min_periods=1) \
    .max() \
    .reset_index(level=[0, 1], drop=True)

    merged_df['total_orders_min_' + str(p)] = merged_df.groupby(['product_unique_id', 'warehouse'])['total_orders'] \
    .rolling(window=p, min_periods=1) \
    .min() \
    .reset_index(level=[0, 1], drop=True)

for lag in range(1, 15):  
    merged_df[f'lag_{lag}'] = merged_df.groupby(['product_unique_id', 'warehouse'])['sales'].shift(lag)
    merged_df[f'lag_{lag}'] = merged_df[f'lag_{lag}'].fillna(merged_df.groupby(['product_unique_id', 'warehouse'])['sales'].transform("last"))


categorical_columns = ['warehouse', 'name', 'L1_category_name_en', 'L2_category_name_en', 
                       'L3_category_name_en', 'L4_category_name_en', 'holiday_name']

    
for col in categorical_columns:
    merged_df[col] = merged_df[col].astype('category')
    merged_df[col] = merged_df[col].cat.codes


merged_df["datetime"] = pd.to_datetime(merged_df["date"])
merged_df["month"] = merged_df["datetime"].dt.month
merged_df["day"] = merged_df["datetime"].dt.day
merged_df["weekday"] = merged_df["datetime"].dt.weekday
merged_df["quarter"] = merged_df["datetime"].dt.quarter
merged_df["week_of_year"] = merged_df["datetime"].dt.isocalendar().week
merged_df["day_of_year"] = merged_df["datetime"].dt.dayofyear
merged_df["is_weekend"] = merged_df["datetime"].dt.weekday.isin([5, 6]).astype(int)
merged_df["is_month_start"] = merged_df["datetime"].dt.is_month_start.astype(int)
merged_df["is_month_end"] = merged_df["datetime"].dt.is_month_end.astype(int)
merged_df['year_sin'] = np.sin(2 * np.pi * merged_df['datetime'].dt.year)
merged_df['year_cos'] = np.cos(2 * np.pi * merged_df['datetime'].dt.year)
merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12) 
merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)
merged_df['day_sin'] = np.sin(2 * np.pi * merged_df['day'] / 31)  
merged_df['day_cos'] = np.cos(2 * np.pi * merged_df['day'] / 31)
    
merged_df.drop(columns=["datetime"], inplace=True)



sales_train_new = merged_df[(merged_df['is_test'] == 0) & (merged_df['date'] < '2024-05-20') & (merged_df['date'] >= '2023-01-01')]
sales_val = merged_df[(merged_df['is_test'] == 0) & (merged_df['date'] >= '2024-05-21') & (merged_df['date'] <= '2024-06-02')]

sales_train_new = sales_train_new.dropna()
sales_val = sales_val.dropna()

sales_test_new = merged_df[merged_df['is_test'] == 1]





sales_test_new.drop(columns=["sales"], inplace=True)


from lightgbm import LGBMRegressor, early_stopping  # Import early_stopping

features = [
    "warehouse", "total_orders", 
    "sell_price_main", "type_0_discount", "type_1_discount", "type_2_discount", 
    "type_3_discount", "type_4_discount", "type_5_discount", "type_6_discount", 
    "name", "L1_category_name_en", "L2_category_name_en", 
    "L3_category_name_en", "L4_category_name_en", "holiday", "holiday_name", 
    "shops_closed", "winter_school_holidays", "school_holidays", 
    "max_discount", "total_orders_mean_3", "total_orders_std_3", "total_orders_max_3", 
    "total_orders_min_3", "total_orders_mean_7", "total_orders_std_7", "total_orders_max_7", 
    "total_orders_min_7", "total_orders_mean_10", "total_orders_std_10", "total_orders_max_10", 
    "total_orders_min_10", "total_orders_mean_14", "total_orders_std_14", "total_orders_max_14", 
    "total_orders_min_14", "month", "day", "weekday", "quarter", "week_of_year", 
    "day_of_year", "is_weekend", "is_month_start", "is_month_end", "year_sin", 
    "year_cos", "month_sin", "month_cos", "day_sin", "day_cos",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7", 
    "lag_8", "lag_9", "lag_10", "lag_11", "lag_12", "lag_13", "lag_14"
] 


X_train = sales_train_new[features]
y_train = np.log1p(sales_train_new['sales'])
X_val = sales_val[features]
y_val = np.log1p(sales_val['sales'])
X_test = sales_test_new[features]



model = CatBoostRegressor(
    iterations=10_000,
    #learning_rate=0.1,
    depth=10,
    loss_function="MAE",
    verbose=100,
    random_seed=0,
    early_stopping_rounds=50,
)


model = LGBMRegressor(
    boosting_type='gbdt',  
    objective='regression',  
    metric='mae',  
    num_leaves=31,  
    learning_rate=0.1,  
    feature_fraction=0.9,  
    bagging_fraction=0.8,  
    bagging_freq=5,  
    verbose=-1,  
    n_jobs=-1,  
    random_state=42,  
    n_estimators=10000
)


params = {
    'eval_metric': 'mae', 
    'learning_rate': 0.1,  
    'max_depth': 11,  
    'min_child_weight': 1,  
    'subsample': 0.8,  
    'colsample_bytree': 0.8,  
    'n_estimators': 10000,  
    'n_jobs': -1,  
    'seed': 42  
}

model = XGBRegressor(**params)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=200, verbose=100)


y_val_pred = model.predict(X_val)
y_test_pred = np.expm1(model.predict(X_test))


mae = mean_absolute_error(y_val, y_val_pred)
print(f'Mean Absolute Error on validation set: {mae}')


sales_test_new['predicted_sales'] = y_test_pred





sales_test_new['id'] = sales_test_new['unique_id'].astype(str) + '_' + sales_test_new['date'].astype(str) 
 
solution = sales_test_new[['id', 'predicted_sales']].copy()
    
solution.columns = ['id', 'sales_hat']
    
solution.to_csv('submission.csv', index=False)


print(solution.tail(20))

