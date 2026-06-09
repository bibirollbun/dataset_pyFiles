import pandas as pd
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import warnings
from datetime import datetime
import os

warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
solution = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])


print(train.info())
print(train.isnull().sum()) #info of train set


unique_ids = [885, 1237, 725, 3778, 5152, 2148, 2424, 3178, 1776, 1689, 612, 2809, 794]
filtered = train[train['unique_id'].isin(unique_ids)]
results = []
for unique_id in unique_ids:
    current_filtered = filtered[filtered['unique_id'] == unique_id]
    null_1 = current_filtered['total_orders'].isnull().sum()
    non_null_1 = current_filtered['total_orders'].notnull().sum()
    null_2 = current_filtered['sales'].isnull().sum()
    non_null_2 = current_filtered['sales'].notnull().sum()
    results.append({
        'unique_id': unique_id,
        'Null values in total_orders': null_1,
        'Non-null values in total_orders': non_null_1,
        'Null values in sales': null_2,
        'Non-null values in sales': non_null_2
    })
result_data = pd.DataFrame(results)
print(result_data)


print(test.info())
print(test.isnull().sum()) #info of test set


print(calendar.info())
print(calendar.isnull().sum()) #info of calendar set


print(inventory.info())
print(inventory.isnull().sum()) #info of inventory set


print(solution.info())
print(solution.isnull().sum()) #info of solution set


print(weights.info())
print(weights.isnull().sum()) #info of test weights set


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

calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Prague_1', 'Prague_2', 'Prague_3'], holidays=czech_holiday)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Brno_1'], holidays=brno_holiday)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Munich_1'], holidays=munich_holidays)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Frankfurt_1'], holidays=frank_holidays)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Budapest_1'], holidays=budapest_holidays)


Frankfurt_1 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Frankfurt_1"')
Prague_2 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Prague_2"')
Brno_1 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Brno_1"')
Munich_1 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Munich_1"')
Prague_3 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Prague_3"')
Prague_1 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Prague_1"')
Budapest_1 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Budapest_1"')


def process_calendar(df):
    df = df.sort_values('date').reset_index(drop=True)
    df['next_holiday_date'] = df.loc[df['holiday'] == 1, 'date'].shift(-1)
    df['next_holiday_date'] = df['next_holiday_date'].bfill()
    df['days_to_holiday'] = (df['next_holiday_date'] - df['date']).dt.days
    df.drop(columns=['next_holiday_date'], inplace=True)
    df['next_shops_closed_date'] = df.loc[df['shops_closed'] == 1, 'date'].shift(-1)
    df['next_shops_closed_date'] = df['next_shops_closed_date'].bfill()
    df['days_to_shops_closed'] = (df['next_shops_closed_date'] - df['date']).dt.days
    df.drop(columns=['next_shops_closed_date'], inplace=True)
    df['day_after_closing'] = (
        (df['shops_closed'] == 0) & (df['shops_closed'].shift(1) == 1)
    ).astype(int)
    
    df['long_weekend'] = (
        (df['shops_closed'] == 1) & (df['shops_closed'].shift(1) == 1)
    ).astype(int)
    
    df['weekday'] = df['date'].dt.weekday 
    return df


dfs = ['Frankfurt_1', 'Prague_2', 'Brno_1', 'Munich_1', 'Prague_3', 'Prague_1', 'Budapest_1']
processed_dfs = [process_calendar(globals()[df]) for df in dfs]
calendar_extended = pd.concat(processed_dfs).sort_values('date').reset_index(drop=True)


train_calendar = train.merge(calendar_extended, on=['date', 'warehouse'], how='left')
train_inventory = train_calendar.merge(inventory, on=['unique_id', 'warehouse'], how='left')
train_data = train_inventory.merge(weights, on=['unique_id'], how='left')

test_calendar = test.merge(calendar_extended, on=['date', 'warehouse'], how='left')
test_datas = test_calendar.merge(inventory, on=['unique_id', 'warehouse'], how='left')


train_data = train_data.drop(columns=['availability'])
train_data.dropna(subset=['sales'], inplace=True)


df = train_data
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['dayofweek'] = df['date'].dt.dayofweek
df['weekofyear'] = df['date'].dt.isocalendar().week
df['dayofyear'] = df['date'].dt.dayofyear
df['is_month_start'] = df['date'].dt.is_month_start
df['is_month_end'] = df['date'].dt.is_month_end
df['quarter'] = df['date'].dt.quarter
df["total_dic"] = df['type_0_discount'] + df['type_0_discount'] + df['type_1_discount'] + df['type_2_discount'] + df['type_3_discount'] + df['type_4_discount'] + df['type_5_discount'] + df['type_6_discount']
df['total_orders_'] = df['total_orders'] / df['sell_price_main']
df['total_orders_dic'] = df['total_orders_'] / df["total_dic"]
df['total_orders_sell_price_main'] = df['sell_price_main'] / df["total_dic"]

for i in range(7):
    df[f'total_orders{i}'] = df[f'type_{i}_discount'] / df["total_orders"]
    df[f'total_orders_sell_price_main_{i}'] = df[f'type_{i}_discount'] / df["total_orders_sell_price_main"]
    df[f'sell_price_main{i}'] = df[f'type_{i}_discount'] / df["sell_price_main"]
    df[f'sell_price_main_x_{i}'] = df[f'type_{i}_discount'] / (df["sell_price_main"] * df["total_orders"])
    df[f'total_orders_dic{i}'] = df[f'type_{i}_discount'] / df["total_orders_dic"]

    df[f'_total_orders{i}'] = df[f'type_{i}_discount'] * df["total_orders"]
    df[f'_total_orders_sell_price_main_{i}'] = df[f'type_{i}_discount'] * df["total_orders_sell_price_main"]
    df[f'_sell_price_main{i}'] = df[f'type_{i}_discount'] * df["sell_price_main"]
    df[f'_total_orders_dic{i}'] = df[f'type_{i}_discount'] * df["total_orders_dic"]

df.fillna(0, inplace=True)


warehouse_yearly_sales = df.groupby(['warehouse', 'year'])['sales'].sum().reset_index()
warehouse_yearly_sales['sales_rise'] = warehouse_yearly_sales.groupby('warehouse')['sales'].diff()
warehouse_yearly_sales['sales_rise_percentage'] = warehouse_yearly_sales.groupby('warehouse')['sales'].pct_change() * 100

warehouse_yearly_sales['sales_rise'].fillna(0, inplace=True)
warehouse_yearly_sales['sales_rise_percentage'].fillna(0, inplace=True)

print(warehouse_yearly_sales)


valid_sales_rise = warehouse_yearly_sales[warehouse_yearly_sales['year'] != 2020]
mean_sales_rise_percentage = valid_sales_rise['sales_rise_percentage'].replace([np.inf, -np.inf], np.nan).mean(skipna=True)

sales_change_multiplier = 1 + (mean_sales_rise_percentage / 100)
sales_change_multiplier = max(min(sales_change_multiplier, 1.05), 1.0)

print("Integrated Sales Change Multiplier:", sales_change_multiplier)


categorical_columns = df.select_dtypes("object").columns

for col in categorical_columns:
    df[col] = df[col].astype('category')


df_test = test_datas
df_test['date'] = pd.to_datetime(df_test['date'])
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df_test['weekday'] = df_test['date'].dt.weekday
df_test['dayofweek'] = df_test['date'].dt.dayofweek
df_test['weekofyear'] = df_test['date'].dt.isocalendar().week
df_test['dayofyear'] = df_test['date'].dt.dayofyear
df_test['is_month_start'] = df_test['date'].dt.is_month_start
df_test['is_month_end'] = df_test['date'].dt.is_month_end
df_test['quarter'] = df_test['date'].dt.quarter

df_test["total_dic"] = df_test['type_0_discount'] + df_test['type_0_discount'] + df_test['type_1_discount'] + df_test['type_2_discount'] + df_test['type_3_discount'] + df_test['type_4_discount'] + df_test['type_5_discount'] + df_test['type_6_discount']
df_test['total_orders_'] = df_test['total_orders'] / df_test['sell_price_main']
df_test['total_orders_dic'] = df_test['total_orders_'] / df_test["total_dic"]
df_test['total_orders_sell_price_main'] = df_test['sell_price_main'] / df_test["total_dic"]

for i in range(7):
    df_test[f'total_orders{i}'] = df_test[f'type_{i}_discount'] / df_test["total_orders"]
    df_test[f'total_orders_sell_price_main_{i}'] = df_test[f'type_{i}_discount'] / df_test["total_orders_sell_price_main"]
    df_test[f'sell_price_main{i}'] = df_test[f'type_{i}_discount'] / df_test["sell_price_main"]
    df_test[f'sell_price_main_x_{i}'] = df_test[f'type_{i}_discount'] / (df_test["sell_price_main"] * df_test["total_orders_sell_price_main"])
    df_test[f'total_orders_dic{i}'] = df_test[f'type_{i}_discount'] / df_test["total_orders_dic"]
    df_test[f'_total_orders{i}'] = df_test[f'type_{i}_discount'] * df_test["total_orders"]
    df_test[f'_total_orders_sell_price_main_{i}'] = df_test[f'type_{i}_discount'] * df_test["total_orders_sell_price_main"]
    df_test[f'_sell_price_main{i}'] = df_test[f'type_{i}_discount'] * df_test["sell_price_main"]
    df_test[f'_total_orders_dic{i}'] = df_test[f'type_{i}_discount'] * df_test["total_orders_dic"]

df_test.fillna(0, inplace=True)

for col in categorical_columns:
    df_test[col] = df_test[col].astype('category')


train_start_date = '2020-08-01'
train_end_date = '2024-03-18'
test_start_date = '2024-03-18'
test_end_date = '2024-06-01'

X = df.drop(['sales', 'date', 'unique_id', 'weight'], axis=1)
y = np.sqrt(np.sqrt(df['sales']))

train_data = df[(df['date'] < train_end_date)]
test_data = df[(df['date'] >= test_start_date)]

X_train = train_data.drop(['sales', 'date', 'unique_id', 'weight'], axis=1)
y_train = np.sqrt(np.sqrt(train_data['sales']))
train_weights = train_data['weight']

X_test = test_data.drop(['sales', 'date', 'unique_id', 'weight'], axis=1)
y_test = np.sqrt(np.sqrt(test_data['sales']))
test_weights = test_data['weight']


cols = X.select_dtypes(["int", "float"]).columns
categorical_feature_indices = [X.columns.get_loc(col) for col in categorical_columns if col in X.columns]

sc = StandardScaler()

for col in cols:
    X[col].replace([np.inf, -np.inf], X[col].min(), inplace=True)
    df_test[col].replace([np.inf, -np.inf], df_test[col].min(), inplace=True)

    X[col].fillna(X[col].mean(), inplace=True)
    df_test[col].fillna(df_test[col].mean(), inplace=True)

X[cols] = sc.fit_transform(X[cols])
df_test[cols] = sc.transform(df_test[cols])


callbacks = [log_evaluation(period=200)]

params = {
    'learning_rate': 0.021796506746095975,
    'num_leaves': 93,
    'max_depth': 10,
    'min_child_samples': 25,
    'subsample': 0.7057135664023435,
    'colsample_bytree': 0.8528497905459008,
    'reg_alpha': 0.036786449788597686,
    'reg_lambda': 0.3151110021900479,
    'num_boost_round': 9800,
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'verbose': -1
}

final_train_dataset = lgb.Dataset(X, label=y, 
                                  categorical_feature=categorical_feature_indices,
                                  weight=df['weight'])
final_model = lgb.train(params, 
                        final_train_dataset, 
                        num_boost_round=params['num_boost_round'],
                        callbacks=callbacks)
print('Model run successfully')


final_y_pred = final_model.predict(df_test.drop(['date', 'unique_id'], axis=1), num_iteration=final_model.best_iteration)
final_y_pred = np.power(np.power(final_y_pred, 2), 2)
sub = df_test.copy()
sub['sales_hat'] = final_y_pred * 1.05
sub['id'] = sub['unique_id'].astype(str) + "_" + sub['date'].astype(str)
sub[['id', 'sales_hat']].to_csv("/kaggle/working/submission.csv", index=False)

print("Files in /kaggle/working/:")
print(os.listdir('/kaggle/working'))
print('submission file created successfully')

