pip install lightgbm


import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
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
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),
]
brno_holiday = [
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),
]

budapest_holidays = []
munich_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
]

frank_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
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
df['date_copy'] = df['date']
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


monthly_sales = train_data.groupby(['year', 'month', 'warehouse'])['sales'].sum().reset_index()

monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))
monthly_sales['date_copy'] = monthly_sales['date']

unique_warehouses = monthly_sales['warehouse'].unique()

for warehouse in unique_warehouses:
    warehouse_data = monthly_sales[monthly_sales['warehouse'] == warehouse].sort_values('date_copy')
    ts = warehouse_data.set_index('date_copy')['sales']
    ts = ts.asfreq('MS').fillna(0)
    result = seasonal_decompose(ts, model='additive', period=12)  

    plt.figure(figsize=(12, 15))
    result.plot()
    plt.suptitle(f'Trend and Seasonality Decomposition - {warehouse}', fontsize=16)
    plt.tight_layout()
    plt.show()


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


X = X.drop(columns=['date_copy'])


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
    'num_boost_round': 6000,
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
sub[['id', 'sales_hat']].to_csv("/kaggle/working/submission_LGB.csv", index=False)

print("Files in /kaggle/working/:")
print(os.listdir('/kaggle/working'))
print('submission file created successfully')


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import sys  # For forcing output

# Load data (Ensure the file paths are correct)
train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])

# --- Feature Engineering ---

def fill_loss_holidays(df_fill, warehouses, holidays):
    df = df_fill.copy()
    for item in holidays:
        dates, holiday_name = item
        generated_dates = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        for generated_date in generated_dates:
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday_name'] = holiday_name
    return df

czech_holiday = [ 
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"), 
]
brno_holiday = [
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),
]
budapest_holidays = []
munich_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
]
frank_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
]

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

# --- Merging Data ---

def create_merged_data(train_data, test_data):
    train_data['date'] = pd.to_datetime(train_data['date'])
    test_data['date'] = pd.to_datetime(test_data['date'])
    merged_df = pd.merge(test_data, calendar_extended, on=['date', 'warehouse'], how='left')
    merged_df = pd.merge(train_data, merged_df, on=['unique_id', 'date', 'warehouse'], how='left')
    return merged_df

# --- Evaluation Function ---

def evaluate_forecast(test_df, forecast_df):
    print("--- evaluate_forecast function called ---")  # ADDED: Debugging
    sys.stdout.flush()  # Force print output

    merged_data = pd.merge(test_df, forecast_df, on=['unique_id', 'date'], how='inner')
    
    # Print merged_data for inspection
    print("Sample of merged_data:")
    print(merged_data.head())
    sys.stdout.flush()

    # Check for NaNs
    print("\nNaNs in sales and forecast:")
    print(merged_data[['sales', 'forecast']].isnull().sum())
    sys.stdout.flush()

    # Verify data types
    print("\nData types of sales and forecast:")
    print(merged_data[['sales', 'forecast']].dtypes)
    sys.stdout.flush()

    # Check if merged_data is empty
    if merged_data.empty:
        print("\nError: merged_data is empty. Check your merge keys and data.")
        return None, None, None

    y_true = merged_data['sales'].values  # Get the raw numpy values
    y_pred = merged_data['forecast'].values # Get the raw numpy values

    # Check if y_true or y_pred are empty
    if y_true.size == 0 or y_pred.size == 0: # Use .size for numpy arrays
        print("\nError: y_true or y_pred is empty. Check your input data.")
        return None, None, None

    # *** NEW: Length Check and Truncation ***
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f'\nMSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    sys.stdout.flush()
    return mse, mae, rmse

# 1.  **Crucially, call the function to create test_merged**
test_merged = create_merged_data(train, test)

# 4. Create forecast_df (replace with your actual forecast generation)
forecast_df = test[['unique_id', 'date']].copy()
forecast_df['forecast'] = np.random.randint(0, 50, len(forecast_df))

print("--- forecast_df head ---")  # Debugging
print(forecast_df.head())
print("--- forecast_df shape ---")
print(forecast_df.shape)
print("--- forecast_df columns ---")
print(forecast_df.columns)

# 5.  Finally, call the evaluation function
evaluate_forecast(test_merged, forecast_df)


def evaluate_forecast(test_df, forecast_df):
    merged_data = pd.merge(test_df, forecast_df, on=['unique_id', 'date'], how='inner')

    y_true = merged_data['sales'].values
    y_pred = merged_data['forecast'].values

    y_true_original_scale = y_true**4
    y_pred_original_scale = y_pred**4

    mse = mean_squared_error(y_true_original_scale, y_pred_original_scale)
    mae = mean_absolute_error(y_true_original_scale, y_pred_original_scale)
    rmse = np.sqrt(mse)

    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    return mse, mae, rmse



train_data['date'] = pd.to_datetime(train_data['date'])

# June 2, 2024 is the last actual date (Sunday)
last_actual_date = pd.Timestamp('2024-06-02')

# Go back 6 weeks from June 2, 2024
start_date = last_actual_date - pd.Timedelta(weeks=4)

# Filter data for the last 6 weeks of actual sales
recent_sales = train_data[train_data['date'] >= start_date].copy()
recent_sales['week'] = recent_sales['date'].dt.to_period('W').astype(str)

# Aggregate weekly actual sales
weekly_sales = (
    recent_sales.groupby(['warehouse', 'week'])['sales']
    .sum()
    .reset_index()
    .rename(columns={'sales': 'value'})
)
weekly_sales['type'] = 'Actual'

# Step 2: Prepare forecast data (submission file)
submission = pd.read_csv('/kaggle/working/submission_LGB.csv')
submission[['unique_id', 'date']] = submission['id'].str.split('_', n=1, expand=True)
submission['date'] = pd.to_datetime(submission['date'])

# Convert IDs for merging
submission['unique_id'] = submission['unique_id'].astype(str)
inventory['unique_id'] = inventory['unique_id'].astype(str)
submission = submission.merge(inventory[['unique_id', 'warehouse']], on='unique_id', how='left')

# Filter forecast data to only include dates after June 2, 2024 (June 3 onward)
submission = submission[submission['date'] > last_actual_date]
submission['week'] = submission['date'].dt.to_period('W').astype(str)

# Aggregate weekly forecast sales
weekly_forecast = (
    submission.groupby(['warehouse', 'week'])['sales_hat']
    .sum()
    .reset_index()
    .rename(columns={'sales_hat': 'value'})
)
weekly_forecast['type'] = 'Forecast'

# Step 3: Plot actual and forecast per warehouse
warehouses = weekly_sales['warehouse'].unique()

for warehouse in warehouses:
    actual = weekly_sales[weekly_sales['warehouse'] == warehouse]
    forecast = weekly_forecast[weekly_forecast['warehouse'] == warehouse]
    
    # Combine actual and forecast data
    combined = pd.concat([actual, forecast]).sort_values('week')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(actual['week'], actual['value'], marker='o', color='steelblue', label='Actual Sales')
    plt.plot(forecast['week'], forecast['value'], marker='x', linestyle='--', color='orange', label='Forecasted Sales')

    plt.title(f'Weekly Sales (Last 6 Weeks + Forecast) - {warehouse}')
    plt.xlabel('Week')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Explicitly call plt.show() to display the plot
    plt.tight_layout()
    plt.show()  # This will show the plot

