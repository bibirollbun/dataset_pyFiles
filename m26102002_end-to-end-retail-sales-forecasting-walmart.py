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
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


# load files

train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip', parse_dates=['Date'])
features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip', parse_dates=['Date'])
test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
store = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')

print(train.shape, features.shape, test.shape, store.shape)
train.head()


# Merge features and stores into train

df = pd.merge(train, features, on=['Store','Date'], how='left')
df = pd.merge(df, store, on = 'Store', how = 'left')
df.sort_values(['Store','Dept','Date'], inplace=True)
df.reset_index(drop=True, inplace=True)


df.info()


df.drop('IsHoliday_x',axis = 1, inplace = True)


df = df.rename(columns = {'IsHoliday_y':'IsHoliday'})


print("Missing per column:\n", df.isnull().sum())

# basic stats
df.describe(include='all')


print("\nDate range:", df['Date'].min(), "to", df['Date'].max())


agg = df.groupby('Date')['Weekly_Sales'].sum().reset_index().sort_values(by='Date', ascending=True)

#visualize

plt.figure(figsize=(12,4))
plt.plot(agg['Date'], agg['Weekly_Sales'])
plt.title('Total Weekly Sales (All stores)')
plt.xlabel('143 unique dates'); plt.ylabel('Sales')
plt.show()


# year-end peaks


holiday_sales = df.groupby('IsHoliday')['Weekly_Sales'].mean()
print("avearge sales - NonHoliday vs Holiday:\n", holiday_sales)


holiday_sales.plot(kind='bar', figsize=(6,3),color=['skyblue','salmon'])
plt.title('Average Weekly Sales: Holiday vs Non-Holiday')
plt.ylabel('Weekly Sales')
plt.show()


# Average sales are significantly higher on holidays → the IsHoliday feature is important for the model.


s = df[(df.Store==1) & (df.Dept==1)].copy()
plt.figure(figsize=(12,4))
plt.plot(s['Date'], s['Weekly_Sales'])
plt.title('Store 1 - Dept 1 Weekly Sales')
plt.show()


# Some series are noisy, some are smooth;


# basic time feature

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsHoliday'] = df['IsHoliday'].astype(int)


# 7.1 - create lags and rolling per store-dept

lag_list = [1,2,3,4,12]   # weeks
for lag in lag_list:
    df[f'lag_{lag}'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(lag)

df['rmean_4'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(1).rolling(window=4).mean()
df['rmean_12'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(1).rolling(window=12).mean()


# 7.2 - fill external regressors missing (forward/backfill small gaps)

df[['Temperature','Fuel_Price','CPI','Unemployment']] = df[['Temperature','Fuel_Price','CPI','Unemployment']].fillna(method='ffill').fillna(method='bfill')


# 8.0 - drop initial rows with NaNs from lags

model_df = df.dropna(subset=[f'lag_{l}' for l in lag_list] + ['rmean_4']).copy()
print("After dropna:", model_df.shape)
# choose feature cols
feature_cols = ['Store','Dept','Year','Month','Week','DayOfWeek','IsHoliday',
                'Temperature','Fuel_Price','CPI','Unemployment'] + \
               [f'lag_{l}' for l in lag_list] + ['rmean_4','rmean_12']
target_col = 'Weekly_Sales'
model_df[feature_cols].head()


# Last 12 weeks = validation, rest = training

cutoff_date = model_df['Date'].max() - pd.Timedelta(weeks=12)

train = model_df[model_df['Date'] <= cutoff_date]
val   = model_df[model_df['Date'] > cutoff_date]

X_train, y_train = train[feature_cols], train[target_col]
X_val, y_val     = val[feature_cols],   val[target_col]

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)


# 10.0 - baseline using lag_1

y_pred_base = X_val['lag_1'].values
def mape(y_true,y_pred): return np.mean(np.abs((y_true - y_pred)/ (y_true + 1e-9))) * 100
print("Baseline RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_base)))
print("Baseline MAE:", mean_absolute_error(y_val, y_pred_base))
print("Baseline MAPE:", mape(y_val, y_pred_base))


# 11.0 - train XGBoost regressor
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=50)

# Predict + eval
y_pred = xgb.predict(X_val)
print("XGB RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))
print("XGB MAE:", mean_absolute_error(y_val, y_pred))
print("XGB MAPE:", mape(y_val, y_pred))


# 12.0 - feature importance
imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(8,6))
imp.head(20).plot(kind='barh')
plt.title('Top Features (XGBoost)')
plt.gca().invert_yaxis()
plt.show()


# - Holidays drive big uplifts. Plan inventory/staffing for known holiday weeks.
# - Recent sales (lag_1, rolling mean) are strongest predictors → short-term momentum matters.
# - External regressors (CPI, Unemployment, Fuel_Price) add small but useful signal.
# - Use XGBoost global model for automated weekly predictions; use Prophet for per-store communication.
# - For production: build iterative prediction pipeline that updates lags each week and ingests known holiday calendar.


# 13. Forecast next N weeks (example: 12 weeks)

# Step 1: create future dates same way you created lags earlier
last_date = df['Date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=12, freq='W-MON')

# Step 2: build future features (lags, rolling means, holidays, store/dept)
# Note: For production-ready pipeline you’d automate lag creation, here is a simple placeholder
future_df = pd.DataFrame({'Date': future_dates})

# Example: keep it minimal if no external regressors prepared
X_future = df[feature_cols].iloc[-12:]   # reuse last 12 rows’ structure for demo

# Step 3: predict
y_future = xgb.predict(X_future)

# Step 4: save results
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted_Sales": y_future
})

forecast_df.to_csv("xgb_forecast_output.csv", index=False)
print("Forecast saved as xgb_forecast_output.csv")

