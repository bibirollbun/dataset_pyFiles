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


data = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")

data


data.info()


import matplotlib.pyplot as plt

# Aggregate the total num_sold by country
country_sales = data.groupby('country')['num_sold'].sum().reset_index()

# Sort by num_sold for better visualization
country_sales = country_sales.sort_values(by='num_sold', ascending=False)

# Plot the aggregated num_sold by country as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(country_sales['country'], country_sales['num_sold'], color='skyblue')

# Customize the plot
plt.title('Aggregated Total Num Sold by Country')
plt.xlabel('Country')
plt.ylabel('Total Number of Items Sold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()



# Aggregate the total num_sold by store
store_sales = data.groupby('store')['num_sold'].sum().reset_index()

# Sort by num_sold for better visualization
store_sales = store_sales.sort_values(by='num_sold', ascending=False)

# Plot the aggregated num_sold by store as a bar chart
plt.figure(figsize=(12, 6))
plt.bar(store_sales['store'], store_sales['num_sold'], color='coral')

# Customize the plot
plt.title('Aggregated Total Num Sold by Store')
plt.xlabel('Store')
plt.ylabel('Total Number of Items Sold')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()



# Aggregate the total num_sold by product
product_sales = data.groupby('product')['num_sold'].sum().reset_index()

# Sort by num_sold for better visualization
product_sales = product_sales.sort_values(by='num_sold', ascending=False)

# Plot the aggregated num_sold by product as a bar chart
plt.figure(figsize=(12, 6))
plt.bar(product_sales['product'], product_sales['num_sold'], color='lightblue')

# Customize the plot
plt.title('Aggregated Total Num Sold by Product')
plt.xlabel('Product')
plt.ylabel('Total Number of Items Sold')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()



# Group the data by date, summing up the num_sold values
overall_time_series = data.groupby('date')['num_sold'].sum().reset_index()

# Plot the overall time-series data
plt.figure(figsize=(12, 6))
plt.plot(overall_time_series['date'], overall_time_series['num_sold'], label='Overall', color='blue')

# Customize the plot
plt.title('Overall Time-Series of Num Sold')
plt.xlabel('Date')
plt.ylabel('Number of Items Sold')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()



import matplotlib.pyplot as plt

# Convert the 'date' column to a datetime format
data['date'] = pd.to_datetime(data['date'])

# Group the data by date and country, summing up the num_sold values
time_series_data = data.groupby(['date', 'country'])['num_sold'].sum().reset_index()

# Plot the time-series data
plt.figure(figsize=(12, 6))

# Plot for each country
for country in time_series_data['country'].unique():
    country_data = time_series_data[time_series_data['country'] == country]
    plt.plot(country_data['date'], country_data['num_sold'], label=country)

# Customize the plot
plt.title('Time-Series of Num Sold by Country')
plt.xlabel('Date')
plt.ylabel('Number of Items Sold')
plt.legend(title='Country')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()



# Group the data by date and store, summing up the num_sold values
store_time_series_data = data.groupby(['date', 'store'])['num_sold'].sum().reset_index()

# Plot the time-series data by store
plt.figure(figsize=(12, 6))

# Plot for each store
for store in store_time_series_data['store'].unique():
    store_data = store_time_series_data[store_time_series_data['store'] == store]
    plt.plot(store_data['date'], store_data['num_sold'], label=store)

# Customize the plot
plt.title('Time-Series of Num Sold by Store')
plt.xlabel('Date')
plt.ylabel('Number of Items Sold')
plt.legend(title='Store')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()



import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

train_file_path = "/kaggle/input/playground-series-s5e1/train.csv"
df = pd.read_csv(train_file_path)

df['date'] = pd.to_datetime(df['date'])
df['num_sold'].fillna(df['num_sold'].median(), inplace=True)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
df['quarter'] = df['date'].dt.quarter

df['lag_1'] = df['num_sold'].shift(1)
df['lag_7'] = df['num_sold'].shift(7)
df['rolling_mean_3'] = df['num_sold'].rolling(window=3, min_periods=1).mean()
df['rolling_mean_7'] = df['num_sold'].rolling(window=7, min_periods=1).mean()

df.drop(columns=['date', 'id'], inplace=True)
df[['country', 'store', 'product']] = df[['country', 'store', 'product']].astype(str)
df = pd.get_dummies(df, columns=['country', 'store', 'product'])

df.fillna(df.median(), inplace=True)
train_columns = list(df.columns)
train_columns.remove('num_sold')
joblib.dump(train_columns, "train_columns.pkl")

X = df.drop(columns=['num_sold'])
y = df['num_sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAE: {mae:.2f}\nMAPE: {mape:.2%}")

joblib.dump(rf_model, "random_forest_model.pkl")

test_file_path = "/kaggle/input/playground-series-s5e1/test.csv"
test_df = pd.read_csv(test_file_path)

test_ids = test_df[['id']]
test_df['date'] = pd.to_datetime(test_df['date'])
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['weekday'] = test_df['date'].dt.weekday
test_df['is_weekend'] = test_df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
test_df['quarter'] = test_df['date'].dt.quarter

test_df.drop(columns=['date', 'id'], inplace=True)
test_df[['country', 'store', 'product']] = test_df[['country', 'store', 'product']].astype(str)

test_df = pd.get_dummies(test_df, columns=['country', 'store', 'product'])
rf_model = joblib.load("random_forest_model.pkl")
train_columns = joblib.load("train_columns.pkl")

missing_cols = set(train_columns) - set(test_df.columns)
extra_cols = set(test_df.columns) - set(train_columns)

for col in missing_cols:
    test_df[col] = 0

test_df = test_df.drop(columns=extra_cols, errors='ignore')
test_df = test_df[train_columns]
test_df = test_df.astype(float)

test_df['num_sold_predicted'] = rf_model.predict(test_df)

output_df = test_ids.copy()
output_df['num_sold_predicted'] = test_df['num_sold_predicted']
output_df.to_csv("/kaggle/working/submission.csv", index=False)
#print("Predictions saved successfully to forecasted_sales.csv!")

output_df


0.90806

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

train_file_path = r"C:\Users\USER\Downloads\train.csv"
df = pd.read_csv(train_file_path)

df['date'] = pd.to_datetime(df['date'])
df['num_sold'].fillna(df['num_sold'].median(), inplace=True)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
df['quarter'] = df['date'].dt.quarter

df['lag_1'] = df['num_sold'].shift(1)
df['lag_7'] = df['num_sold'].shift(7)
df['rolling_mean_3'] = df['num_sold'].rolling(window=3, min_periods=1).mean()
df['rolling_mean_7'] = df['num_sold'].rolling(window=7, min_periods=1).mean()
df['expanding_mean'] = df['num_sold'].expanding().mean()
df['yearly_avg'] = df.groupby(['year'])['num_sold'].transform('mean')

df.drop(columns=['date', 'id'], inplace=True)
df[['country', 'store', 'product']] = df[['country', 'store', 'product']].astype(str)
df = pd.get_dummies(df, columns=['country', 'store', 'product'])

df.fillna(df.median(), inplace=True)
train_columns = list(df.columns)
train_columns.remove('num_sold')
joblib.dump(train_columns, "train_columns.pkl")

X = df.drop(columns=['num_sold'])
y = df['num_sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=1000, max_depth=25, min_samples_split=4, min_samples_leaf=2, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAE: {mae:.2f}\nMAPE: {mape:.2%}")

joblib.dump(rf_model, "random_forest_model.pkl")

test_file_path = r"C:\Users\USER\Downloads\test.csv"
test_df = pd.read_csv(test_file_path)

test_ids = test_df[['id']]
test_df['date'] = pd.to_datetime(test_df['date'])
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['weekday'] = test_df['date'].dt.weekday
test_df['is_weekend'] = test_df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
test_df['quarter'] = test_df['date'].dt.quarter

test_df.drop(columns=['date', 'id'], inplace=True)
test_df[['country', 'store', 'product']] = test_df[['country', 'store', 'product']].astype(str)

test_df = pd.get_dummies(test_df, columns=['country', 'store', 'product'])
rf_model = joblib.load("random_forest_model.pkl")
train_columns = joblib.load("train_columns.pkl")

missing_cols = set(train_columns) - set(test_df.columns)
extra_cols = set(test_df.columns) - set(train_columns)

for col in missing_cols:
    test_df[col] = 0

test_df = test_df.drop(columns=extra_cols, errors='ignore')
test_df = test_df[train_columns]
test_df = test_df.astype(float)

test_df['num_sold_predicted'] = rf_model.predict(test_df)

output_df = test_ids.copy()
output_df['num_sold_predicted'] = test_df['num_sold_predicted']
output_df.to_csv(r"C:\Users\USER\Downloads\forecasted_sales3.csv", index=False)
print("Predictions saved successfully to forecasted_sales.csv!")




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


data_file_path = r"C:\Users\USER\Downloads\train.csv"
data = pd.read_csv(data_file_path)

#Convert date to datetime

data['date'] = pd.to_datetime(data['date'])


data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['weekday'] = data['date'].dt.weekday
data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x >= 5 else 0)
data['quarter'] = data['date'].dt.quarter

data['lag_1'] = data['num_sold'].shift(1)
data['lag_7'] = data['num_sold'].shift(7)
data['rolling_mean_3'] = data['num_sold'].rolling(window=3, min_periods=1).mean()
data['rolling_mean_7'] = data['num_sold'].rolling(window=7, min_periods=1).mean()
data['expanding_mean'] = data['num_sold'].expanding().mean()
data['yearly_avg'] = data.groupby(['year'])['num_sold'].transform('mean')
data['seasonal_avg'] = data.groupby(['month'])['num_sold'].transform('mean')
data['holiday_flag'] = data['weekday'].apply(lambda x: 1 if x in [5, 6] else 0)


data = pd.get_dummies(data, columns=['country', 'store', 'product'], drop_first=True)


data.drop(columns=['id', 'date'], inplace=True)



if 'num_sold' not in data.columns:
    raise KeyError("Column 'num_sold' not found in the dataset")



data = data.replace([np.inf, -np.inf], np.nan).dropna()



X = data.drop(columns=['num_sold'])
y = data['num_sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


xgb_model = xgb.XGBRegressor(
n_estimators=1500, learning_rate=0.03, max_depth=10, subsample=0.85, colsample_bytree=0.85, gamma=0.1, random_state=42
)
xgb_model.fit(X_train_scaled, y_train)


y_pred = xgb_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAE: {mae:.2f}\nMAPE: {mape:.2%}")



test_file_path = r"C:\Users\USER\Downloads\test.csv"
test_df = pd.read_csv(test_file_path)


test_ids = test_df[['id']]


test_df['date'] = pd.to_datetime(test_df['date'])


test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['weekday'] = test_df['date'].dt.weekday
test_df['is_weekend'] = test_df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
test_df['quarter'] = test_df['date'].dt.quarter
test_df['holiday_flag'] = test_df['weekday'].apply(lambda x: 1 if x in [5, 6] else 0)


test_df = pd.get_dummies(test_df, columns=['country', 'store', 'product'], drop_first=True)


missing_cols = set(X.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0  # Add missing columns as 0



test_df = test_df[X.columns]


test_df_scaled = scaler.transform(test_df)


test_df['num_sold_predicted'] = xgb_model.predict(test_df_scaled)


output_df = test_ids.copy()
output_df['num_sold_predicted'] = test_df['num_sold_predicted']


output_df.to_csv(r"C:\Users\USER\Downloads\forecasted_sales_Xbooster.csv", index=False)
print("Predictions saved successfully to forecasted_sales_Xbooster.csv!")




import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

train_file_path = "/kaggle/input/playground-series-s5e1/train.csv"
df = pd.read_csv(train_file_path)

df['date'] = pd.to_datetime(df['date'])
df['num_sold'].fillna(df['num_sold'].median(), inplace=True)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
df['quarter'] = df['date'].dt.quarter

df['lag_1'] = df['num_sold'].shift(1)
df['lag_7'] = df['num_sold'].shift(7)
df['rolling_mean_3'] = df['num_sold'].rolling(window=3, min_periods=1).mean()
df['rolling_mean_7'] = df['num_sold'].rolling(window=7, min_periods=1).mean()
df['expanding_mean'] = df['num_sold'].expanding().mean()
df['yearly_avg'] = df.groupby(['year'])['num_sold'].transform('mean')
df['seasonal_avg'] = df.groupby(['month'])['num_sold'].transform('mean')
df['holiday_flag'] = df['weekday'].apply(lambda x: 1 if x in [5, 6] else 0)

df.drop(columns=['date', 'id'], inplace=True)
df[['country', 'store', 'product']] = df[['country', 'store', 'product']].astype(str)
df = pd.get_dummies(df, columns=['country', 'store', 'product'])
df.fillna(df.median(), inplace=True)

train_columns = list(df.columns)
train_columns.remove('num_sold')
joblib.dump(train_columns, "train_columns.pkl")

X = df.drop(columns=['num_sold'])
y = df['num_sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#rf_model = RandomForestRegressor(n_estimators=1500, max_depth=30, min_samples_split=3, min_samples_leaf=1, random_state=42)
rf_model = RandomForestRegressor(n_estimators=1000, max_depth=30, min_samples_split=3, min_samples_leaf=1, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAE: {mae:.2f}\nMAPE: {mape:.2%}")

#joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl", compress=3)

test_file_path = "/kaggle/input/playground-series-s5e1/test.csv"
test_df = pd.read_csv(test_file_path)

test_ids = test_df[['id']]
test_df['date'] = pd.to_datetime(test_df['date'])
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['weekday'] = test_df['date'].dt.weekday
test_df['is_weekend'] = test_df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
test_df['quarter'] = test_df['date'].dt.quarter
test_df['holiday_flag'] = test_df['weekday'].apply(lambda x: 1 if x in [5, 6] else 0)

test_df.drop(columns=['date', 'id'], inplace=True)
test_df[['country', 'store', 'product']] = test_df[['country', 'store', 'product']].astype(str)
test_df = pd.get_dummies(test_df, columns=['country', 'store', 'product'])
rf_model = joblib.load("random_forest_model.pkl")
train_columns = joblib.load("train_columns.pkl")

missing_cols = set(train_columns) - set(test_df.columns)
extra_cols = set(test_df.columns) - set(train_columns)

for col in missing_cols:
    test_df[col] = 0

test_df = test_df.drop(columns=extra_cols, errors='ignore')
test_df = test_df[train_columns]
test_df = test_df.astype(float)

test_df['num_sold_predicted'] = rf_model.predict(test_df)

output_df = test_ids.copy()
output_df['num_sold_predicted'] = test_df['num_sold_predicted']
output_df.to_csv("/kaggle/working/forecasted_sales4.csv", index=False)
print("Predictions saved successfully to forecasted_sales.csv!")

output_df


