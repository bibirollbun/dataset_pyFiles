import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
paths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        paths.append(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install py7zr


import py7zr
for filenames in paths:
    with py7zr.SevenZipFile(filenames, mode='r') as z_ref:
        z_ref.extractall(path='/kaggle/working')


%%time
data = pd.read_csv("/kaggle/working/train.csv")
data["date"] =  pd.to_datetime(data["date"])


monthly_counts_data = data['date'].dt.to_period('M').value_counts().sort_index()

plt.figure(figsize=(12, 6))
monthly_counts_data.plot(kind='bar', color='skyblue')
plt.title('Count of Data for Each Month for All data', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


%%time
THRESHOLD_TRAIN_DATE = pd.to_datetime("2017-07-01")
THRESHOLD_TEST_DATE = pd.to_datetime("2017-08-01")

data[data["date"] < THRESHOLD_TRAIN_DATE].to_csv("/kaggle/working/train_data.csv")
data[(data["date"] >= THRESHOLD_TRAIN_DATE)&(data["date"] < THRESHOLD_TEST_DATE)].to_csv("/kaggle/working/test_data.csv")

%xdel data


%%time
df = pd.read_csv("/kaggle/working/train_data.csv")
test_df = pd.read_csv("/kaggle/working/test_data.csv")


df["date"] = pd.to_datetime(df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])


monthly_counts = df['date'].dt.to_period('M').value_counts().sort_index().reset_index()
monthly_counts_test = test_df['date'].dt.to_period('M').value_counts().sort_index().reset_index()


monthly_counts['type'] = 'Train'
monthly_counts_test['type'] = 'Test'
combined = pd.concat([monthly_counts, monthly_counts_test])

plt.figure(figsize=(12, 6))
sns.barplot(data=combined, x='date', y='count', hue='type', palette=['blue', 'red'])
plt.title('Count of Data for Each Month (Train vs Test)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Dataset')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


df.head()


holiday_events = pd.read_csv("/kaggle/working/holidays_events.csv")
print(holiday_events.shape)
holiday_events.head()


items = pd.read_csv("/kaggle/working/items.csv")
print(items.shape)
items.head()


oil = pd.read_csv("/kaggle/working/oil.csv")
print(oil.shape)
oil.head()


stores = pd.read_csv("/kaggle/working/stores.csv")
print(stores.shape)
stores.head()


from sklearn.metrics import mean_squared_log_error

min_value = df['unit_sales'].min()
if min_value < 0:
    df['unit_sales'] = df['unit_sales'] - min_value
    test_df['unit_sales'] = test_df['unit_sales'] - min_value

# Calculate mean sales from the training data
mean_sales = df['unit_sales'].mean()

# Generate predictions for the test data
test_df['predicted_sales'] = mean_sales

# Evaluate using NWRMSLE
def nwrmsle(y_true, y_pred, weights=None):
    if weights is None:
        weights = np.ones_like(y_true)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred, sample_weight=weights))
    return rmsle

# Calculate NWRMSLE
nwrmsle_score = nwrmsle(test_df['unit_sales'], test_df['predicted_sales'])
print(f"Baseline NWRMSLE: {nwrmsle_score}")


holiday_events['date'] = pd.to_datetime(holiday_events['date'])
oil['date'] = pd.to_datetime(oil['date'])

# Merge additional datasets
def merge_additional_data(df):
    # Merge holiday data
    df = df.merge(holiday_events, on='date', how='left')
    df['is_holiday'] = df['type'].notna().astype(int)
    
    # Merge item data
    df = df.merge(items, on='item_nbr', how='left')
    
    # Merge oil data
    df = df.merge(oil, on='date', how='left')
    
    # Merge store data
    df = df.merge(stores, on='store_nbr', how='left')
    
    return df

# Apply merging to train and test data
df = merge_additional_data(df)
test_df = merge_additional_data(test_df)

# Feature Engineering
def create_features(df):
    # Time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    
    # Lag features for oil prices
    df['oil_price_lag_7'] = df['dcoilwtico'].shift(7)
    df['oil_price_lag_14'] = df['dcoilwtico'].shift(14)
    
    # Rolling window features for oil prices
    df['oil_price_rolling_mean_7'] = df['dcoilwtico'].rolling(window=7).mean()
    df['oil_price_rolling_std_7'] = df['dcoilwtico'].rolling(window=7).std()
    
    # Holiday features
    df['is_national_holiday'] = (df['locale'] == 'National').astype(int)
    df['is_regional_holiday'] = (df['locale'] == 'Regional').astype(int)
    df['is_local_holiday'] = (df['locale'] == 'Local').astype(int)
    
    # Item features
    df['is_perishable'] = df['perishable'].fillna(0).astype(int)
    
    # Store features
    df['store_type'] = df['type'].astype('category').cat.codes
    df['store_cluster'] = df['cluster'].astype('category').cat.codes
    
    # Drop unnecessary columns
    df = df.drop(columns=['type_x', 'locale', 'description', 'type_y', 'class', 'family', 'city', 'state', 'cluster'])
    
    return df

# Apply feature engineering to train and test data
df = create_features(df)
test_df = create_features(test_df)

# Fill missing values
df = df.fillna(0)
test_df = test_df.fillna(0)

# Display the final datasets
print("Train DataFrame:")
print(train_df.head())
print("\nTest DataFrame:")
print(test_df.head())


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import numpy as np

# Prepare data
X = df.drop(columns=['unit_sales', 'date'])
y = df['unit_sales']

# Encode categorical features
categorical_features = ['family', 'city', 'type', 'store_type', 'store_cluster']
X = pd.get_dummies(X, columns=categorical_features)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# Define hyperparameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'verbose': -1
}

# Train the model
model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000, early_stopping_rounds=100)

# Generate predictions
test_X = test_df.drop(columns=['unit_sales', 'date'])
test_X = pd.get_dummies(test_X, columns=categorical_features)
predictions = model.predict(test_X)

# Evaluate using NWRMSLE
def nwrmsle(y_true, y_pred, weights=None):
    if weights is None:
        weights = np.ones_like(y_true)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred, sample_weight=weights))
    return rmsle

nwrmsle_score = nwrmsle(test_df['unit_sales'], predictions)
print(f"LightGBM NWRMSLE: {nwrmsle_score}")


from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import numpy as np

# Prepare data
X = df.drop(columns=['unit_sales', 'date'])
y = df['unit_sales']

# Define categorical features
categorical_features = ['family', 'city', 'type', 'store_type', 'store_cluster']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CatBoost model
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.01,
    depth=6,
    loss_function='RMSE',
    verbose=100,
    cat_features=categorical_features
)

model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)

# Generate predictions
test_X = test_df.drop(columns=['unit_sales', 'date'])
predictions = model.predict(test_X)

# Evaluate using NWRMSLE
def nwrmsle(y_true, y_pred, weights=None):
    if weights is None:
        weights = np.ones_like(y_true)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred, sample_weight=weights))
    return rmsle

nwrmsle_score = nwrmsle(test_df['unit_sales'], predictions)
print(f"CatBoost NWRMSLE: {nwrmsle_score}")

