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


df1 = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


df1.tail()


df1.info()


df1.describe()


df2 = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


df2.head()


df1.isnull().sum()


duplicate_count = df1.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")



df1['num_sold'].describe()


import matplotlib.pyplot as plt

# Replace 'num_sold' with the column you want to visualize
plt.figure(figsize=(10, 6))
plt.hist(df1['num_sold'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of num_sold', fontsize=16)
plt.xlabel('num_sold', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# df1['num_sold'].fillna(df1['num_sold'].mean(),inplace = True)


import matplotlib.pyplot as plt

# Replace 'num_sold' with the column you want to visualize
plt.figure(figsize=(10, 6))
plt.hist(df1['num_sold'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of num_sold', fontsize=16)
plt.xlabel('num_sold', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


df1['num_sold'].describe()


import seaborn as sns

# Create a Boxplot with Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(data=df1, x='num_sold')
plt.title('Boxplot of column_name')
plt.show()


canada_rows = df1[df1['country'] == 'Canada']


canada_rows['num_sold'].isnull().sum()


Italy_rows = df1[df1['country'] == 'Italy']
Finland_rows = df1[df1['country'] == 'Finland']
Kenya_rows = df1[df1['country'] == 'Kenya']
Norway_rows = df1[df1['country'] == 'Norway']
Singapore_rows = df1[df1['country'] == 'Singapore']
i=Italy_rows['num_sold'].isnull().sum()
f=Finland_rows['num_sold'].isnull().sum()
k=Kenya_rows['num_sold'].isnull().sum()
n=Norway_rows['num_sold'].isnull().sum()
s=Singapore_rows['num_sold'].isnull().sum()
print(i,f,k,n,s)


canada_rows.describe()


Kenya_rows.describe()


df1.loc[df1['country'] == 'Kenya', 'num_sold'] = df1.loc[df1['country'] == 'Kenya', 'num_sold'].fillna(20.43)


df1.loc[df1['country'] == 'Canada', 'num_sold'] = df1.loc[df1['country'] == 'Canada', 'num_sold'].fillna(840.04)


# counts_per_country = df1.groupby('country')['num_sold'].count()





df1['country'].unique()


df1['store'].unique()


df1['product'].unique()



df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d', errors='coerce')


df1['year'] = df1['date'].dt.year
df1['month'] = df1['date'].dt.month
df1['day'] = df1['date'].dt.day
df1['weekday'] = df1['date'].dt.weekday
df1['is_weekend'] = df1['date'].dt.weekday.isin([5, 6]).astype(int)
import numpy as np

# Encode month as a cyclical feature
df1['month_sin'] = np.sin(2 * np.pi * df1['month'] / 12)
df1['month_cos'] = np.cos(2 * np.pi * df1['month'] / 12)

# Encode day of the week as a cyclical feature
df1['weekday_sin'] = np.sin(2 * np.pi * df1['weekday'] / 7)
df1['weekday_cos'] = np.cos(2 * np.pi * df1['weekday'] / 7)


df1.head()


df1.drop(columns=['date'],inplace = True)


df1.drop(columns=['id'],inplace = True)


df1.head()


from sklearn.preprocessing import OneHotEncoder


# List of columns to encode
columns_to_encode = ['country', 'store', 'product']

# Create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Apply one-hot encoding to the selected columns
encoded_array = encoder.fit_transform(df1[columns_to_encode])

# Create a DataFrame for the encoded columns
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(columns_to_encode))

# Concatenate the encoded columns with the original DataFrame
df1 = pd.concat([df1, encoded_df], axis=1)

# Optionally, drop the original categorical columns
df1.drop(columns=columns_to_encode, inplace=True)


df1.head()


X = df1.drop(columns=['num_sold'])
y= df1['num_sold']


X


y


from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.fit_transform(x_test)


X_train


X_train_scaled


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train the regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test_scaled)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


from sklearn.ensemble import RandomForestRegressor

# Create and train the Random Forest model
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(x_test_scaled)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAPE: {mape:.2f}%")


# from sklearn.metrics import mean_absolute_error
# import numpy as np

# # Make predictions using the model (assuming y_pred and y_test are available)
# y_pred = model.predict(x_test_scaled)

# # Calculate Mean Absolute Percentage Error (MAPE)
# mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# print(f"MAPE: {mape:.2f}%")





import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Convert data to DMatrix format (specific to XGBoost)
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(x_test_scaled, label=y_test)

# Set parameters for XGBoost with GPU
params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',  # Mean Absolute Error
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'device': 'cuda',  # Enable GPU support
    'tree_method': 'hist'
}

# Train the XGBoost model
model_xgb = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions on the test set
y_pred_xgb = model_xgb.predict(dtest)

# Evaluate the model using MAPE
mape_xgb = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100
print(f"XGBoost MAPE: {mape_xgb:.2f}%")



import catboost
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Create the CatBoost model
model_catboost = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    cat_features=[],  # Specify categorical feature indices (if any)
    task_type='GPU',  # Enable GPU training
    loss_function='RMSE',  # Use RMSE (Root Mean Squared Error) for regression
    devices='0'  # Specify GPU device ID, '0' usually means the first GPU
)

# Train the model
model_catboost.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_catboost = model_catboost.predict(x_test_scaled)

# Evaluate the model using Mean Absolute Percentage Error (MAPE)
mape_catboost = np.mean(np.abs((y_test - y_pred_catboost) / y_test)) * 100
print(f"CatBoost MAPE: {mape_catboost:.2f}%")



df2.info()


df2.isnull().sum()


df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d', errors='coerce')
df2['year'] = df2['date'].dt.year
df2['month'] = df2['date'].dt.month
df2['day'] = df2['date'].dt.day
df2['weekday'] = df2['date'].dt.weekday
df2['is_weekend'] = df2['date'].dt.weekday.isin([5, 6]).astype(int)
import numpy as np2
# 2ncode month as a cyclical feature
df2['month_sin'] = np.sin(2 * np.pi * df2['month'] / 12)
df2['month_cos'] = np.cos(2 * np.pi * df2['month'] / 12)
# 2ncode day of the week as a cyclical feature
df2['weekday_sin'] = np.sin(2 * np.pi * df2['weekday'] / 7)
df2['weekday_cos'] = np.cos(2 * np.pi * df2['weekday'] / 7)



df2.drop(columns=['date'],inplace = True)
df2.drop(columns=['id'],inplace = True)


df2.head()


temp = df2.copy()


df2.shape


df2


# from sklearn.preprocessing import OneHotEncoder


# # List of columns to encode
# columns_to_encode = ['country', 'store', 'product']

# # Create an instance of OneHotEncoder
# encoder = OneHotEncoder()

# # Apply one-hot encoding to the selected columns
# encoded_array = encoder.fit_transform(df2[columns_to_encode])

# # Create a DataFrame for the encoded columns
# encoded_df2 = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(columns_to_encode))

# # Concatenate the encoded columns with the original DataFrame
# df2 = pd.concat([df2, encoded_df2], axis=1)

# # # Optionally, drop the original categorical columns
# # df2.drop(columns=columns_to_encode, inplace=True)


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# List of columns to encode
columns_to_encode = ['country', 'store', 'product']

# Create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse=False)  # sparse=False to return dense array

# Apply one-hot encoding to the selected columns
encoded_array = encoder.fit_transform(df2[columns_to_encode])

# Create a DataFrame for the encoded columns
encoded_df2 = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(columns_to_encode))

# Concatenate the encoded columns with the original DataFrame
df2 = pd.concat([df2, encoded_df2], axis=1)

# Optionally, drop the original categorical columns
df2.drop(columns=columns_to_encode, inplace=True)




df2.shape



scaled_test = scaler.fit_transform(df2)


scaled_test


prediction = model.predict(scaled_test)


prediction


for_id = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


for_id.head()


result = pd.DataFrame()
result["id"] = for_id.id
result["num_sold"] = prediction


result.head()


result.to_csv("submission.csv", index = False)




