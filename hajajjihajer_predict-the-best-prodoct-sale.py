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


import os  # Operating system interactions
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns  # High-level data visualization based on matplotlib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold,KFold, RandomizedSearchCV  # Model selection and cross-validation
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder  # Preprocessing steps
from sklearn.impute import SimpleImputer  # Handling missing values
from sklearn.compose import ColumnTransformer  # Applying transformers to specific columns
from sklearn.pipeline import Pipeline  # Pipeline assembly
from sklearn.decomposition import PCA  # Dimensionality reduction

# Set display option for pandas
pd.set_option('display.max_rows', None)  # Display all rows in pandas DataFrame

# === Regression Models ===
from sklearn.linear_model import LinearRegression,LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# === Evaluation Metrics ===
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For classification
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error  # For regression

# Ignore all warnings
import warnings  # Suppress warnings
warnings.filterwarnings('ignore')

#Regular expression operations
import re


# Load data
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')


print('***************sales_train*********************')
print(sales_train.nunique())
print('****************items********************')
print(items.nunique())
print('*****************item_categories*******************')
print(item_categories.nunique())
print('*****************shops*******************')
print(shops.nunique())
print('*****************test*******************')
print(test.nunique())


# Convert date to datetime
sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')

# Check for missing values
print(sales_train.isnull().sum())
print('**************************************************')
print(items.isnull().sum())
print('**************************************************')
print(item_categories.isnull().sum())
print('**************************************************')
print(shops.isnull().sum())


sales_train.head(2)


sales_train.shape


sales_train.query('item_cnt_day<=0').shape


sales_train.query('item_price<=0').head()


sales_train = sales_train[sales_train['item_price'] > 0]


sales_train['item_cnt_day'] = sales_train['item_cnt_day'].apply(lambda x: 1 if x <= 0 else x)


sales_train.query('item_cnt_day<=0').shape


sales_train.query('item_id<=0').head()


sales_train = sales_train[sales_train['item_id'] > 0]


sales_train.query('date_block_num<0').shape


#Use duplicated() to find any duplicate rows.
print(sales_train.duplicated().sum())


sales_train = sales_train.drop_duplicates()


print(sales_train.duplicated().sum())


# Aggregate monthly sales
# Plot sales trends over time
plt.figure(figsize=(10,6))
plt.plot(sales_train.groupby('date_block_num')['item_cnt_day'].sum())
plt.title('Monthly Sales Over Time')
plt.xlabel('Month (date_block_num)')
plt.ylabel('Total Sales')
plt.show()


# Top 10 items by total sales
top_items = sales_train.groupby('item_id')['item_cnt_day'].sum().sort_values(ascending=False).head(10)
print(top_items)


# Top 10 shops by total sales
top_shops = sales_train.groupby('shop_id')['item_cnt_day'].sum().sort_values(ascending=False).head(10)
print(top_shops)


# Plot item price distribution
plt.figure(figsize=(10,6))
plt.hist(sales_train['item_price'], bins=100, log=True)
plt.title('Item Price Distribution')
plt.xlabel('Item Price')
plt.ylabel('Frequency (log scale)')
plt.show()


sns.boxplot(x=sales_train['item_price'], orient='h')


sales_train.shape


sales_train=sales_train[sales_train['item_price']<40000]


sales_train.shape


sns.boxplot(x=sales_train['item_price'], orient='h')


sns.distplot(sales_train['item_price'])


sales_train[sales_train['item_price']>5000].shape


sales_train=sales_train[sales_train['item_price']<5000]


sns.distplot(sales_train['item_price'])


sns.boxplot(x=sales_train['item_price'], orient='h')


# Plot item sales distribution
plt.figure(figsize=(10,6))
plt.hist(sales_train['item_cnt_day'], bins=100, log=True)
plt.title('Sales Count per Item Distribution')
plt.xlabel('Sales Count per Day')
plt.ylabel('Frequency (log scale)')
plt.show()


sns.boxplot(x=sales_train['item_cnt_day'], orient='h')


sales_train[sales_train['item_cnt_day']>10].shape


sales_train=sales_train[sales_train['item_cnt_day']<10]


sns.boxplot(x=sales_train['item_cnt_day'], orient='h')


sns.distplot(sales_train['item_cnt_day'])


sales_train.shape


# Correlation matrix
corr_matrix = sales_train[['item_price', 'item_cnt_day']].corr()

# Plot the heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Add 'month' column
sales_train['month'] = sales_train['date'].dt.month

# Aggregate sales by month
monthly_sales_by_month = sales_train.groupby('month')['item_cnt_day'].sum()

# Plot
plt.figure(figsize=(10,6))
monthly_sales_by_month.plot(kind='bar')
plt.title('Total Sales per Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()


# Check the unique shops
print(shops.head())


# Check for sales count outliers
plt.figure(figsize=(10,6))
plt.boxplot(sales_train['item_cnt_day'], vert=False)
plt.title('Sales Count Outliers')
plt.show()

# Check for price outliers
plt.figure(figsize=(10,6))
plt.boxplot(sales_train['item_price'], vert=False)
plt.title('Item Price Outliers')
plt.show()


sales_train.head()


# Aggregate sales to monthly level
sales_train['revenue'] = sales_train['item_price'] * sales_train['item_cnt_day']


monthly_sales = sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({
    'item_cnt_day': 'sum',
    'revenue': 'sum'
}).reset_index()


sales_train.head()


monthly_sales.shape


monthly_sales.head()


# Rename columns for clarity
monthly_sales.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)


monthly_sales.head()


# Merge with test data to align features
test_data = pd.merge(test, items, on='item_id', how='left')


test_data.isnull().sum()


# Add category to the monthly sales
monthly_sales = pd.merge(monthly_sales, items[['item_id','item_category_id']], on='item_id', how='left')


monthly_sales.head(3)


monthly_sales.isnull().sum()


sales_train[sales_train['date_block_num'] == 1]['shop_id'].nunique()


import itertools

# Create a grid to ensure every month has a row for every combination of shop_id and item_id
grid = []
for block_num in range(34):
    cur_shops = sales_train[sales_train['date_block_num'] == block_num]['shop_id'].unique()
    cur_items = sales_train[sales_train['date_block_num'] == block_num]['item_id'].unique()
    
    # Use itertools.product to create the Cartesian product of shops, items, and the current date_block_num
    grid.append(np.array(list(itertools.product(cur_shops, cur_items, [block_num]))))

# Combine all monthly grids into one DataFrame
grid = pd.DataFrame(np.vstack(grid), columns=['shop_id', 'item_id', 'date_block_num'])


# সকল ইউনিক শপ, আইটেম এবং মাসের তালিকা বের করা
shop_s = sales_train['shop_id'].unique()
item_s = sales_train['item_id'].unique()
month_s = sales_train['date_block_num'].unique()

# Step 1: শপ এবং মাসের জন্য কার্টেসিয়ান প্রোডাক্ট তৈরি করা
shops_df = pd.DataFrame({'shop_id': shop_s})
months_df = pd.DataFrame({'date_block_num': month_s})

# শপ এবং মাসের কার্টেসিয়ান প্রোডাক্ট তৈরি করা
shop_month_df = shops_df.merge(months_df, how='cross')

# Step 2: আইটেমের সাথে shop_month_df মিক্স করা (কার্টেসিয়ান প্রোডাক্ট)
items_df = pd.DataFrame({'item_id': item_s})

# shop_month_df এবং items_df এর কার্টেসিয়ান প্রোডাক্ট তৈরি করা
grid_ = shop_month_df.merge(items_df, how='cross')

# Final grid
print(grid_.head())  # প্রথম কিছু রো প্রিন্ট করে দেখা



grid.head()


grid.shape


# Merge the grid with monthly sales data
monthly_sales = pd.merge(grid, monthly_sales, on=['date_block_num', 'shop_id', 'item_id'], how='left').fillna(0)

# Feature engineering: Lag features (previous month sales)
for lag in [1, 2, 3]:
    monthly_sales[f'item_cnt_month_lag_{lag}'] = monthly_sales.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag)



monthly_sales.isnull().sum()


# Drop any rows with NaNs after shifting
monthly_sales.dropna(inplace=True)


monthly_sales.isnull().sum()


monthly_sales.shape


from sklearn.preprocessing import LabelEncoder

# Label Encoding for 'shop_id', 'item_id', and 'category_id'
le_shop = LabelEncoder()
monthly_sales['shop_id'] = le_shop.fit_transform(monthly_sales['shop_id'])

le_item = LabelEncoder()
monthly_sales['item_id'] = le_item.fit_transform(monthly_sales['item_id'])

le_category = LabelEncoder()
monthly_sales['item_category_id'] = le_category.fit_transform(monthly_sales['item_category_id'])


monthly_sales.head()


# Create a new feature to track the relative change in item price
sales_train['price_change'] = sales_train.groupby(['shop_id', 'item_id'])['item_price'].pct_change()

# Fill any NaN values (first observation of the time series will have NaN for pct_change)
sales_train['price_change'].fillna(0, inplace=True)


# Create rolling mean and sum features
monthly_sales['rolling_mean_sales'] = monthly_sales.groupby(['shop_id', 'item_id'])['item_cnt_month'].transform(lambda x: x.rolling(window=3).mean())
monthly_sales['rolling_sum_sales'] = monthly_sales.groupby(['shop_id', 'item_id'])['item_cnt_month'].transform(lambda x: x.rolling(window=3).sum())

# Fill NaN values with 0
monthly_sales['rolling_mean_sales'].fillna(0, inplace=True)
monthly_sales['rolling_sum_sales'].fillna(0, inplace=True)



# Cap extreme values of item_cnt_month and item_price
# monthly_sales['item_cnt_month'] = monthly_sales['item_cnt_month'].clip(0, 20)
# sales_train['item_price'] = sales_train['item_price'].clip(0, 100000)


# Train on all months except the last one (month 33 is the last month in training data)
X_train = monthly_sales[monthly_sales['date_block_num'] < 33]
X_valid = monthly_sales[monthly_sales['date_block_num'] == 33]

# Target variable (sales for the month)
y_train = X_train['item_cnt_month']
y_valid = X_valid['item_cnt_month']

# Drop the target variable from feature sets
X_train = X_train.drop(['item_cnt_month'], axis=1)
X_valid = X_valid.drop(['item_cnt_month'], axis=1)


import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Convert data to DMatrix for XGBoost
train_data = xgb.DMatrix(X_train, label=y_train)
valid_data = xgb.DMatrix(X_valid, label=y_valid)

# Define the model parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'min_child_weight': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse',
    'seed': 42
}

# Train the model
xgb_model = xgb.train(xgb_params, train_data, 500, [(train_data, 'train'), (valid_data, 'eval')], early_stopping_rounds=50, verbose_eval=10)

# Make predictions on validation set
y_pred = xgb_model.predict(valid_data)

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f'Validation RMSE: {rmse}')


X_train.head()


import pandas as pd
import numpy as np
import xgboost as xgb

# Step 1: Load the test data and merge with additional item data
test_data = pd.merge(test, items[['item_id', 'item_category_id']], on='item_id', how='left')

# Step 2: Preprocess test data similarly to the training data
# Assuming test data corresponds to the next date_block_num (which is 34, after the training period)
test_data['date_block_num'] = 34

# Step 3: Rename `item_category_id` to `category_id` if necessary
# If the training set used 'category_id', we need to rename it in the test set to ensure consistency
test_data.rename(columns={'item_category_id': 'item_category_id'}, inplace=True)

# Step 4: Merge with the necessary lag features and other engineered features
# Assuming you have monthly sales data with lag features
test_data = pd.merge(test_data, monthly_sales[['shop_id', 'item_id', 'date_block_num', 
                                               'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 
                                               'item_cnt_month_lag_3', 'rolling_mean_sales', 'rolling_sum_sales']], 
                     on=['shop_id', 'item_id', 'date_block_num'], how='left')

# Step 5: Add missing feature 'revenue' (or any other feature used in training) and handle missing values
test_data['revenue'] = 0  # Add 'revenue' feature to the test data if necessary
test_data.fillna(0, inplace=True)

# Step 6: Ensure the test set has the exact same column order as the training data
train_features = ['shop_id', 'item_id', 'date_block_num', 'revenue', 'item_category_id', 
                  'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3',
                  'rolling_mean_sales', 'rolling_sum_sales']

# Reorder test_data columns to match the order of train_features
X_test = test_data[train_features]

# Step 7: Convert the test set to a DMatrix object (required for XGBoost's Booster API)
test_dmatrix = xgb.DMatrix(X_test)

# Step 8: Apply the trained model to make predictions on the test set
test_preds = xgb_model.predict(test_dmatrix)

# Step 9: Clip predictions to match competition constraints (0 to 20 sales per month)
test_preds = np.clip(test_preds, 0, 20)

# Step 10: Create the submission DataFrame
submission = pd.DataFrame({
    'ID': test['ID'],            # Use the 'ID' column from the original test.csv
    'item_cnt_month': test_preds  # Predicted sales count
})

# Step 11: Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")


# import pandas as pd
# import numpy as np
# import xgboost as xgb

# # Step 1: Load the test data and merge with additional item data
# # Ensure `item_category_id` is used (since it was used in training)
# test_data = pd.merge(test, items[['item_id', 'item_category_id']], on='item_id', how='left')

# # Step 2: Preprocess test data similarly to the training data
# # Assuming test data corresponds to the next date_block_num (which is 34, after the training period)
# test_data['date_block_num'] = 34

# # Step 3: Ensure consistent feature naming
# # Now, we use `item_category_id` instead of `category_id` because that is what the model expects.

# # Step 4: Merge with lag features and other engineered features (as in the training data)
# test_data = pd.merge(test_data, monthly_sales[['shop_id', 'item_id', 'date_block_num', 
#                                                'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 
#                                                'item_cnt_month_lag_3', 'rolling_mean_sales', 'rolling_sum_sales']], 
#                      on=['shop_id', 'item_id', 'date_block_num'], how='left')

# # Step 5: Handle missing values and other necessary columns
# test_data['revenue'] = 0  # Add 'revenue' feature to the test data if necessary
# test_data.fillna(0, inplace=True)

# # Step 6: Ensure the test set has the exact same column order as the training data
# train_features = ['shop_id', 'item_id', 'date_block_num', 'revenue', 'item_category_id', 
#                   'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3',
#                   'rolling_mean_sales', 'rolling_sum_sales']

# # Reorder test_data columns to match the order of train_features
# X_test = test_data[train_features]

# # Step 7: Convert the test set to a DMatrix object (required for XGBoost's Booster API)
# test_dmatrix = xgb.DMatrix(X_test)

# # Step 8: Apply the trained model to make predictions on the test set
# test_preds = xgb_model.predict(test_dmatrix)

# # Step 9: Clip predictions to match competition constraints (0 to 20 sales per month)
# test_preds = np.clip(test_preds, 0, 20)

# # Step 10: Create the submission DataFrame
# submission = pd.DataFrame({
#     'ID': test['ID'],            # Use the 'ID' column from the original test.csv
#     'item_cnt_month': test_preds  # Predicted sales count
# })

# # Step 11: Save the submission to a CSV file
# submission.to_csv('submission.csv', index=False)

# print("Submission file created successfully!")




