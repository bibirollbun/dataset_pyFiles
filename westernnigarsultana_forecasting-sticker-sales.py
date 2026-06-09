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
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

train.head(10)


test.head(10)


train.info()


train.describe(include ='object')


train.describe()


train.isnull().sum()


train=train.dropna()


test['num_sold']=0
data= pd.concat([train,test],axis=0)


train.isnull().sum()


test.isnull().sum()


train['date'] = pd.to_datetime(train['date'], errors='coerce')



train['year'] = train['date'].dt.year  # Extract year
sales_by_year = train.groupby('year')['num_sold'].sum()
sales_by_year.plot(kind='bar', title='Total Sales by Year', figsize=(8, 5))



train['month'] = train['date'].dt.month  # Extract month
sales_by_month = train.groupby('month')['num_sold'].sum()
sales_by_month.plot(kind='bar', title='Total Sales by Month', figsize=(8, 5))



import matplotlib.pyplot as plt
import pandas as pd

# Group total sales by year and month
sales_by_month_year = train.groupby(['year', 'month'])['num_sold'].sum().unstack()

# Plot the grouped bar chart
sales_by_month_year.plot(kind='bar', figsize=(12, 6), title='Total Sales by Month for Each Year')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



train['day_of_week'] = train['date'].dt.dayofweek  # Extract day of the week (0=Monday, 6=Sunday)
sales_by_day = train.groupby('day_of_week')['num_sold'].sum()
sales_by_day.plot(kind='bar', title='Total Sales by Day of the Week', figsize=(8, 5))



sales_by_store = train.groupby('store')['num_sold'].sum()
sales_by_store.plot(kind='bar', title='Total Sales by Store', figsize=(8, 5))



# Group data by both store and year and calculate total sales
sales_by_store_year = train.groupby(['store', 'year'])['num_sold'].sum().unstack()

# Plot total sales by store for each year as a stacked bar chart
sales_by_store_year.plot(kind='bar', stacked=True, figsize=(10, 6), title='Total Sales by Store for Each Year')



# Group sales by year and store
sales_by_store_year = train.groupby([train['date'].dt.year, 'store'])['num_sold'].sum()

# Convert to table format
sales_by_store_year_table = sales_by_store_year.unstack()  # Creates columns for stores
sales_by_store_year_table.index.name = 'Year'
sales_by_store_year_table.columns.name = 'Store'
sales_by_store_year_table.fillna(0, inplace=True)  # Fill missing stores with 0
sales_by_store_year_table



sales_by_product = train.groupby('product')['num_sold'].sum()
sales_by_product.plot(kind='bar', title='Total Sales by Product', figsize=(8, 5))



# Group data by both product and year and calculate total sales
sales_by_product_year = train.groupby(['product', 'year'])['num_sold'].sum().unstack()

# Plot total sales by product for each year as a stacked bar chart
sales_by_product_year.plot(kind='bar', stacked=True, figsize=(10, 6), title='Total Sales by Product for Each Year')



# Group sales by year and product
sales_by_product_year = train.groupby([train['date'].dt.year, 'product'])['num_sold'].sum()

# Convert to table format
sales_by_product_year_table = sales_by_product_year.unstack()  # Creates columns for products
sales_by_product_year_table.index.name = 'Year'
sales_by_product_year_table.columns.name = 'Product'
sales_by_product_year_table.fillna(0, inplace=True)  # Fill missing products with 0
sales_by_product_year_table



sales_by_date = train.groupby('date')['num_sold'].sum()
sales_by_date.plot(figsize=(12, 6), title='Total Sales by Date')



sales_by_date = train.groupby('date')['num_sold'].sum().reset_index()
sales_by_date.sort_values(by='num_sold', ascending=False).head(10)  # Top 10 spikes



import matplotlib.pyplot as plt

train['year'] = train['date'].dt.year
for year in train['year'].unique():
    yearly_sales = sales_by_date[sales_by_date['date'].dt.year == year]
    plt.plot(yearly_sales['date'], yearly_sales['num_sold'], label=str(year))

plt.title('Sales by Date with Yearly Trends')
plt.legend()
plt.show()



train.duplicated().sum()


train_encoded = pd.get_dummies(train, columns=['country','store','product'], drop_first=False)
test_encoded = pd.get_dummies(test, columns=['country', 'store','product'], drop_first=False)



#  'date' column is in datetime format
train_encoded['date'] = pd.to_datetime(train['date'], errors='coerce')

# Extract Month from 'date'
train_encoded['month'] = train['date'].dt.month

# Extract Day of the Week from 'date'
train_encoded['days_week'] = train['date'].dt.dayofweek  # Monday=0, Sunday=6
# Extract year
train_encoded['year']= train['date'].dt.year
# extract day of month
train_encoded['day_of_month']= train['date'].dt.day
train_encoded['quarter'] = train['date'].dt.quarter
train_encoded['week'] = train['date'].dt.isocalendar().week
train_encoded['day']=train['date'].dt.day


train_encoded.drop('date', axis=1, inplace=True)
test_encoded.drop('date', axis=1, inplace=True)


train_encoded['num_sold'] = np.log1p(train['num_sold'])



print(train_encoded.columns)



print(train_encoded[[ 'month', 'days_week','year']].head())



from sklearn.model_selection import train_test_split
features = [ 'year', 'month','quarter','week','day_of_week', 'country_Finland', 'country_Italy', 'country_Kenya', 'country_Norway', 'country_Singapore','store_Discount Stickers', 'store_Premium Sticker Mart', 'store_Stickers for Less','product_Holographic Goose', 'product_Kaggle', 'product_Kaggle Tiers', 'product_Kerneler', 'product_Kerneler Dark Mode']
X = train_encoded[features]
y = train_encoded['num_sold']

X_train_encoded , X_val_encoded, y_train, y_val = train_test_split(train_encoded[features], train_encoded['num_sold'],test_size = 0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('model', XGBRegressor(
        n_estimators=3000,
        learning_rate=0.00990161328639894,
        max_depth=17,
        min_child_weight=58,
        subsample=0.7337527286687829,
        colsample_bytree=0.4544157822113165,
        gamma=0.001976067190765828,
        reg_alpha=0.7647218923252306,  # Ensure CUDA is supported, or replace/remove
        random_state=0,
        early_stopping_rounds=200
    ))
])


 #Fit the pipeline
my_pipeline.named_steps['model'].fit(
    X_train_encoded, y_train,
    eval_set=[(X_val_encoded, y_val)],  # Pass validation data for early stopping
    eval_metric="rmse",  # Optional: You can change the evaluation metric
    verbose=True  # Logs the training progress
)

# Predict the validation set
predictions = my_pipeline.named_steps['model'].predict(X_val_encoded)


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_val, predictions)

print("Mean Absolute Error (MAE):", mae)


from sklearn.metrics import mean_absolute_percentage_error

# Calculate MAE
mae = mean_absolute_error(y_val, predictions)
print("Mean Absolute Error (MAE):", mae)

# Calculate MAPE
mape = mean_absolute_percentage_error(y_val, predictions)
print("Mean Absolute Percentage Error (MAPE):", mape * 100, "%")



print(test.columns)



import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Assuming X_train_encoded and features are defined earlier
# For example:
# X_train_encoded = pd.get_dummies(X_train, columns=['country', 'store', 'product'], drop_first=False)
# features = X_train_encoded.columns.tolist()
# y_train = np.log1p(y_train)  # If applying log transformation

# Define and train your pipeline (ensure this is done before making predictions)
# my_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('xgb', XGBRegressor(...))  # include your trained parameters
# ])
# my_pipeline.fit(X_train_encoded, y_train)

# 1. Load and preprocess test data
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

# Convert 'date' to datetime
test['date'] = pd.to_datetime(test['date'], errors='coerce')

# Handle any NaT values
if test['date'].isnull().any():
    test['date'].fillna(pd.Timestamp('2020-01-01'), inplace=True)  # Example strategy

# Extract features
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day_of_week'] = test['date'].dt.dayofweek
test['quarter'] = test['date'].dt.quarter
test['week'] = test['date'].dt.isocalendar().week

# One-hot encode categorical variables
test_encoded = pd.get_dummies(test, columns=['country', 'store', 'product'], drop_first=False)

# Ensure all dummy columns are present and align with training features
test_encoded = test_encoded.reindex(columns=features, fill_value=0)

# Align test data with training features
X_test = test_encoded[features]

# 2. Make predictions
test_predictions_log = my_pipeline.predict(X_test)

# Inverse transform if log1p was used during training
test_predictions = np.expm1(test_predictions_log)

# Ensure no negative predictions
test_predictions = np.maximum(test_predictions, 0)

# Optionally convert to integer if required
test_predictions = test_predictions.astype(int)

# 3. Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'num_sold': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("submission.csv created!")

# 4. Optional: Validate submission
print(submission.head())
print(submission.shape)



test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
print(test['id'].head())
print(submission['id'].head())


