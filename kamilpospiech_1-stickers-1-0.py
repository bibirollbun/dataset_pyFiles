# Importing basic modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset

train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


# Data statistics

print(train.head())
print(train.info())
print(train.describe())


# Checking unique features

unique_dates = list(train['date'].unique())
unique_countries = list(train['country'].unique())
unique_stores = list(train['store'].unique())
unique_products = list(train['product'].unique())

print(f"Dates: {len(unique_dates)}\n")
print(f"Countries: {unique_countries}\n")
print(f"Stores: {unique_stores}\n")
print(f"Products: {unique_products}")


# Converting date to datetime

train['date'] = pd.to_datetime(train['date'])

train.info()


# Checking target variable (num_sold)

target_distribution = train['num_sold'].value_counts()

# Bar plot
plt.figure(figsize=(8,6))
sns.barplot(x=target_distribution.index, y=target_distribution.values, palette='viridis')
plt.title('Sales distribution')
plt.xlabel('Sold volume')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--',alpha=0.7)
plt.show()

# Checking the missing values
missing_values = train['num_sold'].isna().sum()
print(f"Missing {missing_values} target values")


# Extracting some useful features from the date

# Year, month, day
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day

# Day of the week
train['day_of_week'] = train['date'].dt.dayofweek

day_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday',
}

train['day_of_week'] = train['day_of_week'].map(day_mapping)

# Dropping the date
train = train.drop(columns='date')

print(train.head(10))


# Missing values

print("Missing values: \n", train.isna().sum())

# Duplicates

print("\nDuplicates: \n", train.duplicated().sum())


# Analyze sales

# Grouping to see the sales by country, store, product, and year
sales_analysis = train.groupby(['product', 'country', 'store', 'year'])['num_sold'].mean().reset_index()

# Sorting the values
sales_analysis = sales_analysis.sort_values(by=['product', 'country', 'store', 'year'])

print(sales_analysis.head(20))


# Pivot table for better visualization
sales_pivot = sales_analysis.pivot_table(
    index=['product', 'country', 'store'],
    columns='year',
    values='num_sold',
    aggfunc='mean'
)

print(sales_pivot)


agg_sales = (
    train.groupby(['product', 'country', 'year', 'store'])['num_sold']
    .mean()
    .reset_index()
)

plt.figure(figsize=(15,10))

# Product sales per country
sns.barplot(
    data=agg_sales,
    x='country',
    y='num_sold',
    hue='product',
    errorbar=None
)
plt.title('Mean product sales per country')
plt.ylabel('Mean sales')
plt.xticks(rotation=45)
plt.legend(title='Product')
plt.tight_layout()
plt.show()

# Product sales per year
plt.figure(figsize=(15,10))
sns.lineplot(
    data=agg_sales,
    x='year',
    y='num_sold',
    hue='product',
    style='country',
    markers=True,
    dashes=False
)
plt.title('Mean product sales per year by country')
plt.ylabel('Mean sales')
plt.xticks(rotation=45)
plt.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Product sales per store
plt.figure(figsize=(15,10))
sns.barplot(
    data=agg_sales,
    x='store',
    y='num_sold',
    hue='product',
    errorbar=None
)
plt.title('Mean product sales per store')
plt.ylabel('Mean sales')
plt.xticks(rotation=45)
plt.legend(title='Product')
plt.tight_layout()
plt.show()


# Filling the NaNs with the mean value of given product sales per store
# If the combination doesn't exist, it should be replaced with 0

# Mean sales for each product, store, and country
mean_sales = train.groupby(['store', 'product'])['num_sold'].mean().to_dict()

def fill_nan(row):
    if pd.isna(row['num_sold']):
        return mean_sales.get((row['store'], row['product']), 0)
    return row['num_sold']

train['num_sold'] = train.apply(fill_nan, axis=1)
train['num_sold'] = train['num_sold'].astype('int')
 
print(train['num_sold'].isna().sum())


train.info()


# Let's check how the product was selling per day of the week

sales_per_day = train.groupby(['product', 'day_of_week'])['num_sold'].mean().reset_index()

plt.figure(figsize=(12,8))
sns.barplot(data=sales_per_day, x='day_of_week', y='num_sold', hue='product')
plt.title('Mean product sales per day of the week')
plt.xlabel('Day of the week')
plt.ylabel('Mean sales')
plt.xticks(rotation=45)
plt.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


sales_per_day_of_month = train.groupby(['product', 'day'])['num_sold'].mean().reset_index()

plt.figure(figsize=(12,8))
sns.barplot(data=sales_per_day_of_month, x='day', y='num_sold', hue='product')
plt.title('Mean product sales per day of the month')
plt.xlabel('Day of the month')
plt.ylabel('Mean sales')
plt.xticks(rotation=45)
plt.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


sales_per_month = train.groupby(['product', 'month'])['num_sold'].mean().reset_index()

plt.figure(figsize=(12,8))
sns.barplot(data=sales_per_month, x='month', y='num_sold', hue='product')
plt.title('Mean product sales per month')
plt.xlabel('Month')
plt.ylabel('Mean sales')
plt.xticks(rotation=45)
plt.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


sales_per_year = train.groupby(['product', 'year'])['num_sold'].mean().reset_index()

plt.figure(figsize=(12,8))
sns.barplot(data=sales_per_year, x='year', y='num_sold', hue='product')
plt.title('Mean product sales per year')
plt.xlabel('Year')
plt.ylabel('Mean sales')
plt.xticks(rotation=45)
plt.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# One-hot encoding the labels and trying to run the first random forest model

# Modules
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


# Split data
X = train.drop(columns=['num_sold'])
y = train['num_sold']

# Categorical and numerical columns
categorical_columns = ['country', 'store', 'product', 'day_of_week']
numerical_columns = [col for col in X.columns if col not in categorical_columns]

# One-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'
)

# Transform features
X_encoded = preprocessor.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

#MAPE
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
print(f"Linear Regressiono MAPE: {mape_lr:.2%}")


# Random Forest
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

#MAPE
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)
print(f"Random Forest MAPE: {mape_rf:.2%}")


# Running the RF model on the proper test data provided in the competition

test.head()


# Preprocessing the data in the same way as trained

# Date
test['date'] = pd.to_datetime(test['date'])


test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day

test['day_of_week'] = test['date'].dt.dayofweek

day_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday',
}

test['day_of_week'] = test['day_of_week'].map(day_mapping)

test = test.drop(columns='date')

# # NaNs handling
# mean_sales_test = test.groupby(['store', 'product'])['num_sold'].mean().to_dict()

# def fill_nan(row):
#     if pd.isna(row['num_sold']):
#         return mean_sales_test.get((row['store'], row['product']), 0)
#     return row['num_sold']

# test['num_sold'] = test.apply(fill_nan, axis=1)
# test['num_sold'] = test['num_sold'].astype('int')
 
# print(test['num_sold'].isna().sum())


# Preprocess the test data

X_test_final = preprocessor.transform(test)


# Make RF predictions
test_predictions = rf_model.predict(X_test_final)

# Add predictions to the dataframe
test['num_sold_predicted'] = test_predictions


test.head(20)


submission = pd.DataFrame({
    'id': test['id'],
    'num_sold': test['num_sold_predicted']
})

submission.to_csv('/kaggle/working/submission.csv', index=False)

