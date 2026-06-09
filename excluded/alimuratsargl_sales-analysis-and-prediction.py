# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import warnings
warnings.filterwarnings("ignore")



train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


train.head(10)


train.info()


train.isnull().sum()


# Fill missing values based on the median of each product
train['num_sold'] = train.groupby('product')['num_sold'].transform(lambda x: x.fillna(x.median()))


train.isnull().sum()


def prep_data(df):
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df = df.drop(columns=["date", "id"])
    df = pd.get_dummies(df, columns=["country", "store", "product"], drop_first=True)
    return df


train_df_pro = prep_data(train)
test_df_pro = prep_data(test)


# Top-selling products by country
country_sales = train.groupby('country')['num_sold'].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=country_sales.index, y=country_sales.values, hue = country_sales.index , palette='viridis')
plt.title('Country-wise Total Sales')
plt.xlabel('Country')
plt.ylabel('Total Sales')
plt.show()


# Top-selling products by store
store_sales = train.groupby('store')['num_sold'].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=store_sales.index, y=store_sales.values, hue = store_sales.index ,  palette='plasma')
plt.title('Store-wise Total Sales')
plt.xlabel('Store')
plt.ylabel('Total Sales')
plt.show()


# Sales trend over time with resampling and smoothing
date_sales = train.groupby('date')['num_sold'].sum()

# Resample the data to monthly frequency, taking the sum of sales in each month (month-end)
date_sales_monthly = date_sales.resample('M').sum()

# Smooth the data with a rolling average (window size of 3 months)
date_sales_smooth = date_sales_monthly.rolling(window=3).mean()

plt.figure(figsize=(10, 6))
plt.plot(date_sales_smooth.index, date_sales_smooth.values, marker='o', color='blue')
plt.title('Total Sales Over Time (Monthly with Smoothed Trend)')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()


# Total sales by product
product_sales = train.groupby('product')['num_sold'].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=product_sales.index, y=product_sales.values, color='blue')
plt.title('Product-wise Total Sales')
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.show()


# Sales by product and store
product_store_sales = train.groupby(['product', 'store'])['num_sold'].sum().unstack()

plt.figure(figsize=(10, 6))
product_store_sales.plot(kind='bar', stacked=True, colormap='tab10')
plt.title('Product and Store-wise Total Sales')
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.legend(title='Store')
plt.show()


# Monthly total sales
monthly_sales = train.groupby('month')['num_sold'].sum()

plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o', color='green')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(monthly_sales.index)
plt.show()


# Distribution of 'country', 'store', and 'product' categories
plt.figure(figsize=(12, 6))

# By country
plt.subplot(1, 3, 1)
sns.countplot(x='country', data=train, hue='country', palette='Set2')
plt.title('Distribution of Countries')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability

# By store
plt.subplot(1, 3, 2)
sns.countplot(x='store', data=train, hue='store', palette='Set3')
plt.title('Distribution of Stores')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability

# By product
plt.subplot(1, 3, 3)
sns.countplot(x='product', data=train, hue='product', palette='Set1')
plt.title('Distribution of Products')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability

plt.tight_layout()
plt.show()


# Distribution of sales numbers using histogram and boxplot
plt.figure(figsize=(14, 6))

# Histogram
plt.subplot(1, 2, 1)
sns.histplot(train['num_sold'], kde=True, color='blue', bins=30)  # Adjust the number of bins for better visualization
plt.title('Distribution of Num Sold')
plt.xlabel('Number of Products Sold')  # Adding more descriptive labels
plt.ylabel('Frequency')

# Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x=train['num_sold'], color='red')
plt.title('Boxplot of Num Sold')
plt.xlabel('Number of Products Sold')

plt.tight_layout()
plt.show()


X = train_df_pro.drop(columns=['num_sold'])
y = train_df_pro['num_sold']


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df_pro)


rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_val_pred = rf_model.predict(X_val_scaled)


xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_val_pred = xgb_model.predict(X_val_scaled)


lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_val_pred = lr_model.predict(X_val_scaled)


# Function to calculate performance metrics
def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": model_name,
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }

# Calculate metrics and append them
metrics = []
metrics.append(print_metrics(y_val, rf_val_pred, "Random Forest"))
metrics.append(print_metrics(y_val, xgb_val_pred, "XGBoost"))
metrics.append(print_metrics(y_val, lr_val_pred, "Linear Regression"))

# Convert the results into a pandas DataFrame for a clean table display
metrics_df = pd.DataFrame(metrics)

# Print the results in a more organized format
print(metrics_df.to_string(index=False))


plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.scatter(y_val, rf_val_pred, color='blue', alpha=0.5)  # Add transparency to points for clarity
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', lw=2, linestyle='--')  # Dashed line for better distinction
plt.title("Random Forest Error Plot")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.grid(True)  # Add grid for better readability
plt.tight_layout()  # Ensure everything fits well

plt.show()


plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 2)
plt.scatter(y_val, xgb_val_pred, color='green', alpha=0.5)  # Add transparency to points for clarity
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', lw=2, linestyle='--')  # Dashed line for better distinction
plt.title("XGBoost Error Plot")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.grid(True)  # Add grid for better readability
plt.tight_layout()  # Ensure everything fits well

plt.show()


plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 3)
plt.scatter(y_val, lr_val_pred, color='purple', alpha=0.5)  # Add transparency to points for better clarity
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', lw=2, linestyle='--')  # Dashed line for better distinction
plt.title("Linear Regression Error Plot")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.grid(True)  # Add grid for better readability
plt.tight_layout()  # Ensure everything fits well

plt.show()


rf_test_predictions = rf_model.predict(X_test_scaled)
xgb_test_predictions = xgb_model.predict(X_test_scaled)
lr_test_predictions = lr_model.predict(X_test_scaled)

test['rf_num_sold'] = rf_test_predictions
test['xgb_num_sold'] = xgb_test_predictions
test['lr_num_sold'] = lr_test_predictions

print("\nTest Set Predictions:")
print(test[['id', 'rf_num_sold', 'xgb_num_sold', 'lr_num_sold']])


best_model = None
if r2_score(y_val, rf_val_pred) > r2_score(y_val, xgb_val_pred) and r2_score(y_val, rf_val_pred) > r2_score(y_val, lr_val_pred):
    best_model = rf_model
    best_model_name = "Random Forest"
elif r2_score(y_val, xgb_val_pred) > r2_score(y_val, rf_val_pred) and r2_score(y_val, xgb_val_pred) > r2_score(y_val, lr_val_pred):
    best_model = xgb_model
    best_model_name = "XGBoost"
else:
    best_model = lr_model
    best_model_name = "Linear Regression"

print(f"The best model is: {best_model_name}")


submission = test[['id']].copy()

# Add the predictions from the best model (based on your evaluation)
submission['num_sold'] = best_model.predict(X_test_scaled)

# Save the submission file
submission.to_csv('sales_predictions.csv', index=False)

print("Submission file created successfully!")

