# Installing packages
!pip install watermark


# Import of libraries

# System libraries
import re
import unicodedata
import itertools

# Library for file manipulation
import pandas as pd
import numpy as np
import pandas

# Data visualization
import seaborn as sns
import matplotlib.pylab as pl
import matplotlib as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib import pyplot as plt

# Configuration for graph width and layout
sns.set_theme(style='whitegrid')
palette='viridis'

# Warnings remove alerts
import warnings
warnings.filterwarnings("ignore")

# Python version
from platform import python_version
print('Python version in this Jupyter Notebook:', python_version())

# Load library versions
import watermark

# Library versions
%reload_ext watermark
%watermark -a "Library versions" --iversions


# Database
train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])

# Viewing dataset
train_df


# Viewing first 5 data
train_df.head()


# Viewing 5 latest data
train_df.tail()


# Info data
train_df.info()


# Type data
train_df.dtypes


# Viewing rows and columns
train_df.shape


print("Checking for missing values in each column:")
print(train_df.isnull().sum())


# Checking the number of null values ​​in specific columns
print(train_df[['num_sold']].isnull().sum())


# Drop 'id' column
train_df = train_df.drop(['id'], axis=1)
train_df


def remove_outliers(df, columns, min_rows=10):
    """
    Removes outliers from the specified numeric columns using the IQR method,
    ensuring that at least `min_rows` remain in the DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame from which outliers will be removed.
    columns (list): A list of column names where outliers should be removed.
    min_rows (int): The minimum number of rows that should remain in the DataFrame 
                    after removing outliers. Default is 10.
    
    Returns:
    DataFrame: A DataFrame with outliers removed from the specified columns.
    """
    
    for col in columns:
        # Attempt to convert the column to numeric, ignoring errors
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values with the median of the column (or another appropriate method)
        df[col].fillna(df[col].median(), inplace=True)
        
        # Calculate the first and third quartiles and the interquartile range (IQR)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Apply the outlier filter only if it leaves at least `min_rows` in the DataFrame
        filtered_df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        if filtered_df.shape[0] >= min_rows:
            df = filtered_df
        else:
            # Print a message if the filter would remove too many rows
            print(f"Column: {col} - Unable to apply outlier filter without removing all rows.")
    
    return df

# Applying the function to the train_df DataFrame
numeric_columns = ['num_sold']
train_df = remove_outliers(train_df, numeric_columns)

train_df.head(n=20)


# Assuming you have a function for data cleaning and outlier removal
def clean_and_remove_outliers(df):
    # Apply data cleaning and outlier removal
    # Example:
    # df = df.dropna()  # Remove missing values
    # df = df[df['column_name'] < threshold]  # Remove outliers based on a threshold
    return df

# Apply the same transformations to the test set
X_test_kaggle = clean_and_remove_outliers(test_df.drop(columns=['id']))  # Removing only the 'id' column

def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Applying outlier removal to the correct columns in the test set
# Replace with the correct column names
X_test_kaggle = remove_outliers(test_df.drop(columns=['id']), columns=['date']) 


# Checking the number of null values ​​in specific columns
print(train_df[['num_sold']].isnull().sum())


print("Checking for missing values in each column:")
print(train_df.isnull().sum())


# Copy data
data = train_df.copy()

# Salvando dataset
data.to_csv("dataset_clear.csv")


# Convert 'date' column to datetime format
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# Extract temporal features
train_df['day_of_week'] = train_df['date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
train_df['month'] = train_df['date'].dt.month
train_df['year'] = train_df['date'].dt.year
train_df['is_weekend'] = train_df['day_of_week'].isin([5, 6])  # Saturday, Sunday


# Viewing descriptive statistics
train_df.describe()


# Setting up the seaborn style
sns.set(style="whitegrid")

# Histogram for the 'price' variable
plt.figure(figsize=(12, 7))
sns.histplot(train_df['num_sold'], kde=True, color='skyblue', bins=40, alpha=0.7, line_kws={'linewidth': 2, 'color': 'blue'})
plt.title('Distribution of Car Prices', fontsize=16)
plt.xlabel('Price', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Adding mean and median lines
mean_price = train_df['num_sold'].mean()
median_price = train_df['num_sold'].median()
plt.axvline(mean_price, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:.2f}')
plt.axvline(median_price, color='green', linestyle='-', linewidth=2, label=f'Median: ${median_price:.2f}')

# Adding a legend
plt.legend()
plt.grid(False)

# Displaying the plot
plt.show()


from datetime import datetime
import holidays

# Add holiday flag (example: US holidays)
us_holidays = holidays.US()
train_df['is_holiday'] = train_df['date'].isin(us_holidays)

# Group by date to analyze trends
daily_sales = train_df.groupby('date')['num_sold'].sum().reset_index()

# Visualization 1: Time series plot of sales
daily_sales.set_index('date')['num_sold'].plot(figsize=(12, 6), title='Daily Sales Over Time')
plt.ylabel('Number Sold')
plt.grid(False)
plt.show()


# Visualization 2: Boxplot of sales by day of the week
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='day_of_week', y='num_sold',
            palette='viridis', showfliers=False)
plt.title('Sales Distribution by Day of the Week')
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.ylabel('Number Sold')
plt.show()


# Visualization 3: Heatmap of monthly sales trends
monthly_sales = train_df.groupby(['year', 'month'])['num_sold'].sum().unstack()
plt.figure(figsize=(20, 10))
sns.heatmap(monthly_sales, cmap='coolwarm', annot=True, fmt='.0f', linewidths=.5)
plt.title('Monthly Sales Heatmap')
plt.ylabel('Year')
plt.xlabel('Month')
plt.show()


# Feature Engineering: Seasonal and cyclic features
train_df['day_of_year'] = train_df['date'].dt.dayofyear
train_df['sin_day_of_year'] = np.sin(2 * np.pi * train_df['day_of_year'] / 365)
train_df['cos_day_of_year'] = np.cos(2 * np.pi * train_df['day_of_year'] / 365)

# Distribution of sales
plt.figure(figsize=(10, 6))
sns.histplot(train_df['num_sold'], kde=True, bins=30, color='blue')
plt.title('Distribution of Sales')
plt.xlabel('Number Sold')
plt.ylabel('Frequency')
plt.show()


# Correlation Heatmap
plt.figure(figsize=(10, 6))
corr_matrix = train_df[['num_sold', 'day_of_week', 'month', 'year', 'is_weekend']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# Setting up the figure and axes for subplots with tight layout
fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 3 columns

# First boxplot - Fuel Type vs Price
sns.boxplot(x="product", y="num_sold", data=train_df, palette="Set2", ax=axes[0])
axes[0].set_title('Price Distribution by Fuel Type', fontsize=14)
axes[0].set_xlabel('Fuel Type', fontsize=12)
axes[0].set_ylabel('Price (USD)', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# Third boxplot - Model Year vs Price
sns.boxplot(x="store", y="num_sold", data=train_df, palette="Set2", ax=axes[1])
axes[1].set_title('Price Distribution by Model Year', fontsize=14)
axes[1].set_xlabel('Model Year', fontsize=12)
axes[1].set_ylabel('Price (USD)', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

# Automatically adjust subplot parameters for better fit
plt.tight_layout()

# Display the plots
plt.show()


### Outlier removal

# Assuming 'train_df' is the DataFrame where you want to remove the outliers
Q1 = train_df['num_sold'].quantile(0.25)  # Calculate the first quartile (25th percentile) of the 'price' column
Q3 = train_df['num_sold'].quantile(0.75)  # Calculate the third quartile (75th percentile) of the 'price' column
IQR = Q3 - Q1  # Calculate the interquartile range (IQR) for 'price'

# Define the boundaries to consider a value as an outlier
lower_bound = Q1 - 1.5 * IQR  # Calculate the lower bound (below which values will be considered outliers)
upper_bound = Q3 + 1.5 * IQR  # Calculate the upper bound (above which values will be considered outliers)

# Filter out the outliers from the 'price' column
# Keep only the data points within the bounds
train_df = train_df[(train_df['num_sold'] >= lower_bound) & (train_df['num_sold'] <= upper_bound)]  
train_df.head(n=20)


fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Boxplot
sns.boxplot(x=train_df['num_sold'], ax=axes[0], palette="Set2")
axes[0].set_title('Boxplot of num sold - After Outlier Removal', fontsize=16)
axes[0].set_xlabel('Price', fontsize=14)
axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)  # Adding grid lines for better readability

# Histogram
sns.histplot(train_df['num_sold'], bins=40, kde=True, ax=axes[1], palette="Set2")  # Consistent color
axes[1].set_title('Histogram of num sold - After Outlier Removal', fontsize=16)
axes[1].set_xlabel('Price', fontsize=14)
axes[1].set_ylabel('Frequency', fontsize=14)
axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)  # Adding grid lines

plt.tight_layout()
plt.show()


# Function to create features
def create_date_features(df):
    train_df["year"] = train_df["date"].dt.year  # Extract year from the date column
    train_df["month"] = train_df["date"].dt.month  # Extract month from the date column
    train_df["day"] = train_df["date"].dt.day  # Extract day from the date column
    train_df["dayofweek"] = train_df["date"].dt.dayofweek  # Monday=0, Sunday=6
    train_df["weekofyear"] = train_df["date"].dt.isocalendar().week.astype(int)  # Week of the year
    
    # Weekend indicator (code added here) # adicionar
    train_df["is_weekend"] = train_df["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
    return train_df

# Apply date feature creation to train and test datasets # adicionar
train_df = create_date_features(train_df)  # adicionar
test_df = create_date_features(test_df)  # adicionar


# Importing library
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Creating the Label encoder
Label_pre = LabelEncoder()
data_cols=train_df.select_dtypes(exclude=['int','float']).columns
label_col =list(data_cols)

# Applying encoder
train_df[label_col]=train_df[label_col].apply(lambda col:Label_pre.fit_transform(col))

# Viewing
Label_pre


# Viewing dataset after applying label encoder
train_df.head()


# Selecting variables for model
X = train_df.drop('num_sold', axis=1)
y = train_df['num_sold']


# Visualizing data x
X.shape


# Visualizing data y
y.shape


# Importing libraries
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Viewing X_train rows and columns
print("Viewing X train data:", X_train.shape)
print("Viewing y train data:", y_train.shape)


%%time

import xgboost as xgb
import lightgbm as lgb
from lightgbm import log_evaluation
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# XGBoost model parameters with GPU and other adjustments
xgb_params = {'tree_method': 'gpu_hist',           # Use GPU-optimized tree method
              'predictor': 'gpu_predictor',        # Use GPU for prediction
              'objective': 'reg:squarederror',     # Objective function for regression
              'n_estimators': 1000,                # Number of trees (estimators)
              'learning_rate': 0.05,               # Learning rate
              'max_depth': 6,                      # Maximum depth of trees
              'subsample': 0.8,                    # Ratio of samples for subsampling
              'colsample_bytree': 0.8,             # Proportion of columns for subsampling
              'reg_alpha': 0.1,                    # L1 Regularization
              'reg_lambda': 0.1,                   # L2 Regularization
              'verbosity': 1                       # Verbosity level for output (1 for detailed output)
             }

# LightGBM model parameters with GPU and other adjustments
lgbm_params = {'boosting_type': 'gbdt',             # Type of boosting (Gradient Boosted Decision Trees)
               'objective': 'regression',           # Objective function for regression
               'metric': 'rmse',                    # Evaluation metric: Root Mean Squared Error (RMSE)
               'device': 'gpu',                     # Use GPU for training
               'gpu_platform_id': 0,                # GPU platform ID (set as needed)
               'gpu_device_id': 0,                  # GPU device ID (set as needed)
               'num_leaves': 31,                    # Number of leaves in the tree
               'learning_rate': 0.05,               # Learning rate
               'n_estimators': 1000,                # Number of trees (estimators)
               'max_depth': -1,                     # Maximum depth of trees (-1 means no limit)
               'min_child_samples': 20,             # Minimum number of samples in a child node
               'subsample': 0.8,                    # Ratio of samples for subsampling
               'colsample_bytree': 0.8,             # Proportion of columns for subsampling
               'reg_alpha': 0.1,                    # L1 Regularization
               'reg_lambda': 0.1,                   # L2 Regularization
               'verbose': -1                        # Verbosity level (-1 for silent)
              }

# Initializing and training the XGBoost model
xgb_model = xgb.XGBRegressor(**xgb_params)

#
xgb_model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              early_stopping_rounds=10, 
              verbose=True)

# Making predictions with the trained XGBoost model
xgb_model_pred = xgb_model.predict(X_test)

# Initializing and training the LightGBM model
lgbm_model = lgb.LGBMRegressor(**lgbm_params)

#
lgbm_model.fit(X_train, y_train,
               eval_set=[(X_test, y_test)],  
               callbacks=[log_evaluation(10)])

# Making predictions with the trained LightGBM model
lgbm_predictions = lgbm_model.predict(X_test)

# Calculating RMSE for the LightGBM model
lgbm_rmse = sqrt(mean_squared_error(y_test, lgbm_predictions))
print(f"LightGBM RMSE: {lgbm_rmse:.4f}")

# Calculating RMSE for the XGBoost model
xgb_rmse = sqrt(mean_squared_error(y_test, xgb_model_pred))
print(f"XGBoost RMSE: {xgb_rmse:.4f}")


# Plotting Feature Importance for the XGBoost Model
xgb.plot_importance(xgb_model, max_num_features=25, importance_type='weight')
plt.title('Importance of Features - XGBoost')
plt.show()


# Importing the feature importances from the LightGBM model into a DataFrame
lgbm_feature_importances = pd.DataFrame({'Feature': X_train.columns,
                                         'Importance': lgbm_model.feature_importances_})

# Sorting the features by their importance in descending order
lgbm_feature_importances = lgbm_feature_importances.sort_values(by='Importance', ascending=False)

# Plotting the top 20 most important features
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=lgbm_feature_importances.head(20))
plt.title('Feature Importance - LightGBM')
plt.show()


from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculating the metrics for the LightGBM model
lgbm_r2 = r2_score(y_test, lgbm_predictions)
lgbm_rmse = sqrt(mean_squared_error(y_test, lgbm_predictions))
lgbm_mape = mean_absolute_percentage_error(y_test, lgbm_predictions)

# Calculating the metrics for the XGBoost model
xgb_r2 = r2_score(y_test, xgb_model_pred)
xgb_rmse = sqrt(mean_squared_error(y_test, xgb_model_pred))
xgb_mape = mean_absolute_percentage_error(y_test, xgb_model_pred)

# Storing the results in a dictionary
results = {"Model": ["LightGBM", "XGBoost"],
           "R²": [lgbm_r2, xgb_r2],
           "RMSE": [lgbm_rmse, xgb_rmse],
           "MAPE (%)": [lgbm_mape, xgb_mape]}

# Converting the dictionary into a DataFrame
results_df = pd.DataFrame(results)

# Displaying the DataFrame with the results
results_df


# Ensembling (opcional)
# Combine predictions from LightGBM and XGBoost using their average
y_pred_ensemble = (lgbm_predictions + xgb_model_pred) / 2

# Calculating MAPE for the ensemble predictions
ensemble_mape = mean_absolute_percentage_error(y_test, y_pred_ensemble)

# Displaying the MAPE of the ensemble model
print(f"MAPE (Ensemble) on validation: {ensemble_mape:.4f}%")


from sklearn.metrics import mean_absolute_error
import numpy as np

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Ensemble: Averaging predictions from LightGBM and XGBoost
y_pred_ensemble = (lgbm_predictions + xgb_model_pred) / 2

# Calculating metrics for the ensemble
ensemble_rmse = sqrt(mean_squared_error(y_test, y_pred_ensemble))
ensemble_mape = mean_absolute_percentage_error(y_test, y_pred_ensemble)

# Printing the results
print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
print(f"Ensemble MAPE: {ensemble_mape:.4f}%")

# Adding ensemble results to the DataFrame
results["Model"].append("Ensemble")
results["R²"].append(r2_score(y_test, y_pred_ensemble))
results["RMSE"].append(ensemble_rmse)
results["MAPE (%)"].append(ensemble_mape)

# Converting to DataFrame
results_df = pd.DataFrame(results)

# Displaying the results
results_df


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the test dataset
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

# Create derived columns in the test dataset
test['year'] = pd.to_datetime(test['date']).dt.year
test['month'] = pd.to_datetime(test['date']).dt.month
test['day'] = pd.to_datetime(test['date']).dt.day
test['dayofweek'] = pd.to_datetime(test['date']).dt.dayofweek  # Monday=0, Sunday=6
test['is_weekend'] = test['dayofweek'].isin([5, 6]).astype(int)  # Weekend indicator
test['weekofyear'] = pd.to_datetime(test['date']).dt.isocalendar().week  # Week of the year

# Define initial feature columns
feature_cols = [
    'year', 'month', 'day', 'dayofweek', 'is_weekend', 'weekofyear', 
    'country_1', 'country_2', 'country_3', 'country_4', 'country_5', 'store_1', 
    'store_2', 'product_1', 'product_2', 'product_3', 'product_4'
]

# Adjust feature_cols to include only columns present in the test dataset
feature_cols = [col for col in feature_cols if col in test.columns]

# Create X_test (features for the test dataset)
X_test = test[feature_cols]

# Remove the 'date' column if it is not needed
if 'date' in X_test.columns:
    X_test = X_test.drop(columns=['date'])

# Encode categorical columns using Label Encoding
categorical_cols = ['country', 'store', 'product']
for col in categorical_cols:
    if col in X_test.columns:
        le = LabelEncoder()
        X_test[col] = le.fit_transform(X_test[col])

# Ensure all columns are numeric
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Ensure test has the same columns as the training set
feature_cols_train = X_train.columns.tolist()

# Add missing columns to test with a default value (e.g., 0)
for col in feature_cols_train:
    if col not in test.columns:
        test[col] = 0

# Select the feature columns from the test dataset
X_test = test[feature_cols_train]

# Reorder columns to match the training set
X_test = X_test.reindex(columns=feature_cols_train, fill_value=0)

# Remove the 'date' column again if it exists
if 'date' in X_test.columns:
    X_test = X_test.drop(columns=['date'])

# Final check to ensure all columns are numeric
X_test = X_test.apply(pd.to_numeric, errors='coerce')


# Ensure X_test has the same columns as X_train
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0  # Add missing columns with default value
for col in X_test.columns:
    if col not in X_train.columns:
        X_test = X_test.drop(columns=[col])  # Drop extra columns

# Reorder columns to match the training set
X_test = X_test[X_train.columns]

# Make predictions
y_pred_test_lgb = lgbm_model.predict(X_test)  # LightGBM
y_pred_test_xgb = xgb_model.predict(X_test)  # XGBoost

# Ensemble of predictions
y_pred_test_ensemble = (y_pred_test_lgb + y_pred_test_xgb) / 2

# Create the submission DataFrame with the correct column name 'Id'
submission = pd.DataFrame({'id': test['id'],  
                           'Prediction': y_pred_test_ensemble})

# Save the submission file
submission.to_csv('submission_ensemble.csv', index=False)
print("Submission successfully created! File saved as 'submission_ensemble.csv'.")




