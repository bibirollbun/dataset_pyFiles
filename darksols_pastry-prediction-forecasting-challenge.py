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
import requests
from bs4 import BeautifulSoup
from io import StringIO
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import openpyxl
import plotly.graph_objects as go
from matplotlib.sankey import Sankey
from xgboost import XGBRegressor


pastry_train = "/kaggle/input/pastry-prediction/train.csv"
pastry_test = "/kaggle/input/pastry-prediction/test.csv"


pastry_train_df = pd.read_csv(pastry_train)


pastry_test_df = pd.read_csv(pastry_test)


numeric_cols_train = pastry_train_df.select_dtypes(include=['number']).columns
print("Numeric Columns in Train:", numeric_cols_train)


categorical_cols_train = pastry_train_df.select_dtypes(include=['object']).columns
print("Categorical Columns in Train:", categorical_cols_train)


numeric_cols_test = pastry_test_df.select_dtypes(include=['number']).columns
print("Numeric Columns in Test:", numeric_cols_test)


categorical_cols_test = pastry_test_df.select_dtypes(include=['object']).columns
print("Categorical Columns in Test:", categorical_cols_test)


# Convert all categorical columns to numeric columns in both datasets
for df in [pastry_train_df, pastry_test_df]:
    df['date'] = pd.to_datetime(df['date'])  # Convert to datetime
    df['store_num'] = df['store'].astype('category').cat.codes
    df['is_state_holiday_num'] = df['is_state_holiday'].astype('category').cat.codes
    df['is_school_holiday_num'] = df['is_school_holiday'].astype('category').cat.codes
    df['is_special_day_num'] = df['is_special_day'].astype('category').cat.codes


print (pastry_train_df.min())


print (pastry_test_df.min())


print (pastry_train_df.max())


print (pastry_test_df.max())


# Finding the mode of temperature columns in sales_train_df
train_temp_mode = pastry_train_df[['temperature_max', 'temperature_min', 'temperature_mean']].mode().iloc[0]

# Finding the mode of temperature columns in sales_test_df
test_temp_mode = pastry_test_df[['temperature_max', 'temperature_min', 'temperature_mean']].mode().iloc[0]

# Display the modes
print("Mode of temperature columns in pastry_train_df:\n", train_temp_mode)
print("\nMode of temperature columns in pastry_test_df:\n", test_temp_mode)


print(pastry_train_df["store"].value_counts())


print(pastry_test_df["store"].value_counts())


# print (pastry_test_df["is_special_day"].tolist())
# print (pastry_test_df["is_school_holiday"].tolist())
# print (pastry_test_df["is_state_holiday"].tolist())


plt.figure(figsize=(10, 6))  # Adjust figure size if needed
ax = sns.countplot(x=pastry_train_df["is_special_day"])

# Add labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xticks(rotation=45)  # Rotate labels if necessary
plt.title("Category Distribution of Special Day in Train")
plt.show()


plt.figure(figsize=(10, 6))  # Adjust figure size if needed
ax = sns.countplot(x=pastry_train_df["is_school_holiday"])

# Add labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xticks(rotation=45)  # Rotate labels if necessary
plt.title("Category Distribution of School Holiday in Train")
plt.show()


plt.figure(figsize=(10, 6))  # Adjust figure size if needed
ax = sns.countplot(x=pastry_train_df["is_state_holiday"])

# Add labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xticks(rotation=45)  # Rotate labels if necessary
plt.title("Category Distribution of State Holiday in Test")
plt.show()


plt.figure(figsize=(10, 6))  # Adjust figure size if needed
ax = sns.countplot(x=pastry_test_df["is_state_holiday"])

# Add labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xticks(rotation=45)  # Rotate labels if necessary
plt.title("Category Distribution of State Holiday in Test")
plt.show()


plt.figure(figsize=(10, 6))  # Adjust figure size if needed
ax = sns.countplot(x=pastry_test_df["is_school_holiday"])

# Add labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xticks(rotation=45)  # Rotate labels if necessary
plt.title("Category Distribution of School Holiday in Test")
plt.show()


plt.figure(figsize=(10, 6))  # Adjust figure size if needed
ax = sns.countplot(x=pastry_test_df["is_special_day"])

# Add labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xticks(rotation=45)  # Rotate labels if necessary
plt.title("Category Distribution of Special Day in Test")
plt.show()


print (pastry_test_df['precipitation_sum'].value_counts())


print (pastry_train_df['precipitation_sum'].value_counts())


# Check the correlation between sunshine and sales in Train
discount_precipitation_corr = pastry_train_df[['precipitation_sum', 'sales']].corr()
print(discount_precipitation_corr)


print (pastry_train_df['sunshine_sum'].value_counts())


print (pastry_test_df['sunshine_sum'].value_counts())


# Check the correlation between sunshine and sales in Train
discount_sunshine_corr = pastry_train_df[['sunshine_sum', 'sales']].corr()
print(discount_sunshine_corr)


# Check the correlation between store and sales in Train
store_sales_corr_train = pastry_train_df[['store_num', 'sales']].corr()
print(store_sales_corr_train)


# Check the correlation between state holiday and sales in Train
state_holiday_corr_train = pastry_train_df[['is_state_holiday_num', 'sales']].corr()
print(state_holiday_corr_train)


# Check the correlation between school holiday and sales in Train
school_holiday_corr_train = pastry_train_df[['is_school_holiday_num', 'sales']].corr()
print(school_holiday_corr_train)


# Check the correlation between special day and sales in Train
special_day_corr_train = pastry_train_df[['is_special_day_num', 'sales']].corr()
print(special_day_corr_train)


# Check the correlation between special day and precipitation in Train
precipitation_corr_train = pastry_train_df[['is_special_day_num', 'precipitation_sum']].corr()
print(precipitation_corr_train)


# Check the correlation between special day and sunshine in Train
sunshine_corr_train = pastry_train_df[['is_special_day_num', 'sunshine_sum']].corr()
print(sunshine_corr_train)


for df in [pastry_train_df, pastry_test_df]:
    df.drop(columns=['store', 'is_state_holiday', 'is_school_holiday', 'is_special_day'], inplace=True)


duplicates_pastry_train = pastry_train_df.duplicated()
duplicates_pastry_test = pastry_test_df.duplicated()


print (duplicates_pastry_train.sum())


print (duplicates_pastry_test.sum())


pastry_train_df.drop_duplicates()
print (f"After removing fully duplicated rows from pastry_train_df: {pastry_train_df.shape[0]} rows")
pastry_test_df.drop_duplicates()
print (f"After removing fully duplicated rows from pastry_test_df: {pastry_test_df.shape[0]} rows")


print(pastry_train_df.isnull().sum())


pastry_train_df['unsold'] = pastry_train_df['unsold'].fillna("0")


pastry_train_df['ordered'] = pastry_train_df['ordered'].fillna("0")


print(pastry_train_df.isnull().sum())


print(pastry_test_df.isnull().sum())


# 2021
weekend_dates_2021 = [
    # August
    "2021-08-01", "2021-08-07", "2021-08-08", "2021-08-14", "2021-08-15",
    "2021-08-21", "2021-08-22", "2021-08-28", "2021-08-29",

    # September
    "2021-09-04", "2021-09-05", "2021-09-11", "2021-09-12", "2021-09-18",
    "2021-09-19", "2021-09-25", "2021-09-26",

    # October
    "2021-10-02", "2021-10-03", "2021-10-09", "2021-10-10", "2021-10-16",
    "2021-10-17", "2021-10-23", "2021-10-24", "2021-10-30", "2021-10-31",

    # November
    "2021-11-06", "2021-11-07", "2021-11-13", "2021-11-14", "2021-11-20",
    "2021-11-21", "2021-11-27", "2021-11-28",

    # December
    "2021-12-04", "2021-12-05", "2021-12-11", "2021-12-12", "2021-12-18",
    "2021-12-19", "2021-12-25", "2021-12-26"
]


weekend_dates_2022 = [
    # January
    "2022-01-01", "2022-01-02", "2022-01-08", "2022-01-09", "2022-01-15",
    "2022-01-16", "2022-01-22", "2022-01-23", "2022-01-29", "2022-01-30",

    # February
    "2022-02-05", "2022-02-06", "2022-02-12", "2022-02-13", "2022-02-19",
    "2022-02-20", "2022-02-26", "2022-02-27",

    # March
    "2022-03-05", "2022-03-06", "2022-03-12", "2022-03-13", "2022-03-19",
    "2022-03-20", "2022-03-26", "2022-03-27",

    # April
    "2022-04-02", "2022-04-03", "2022-04-09", "2022-04-10", "2022-04-16",
    "2022-04-17", "2022-04-23", "2022-04-24", "2022-04-30",

    # May
    "2022-05-01", "2022-05-07", "2022-05-08", "2022-05-14", "2022-05-15",
    "2022-05-21", "2022-05-22", "2022-05-28", "2022-05-29",

    # June
    "2022-06-04", "2022-06-05", "2022-06-11", "2022-06-12", "2022-06-18",
    "2022-06-19", "2022-06-25", "2022-06-26",

    # July
    "2022-07-02", "2022-07-03", "2022-07-09", "2022-07-10", "2022-07-16",
    "2022-07-17", "2022-07-23", "2022-07-24", "2022-07-30", "2022-07-31",

    # August
    "2022-08-06", "2022-08-07", "2022-08-13", "2022-08-14", "2022-08-20",
    "2022-08-21", "2022-08-27", "2022-08-28",

    # September
    "2022-09-03", "2022-09-04", "2022-09-10", "2022-09-11", "2022-09-17",
    "2022-09-18", "2022-09-24", "2022-09-25",
    
    # October
    "2022-10-01", "2022-10-02", "2022-10-08", "2022-10-09", "2022-10-15",
    "2022-10-16", "2022-10-22", "2022-10-23", "2022-10-29", "2022-10-30",

    # November
    "2022-11-05", "2022-11-06", "2022-11-12", "2022-11-13", "2022-11-19",
    "2022-11-20", "2022-11-26", "2022-11-27",

    # December
    "2022-12-03", "2022-12-04", "2022-12-10", "2022-12-11", "2022-12-17",
    "2022-12-18", "2022-12-24", "2022-12-25", "2022-12-31"
]


weekend_dates_2023 = [
    # January
    "2023-01-01", "2023-01-07", "2023-01-08", "2023-01-14", "2023-01-15",
    "2023-01-21", "2023-01-22", "2023-01-28", "2023-01-29",

    # February
    "2023-02-04", "2023-02-05", "2023-02-11", "2023-02-12", "2023-02-18",
    "2023-02-19", "2023-02-25", "2023-02-26",

    # March
    "2023-03-04", "2023-03-05", "2023-03-11", "2023-03-12", "2023-03-18",
    "2023-03-19", "2023-03-25", "2023-03-26",

    # April
    "2023-04-01", "2023-04-02", "2023-04-08", "2023-04-09", "2023-04-15",
    "2023-04-16", "2023-04-22", "2023-04-23", "2023-04-29", "2023-04-30",

    # May
    "2023-05-06", "2023-05-07", "2023-05-13", "2023-05-14", "2023-05-20",
    "2023-05-21", "2023-05-27", "2023-05-28",

    # June
    "2023-06-03", "2023-06-04", "2023-06-10", "2023-06-11", "2023-06-17",
    "2023-06-18", "2023-06-24", "2023-06-25",

    # July
    "2023-07-01", "2023-07-02", "2023-07-08", "2023-07-09", "2023-07-15",
    "2023-07-16", "2023-07-22", "2023-07-23", "2023-07-29", "2023-07-30",

    # August
    "2023-08-05", "2023-08-06", "2023-08-12", "2023-08-13", "2023-08-19",
    "2023-08-20", "2023-08-26", "2023-08-27",

    # September
    "2023-09-02", "2023-09-03", "2023-09-09", "2023-09-10", "2023-09-16",
    "2023-09-17", "2023-09-23", "2023-09-24", "2023-09-30"
    
    # October
    "2023-10-01", "2023-10-07", "2023-10-08", "2023-10-14", "2023-10-15",
    "2023-10-21", "2023-10-22", "2023-10-28", "2023-10-29",

    # November
    "2023-11-04", "2023-11-05", "2023-11-11", "2023-11-12", "2023-11-18",
    "2023-11-19", "2023-11-25", "2023-11-26"
]


# 1. Properly concatenate all weekend date lists
weekend_dates_list = weekend_dates_2021 + weekend_dates_2022 + weekend_dates_2023  # Proper list concatenation

# 2. Convert to datetime format, handling errors gracefully
weekend_dates_dt = pd.to_datetime(weekend_dates_list, format='%Y-%m-%d', errors='coerce')  # Ensures correct parsing

# 3. Ensure the 'date' column in pastry_train_df is in datetime format
pastry_train_df['date'] = pd.to_datetime(pastry_train_df['date'], format='%Y-%m-%d', errors='coerce')

# 4. Assign "Weekend" or "Workday" based on whether the date falls in the weekend list
pastry_train_df['Weekend or Workday'] = pastry_train_df['date'].apply(lambda x: "Weekend" if x in weekend_dates_dt.values else "Workday")


weekend_dates_2023_dec = [
    "2023-12-02", "2023-12-03",
    "2023-12-09", "2023-12-10",
    "2023-12-16", "2023-12-17",
    "2023-12-23", "2023-12-24",
    "2023-12-30", "2023-12-31"
]


weekend_dates_2024_jan_may = [
    # January 2024
    "2024-01-06", "2024-01-07",
    "2024-01-13", "2024-01-14",
    "2024-01-20", "2024-01-21",
    "2024-01-27", "2024-01-28",
    
    # February 2024
    "2024-02-03", "2024-02-04",
    "2024-02-10", "2024-02-11",
    "2024-02-17", "2024-02-18",
    "2024-02-24", "2024-02-25",
    
    # March 2024
    "2024-03-02", "2024-03-03",
    "2024-03-09", "2024-03-10",
    "2024-03-16", "2024-03-17",
    "2024-03-23", "2024-03-24",
    "2024-03-30", "2024-03-31",
    
    # April 2024
    "2024-04-06", "2024-04-07",
    "2024-04-13", "2024-04-14",
    "2024-04-20", "2024-04-21",
    "2024-04-27", "2024-04-28",
    
    # May 2024
    "2024-05-04", "2024-05-05",
    "2024-05-11", "2024-05-12",
    "2024-05-18", "2024-05-19",
    "2024-05-25", "2024-05-26",
]



# 1. Properly concatenate all weekend date lists
weekend_dates_list_test = weekend_dates_2023_dec + weekend_dates_2024_jan_may  # Proper list concatenation

# 2. Convert to datetime format, handling errors gracefully
weekend_dates_dt_test = pd.to_datetime(weekend_dates_list_test, format='%Y-%m-%d', errors='coerce')  # Ensures correct parsing

# 3. Ensure the 'date' column in pastry_train_df is in datetime format
pastry_test_df['date'] = pd.to_datetime(pastry_test_df['date'], format='%Y-%m-%d', errors='coerce')

# 4. Assign "Weekend" or "Workday" based on whether the date falls in the weekend list
pastry_test_df['Weekend or Workday'] = pastry_test_df['date'].apply(lambda x: "Weekend" if x in weekend_dates_dt.values else "Workday")

pastry_test_df


# Define feature columns
feature_columns = ['store_num', 'is_state_holiday_num', 'is_school_holiday_num', 'is_special_day_num',
                   'temperature_max', 'temperature_min', 'temperature_mean',
                   'sunshine_sum', 'precipitation_sum'
]


# Split training data
X_train = pastry_train_df[feature_columns]
y_train = pastry_train_df['sales']


from xgboost import XGBRegressor

# Initialize and train the model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)


X_test = pastry_test_df[feature_columns]
pastry_test_df['sales'] = xgb_model.predict(X_test)  # Add predictions to test DataFrame


submission = pastry_test_df[['row_id', 'sales']]

# If ordered and un_ordered are required and available in test:
# submission['ordered'] = pastry_test_df['ordered']
# submission['un_ordered'] = pastry_test_df['un_ordered']

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file saved successfully!")

