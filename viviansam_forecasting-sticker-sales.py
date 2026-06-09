import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# import library
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve, roc_auc_score


df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df.info()


df.head()


# Drop the 'ID' column
df = df.drop(columns=['id'])


# Check unique value of categorical variables
# List of categorical variables
categorical_variables = [
    'country',
    'store',
    'product'
]

# Check unique values for each categorical variable
for variable in categorical_variables:
    unique_values = df[variable].unique()
    print(f"Unique values for {variable}: {unique_values}")


# Impute missing values in 'num_sold' with mean
df['num_sold'] = df['num_sold'].fillna(df['num_sold'].mean())

# Convert target from float to int 
df['num_sold'] = df['num_sold'].astype(int)


# Check min / max for target variable
min_value = df['num_sold'].min()
max_value = df['num_sold'].max()
print(f"num_sold - Min: {min_value}, Max: {max_value}")


df.info()


import holidays

# Load and preprocess date
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Define holiday calendars for each country
country_holidays = {
    'Canada': holidays.CountryHoliday('CA'),
    'Finland': holidays.CountryHoliday('FI'),
    'Italy': holidays.CountryHoliday('IT'),
    'Kenya': holidays.CountryHoliday('KE'),
    'Norway': holidays.CountryHoliday('NO'),
    'Singapore': holidays.CountryHoliday('SG'),
}

# Add a column to indicate if the date is a holiday
df['is_holiday'] = df.apply(
    lambda row: 1 if row['date'] in country_holidays.get(row['country'], []) else 0, 
    axis=1
)

# Define season 
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

# Add season column
df['season'] = df['month'].apply(get_season)

# Add cyclic features for day of year
df['day_of_year'] = df['date'].dt.dayofyear
df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)


df.info()


# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['country', 'store', 'product', 'season'], drop_first=True)
df_encoded.info()


# Data splitting
X = df_encoded.drop(columns=['num_sold', 'date']) # features
y = df_encoded['num_sold'] # target variable

# Split into at 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on test set
y_pred_lr = lr.predict(X_test)

# Calculate MAPE
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
print("MAPE for Lin Reg:", mape_lr)


# Random Forest
rf = RandomForestRegressor(random_state=123)
rf.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf.predict(X_test)

# Calculate MAPE
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)
print("MAPE for Random Forest:", mape_rf)


# XG Boost
xgb = XGBRegressor(random_state=123, learning_rate=0.1)
xgb.fit(X_train, y_train)

# Predict on test set
y_pred_xgb = xgb.predict(X_test)

# Calculate MAPE
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)
print("MAPE for XG Boost:", mape_xgb)


# Prepare to submit test set
df_test.info()


# Keep the 'ID' column separate
id_test = df_test['id']  

# Drop the 'ID' column from df_test
df_test = df_test.drop(columns=['id'])


# Load and preprocess date
df_test['date'] = pd.to_datetime(df_test['date'])
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df_test['day_of_week'] = df_test['date'].dt.dayofweek
df_test['is_weekend'] = df_test['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Add a column to indicate if the date is a holiday
df_test['is_holiday'] = df_test.apply(
    lambda row: 1 if row['date'] in country_holidays.get(row['country'], []) else 0, 
    axis=1
)

# Add season column
df_test['season'] = df_test['month'].apply(get_season)

# Add cyclic features for day of year
df_test['day_of_year'] = df_test['date'].dt.dayofyear
df_test['sin_day_of_year'] = np.sin(2 * np.pi * df_test['day_of_year'] / 365)
df_test['cos_day_of_year'] = np.cos(2 * np.pi * df_test['day_of_year'] / 365)


# One-Hot Encoding
df_test_encoded = pd.get_dummies(df_test, columns=['country', 'store', 'product', 'season'], drop_first=True)
df_test_encoded.info()


# Drop unrelated features
df_test_encoded = df_test_encoded.drop(columns=['date']) 


# Feature scaling
df_test_encoded = scaler.transform(df_test_encoded)


# Predict using Random Forest
y_pred_test_2 = rf.predict(df_test_encoded)

# Convert predictions to integers
y_pred_test_2 = y_pred_test_2.astype(int)


# Create a DataFrame with 'ID' and 'num_sold' columns
output = pd.DataFrame({'id': id_test, 'num_sold': y_pred_test_2})
output.head()


output.to_csv('submission.csv', index=False)

