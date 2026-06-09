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


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error


import warnings
warnings.filterwarnings('ignore')


%pip install wbgapi


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

display(train_df.head())
display(test_df.head())


print("Shape of train_df:", train_df.shape)
print("Shape of test_df:", test_df.shape)

print("\nInfo of train_df:")
train_df.info()

print("\nInfo of test_df:")
test_df.info()

print("\nDescription of train_df (num_sold):")
display(train_df['num_sold'].describe())

print("\nMissing values in train_df:")
display(train_df.isnull().sum())

print("\nMissing values in test_df:")
display(test_df.isnull().sum())

train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

print("\nData types after converting date columns:")
print(train_df.info())
print(test_df.info())

import matplotlib.pyplot as plt

train_sales_by_date = train_df.groupby('date')['num_sold'].sum()

plt.figure(figsize=(12, 6))
plt.plot(train_sales_by_date.index, train_sales_by_date.values)
plt.title('Total Sticker Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Number Sold')
plt.grid(True)
plt.show()


train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['day_of_week'] = train_df['date'].dt.dayofweek
train_df['day_of_year'] = train_df['date'].dt.dayofyear
train_df['week_of_year'] = train_df['date'].dt.isocalendar().week.astype(int)
train_df['quarter'] = train_df['date'].dt.quarter

test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['day_of_week'] = test_df['date'].dt.dayofweek
test_df['day_of_year'] = test_df['date'].dt.dayofyear
test_df['week_of_year'] = test_df['date'].dt.isocalendar().week.astype(int)
test_df['quarter'] = test_df['date'].dt.quarter

display(train_df.head())
display(test_df.head())


train_df["num_sold"].isna().sum()


train_df.shape


train_df['num_sold_lag_1'] = train_df.groupby(['country', 'store', 'product'])['num_sold'].shift(1).fillna(0)
train_df['num_sold_lag_7'] = train_df.groupby(['country', 'store', 'product'])['num_sold'].shift(7).fillna(0)
train_df['num_sold_lag_30'] = train_df.groupby(['country', 'store', 'product'])['num_sold'].shift(30).fillna(0)

train_df['num_sold_rolling_mean_7'] = train_df.groupby(['country', 'store', 'product'])['num_sold'].rolling(window=7).mean().reset_index(level=[0,1,2], drop=True).fillna(0)
train_df['num_sold_rolling_mean_30'] = train_df.groupby(['country', 'store', 'product'])['num_sold'].rolling(window=30).mean().reset_index(level=[0,1,2], drop=True).fillna(0)
train_df['num_sold_rolling_std_7'] = train_df.groupby(['country', 'store', 'product'])['num_sold'].rolling(window=7).std().reset_index(level=[0,1,2], drop=True).fillna(0)
train_df['num_sold_rolling_std_30'] = train_df.groupby(['country', 'store', 'product'])['num_sold'].rolling(window=30).std().reset_index(level=[0,1,2], drop=True).fillna(0)

display(train_df.head())


# Define features (X) and target (y)
features = [col for col in train_df.columns if col not in ['id', 'date', 'num_sold']]
X = train_df[features]
y = train_df['num_sold']

# Handle missing values in the target variable by dropping rows
# This is a simple approach, other strategies like imputation could be used

X = X[y.notna()]
y = y.dropna()

# Split data into training and validation sets 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical and numerical features
categorical_features = ['country', 'store', 'product']
numerical_features = [col for col in features if col not in categorical_features]

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)


# Create a pipeline with the preprocessor and a RandomForestRegressor model
model_rf = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(n_estimators= 500, random_state=42, n_jobs=-1))])

# Train the pipeline
model_rf.fit(X_train, y_train)

print("Random Forest Pipeline trained successfully!")


# Make predictions on the validation set using the Random Forest model
y_pred_rf = model_rf.predict(X_val)

# Calculate MAPE for the Random Forest model
mape_rf = mean_absolute_percentage_error(y_val, y_pred_rf)

print(f"Mean Absolute Percentage Error (MAPE) for Random Forest on the validation set: {mape_rf}")


print("Training Random Forest on full train dataset")
model_rf.fit(X, y)


df_combined = pd.concat([train_df.drop(["num_sold"], axis =1), test_df], ignore_index=True)


print("Creating continuous series for 'Last Known Value' lag initialization...")

# 1. Create imputed training sales features (X + Y)
sales_train = train_df.drop(["num_sold"], axis = 1)
sales_train['num_sold'] = train_df["num_sold"].fillna(0)

# 2. Create test sales features (X + NaN Placeholder for Y)
# Using NaN here allows the 'shift' operation to correctly use the training sales
# values that immediately precede the test set's start date.
sales_test = test_df.copy()
sales_test['num_sold'] = np.nan 

# 3. Concatenate ALL data in time-ordered sequence
# This creates the continuous series required for boundary calculations.
sales_full = pd.concat([sales_train, sales_test], ignore_index=True)

# 4. Sort strictly by group and time (essential for correct lag calculation)
sales_full['date'] = pd.to_datetime(sales_full['date'])
sales_full = sales_full.sort_values(['country', 'store', 'product', 'date']).reset_index(drop=True)

# --- 5. New Lag Feature Calculation on Full Data ---

def create_lag_features_full_clean(df_with_sales):
    """Creates ALL 7 required lag and rolling features and fills NaNs with 0."""
    df_lags = df_with_sales.copy()
    
    # --- LAG FEATURES (Matching your original training features) ---
    for lag in [1, 7, 30]:
        col_name = f'num_sold_lag_{lag}'
        # shift() works across the boundary, then fill the remaining NaNs (at series start) with 0
        df_lags[col_name] = df_lags.groupby(['country', 'store', 'product'])['num_sold'].shift(lag).fillna(0)
    
    # --- ROLLING MEAN/STD FEATURES (Matching your original training features) ---
    for window in [7, 30]:
        # Rolling mean
        df_lags[f'num_sold_rolling_mean_{window}'] = df_lags.groupby(['country', 'store', 'product'])['num_sold'].transform(
            lambda x: x.shift(1).rolling(window=window).mean()
        ).fillna(0) # Fill NaNs (at series start) with 0
        
        # Rolling standard deviation
        df_lags[f'num_sold_rolling_std_{window}'] = df_lags.groupby(['country', 'store', 'product'])['num_sold'].transform(
            lambda x: x.shift(1).rolling(window=window).std()
        ).fillna(0) # Fill NaNs (at series start) with 0
        
    # Return all 7 clean features plus the ID column
    return df_lags.filter(regex='^(id|num_sold_lag|num_sold_rolling)').copy()

# Apply the lag calculation to the full dataset
lag_features_full = create_lag_features_full_clean(sales_full) 

# --- 6. Merge the new lag features back to df_combined ---

# --- Merge the clean features back to df_combined ---

lag_features_full = lag_features_full.set_index('id')
df_combined = df_combined.set_index('id')

# CRUCIAL: Drop ALL old lag/roll columns before joining
cols_to_drop = [c for c in df_combined.columns if c.startswith('num_sold_lag') or c.startswith('num_sold_rolling') or c.startswith('sales_lag') or c.startswith('sales_roll')]
df_combined = df_combined.drop(columns=cols_to_drop, errors='ignore')

# Join the new, clean features
df_combined = df_combined.join(lag_features_full, how='left').reset_index()

print("Lag features created and merged successfully.")


# 3. Final Prediction

# Get the final X_test
train_len = len(train_df)
test_df_from_combined = df_combined.iloc[train_len:].copy()
X_test = test_df_from_combined.drop(['id', 'date'], axis = 1)

# Now, prediction should work without null errors!
test_predictions_rf = model_rf.predict(X_test)


# Round predictions to the nearest integer as sales are typically whole numbers
test_predictions_rf = np.round(test_predictions_rf).astype(int)

# Ensure no negative predictions
test_predictions_rf[test_predictions_rf < 0] = 0


# Create a submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id'], 'num_sold': test_predictions_rf})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' generated successfully!")
display(submission_df.head())

