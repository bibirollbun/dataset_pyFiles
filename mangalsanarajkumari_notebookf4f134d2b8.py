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


# Load datasets
train_df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
sample_submission = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')

# Display first few rows of each dataset
print("Train Data:")
display(train_df.head())

print("\nTest Data:")
display(test_df.head())

print("\nSample Submission:")
display(sample_submission.head())


print("Missing values in train set:")
print(train_df.isnull().sum())

print("\nMissing values in test set:")
print(test_df.isnull().sum())


print("\nTrain Data Info:")
print(train_df.info())

print("\nTrain Data Statistics:")
print(train_df.describe())





import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(train_df['price'], bins=50, kde=True)
plt.xlabel("Price")
plt.ylabel("Count")
plt.title("Distribution of Prices")
plt.show()


upper_limit = train_df['price'].quantile(0.99)  # 99th percentile
train_df = train_df[train_df['price'] <= upper_limit]


import numpy as np  

train_df['log_price'] = np.log1p(train_df['price'])  # log1p avoids log(0) errors

plt.figure(figsize=(8, 5))
sns.histplot(train_df['log_price'], bins=50, kde=True)
plt.xlabel("Log Price")
plt.ylabel("Count")
plt.title("Log-Transformed Distribution of Prices")
plt.show()


plt.figure(figsize=(8, 4))
sns.boxplot(x=train_df['price'])
plt.title("Box Plot of Prices")
plt.show()


# Convert date column to datetime format
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# Extract date-related features
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['day_of_week'] = train_df['date'].dt.dayofweek

test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['day_of_week'] = test_df['date'].dt.dayofweek

# Drop the original date column (if not needed)
train_df.drop(columns=['date'], inplace=True)
test_df.drop(columns=['date'], inplace=True)


train_df.head()


test_df.head()


from supplemental_english import REGION_CODES

# Extract last 2 or 3 digits from plate number (region code)
train_df['region_code'] = train_df['plate'].str.extract(r'(\d{2,3})$').astype(float)
test_df['region_code'] = test_df['plate'].str.extract(r'(\d{2,3})$').astype(float)


from supplemental_english import GOVERNMENT_CODES

# Create a new feature: 1 if plate is a government plate, 0 otherwise
train_df['is_gov_plate'] = train_df['plate'].isin(GOVERNMENT_CODES).astype(int)
test_df['is_gov_plate'] = test_df['plate'].isin(GOVERNMENT_CODES).astype(int)


# Map region codes to actual region names
train_df['region_name'] = train_df['region_code'].map(REGION_CODES)
test_df['region_name'] = test_df['region_code'].map(REGION_CODES)


region_price_mean = train_df.groupby('region_code')['price'].mean()
train_df['region_avg_price'] = train_df['region_code'].map(region_price_mean)
test_df['region_avg_price'] = test_df['region_code'].map(region_price_mean)


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
train_df['region_encoded'] = encoder.fit_transform(train_df['region_name'].astype(str))
test_df['region_encoded'] = encoder.transform(test_df['region_name'].astype(str))

# Drop text column if not needed
train_df.drop(columns=['region_name'], inplace=True)
test_df.drop(columns=['region_name'], inplace=True)


train_df[['plate', 'is_gov_plate']].head(20)


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=train_df['is_gov_plate'], y=train_df['price'])
plt.title("Price Distribution: Government vs. Regular Plates")
plt.xlabel("Government Plate (0 = No, 1 = Yes)")
plt.ylabel("Price")
plt.show()


from sklearn.model_selection import train_test_split

# Define features and target
features = ['region_code', 'year', 'month', 'day']
target = 'price'

X = train_df[features]
y = train_df[target]

# Split into train & validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Define parameter grid
param_grid = {
    'n_estimators': [300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8]
}

# Initialize GridSearchCV
grid = GridSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,  # You can change to 5 if needed
    n_jobs=-1,  # Use all CPU cores
    verbose=2
)

# Train model
grid.fit(X_train, y_train)

# Display best parameters
print(f"Best parameters: {grid.best_params_}")

# Train final model with best parameters
best_xgb = xgb.XGBRegressor(
    n_estimators=grid.best_params_['n_estimators'],
    learning_rate=grid.best_params_['learning_rate'],
    max_depth=grid.best_params_['max_depth'],
    objective='reg:squarederror',
    random_state=42
)
best_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=True)

# Predict on validation set
y_pred = best_xgb.predict(X_val)

# Evaluate model
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Absolute Error: {mae}")


# Train the final model on the entire dataset
best_xgb = xgb.XGBRegressor(
    n_estimators=grid.best_params_['n_estimators'],
    learning_rate=grid.best_params_['learning_rate'],
    max_depth=grid.best_params_['max_depth'],
    objective='reg:squarederror',
    random_state=42
)

# Fit on full training data
best_xgb.fit(X, y)


# Ensure test dataset contains the correct features
X_test = test_df[X.columns]  

# Predict prices
test_df['price'] = best_xgb.predict(X_test)


# Create submission DataFrame
submission = test_df[['id', 'price']]

# Save as CSV (without index)
submission.to_csv('submission.csv', index=False)

print("✅ Submission file 'submission.csv' created successfully!")


X_train.shape




