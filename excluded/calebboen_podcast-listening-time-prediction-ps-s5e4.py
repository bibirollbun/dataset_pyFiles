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


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# load and inspect data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
train.head()


# basic info
print("Training set info:\n")
train.info()


print("Testing set info:\n")
test.info()


# train null values
print("Training set number of null values:\n")
train.isnull().sum()


# test null values
print("Testing set number of null values:\n")
test.isnull().sum()


# train duplicates
train.duplicated().sum()


# test duplicates
test.duplicated().sum()


# view training set summary statistics
train.describe()


test.describe()


# Clip outlier
train['Episode_Length_minutes'] = train['Episode_Length_minutes'].clip(upper=325.24)
test['Episode_Length_minutes'] = test['Episode_Length_minutes'].clip(upper=325.24)


# Imputation
for df in [train, test]:
    df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)
    df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median(), inplace=True)
    if 'Number_of_Ads' in df.columns:
        df['Number_of_Ads'].fillna(df['Number_of_Ads'].median(), inplace=True)

# Verify
print("\nTrain Nulls Post-Imputation:\n", train.isnull().sum())
print("\nTest Nulls Post-Imputation:\n", test.isnull().sum())
print("\nTrain Describe Post-Imputation:\n", train.describe())
print("\nTest Describe Post-Imputation:\n", test.describe())


# Numerical distributions
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                  'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(train[col], bins=50, kde=True)
    plt.title(f'{col} Distribution (Train)')
plt.tight_layout()
plt.show()


# Scatter: Target vs. Episode length by Genre
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train, x='Episode_Length_minutes', y='Listening_Time_minutes', hue='Genre', alpha=0.5)
plt.title('Listening Time vs. Episode Length by Genre')
plt.show()


# Categorical box plots
categorical_cols = ['Genre', 'Publication_Day', 'Episode_Sentiment']
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=train, x=col, y='Listening_Time_minutes')
    plt.xticks(rotation=45)
    plt.title(f'Listening Time by {col}')
plt.tight_layout()
plt.show()


# Correlation matrix
corr = train[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix (Train)')
plt.show()


# Drop irrelevant columns
train = train.drop(columns=['id', 'Podcast_Name', 'Episode_Title'])
test_ids = test['id']  # Save for submission
test = test.drop(columns=['id', 'Podcast_Name', 'Episode_Title'])


# Categorical encoding
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)


# Align train and test columns
train, test = train.align(test, join='left', axis=1, fill_value=0)


# Numerical scaling
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                  'Guest_Popularity_percentage', 'Number_of_Ads']
scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])


# Target transformation
train['Listening_Time_minutes'] = np.log1p(train['Listening_Time_minutes'])

# Output
print("Train Shape:", train.shape)
print("\nTrain Head:\n", train.head())
print("\nTest Shape:", test.shape)
print("\nTest Head:\n", test.head())


# Feature engineering
for df in [train, test]:
    # Host to Guest Popularity Ratio (avoid division by 0)
    df['Host_to_Guest_Popularity'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1e-6)
    # Ads per Minute (avoid division by 0)
    df['Ads_per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1e-6)
    # Length Category (bins based on quartiles)
    df['Length_Category'] = pd.cut(df['Episode_Length_minutes'], 
                                   bins=[-float('inf'), -0.5, 0.5, float('inf')], 
                                   labels=['Short', 'Medium', 'Long'])


# Encode Length_Category
train = pd.get_dummies(train, columns=['Length_Category'], drop_first=True)
test = pd.get_dummies(test, columns=['Length_Category'], drop_first=True)


# Drop Listening_Time_minutes from test (artifact)
test = test.drop(columns=['Listening_Time_minutes'])


# Scale new numerical features
new_numerical = ['Host_to_Guest_Popularity', 'Ads_per_Minute']
scaler = StandardScaler()
train[new_numerical] = scaler.fit_transform(train[new_numerical])
test[new_numerical] = scaler.transform(test[new_numerical])


# Output
print("Train Shape:", train.shape)
print("\nTrain Head:\n", train.head())
print("\nTest Shape:", test.shape)
print("\nTest Head:\n", test.head())


# Correlation check (train only)
corr = train[['Listening_Time_minutes'] + new_numerical].corr()
print("\nNew Feature Correlations:\n", corr)


# Split data
X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
}


# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log).clip(0, 119.97)  # Back to original scale, capped
    y_val_orig = np.expm1(y_val)
    rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred))
    print(f"\n{name} RMSE: {rmse:.4f}")
    print(f"Sample y_pred (orig): {y_pred[:5]}")# Train and evaluate


# Tune XGBoost
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7],
    'learning_rate': [0.05, 0.1]
}
grid = GridSearchCV(XGBRegressor(random_state=42, n_jobs=-1), param_grid, cv=3, 
                    scoring='neg_mean_squared_error', verbose=1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred_log = best_model.predict(X_val)
y_pred = np.expm1(y_pred_log).clip(0, 119.97)
y_val_orig = np.expm1(y_val)
rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred))
print("\nBest XGBoost Params:", grid.best_params_)
print(f"Tuned XGBoost RMSE: {rmse:.4f}")
print(f"Sample y_pred (orig): {y_pred[:5]}")


# Submission with tuned model
best_model.fit(X, y)  # Full train
test_pred_log = best_model.predict(test)
test_pred = np.expm1(test_pred_log).clip(0, 119.97)
submission = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': test_pred})
submission.to_csv('submission.csv', index=False)
print("\nSubmission Head:\n", submission.head())




