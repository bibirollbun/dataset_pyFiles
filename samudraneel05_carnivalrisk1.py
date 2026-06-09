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

train_df = pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/train.csv')

print("Train DataFrame:")
display(train_df.head())

print("\nTest DataFrame:")
display(test_df.head())


print("Number of duplicate rows in train_df before removal:", train_df.duplicated().sum())
train_df.drop_duplicates(inplace=True)
print("Number of duplicate rows in train_df after removal:", train_df.duplicated().sum())


print("Missing values in train_df:")
display(train_df.isnull().sum())

print("\nMissing values in test_df:")
display(test_df.isnull().sum())


# def impute_missing_values(df):
#     """
#     Imputes missing values in the DataFrame based on specified strategies.

#     Args:
#         df: pandas DataFrame.

#     Returns:
#         pandas DataFrame with imputed missing values.
#     """
#     # Impute 'Age' with the mean
#     df['Age'] = df['Age'].fillna(df['Age'].mean())

#     # Impute 'Annual Income' with the mean
#     df['Annual Income'] = df['Annual Income'].fillna(df['Annual Income'].mean())

#     # Impute 'Marital Status' with the mode
#     df['Marital Status'] = df['Marital Status'].fillna(df['Marital Status'].mode()[0])

#     # Impute 'Number of Children' based on age
#     df['Number of Children'] = df.apply(
#         lambda row: df['Number of Children'].mode()[0] if pd.isnull(row['Number of Children']) and row['Age'] > 25 else (0 if pd.isnull(row['Number of Children']) else row['Number of Children']), axis=1
#     )

#     # Impute 'Occupation' with the mode
#     df['Occupation'] = df['Occupation'].fillna(df['Occupation'].mode()[0])

#     # Impute 'Health Score' with the mean
#     df['Health Score'] = df['Health Score'].fillna(df['Health Score'].mean())

#     # Impute 'Previous Claims' with the mode
#     df['Previous Claims'] = df['Previous Claims'].fillna(df['Previous Claims'].mode()[0])

#     # Impute 'Vehicle Age' with the mean
#     df['Vehicle Age'] = df['Vehicle Age'].fillna(df['Vehicle Age'].mean())

#     # Impute 'Credit Score' with the mean
#     df['Credit Score'] = df['Credit Score'].fillna(df['Credit Score'].mean())

#     # Impute 'Insurance Duration' with the mean (only in train_df as test_df has no missing values)
#     if 'Insurance Duration' in df.columns:
#         df['Insurance Duration'] = df['Insurance Duration'].fillna(df['Insurance Duration'].mean())


#     # Impute 'Customer Feedback' with the mode
#     df['Customer Feedback'] = df['Customer Feedback'].fillna(df['Customer Feedback'].mode()[0])

#     return df

# # Apply imputation to both dataframes
# train_df = impute_missing_values(train_df)
# test_df = impute_missing_values(test_df)

import pandas as pd
import numpy as np

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiently imputes missing values using vectorized operations.
    Optimized for large DataFrames (~500k+ rows).
    """
    df = df.copy()

    # Precompute means and modes once
    means = df[['Age', 'Annual Income', 'Health Score', 'Vehicle Age', 'Credit Score',
                'Insurance Duration' if 'Insurance Duration' in df.columns else None]].mean(numeric_only=True)
    modes = df[['Marital Status', 'Number of Children', 'Occupation',
                'Previous Claims', 'Customer Feedback']].mode().iloc[0]

    # Numeric columns - use vectorized fillna
    df['Age'].fillna(means['Age'], inplace=True)
    df['Annual Income'].fillna(means['Annual Income'], inplace=True)
    df['Health Score'].fillna(means['Health Score'], inplace=True)
    df['Vehicle Age'].fillna(means['Vehicle Age'], inplace=True)
    df['Credit Score'].fillna(means['Credit Score'], inplace=True)
    if 'Insurance Duration' in df.columns:
        df['Insurance Duration'].fillna(means['Insurance Duration'], inplace=True)

    # Categorical columns - use mode
    df['Marital Status'].fillna(modes['Marital Status'], inplace=True)
    df['Occupation'].fillna(modes['Occupation'], inplace=True)
    df['Previous Claims'].fillna(modes['Previous Claims'], inplace=True)
    df['Customer Feedback'].fillna(modes['Customer Feedback'], inplace=True)

    # Number of Children logic (avoid .apply)
    num_children_mode = modes['Number of Children']
    mask_null_children = df['Number of Children'].isna()
    df.loc[mask_null_children & (df['Age'] > 25), 'Number of Children'] = num_children_mode
    df.loc[mask_null_children & (df['Age'] <= 25), 'Number of Children'] = 0

    return df


# Apply to both datasets
train_df = impute_missing_values(train_df)
test_df = impute_missing_values(test_df)

print("Missing values in train_df after imputation:")
display(train_df.isnull().sum())

print("\nMissing values in test_df after imputation:")
display(test_df.isnull().sum())

# print("Missing values in train_df after imputation:")
# display(train_df.isnull().sum())

# print("\nMissing values in test_df after imputation:")
# display(test_df.isnull().sum())


!pip install category_encoders


# Identify categorical columns
categorical_cols_train = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols_test = test_df.select_dtypes(include=['object', 'category']).columns.tolist()

print("Categorical columns in train_df:", categorical_cols_train)
print("Categorical columns in test_df:", categorical_cols_test)


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import category_encoders as ce

# Drop 'id' column
train_df.drop('id', axis=1, inplace=True)
test_df.drop('id', axis=1, inplace=True)

# Binary encoding for 'Marital Status' (assuming 'Single' is one category and others are the other)
train_df['Marital Status'] = train_df['Marital Status'].apply(lambda x: 1 if x == 'Married' else 0)
test_df['Marital Status'] = test_df['Marital Status'].apply(lambda x: 1 if x == 'Married' else 0)

# Label Encoding for 'Education Level', 'Occupation', 'Location', 'Customer Feedback', 'Smoking Status', 'Property Type'
label_encode_cols = ['Education Level', 'Occupation', 'Location', 'Customer Feedback', 'Smoking Status', 'Property Type']
for col in label_encode_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col]) # Use transform on test to avoid new categories

# Ordinal Encoding for 'Policy Type'
# Check for other tiers
print("Unique values in Policy Type:", train_df['Policy Type'].unique())
policy_order = ['Basic', 'Premium', 'Comprehensive']
oe = OrdinalEncoder(categories=[policy_order])
train_df['Policy Type'] = oe.fit_transform(train_df[['Policy Type']])
test_df['Policy Type'] = oe.transform(test_df[['Policy Type']])

# Feature Engineering 'Policy Start Date'
train_df['Policy Start Date'] = pd.to_datetime(train_df['Policy Start Date'])
test_df['Policy Start Date'] = pd.to_datetime(test_df['Policy Start Date'])

# Calculate 'time since policy bought'. Assuming 'today' is a fixed date for consistency.
# In a real scenario, you might use the current date or a reference date from the dataset.
# Let's use the latest date in the training data as a reference.
latest_date = train_df['Policy Start Date'].max()

train_df['time_since_policy_bought'] = (latest_date - train_df['Policy Start Date']).dt.days
test_df['time_since_policy_bought'] = (latest_date - test_df['Policy Start Date']).dt.days

# Drop the original 'Policy Start Date' column
train_df.drop('Policy Start Date', axis=1, inplace=True)
test_df.drop('Policy Start Date', axis=1, inplace=True)

# Target Encoding for 'Exercise Frequency' using 'Premium Amount' as the target
target_encode_col = 'Exercise Frequency'
target = 'Premium Amount'

# Ensure the target variable exists in the training data
if target in train_df.columns:
    # Initialize TargetEncoder
    encoder = ce.TargetEncoder(cols=[target_encode_col])

    # Fit and transform on the training data
    train_df[target_encode_col] = encoder.fit_transform(train_df[target_encode_col], train_df[target])

    # Transform on the testing data (use the mappings learned from the training data)
    test_df[target_encode_col] = encoder.transform(test_df[target_encode_col])
else:
    print(f"Warning: Target variable '{target}' not found in the training data. Skipping target encoding for '{target_encode_col}'.")

print("\nTrain DataFrame after encoding and feature engineering:")
display(train_df.head())

print("\nTest DataFrame after encoding and feature engineering:")
display(test_df.head())


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
correlation_matrix = train_df.corr()

# Display the correlation matrix
print("Correlation Matrix:")
display(correlation_matrix)

# Optionally, visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Training Data')
plt.show()


X = train_df.drop('Premium Amount', axis=1)
y = train_df['Premium Amount']

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor

# Initialize models with default parameters
xgb_model = xgb.XGBRegressor(random_state=42)
ada_model = AdaBoostRegressor(random_state=42)
lgbm_model = LGBMRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)

print("Initialized XGBoost Regressor:", xgb_model)
print("Initialized AdaBoost Regressor:", ada_model)
print("Initialized LightGBM Regressor:", lgbm_model)
print("Initialized Random Forest Regressor:", rf_model)


# Define hyperparameter grids for each model

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

ada_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'loss': ['linear', 'square', 'exponential']
}

lgbm_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [31, 62, 124],
    'max_depth': [-1, 10, 20],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("XGBoost Parameter Grid:", xgb_param_grid)
print("AdaBoost Parameter Grid:", ada_param_grid)
print("LightGBM Parameter Grid:", lgbm_param_grid)
print("Random Forest Parameter Grid:", rf_param_grid)


from sklearn.model_selection import KFold

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("5-fold cross-validation object:", kf)


from sklearn.model_selection import GridSearchCV

# List of models and their parameter grids
models = [
    ('XGBoost', xgb_model, xgb_param_grid),
    ('AdaBoost', ada_model, ada_param_grid),
    ('LightGBM', lgbm_model, lgbm_param_grid),
    ('RandomForest', rf_model, rf_param_grid)
]

best_estimators = {}

for name, model, param_grid in models:
    print(f"Running GridSearchCV for {name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validation score for {name}: {-grid_search.best_score_}") # Convert negative MSE to positive
    best_estimators[name] = grid_search.best_estimator_
    print("-" * 30)

print("\nBest estimators found:")
for name, estimator in best_estimators.items():
    print(f"{name}: {estimator}")



test_predictions = final_xgb_model.predict(test_df)

try:
    submission_df = pd.DataFrame({'id': test_df_original['id'], 'Premium Amount': test_predictions})
except NameError:
    print("Warning: 'test_df_original' not found. Assuming the original test data needs to be reloaded to get the 'id' column.")
    test_df_original = pd.read_csv('test.csv')
    submission_df = pd.DataFrame({'id': test_df_original['id'], 'Premium Amount': test_predictions})


print("Submission DataFrame:")
display(submission_df.head())

submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully.")

