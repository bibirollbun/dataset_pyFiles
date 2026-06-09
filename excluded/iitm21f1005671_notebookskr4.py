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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

# Feature engineering
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

for df in [train, test]:
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Add cyclic features for 'month' and 'day_of_week'
for df in [train, test]:
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Separate features (X) and target (y)
X = train.drop(columns=['num_sold', 'id', 'date', 'month', 'day_of_week'])
y = train['num_sold']

# Preprocessing pipelines
categorical_features = ['store', 'country', 'product']
numerical_features = [col for col in X.columns if col not in categorical_features]

from sklearn.preprocessing import OneHotEncoder

def impute_missing_target(y_train, X_train):
    # Find rows where target y_train is NaN
    missing_mask = y_train.isna()
    
    if missing_mask.sum() > 0:
        # Use only non-missing values for training the model
        X_train_non_missing = X_train[~missing_mask]
        y_train_non_missing = y_train[~missing_mask]
        
        # Encode categorical features
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Fit and transform the non-missing categorical data
        X_train_non_missing_encoded = encoder.fit_transform(X_train_non_missing[categorical_features])
        X_train_non_missing_combined = np.hstack(
            [X_train_non_missing_encoded, X_train_non_missing[numerical_features].values]
        )
        
        # Transform the missing categorical data
        X_train_missing_encoded = encoder.transform(X_train[missing_mask][categorical_features])
        X_train_missing_combined = np.hstack(
            [X_train_missing_encoded, X_train[missing_mask][numerical_features].values]
        )
        
        # Train a RandomForestRegressor using only non-missing data
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_non_missing_combined, y_train_non_missing)
        
        # Predict missing target values
        y_train[missing_mask] = model.predict(X_train_missing_combined)
    
    return y_train


y = impute_missing_target(y, X)
print(y.isna().sum())

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# y_val = impute_missing_target(y_val, X_val)
# test['num_sold'] = y_val
# print(test.isna().sum())


categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Build model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='neg_mean_absolute_percentage_error', verbose=2)
grid_search.fit(X_train, y_train)

# Best pipeline
best_pipeline = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Predict and evaluate
y_val_pred = best_pipeline.predict(X_val)
validation_mape = mean_absolute_percentage_error(y_val, y_val_pred)
print(f"Validation MAPE: {validation_mape:.4f}")

# Test predictions
X_test = test.drop(columns=['id', 'date', 'month', 'day_of_week'])
test_predictions = best_pipeline.predict(X_test).round().astype(int)

# Evaluate submission
# submission_mape = mean_absolute_percentage_error(test['num_sold'], test_predictions)
# print(f"Test MAPE: {submission_mape:.4f}")

# Prepare submission file
submission = test[['id']].copy()
submission['num_sold'] = test_predictions
submission.to_csv('submission.csv', index=False)

# Print sample submission
print("\nSample of submission file:")
print(submission.head())


