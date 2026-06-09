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

# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
training_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')

# Explore the datasets
print(train_data.head())
print(train_data.info())
print(test_data.head())
print(test_data.info())
print(training_extra.head())
print(training_extra.info())
print(sample_submission.head())



# Handle missing values in train_data
train_data['Brand'].fillna('Unknown', inplace=True)
train_data['Material'].fillna('Unknown', inplace=True)
train_data['Size'].fillna('Unknown', inplace=True)
train_data['Laptop Compartment'].fillna('No', inplace=True)
train_data['Waterproof'].fillna('No', inplace=True)
train_data['Style'].fillna('Unknown', inplace=True)
train_data['Color'].fillna('Unknown', inplace=True)
train_data['Weight Capacity (kg)'].fillna(train_data['Weight Capacity (kg)'].median(), inplace=True)

# Repeat the same preprocessing for the test and extra training data
test_data['Brand'].fillna('Unknown', inplace=True)
test_data['Material'].fillna('Unknown', inplace=True)
test_data['Size'].fillna('Unknown', inplace=True)
test_data['Laptop Compartment'].fillna('No', inplace=True)
test_data['Waterproof'].fillna('No', inplace=True)
test_data['Style'].fillna('Unknown', inplace=True)
test_data['Color'].fillna('Unknown', inplace=True)
test_data['Weight Capacity (kg)'].fillna(test_data['Weight Capacity (kg)'].median(), inplace=True)

training_extra['Brand'].fillna('Unknown', inplace=True)
training_extra['Material'].fillna('Unknown', inplace=True)
training_extra['Size'].fillna('Unknown', inplace=True)
training_extra['Laptop Compartment'].fillna('No', inplace=True)
training_extra['Waterproof'].fillna('No', inplace=True)
training_extra['Style'].fillna('Unknown', inplace=True)
training_extra['Color'].fillna('Unknown', inplace=True)
training_extra['Weight Capacity (kg)'].fillna(training_extra['Weight Capacity (kg)'].median(), inplace=True)

# Verify the changes
print(train_data.isnull().sum())
print(test_data.isnull().sum())
print(training_extra.isnull().sum())



from sklearn.preprocessing import OneHotEncoder

# Categorical columns to encode
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Initialize OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False)

# Fit and transform the train data
encoded_train = pd.DataFrame(encoder.fit_transform(train_data[categorical_columns]))
encoded_train.columns = encoder.get_feature_names_out(categorical_columns)

# Join the encoded columns back to the train data
train_data = train_data.drop(categorical_columns, axis=1)
train_data = train_data.join(encoded_train)

# Repeat for the test and training_extra data
encoded_test = pd.DataFrame(encoder.transform(test_data[categorical_columns]))
encoded_test.columns = encoder.get_feature_names_out(categorical_columns)
test_data = test_data.drop(categorical_columns, axis=1)
test_data = test_data.join(encoded_test)

encoded_extra = pd.DataFrame(encoder.transform(training_extra[categorical_columns]))
encoded_extra.columns = encoder.get_feature_names_out(categorical_columns)
training_extra = training_extra.drop(categorical_columns, axis=1)
training_extra = training_extra.join(encoded_extra)

# Verify the changes
print(train_data.head())
print(test_data.head())
print(training_extra.head())



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Split features and target variable
X = train_data.drop(['id', 'Price'], axis=1)
y = train_data['Price']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_valid)

# Evaluate the model
mae = mean_absolute_error(y_valid, y_pred)
print(f'RF Mean Absolute Error: {mae}')



from sklearn.linear_model import LinearRegression

# Initialize and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_linear = linear_model.predict(X_valid)
mae_linear = mean_absolute_error(y_valid, y_pred_linear)
print(f'Linear Regression MAE: {mae_linear}')



# Define X_test
X_test = test_data.drop(['id'], axis=1


from sklearn.tree import DecisionTreeRegressor

# Initialize and train the Decision Tree Regressor
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_tree = tree_model.predict(X_valid)
mae_tree = mean_absolute_error(y_valid, y_pred_tree)
print(f'Decision Tree Regressor MAE: {mae_tree}')



import lightgbm as lgb

# Initialize and train the LightGBM Regressor
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_lgb = lgb_model.predict(X_valid)
mae_lgb = mean_absolute_error(y_valid, y_pred_lgb)
print(f'LightGBM Regressor MAE: {mae_lgb}')



import xgboost as xgb

# Initialize and train the XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_xgb = xgb_model.predict(X_valid)
mae_xgb = mean_absolute_error(y_valid, y_pred_xgb)
print(f'XGBoost Regressor MAE: {mae_xgb}')



from catboost import CatBoostRegressor

# Initialize and train the CatBoost Regressor
catboost_model = CatBoostRegressor(random_state=42, verbose=0)
catboost_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_catboost = catboost_model.predict(X_valid)
mae_catboost = mean_absolute_error(y_valid, y_pred_catboost)
print(f'CatBoost Regressor MAE: {mae_catboost}')



from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}

# Initialize the LightGBM Regressor
lgb_model = lgb.LGBMRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters and best score
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best MAE: {-grid_search.best_score_}')



# Initialize the LightGBM Regressor with the best parameters
lgb_model = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=100, num_leaves=31, random_state=42)

# Train the model on the training data
lgb_model.fit(X_train, y_train)

# Predict the prices for the validation set
y_pred_lgb_tuned = lgb_model.predict(X_valid)
mae_lgb_tuned = mean_absolute_error(y_valid, y_pred_lgb_tuned)
print(f'Tuned LightGBM Regressor MAE: {mae_lgb_tuned}')

# Predict the prices for the test set using the tuned LightGBM model
test_predictions_tuned = lgb_model.predict(X_test)

# Create the submission file
submission_tuned = pd.DataFrame({'id': test_data['id'], 'Price': test_predictions_tuned})
submission_tuned.to_csv('submission_tuned.csv', index=False)

# Verify the submission file
print(submission_tuned.head())





