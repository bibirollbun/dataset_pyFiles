# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train.head()


train.info()


train.shape


# Handle missing values for categorical columns by imputing with the mode (most frequent value)
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
for col in categorical_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    train_extra[col].fillna(train_extra[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)

# Handle missing values for numerical columns like 'Weight Capacity (kg)' by imputing with the mean
train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean(), inplace=True)
train_extra['Weight Capacity (kg)'].fillna(train_extra['Weight Capacity (kg)'].mean(), inplace=True)
test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean(), inplace=True)

# Check again for missing values to ensure they have been handled
print(train.isnull().sum())
print(train_extra.isnull().sum())
print(test.isnull().sum())


# One-hot encoding categorical features
train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
train_extra = pd.get_dummies(train_extra, columns=categorical_cols, drop_first=True)
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)


train_combined = pd.concat([train, train_extra], axis=0)


# Separate the features and target variable
X_combined = train_combined.drop(columns=['id', 'Price'])
y_combined = train_combined['Price']



# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Check the shapes of the resulting datasets
print(X_train.shape, X_val.shape)



# # Train an XGBoost Regressor
# xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
# xgb_model.fit(X_train, y_train)

# # Predict on validation set
# y_pred_xgb = xgb_model.predict(X_val)

# # Evaluate the model
# mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
# rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))

# print(f"XGBoost - MAE: {mae_xgb}, RMSE: {rmse_xgb}")


from sklearn.model_selection import RandomizedSearchCV

# Set the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'lambda': [0, 1, 2],
    'alpha': [0, 1, 2]
}

# Initialize the XGBRegressor
xgb_model = xgb.XGBRegressor(random_state=42)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_model, param_grid, n_iter=50, cv=3, random_state=42, n_jobs=-1, scoring='neg_root_mean_squared_error')
random_search.fit(X_train, y_train)

# Get the best model from the search
best_model = random_search.best_estimator_

# Predict on the validation set using the best model
y_pred_xgb_best = best_model.predict(X_val)

# Evaluate the improved model
mae_xgb_best = mean_absolute_error(y_val, y_pred_xgb_best)
rmse_xgb_best = np.sqrt(mean_squared_error(y_val, y_pred_xgb_best))

print(f"Improved XGBoost - MAE: {mae_xgb_best}, RMSE: {rmse_xgb_best}")



X_test = test.drop(columns=['id'])  # Drop 'id' column from test set

#Make predictions on the test set using the final model
test_predictions = xgb_model.predict(X_test)

#Create the submission file
submission = pd.DataFrame({'id': test['id'], 'Price': test_predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file has been created successfully.")







