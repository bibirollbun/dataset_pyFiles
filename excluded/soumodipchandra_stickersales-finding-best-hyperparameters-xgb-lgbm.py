import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
import matplotlib.pyplot as plt  
import seaborn as sns


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


# Visualistaion of the data
print("---",train.head())
print(train.describe())
print("----",train.info())
print(train.shape)


# Checking for missing values
train.isnull().sum()


# droping the missing value rows
train = train.dropna()


# Now we will marge the train and test data in num_sold of test i willput 0
test['num_sold'] = 0
data = pd.concat([train, test], axis=0)


data.columns


# Select columns with object data type (character values)
character_columns = data.select_dtypes(include=['object']).columns
print("Columns with character values:", character_columns)


data['country'].unique()


# Visualisation of country with the num_sold
plt.figure(figsize=(12, 6))
sns.barplot(x='country', y='num_sold', data=train, estimator=sum)
plt.title('Total Number of Products Sold by Country')
plt.xlabel('Country')
plt.ylabel('Total Number of Products Sold')
plt.show()


data['store'].unique()


# Visualiastion of store data with num_sold
plt.figure(figsize=(12, 6))
sns.barplot(x='store', y='num_sold', data=train, estimator=sum)
plt.title('Total Number of Products Sold by Store')
plt.xlabel('Store')
plt.ylabel('Total Number of Products Sold')
plt.show()


data['product'].unique()


# Visualisation of product with num_sold
plt.figure(figsize=(12, 6))
sns.barplot(x='product', y='num_sold', data=train, estimator=sum)
plt.title('Total Number of Products Sold by Product')
plt.xlabel('Product')
plt.ylabel('Total Number of Products Sold')
plt.show()


# Convert 'date' column to datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# Extract date features for train data
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['day_of_week'] = train['date'].dt.weekday
train['quarter'] = train['date'].dt.quarter
train['week'] = train['date'].dt.isocalendar().week

# Extract date features for test data
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['day_of_week'] = test['date'].dt.weekday
test['quarter'] = test['date'].dt.quarter
test['week'] = test['date'].dt.isocalendar().week



train.drop('date', axis=1, inplace=True)
test.drop('date', axis=1, inplace=True)


data.head()


train.set_index('id', inplace=True)
test.set_index('id', inplace=True)
data.set_index('id', inplace=True)


train['num_sold'] = np.log1p(train['num_sold'])


train['num_sold'].hist() # check the distribution of the target variable


train.head()



num_cols = list(train.select_dtypes(exclude=['object']).columns.difference(['num_sold']))
cat_cols = list(train.select_dtypes(include=['object']).columns)

num_cols_test = list(test.select_dtypes(exclude=['object']).columns.difference(['id']))
cat_cols_test = list(test.select_dtypes(include=['object']).columns)


#Encoding the categorical columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cat_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])



# Keep only the specified columns
columns_to_keep = ['country', 'store', 'product', 'num_sold', 'year', 'month', 'day']

train = train[columns_to_keep]
test = test[columns_to_keep]
data = data[columns_to_keep]

# Display the updated dataframes
print(train.head())
print(test.head())
print(data.head())


test.drop(columns=['num_sold'], inplace=True)


# Calculate the correlation matrix
correlation_matrix = train.corr()

# Plot the heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split

X = train.drop(columns=['num_sold'])
y = train['num_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Define the hyperparameter space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.3),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
    }

    # Define the model
    model = XGBRegressor(**params, random_state=42)
    
    # Evaluate using cross-validation
    score = -cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return score



# Create a study and optimize it
study = optuna.create_study(direction='minimize')  # 'minimize' for MSE or similar metrics
study.optimize(objective, n_trials=100, n_jobs=-1)  # Increase n_trials for better results

# Get the best parameters
best_params = study.best_params
print("Best Parameters:", best_params)

# Save the parameters for future use
import json
with open('best_xgb_parameters.json', 'w') as f:
    json.dump(best_params, f)



xgb_params = # find your bast hyper parameter through the previous step and put it here


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

# Define MAPE metric
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

# Cross-validation for XGBRegressor
def cross_val_xgbr_mape(X, y, test, n_splits=5, **xgb_params):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        # Ensure data types for indexing
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        else:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

        # Initialize and train the model
        model = XGBRegressor(random_state=42, **xgb_params)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_valid)
        score = mape(y_valid, y_pred)
        mape_scores.append(score)

        # Predict on the test set
        preds.append(model.predict(test))

    # Average predictions over all folds
    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean

average_mape, xgb_preds = cross_val_xgbr_mape(X, y, test, n_splits=5, **xgb_params)

print(f"Average MAPE across folds: {average_mape:.4f}")

# Reset index to get 'id' column back
test_reset = test.reset_index()

# Save predictions for submission
submission = pd.DataFrame({'id': test_reset['id'], 'num_sold': np.expm1(xgb_preds)})
print(submission.head())
submission.to_csv('gg_xgb.csv', index=False)


import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score

# Objective function for LightGBM optimization
def objective(trial):
    # Define the hyperparameter space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', -1, 20),  # -1 means no limit
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
    }

    # Create and train the LightGBM model
    model = LGBMRegressor(random_state=42, **params)
    
    # Evaluate using cross-validation
    score = -cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return score



# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize')  # Minimize MSE or similar metric
study.optimize(objective, n_trials=100, n_jobs=-1)  # Increase n_trials for better results

# Extract the best parameters
best_params = study.best_params
print("Best Parameters:", best_params)

# Save parameters for future use
import json
with open('best_lgbm_parameters.json', 'w') as f:
    json.dump(best_params, f)



lgbm_params=# find your bast hyper parameter through the previous step and put it here


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

# Define MAPE metric
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

# Cross-validation for LGBMRegressor
def cross_val_lgbm_mape(X, y, test, n_splits=5, **lgbm_params):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        # Ensure data types for indexing
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        else:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

        # Initialize and train the model
        model = LGBMRegressor(random_state=42, **lgbm_params)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_valid)
        score = mape(y_valid, y_pred)
        mape_scores.append(score)

        # Predict on the test set
        preds.append(model.predict(test))

    # Average predictions over all folds
    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean


average_mape, lgb_preds = cross_val_lgbm_mape(X, y, test, n_splits=5, **lgbm_params)

print(f"Average MAPE across folds: {average_mape:.4f}")

# Reset index to get 'id' column back
test_reset = test.reset_index()

# Save predictions for submission
submission = pd.DataFrame({'id': test_reset['id'], 'num_sold': np.expm1(lgb_preds)})
print(submission.head())
submission.to_csv('submission_lgb.csv', index=False)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

# Define MAPE metric
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

# Cross-validation for CatBoostRegressor
def cross_val_catboost_mape(X, y, test, n_splits=5, cat_features=None, **cat_params):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        # Ensure data types for indexing
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        else:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

        # Initialize and train the model
        model = CatBoostRegressor(cat_features=cat_features, random_state=42, silent=True, **cat_params)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50, verbose=False)

        # Predictions and evaluation
        y_pred = model.predict(X_valid)
        score = mape(y_valid, y_pred)
        mape_scores.append(score)

        # Predict on the test set
        preds.append(model.predict(test))

    # Average predictions over all folds
    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean

# Example usage
cat_params = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "subsample": 0.8,
    "random_strength": 1,
    "loss_function": "MAPE"
}

average_mape, cat_preds = cross_val_catboost_mape(X, y, test, n_splits=5, cat_features=None, **cat_params)

print(f"Average MAPE across folds: {average_mape:.4f}")

# Reset index to get 'id' column back
test_reset = test.reset_index()

# Save predictions for submission
submission = pd.DataFrame({'id': test_reset['id'], 'num_sold': np.expm1(cat_preds)})
print(submission.head())
submission.to_csv('gg_catboost.csv', index=False)


