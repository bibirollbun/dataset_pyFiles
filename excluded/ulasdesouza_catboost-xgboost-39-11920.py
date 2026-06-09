import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
import missingno as msno

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col=[0])
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col=[0])

display(train.head(), test.head())


# Extract the Target Column
target_column = (set(train.columns) - set(test.columns)).pop()

print(f"Target column: {target_column}")
print(f"Data type: {train[target_column].dtype}")


train.info()


test.info()


train.isnull().sum()


test.isnull().sum()


# Fill missing values in numeric columns
numeric_columns = train.select_dtypes(include=['number']).columns
for col in numeric_columns:
    if col in test.columns:
        median_value = train[col].median()  # Calculate the median
        train[col].fillna(median_value, inplace=True)
        test[col].fillna(median_value, inplace=True)

# Fill missing values in object columns
object_columns = train.select_dtypes(include=['object']).columns
for col in object_columns:
    if col in test.columns:
        train[col].fillna("Unknown", inplace=True)
        test[col].fillna("Unknown", inplace=True)


train.isnull().sum()


test.isnull().sum()


le = LabelEncoder()
object_columns = train.select_dtypes(include=['object']).columns
for column_name in object_columns:
    train[column_name] = le.fit_transform(train[column_name])    
    test[column_name] = le.transform(test[column_name])
    
display(train.head(), test.head())


# Select numerical columns
numerical_columns = train.select_dtypes(include=['float64']).columns
numerical_columns = numerical_columns[numerical_columns != target_column]

# Applying Normalization
scaler = StandardScaler()
train[numerical_columns] = scaler.fit_transform(train[numerical_columns])
test[numerical_columns] = scaler.transform(test[numerical_columns])

display(train.head(), test.head())


# Prepare training data
X_train = train.drop([target_column], axis=1)
y_train = train[target_column]

display(X_train.dtypes, y_train.dtypes)


# Initialize the XGBoost regressor model
xgboost_params = {'random_state': 42}

# Hyperparameters for BayesSearchCV tuning for XGBoost
xgboost_search_spaces = {
    'n_estimators': (10, 200),
    'max_depth': (2, 10),
    'reg_alpha': (0.00005, 0.2, 'log-uniform'),
    'reg_lambda': (0.005, 100, 'log-uniform')
}

xgboost_search = BayesSearchCV(
    estimator=XGBRegressor(**xgboost_params),
    search_spaces=xgboost_search_spaces,
    n_iter=100,
    cv=5,
    verbose=1,
    scoring="neg_root_mean_squared_error",
    random_state=42
)

# Perform the Bayesian optimization with cross-validation for XGBoost
xgboost_search.fit(X_train, y_train)
xgboost_rmse = -xgboost_search.best_score_


print("XGBoost Best params: ", xgboost_search.best_params_)
print("XGBoost Best RMSE: ", xgboost_rmse)


# Initialize the CatBoost regressor model
catboost_params = {'random_seed': 42, 'verbose': False}

# Hyperparameters for BayesSearchCV tuning for CatBoost
catboost_search_spaces = {
    'iterations': (10, 200),
    'depth': (2, 10),
    'l2_leaf_reg': (0.0001, 0.8, 'log-uniform'),
    'learning_rate': (0.01, 0.5, 'log-uniform')
}

# Set up BayesSearchCV for hyperparameter tuning
catboost_search = BayesSearchCV(
    estimator=CatBoostRegressor(**catboost_params),
    search_spaces=catboost_search_spaces,
    n_iter=100,
    cv=5,
    verbose=1,
    scoring="neg_root_mean_squared_error",
    random_state=42
)

# Perform the Bayesian optimization with cross-validation for CatBoost
catboost_search.fit(X_train, y_train)
catboost_rmse = -catboost_search.best_score_


print("CatBoost Best params: ", catboost_search.best_params_)
print("CatBoost Best RMSE: ", catboost_rmse)


# En iyi modelleri ve ağırlıkları alma
best_cat = catboost_search.best_estimator_
best_xgb = xgboost_search.best_estimator_

# Validation set üzerinde ağırlık optimizasyonu
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

best_cat.fit(X_train_sub, y_train_sub)
best_xgb.fit(X_train_sub, y_train_sub)

cat_pred = best_cat.predict(X_val)
xgb_pred = best_xgb.predict(X_val)

# Ağırlık optimizasyonu
def objective(weights):
    return np.sqrt(mean_squared_error(y_val, weights[0]*cat_pred + weights[1]*xgb_pred))

result = minimize(objective, [0.5, 0.5], bounds=[(0,1), (0,1)])
best_weights = result.x

# Full model training
best_cat.fit(X_train, y_train)
best_xgb.fit(X_train, y_train)

# Tahminler ve ensemble
cat_test_pred = best_cat.predict(test)
xgb_test_pred = best_xgb.predict(test)
ensemble_pred = best_weights[0]*cat_test_pred + best_weights[1]*xgb_test_pred


# Submission
submission = pd.DataFrame({
    'id': test.index,
    target_column: ensemble_pred
})
submission.to_csv('submission.csv', index=False)

