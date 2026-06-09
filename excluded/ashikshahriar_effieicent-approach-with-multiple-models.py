import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


train_data=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_data.head()


train_data.info()


import pandas as pd
from sklearn.impute import SimpleImputer

# Create a copy of your train_data to avoid changing the original dataframe
train_data_imputed = train_data.copy()

# Impute Numerical Columns with Mean or Median
numerical_columns = ['Compartments', 'Weight Capacity (kg)']
for col in numerical_columns:
    # Using median for numerical columns to avoid outliers
    train_data_imputed[col] = train_data_imputed[col].fillna(train_data_imputed[col].median())

# Impute Categorical Columns with Mode (Most Frequent Value)
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
for col in categorical_columns:
    train_data_imputed[col] = train_data_imputed[col].fillna(train_data_imputed[col].fillna('Unknown'))


# Check if there are still any missing values
missing_values = train_data_imputed.isnull().sum()
print(missing_values)



import numpy as np

# Step 1: Feature Engineering

train_data_imputed['Compartments_bin'] = pd.cut(train_data_imputed['Compartments'], bins=[0, 3, 6, 10], labels=['Low', 'Medium', 'High'])
train_data_imputed['Weight Capacity bin'] = pd.cut(train_data_imputed['Weight Capacity (kg)'], bins=[0, 5, 10, 20], labels=['Light', 'Medium', 'Heavy'])

train_data_imputed['Log_Weight_Capacity'] = train_data_imputed['Weight Capacity (kg)'].apply(lambda x: np.log(x + 1) if x > 0 else 0)

train_data_imputed['Compartments_x_Weight'] = train_data_imputed['Compartments'] * train_data_imputed['Weight Capacity (kg)']

train_data_imputed['Waterproof'] = train_data_imputed['Waterproof'].map({'Yes': 1, 'No': 0})


# Step 2: One-Hot Encoding Categorical Features
train_data_imputed = pd.get_dummies(train_data_imputed, drop_first=True)




train_data_imputed.head()


train_data_imputed['Waterproof'].value_counts()


train_data_imputed.info()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Split the data into features (X) and target (y)
X = train_data_imputed.drop(columns=['id', 'Price'])  # Drop 'id' and target column 'Price'
y = train_data_imputed['Price']

# Step 2: Train-Test Split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.impute import SimpleImputer

# Step 1: Impute missing values in X_train and X_test
imputer = SimpleImputer(strategy='mean')  # You can use median or most_frequent as well

# Apply the imputer to both training and testing data
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Step 2: Apply a Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train_imputed, y_train)

# Step 3: Predict on the test set
y_pred = regressor.predict(X_test_imputed)

# Step 4: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')



import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
lgb_model.fit(X_train_imputed, y_train)

y_pred_lgb = lgb_model.predict(X_test_imputed)

mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

print(f'LightGBM Regressor:')
print(f'MAE: {mae_lgb}')
print(f'MSE: {mse_lgb}')
print(f'R²: {r2_lgb}')



from catboost import CatBoostRegressor

catboost_model = CatBoostRegressor(n_estimators=200, learning_rate=0.05, depth=4, random_state=42, verbose=0)
catboost_model.fit(X_train_imputed, y_train)

y_pred_catboost = catboost_model.predict(X_test_imputed)

mae_catboost = mean_absolute_error(y_test, y_pred_catboost)
mse_catboost = mean_squared_error(y_test, y_pred_catboost)
r2_catboost = r2_score(y_test, y_pred_catboost)

print(f'CatBoost Regressor:')
print(f'MAE: {mae_catboost}')
print(f'MSE: {mse_catboost}')
print(f'R²: {r2_catboost}')



from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

base_models = [
    ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)),
    ('lgb', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)),
    ('cat', CatBoostRegressor(n_estimators=200, learning_rate=0.05, depth=4, random_state=42, verbose=0))
]

meta_model = Ridge()

stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stacking_model.fit(X_train_imputed, y_train)

y_pred_stacking = stacking_model.predict(X_test_imputed)

mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
r2_stacking = r2_score(y_test, y_pred_stacking)

print(f'Stacking Regressor:')
print(f'MAE: {mae_stacking}')
print(f'MSE: {mse_stacking}')
print(f'R²: {r2_stacking}')



from sklearn.ensemble import VotingRegressor

voting_model = VotingRegressor(estimators=[
    ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)),
    ('lgb', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42))
], weights=[2, 1, 1])  # Giving XGBoost more weight

voting_model.fit(X_train_imputed, y_train)

y_pred_voting = voting_model.predict(X_test_imputed)

mae_voting = mean_absolute_error(y_test, y_pred_voting)
mse_voting = mean_squared_error(y_test, y_pred_voting)
r2_voting = r2_score(y_test, y_pred_voting)

print(f'Voting Regressor:')
print(f'MAE: {mae_voting}')
print(f'MSE: {mse_voting}')
print(f'R²: {r2_voting}')



from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Step 2: Train Ridge Regression model
ridge_regressor = Ridge(alpha=1.0)
ridge_regressor.fit(X_train_imputed, y_train)

# Step 3: Predict on test set
y_pred_ridge = ridge_regressor.predict(X_test_imputed)

# Step 4: Evaluate model
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f'Ridge Regression:')
print(f'MAE: {mae_ridge}')
print(f'MSE: {mse_ridge}')
print(f'R²: {r2_ridge}')



from sklearn.ensemble import RandomForestRegressor

# Step 1: Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Step 2: Train Random Forest model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_imputed, y_train)

# Step 3: Predict on test set
y_pred_rf = rf_regressor.predict(X_test_imputed)

# Step 4: Evaluate model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Regression:')
print(f'MAE: {mae_rf}')
print(f'MSE: {mse_rf}')
print(f'R²: {r2_rf}')



from xgboost import XGBRegressor

# Step 1: Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Step 2: Train XGBoost model
xgb_regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_regressor.fit(X_train_imputed, y_train)

# Step 3: Predict on test set
y_pred_xgb = xgb_regressor.predict(X_test_imputed)

# Step 4: Evaluate model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f'XGBoost Regression:')
print(f'MAE: {mae_xgb}')
print(f'MSE: {mse_xgb}')
print(f'R²: {r2_xgb}')



from sklearn.ensemble import GradientBoostingRegressor

# Step 1: Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Step 2: Train Gradient Boosting model
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_regressor.fit(X_train_imputed, y_train)

# Step 3: Predict on test set
y_pred_gb = gb_regressor.predict(X_test_imputed)

# Step 4: Evaluate model
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f'Gradient Boosting Regression:')
print(f'MAE: {mae_gb}')
print(f'MSE: {mse_gb}')
print(f'R²: {r2_gb}')



from sklearn.ensemble import VotingRegressor
import xgboost as xgb

# Create individual models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
xg_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

# Create the ensemble model (voting)
ensemble_regressor = VotingRegressor(estimators=[('rf', rf_model), ('gb', gb_model), ('xg', xg_model)])

# Step 1: Fit the ensemble model
ensemble_regressor.fit(X_train_imputed, y_train)

# Step 2: Predict on test set
y_pred_ensemble = ensemble_regressor.predict(X_test_imputed)

# Step 3: Evaluate ensemble model
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)

print(f'Ensemble Model (Voting Regressor):')
print(f'MAE: {mae_ensemble}')
print(f'MSE: {mse_ensemble}')
print(f'R²: {r2_ensemble}')



from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

# Step 1: Define the base model (e.g., Linear Regression)
base_model = LinearRegression()

# Step 2: Define the Bagging Regressor with the base model
bagging_model = BaggingRegressor(base_model, n_estimators=100, random_state=42)

# Step 3: Train the bagging model
bagging_model.fit(X_train_imputed, y_train)

# Step 4: Predict on the test set
y_pred_bagging = bagging_model.predict(X_test_imputed)

# Step 5: Evaluate the model
mae_bagging = mean_absolute_error(y_test, y_pred_bagging)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)

print(f'Bagging Regressor:')
print(f'MAE: {mae_bagging}')
print(f'MSE: {mse_bagging}')
print(f'R²: {r2_bagging}')



from sklearn.ensemble import VotingRegressor

# Step 1: Define the base models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
xg_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

# Step 2: Define the Voting Regressor (Weighted)
ensemble_model = VotingRegressor(estimators=[('rf', rf_model), ('gb', gb_model), ('xg', xg_model)],
                                 weights=[1, 2, 1])  # Custom weights for the base models

# Step 3: Train the ensemble model
ensemble_model.fit(X_train_imputed, y_train)

# Step 4: Predict on test set
y_pred_ensemble = ensemble_model.predict(X_test_imputed)

# Step 5: Evaluate the model
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)

print(f'Weighted Voting Regressor:')
print(f'MAE: {mae_ensemble}')
print(f'MSE: {mse_ensemble}')
print(f'R²: {r2_ensemble}')



test_data=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
test_data.head()


test_data.info()


train_data_imputed.columns


test_data_imputed = test_data.copy()

# Impute Numerical Columns with Mean or Median
numerical_columns = ['Compartments', 'Weight Capacity (kg)']
for col in numerical_columns:
    # Using median for numerical columns to avoid outliers
    test_data_imputed[col] = test_data_imputed[col].fillna(test_data_imputed[col].median())

# Impute Categorical Columns with Mode (Most Frequent Value)
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
for col in categorical_columns:
    test_data_imputed[col] = test_data_imputed[col].fillna(test_data_imputed[col].fillna('Unknown'))


# Check if there are still any missing values
missing_values = test_data_imputed.isnull().sum()
print(missing_values)


test_data_imputed['Compartments_bin'] = pd.cut(test_data_imputed['Compartments'], bins=[0, 3, 6, 10], labels=['Low', 'Medium', 'High'])
test_data_imputed['Weight Capacity bin'] = pd.cut(test_data_imputed['Weight Capacity (kg)'], bins=[0, 5, 10, 20], labels=['Light', 'Medium', 'Heavy'])

test_data_imputed['Log_Weight_Capacity'] = test_data_imputed['Weight Capacity (kg)'].apply(lambda x: np.log(x + 1) if x > 0 else 0)

test_data_imputed['Compartments_x_Weight'] = test_data_imputed['Compartments'] * test_data_imputed['Weight Capacity (kg)']

test_data_imputed['Waterproof'] = test_data_imputed['Waterproof'].map({'Yes': 1, 'No': 0})



# Step 2: One-Hot Encoding Categorical Features
test_data_imputed = pd.get_dummies(test_data_imputed, drop_first=True)

# Check the updated dataframe



test_data_imputed.info()


# Ensure test data has the same columns as the training data
X_test = test_data_imputed.drop(['id'], axis=1)

# Get the feature columns used in training
expected_features = X_train.columns  # X_train is the feature matrix from training

# Reorder columns and fill missing ones with 0
X_test = X_test.reindex(columns=expected_features, fill_value=0)

# Apply the trained imputer
X_test_imputed = imputer.transform(X_test)

# Predict the price
y_pred_test =ensemble_regressor.predict(X_test_imputed)

# Create submission file
submission = pd.DataFrame({
    'id': test_data_imputed['id'],
    'Price': y_pred_test
})

# Save the submission file
submission.to_csv('submission_ensemble.csv', index=False)

print('Submission file has been created: submission.csv')



# Ensure test data has the same columns as the training data
X_test = test_data_imputed.drop(['id'], axis=1)

# Get the feature columns used in training
expected_features = X_train.columns  # X_train is the feature matrix from training

# Reorder columns and fill missing ones with 0
X_test = X_test.reindex(columns=expected_features, fill_value=0)

# Apply the trained imputer
X_test_imputed = imputer.transform(X_test)

# Predict the price
y_pred_test =ensemble_model.predict(X_test_imputed)

# Create submission file
submission = pd.DataFrame({
    'id': test_data_imputed['id'],
    'Price': y_pred_test
})

# Save the submission file
submission.to_csv('submission_ensmblmodel.csv', index=False)

print('Submission file has been created: submission.csv')



# Ensure test data has the same columns as the training data
X_test = test_data_imputed.drop(['id'], axis=1)

# Get the feature columns used in training
expected_features = X_train.columns  # X_train is the feature matrix from training

# Reorder columns and fill missing ones with 0
X_test = X_test.reindex(columns=expected_features, fill_value=0)

# Apply the trained imputer
X_test_imputed = imputer.transform(X_test)

# Predict the price
y_pred_test =bagging_model.predict(X_test_imputed)

# Create submission file
submission = pd.DataFrame({
    'id': test_data_imputed['id'],
    'Price': y_pred_test
})

# Save the submission file
submission.to_csv('submission_bagging.csv', index=False)

print('Submission file has been created: submission.csv')


# Ensure test data has the same columns as the training data
X_test = test_data_imputed.drop(['id'], axis=1)

# Get the feature columns used in training
expected_features = X_train.columns  # X_train is the feature matrix from training

# Reorder columns and fill missing ones with 0
X_test = X_test.reindex(columns=expected_features, fill_value=0)

# Apply the trained imputer
X_test_imputed = imputer.transform(X_test)

# Predict the price
y_pred_test =lgb_model.predict(X_test_imputed)

# Create submission file
submission = pd.DataFrame({
    'id': test_data_imputed['id'],
    'Price': y_pred_test
})

# Save the submission file
submission.to_csv('submission_lgb.csv', index=False)

print('Submission file has been created: submission.csv')


# Ensure test data has the same columns as the training data
X_test = test_data_imputed.drop(['id'], axis=1)

# Get the feature columns used in training
expected_features = X_train.columns  # X_train is the feature matrix from training

# Reorder columns and fill missing ones with 0
X_test = X_test.reindex(columns=expected_features, fill_value=0)

# Apply the trained imputer
X_test_imputed = imputer.transform(X_test)

# Predict the price
y_pred_test =stacking_model.predict(X_test_imputed)

# Create submission file
submission = pd.DataFrame({
    'id': test_data_imputed['id'],
    'Price': y_pred_test
})

# Save the submission file
submission.to_csv('submission_stacking.csv', index=False)

print('Submission file has been created: submission.csv')

