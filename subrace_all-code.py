


# Reload the necessary libraries and data due to the environment reset
import pandas as pd

# Reloading dataset paths
train_path = '/kaggle/input/playground-series-s4e12/train.csv'
test_path = '/kaggle/input/playground-series-s4e12/test.csv'

# Load the datasets
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Display basic information about the training dataset
train_data_info = train_data.info()
train_data_head = train_data.head()

# Display basic information about the test dataset
test_data_info = test_data.info()
test_data_head = test_data.head()

train_data_info, train_data_head, test_data_info, test_data_head



# Get the total rows and columns for both datasets
train_shape = train_data.shape
test_shape = test_data.shape

train_shape, test_shape



# Subset size for prototyping
subset_size = 100000

# Load a subset of the training data
train_data_subset = pd.read_csv(train_path, nrows=subset_size)

# Drop irrelevant columns
train_data_subset = train_data_subset.drop(columns=['Policy Start Date', 'id'])

# Display the shape of the subset and a quick overview
train_data_subset_shape = train_data_subset.shape
train_data_subset_head = train_data_subset.head()

train_data_subset_shape, train_data_subset_head


# Split the subset into features (X) and target (y)
target_column = 'Premium Amount'
X = train_data_subset.drop(columns=[target_column])
y = train_data_subset[target_column]

# Display shapes of features and target
X_shape = X.shape
y_shape = y.shape

X_shape, y_shape



# Encode categorical variables using one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Display the shape of the encoded features
X_encoded_shape = X_encoded.shape

X_encoded_shape



# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from scipy.stats import uniform

# Define RSMLE calculation
def rsmle(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

# Subset size for prototyping
subset_size = 100000

# Load a subset of the training data
train_data_subset = pd.read_csv(train_path, nrows=subset_size)

# Drop irrelevant columns
train_data_subset = train_data_subset.drop(columns=['Policy Start Date', 'id'])

# Feature Engineering: Interaction and Transformation
train_data_subset['Income_Health'] = train_data_subset['Annual Income'] * train_data_subset['Health Score']
train_data_subset['Annual Income'] = np.log1p(train_data_subset['Annual Income'])

# Split the subset into features and target
target_column = 'Premium Amount'
X = train_data_subset.drop(columns=[target_column])
y = train_data_subset[target_column]

# Encode categorical variables using one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Log-transform the target variable for training
y_train_log = np.log1p(y_train)

# Define parameter grid for RandomizedSearchCV
param_distributions = {
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [100, 200, 300, 500],
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'min_child_weight': [1, 3, 5, 7],
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(1, 10)
}

# Initialize the XGBoost regressor
xgb_model = xgb.XGBRegressor(random_state=42)

# Perform RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=50,  # Number of random samples to evaluate
    scoring='neg_root_mean_squared_error',
    cv=3,
    random_state=42,
    verbose=1
)

# Fit the RandomizedSearchCV
random_search.fit(X_train, y_train_log)

# Get the best parameters
best_params = random_search.best_params_
print(f"Best Parameters: {best_params}")

# Train the model with the best parameters
best_xgb_model = xgb.XGBRegressor(**best_params, random_state=42)
best_xgb_model.fit(
    X_train, y_train_log,
    early_stopping_rounds=20,
    eval_set=[(X_val, np.log1p(y_val))],
    verbose=True
)

# Make predictions and inverse log-transform
y_pred_log = best_xgb_model.predict(X_val)
y_pred = np.expm1(y_pred_log)

# Cross-Validation on Training Data
cross_val_rmse = cross_val_score(
    best_xgb_model,
    X_train,
    y_train_log,
    scoring='neg_root_mean_squared_error',
    cv=5
)
print(f"Cross-Validation RMSE: {-cross_val_rmse.mean():.2f}")

# Calculate evaluation metrics
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
rsmle_score = rsmle(y_val, y_pred)

# Print evaluation results
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Root Mean Squared Logarithmic Error (RSMLE): {rsmle_score:.2f}")



import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import early_stopping, log_evaluation

# Log-transform the target variable for LightGBM
y_train_log = np.log1p(y_train)

# Define LightGBM model
lgb_model = lgb.LGBMRegressor(random_state=42)

# Define parameter grid for LightGBM
param_distributions = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 1, 5, 10]
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='neg_root_mean_squared_error',
    cv=3,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train_log)

# Get the best parameters
best_params = random_search.best_params_
print(f"Best Parameters for LightGBM: {best_params}")

# Train the LightGBM model with the best parameters and early stopping
best_lgb_model = lgb.LGBMRegressor(**best_params, random_state=42)
best_lgb_model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, np.log1p(y_val))],
    callbacks=[
        early_stopping(stopping_rounds=20),
        log_evaluation(period=10)
    ]
)

# Make predictions and inverse log-transform
y_pred_log = best_lgb_model.predict(X_val)
y_pred = np.expm1(y_pred_log)

# Evaluate the model
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
rsmle_score = rsmle(y_val, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Root Mean Squared Logarithmic Error (RSMLE): {rsmle_score:.2f}")



from catboost import CatBoostRegressor

# Train the CatBoost model
cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=100
)

cat_model.fit(X_train, y_train_log, eval_set=(X_val, np.log1p(y_val)), early_stopping_rounds=20)

# Make predictions and inverse log-transform
y_pred_log = cat_model.predict(X_val)
y_pred = np.expm1(y_pred_log)

# Evaluate the model
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
rsmle_score = rsmle(y_val, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Root Mean Squared Logarithmic Error (RSMLE): {rsmle_score:.2f}")



from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor

# Create a pipeline for ElasticNet with preprocessing
elasticnet_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler()),                # Scale the features
    ('elasticnet', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))  # ElasticNet model
])

# Add ElasticNet to the stacking ensemble
stacking_model = StackingRegressor(
    estimators=[
        ('lgb', lgb.LGBMRegressor(**best_params, random_state=42)),
        ('cat', CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_seed=42, loss_function='RMSE', verbose=False)),
        ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)),
        ('hgb', HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=10, random_state=42)),
        ('elasticnet', elasticnet_pipeline)
    ],
    final_estimator=Ridge(alpha=1.0)  # Meta-model
)

# Train the stacking model
stacking_model.fit(X_train, y_train_log)

# Predict with the stacking model and inverse log-transform
y_pred_stack_log = stacking_model.predict(X_val)
y_pred_stack = np.expm1(y_pred_stack_log)

# Evaluate the stacking model
mae_stack = mean_absolute_error(y_val, y_pred_stack)
mse_stack = mean_squared_error(y_val, y_pred_stack)
rmse_stack = mse_stack ** 0.5
rsmle_stack = rsmle(y_val, y_pred_stack)

# Print evaluation results
print(f"Stacking Model MAE: {mae_stack:.2f}")
print(f"Stacking Model MSE: {mse_stack:.2f}")
print(f"Stacking Model RMSE: {rmse_stack:.2f}")
print(f"Stacking Model RSMLE: {rsmle_stack:.2f}")



from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define Histogram Gradient Boosting model
hgb_model = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.05,
    max_depth=10,
    l2_regularization=1.0,
    random_state=42
)

# Train the Histogram Gradient Boosting model
hgb_model.fit(X_train, y_train_log)

# Predict with Histogram Gradient Boosting and inverse log-transform
y_pred_hgb_log = hgb_model.predict(X_val)
y_pred_hgb = np.expm1(y_pred_hgb_log)

# Evaluate the model
mae_hgb = mean_absolute_error(y_val, y_pred_hgb)
mse_hgb = mean_squared_error(y_val, y_pred_hgb)
rmse_hgb = mse_hgb ** 0.5
rsmle_hgb = rsmle(y_val, y_pred_hgb)

print(f"Histogram Gradient Boosting MAE: {mae_hgb:.2f}")
print(f"Histogram Gradient Boosting MSE: {mse_hgb:.2f}")
print(f"Histogram Gradient Boosting RMSE: {rmse_hgb:.2f}")
print(f"Histogram Gradient Boosting RSMLE: {rsmle_hgb:.2f}")



from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

stacking_model = StackingRegressor(
    estimators=[
        ('lgb', lgb.LGBMRegressor(**best_params, random_state=42)),
        ('cat', CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_seed=42, loss_function='RMSE', verbose=False)),
        ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)),
        ('hgb', HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=10, random_state=42)),
        ('elasticnet', elasticnet_pipeline)  # Preprocessing pipeline for ElasticNet
    ],
    final_estimator=Ridge(alpha=1.0)
)


# Train the stacking modela
stacking_model.fit(X_train, y_train_log)

# Predict with the stacking model and inverse log-transform
y_pred_stack_log = stacking_model.predict(X_val)
y_pred_stack = np.expm1(y_pred_stack_log)

# Evaluate the stacking model
mae_stack = mean_absolute_error(y_val, y_pred_stack)
mse_stack = mean_squared_error(y_val, y_pred_stack)
rmse_stack = mse_stack ** 0.5
rsmle_stack = rsmle(y_val, y_pred_stack)

print(f"Stacking Model MAE: {mae_stack:.2f}")
print(f"Stacking Model MSE: {mse_stack:.2f}")
print(f"Stacking Model RMSE: {rmse_stack:.2f}")
print(f"Stacking Model RSMLE: {rsmle_stack:.2f}")



import pandas as pd
submission_file = pd.read_csv("/kaggle/input/playground-series-s4e12/sample_submission.csv")
submission_file.head()



