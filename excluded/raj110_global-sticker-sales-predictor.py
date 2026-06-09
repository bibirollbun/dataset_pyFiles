# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# Step 2: Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

# Step 3: Data Preprocessing
## Convert 'date' to datetime format
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

## Handle missing values (if any)
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

## Feature Engineering: Extract useful features from 'date'
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['weekday'] = train['date'].dt.weekday
train['is_weekend'] = train['weekday'] >= 5  # Friday-Sunday as weekend
train['is_holiday'] = train['month'].isin([12, 1])  # Assuming December and January as holidays

test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['weekday'] = test['date'].dt.weekday
test['is_weekend'] = test['weekday'] >= 5
test['is_holiday'] = test['month'].isin([12, 1])

# Step 4: Handle Outliers
## Cap the 'num_sold' column to remove extreme outliers
q1 = train['num_sold'].quantile(0.25)
q3 = train['num_sold'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
train['num_sold'] = np.clip(train['num_sold'], lower_bound, upper_bound)

# Step 5: Feature Encoding (for categorical variables)
train = pd.get_dummies(train, columns=['country', 'store', 'product'], drop_first=True)
test = pd.get_dummies(test, columns=['country', 'store', 'product'], drop_first=True)

# Step 6: Define features and target
X = train.drop(columns=['id', 'date', 'num_sold'])
y = train['num_sold']

# Step 7: Scaling Features using MinMaxScaler (better for XGBoost and tree-based models)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Time-Series Cross-Validation (using TimeSeriesSplit for cross-validation)
tscv = TimeSeriesSplit(n_splits=5)
train_index, val_index = list(tscv.split(X))[0]

X_train, X_val = X_scaled[train_index], X_scaled[val_index]
y_train, y_val = y.iloc[train_index], y.iloc[val_index]

# Step 9: Model Training (XGBoost Regressor)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train)

# Step 10: Evaluate Model using MAPE
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    
    # Avoid dividing by zero or very small values
    epsilon = 1e-10
    y_val = np.where(y_val == 0, epsilon, y_val)  # Replace zero values with a small epsilon to avoid division by zero
    
    mape = np.mean(np.abs((y_val - y_pred) / y_val))  # Mean Absolute Percentage Error
    return mape

# Calculate MAPE for the XGBoost model
xgb_mape = evaluate_model(xgb_model, X_val, y_val)
print(f'XGBoost MAPE: {xgb_mape}')

# Step 11: Hyperparameter Tuning using RandomizedSearchCV (for XGBoost)
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
}

random_search = RandomizedSearchCV(estimator=XGBRegressor(random_state=42), param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')
random_search.fit(X_train, y_train)

print("Best parameters for XGBoost:", random_search.best_params_)

# Step 12: Retrain the final model with best hyperparameters
best_model = random_search.best_estimator_
best_model.fit(X_scaled, y)

# Step 13: Predict on the test set
X_test = test.drop(columns=['id', 'date'])
X_test_scaled = scaler.transform(X_test)

test['num_sold'] = best_model.predict(X_test_scaled)

# Step 14: Round predictions to 4 decimal places (to match required format)
test['num_sold'] = test['num_sold'].round(4)

# Step 15: Create a Submission File
submission = test[['id', 'num_sold']]
submission.to_csv('optimized_submission.csv', index=False)

print("submission file created!")


