!pip install cudf cuml

# Import necessary libraries
import cudf
import numpy as np
from cuml.ensemble import RandomForestRegressor
from cuml.preprocessing import LabelEncoder
from cuml.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error  # Use from sklearn instead
import gc

# Load the data using cuDF for GPU acceleration
train_data = cudf.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = cudf.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

# Basic exploration
print(train_data.head())
print(train_data.info())

# Check for missing values
print("Missing values in training data:", train_data.isnull().sum().sum())
print("Missing values in test data:", test_data.isnull().sum().sum())

# Preprocess the data
# Convert date to datetime and extract features
train_data['date'] = cudf.to_datetime(train_data['date'])
test_data['date'] = cudf.to_datetime(test_data['date'])

for dataset in [train_data, test_data]:
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['day_of_week'] = dataset['date'].dt.dayofweek

# Encode categorical variables
le = LabelEncoder()
for column in ['country', 'store', 'product']:
    train_data[column] = le.fit_transform(train_data[column])
    test_data[column] = le.transform(test_data[column])  # Use the same encoder

# Check for NaN values in the target variable
print("NaN values in num_sold:", train_data['num_sold'].isna().sum())

# Handle NaN values in 'num_sold'
train_data = train_data.dropna(subset=['num_sold'])  # Dropping rows with NaN in num_sold

# Re - check for NaN values after handling
print("NaN values after handling:", train_data['num_sold'].isna().sum())

# Additional Feature Engineering
# Aggregation Features
monthly_avg = train_data.groupby(['country', 'store', 'product', 'year', 'month'])['num_sold'].mean().reset_index(name='monthly_avg_sold')
train_data = cudf.merge(train_data, monthly_avg, on=['country', 'store', 'product', 'year', 'month'], how='left')
test_data = cudf.merge(test_data, monthly_avg, on=['country', 'store', 'product', 'year', 'month'], how='left')
test_data['monthly_avg_sold'] = test_data['monthly_avg_sold'].fillna(train_data['monthly_avg_sold'].mean())
del monthly_avg
gc.collect()

# Trend Features
train_data = train_data.sort_values(by=['country', 'store', 'product', 'year', 'month'])
train_data['sales_trend'] = train_data.groupby(['country', 'store', 'product'])['num_sold'].diff()
train_data['sales_trend'] = train_data['sales_trend'].fillna(0)
avg_trend = train_data['sales_trend'].mean()
test_data['sales_trend'] = avg_trend

# Prepare features and target again
features = ['year', 'month', 'day', 'day_of_week', 'country', 'store', 'product', 'monthly_avg_sold', 'sales_trend']
X = train_data[features]
y = train_data['num_sold']

# Convert to smaller data types if possible
for col in X.columns:
    if X[col].dtype == 'float64':
        X[col] = X[col].astype('float32')
    elif X[col].dtype == 'int64':
        X[col] = X[col].astype('int32')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
from cuml.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, n_jobs=1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Make predictions on validation set
val_predictions = best_model.predict(X_val)

# Convert to Pandas for metric calculation
y_val_pandas = y_val.to_pandas()
val_predictions_pandas = val_predictions.to_pandas()

# Calculate the MAPE for validation
mape = mean_absolute_percentage_error(y_val_pandas, val_predictions_pandas)
print(f"Validation MAPE after hyperparameter tuning: {mape}")

# Check if test data has all the required features
for feature in features:
    if feature not in test_data.columns:
        raise ValueError(f"Test data is missing feature: {feature}")

# Make predictions on test set
test_predictions = best_model.predict(test_data[features])

# Create submission file
submission = cudf.DataFrame({
    'id': test_data['id'],
    'num_sold': test_predictions
})

# Convert to pandas DataFrame for CSV writing
submission = submission.to_pandas()

# Save the submission file
submission.to_csv('submission.csv', index=False)
print(submission.head())

