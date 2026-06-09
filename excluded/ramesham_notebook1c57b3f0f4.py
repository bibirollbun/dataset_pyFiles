import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Load datasets
train_data = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=['datetime'])
test_data = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv", parse_dates=['datetime'])

# Extract datetime features
def add_datetime_features(df):
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday
    return df

train_data = add_datetime_features(train_data)
test_data = add_datetime_features(test_data)

# Separate features and target
features = train_data.drop(columns=['datetime', 'casual', 'registered', 'count'])
target = train_data['count']
test_features = test_data.drop(columns=['datetime'])

# Apply one-hot encoding
X_encoded = pd.get_dummies(features, columns=['season', 'weather', 'month', 'hour', 'weekday', 'year'], drop_first=True)
X_test_encoded = pd.get_dummies(test_features, columns=['season', 'weather', 'month', 'hour', 'weekday', 'year'], drop_first=True)

# Align test features with training features
X_test_encoded = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Split data into training and validation sets
X_train_set, X_valid_set, y_train_set, y_valid_set = train_test_split(X_encoded, target, test_size=0.2, random_state=42)

# Initialize and train the model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_set, y_train_set)

# Predict and evaluate
val_predictions = rf_regressor.predict(X_valid_set)
rmsle_score = np.sqrt(mean_squared_log_error(y_valid_set, val_predictions))
print(f"Validation RMSLE: {rmsle_score:.5f}")

# Generate predictions for the test dataset
final_predictions = rf_regressor.predict(X_test_encoded)

# Load submission format and assign predictions
submission_df = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")
submission_df['count'] = final_predictions

# Save submission file
output_path = "/kaggle/working/submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission saved as {output_path}")


