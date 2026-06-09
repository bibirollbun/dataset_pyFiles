import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Load data
train_data = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=['datetime'])
test_data = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv", parse_dates=['datetime'])

# Feature engineering
def enrich_datetime(df):
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 17, 18]).astype(int)
    return df

train_data = enrich_datetime(train_data)
test_data = enrich_datetime(test_data)

# Log-transform target to reduce RMSLE sensitivity
train_data['log_count'] = np.log1p(train_data['count'])

# Prepare features and targets
drop_cols = ['datetime', 'casual', 'registered', 'count']
X = train_data.drop(columns=drop_cols)
y = train_data['log_count']
X_test = test_data.drop(columns=['datetime'])

# One-hot encoding
X = pd.get_dummies(X, columns=['season', 'weather', 'month', 'hour', 'weekday', 'year'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['season', 'weather', 'month', 'hour', 'weekday', 'year'], drop_first=True)

# Align test columns to training columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model with tuned hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=200, 
    max_depth=20, 
    min_samples_split=5, 
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Predict and evaluate
val_preds_log = rf_model.predict(X_val)
val_preds = np.expm1(val_preds_log)  # revert log1p
true_vals = np.expm1(y_val)          # revert log1p

rmsle_score = np.sqrt(mean_squared_log_error(true_vals, val_preds))
print(f"Validation RMSLE: {rmsle_score:.5f}")

# Final prediction on test set
test_preds_log = rf_model.predict(X_test)
test_preds = np.expm1(test_preds_log)  # revert log1p

# Submission
submission = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")
submission['count'] = test_preds
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved as /kaggle/working/submission.csv")


