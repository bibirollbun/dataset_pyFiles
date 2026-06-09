import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Load the dataset
train_data = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test_data = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

# Feature engineering
def preprocess_data(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_workhour'] = df['hour'].apply(lambda x: 1 if 8 <= x <= 18 else 0)
    df['datetime'] = df['datetime']  # Preserve for plotting/submission
    return df

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Features
feature_cols = ['hour', 'day', 'month', 'year', 'dayofweek', 'is_weekend', 'is_workhour',
                'temp', 'atemp', 'humidity', 'windspeed', 'weather', 'season', 'holiday', 'workingday']

X = train_data[feature_cols]
y = np.log1p(train_data['count'])  # log(1 + y) for RMSLE optimization

# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: XGBoost
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist"
)
model.fit(X_train, y_train)

# Predict and evaluate
val_preds_log = model.predict(X_val)
val_preds = np.expm1(val_preds_log)  # inverse of log1p
val_actual = np.expm1(y_val)
val_score = np.sqrt(mean_squared_log_error(val_actual, val_preds))
print(f"Validation RMSLE: {val_score:.5f}")

# Predictions on test set
test_X = test_data[feature_cols]
test_preds = np.expm1(model.predict(test_X))  # inverse log1p

# Create submission
submission = pd.DataFrame({
    'datetime': test_data['datetime'],
    'count': test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(train_data['datetime'], train_data['count'], label="Training Data")
plt.xlabel("Datetime")
plt.ylabel("Count")
plt.title("Bike Sharing Demand Over Time")
plt.legend()
plt.show()


