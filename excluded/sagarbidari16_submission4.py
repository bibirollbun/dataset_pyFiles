import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Load data
train_data = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test_data = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

# Feature engineering
def preprocess(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if x in [7, 8, 17, 18] else 0)
    df['temp_atemp_diff'] = df['atemp'] - df['temp']
    df['year_month'] = df['year'] * 100 + df['month']
    return df

train_data = preprocess(train_data)
test_data = preprocess(test_data)

# Target and features
target = np.log1p(train_data['count'])  # Use log1p for RMSLE optimization
features = ['hour', 'day', 'month', 'year', 'dayofweek', 'is_weekend', 'is_peak_hour',
            'temp', 'atemp', 'humidity', 'windspeed', 'weather', 'season',
            'holiday', 'workingday', 'temp_atemp_diff', 'year_month']

X = train_data[features]
X_test = test_data[features]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, target, test_size=0.2, random_state=42)

# LightGBM model (no early stopping to avoid version issues)
model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Validate
val_preds_log = model.predict(X_val)
val_preds = np.expm1(val_preds_log)
val_actual = np.expm1(y_val)
val_score = np.sqrt(mean_squared_log_error(val_actual, val_preds))
print(f"Validation RMSLE: {val_score:.5f}")

# Predict on test set
test_preds = np.expm1(model.predict(X_test))

# Save submission
submission = pd.DataFrame({
    'datetime': test_data['datetime'],
    'count': test_preds
})
submission.to_csv("submission_lgbm.csv", index=False)
print("Submission file created: submission_lgbm.csv")

# Plot demand over time
plt.figure(figsize=(10, 5))
plt.plot(train_data['datetime'], train_data['count'], label="Training Data")
plt.xlabel("Datetime")
plt.ylabel("Count")
plt.title("Bike Sharing Demand Over Time")
plt.legend()
plt.show()


