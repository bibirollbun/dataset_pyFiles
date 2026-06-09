import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
# Load the dataset
train_data = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test_data = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

# Check the data
print(train_data.head())
print(test_data.head())
# Preprocess the data
def preprocess_data(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['dayofweek'] = df['datetime'].dt.dayofweek
    return df

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Features and target
X = train_data[['hour', 'day', 'month', 'year', 'dayofweek']]
y = train_data['count']
# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(max_depth=10, max_features='sqrt', max_samples=0.6, random_state=42)
model.fit(X_train, y_train)

# Validation predictions
val_preds = model.predict(X_val)
val_score = np.sqrt(mean_squared_log_error(y_val, val_preds))
print(f"Validation RMSLE: {val_score}")
# Predictions on test data
test_features = test_data[['hour', 'day', 'month', 'year', 'dayofweek']]
test_preds = model.predict(test_features)

# Save predictions
submission = pd.DataFrame({
    'datetime': test_data['datetime'],
    'count': test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
plt.figure(figsize=(10, 5))
plt.plot(train_data['datetime'], train_data['count'], label="Training Data")
plt.xlabel("Datetime")
plt.ylabel("Count")
plt.title("Bike Sharing Demand Over Time")
plt.legend()
plt.show()

