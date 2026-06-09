import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
train_data


test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
test_data


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

y_train = train_data["accident_risk"]

features = ["id", "road_type", "num_lanes", 
            "curvature", "speed_limit", "lighting",
            "weather", "road_signs_present", 
            "public_road", "time_of_day", "holiday",
            "school_season", "num_reported_accidents"]

X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Create and train the model (on an absurd number of estimators OVERFIT WE GOOOO ðŸ˜­)
model_ML = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=1)


model_ML.fit(X_train, y_train)

# Make predictions
predictions_ML = model_ML.predict(X_test)


from sklearn.metrics import mean_squared_error
train_rmse = mean_squared_error(y_train, model_ML.predict(X_train), squared=False)
test_rmse = mean_squared_error(y_test, model_ML.predict(X_test), squared=False)
print(f"Mean Squared Error of the Training Data: {train_rmse}")
print(f"Mean Squared Error of the Testing Data: {test_rmse}")


predictions_ML


y_train = train_data["accident_risk"]
features = ["id", "road_type", "num_lanes", 
            "curvature", "speed_limit", "lighting",
            "weather", "road_signs_present", 
            "public_road", "time_of_day", "holiday",
            "school_season", "num_reported_accidents"]
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
model_ML = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=1)
model_ML.fit(X_train, y_train)
predictions_ML = model_ML.predict(X_test)


import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Get RandomForestRegressorPredictions
rf_train_preds = model_ML.predict(X_train).reshape(-1, 1)
rf_test_preds = model_ML.predict(X_test).reshape(-1, 1)

# Combine RF predictions with original features
X_train_combined = np.hstack((X_train, rf_train_preds))
X_test_combined = np.hstack((X_test, rf_test_preds))

# Scale features for NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)

X_train_scaled.shape


# --- Build Sequential Neural Network ---
model_NN = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Regression output (no activation)
])

EarlyStopping(
    monitor="val_mae",
    patience=5,
    restore_best_weights=True,
)
model_NN.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_NN.fit(X_train_scaled, y_train, epochs=15, validation_split=0.1)


# --- Predict ---
predictions_NN = model_NN.predict(X_test_scaled).flatten()


predictions_NN


output = pd.DataFrame({
    "id": X_test["id"],  # assuming your test data has an 'id' column
    "accident_risk": predictions_NN
})

output.to_csv("submission.csv", index=False)




