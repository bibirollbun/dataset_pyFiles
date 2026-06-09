import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"


train = pd.read_csv(train_path, index_col='id')
test = pd.read_csv(test_path, index_col='id')
submission = pd.read_csv(submission_path)


print("\nTrain Data Info:")
print(train.info())
print("\nTest Data Info:")
print(test.info())


test.fillna(test.median(), inplace=True) 


print("Statistical Summary of Train Data:")
print(train.describe())


plt.figure(figsize=(10, 6))
sns.histplot(train['rainfall'], bins=30, kde=True)
plt.title("Distribution of Rainfall")
plt.xlabel("Rainfall")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


X = train.drop(columns=['rainfall'])
y = train['rainfall']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test)


xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=5, random_state=42)
xgb_model.fit(X_train_scaled, y_train)


y_pred_xgb = xgb_model.predict(X_val_scaled)
y_pred_xgb_rounded = np.round(y_pred_xgb) 
mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
mse_xgb = mean_squared_error(y_val, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_val, y_pred_xgb)
accuracy_xgb = accuracy_score(y_val, y_pred_xgb_rounded)
print(f"\nXGBoost Model Evaluation:\nMAE: {mae_xgb:.4f}, MSE: {mse_xgb:.4f}, RMSE: {rmse_xgb:.4f}, R2: {r2_xgb:.4f}, Accuracy: {accuracy_xgb:.4f}")



nn_model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])


nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])


history = nn_model.fit(X_train_scaled, y_train, epochs=40, batch_size=32, validation_data=(X_val_scaled, y_val), verbose=1)


y_pred_nn = nn_model.predict(X_val_scaled).flatten()
y_pred_nn_rounded = np.round(y_pred_nn)
mae_nn = mean_absolute_error(y_val, y_pred_nn)
mse_nn = mean_squared_error(y_val, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(y_val, y_pred_nn)
accuracy_nn = accuracy_score(y_val, y_pred_nn_rounded)
print(f"\nNeural Network Model Evaluation:\nMAE: {mae_nn:.4f}, MSE: {mse_nn:.4f}, RMSE: {rmse_nn:.4f}, R2: {r2_nn:.4f}, Accuracy: {accuracy_nn:.4f}")



plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


lstm_model = keras.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)),
    layers.Dropout(0.3),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])


lstm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])


history = lstm_model.fit(X_train_scaled, y_train, epochs=40, batch_size=64, validation_data=(X_val_scaled, y_val), verbose=1)


y_pred_nn = lstm_model.predict(X_val_scaled).flatten()
y_pred_nn_rounded = np.round(y_pred_nn)
mae_nn = mean_absolute_error(y_val, y_pred_nn)
mse_nn = mean_squared_error(y_val, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(y_val, y_pred_nn)
accuracy_nn = accuracy_score(y_val, y_pred_nn_rounded)
print(f"\nNeural Network Model Evaluation:\nMAE: {mae_nn:.4f}, MSE: {mse_nn:.4f}, RMSE: {rmse_nn:.4f}, R2: {r2_nn:.4f}, Accuracy: {accuracy_nn:.4f}")



plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


predictions = xgb_model.predict(X_test_scaled)
predictions


submission['rainfall'] = predictions
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved successfully!")

