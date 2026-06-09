import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Load dataset (replace with actual Kaggle dataset)
df = pd.read_csv("/kaggle/input/playground-series-s3e19/train.csv", parse_dates=["date"], index_col="date")
df = df.sort_index()


df.head(10)


# Visualize data
plt.figure(figsize=(12, 6))
plt.plot(df["num_sold"], label="Sales Data")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Data Over Time")
plt.legend()
plt.show()


# Data preprocessing
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["num_sold"]])


def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)


#data_array = df["num_sold"].values
seq_length = 30  # Using past 30 days to predict next day
X, y = create_sequences(df_scaled, seq_length)


print(X[1])


print(y[1])


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


print(len(X_train), len(y_train), len(X_test), len(y_test))


# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])


model.compile(optimizer='adam', loss='mse')


# Train model
history = model.fit(X_train, y_train, epochs=1, batch_size=16, validation_data=(X_test, y_test))


# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2, linestyle='--')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# Make predictions
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))


# Example prediction
example_idx = 100
example_date = df.index[len(df) - len(y_test) + example_idx]
print(f"Example Prediction Date: {example_date.strftime('%B %Y')}")
print(f"Actual Sales: {y_test_inv[example_idx][0]:.2f}")
print(f"Predicted Sales: {y_pred_inv[example_idx][0]:.2f}")


# Example prediction
example_idx = 6
example_date = df.index[len(df) - len(y_test) + example_idx]
print(f"Example Prediction Date: {example_date.strftime('%B %Y')}")
print(f"Actual Sales: {y_test_inv[example_idx][0]:.2f}")
print(f"Predicted Sales: {y_pred_inv[example_idx][0]:.2f}")


# Plot predictions
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")
plt.plot(y_test_inv, label='Actual Sales', linestyle='-', marker='o', color='blue', alpha=0.8, linewidth=2)
plt.plot(y_pred_inv, label='Predicted Sales', linestyle='--', marker='s', color='red', alpha=0.8, linewidth=2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Actual vs Predicted Sales', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()




