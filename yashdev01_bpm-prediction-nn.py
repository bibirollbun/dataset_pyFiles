import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


mixed_precision.set_global_policy("mixed_float16")


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.columns


X = train.drop(columns=['BeatsPerMinute'])
y = train['BeatsPerMinute'].values


scaler = StandardScaler()
X = scaler.fit_transform(X)


y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std


X_train, X_val, y_train, y_val = train_test_split(X, y_scaled, test_size=0.2, random_state=42)


BATCH_SIZE = 128

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(2048).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)


def build_model(input_dim):
    model = keras.Sequential([
        layers.Dense(512, activation="relu", input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, dtype="float32")  # ensure output is float32
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError()]
    )
    return model


model = build_model(X_train.shape[1])


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)


reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, verbose=1
)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    batch_size=16,   # smaller batch size often works better
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


# Convert history.history into variables
loss = history.history['loss']
val_loss = history.history['val_loss']
rmse = history.history['root_mean_squared_error']
val_rmse = history.history['val_root_mean_squared_error']
lr = history.history['learning_rate']
epochs = range(1, len(loss) + 1)


plt.figure(figsize=(8,5))
plt.plot(epochs, loss, label="Training Loss", marker='o')
plt.plot(epochs, val_loss, label="Validation Loss", marker='x')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))
plt.plot(epochs, rmse, label="Training RMSE", marker='o')
plt.plot(epochs, val_rmse, label="Validation RMSE", marker='x')
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.title("Training vs Validation RMSE")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))
plt.plot(epochs, lr, label="Learning Rate", color="purple")
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid(True)
plt.show()


# Predict on validation/test
y_val_pred_scaled = model.predict(val_ds).flatten()

# Inverse transform
y_val_pred = y_val_pred_scaled * y_std + y_mean
y_val_true = y_val * y_std + y_mean


y_pred = model.predict(X_val).flatten()
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"Validation RMSE: {rmse:.5f}")


test_scaled = scaler.transform(test)  # scale test features
test_preds_scaled = model.predict(test_scaled).flatten()


X_test = scaler.transform(test_scaled)


test_preds = model.predict(X_test).flatten()

# Inverse transform
test_preds = test_preds_scaled * y_std + y_mean


submission = pd.DataFrame({
    "id": test['id'],
    "BeatsPerMinute": test_preds 
})

submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as submission.csv")




