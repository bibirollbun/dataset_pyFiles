import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# =========================
# Load & preprocess data
# =========================
df = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
X = df.drop(['yield', 'field_id','humidity','ndvi'], axis=1)
y = df['yield'].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# =========================
# Build Model
# =========================
def build_model():
    model = models.Sequential([
        layers.Dense(128, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001), input_shape=(X_train_scaled.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model

model = build_model()


model.summary()


# =========================
# Callbacks
# =========================
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

lr_reduce = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1)


# =========================
# Train Model
# =========================
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=150,
    batch_size=64,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)


# =========================
# Plot Loss & MAE
# =========================
plt.figure(figsize=(14, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()

# MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Mean Absolute Error Curve')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predict on validation set
val_preds = model.predict(X_val_scaled).flatten()

# Evaluate
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
mae = mean_absolute_error(y_val, val_preds)
r2 = r2_score(y_val, val_preds)

print(f"\n✅ Final Validation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")


# Predict on scaled test
test = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')
test_preds = model.predict(X_val_scaled).flatten()

# Save to CSV
pd.DataFrame({
    "field_id": test["field_id"],
    "yield": test_preds
}).to_csv("predicted_yield_tf_1.csv", index=False)


