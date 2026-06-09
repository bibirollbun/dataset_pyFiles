import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


# Load datasets
train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


print("\nğŸ”¹ Dataset Info:")
print(train_df.info())


print("\nğŸ”¹ Dataset Description:")
print(train_df.describe())


print("\nğŸ”¹ First 5 Rows:")
print(train_df.head())


plt.figure(figsize=(10, 5))
sns.histplot(train_df["rainfall"], bins=30, kde=True, color='blue')
plt.title("Rainfall Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Rainfall")
plt.ylabel("Frequency")
plt.show()


test_df["winddirection"] = test_df["winddirection"].fillna(test_df["winddirection"].mean())


features = ["pressure", "maxtemp", "temparature", "mintemp", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
X = train_df[features]
y = train_df["rainfall"]


poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(test_df[features])


smote = SMOTE(random_state=42)
X_poly_resampled, y_resampled = smote.fit_resample(X_poly, y)


X_train, X_val, y_train, y_val = train_test_split(X_poly_resampled, y_resampled, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test_poly)


plt.figure(figsize=(12, 6))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.show()


def create_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(512, activation='swish', kernel_regularizer=l2(0.0001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(256, activation='swish', kernel_regularizer=l2(0.0001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(128, activation='swish', kernel_regularizer=l2(0.0001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.05),
        keras.layers.Dense(64, activation='swish', kernel_regularizer=l2(0.0001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='swish', kernel_regularizer=l2(0.0001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
    return model

model = create_model()


early_stopping = keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6)

history = model.fit(X_train, y_train, epochs=300, validation_data=(X_val, y_val), batch_size=1024, callbacks=[early_stopping, reduce_lr])


plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy", fontsize=14, fontweight='bold')
plt.legend()
plt.show()


val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")


test_predictions = (model.predict(X_test) > 0.5).astype(int)


submission = pd.DataFrame({"id": test_df["id"], "rainfall": test_predictions.flatten()})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\nğŸ“‚ Submission file saved successfully!")

