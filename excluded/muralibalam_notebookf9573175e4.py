# ===== GOOGLE AI CAPSTONE – BEGINNER NOTEBOOK (SINGLE PAGE CODE) =====

# Install (colab already has most packages)
!pip install -q tensorflow pandas scikit-learn matplotlib

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)

# ===== Load dataset (using IRIS for beginner example) =====
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target

data = pd.DataFrame(X, columns=iris.feature_names)
data["target"] = y

print("Sample data:")
print(data.head())

# ===== Check data =====
print("\nShape:", data.shape)
print("Missing values:\n", data.isnull().sum())

# ===== Prepare features & labels =====
TARGET_COLUMN = "target"
X = data.drop(columns=[TARGET_COLUMN]).values
y = data[TARGET_COLUMN].values

# ===== Train / Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== Scaling =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== Build Model =====
input_dim = X_train_scaled.shape[1]
num_classes = len(np.unique(y))

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

# ===== Train the Model =====
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    verbose=1
)

# ===== Plot Accuracy =====
plt.figure()
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy")
plt.show()

# ===== Plot Loss =====
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.show()

# ===== Evaluate =====
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print("\nTest Accuracy:", test_acc)

y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===== Save the Model =====
model.save("capstone_model.h5")
print("\nModel saved as capstone_model.h5")

# ===== END =====


