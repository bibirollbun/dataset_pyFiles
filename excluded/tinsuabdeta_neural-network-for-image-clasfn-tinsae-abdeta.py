# ==============================
# Assignment 1 - Kannada MNIST
# Neural Network (Fully Connected)
# Author: Tinsae Abdeta
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------
# 1. Load the dataset (Kaggle path)
# --------------------------
train_df = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
test_df  = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

# Training features and labels
X = train_df.drop("label", axis=1).values
y = train_df["label"].values

# Handle test set (drop 'id' if present)
if "id" in test_df.columns:
    X_test = test_df.drop("id", axis=1).values
else:
    X_test = test_df.values

print("Train shape:", X.shape)
print("Test shape:", X_test.shape)

# --------------------------
# 2. Preprocess the data
# --------------------------
# Normalize pixel values to [0,1]
X = X / 255.0
X_test = X_test / 255.0

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# One-hot encode labels
y_train_encoded = to_categorical(y_train, num_classes=10)
y_val_encoded   = to_categorical(y_val, num_classes=10)

# --------------------------
# 3. Build the model
# --------------------------
model = Sequential([
    Input(shape=(784,)),  # Explicit input layer
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes (digits 0–9)
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# --------------------------
# 4. Train the model
# --------------------------
history = model.fit(
    X_train, y_train_encoded,
    validation_data=(X_val, y_val_encoded),
    epochs=20,
    batch_size=128,
    verbose=2
)

# --------------------------
# 5. Evaluate performance
# --------------------------
val_loss, val_acc = model.evaluate(X_val, y_val_encoded, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")

# Plot accuracy and loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Model Accuracy")
plt.xlabel("Epochs"); plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Model Loss")
plt.xlabel("Epochs"); plt.ylabel("Loss")
plt.legend()
plt.show()

# Confusion matrix on validation set
y_val_pred = np.argmax(model.predict(X_val), axis=1)
conf_matrix = confusion_matrix(y_val, y_val_pred)

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Validation Confusion Matrix")
plt.show()

print("Classification Report:\n", classification_report(y_val, y_val_pred))

# --------------------------
# 6. Kaggle Submission
# --------------------------
pred_probs = model.predict(X_test)
pred_labels = np.argmax(pred_probs, axis=1)

# If test set has 'id' column, use it, else generate sequential IDs
if "id" in test_df.columns:
    submission = pd.DataFrame({
        "id": test_df["id"],
        "label": pred_labels
    })
else:
    submission = pd.DataFrame({
        "id": np.arange(len(pred_labels)),
        "label": pred_labels
    })

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv for Kaggle upload!")


