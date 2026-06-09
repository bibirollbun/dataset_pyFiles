# ====================================================
# 1. Import Libraries
# ====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# ====================================================
# 2. Load Image Dataset (Kaggle Input Folder)
# ====================================================

data_path = "/kaggle/input/your-dataset-folder"   # <<< CHANGE THIS PATH

classes = os.listdir(data_path)
print("Classes found:", classes)

images = []
labels = []

for label, folder in enumerate(classes):
    folder_path = os.path.join(data_path, folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        # Load image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))

        images.append(img)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

print("Total Images:", len(images))

# ====================================================
# 3. Preprocessing
# ====================================================

images = images / 255.0                # Normalize
labels = to_categorical(labels)        # One-hot encoding

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ====================================================
# 4. Build a Simple CNN Model
# ====================================================

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(classes), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ====================================================
# 5. Train the Model
# ====================================================

history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2
)

# ====================================================
# 6. Evaluate the Model
# ====================================================

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Accuracy:", accuracy_score(y_true, y_pred_labels))

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ====================================================
# 7. Save the Model
# ====================================================

model.save("card_thumbnail_model.h5")
print("Model Saved Successfully!")

# ====================================================
# 8. Prediction on a New Image
# ====================================================

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label_index = np.argmax(prediction)

    return classes[label_index]

# Example:
# print(predict_image("/kaggle/input/your-dataset-folder/card1/img.png"))


