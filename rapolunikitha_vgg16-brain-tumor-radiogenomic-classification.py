# Import libraries
import numpy as np
import pandas as pd
import os
import cv2
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, classification_report, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Set paths
train_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/'
labels_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv'

labels_df = pd.read_csv(labels_path)



# Load one middle slice
def load_image(patient_id, img_size=(128,128)):
    patient_folder = os.path.join(train_path, str(patient_id).zfill(5), "T1w")
    if not os.path.exists(patient_folder):
        return None
    slices = sorted(os.listdir(patient_folder))
    if len(slices) == 0:
        return None
    slice_path = os.path.join(patient_folder, slices[len(slices)//2])
    dcm = pydicom.dcmread(slice_path)
    img = dcm.pixel_array
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return img

# Load dataset
X = []
y = []

for idx, row in labels_df.iterrows():
    img = load_image(row['BraTS21ID'])
    if img is not None:
        X.append(img)
        y.append(row['MGMT_value'])

X = np.array(X)
X = np.expand_dims(X, axis=-1)  # (batch, 128,128,1)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

# Repeat channels if grayscale
X_train = np.repeat(X_train, 3, axis=-1)
X_val = np.repeat(X_val, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)


# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8,1.2]
)
datagen.fit(X_train)

# Build improved VGG16
base_model = VGG16(weights=None, include_top=False, input_shape=(128,128,3))

# Freeze VGG16 layers
for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)



# Train model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=50,
                    callbacks=[early_stop])

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Metrics
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)



# Printing Results
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"AUC: {roc_auc:.4f}")

# Plot Accuracy and Loss
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')

plt.show()



# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc_value = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


