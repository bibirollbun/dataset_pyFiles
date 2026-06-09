!pip install -q pydicom

# ✅ Imports
import os
import numpy as np
import pandas as pd
import cv2
import pydicom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ✅ Load MGMT labels
train_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/'
labels_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv'
labels_df = pd.read_csv(labels_path)

# ✅ Load one center slice per patient
def load_image(patient_id, img_size=(224, 224)):
    folder = os.path.join(train_path, str(patient_id).zfill(5), "T1w")
    if not os.path.exists(folder): return None
    files = sorted(os.listdir(folder))
    if not files: return None
    path = os.path.join(folder, files[len(files)//2])
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array
    img = cv2.resize(img, img_size)
    return img / 255.0

X, y = [], []
for _, row in labels_df.iterrows():
    img = load_image(row['BraTS21ID'])
    if img is not None:
        X.append(img)
        y.append(row['MGMT_value'])

X = np.expand_dims(np.array(X), -1)      # (N, 224, 224, 1)
X = np.repeat(X, 3, axis=-1)             # (N, 224, 224, 3) for VGG19
y = np.array(y)

# ✅ Train/val/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

# ✅ Data augmentation
aug = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
aug.fit(X_train)

# ✅ Build VGG19 model (from scratch)
base_model = VGG19(weights=None, include_top=False, input_shape=(224,224,3))

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ✅ Train model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    aug.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10,
    callbacks=[early_stop]
)

# ✅ Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"AUC: {roc_auc:.4f}")




# ✅ Accuracy & Loss Plots
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# ✅ ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_val = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_val:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - VGG19')
plt.legend()
plt.show()

# ✅ Sample MRI images
plt.figure(figsize=(12, 8))
for i in range(9):
    idx = np.random.randint(0, len(X_train))
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[idx].squeeze(), cmap='gray')
    plt.title(f"Label: {y_train[idx]}")
    plt.axis('off')
plt.suptitle("Sample MRI Slices (VGG19 Input)")
plt.tight_layout()
plt.show()



