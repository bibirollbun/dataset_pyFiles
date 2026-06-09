!pip install -q pydicom

import numpy as np
import pandas as pd
import os, cv2, pydicom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, VGG19, Xception, InceptionV3, InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Load Data
labels_df = pd.read_csv('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv')
train_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/'

def load_image(patient_id, img_size=(224,224)):
    folder = os.path.join(train_path, str(patient_id).zfill(5), "T1w")
    files = sorted(os.listdir(folder)) if os.path.exists(folder) else []
    if len(files) == 0: return None
    path = os.path.join(folder, files[len(files)//2])
    dcm = pydicom.dcmread(path)
    img = cv2.resize(dcm.pixel_array, img_size) / 255.0
    return img

X, y = [], []
for _, row in labels_df.iterrows():
    img = load_image(row['BraTS21ID'])
    if img is not None:
        X.append(img)
        y.append(row['MGMT_value'])

X = np.expand_dims(np.array(X), -1)
X = np.repeat(X, 3, axis=-1)
y = np.array(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

# Augmentation
aug = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                         zoom_range=0.1, horizontal_flip=True)
aug.fit(X_train)

# Model Builder
def build_model(base, input_shape=(224,224,3)):
    base_model = base(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
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
    return model

# Train Models & Predict
histories = {}
predictions = {}

for name, arch in {
    'VGG16': VGG16,
    'VGG19': VGG19,
    'Xception': Xception,
    'InceptionV3': InceptionV3,
    'InceptionResNetV2': InceptionResNetV2
}.items():
    print(f"\nTraining {name}...")
    model = build_model(arch)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        aug.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=30,
        callbacks=[early_stop],
        verbose=1
    )
    histories[name] = history
    predictions[name] = model.predict(X_test).flatten()

# Ensemble Predictions & Metrics
ensemble_probs = sum(predictions.values()) / len(predictions)
ensemble_preds = (ensemble_probs > 0.5).astype(int)

fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
roc_auc = auc(fpr, tpr)
acc = accuracy_score(y_test, ensemble_preds)
f1 = f1_score(y_test, ensemble_preds)
kappa = cohen_kappa_score(y_test, ensemble_preds)

# Metric Summary
print("\n===== Fuzzy Ensemble Evaluation Metrics =====")
print(f"Test Accuracy     : {acc:.4f}")
print(f"F1 Score          : {f1:.4f}")
print(f"Cohen’s Kappa     : {kappa:.4f}")
print(f"AUC               : {roc_auc:.4f}")

print("\nClassification Report:\n", classification_report(y_test, ensemble_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, ensemble_preds))

# Training Summary Table 
results = []
for name, history in histories.items():
    results.append({
        'Model': name,
        'Train Accuracy': round(history.history['accuracy'][-1], 4),
        'Validation Accuracy': round(history.history['val_accuracy'][-1], 4),
        'Train Loss': round(history.history['loss'][-1], 4),
        'Validation Loss': round(history.history['val_loss'][-1], 4)
    })

df_results = pd.DataFrame(results)
print("\n=== Training & Validation Summary ===")
print(df_results.to_string(index=False))

# === 10. ROC Curve ===
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'Fuzzy Ensemble ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Fuzzy Ensemble (New Method II)')
plt.legend()
plt.grid()
plt.show()

# Accuracy/Loss Curves
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=name)
plt.title('Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
for name, history in histories.items():
    plt.plot(history.history['val_loss'], label=name)
plt.title('Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.suptitle('Training Curves for New Method II - 5 CNN Models')
plt.tight_layout()
plt.show()


