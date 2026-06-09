# Import libraries
import numpy as np
import pandas as pd
import os
import cv2
import pydicom
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping

# Load labels
train_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/'
labels_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv'
labels_df = pd.read_csv(labels_path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings (INFO and WARNING)
warnings.filterwarnings('ignore')  # Suppress Python warnings in general



# Load images
def load_image(patient_id, img_size=(150,150)):
    folder = os.path.join(train_path, str(patient_id).zfill(5), "T1w")
    if not os.path.exists(folder): return None
    files = sorted(os.listdir(folder))
    if not files: return None
    dcm = pydicom.dcmread(os.path.join(folder, files[len(files)//2]))
    img = dcm.pixel_array
    img = cv2.resize(img, img_size)
    return img / 255.0

X, y = [], []
for _, row in labels_df.iterrows():
    img = load_image(row['BraTS21ID'])
    if img is not None:
        X.append(img)
        y.append(row['MGMT_value'])

X = np.expand_dims(np.array(X), axis=-1)  # Shape: (N,150,150,1)
y = np.array(y)




# Prepare dataset
X = np.repeat(X, 3, axis=-1)  # InceptionV3 expects 3 channels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)



plt.figure(figsize=(12,8))
for i in range(9):
    idx = np.random.randint(0, len(X_train))
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[idx])
    plt.title(f"Label: {y_train[idx]}")
    plt.axis('off')
plt.suptitle('Sample Training MRI Slices', fontsize=16)
plt.tight_layout()
plt.show()


# Data augmentation
aug = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
aug.fit(X_train)



# Build InceptionV3 model
base_model = InceptionV3(weights=None, include_top=False, input_shape=(150,150,3))

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)




# Train model
history = model.fit(
    aug.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=30,
    callbacks=[early_stop]
)



# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = (model.predict(X_test) > 0.5).astype("int32")



# Metrics
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)


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
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()




# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_val = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_val:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - InceptionV3')
plt.legend()
plt.show()

