import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pydicom as dicom
import cv2
import ast

import warnings
warnings.filterwarnings("ignore")


path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/'
os.listdir(path)


path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/'
os.listdir(path)
train_data = pd.read_csv(path+'train_labels.csv')
samp_subm = pd.read_csv(path+'sample_submission.csv')


print('Samples train:', len(train_data))
print('Samples test:', len(samp_subm))


train_data.head()


train_data["MGMT_value"].value_counts().head(2).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


train_data["MGMT_value"].value_counts()


samp_subm.head()


folder = str(train_data.loc[0, 'BraTS21ID']).zfill(5)
folder


os.listdir(path+'train/'+folder)


print('Number of FLAIR images:', len(os.listdir(path+'train/'+folder+'/'+'FLAIR')))
print('Number of T1w images:', len(os.listdir(path+'train/'+folder+'/'+'T1w')))
print('Number of T1wCE images:', len(os.listdir(path+'train/'+folder+'/'+'T1wCE')))
print('Number of T2w images:', len(os.listdir(path+'train/'+folder+'/'+'T2w')))


path_file = ''.join([path, 'train/', folder, '/', 'FLAIR/'])
image = os.listdir(path_file)[0]
data_file = dicom.dcmread(path_file+image)
img = data_file.pixel_array
print('Image shape:', img.shape)


#Flair Image
def plot_examples(row = 0, cat = 'FLAIR'): 
    folder = str(train_data.loc[row, 'BraTS21ID']).zfill(5)
    path_file = ''.join([path, 'train/', folder, '/', cat, '/'])
    images = os.listdir(path_file)
    
    fig, axs = plt.subplots(1, 5, figsize=(30, 30))
    fig.subplots_adjust(hspace = .2, wspace=.2)
    axs = axs.ravel()
    
    for num in range(5):
        data_file = dicom.dcmread(path_file+images[num])
        img = data_file.pixel_array
        axs[num].imshow(img, cmap='gray')
        axs[num].set_title(cat+' '+images[num])
        axs[num].set_xticklabels([])
        axs[num].set_yticklabels([])
        
row = 0
plot_examples(row = row, cat = 'FLAIR')
plt.show()


#T1w Images
plot_examples(row = row, cat = 'T1w')
plt.show()


#T1wCE Images
plot_examples(row = row, cat = 'T1wCE')
plt.show()


#T2w Images
plot_examples(row = row, cat = 'T2w')
plt.show()


import os
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Path dataset
path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/'
train_labels = pd.read_csv(path + 'train_labels.csv')

# Konfigurasi
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
MODALITY = 'FLAIR' 

# Fungsi untuk membaca dan memproses gambar DICOM
def load_dicom_image(filepath, img_size=IMG_SIZE):
    dicom = pydicom.dcmread(filepath)
    img = dicom.pixel_array.astype(float)
    
    # Normalisasi
    img = (img - img.min()) / (img.max() - img.min())
    
    # Konversi ke uint8
    img = (img * 255).astype(np.uint8)
    
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    
    # Stack ke 3 channel
    img = np.stack([img]*3, axis=-1)
    return img

# Fungsi untuk memuat data pasien
def load_patient_data(patient_id, num_slices=16):
    patient_path = os.path.join(path, 'train', str(patient_id).zfill(5), MODALITY)
    slices = []
    
    if not os.path.exists(patient_path):
        print(f"Data tidak ditemukan untuk pasien {patient_id}")
        return None
    
    # Dapatkan semua file DICOM
    dicom_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.dcm')])
    
    if not dicom_files:
        print(f"Tidak ada file DICOM untuk pasien {patient_id}")
        return None
    
    # Pilih slice secara merata
    step = max(1, len(dicom_files) // num_slices)
    selected_files = dicom_files[::step][:num_slices]
    
    # Muat slice yang dipilih
    for filename in selected_files:
        img_path = os.path.join(patient_path, filename)
        img = load_dicom_image(img_path)
        slices.append(img)
    
    # Jika tidak cukup slice, duplikat yang terakhir
    while len(slices) < num_slices:
        slices.append(slices[-1].copy())  # Gunakan copy untuk menghindari reference yang sama
    
    return np.array(slices)


# Membuat dataset
X = []
y = []

print("Memuat data training...")
for idx, row in train_labels.iterrows():
    patient_id = row['BraTS21ID']
    label = row['MGMT_value']
    
    patient_data = load_patient_data(patient_id)
    if patient_data is not None:
        X.append(patient_data)
        y.append(label)

# Konversi ke numpy array
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"Total data yang dimuat: {len(X)} sampel")
print(f"Distribusi kelas: {np.sum(y == 1)} positif, {np.sum(y == 0)} negatif")


# Split data: training (60%), validation (20%), test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("\nDistribusi dataset:")
print(f"Training:   {len(X_train)} sampel")
print(f"Validation: {len(X_val)} sampel")
print(f"Test:       {len(X_test)} sampel")


# Arsitektur model CNN 3D
def build_3d_cnn(input_shape, num_classes):
    model = models.Sequential([
        # Blok konvolusi 1
        layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.2),
        
        # Blok konvolusi 2
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.3),
        
        # Blok konvolusi 3
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.4),
        
        layers.GlobalAveragePooling3D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model


# Bangun model
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])
print(f"\nInput shape: {input_shape}")
model = build_3d_cnn(input_shape, num_classes=1)

# Ringkasan model
model.summary()


# Kompilasi model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Callback
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5, 
        monitor='val_auc', 
        mode='max', 
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        save_best_only=True,
        monitor='val_auc',
        mode='max',
        verbose=1
    )
]


#Training
print("\nMemulai training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Plot history training
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot AUC
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_history(history)


# Evaluasi pada validation set
print("\nEvaluasi pada validation set:")
val_loss, val_acc, val_auc = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation AUC: {val_auc:.4f}")

# Evaluasi pada test set
print("\nEvaluasi pada test set:")
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")


# Prediksi pada test set
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)


# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()


# Simpan model akhir
model.save('final_model.h5')
print("\nModel akhir disimpan sebagai 'final_model.h5'")

