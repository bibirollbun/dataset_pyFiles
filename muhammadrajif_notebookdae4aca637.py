import os
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns


# Path dataset
path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/'
train_labels = pd.read_csv(path + 'train_labels.csv')

# Konfigurasi
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
MODALITY = 'FLAIR'
NUM_SLICES = 16
TEST_SIZE = 0.15
VAL_SIZE = 0.15   


# Fungsi untuk membaca dan memproses gambar DICOM
def load_dicom_image(filepath, img_size=IMG_SIZE):
    dicom = pydicom.dcmread(filepath)
    img = dicom.pixel_array.astype(float)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = cv2.resize(img.astype(np.uint8), (img_size, img_size))
    return np.stack([img]*3, axis=-1)  # Convert to 3-channel

# Fungsi untuk memuat data pasien
def load_patient_data(patient_id, num_slices=NUM_SLICES):
    patient_path = os.path.join(path, 'train', str(patient_id).zfill(5), MODALITY)
    if not os.path.exists(patient_path): 
        return None
    
    dicom_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.dcm')])
    if not dicom_files: 
        return None
    
    step = max(1, len(dicom_files) // num_slices)
    selected_files = dicom_files[::step][:num_slices]
    
    slices = []
    for filename in selected_files:
        img = load_dicom_image(os.path.join(patient_path, filename))
        slices.append(img)
    
    # Padding jika slice kurang
    while len(slices) < num_slices:
        slices.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    
    return np.array(slices)


# Membuat dataset
X, y = [], []
print("Memuat data training...")
for idx, row in train_labels.iterrows():
    patient_data = load_patient_data(row['BraTS21ID'])
    if patient_data is not None:
        X.append(patient_data)
        y.append(row['MGMT_value'])

X = np.array(X, dtype=np.float32) / 255.0  # Normalisasi
y = np.array(y, dtype=np.float32)
print(f"Shape data: {X.shape}, Distribusi kelas: MGMT+={sum(y==1)}, MGMT-={sum(y==0)}")


# Split data menjadi train, validation, dan test
# train (85%) dan test (15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)


# Bagi train_val menjadi train dan validation
val_ratio = VAL_SIZE / (1 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=42
)


print(f"\nSplit Dataset:")
print(f"Train:      {X_train.shape[0]} sampel ({(X_train.shape[0]/len(X))*100:.1f}%)")
print(f"Validation: {X_val.shape[0]} sampel ({(X_val.shape[0]/len(X))*100:.1f}%)")
print(f"Test:       {X_test.shape[0]} sampel ({(X_test.shape[0]/len(X))*100:.1f}%)")


# 1. MODEL 3D CNN SEDERHANA
def build_3d_cnn(input_shape=(NUM_SLICES, IMG_SIZE, IMG_SIZE, 3)):
    model = models.Sequential([
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.4),
        
        layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.5),
        
        layers.GlobalAveragePooling3D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


# 2. MODEL 3D EFFICIENTNET B0
def build_3d_efficientnet(input_shape=(NUM_SLICES, IMG_SIZE, IMG_SIZE, 3)):
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze initial layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    slice_outputs = []
    
    for i in range(NUM_SLICES):
        slice = layers.Lambda(lambda x: x[:, i, :, :, :])(inputs)
        x = base_model(slice)
        x = layers.GlobalAveragePooling2D()(x)
        slice_outputs.append(x)
    
    x = layers.Concatenate(axis=1)(slice_outputs)
    x = layers.Reshape((NUM_SLICES, -1))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


# 3. MODEL 3D RESNET50
def build_3d_resnet(input_shape=(NUM_SLICES, IMG_SIZE, IMG_SIZE, 3)):
    def residual_block(x, filters, stride=1):
        shortcut = x
        
        x = layers.Conv3D(filters, (3, 3, 3), strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv3D(filters, (3, 3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv3D(filters, (1, 1, 1), strides=stride)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(64, (7, 7, 7), strides=(2, 2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model



# FUNGSI EVALUASI MODEL
def evaluate_model(model, X_test, y_test, name):
    print(f"\n{'='*50}")
    print(f"Evaluasi Model {name} pada Data Test")
    print(f"{'='*50}")
    
    # Prediksi
    y_pred = model.predict(X_test)
    y_pred_bin = (y_pred > 0.5).astype(int)
    
    # Classification Report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_bin))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_bin)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Unmethylated', 'Methylated'], 
                yticklabels=['Unmethylated', 'Methylated'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{name} - Confusion Matrix (Test Set)')
    plt.savefig(f'{name}_cm_test.png')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} - ROC Curve (Test Set)')
    plt.legend(loc="lower right")
    plt.savefig(f'{name}_roc_test.png')
    plt.show()
    
    return roc_auc


# TRAINING DAN VALIDASI MODEL
def train_and_evaluate(model, name, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n{'='*50}")
    print(f"Training {name} Model")
    print(f"{'='*50}")
    model.summary()
    
    # Callback
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(f'best_{name}.h5', save_best_only=True, monitor='val_auc', mode='max')
    ]
    
    # Class weighting
    class_weight = {0: 1., 1: len(y_train[y_train==0])/len(y_train[y_train==1])}
    
    # Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight
    )
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{name} - Loss Curve')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title(f'{name} - AUC Curve')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{name} - Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{name}_training_curves.png')
    plt.show()
    
    # Evaluasi pada validation set
    val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Metrics ({name}):")
    print(f"Loss:     {val_loss:.4f}")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"AUC:      {val_auc:.4f}")
    
    # Evaluasi pada test set
    test_auc = evaluate_model(model, X_test, y_test, name)
    
    return model, history, val_auc, test_auc


# =============================
# TRAIN DAN EVALUASI SEMUA MODEL
# =============================
results = []

# Bangun dan latih model
models_dict = {
    "3D_CNN": build_3d_cnn(),
    "3D_EfficientNet": build_3d_efficientnet(),
    "3D_ResNet": build_3d_resnet()
}

for name, model in models_dict.items():
    trained_model, history, val_auc, test_auc = train_and_evaluate(
        model, name, X_train, y_train, X_val, y_val, X_test, y_test
    )
    results.append({
        'Model': name,
        'Validation AUC': val_auc,
        'Test AUC': test_auc
    })
    trained_model.save(f"{name}_model.h5")

# Tampilkan ringkasan hasil
results_df = pd.DataFrame(results)
print("\nRingkasan Hasil Evaluasi Model:")
print(results_df)

# Plot perbandingan performa model
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Validation AUC'], alpha=0.7, label='Validation AUC')
plt.bar(results_df['Model'], results_df['Test AUC'], alpha=0.7, label='Test AUC')
plt.ylabel('AUC Score')
plt.title('Perbandingan Performa Model')
plt.ylim(0.5, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('model_comparison.png')
plt.show()

