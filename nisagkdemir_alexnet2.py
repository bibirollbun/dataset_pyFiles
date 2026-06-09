import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


IMG_SIZE = 227 
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-5 

DICOM_DATA_DIR = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images"
LABELS_CSV = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
PNG_OUTPUT_DIR = "/kaggle/working/rsna_pneumonia_png_images"


os.makedirs(PNG_OUTPUT_DIR, exist_ok=True) # Çıktı dizinini oluştur

# Etiket DataFrame'ini yükle
df_labels = pd.read_csv(LABELS_CSV)
patient_ids = df_labels['patientId'].unique()

print(f"{len(patient_ids)} adet DICOM dosyası PNG'ye dönüştürülüyor...")

for i, patient_id in enumerate(patient_ids):
    dicom_path = os.path.join(DICOM_DATA_DIR, patient_id + ".dcm")
    output_path = os.path.join(PNG_OUTPUT_DIR, patient_id + ".png")

    # Eğer dosya zaten dönüştürülmüşse atla
    if os.path.exists(output_path):
        continue

    try:
        dicom = pydicom.dcmread(dicom_path)
        img = dicom.pixel_array.astype(np.float32)

        # DICOM Pencereleme Uygula
        if 'WindowCenter' in dicom and 'WindowWidth' in dicom:
            window_center = dicom.WindowCenter
            window_width = dicom.WindowWidth

            # Birden fazla pencere ayarı varsa ilkini kullan
            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = window_center[0]
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = window_width[0]

            min_val = window_center - window_width / 2
            max_val = window_center + window_width / 2

            img = np.clip(img, min_val, max_val)
            # 0-255 aralığına ölçekle (PNG olarak kaydetmek için)
            img = ((img - min_val) / (max_val - min_val + 1e-5)) * 255
        else:
            # Pencere bilgisi yoksa veya tanımsızsa basit min-max normalizasyona geri dön
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5) * 255

        img = img.astype(np.uint8)
        
        # Görüntüyü yeniden boyutlandır 
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # PNG olarak kaydet
        cv2.imwrite(output_path, img)

    except Exception as e:
        print(f"Hata: {patient_id}.dcm dönüştürülürken hata oluştu: {e}")
        continue
    
    if (i + 1) % 1000 == 0:
        print(f"{i + 1} görüntü dönüştürüldü.")

print("Tüm DICOM dosyaları PNG'ye dönüştürüldü.")


df = pd.read_csv(LABELS_CSV)
df = df.drop_duplicates(subset="patientId")[['patientId', 'Target']]
df["filename"] = df["patientId"] + ".png"

sns.countplot(x=df["Target"])
plt.title("Sınıf Dağılımı")
plt.show()


# Eğitim-Doğrulama-Test Ayırma
train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Target"], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1, stratify=train_val_df["Target"], random_state=42)

train_df['Target'] = train_df['Target'].astype(str)
val_df['Target'] = val_df['Target'].astype(str)
test_df['Target'] = test_df['Target'].astype(str)

print(f"Eğitim seti boyutu: {len(train_df)}")
print(f"Doğrulama seti boyutu: {len(val_df)}")
print(f"Test seti boyutu: {len(test_df)}")


# Sınıf ağırlıkları
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_df["Target"]), y=train_df["Target"])
class_weights = dict(enumerate(class_weights))
print("Sınıf Ağırlıkları (Class Weights):", class_weights)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,       # Maksimum 15 derece döndürme
    zoom_range=0.15,         # %15'e kadar yakınlaştırma
    width_shift_range=0.1,   # %10 genişlik kaydırma
    height_shift_range=0.1,  # %10 yükseklik kaydırma
    horizontal_flip=True,    # Yatay çevirme 
    fill_mode='nearest'      # Boş kalan pikselleri en yakın değerle doldur
)

# Doğrulama ve test için sadece yeniden ölçeklendirme yapılır, veri artırma uygulanmaz.
val_test_datagen = ImageDataGenerator(rescale=1./255)

# GENERATORLAR (flow_from_dataframe kullanılarak)
# directory: Dönüştürülmüş PNG'lerin bulunduğu dizini gösterir.
# x_col: DataFrame'deki dosya adlarını içeren sütun adı.
# y_col: DataFrame'deki etiketleri içeren sütun adı.
# target_size: Görüntülerin yeniden boyutlandırılacağı hedef boyut.
# class_mode: İkili sınıflandırma (0 veya 1) olduğu için 'binary'.
# color_mode: AlexNet 3 kanal beklediği için 'rgb' kullanıldı. Gri tonlamalı PNG'leri 3 kanala kopyalayacak.
# shuffle: Eğitim ve doğrulama için karıştırma açılır, test için kapatılır.

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=PNG_OUTPUT_DIR,
    x_col="filename",
    y_col="Target",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb", 
    shuffle=True
)

val_gen = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=PNG_OUTPUT_DIR, 
    x_col="filename",
    y_col="Target",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb", 
    shuffle=False # Doğrulama için karıştırma kapalı
)

test_gen = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=PNG_OUTPUT_DIR, 
    x_col="filename",
    y_col="Target",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb", 
    shuffle=False # Test için karıştırma kapalı (önemli: y_true ile eşleşmesi için)
)



def build_alexnet_classic(input_shape=(227, 227, 3)):
    model = models.Sequential([
        layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2),

        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2),

        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2),

        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', dtype='float32')  # İkili sınıflandırma için sigmoid
    ])
    
    return model

model = build_alexnet_classic()
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])



early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, # Öğrenme oranını %20'ye düşür
                               patience=3, 
                               min_lr=1e-7, 
                               verbose=1)


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights, 
    callbacks=[early_stop, reduce_lr]
)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.title("Model Doğruluğu")
plt.xlabel("Epok")
plt.ylabel("Doğruluk")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.title("Model Kaybı")
plt.xlabel("Epok")
plt.ylabel("Kayıp")

plt.tight_layout()
plt.show()


test_steps = int(np.ceil(len(test_df) / BATCH_SIZE))
y_pred_probs = model.predict(test_gen, steps=test_steps)
y_pred = (y_pred_probs > 0.6).astype(int).flatten() 
y_true = test_gen.labels[:len(y_pred)] 

# Karışıklık Matrisi
cm = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
sensitivity = recall_score(y_true, y_pred) 
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) 
f1 = f1_score(y_true, y_pred)

print(pd.DataFrame({
    "Metrik": ["Doğruluk (Accuracy)", "Kesinlik (Precision)", "Duyarlılık (Sensitivity)", "Özgüllük (Specificity)", "F1 Skoru"],
    "Değer": [accuracy, precision, sensitivity, specificity, f1]
}))


plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tahmin: Normal', 'Tahmin: Pnömoni'],
            yticklabels=['Gerçek: Normal', 'Gerçek: Pnömoni'])
plt.title("Karışıklık Matrisi")
plt.xlabel("Tahmin Edilen Etiket")
plt.ylabel("Gerçek Etiket")
plt.show()

