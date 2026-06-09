# Dogs vs Cats Classification - Final Version
# Bu proje köpek ve kedi görüntülerini sınıflandırmak için CNN kullanır

import warnings
warnings.filterwarnings('ignore')
import os
import random
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, load_img
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Temel parametreler
FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3


# Test verisini çıkar
path_to_zip_file = '/kaggle/input/dogs-vs-cats/test1.zip'
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall('.')


# Eğitim verisini çıkar
path_to_zip_file = '/kaggle/input/dogs-vs-cats/train.zip'
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall('.')


# Dosya isimlerinden kategori bilgisini çıkar
filenames = os.listdir("train/")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

# DataFrame oluştur
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

# Test dosyalarını da DataFrame'e ekle
test_filenames = os.listdir("test1/")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]


# 0 cat 1 dog
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 

# Train-Validation-Test bölümlemesi (60-20-20 oranında)
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['category'])
validate_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['category'])

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Veri seti dağılımı:")
print(f"Toplam veri: {len(df):,}")
print(f"Eğitim: {len(train_df):,}")
print(f"Doğrulama: {len(validate_df):,}") 
print(f"Test: {len(test_df):,}")


df.head()


# Rastgele örnekler göster
df.sample(5)


df.info()


df.isna().sum()


df.duplicated().sum()


# Detaylı veri analizi ve görselleştirme
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()  # 2D array'i flatten et

# 1) Sınıf bar chart
df['category'].value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'lightcoral'])
axes[0].set_title('Sınıf Dağılımı')
axes[0].set_ylabel('Görüntü Sayısı')
axes[0].tick_params(axis='x', rotation=0)

# Set bazında dağılım
set_data = pd.DataFrame({
    'Train': train_df['category'].value_counts(),
    'Validation': validate_df['category'].value_counts(),
    'Test': test_df['category'].value_counts()
})
set_data.plot(kind='bar', ax=axes[1], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1].set_title('Set Bazında Sınıf Dağılımı')
axes[1].set_ylabel('Görüntü Sayısı')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend()

# Rastgele görüntü örnekleri
for i in range(2):
    sample_file = random.choice(filenames)
    img_path = f"train/{sample_file}"
    img = load_img(img_path, target_size=(150, 150))
    category = sample_file.split('.')[0]
    
    axes[i+2].imshow(img)
    axes[i+2].set_title(f'Örnek: {category.upper()}')
    axes[i+2].axis('off')

plt.tight_layout()
plt.show()



# CNN modelini oluştur
model = Sequential()

# İlk konvolüsyon bloğu
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# İkinci konvolüsyon bloğu
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Üçüncü konvolüsyon bloğu
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Tam bağlantılı katmanlar
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

# Çıkış katmanı (2 sınıf: kedi ve köpek)
model.add(Dense(2, activation='softmax'))

# Modeli derle
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

print(f"Toplam parametre sayısı: {model.count_params():,}")
model.summary()


# Callback fonksiyonlarını tanımla
# Early Stopping: Validation loss iyileşmezse eğitimi durdur
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Learning Rate Reducer: Plateau durumunda learning rate'i azalt
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Model Checkpoint: En iyi modeli kaydet
checkpoint = ModelCheckpoint(
    'best_catdog_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Tüm callback'leri topla
callbacks = [early_stop, lr_reducer, checkpoint]

print("Callback fonksiyonları hazırlandı")


# Data Augmentation - Veri artırma teknikleri
train_datagen = ImageDataGenerator(
    rotation_range=15,        # 15 derece döndürme
    rescale=1./255,          # Normalizasyon
    shear_range=0.1,         # Kayma dönüşümü
    zoom_range=0.2,          # Zoom
    horizontal_flip=True,     # Yatay çevirme
    width_shift_range=0.1,   # Genişlik kaydırma
    height_shift_range=0.1,  # Yükseklik kaydırma
    fill_mode='nearest'      # Boş alan doldurma
)

# Validation için sadece normalizasyon
validation_datagen = ImageDataGenerator(rescale=1./255)

# Training data generator
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=32 if FAST_RUN else 64
)

# Validation data generator
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=32 if FAST_RUN else 64
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Class indices: {train_generator.class_indices}")


# Model eğitimi
epochs = 50 if not FAST_RUN else 3

print("Model eğitimi başlıyor...")
print(f"Epoch sayısı: {epochs}")
print(f"Eğitim veri: {len(train_df):,}")
print(f"Doğrulama veri: {len(validate_df):,}")

# Modeli eğit
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    verbose=1
)

print(f"En iyi validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"En düşük validation loss: {min(history.history['val_loss']):.4f}")


# Eğitim sonuçlarını görselleştir
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy grafiği
ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss grafiği
ax2.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Learning rate geçmişi (varsa)
if 'lr' in history.history:
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['lr'], color='orange', linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()

# Sonuç özeti
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print("Sonuç özeti:")
print(f"Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"Overfitting farkı: {abs(final_train_acc - final_val_acc):.4f}")


# Model değerlendirme - Confusion Matrix ve Classification Report
# En iyi modeli yükle
try:
    model.load_weights('best_catdog_model.h5')
    print("En iyi model ağırlıkları yüklendi")
except:
    print("Model checkpoint bulunamadı, mevcut ağırlıklar kullanılacak")

# Validation seti üzerinde tahmin yap
validation_generator.reset()
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Gerçek etiketleri al
y_true = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# Confusion Matrix oluştur
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels,
            cbar_kws={'label': 'Sayı'})
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Final validation accuracy
final_accuracy = np.sum(y_pred == y_true) / len(y_true)
print(f"Final Validation Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

# Model özeti
print("Model özeti:")
print(f"- Toplam parametre sayısı: {model.count_params():,}")
print(f"- Model boyutu: ~{model.count_params() * 4 / (1024*1024):.1f} MB")
print(f"- Eğitim epoch sayısı: {len(history.history['loss'])}")


# rastgele örnek tahminler göster
def show_predictions(num_samples=6):
   
    sample_files = random.sample(validate_df['filename'].tolist(), num_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    correct_predictions = 0
    
    for i, filename in enumerate(sample_files):
        # Görüntüyü yükle ve işle
        img_path = f"train/{filename}"
        img = load_img(img_path, target_size=IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Tahmin yap
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction)
        
        
        # Gerçek sınıf
        actual_class = validate_df[validate_df['filename'] == filename]['category'].iloc[0]
        
        # Görüntüyü göster
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Başlık oluştur
        pred_label = class_labels[predicted_class]
        is_correct = pred_label == actual_class
        if is_correct:
            correct_predictions += 1
        
        color = 'green' if is_correct else 'red'
        title = f"Actual: {actual_class.upper()}\nPredicted: {pred_label.upper()}"
        axes[i].set_title(title, color=color)
    
    plt.tight_layout()
    plt.show()
    
    return correct_predictions / num_samples

# Örnek tahminleri göster
sample_accuracy = show_predictions(6)



# İkinci model - Düşük learning rate (0.0001)
model2 = Sequential()

# Aynı mimari
model2.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))) 
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.1))

model2.add(Conv2D(64, (3, 3), padding='same'))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.2))

model2.add(Conv2D(128, (3, 3), padding='same'))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(256))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(0.3))
model2.add(Dense(2, activation='softmax'))

# Farklı learning rate ile derle
model2.compile(loss='categorical_crossentropy', 
               optimizer=Adam(learning_rate=0.0001),  # Düşük learning rate
               metrics=['accuracy'])



# Callback'leri yeniden tanımla
early_stop2 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr_reducer2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
checkpoint2 = ModelCheckpoint('best_catdog_model_lr0001.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

callbacks2 = [early_stop2, lr_reducer2, checkpoint2]

# Generatorları yeniden başlat
train_generator.reset()
validation_generator.reset()

# Model eğitimi
history2 = model2.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks2,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    verbose=1
)

print(f"En iyi validation accuracy: {max(history2.history['val_accuracy']):.4f}")


# İki modelin karşılaştırması
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Model 1 (LR=0.001) - Accuracy
ax1.plot(history.history['accuracy'], label='Training', color='blue', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation', color='red', linewidth=2)
ax1.set_title('Model 1 - Accuracy (LR=0.001)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Model 1 (LR=0.001) - Loss
ax2.plot(history.history['loss'], label='Training', color='blue', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation', color='red', linewidth=2)
ax2.set_title('Model 1 - Loss (LR=0.001)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Model 2 (LR=0.0001) - Accuracy
ax3.plot(history2.history['accuracy'], label='Training', color='green', linewidth=2)
ax3.plot(history2.history['val_accuracy'], label='Validation', color='orange', linewidth=2)
ax3.set_title('Model 2 - Accuracy (LR=0.0001)')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Model 2 (LR=0.0001) - Loss
ax4.plot(history2.history['loss'], label='Training', color='green', linewidth=2)
ax4.plot(history2.history['val_loss'], label='Validation', color='orange', linewidth=2)
ax4.set_title('Model 2 - Loss (LR=0.0001)')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Performans karşılaştırması
print("LEARNING RATE KARŞILAŞTIRMASI")
print("="*50)

# Model 1 sonuçları
final_train_acc1 = history.history['accuracy'][-1]
final_val_acc1 = history.history['val_accuracy'][-1]
best_val_acc1 = max(history.history['val_accuracy'])
final_loss1 = history.history['val_loss'][-1]

# Model 2 sonuçları  
final_train_acc2 = history2.history['accuracy'][-1]
final_val_acc2 = history2.history['val_accuracy'][-1]
best_val_acc2 = max(history2.history['val_accuracy'])
final_loss2 = history2.history['val_loss'][-1]

print(f"\nMODEL 1 (LR=0.001):")
print(f"  Final Training Accuracy: {final_train_acc1:.4f} ({final_train_acc1*100:.2f}%)")
print(f"  Final Validation Accuracy: {final_val_acc1:.4f} ({final_val_acc1*100:.2f}%)")
print(f"  Best Validation Accuracy: {best_val_acc1:.4f} ({best_val_acc1*100:.2f}%)")
print(f"  Final Validation Loss: {final_loss1:.4f}")
print(f"  Epoch Sayısı: {len(history.history['loss'])}")

print(f"\nMODEL 2 (LR=0.0001):")
print(f"  Final Training Accuracy: {final_train_acc2:.4f} ({final_train_acc2*100:.2f}%)")
print(f"  Final Validation Accuracy: {final_val_acc2:.4f} ({final_val_acc2*100:.2f}%)")
print(f"  Best Validation Accuracy: {best_val_acc2:.4f} ({best_val_acc2*100:.2f}%)")
print(f"  Final Validation Loss: {final_loss2:.4f}")
print(f"  Epoch Sayısı: {len(history2.history['loss'])}")

print(f"\nSonuç:")
if best_val_acc1 > best_val_acc2:
    print(f"Model 1 (LR=0.001) daha başarılı: {best_val_acc1*100:.2f}% vs {best_val_acc2*100:.2f}%")
    print(f"Fark: {(best_val_acc1-best_val_acc2)*100:.2f} puan")
else:
    print(f"Model 2 (LR=0.0001) daha başarılı: {best_val_acc2*100:.2f}% vs {best_val_acc1*100:.2f}%")
    print(f"Fark: {(best_val_acc2-best_val_acc1)*100:.2f} puan")


# Her iki modelin validation seti üzerinde performansı
validation_generator.reset()

# Model 1 tahminleri
Y_pred1 = model.predict(validation_generator)
y_pred1 = np.argmax(Y_pred1, axis=1)

validation_generator.reset()

# Model 2 tahminleri  
Y_pred2 = model2.predict(validation_generator)
y_pred2 = np.argmax(Y_pred2, axis=1)

# Gerçek etiketler
y_true = validation_generator.classes

# Accuracy hesaplama
acc1 = np.sum(y_pred1 == y_true) / len(y_true)
acc2 = np.sum(y_pred2 == y_true) / len(y_true)

print("VALIDATION SETİ FINAL PERFORMANSI")
print("="*40)
print(f"Model 1 (LR=0.001): {acc1:.4f} ({acc1*100:.2f}%)")
print(f"Model 2 (LR=0.0001): {acc2:.4f} ({acc2*100:.2f}%)")

if acc1 > acc2:
    print(f"\nKazanan: Model 1 (Yüksek Learning Rate)")
    print(f"Performans farkı: +{(acc1-acc2)*100:.2f} puan")
else:
    print(f"\nKazanan: Model 2 (Düşük Learning Rate)")
    print(f"Performans farkı: +{(acc2-acc1)*100:.2f} puan")

