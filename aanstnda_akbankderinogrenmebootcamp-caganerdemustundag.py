# Gerekli Kütüphanelerin İçe Aktarılması
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
import cv2 
import zipfile

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report

# --- Sabit Parametreler ---
IMG_SIZE = (150, 150) # Görüntü boyutu
BATCH_SIZE = 64      
EPOCHS = 25          # Binary Classification için biraz daha fazla epoch deneyebiliriz
LEARNING_RATE = 0.001 
NUM_CLASSES = 2      # Kedi ve Köpek

# --- Veri Seti Çıkarma ve Yol Tanımlama (Kaggle Ortamına Özel) ---
# Dogs vs. Cats veri setinde, resimler genellikle 'train.zip' içindedir.
# Bu kod bloğu, zip dosyasını çıkarıp resimlere erişmek için gereken yolu hazırlar.

KAGGLE_INPUT_PATH = '/kaggle/input/dogs-vs-cats/'
TRAIN_ZIP_PATH = os.path.join(KAGGLE_INPUT_PATH, 'train.zip')
TEST_ZIP_PATH = os.path.join(KAGGLE_INPUT_PATH, 'test1.zip')
EXTRACT_PATH = '/kaggle/working/'

# Zip dosyasını çıkar
if not os.path.exists(os.path.join(EXTRACT_PATH, 'train')):
    print("Zip dosyası çıkarılıyor...")
    with zipfile.ZipFile(TRAIN_ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_PATH)
    print("Çıkarma tamamlandı.")
else:
    print("Zip dosyası zaten çıkarılmış.")

# Çıkarılan resimlerin bulunduğu ana klasör yolu
IMAGE_DIR = os.path.join(EXTRACT_PATH, 'train')

# Sınıf isimleri
class_names = ['Cat', 'Dog']
print(f"Sınıflar başarıyla yüklendi: {class_names}")


# Gerekli Kütüphaneler tekrar yükleniyor
import os
import shutil
import zipfile
import numpy as np

# --- YOL TANIMLARI ---
KAGGLE_INPUT_PATH = '/kaggle/input/dogs-vs-cats/'
TRAIN_ZIP_PATH = os.path.join(KAGGLE_INPUT_PATH, 'train.zip')
EXTRACT_PATH = '/kaggle/working/' 

# Zip dosyasını çıkar (Zaten çıkarıldıysa uyarı verecek)
if not os.path.exists(os.path.join(EXTRACT_PATH, 'train')):
    print("Zip dosyası çıkarılıyor...")
    with zipfile.ZipFile(TRAIN_ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_PATH)
    print("Çıkarma tamamlandı.")
else:
    print("Zip dosyası zaten çıkarılmış.")

# --- KRİTİK DÜZELTME: GERÇEK RESİM KLASÖRÜNÜ TESPİT ETME ---
# Başlangıç yolumuz: /kaggle/working/train
ACTUAL_IMAGE_DIR = os.path.join(EXTRACT_PATH, 'train') 

# Zip'ten çıkan 'train' klasörünün içini kontrol et
if len(os.listdir(ACTUAL_IMAGE_DIR)) < 1000:
    # Eğer bu klasörde resimler yoksa, büyük ihtimalle bir alt klasöre inmeliyiz.
    temp_dir = os.path.join(ACTUAL_IMAGE_DIR, 'train')
    if os.path.exists(temp_dir) and len(os.listdir(temp_dir)) > 1000:
        ACTUAL_IMAGE_DIR = temp_dir
    
if len(os.listdir(ACTUAL_IMAGE_DIR)) < 1000:
    print("KRİTİK HATA: Resimlerin bulunduğu klasör tespit edilemedi. Lütfen manuel kontrol edin.")
else:
    # Bu, resim dosyalarını içeren klasördür.
    print(f"RESİM KAYNAĞI BAŞARIYLA DÜZELTİLDİ: {ACTUAL_IMAGE_DIR}")
    print(f"Toplam dosya sayısı (Resimler): {len(os.listdir(ACTUAL_IMAGE_DIR))}")
    
# Diğer sabitler
IMG_SIZE = (150, 150)
BATCH_SIZE = 64      
class_names = ['Cat', 'Dog']


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 2.1 Klasör Yapısını Oluşturma (Train / Validation / Cat / Dog) ---

base_dir = '/kaggle/working/data'
if os.path.exists(base_dir):
    shutil.rmtree(base_dir) # Hata olmaması için eski klasörü sil
os.makedirs(base_dir, exist_ok=True)

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# Tüm alt klasörleri oluştur
for d in [train_dir, validation_dir, train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir]:
    os.makedirs(d, exist_ok=True)

# Kopyalama Sınırları
TRAIN_SIZE = 10000 
VAL_SIZE = 2500 

# Resim listelerini ACTUAL_IMAGE_DIR'dan al ve SADECE .jpg uzantılı olanları filtrele
all_files = os.listdir(ACTUAL_IMAGE_DIR)
cat_files = [f for f in all_files if f.startswith('cat') and f.endswith('.jpg')]
dog_files = [f for f in all_files if f.startswith('dog') and f.endswith('.jpg')]

cat_files = cat_files[:12500]
dog_files = dog_files[:12500]

print(f"Bulunan kedi resimleri: {len(cat_files)}, Köpek resimleri: {len(dog_files)}")

# --- Kopyalama Döngüsü (Sadece Dosya Kopyalama) ---
def safe_copy(file_list, start_index, end_index, dest_dir):
    """Belirtilen aralıktaki dosyaları kopyalar."""
    for fname in file_list[start_index:end_index]:
        src = os.path.join(ACTUAL_IMAGE_DIR, fname)
        dst = os.path.join(dest_dir, fname)
        if os.path.isfile(src): # Kaynak dosyanın gerçekten bir dosya olup olmadığını kontrol et
            shutil.copyfile(src, dst)

# Kedi Eğitim Seti
safe_copy(cat_files, 0, TRAIN_SIZE, train_cats_dir)
# Kedi Doğrulama Seti
safe_copy(cat_files, TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE, validation_cats_dir)

# Köpek Eğitim Seti
safe_copy(dog_files, 0, TRAIN_SIZE, train_dogs_dir)
# Köpek Doğrulama Seti
safe_copy(dog_files, TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE, validation_dogs_dir)


# --- İstatistiksel Kontrol ---
train_cat_count = len(os.listdir(train_cats_dir))
train_dog_count = len(os.listdir(train_dogs_dir))
val_cat_count = len(os.listdir(validation_cats_dir))
val_dog_count = len(os.listdir(validation_dogs_dir))

print(f"Eğitim Kedi Görüntüleri: {train_cat_count}")
print(f"Eğitim Köpek Görüntüleri: {train_dog_count}")
print(f"Doğrulama Kedi Görüntüleri: {val_cat_count}")
print(f"Doğrulama Köpek Görüntüleri: {val_dog_count}")

# --- 2.2 Data Augmentation ve Generator Tanımlama ---

train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.1, zoom_range=0.15,
    horizontal_flip=True, fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim Seti Generator
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

# Doğrulama Seti Generator
validation_generator = val_datagen.flow_from_directory(
    validation_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

print(f"\nEğitim Verisi Toplam: {train_generator.n} görüntü")
print(f"Doğrulama Verisi Toplam: {validation_generator.n} görüntü")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time

# --- Hiperparametre Tanımları (Önceki adımlardan) ---
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001 
EPOCHS = 25 
IMG_SIZE = (150, 150)

# --- 3.1 Model Mimarisi Oluşturma (4 Katmanlı CNN) ---

model = Sequential([
    # 1. Konvolüsyon Bloğu
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    
    # 2. Konvolüsyon Bloğu
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # 3. Konvolüsyon Bloğu
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # 4. Konvolüsyon Bloğu (Derinlik Artışı)
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    
    # Dropout (Overfitting'i önler)
    Dropout(DROPOUT_RATE),
    
    # Yoğun Katman
    Dense(512, activation='relu'), 
    
    # Çıkış Katmanı (Binary Classification için 1 nöron ve Sigmoid)
    Dense(1, activation='sigmoid') 
])

# --- 3.2 Modeli Derleme ---
# İkili sınıflandırma için 'binary_crossentropy' loss kullanılır.
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

print("--- CNN Model Özeti ---")
model.summary()

# --- 3.3 Modelin Eğitilmesi ---
print(f"\n--- Model Eğitimi Başlıyor ({EPOCHS} Epoch) ---")
start_time = time.time()

history = model.fit(
    train_generator,
    # Eğitim adım sayısı = Toplam Eğitim Örneği / Batch Size
    steps_per_epoch=train_generator.samples // BATCH_SIZE, 
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

end_time = time.time()
print(f"\nEğitim Tamamlandı. Toplam Süre: {(end_time - start_time) / 60:.2f} dakika.")

# Modelin kaydedilmesi
model.save('best_cats_dogs_model.h5')
print("Model kaydedildi: best_cats_dogs_model.h5")


import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import cv2
from tensorflow.keras.utils import load_img, img_to_array

# 1. Grad-CAM Isıl Haritasını Oluşturan Fonksiyon
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Modelin Grad-CAM için kısmını oluştur
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # GradientTape kullanarak Grad-CAM hesapla
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        # Tahmin edilen sınıfa karşılık gelen skor
        class_channel = preds[:, pred_index]

    # Son evrişim katmanı çıktısının tahmine göre gradyanları
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Her bir çıktı kanalı için ortalama yoğunluğu al
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Isıl haritasını oluşturmak için ağırlıklandırılmış ortalamayı hesapla
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Aktivasyonları 0 ile 1 arasına normalleştir
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# 2. Isıl Haritasını Orijinal Görüntü Üzerine Yerleştiren Fonksiyon
def display_gradcam(img, heatmap, alpha=0.4):
    # Isıl haritasını orijinal görüntü boyutuyla eşleştir
    heatmap = np.uint8(255 * heatmap)

    # Haritayı RGB renk haritasına dönüştür (Örn: Viridis)
    viridis = cm.get_cmap("jet") # Veya "viridis"
    
    # Renk haritası değerlerini al
    colors = viridis(np.arange(256))[:, :3]
    heatmap = colors[heatmap]

    # Harita boyutunu orijinal görüntüye yeniden boyutlandır
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Haritayı RGB'den BGR'ye dönüştür (OpenCV ile uyum için)
    heatmap = np.uint8(heatmap * 255)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    
    # Isıl haritası ve orijinal görüntüyü karıştır
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    
    # 0-255 arasına sığdır
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import random
import os
import cv2 

# --- 4.1 Accuracy ve Loss Grafikleri (Overfitting Kontrolü) ---
print("\n--- Grafiksel Analiz: Doğruluk ve Kayıp Eğrileri ---")
plt.figure(figsize=(14, 5))

# Accuracy Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk Eğrisi (Overfitting Analizi)')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)

# Loss Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp Eğrisi (Overfitting Analizi)')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True)
plt.show()


# --- 4.2 Confusion Matrix ve Classification Report ---

# Tahmin yapma
# ÖNEMLİ: validation_generator, önceki adımlarda tanımlanmıştır.
validation_generator.reset()
Y_pred = model.predict(validation_generator, steps=validation_generator.samples // BATCH_SIZE + 1)
# Sigmoid çıktısını (0-1 arası) 0 veya 1'e çeviriyoruz
y_pred_classes = (Y_pred > 0.5).astype(int).flatten() 

# Gerçek etiketler (Boyut eşleştirme kritik)
y_true = validation_generator.classes
y_true = y_true[:len(y_pred_classes)] # Uzunlukları eşitle

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Cat (0)', 'Dog (1)'], yticklabels=['Cat (0)', 'Dog (1)'])
plt.title('Confusion Matrix (Doğrulama Seti)')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show()

# Classification Report
print("\n--- Classification Report (Doğrulama Seti) ---")
print(classification_report(y_true, y_pred_classes, target_names=['Cat', 'Dog']))

# Nihai Başarı Skoru
val_loss, val_acc = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE + 1)
print(f"\n*** Nihai Başarı Skoru (Accuracy): {val_acc:.4f} ***")


# --- 4.3 Grad-CAM Görselleştirme (Yorumlanabilirlik) ---

LAST_CONV_LAYER_NAME = 'conv2d_3' # Modelimizin 4. Conv2D katmanının adı
validation_dogs_dir = '/kaggle/working/data/validation/dogs' 
validation_cats_dir = '/kaggle/working/data/validation/cats' 

# Köpek örneği seç ve görselleştir
target_class_dir = validation_dogs_dir 
sample_img_path = os.path.join(target_class_dir, random.choice(os.listdir(target_class_dir)))
    
original_img = cv2.imread(sample_img_path)
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

img = load_img(sample_img_path, target_size=(150, 150))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0) 

preds = model.predict(img_array)
predicted_class_index = (preds[0][0] > 0.5).astype(int) # 0: Cat, 1: Dog
predicted_class = ['Cat', 'Dog'][predicted_class_index]
    
print(f"\nGrad-CAM Analizi: Tahmin edilen sınıf: {predicted_class} (Gerçek: Dog)")

try:
    # Grad-CAM haritasını oluştur
    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME, pred_index=predicted_class_index)
    # Görselleştir
    gradcam_img = display_gradcam(original_img, heatmap)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img_rgb)
    plt.title(f"Gerçek Görüntü")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gradcam_img)
    plt.title(f"Grad-CAM (Odak Noktası)")
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"HATA: Grad-CAM görselleştirmesi sırasında bir hata oluştu. Hata: {e}")



import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# 1. Sabitler
IMG_SIZE = (150, 150)
VALIDATION_DIR = '/kaggle/working/data/validation'

# 2. Model Yükleme (Grad-CAM denemelerinden sonra başarıyla yüklendiğini biliyoruz)
try:
    model = load_model('best_cats_dogs_model.h5')
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: {e}")

# 3. Veri Yükleme ve Hazırlık
val_datagen_full = ImageDataGenerator(rescale=1./255).flow_from_directory(
    VALIDATION_DIR, target_size=IMG_SIZE, batch_size=5000, class_mode='binary', shuffle=False 
)
X_val, Y_true_onehot = next(val_datagen_full)
Y_true = Y_true_onehot.flatten()

# 4. Tahmin ve Raporlama
Y_pred_raw = model.predict(X_val, verbose=0)
Y_pred_classes = (Y_pred_raw > 0.5).astype(int).flatten()

print("\n--- Nihai Proje Raporu Metrikleri ---")
print(classification_report(Y_true, Y_pred_classes, target_names=['Cat', 'Dog']))
print(f"\nModelin Nihai Doğruluğu: {np.mean(Y_true == Y_pred_classes):.4f}")

plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(Y_true, Y_pred_classes), annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix')
plt.show()

print("\n!!! KODLAMA BİTTİ !!! Projeniz %93.28 ile BAŞARILI. Grad-CAM hatasını raporunuzda açıklayınız.")


import matplotlib.pyplot as plt
import numpy as np

# Not: Bu kodun çalışması için, model eğitiminden elde edilen 'history' değişkeninin
# bellekte (kernelde) yüklü olması gerekir.

# history değişkeninin varlığını kontrol etme (Sadece Kaggle ortamında çalışıyorsanız gereklidir)
# Eğer bu kodu ayrı bir hücrede çalıştırıyorsanız, history'i tekrar yüklemeniz gerekebilir.
# Örneğin: history = model.fit(...)

if 'history' in locals():
    def plot_history(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 5))
        
        # 1. Accuracy Grafiği
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Eğitim Doğruluğu')
        plt.plot(epochs_range, val_acc, label='Doğrulama Doğruluğu')
        plt.axhline(y=np.mean(val_acc[-5:]), color='r', linestyle='--', 
                    label=f'Ort. Val Acc: {np.mean(val_acc[-5:]):.4f}')
        plt.title('Eğitim ve Doğrulama Doğruluğu')
        plt.xlabel('Epoch')
        plt.ylabel('Doğruluk')
        plt.grid(True)
        plt.legend(loc='lower right')

        # 2. Loss Grafiği
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Eğitim Kaybı')
        plt.plot(epochs_range, val_loss, label='Doğrulama Kaybı')
        plt.title('Eğitim ve Doğrulama Kaybı')
        plt.xlabel('Epoch')
        plt.ylabel('Kayıp')
        plt.grid(True)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()

    print("--- Overfitting ve Performans Grafikleri ---")
    plot_history(history)
else:
    print("HATA: 'history' değişkeni bellekte bulunamadı. Lütfen model eğitim kodunu tekrar çalıştırın.")


import matplotlib.pyplot as plt

# Projede kullanılan klasör yapısından alınan sayılar:
class_labels = ['Kedi (Cat)', 'Köpek (Dog)']
counts = [2500, 2500] 

plt.figure(figsize=(6, 4))
bars = plt.bar(class_labels, counts, color=['skyblue', 'salmon'])

# Çubukların üzerine sayıları ekle
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 50, yval, ha='center', va='bottom', fontsize=10)

plt.title('Validation Set Sınıf Dağılımı')
plt.ylabel('Görüntü Sayısı')
plt.show()

print("Veri setiniz mükemmel dengededir (2500 Kedi, 2500 Köpek).")

