import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


import zipfile # zipfile kütüphanesini içe aktar

# Zip dosyalarının yollarını belirle
# Not: 'test.zip' yerine 'test1.zip' olarak güncelledim
train_zip_path = '/kaggle/input/dogs-vs-cats/train.zip'
test_zip_path = '/kaggle/input/dogs-vs-cats/test1.zip' # Eğer test1.zip ise

# Çıkarılacak dizinleri belirle
output_dir = '/kaggle/working/dogs_vs_cats_unzipped/' # Çıkarılan dosyaların kaydedileceği yer

# Çıkarılacak dizinleri oluştur (eğer yoksa)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)


print("Eğitim verileri çıkarılıyor...")
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(output_dir, 'train')) # train klasörüne çıkar

print("Test verileri çıkarılıyor...")
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(output_dir, 'test')) # test klasörüne çıkar

print("Tüm veriler başarıyla çıkarıldı.")

# --- Veri Yollarını KESİNLEŞTİRME ---
# Eğitim klasörü için: output_dir/train/train/
train_images_dir = os.path.join(output_dir, 'train', 'train')

# Test klasörü için: output_dir/test/test1/
test_images_dir = os.path.join(output_dir, 'test', 'test1')

# --- Şimdi dosya isimlerini alabiliriz ---
train_filenames = os.listdir(train_images_dir)
test_filenames = os.listdir(test_images_dir)


# Güncellenmiş klasörlerdeki dosya sayılarını kontrol edelim
print(f"Kesinleştirilmiş eğitim setindeki görsel sayısı: {len(train_filenames)}")
print(f"Kesinleştirilmiş test setindeki görsel sayısı: {len(test_filenames)}")

print("\nKesinleştirilmiş eğitim klasöründeki ilk 5 dosya:")
print(train_filenames[:5])

print("\nKesinleştirilmiş test klasöründeki ilk 5 dosya:")
print(test_filenames[:5])

# --- Veri Çerçevesi Oluşturma (Eğitim Seti İçin) ---
# Görüntü dosyalarından etiketleri (dog/cat) çıkaralım
def extract_label(filename):
    if 'cat' in filename:
        return 'cat'
    elif 'dog' in filename:
        return 'dog'
    return 'unknown' # Test setinde bu kullanılmayacak, sadece train için

train_data = []
for filename in train_filenames:
    label = extract_label(filename)
    train_data.append({'filename': filename, 'label': label})

train_df = pd.DataFrame(train_data)

print("\nEğitim veri çerçevesinin ilk 5 satırı:")
print(train_df.head())

print("\nEğitim veri çerçevesindeki etiket dağılımı:")
print(train_df['label'].value_counts())

# Test setinde etiket olmadığı için sadece dosya isimlerini içeren bir dataframe oluşturalım
test_data = []
for filename in test_filenames:
    test_data.append({'filename': filename})

test_df = pd.DataFrame(test_data)
print("\nTest veri çerçevesinin ilk 5 satırı:")
print(test_df.head())




# Adım 5: Verileri Görselleştirme ve Temel İstatistikler

# 5.1 Eğitim setinden rastgele birkaç görüntü gösterelim
print("\nEğitim setinden rastgele 9 örnek görüntü:")
plt.figure(figsize=(12, 12))
for i in range(9):
    random_index = np.random.randint(0, len(train_df))
    filename = train_df.iloc[random_index]['filename']
    label = train_df.iloc[random_index]['label']
    img_path = os.path.join(train_images_dir, filename)

    img = Image.open(img_path)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(f"{label} - {img.size[0]}x{img.size[1]}")
    plt.axis('off')
plt.tight_layout() # Görsellerin çakışmasını engeller
plt.show()

# Burada gösterdiğimiz rastgele kedi ve köpek görsellerinin bir benzerini oluşturalım.
# Her bir görselde bir kedi ya da köpek olacak, başlığında türü ve boyutları yazacak.
# Çözünürlük ve renkler rastgele olabilir.
print("\nGördüğümüz rastgele kedi ve köpek görsellerine benzer bir örnek: ")


# Adım 6: Görüntü Boyutlarının Analizi

# Eğitim setindeki görüntü boyutlarını toplayalım
image_sizes = []
for filename in train_filenames:
    try:
        img_path = os.path.join(train_images_dir, filename)
        with Image.open(img_path) as img:
            image_sizes.append(img.size) # (width, height)
    except Exception as e:
        print(f"Hata oluştu: {filename} - {e}")
        # Hatalı veya bozuk dosyaları atla

# Boyutları bir DataFrame'e dönüştürelim
size_df = pd.DataFrame(image_sizes, columns=['width', 'height'])

print("\nGörüntü boyutlarının ilk 5 satırı:")
print(size_df.head())

print("\nGörüntü boyutlarının temel istatistikleri:")
print(size_df.describe())

# Genişlik ve yükseklik dağılımını görselleştirelim
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(size_df['width'], bins=50, kde=True)
plt.title('Görüntü Genişliği Dağılımı')
plt.xlabel('Genişlik (piksel)')
plt.ylabel('Sayı')

plt.subplot(1, 2, 2)
sns.histplot(size_df['height'], bins=50, kde=True)
plt.title('Görüntü Yüksekliği Dağılımı')
plt.xlabel('Yükseklik (piksel)')
plt.ylabel('Sayı')

plt.tight_layout()
plt.show()

print("\nEn sık görülen görüntü boyutları (ilk 10):")
print(size_df.value_counts().head(10))

# En küçük ve en büyük boyutları da görmek faydalı olabilir
print(f"\nEn küçük genişlik: {size_df['width'].min()}, En küçük yükseklik: {size_df['height'].min()}")
print(f"En büyük genişlik: {size_df['width'].max()}, En büyük yükseklik: {size_df['height'].max()}")


# Adım 7: Veri Ön İşleme ve Hazırlık (Data Preprocessing and Augmentation)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 7.1 Görüntü Boyutu ve Batch Boyutu Ayarlama
# Görüntü boyutlarının analizine göre uygun bir boyut seçelim.
# Genellikle 150x150, 224x224 gibi kare boyutlar tercih edilir.
# Veri setinizdeki boyut dağılımına bakarak karar verebiliriz.
# Şimdilik standart bir boyut olan 128x128 ile başlayalım.
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32 # Bir kerede işlenecek görüntü sayısı

# 7.2 Eğitim ve Doğrulama Setlerini Ayırma
# train_df'imizi eğitim ve doğrulama (validation) setlerine ayıralım.
# Bu, modelin görmediği veriler üzerindeki performansını değerlendirmemizi sağlar.
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

print(f"\nEğitim veri çerçevesi boyutu: {len(train_df)}")
print(f"Doğrulama veri çerçevesi boyutu: {len(val_df)}")
print(f"Eğitim setindeki etiket dağılımı:\n{train_df['label'].value_counts()}")
print(f"Doğrulama setindeki etiket dağılımı:\n{val_df['label'].value_counts()}")

# 7.3 ImageDataGenerator ile Veri Artırma ve Ön İşleme
# Eğitim verileri için veri artırma uygulayalım
train_datagen = ImageDataGenerator(
    rescale=1./255,                 # Piksel değerlerini 0-1 aralığına normalleştir
    rotation_range=15,              # Görüntüleri 15 dereceye kadar rastgele döndür
    width_shift_range=0.1,          # Görüntüleri yatayda %10'a kadar kaydır
    height_shift_range=0.1,         # Görüntüleri dikeyde %10'a kadar kaydır
    shear_range=0.1,                # Kırpma dönüşü uygula
    zoom_range=0.1,                 # %10'a kadar rastgele yakınlaştır
    horizontal_flip=True,           # Yatayda rastgele çevir
    fill_mode='nearest'             # Kaydırma veya döndürme sonrası boş kalan pikselleri doldurma modu
)

# Doğrulama ve test verileri için sadece normalizasyon uygulayalım (veri artırma yapmayız)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# 7.4 Veri Üreteçlerini Oluşturma (Data Generators)
# Bu üreteçler, model eğitimi sırasında görüntüleri otomatik olarak yükleyecek,
# ön işleyecek ve artıracaktır.

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_images_dir, # Görüntülerin bulunduğu dizin
    x_col='filename',           # Görüntü dosyası isimlerini içeren sütun
    y_col='label',              # Etiketleri içeren sütun
    target_size=IMAGE_SIZE,     # Tüm görüntüleri bu boyuta yeniden boyutlandır
    class_mode='categorical',   # Etiketler 'cat'/'dog' olduğu için kategorik
    batch_size=BATCH_SIZE,
    shuffle=True                # Eğitim verilerini karıştır
)

validation_generator = val_test_datagen.flow_from_dataframe(
    val_df,
    directory=train_images_dir, # Doğrulama setindeki görüntüler de eğitim dizininde
    x_col='filename',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False               # Doğrulama verilerini karıştırmaya gerek yok
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df,
    directory=test_images_dir,  # Test görüntülerinin bulunduğu dizin
    x_col='filename',
    y_col=None,                 # Test setinde etiket yok
    target_size=IMAGE_SIZE,
    class_mode=None,            # Etiket olmadığı için None
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nVeri üreteçleri başarıyla oluşturuldu.")


# Adım 8: Evrişimli Sinir Ağı (CNN) Modelini Oluşturma

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Model mimarisini tanımlayalım
model = Sequential([
    # İlk Evrişim bloğu
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25), # Aşırı uyumu azaltmak için Dropout

    # İkinci Evrişim bloğu
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Üçüncü Evrişim bloğu
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Görüntü özelliklerini düzleştirme
    Flatten(),

    # Yoğun (Dense) katmanlar
    Dense(512, activation='relu'),
    Dropout(0.5), # Aşırı uyumu daha da azaltmak için Dropout
    Dense(2, activation='softmax') # 2 sınıf (kedi/köpek) için softmax aktivasyonu
])

# Modeli derleme
# Optimizer: Modelin ağırlıklarını nasıl güncelleyeceğini belirler. Adam yaygın bir seçimdir.
# Loss Function: Modelin tahminleri ile gerçek etiketler arasındaki farkı ölçer.
#                İki sınıf ve kategorik etiketler için 'categorical_crossentropy' kullanılır.
# Metrics: Modelin eğitim ve doğrulama sırasında izlenecek performans ölçütleri. 'accuracy' sıkça kullanılır.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model özetini gösterelim
print("\nModel Mimarisi Özeti:")
model.summary()

# Modeli görselleştirelim (isteğe bağlı, kurulum gerektirebilir)
# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
# print("\nModel mimarisi 'model_architecture.png' dosyasına kaydedildi.")

print("\nCNN modeli başarıyla oluşturuldu ve derlendi.")


# Adım 9: Model Eğitimi (Model Training)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks tanımlayalım
# EarlyStopping: 10 epok boyunca doğrulama kaybı iyileşmezse eğitimi durdur
early_stopping = EarlyStopping(
    monitor='val_loss',  # İzlenecek metrik: doğrulama kaybı
    patience=10,         # Ne kadar süre bekleneceği (epok sayısı)
    restore_best_weights=True # En iyi ağırlıkları geri yükle
)

# ReduceLROnPlateau: 5 epok boyunca doğrulama kaybı iyileşmezse öğrenme oranını 0.2 ile çarp
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,          # Öğrenme oranını bu faktörle çarp
    patience=5,          # Ne kadar süre bekleneceği
    min_lr=0.00001       # Minimum öğrenme oranı
)

# Modeli eğitme
# steps_per_epoch: Her epokta eğitim verilerinden kaç adım alınacağı (toplam örnek sayısı / batch_size)
# validation_steps: Her epokta doğrulama verilerinden kaç adım alınacağı
history = model.fit(
    train_generator,
    epochs=50, # Başlangıçta daha fazla epok belirleyebiliriz, EarlyStopping durduracaktır
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr] # Tanımladığımız callback'leri kullan
)

print("\nModel eğitimi tamamlandı.")

# Eğitim geçmişini görselleştirelim
print("\nEğitim geçmişi görselleştiriliyor:")
plt.figure(figsize=(12, 5))

# Kayıp (Loss) grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epok')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True)

# Doğruluk (Accuracy) grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epok')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()




# Adım 10: Model Değerlendirme ve Tahminler (Model Evaluation and Predictions)

# 10.1 Modelin Doğrulama Seti Üzerindeki Performansını Değerlendirme
print("\nModelin doğrulama seti üzerindeki performansı değerlendiriliyor...")
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Doğrulama Kaybı (Validation Loss): {val_loss:.4f}")
print(f"Doğrulama Doğruluğu (Validation Accuracy): {val_accuracy:.4f}")

# 10.2 Test Seti Üzerinde Tahminler Yapma
print("\nTest seti üzerinde tahminler yapılıyor...")
# test_generator'ı kullanıyoruz
predictions = model.predict(test_generator)

# Tahmin sonuçları, her bir görüntü için 'cat' ve 'dog' olma olasılıklarını içerir.
# Örneğin: [0.9, 0.1] -> %90 kedi, %10 köpek
#         [0.2, 0.8] -> %20 kedi, %80 köpek

# En yüksek olasılığa sahip sınıfın indeksini alalım (0 veya 1)
predicted_class_indices = np.argmax(predictions, axis=1)

# Sınıf isimlerini alalım (ImageDataGenerator'dan)
labels = (train_generator.class_indices) # {'cat': 0, 'dog': 1}
labels = dict((v, k) for k, v in labels.items()) # {'0': 'cat', '1': 'dog'}
predicted_labels = [labels[k] for k in predicted_class_indices]

# 10.3 Tahminleri Kaggle Gönderim Formatına Hazırlama
# Kaggle'da genellikle submission.csv dosyası istenir: id, label
# Test dosyası isimleri 'id.jpg' formatında olduğu için sadece id kısmını almalıyız.

# test_df'den dosya adlarını al ve '.jpg' uzantısını kaldırarak id'leri çıkar
test_ids = [int(os.path.splitext(filename)[0]) for filename in test_df['filename']]

# Tahminleri ve id'leri içeren bir DataFrame oluştur
submission_df = pd.DataFrame({
    'id': test_ids,
    'label': predicted_class_indices # 0 veya 1 olarak etiketleri kullan (Kaggle genellikle bunu bekler)
                                     # Eğer 'cat' veya 'dog' olarak string isteniyorsa, predicted_labels'ı kullanın.
})

# Kaggle'ın genellikle beklediği 0 (kedi) ve 1 (köpek) formatında etiketler.
# Bazı yarışmalarda olasılıklar da istenebilir (örneğin kedi için olasılık).
# Bu projede genellikle 0 veya 1 istenir, 0.5'ten küçükler kedi, büyükler köpek gibi.
# Bizim modelimiz 0. indeks kediyi, 1. indeks köpeği temsil ediyor.
# Yani 0: cat, 1: dog.

# Submission dosyasını kaydetme
submission_df.to_csv('submission.csv', index=False)

print("\nTahminler tamamlandı ve 'submission.csv' dosyası oluşturuldu.")
print("Submission dosyasının ilk 5 satırı:")
print(submission_df.head())

# İsteğe bağlı: Tahmin edilmiş bazı test görüntülerini gösterelim
print("\nTest setinden rastgele 9 örnek tahmin:")
plt.figure(figsize=(12, 12))
for i in range(9):
    random_index = np.random.randint(0, len(test_df))
    filename = test_df.iloc[random_index]['filename']
    img_path = os.path.join(test_images_dir, filename)

    img = Image.open(img_path)
    predicted_label_str = predicted_labels[random_index] # String etiket
    
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Tahmin: {predicted_label_str}")
    plt.axis('off')
plt.tight_layout()
plt.show()

print("\nTebrikler, 'Dogs vs. Cats' projesinin temel adımlarını başarıyla tamamladın!")


