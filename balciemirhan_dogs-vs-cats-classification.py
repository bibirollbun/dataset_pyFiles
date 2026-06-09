# Gerekli kütüphaneleri içe aktarıyoruz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile

# TensorFlow ve Keras kütüphaneleri
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("Kütüphaneler başarıyla yüklendi.")

# Kaggle'daki veri seti yolları
# Veri seti genellikle bir zip dosyası olarak gelir, önce onu açalım.
# Eğer dosyalar zaten açıksa bu adımı atlayabilirsiniz.

# train.zip ve test1.zip dosyalarını açma
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as z:
    z.extractall('.')

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as z:
    z.extractall('.')

# Dosya yollarını ve etiketleri hazırlama
TRAIN_DIR = 'train/'
filenames = os.listdir(TRAIN_DIR)
labels = []
for filename in filenames:
    label = filename.split('.')[0] # Dosya adı 'cat.1.jpg' veya 'dog.1.jpg' şeklindedir
    if label == 'cat':
        labels.append('cat')
    else:
        labels.append('dog')

# Dosya adları ve etiketlerden bir DataFrame oluşturuyoruz.
# Bu yapı, veriyi yönetmeyi çok kolaylaştırır.
df = pd.DataFrame({
    'filename': filenames,
    'label': labels
})

# Oluşturduğumuz DataFrame'in ilk 5 satırına bakalım
print(df.head())
print("\nVeri setindeki toplam resim sayısı:", len(df))
print("\nSınıflara göre resim sayıları:\n", df['label'].value_counts())


# DataFrame'i eğitim (%80) ve doğrulama (%20) olarak ikiye ayırıyoruz
train_df, validation_df = train_test_split(df, test_size=0.20, random_state=42)

# Eğitim setindeki veriyi yeniden dengelemek (eğer dengesiz ise) ve
# boyutunu kontrol etmek için boyutlarını yeniden ayarlıyoruz
train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)

# Görüntü boyutları ve diğer parametreler
IMG_WIDTH = 150
IMG_HEIGHT = 150
IMG_CHANNELS = 3 # Renkli resimler için 3 (RGB)
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32

# Veri Çoğaltma (Data Augmentation) ile ImageDataGenerator oluşturma
# Bu işlem, eğitim verilerini yapay olarak artırarak modelin genelleme yeteneğini geliştirir.
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Piksel değerlerini 0-1 arasına ölçekle
    rotation_range=40,          # Resmi rastgele 40 dereceye kadar döndür
    width_shift_range=0.2,      # Resmi yatayda %20'ye kadar kaydır
    height_shift_range=0.2,     # Resmi dikeyde %20'ye kadar kaydır
    shear_range=0.2,            # Resmi eğ/bük
    zoom_range=0.2,             # Resme %20'ye kadar zoom yap
    horizontal_flip=True,       # Resmi yatayda rastgele çevir
    fill_mode='nearest'         # Oluşan boş pikselleri en yakın pikselle doldur
)

# Doğrulama verisi için çoğaltma yapmıyoruz, sadece ölçekleme yapıyoruz!
# Çünkü modelin performansını orijinal resimler üzerinde ölçmek isteriz.
validation_datagen = ImageDataGenerator(rescale=1./255)

# DataFrame'den akan veri üreteçleri oluşturuyoruz
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col='filename',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='binary', # İki sınıfımız var: kedi ve köpek
    batch_size=BATCH_SIZE
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    directory=TRAIN_DIR,
    x_col='filename',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE
)





# Model mimarisini oluşturuyoruz
model = Sequential([
    # 1. Evrişim ve Havuzlama Bloğu
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)),
    MaxPooling2D((2, 2)),
    
    # 2. Evrişim ve Havuzlama Bloğu
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # 3. Evrişim ve Havuzlama Bloğu
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Özellik haritalarını tek boyutlu bir vektöre düzleştirme
    Flatten(),
    
    # Sınıflandırma için tam bağlantılı katman
    Dense(512, activation='relu'),
    
    # Overfitting'i önlemek için Dropout katmanı
    Dropout(0.5),
    
    # Çıkış katmanı (1 nöron, çünkü kedi mi değil mi diye soruyoruz - binary)
    # Sigmoid aktivasyonu, sonucu 0 ile 1 arasında bir olasılık değerine dönüştürür.
    Dense(1, activation='sigmoid')
])

# Modeli derliyoruz
model.compile(optimizer='adam',
              loss='binary_crossentropy', # İkili sınıflandırma için en uygun kayıp fonksiyonu
              metrics=['accuracy'])

# Modelin mimarisini özet olarak görelim
model.summary()


# Modeli eğitiyoruz
# Bu işlem GPU'nuzun hızına bağlı olarak zaman alabilir. Kaggle'da GPU'yu aktif etmeyi unutmayın!
EPOCHS = 15 # Veri setinin tamamının üzerinden kaç kez geçileceği

history = model.fit(
    train_generator,
    steps_per_epoch=train_df.shape[0] // BATCH_SIZE, # Bir epoch'ta kaç adım atılacağı
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_df.shape[0] // BATCH_SIZE # Doğrulama için kaç adım atılacağı
)

# Eğitilen modeli daha sonra kullanmak üzere kaydedebilirsiniz
# model.save('cat_dog_classifier.h5')


# Eğitim ve Doğrulama Doğruluk/Kayıp grafiklerini çizdirme
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Doğruluğu')
plt.plot(epochs_range, val_acc, label='Doğrulama Doğruluğu')
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Doğruluğu')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Kaybı')
plt.plot(epochs_range, val_loss, label='Doğrulama Kaybı')
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.show()

# Karmaşıklık Matrisi (Confusion Matrix) ve Sınıflandırma Raporu
# Modelin doğrulama seti üzerindeki tahminlerini alalım
predictions = model.predict(validation_generator)
# Sigmoid çıktısını (0-1 arası olasılık) 0 veya 1 sınıfına dönüştürelim
predicted_classes = np.where(predictions > 0.5, 1, 0) 
true_classes = validation_generator.classes

print("\nSınıflandırma Raporu:")
print(classification_report(true_classes, predicted_classes, target_names=['Cat', 'Dog']))

print("\nKarmaşıklık Matrisi:")
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()




