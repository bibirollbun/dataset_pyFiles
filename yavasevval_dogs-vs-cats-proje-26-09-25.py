# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Temel veri işleme ve görselleştirme kütüphaneleri
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# TensorFlow ve Keras, derin öğrenme için kullanacağımız ana kütüphaneler
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Veri setinin tam dosya yapısını listeliyoruz
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Gerekli kütüphaneyi yüklüyoruz
import zipfile
import os

# Zip dosyalarının bulunduğu yolları tanımlıyoruz
zip_dir = '../input/dogs-vs-cats/'
train_zip_path = os.path.join(zip_dir, 'train.zip')
test_zip_path = os.path.join(zip_dir, 'test1.zip')

# Çıkarılacak klasörleri tanımlıyoruz
output_dir = '/kaggle/working/'

# train.zip dosyasını açıyoruz
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    print("train.zip dosyası açılıyor...")
    zip_ref.extractall(output_dir)
    print("train.zip açma işlemi tamamlandı.")

# test1.zip dosyasını açıyoruz
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    print("test1.zip dosyası açılıyor...")
    zip_ref.extractall(output_dir)
    print("test1.zip açma işlemi tamamlandı.")


# Yeni klasörlerin yollarını tanımlıyoruz
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test1')

# Dosya sayısını kontrol ediyoruz
print(f'Eğitim klasöründeki dosya sayısı: {len(os.listdir(train_dir))}')
print(f'Test klasöründeki dosya sayısı: {len(os.listdir(test_dir))}')


# --- 1. Veri Hazırlığı ---
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import random
import zipfile

print("1. Veri setinin zip dosyaları açılıyor...")
zip_dir = '../input/dogs-vs-cats/'
output_dir = '/kaggle/working/'
if not os.path.exists(os.path.join(output_dir, 'train')):
    try:
        with zipfile.ZipFile(os.path.join(zip_dir, 'train.zip'), 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        with zipfile.ZipFile(os.path.join(zip_dir, 'test1.zip'), 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    except FileNotFoundError:
        print("Zip dosyaları bulunamadı. Lütfen Kaggle veri setini kontrol edin.")

print("2. Yeni klasör yapısı oluşturuluyor ve dosyalar taşınıyor...")
base_dir = '/kaggle/working/data/'
os.makedirs(base_dir, exist_ok=True)
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.makedirs(train_cats_dir, exist_ok=True)
os.makedirs(train_dogs_dir, exist_ok=True)
os.makedirs(validation_cats_dir, exist_ok=True)
os.makedirs(validation_dogs_dir, exist_ok=True)

train_source_dir = '/kaggle/working/train/'
filenames = os.listdir(train_source_dir)
if not os.path.exists(os.path.join(train_cats_dir, os.listdir(train_cats_dir)[0] if os.listdir(train_cats_dir) else "placeholder.jpg")):
    cats_filenames = [f for f in filenames if 'cat' in f]
    dogs_filenames = [f for f in filenames if 'dog' in f]
    random.shuffle(cats_filenames)
    random.shuffle(dogs_filenames)
    train_cat_split = int(len(cats_filenames) * 0.8)
    train_dog_split = int(len(dogs_filenames) * 0.8)

    for filename in cats_filenames[:train_cat_split]:
        shutil.copy(os.path.join(train_source_dir, filename), os.path.join(train_cats_dir, filename))
    for filename in cats_filenames[train_cat_split:]:
        shutil.copy(os.path.join(train_source_dir, filename), os.path.join(validation_cats_dir, filename))
    for filename in dogs_filenames[:train_dog_split]:
        shutil.copy(os.path.join(train_source_dir, filename), os.path.join(train_dogs_dir, filename))
    for filename in dogs_filenames[train_dog_split:]:
        shutil.copy(os.path.join(train_source_dir, filename), os.path.join(validation_dogs_dir, filename))

print("3. Veri jeneratörleri oluşturuluyor...")
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary', subset='training')
validation_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary', subset='validation')


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("Model oluşturuluyor...")

# Sequential (sıralı) bir model oluşturuyoruz
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Modeli derliyoruz
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
              metrics=['accuracy'])

# Modelin özetini gösteriyoruz
model.summary()

print("Model başarıyla oluşturuldu.")


# Modeli eğitmeye başlıyoruz.
# history değişkeni, eğitim sürecindeki performans metriklerini (accuracy ve loss) saklayacak.
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15, # Modeli 15 kez tüm veri seti üzerinde eğitiyoruz
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

print("Eğitim tamamlandı.")


# Gerekli kütüphaneleri yüklüyoruz
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- Kayıp ve Doğruluk Grafiği ---
# Eğitimdeki kayıp (loss) ve doğruluk (accuracy) değerlerini alıyoruz
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Doğruluk grafiğini çiziyoruz
plt.plot(epochs, acc, 'bo', label='Eğitim Doğruluğu')
plt.plot(epochs, val_acc, 'b', label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Döngü (Epochs)')
plt.ylabel('Doğruluk (Accuracy)')
plt.legend()
plt.figure()

# Kayıp grafiğini çiziyoruz
plt.plot(epochs, loss, 'bo', label='Eğitim Kaybı')
plt.plot(epochs, val_loss, 'b', label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Döngü (Epochs)')
plt.ylabel('Kayıp (Loss)')
plt.legend()
plt.show()

# --- Karmaşıklık Matrisi (Confusion Matrix) ve Sınıflandırma Raporu ---
# Tahminleri almak için doğrulama setini kullanıyoruz
# Bu, modelin tahminlerini ve gerçek etiketleri karşılaştırmamızı sağlar
validation_steps = validation_generator.samples // validation_generator.batch_size
y_true = validation_generator.classes[validation_generator.index_array]
y_pred_proba = model.predict(validation_generator, steps=validation_steps)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Karmaşıklık Matrisi oluşturma
cm = confusion_matrix(y_true[:len(y_pred)], y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Kedi', 'Köpek'], yticklabels=['Kedi', 'Köpek'])
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Karmaşıklık Matrisi (Confusion Matrix)')
plt.show()

# Sınıflandırma Raporu oluşturma
print("\nSınıflandırma Raporu:")
print(classification_report(y_true[:len(y_pred)], y_pred, target_names=['Kedi', 'Köpek']))


# Yeni bir model oluşturuyoruz.
# Modelin mimarisi aynı kalacak.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

new_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Modeli daha yüksek bir öğrenme oranı ile derliyoruz.
new_model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3), # Önemli değişiklik: 1e-4 yerine 1e-3
                  metrics=['accuracy'])

# Yeni modeli yeniden eğitiyoruz.
print("Yeni modelin eğitimi başladı...")
history_new = new_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import random
import zipfile
from tensorflow.keras.preprocessing import image

# --- 1. Veri Hazırlığı ---
print("1. Veri setinin zip dosyaları açılıyor...")
zip_dir = '../input/dogs-vs-cats/'
output_dir = '/kaggle/working/'

try:
    with zipfile.ZipFile(os.path.join(zip_dir, 'train.zip'), 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    with zipfile.ZipFile(os.path.join(zip_dir, 'test1.zip'), 'r') as zip_ref:
        zip_ref.extractall(output_dir)
except FileNotFoundError:
    pass

print("2. Yeni klasör yapısı oluşturuluyor ve dosyalar taşınıyor...")
base_dir = '/kaggle/working/data/'
os.makedirs(base_dir, exist_ok=True)
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.makedirs(train_cats_dir, exist_ok=True)
os.makedirs(train_dogs_dir, exist_ok=True)
os.makedirs(validation_cats_dir, exist_ok=True)
os.makedirs(validation_dogs_dir, exist_ok=True)

train_source_dir = '/kaggle/working/train/'
filenames = os.listdir(train_source_dir)
if not os.listdir(train_cats_dir):
    cats_filenames = [f for f in filenames if 'cat' in f]
    dogs_filenames = [f for f in filenames if 'dog' in f]
    random.shuffle(cats_filenames)
    random.shuffle(dogs_filenames)
    train_cat_split = int(len(cats_filenames) * 0.8)
    train_dog_split = int(len(dogs_filenames) * 0.8)

    for filename in cats_filenames[:train_cat_split]:
        shutil.copy(os.path.join(train_source_dir, filename), os.path.join(train_cats_dir, filename))
    for filename in cats_filenames[train_cat_split:]:
        shutil.copy(os.path.join(train_source_dir, filename), os.path.join(validation_cats_dir, filename))
    for filename in dogs_filenames[:train_dog_split]:
        shutil.copy(os.path.join(train_source_dir, filename), os.path.join(train_dogs_dir, filename))
    for filename in dogs_filenames[train_dog_split:]:
        shutil.copy(os.path.join(train_source_dir, filename), os.path.join(validation_dogs_dir, filename))

print("3. Veri jeneratörleri oluşturuluyor...")
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

# --- 2. Modeli Eğitme ---
print("4. Model eğitiliyor...")
inputs = Input(shape=(150, 150, 3))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3), metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=train_generator.samples // train_generator.batch_size, epochs=3, validation_data=validation_generator, validation_steps=validation_generator.samples // validation_generator.batch_size)

# --- 3. Grad-CAM Oluşturma ---
print("5. Grad-CAM ısı haritası oluşturuluyor...")
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4, save_path=None):
    img = cv2.imread(img_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + image.img_to_array(img)
    superimposed_img = image.array_to_img(superimposed_img)
    
    plt.imshow(superimposed_img)
    plt.title('Grad-CAM Isı Haritası')
    plt.axis('off')
    plt.show()

    if save_path:
        superimposed_img.save(save_path)
        print(f"Görsel '{save_path}' adresine kaydedildi.")

# --- Grad-CAM Uygulama ---
try:
    last_conv_layer = None
    for layer in model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    last_conv_layer_name = last_conv_layer.name if last_conv_layer else None
    
    if last_conv_layer_name:
        img_path = random.choice(validation_generator.filepaths)
        img_array = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        display_gradcam(img_path, heatmap, save_path='/kaggle/working/gradcam_final.png')
        print(f"Seçilen görsel: {img_path}")
    else:
        print("Modelde Conv2D katmanı bulunamadı.")
except Exception as e:
    print(f"Bir hata oluştu: {e}. Lütfen notebook'u tekrar baştan çalıştırın.")

