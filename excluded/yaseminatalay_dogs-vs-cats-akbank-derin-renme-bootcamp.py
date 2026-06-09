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


import warnings
warnings.filterwarnings('ignore')  # Uyarıları görmezden gel

import zipfile

# Train ve test zip dosyalarını çıkar
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/train_data')

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/test_data')

print("Train ve test veri setleri çıkarıldı!")



import os
import shutil
from sklearn.model_selection import train_test_split

# Ana klasör
train_dir = "/kaggle/working/train_data/train"

# Kedi ve köpek dosyaları
cats = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if 'cat' in f]
dogs = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if 'dog' in f]

# Eğitim ve doğrulama böl
train_cats, val_cats = train_test_split(cats, test_size=0.2, random_state=42)
train_dogs, val_dogs = train_test_split(dogs, test_size=0.2, random_state=42)

print(f"Eğitim: {len(train_cats) + len(train_dogs)}, Doğrulama: {len(val_cats) + len(val_dogs)}")



import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import random



train_dir = '/kaggle/working/train_data/train'  # klasör yolunu kendine göre ayarla

cats = [f for f in os.listdir(train_dir) if 'cat' in f]
dogs = [f for f in os.listdir(train_dir) if 'dog' in f]

print(f"Toplam kedi sayısı: {len(cats)}")
print(f"Toplam köpek sayısı: {len(dogs)}")

# Rastgele birkaç görsel göster
fig, axes = plt.subplots(2, 5, figsize=(15,6))
for i, ax in enumerate(axes.flatten()):
    if i < 5:
        img_path = os.path.join(train_dir, random.choice(cats))
    else:
        img_path = os.path.join(train_dir, random.choice(dogs))
    img = image.load_img(img_path, target_size=(150,150))
    ax.imshow(img)
    ax.axis('off')
plt.show()



train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,       # 0-30 derece döndürme
    width_shift_range=0.2,   # yatay kaydırma
    height_shift_range=0.2,  # dikey kaydırma
    shear_range=0.2,         # kesme (shear)
    zoom_range=0.2,          # yakınlaştırma
    horizontal_flip=True,    # yatay çevirme
    brightness_range=[0.8,1.2],  # renk/brightness jitter
    validation_split=0.2     # train-validation ayrımı
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)



import os
import shutil

base_dir = '/kaggle/working/train_data/train'
cat_dir = os.path.join(base_dir, 'cats')
dog_dir = os.path.join(base_dir, 'dogs')

os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

for fname in os.listdir(base_dir):
    src_path = os.path.join(base_dir, fname)

    # Sadece dosyaları taşı, klasörleri atla
    if not os.path.isfile(src_path):
        continue

    if fname.startswith('cat'):
        dst_path = os.path.join(cat_dir, fname)
    elif fname.startswith('dog'):
        dst_path = os.path.join(dog_dir, fname)
    else:
        continue

    # Eğer hedefte aynı dosya varsa isim değiştir
    if os.path.exists(dst_path):
        name, ext = os.path.splitext(fname)
        counter = 1
        while True:
            new_name = f"{name}_{counter}{ext}"
            new_dst_path = os.path.join(os.path.dirname(dst_path), new_name)
            if not os.path.exists(new_dst_path):
                dst_path = new_dst_path
                break
            counter += 1

    shutil.move(src_path, dst_path)
    print(f"Taşındı: {src_path} -> {dst_path}")



print("Base dizinindeki dosyalar:")
for f in os.listdir(base_dir):
    print(f)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation ve normalizasyon
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Eğitim/validation ayırmak için
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)



import os
import random
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Klasör yolları
cat_dir = '/kaggle/working/train_data/train/cats'
dog_dir = '/kaggle/working/train_data/train/dogs'

# Klasördeki dosyaları listele
cats = [f for f in os.listdir(cat_dir) if os.path.isfile(os.path.join(cat_dir, f))]
dogs = [f for f in os.listdir(dog_dir) if os.path.isfile(os.path.join(dog_dir, f))]

# Dosyaların boş olup olmadığını kontrol et
if len(cats) == 0 or len(dogs) == 0:
    raise ValueError("Cat veya dog klasörü boş! Dosyaları doğru şekilde yüklediğinizden emin olun.")

# Örnek olarak kedi veya köpek resmini random seç
label_choice = random.choice(['cat', 'dog'])
if label_choice == 'cat':
    sample_img_path = os.path.join(cat_dir, random.choice(cats))
else:
    sample_img_path = os.path.join(dog_dir, random.choice(dogs))

print(f"Seçilen sınıf: {label_choice}, dosya: {sample_img_path}")

# Resmi yükle ve boyutlandır
img_size = (224, 224)
img = image.load_img(sample_img_path, target_size=img_size)
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

# Augmentation ve görselleştirme
i = 0
fig, axes = plt.subplots(1, 5, figsize=(15,5))
for batch in train_datagen.flow(x, batch_size=1):
    ax = axes[i]
    # 0-1 float değerler ile imshow doğrudan çalışır, uint8 yapmaya gerek yok
    ax.imshow(batch[0])
    ax.axis('off')
    i += 1
    if i % 5 == 0:
        break
plt.show()



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

img_size = (150, 150, 3)

model = Sequential([
    Input(shape=img_size),  # input_shape yerine Input katmanı
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])



model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Çünkü binary classification: cat vs dog
    metrics=['accuracy']
)



model.summary()



from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation ve normalizasyon
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Jeneratörleri oluştur
img_size = (150, 150)
batch_size = 32
base_dir = '/kaggle/working/train_data/train'

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)



epochs = 10  # CPU kullanıyorsan 5-10 epoch başlangıç için yeterli
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)



from sklearn.metrics import f1_score
import numpy as np

# Validation set üzerinden tahmin
y_true = val_generator.classes                  # Gerçek etiketler
y_pred = model.predict(val_generator)          # Tahminler
y_pred = np.round(y_pred)                      # 0 veya 1'e yuvarla

f1 = f1_score(y_true, y_pred)
print("Validation F1 Score:", f1)



import matplotlib.pyplot as plt

# Accuracy grafiği
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss grafiği
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



model.save('/kaggle/working/cat_dog_cnn_model.h5')



!pip install keras-tuner --quiet 

import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam



def build_model(hp):
    model = Sequential()
    
    # Conv2D Katmanları
    for i in range(hp.Int('conv_layers', 1, 3)):  # Katman sayısını 1-3 arasında dene
        filters = hp.Choice(f'filters_{i}', values=[32, 64, 128])
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[3,5])
        if i == 0:
            model.add(Conv2D(filters, (kernel_size, kernel_size), activation='relu', input_shape=(150,150,3)))
        else:
            model.add(Conv2D(filters, (kernel_size, kernel_size), activation='relu'))
        model.add(MaxPooling2D(2,2))
    
    model.add(Flatten())
    
    # Dense katmanı
    dense_units = hp.Choice('dense_units', [128, 256, 512])
    model.add(Dense(dense_units, activation='relu'))
    
    # Dropout
    dropout_rate = hp.Float('dropout', 0.2, 0.5, step=0.1)
    model.add(Dropout(dropout_rate))
    
    # Output
    model.add(Dense(1, activation='sigmoid'))
    
    # Learning rate
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model



tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',  # Maksimize edilecek değer
    max_trials=10,              # Toplam deneme sayısı
    executions_per_trial=1,     # Her deneme için tekrar sayısı
    directory='tuner_dir',
    project_name='cat_dog_cnn'
)



tuner.search(train_generator,
             validation_data=val_generator,
             epochs=10,
             batch_size=32)



best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("En iyi hiperparametreler:")
print(best_hyperparameters.values)



history = best_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Grafikleri çiz (daha önceki bölümdeki gibi)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



# TensorBoard callback
from tensorflow.keras.callbacks import TensorBoard
tensorboard_cb = TensorBoard(log_dir='./logs', histogram_freq=1)

history = best_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[tensorboard_cb]
)


