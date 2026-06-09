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


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import os
import zipfile
from tqdm import tqdm
import matplotlib.pyplot as plt


with zipfile.ZipFile("/kaggle/input/street-view-getting-started-with-julia/trainResized.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working")

with zipfile.ZipFile("/kaggle/input/street-view-getting-started-with-julia/testResized.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working")

train_labels_file = pd.read_csv("/kaggle/input/street-view-getting-started-with-julia/trainLabels.csv")


# Random seed
np.random.seed(42)
tf.random.set_seed(42)


train_path = '/kaggle/working/trainResized/'
test_path = '/kaggle/working/testResized/'  # FIXED: Added missing slash
train_labels_file = '/kaggle/input/street-view-getting-started-with-julia/trainLabels.csv'

IMG_SIZE = 64


# Etiketler
labels_df = pd.read_csv(train_labels_file) 

print(f"\nEtiket dağılımı:\n{labels_df['Class'].value_counts().sort_index()}")


# Label Encoding
label_encoder = LabelEncoder()
labels_df['Class_Encoded'] = label_encoder.fit_transform(labels_df['Class'])


num_classes = len(label_encoder.classes_)


# Eğitim verileri
train_images = []
train_labels = []

for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    img_id = row['ID']
    label = row['Class_Encoded']
    
    img_file = os.path.join(train_path, f"{img_id}.Bmp")
    
    if os.path.exists(img_file):
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        
        if img is not None: 
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            train_images.append(img_resized)
            train_labels.append(label)

# Numpy dizilerine çevir
X_train = np.array(train_images)
y_train = np.array(train_labels)


# Özellik matris boyutu 
X_train.shape


# Etiket dizisi boyutu 
y_train.shape


# Veriyi normalize et ve reshape
X_train = X_train.astype('float32') / 255.0
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# One-hot encoding
y_train_cat = keras.utils.to_categorical(y_train, num_classes)


# Train-validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_cat, test_size=0.15, random_state=42, stratify=y_train
)


# CNN Model 
model = keras.Sequential()

# İlk Conv Bloğu
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# İkinci Conv Bloğu
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Üçüncü Conv Bloğu
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Dördüncü Conv Bloğu
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Dense Katmanlar
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))



model.summary()


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)



# Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])


# Augmented veri oluştur
X_tr_augmented = data_augmentation(X_tr, training=True)


history = model.fit(
    X_tr_augmented, y_tr,
    batch_size=30,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)


# Eğitim Sonuçları

plt.figure(figsize=(14, 5))

# Accuracy grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()



# Son doğrulama skoru
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"\nFinal Validation Accuracy: {val_accuracy:.4f}")
print(f"Final Validation Loss: {val_loss:.4f}")



test_files = sorted([f for f in os.listdir(test_path) if f.endswith('.Bmp')])
test_images = []
test_ids = []

for img_file in tqdm(test_files):
    img_path = os.path.join(test_path, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_images.append(img_resized)
        test_ids.append(int(img_file.replace('.Bmp', '')))

# Test verilerini hazırla
X_test = np.array(test_images)
X_test = X_test.astype('float32') / 255.0
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# Tahminler
predictions_prob = model.predict(X_test, batch_size=128, verbose=1)
predictions_encoded = np.argmax(predictions_prob, axis=1)

# Label'ları geri çevir
predictions = label_encoder.inverse_transform(predictions_encoded)

# Submission dosyası oluştur
submission = pd.DataFrame({
    'ID': test_ids,
    'Class': predictions
})

# ID'ye göre sırala
submission = submission.sort_values('ID')

# Kaydet
submission.to_csv('submission.csv', index=False)




predictions_prob = model.predict(X_test, batch_size=128, verbose=1)
predictions_encoded = np.argmax(predictions_prob, axis=1)

# Label'ları geri çevir
predictions = label_encoder.inverse_transform(predictions_encoded)

# Submission dosyası oluştur
submission = pd.DataFrame({
    'ID': test_ids,
    'Class': predictions
})

# ID'ye göre sırala
submission = submission.sort_values('ID')
 
submission.to_csv('submission.csv', index=False)



submission.head()




