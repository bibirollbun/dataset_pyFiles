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


import tensorflow as tf
import sys
print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("GPU listesi:", tf.config.list_physical_devices('GPU'))


import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


print("Input klasÃ¶rleri:")
print(os.listdir("/kaggle/input"))


BASE = "/kaggle/input/plant-pathology-2020-fgvc7"
print("Dosyalar:", os.listdir(BASE))

# Train ve Test csv dosyalarÄ±nÄ± yÃ¼kleyelim
train_df = pd.read_csv(os.path.join(BASE, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE, "test.csv"))

print("Train CSV boyutu:", train_df.shape)
print(train_df.head())

print("Test CSV boyutu:", test_df.shape)
print(test_df.head())


import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

# label ile sÃ¼tÃ¼n ekliyoruz
# 4 etiket sÃ¼tununu belirle
label_cols = ['healthy','multiple_diseases','rust','scab']

# Hangi sÃ¼tun 1 ise o sÄ±nÄ±f olacak
train_df['label'] = train_df[label_cols].idxmax(axis=1)

# SÄ±nÄ±f daÄŸÄ±lÄ±mÄ±
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='label')
plt.title("SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ±")
plt.show()

# Ã–rnek 8 gÃ¶rsel gÃ¶ster
IMG_DIR = os.path.join(BASE, "images")

def show_images(df, n=8):
    sample = df.sample(n).reset_index(drop=True)
    plt.figure(figsize=(14,6))
    for i, row in sample.iterrows():
        img_path = os.path.join(IMG_DIR, row['image_id'] + ".jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, n//2, i+1)
        plt.imshow(img)
        plt.title(row['label'])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_images(train_df, n=8)



# Etiket sÃ¼tununu yeniden oluÅŸtur
label_cols = ['healthy','multiple_diseases','rust','scab']
train_df['label'] = train_df[label_cols].idxmax(axis=1)

print(train_df[['image_id','label']].head())



from sklearn.model_selection import train_test_split
import numpy as np, random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_sub, val_sub = train_test_split(
    train_df,
    test_size=0.15,
    stratify=train_df['label'],
    random_state=SEED
)

print("Train set boyutu:", len(train_sub))
print("Validation set boyutu:", len(val_sub))



from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32

# Train augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation sadece normalize (rescale)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Dosya adlarÄ±nÄ± dataframe iÃ§ine ekleyelim
train_sub['filename'] = train_sub['image_id'] + ".jpg"
val_sub['filename']   = val_sub['image_id'] + ".jpg"

# Generator tanÄ±mlarÄ±
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_sub,
    directory=os.path.join(BASE, "images"),
    x_col="filename",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=val_sub,
    directory=os.path.join(BASE, "images"),
    x_col="filename",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)



import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_baseline(input_shape=(224,224,3), n_classes=4):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

# Modeli oluÅŸtur
baseline = build_baseline()

# Derleme
baseline.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model Ã¶zetini yazdÄ±r
baseline.summary()

# EÄŸitim
EPOCHS = 6
history_base = baseline.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS
)



import matplotlib.pyplot as plt

def plot_training(history):
    plt.figure(figsize=(12,4))

    # Accuracy grafiÄŸi
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy EÄŸrisi")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss grafiÄŸi
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss EÄŸrisi")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

plot_training(history_base)



from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

# Base model: ResNet50 (ImageNet aÄŸÄ±rlÄ±klarÄ±yla, include_top=False)
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# KatmanlarÄ± dondur (Ã¶nceden Ã¶ÄŸrendiklerini koru)
for layer in base_model.layers:
    layer.trainable = False

# Ãœst katmanlarÄ± ekle
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(4, activation='softmax')(x)

model_resnet = models.Model(inputs=base_model.input, outputs=output)

# Derleme
model_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Ã–zet
model_resnet.summary()

# EÄŸitim
EPOCHS = 8
history_resnet = model_resnet.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS
)



# ResNet50 katmanlarÄ±nÄ±n son 50 tanesini eÄŸitilebilir hale getir
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Daha kÃ¼Ã§Ã¼k learning rate
model_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Tekrar eÄŸitim
EPOCHS_FINE = 10
history_fine = model_resnet.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS_FINE
)



def plot_training(history, title="Model"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,4))

    # Accuracy grafiÄŸi
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss grafiÄŸi
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

plot_training(history_fine, title="ResNet50 (Fine-Tuned)")



# Test CSV'yi yeniden yÃ¼kle
import os
import pandas as pd

BASE = "/kaggle/input/plant-pathology-2020-fgvc7"

test_df = pd.read_csv(os.path.join(BASE, "test.csv"))
print(test_df.shape)
print(test_df.head())



from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(4, activation='softmax')(x)

model_resnet = models.Model(inputs=base_model.input, outputs=output)

model_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

BASE = "/kaggle/input/plant-pathology-2020-fgvc7"

# Tekrar tanÄ±mla
IMG_SIZE = 224
BATCH_SIZE = 32

# Test CSV yÃ¼kle
test_df = pd.read_csv(os.path.join(BASE, "test.csv"))
test_df['filename'] = test_df['image_id'] + ".jpg"

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=os.path.join(BASE, "images"),
    x_col="filename",
    y_col=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Tahmin al
preds = model_resnet.predict(test_generator, verbose=1)

# Submission dataframe
submission = pd.DataFrame(
    preds,
    columns=['healthy','multiple_diseases','rust','scab']
)
submission.insert(0, 'image_id', test_df['image_id'])

# CSV olarak kaydet
submission.to_csv("submission.csv", index=False)
print("submission.csv oluÅŸturuldu!")
submission.head()



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Modeli compile et (kÃ¼Ã§Ã¼k lr ile)
model_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

EPOCHS_BONUS = 20
history_bonus = model_resnet.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS_BONUS,
    callbacks=[early_stop, checkpoint]
)


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Label deÄŸerlerini sayÄ±sal index haline getirelim
labels = train_generator.class_indices
print("Class Indices:", labels)

# Class weight hesapla
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

# SÃ¶zlÃ¼k haline getir
class_weights_dict = {i: w for i, w in enumerate(class_weights)}
print("Class Weights:", class_weights_dict)

# Modeli tekrar compile et
model_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# EÄŸitim
EPOCHS_CW = 15
history_cw = model_resnet.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS_CW,
    class_weight=class_weights_dict,
    callbacks=[early_stop, checkpoint]
)



from tensorflow.keras.models import load_model

# Kaydedilen en iyi modeli yÃ¼kle
model_resnet = load_model("best_model.h5")

# Model yÃ¼klendiÄŸinde tekrar test et
val_preds = model_resnet.predict(valid_generator)
val_preds_classes = np.argmax(val_preds, axis=1)
true_classes = valid_generator.classes
class_labels = list(valid_generator.class_indices.keys())

# Classification report
print("ğŸ“Š Classification Report:")
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print(classification_report(true_classes, val_preds_classes, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(true_classes, val_preds_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Validation Set")
plt.show()



import os
BASE = "/kaggle/input/plant-pathology-2020-fgvc7"



from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

# Test verisini yeniden hazÄ±rlayalÄ±m
test_df = pd.read_csv(os.path.join(BASE, "test.csv"))
test_df['filename'] = test_df['image_id'] + ".jpg"

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=os.path.join(BASE, "images"),
    x_col="filename",
    y_col=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Kaydedilen en iyi modeli yÃ¼kle
from tensorflow.keras.models import load_model
model_resnet = load_model("best_model.h5")

# Tahmin al
preds = model_resnet.predict(test_generator, verbose=1)

# Submission dosyasÄ± oluÅŸtur
submission = pd.DataFrame(preds, columns=['healthy', 'multiple_diseases', 'rust', 'scab'])
submission.insert(0, 'image_id', test_df['image_id'])
submission.to_csv("submission.csv", index=False)

print("âœ… Submission dosyasÄ± oluÅŸturuldu!")
submission.head()


