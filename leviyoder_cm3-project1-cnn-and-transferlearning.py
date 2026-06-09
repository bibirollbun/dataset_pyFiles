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
import zipfile
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from functools import partial
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

tf.random.set_seed(72)


zip_dir = '/kaggle/input/dogs-vs-cats'
with zipfile.ZipFile(os.path.join(zip_dir, 'train.zip'), 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/train')

with zipfile.ZipFile(os.path.join(zip_dir, 'test1.zip'), 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/test')


#!unzip -q /kaggle/input/dogs-vs-cats/train.zip -d /kaggle/working/train
#!unzip -q /kaggle/input/dogs-vs-cats/test1.zip -d /kaggle/working/test


train_path = '/kaggle/working/train/train'
test_path = '/kaggle/working/test/test1'

print("Number of training images:", len(os.listdir(train_path)))
print("Number of test images:", len(os.listdir(test_path)))


import shutil

train_path = '/kaggle/working/train/train'
sorted_train_path = '/kaggle/working/train/sorted_train'

os.makedirs(os.path.join(sorted_train_path, 'dogs'), exist_ok=True)
os.makedirs(os.path.join(sorted_train_path, 'cats'), exist_ok=True)

for filename in os.listdir(train_path):
    if filename.startswith('dog'):
        shutil.move(os.path.join(train_path, filename), os.path.join(sorted_train_path, 'dogs', filename))
    elif filename.startswith('cat'):
        shutil.move(os.path.join(train_path, filename), os.path.join(sorted_train_path, 'cats', filename))


img_size = (224, 224)  

datagen = ImageDataGenerator(
    rescale=1./255,       
    validation_split=0.2  
)

train_gen = datagen.flow_from_directory(
    '/kaggle/working/train/sorted_train',
    target_size=img_size,     
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    '/kaggle/working/train/sorted_train',
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


images_batch, labels_batch = next(train_gen)

img = images_batch[0]
img2 = images_batch[1]
label = labels_batch[0]
label2 = labels_batch[1]

# Plot
plt.imshow(img)  
plt.title('Dog' if label == 1 else 'Cat')
plt.axis('off')
plt.show()

plt.imshow(img2)
plt.title('Dog' if label2 == 1 else 'Cat')
plt.axis('off')
plt.show()


test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    '/kaggle/working/test',  
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  
    shuffle=False     
)


DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")

model_1 = tf.keras.Sequential([
    Input(shape=(224, 224, 3)),
    DefaultConv2D(filters=64, kernel_size=7,),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
])


model_1.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(
    weights='imagenet',        
    include_top=False,         
    input_shape=(224, 224, 3) 
)

base_model.trainable = False


model_2 = Sequential([
    base_model,
    GlobalAveragePooling2D(),  
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])


model_2.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


history_1 = model_1.fit(
    train_gen,               
    validation_data=val_gen, 
    epochs=5,               
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_steps=val_gen.samples // val_gen.batch_size)


history_2 = model_2.fit(
    train_gen,               
    validation_data=val_gen, 
    epochs=5,               
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_steps=val_gen.samples // val_gen.batch_size)


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import math


val_gen.reset() 
y_pred_prob = model_1.predict(val_gen, steps=math.ceil(val_gen.samples / val_gen.batch_size))


y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)


y_true = val_gen.classes  

val_gen = datagen.flow_from_directory(
    '/kaggle/working/train/sorted_train',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False   
)


cm = confusion_matrix(y_true, y_pred)


plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_gen.class_indices.keys(), yticklabels=val_gen.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


val_gen.reset() 
y_pred_prob = model_2.predict(val_gen, steps=math.ceil(val_gen.samples / val_gen.batch_size))

y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

y_true = val_gen.classes  

val_gen = datagen.flow_from_directory(
    '/kaggle/working/train/sorted_train',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False  
)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_gen.class_indices.keys(), yticklabels=val_gen.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


h1 = history_1.history
h2 = history_2.history

epochs_1 = range(1, len(h1['loss']) + 1)
epochs_2 = range(1, len(h2['loss']) + 1)

plt.figure(figsize=(16, 6))

# Plot Training & Validation Loss 
plt.subplot(1, 2, 1)
plt.plot(epochs_1, h1['loss'], 'b-', label='Model 1 Training Loss')
plt.plot(epochs_1, h1['val_loss'], 'b--', label='Model 1 Validation Loss')
plt.plot(epochs_2, h2['loss'], 'r-', label='Model 2 Training Loss')
plt.plot(epochs_2, h2['val_loss'], 'r--', label='Model 2 Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Training & Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_1, h1['accuracy'], 'b-', label='Model 1 Training Accuracy')
plt.plot(epochs_1, h1['val_accuracy'], 'b--', label='Model 1 Validation Accuracy')
plt.plot(epochs_2, h2['accuracy'], 'r-', label='Model 2 Training Accuracy')
plt.plot(epochs_2, h2['val_accuracy'], 'r--', label='Model 2 Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


steps = int(np.ceil(test_gen.samples / test_gen.batch_size))

pred_probs = model_2.predict(test_gen, steps=steps)


pred_labels = (pred_probs > 0.5).astype(int).reshape(-1)


filenames = test_gen.filenames
results = list(zip(filenames, pred_labels))


for fname, label in results[:5]:
    print(fname, 'Dog' if label==1 else 'Cat')


submission = pd.DataFrame({
    'id': filenames,
    'label': pred_labels
})
submission.to_csv('submission.csv', index=False)

