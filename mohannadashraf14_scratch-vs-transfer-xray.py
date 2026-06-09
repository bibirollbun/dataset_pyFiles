import kagglehub

# Download latest version
path = kagglehub.dataset_download("bmadushanirodrigo/fracture-multi-region-x-ray-data")

print("Path to dataset files:", path)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D,BatchNormalization,MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from IPython.display import SVG, Image
from sklearn.model_selection import train_test_split
import os
import cv2
import random
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image, UnidentifiedImageError
import shutil
from shutil import copyfile
import plotly.express as px
import glob


train=path+'/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train'
test=path+'/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/test'
val=path+'/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/val'


os.listdir(train)


fractured_train_path = os.path.join(train, 'fractured')
not_fractured_train_path = os.path.join(train, 'not fractured')

fractured_train_files = os.listdir(fractured_train_path)
not_fractured_train_files = os.listdir(not_fractured_train_path)

print(fractured_train_files[:5])
print(not_fractured_train_files[:5])
print(len(fractured_train_files))
print(len(not_fractured_train_files))


plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = cv2.imread(os.path.join(fractured_train_path, fractured_train_files[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title('fractured')
    plt.tight_layout()
plt.show()


plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    img = cv2.imread(os.path.join(not_fractured_train_path, not_fractured_train_files[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title('not fractured')
    plt.tight_layout()
plt.show()


labels = {'fractured':1, 'not fractured':0}
data = [train, test, val]


plt.figure(figsize=(10,10))
for i in data:
    plt.subplot(2, 3, data.index(i)+1)
    #pie
    n_fractured = len(os.listdir(i+'/fractured'))
    n_not_fractured = len(os.listdir(i+'/not fractured'))
    plt.pie([n_fractured, n_not_fractured], labels=labels.keys(), autopct='%1.1f%%')
    plt.title(i.split('/')[-1])
plt.show()


from PIL import ImageFile, UnidentifiedImageError

def remove_truncate(path):
    removed = 0
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            try:
                img = Image.open(os.path.join(path, folder, file))
                img.verify()
            except (UnidentifiedImageError, OSError, IOError, SyntaxError) as e:
                print('Bad file:', os.path.join(path, folder, file))
                os.remove(os.path.join(path, folder, file))
                removed += 1
    return removed
print(remove_truncate(train))
print(remove_truncate(test))
print(remove_truncate(val))


ImageFile.LOAD_TRUNCATED_IMAGES = True
img_sizes =[]
for folder in os.listdir(train):
    for file in os.listdir(os.path.join(train, folder)):
      try:
        img = plt.imread(os.path.join(train, folder, file))
        img_sizes.append(img.shape)
      except Exception as e:
        print(e)


pd.Series(img_sizes).value_counts()


print(len(os.listdir(train+'/fractured')))
print(len(os.listdir(train+'/not fractured')))
print(len(os.listdir(test+'/fractured')))
print(len(os.listdir(test+'/not fractured')))
print(len(os.listdir(val+'/fractured')))
print(len(os.listdir(val+'/not fractured')))


train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train,
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'binary',
                                                 shuffle=True)

test_set = test_datagen.flow_from_directory(test,
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'binary')
validation_set = val_datagen.flow_from_directory(val,
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'binary')


class_names = training_set.class_indices
class_names = {v: k for k, v in class_names.items()}
def plot_data(set, n_images):
    images, labels = next(set)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
    plt.show()
plot_data(training_set, 25)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


improvements = [
    ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
]


history = model.fit(training_set, validation_data=validation_set, epochs=15, callbacks=improvements)


model.evaluate(test_set)


plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], color='green', linestyle='-', marker='o', markersize=5, label='Train Accuracy')
plt.plot(history.history['val_accuracy'], color='red', linestyle='--', marker='x', markersize=5, label='Validation Accuracy')

plt.title('Model Accuracy Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)

plt.legend(loc='upper left', fontsize=12)

plt.ylim(0, 1)

plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], color='green', linestyle='-', marker='o', markersize=5, label='Train Loss')
plt.plot(history.history['val_loss'], color='red', linestyle='--', marker='x', markersize=5, label='Validation Loss')

plt.title('Model Loss Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)

plt.legend(loc='upper left', fontsize=12)

plt.xlim(0, len(history.history['loss']) - 1)

plt.tight_layout()
plt.show()



from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input


input_shape = (224, 224, 3)


xception_mode=Xception(weights='imagenet',include_top=False,input_shape=input_shape)
xception_mode.summary()


for layer in xception_mode.layers:
      if isinstance(layer, BatchNormalization):
          layer.trainable = True
      else:
          layer.trainable = False


model=Sequential()
x = GlobalAveragePooling2D()(xception_mode.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output_layer = Dense(1, activation='sigmoid')(x)
model = Model(inputs=xception_mode.input, outputs=output_layer)
model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(training_set, validation_data=validation_set, epochs=15, callbacks=improvements)


model.evaluate(test_set)

