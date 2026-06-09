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


import matplotlib.pyplot as plt 
import seaborn as sns 
import cv2 

import json


base_dir = '/kaggle/input/cassava-leaf-disease-classification'

with open(os.path.join(base_dir, 'label_num_to_disease_map.json')) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k): v for k,v in map_classes.items()}


print('class_mapping')
print(json.dumps(map_classes, indent=1))


df_train = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')
df_train.head()


df_train['class_name'] = df_train['label'].map(map_classes)
df_train.head()


df_train['class_name'].value_counts()


train_path = '/kaggle/input/cassava-leaf-disease-classification/train_images'


import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers


from sklearn.model_selection import train_test_split

validation_split = .2

train_df, val_df = train_test_split(
    df_train, 
    test_size=validation_split,
    stratify = df_train['label'],
    random_state=2
)

print(f'train_shape: {train_df.shape}')
print(f'val_shape: {val_df.shape}')


def process_image(file_name, label):
    file_path = tf.strings.join([train_path, '/', file_name])
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224,224])
    return image, label


buffer_size=1000
train_ds = tf.data.Dataset.from_tensor_slices((train_df['image_id'].values,
                                             train_df['label'].values))

train_ds = train_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.shuffle(buffer_size).batch(32).prefetch(tf.data.AUTOTUNE)


val_ds = tf.data.Dataset.from_tensor_slices((val_df['image_id'].values, 
                                             val_df['label'].values))

val_ds = val_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)


model = Sequential()

model.add(layers.Input(shape=(224,224,3)))

model.add(layers.Rescaling(1/255.))
model.add(layers.RandomZoom(0.1))
model.add(layers.RandomFlip('horizontal'))
model.add(layers.RandomShear(.1))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D())

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D())

model.add(layers.Conv2D(256, (3,3), activation='relu'))
model.add(layers.Conv2D(256, (3,3), activation='relu'))
model.add(layers.MaxPool2D())

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.summary()


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model_in_custom_cnn.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

callbacks=[checkpoint]


history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()


base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3)
)


base_model.trainable = False


model = Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Lambda(tf.keras.applications.efficientnet.preprocess_input),


    layers.RandomZoom(0.1),
    layers.RandomFlip('horizontal'),
    layers.RandomShear(.1),

    base_model,

    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dropout(.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax'),
])

model.summary()


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model_in_transfer_learning.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

callbacks=[checkpoint]


history = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()


base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3)
)


for layer in base_model.layers[:-3]:
    layer.trainable = False

for layer in base_model.layers[-3:]:
    layer.trainable = True

model = Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Lambda(tf.keras.applications.efficientnet.preprocess_input),


    layers.RandomZoom(0.1),
    layers.RandomFlip('horizontal'),
    layers.RandomShear(.1),

    base_model,

    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(.2),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(.2),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax'),
])

model.summary()


model.compile(optimizer='rmsprop', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_fine_tuned_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

callbacks=[checkpoint]


history = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()


sample = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv')
sample.head(10)


def parse_test_tfrecord(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example, feature_description)

    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.resize(image, [224, 224])

    return image



import glob

test_tfrecords = glob.glob(
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"
)

test_ds = tf.data.TFRecordDataset(test_tfrecords)
test_ds = test_ds.map(parse_test_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)


predictions = []

for images in test_ds:
    preds = model.predict(images)
    labels = np.argmax(preds, axis=1)
    predictions.extend(labels)


sample = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

sample['label'] = predictions
sample.to_csv("submission.csv", index=False)

sample.head()


print(len(predictions), len(sample))



np.bincount(predictions)


print(preds.shape)




