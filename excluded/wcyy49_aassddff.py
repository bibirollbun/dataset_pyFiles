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


import zipfile

# train.zipの解凍
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/train')

# test1.zipの解凍
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/test')


import shutil
import os
from sklearn.model_selection import train_test_split

original_dir = '/kaggle/working/train/train'
base_dir = '/kaggle/working/data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

for category in ['dog', 'cat']:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

all_filenames = os.listdir(original_dir)
train_filenames, val_filenames = train_test_split(all_filenames, test_size=0.2, random_state=42)


for filename in train_filenames:
    if 'dog' in filename:
        shutil.copy(os.path.join(original_dir, filename), os.path.join(train_dir, 'dog', filename))
    else:
        shutil.copy(os.path.join(original_dir, filename), os.path.join(train_dir, 'cat', filename))

for filename in val_filenames:
    if 'dog' in filename:
        shutil.copy(os.path.join(original_dir, filename), os.path.join(val_dir, 'dog', filename))
    else:
        shutil.copy(os.path.join(original_dir, filename), os.path.join(val_dir, 'cat', filename))


import tensorflow as tf
from tensorflow.keras import layers, models

img_size = 128
batch_size = 517

from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = 150
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_size, img_size),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(img_size, img_size),
                                                batch_size=batch_size,
                                                class_mode='binary')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


history = model.fit(train_generator,
                    epochs=10,
                    validation_data=val_generator)


test_dir = '/kaggle/working/test/test'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '/kaggle/working/',
    classes=['test'],
    target_size=(img_size, img_size),
    batch_size=1,
    class_mode=None,
    shuffle=False)

predictions = model.predict(test_generator, verbose=1)
predicted_classes = predictions.ravel()

import pandas as pd

filenames = test_generator.filenames
ids = [int(os.path.basename(fname).split('.')[0]) for fname in filenames]

submission = pd.DataFrame({'id': ids, 'label': predicted_classes})
submission.sort_values('id', inplace=True)
submission.to_csv('/kaggle/working/submission.csv', index=False)

import os
print(os.listdir())

