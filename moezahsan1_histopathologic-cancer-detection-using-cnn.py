# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np 
import pandas as pd

import os
base_dir = '../input/'
print(os.listdir(base_dir))


import matplotlib.pyplot as plt
plt.style.use("ggplot")


import cv2


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


import sklearn
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image


full_train_df = pd.read_csv("../input/histopathologic-cancer-detection/train_labels.csv")
full_train_df.head()



print("Train Size: {}".format(len(os.listdir('../input/histopathologic-cancer-detection/train/'))))
print("Test Size: {}".format(len(os.listdir('../input/histopathologic-cancer-detection/test/'))))


labels_count = full_train_df.label.value_counts()
plt.figure(figsize=(6,4))

labels_count.plot(kind='barh')

plt.title("Class Distribution")
plt.xlabel("Number of Samples")
plt.ylabel("Class")
plt.grid(axis='x', alpha=0.3)

plt.show()





SAMPLE_SIZE = 80000


train_path = '../input/histopathologic-cancer-detection/train/'
test_path = '../input/histopathologic-cancer-detection/test/'


df_negatives = full_train_df[full_train_df['label'] == 0].sample(SAMPLE_SIZE, random_state=42)
df_positives = full_train_df[full_train_df['label'] == 1].sample(SAMPLE_SIZE, random_state=42)


train_df = sklearn.utils.shuffle(pd.concat([df_positives, df_negatives], axis=0).reset_index(drop=True))

train_df.shape




train_df['id'] = train_df['id'].apply(lambda x: x + '.tif')
full_train_df['id'] = full_train_df['id'].apply(lambda x: x + '.tif')


train_split, valid_split = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)



test_datagen = ImageDataGenerator(rescale=1./255)

sample_sub = pd.read_csv("../input/histopathologic-cancer-detection/sample_submission.csv")
sample_sub['id'] = sample_sub['id'].apply(lambda x: x + '.tif')

test_generator = test_datagen.flow_from_dataframe(
    dataframe=sample_sub,
    directory=test_path,
    x_col='id',
    y_col=None,
    target_size=(96, 96),
    batch_size=128,
    class_mode=None,
    shuffle=False
)



train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)


valid_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_split,
    directory=train_path,
    x_col='id',
    y_col='label',
    target_size=(96, 96),
    batch_size=128,
    class_mode='raw',
    seed=42
)


valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_split,
    directory=train_path,
    x_col='id',
    y_col='label',
    target_size=(96, 96),
    batch_size=128,
    class_mode='raw',
    seed=42
)



model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (2, 2), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.4),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.00015), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model.summary()



early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=16,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    callbacks=[early_stop, reduce_lr]
)



model.save('best_model.h5')



preds = model.predict(test_generator, steps=len(test_generator))


preds = preds.flatten()


sample_sub['label'] = preds
sample_sub['id'] = sample_sub['id'].str.replace('.tif', '')  
sample_sub.to_csv('submission.csv', index=False)
sample_sub.head()


print("Hello")

