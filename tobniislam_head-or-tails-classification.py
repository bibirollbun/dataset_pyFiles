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
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight


tf.random.set_seed(42)
np.random.seed(42)


# Configuration
IMG_SIZE = (576, 576)
BATCH_SIZE = 128
EPOCHS = 200
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 2e-4


from PIL import Image
import matplotlib.pyplot as plt


heads=os.listdir('/kaggle/input/heads-or-tails-image-classification/train/heads') 
tails=os.listdir('/kaggle/input/heads-or-tails-image-classification/train/tails')

plt.figure(figsize=(10, 5))
for i in range(3):
    img=Image.open('/kaggle/input/heads-or-tails-image-classification/train/heads/'+heads[i])
    plt.subplot(2, 3, i+1)
    plt.title('Head '+str(i+1))
    plt.axis('off')
    plt.imshow(img)
for i in range(3):
    img=Image.open('/kaggle/input/heads-or-tails-image-classification/train/tails/'+tails[i])
    plt.subplot(2, 3, i+4)
    plt.title('Tail '+str(i+4))
    plt.axis('off')
    plt.imshow(img)


datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT
)


import os
#Class weights
num_heads=len(os.listdir('/kaggle/input/heads-or-tails-image-classification/train/heads'))
num_tails=len(os.listdir('/kaggle/input/heads-or-tails-image-classification/train/tails'))

total = num_heads+num_tails
class_weights={
    '1': total / (2*num_heads),
    '0': total / (2*num_tails)
}


train_ds = datagen.flow_from_directory(
    '/kaggle/input/heads-or-tails-image-classification/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training',
    shuffle=True,
    class_mode='binary',
    color_mode='grayscale',
    seed=42
)


val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)


val_ds = val_gen.flow_from_directory(
    '/kaggle/input/heads-or-tails-image-classification/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation',
    shuffle=True,
    class_mode='binary',
    color_mode='grayscale',
    seed=42
)


model = Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    Conv2D(64, (3, 3), (2, 2), padding='valid', activation='relu', 
          kernel_initializer='uniform'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), (2, 2), padding='valid', activation='relu',
          kernel_initializer='uniform'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), (1, 1), padding='valid', activation='relu',
          kernel_initializer='uniform'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.4),
    Dense(1023, activation='relu', kernel_initializer='uniform'),
    Dropout(0.4),
    Dense(512, activation='relu', kernel_initializer='uniform'),
    Dropout(0.3),
    Dense(256, activation='relu', kernel_initializer='uniform'),
    Dense(1, activation='sigmoid')
])


model.summary()


early_stopping = EarlyStopping(
    monitor='val_auc',
    patience=20,
    verbose=1,
    mode='max',
    restore_best_weights=True,
)

model_checkpoint = ModelCheckpoint(
    'best_efficientnet.h5',
    monitor='val_auc',
    mode='max',
    verbose=1,
    save_best_only=True
)


model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[AUC(name='auc'), 'accuracy']
)


history=model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=[early_stopping, model_checkpoint]
)


def plot_training_history(history):
    plt.figure(figsize=(20, 7))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)


test_gen = ImageDataGenerator(
    rescale=1./255,
)


test_ds = test_gen.flow_from_directory(
    '/kaggle/input/heads-or-tails-image-classification',
    classes=['test'],
    batch_size=1,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    shuffle=False
)


pred = model.predict(test_ds)


submission=pd.DataFrame({
    'prediction_id': list(range(1,len(pred)+1)),
    'probability_of_heads': 1-pred.reshape((len(pred),))
})
submission.head()


submission.to_csv('submission3.csv', index=False)

