# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


BASE_FOLDER = '/kaggle/input/cidaut-ai-fake-scene-classification-2024'
TRAIN_PATH = BASE_FOLDER + "/Train"
TEST_PATH = BASE_FOLDER + "/Test"
NUM_IMAGES = 720
BATCH_SIZE = 8
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720
LABELS_CSV_FILEPATH = BASE_FOLDER + "/train.csv"
WEIGHT_FILEPATH = "tmp/model.weights.h5"
PREDICTIONS_FILEPATH = "submission.csv"


import csv
import os
import tensorflow as tf
import keras
import numpy as np

def load_labels_from_csv_file():
    '''
        Load labels for files in alphanumerical order
        
        Returns:
            labels: array of size (NUM_IMAGES) where label = 0 (fake) or 1
    '''
    with open(LABELS_CSV_FILEPATH) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        image_label_pairs = [(row[0], 1 if row[1] == "real" else 0) for row in reader]
        image_label_pairs = sorted(image_label_pairs, key=lambda t: t[0])
        labels = [t[1] for t in image_label_pairs]
        print(image_label_pairs[:10])
        return labels

def build_dataset_from_directory():
    '''
        Returns:
            dataset: tf.data.Dataset[NUM_IMAGES]
    '''
    labels = load_labels_from_csv_file()
    train_dataset, val_dataset = keras.utils.image_dataset_from_directory(
        directory=TRAIN_PATH,
        labels=labels,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        validation_split=0.1,
        subset="both",
        shuffle=False
    )
    return train_dataset, val_dataset

def load_test_dataset_from_directory():
    '''
        Returns:
            dataset: tf.data.Dataset[NUM_IMAGES]
    '''
    test_dataset = keras.utils.image_dataset_from_directory(
        directory=TEST_PATH,
        labels=None,
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        shuffle=False
    )
    return test_dataset

train_dataset, val_dataset = build_dataset_from_directory()

test_filenames = sorted(list(os.listdir(TEST_PATH)))
test_dataset = load_test_dataset_from_directory()


# Hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 16
num_epochs = 30
image_width, image_height = 512, 512 # Resize to this size


import keras
from keras import layers
from keras import ops
import tensorflow as tf
from tensorflow.python.ops.signal.dct_ops import dct

regularizer = keras.regularizers.l2()

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(height_factor=(-0.2, -0.1), width_factor=(-0.2, -0.1)),
        layers.RandomBrightness(0.2)
    ],
    name="data_augmentation",
)

class DCTLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DCTLayer, self).__init__(**kwargs)

    def call(self, images, training = None):
        images = tf.transpose(images, [0, 3, 2, 1]) # after -> B, C, W, H
        images = dct(images, type=2, norm='ortho', axis=-1) # height
        images = tf.transpose(images, [0, 1, 3, 2]) # after -> B, C, H, W
        images = dct(images, type=2, norm='ortho', axis=-1) # width
        images = tf.transpose(images, [0, 2, 3, 1]) # after -> B, H, W, C
        images = tf.math.log(tf.abs(images) + 1e-13)
        return images


import keras
from keras import layers
from keras import ops

def create_classifier():
    inputs = keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    x = data_augmentation(inputs)
    x = DCTLayer()(x)
    
    x = layers.BatchNormalization()(x)
   
    x = layers.Conv2D(3, 3, padding="same", activation="relu", kernel_regularizer=regularizer)(x)
    x = layers.Conv2D(8, 3, padding="same", activation="relu", kernel_regularizer=regularizer)(x)
    x = layers.AveragePooling2D((2, 2))(x)  # 64
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(16, 3, padding="same", activation="relu", kernel_regularizer=regularizer)(x)
    x = layers.AveragePooling2D((2, 2))(x)  # 32
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=regularizer)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    return model


import os
import keras

def create_checkpoint_callback():
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        WEIGHT_FILEPATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )
    return checkpoint_callback

def run_experiment(model):
    os.makedirs("tmp", exist_ok=True)

    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC()
        ],
    )
    history = model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[create_checkpoint_callback()],
    )

    preds = model.predict(test_dataset)
    preds = np.squeeze(np.where(preds > 0.5, 1, 0))

    return history, preds

classifier = create_classifier()
history, preds = run_experiment(classifier)


import csv
import keras
import numpy as np
import os

def save_preds_to_file(preds):
    if os.path.exists(PREDICTIONS_FILEPATH):
      os.remove(PREDICTIONS_FILEPATH)
        
    header_row = ['image', 'label']
    data = zip(test_filenames, preds)
    data = sorted(data, key=lambda t: t[0])
    with open(PREDICTIONS_FILEPATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_row)
        writer.writerows(data)

save_preds_to_file(preds)

