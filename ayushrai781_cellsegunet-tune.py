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
print("GPUs available:", tf.config.list_physical_devices('GPU'))



import os
import sys
import random

import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_mode
from keras.layers import Input,Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D,concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
np.random.seed(42)


import os

# List all datasets in the Kaggle input directory
print("Available datasets in /kaggle/input/:")
print(os.listdir("/kaggle/input/data-science-bowl-2018"))



from zipfile import ZipFile
import os

# Paths
zip_path = "/kaggle/input/data-science-bowl-2018/stage1_train.zip"  # Path to the zip file
extract_path = "./stage1_train"  # Path where files will be extracted

# Extract the zip file
with ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction completed!")



from zipfile import ZipFile
import os

# Paths
zip_path = "/kaggle/input/data-science-bowl-2018/stage1_test.zip"  # Path to the zip file
extract_path = "./stage1_test"  # Path where files will be extracted

# Extract the zip file
with ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction completed!")


TRAIN_PATH = "./stage1_train/"
TEST_PATH="./stage1_test/"


import os
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import numpy as np

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3

# Updated TRAIN_PATH
TRAIN_PATH = "./stage1_train/"
train_ids = next(os.walk(TRAIN_PATH))[1]

# Initialize arrays
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

print('Getting and resizing train images and masks ... ')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    # Construct paths
    path = os.path.join(TRAIN_PATH, id_)
    image_path = os.path.join(path, 'images', f"{id_}.png")
    
    try:
        # Load and resize the image
        img = imread(image_path)[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img

        # Process masks
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
        mask_dir = os.path.join(path, 'masks')
        for mask_file in next(os.walk(mask_dir))[2]:
            mask_ = imread(os.path.join(mask_dir, mask_file))
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask

    except FileNotFoundError:
        print(f"File not found: {image_path}")
    except NotADirectoryError:
        print(f"Error: {path} is not a directory. Did you extract the zip file?")



X_train.shape,Y_train.shape


print('Getting and resizing test images ... ')
test_ids = next(os.walk(TEST_PATH))[1]
sizes_test = []
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    # Construct paths
    path = os.path.join(TEST_PATH, id_)
    image_path = os.path.join(path, 'images', f"{id_}.png")
    
    try:
        # Load the original image and store its size
        img = imread(image_path)[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])

        # Resize the image
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
    except FileNotFoundError:
        print(f"Test image not found: {image_path}")
    except NotADirectoryError:
        print(f"Error: {path} is not a directory. Did you extract the zip file?")

print('Done!')



X_test.shape


from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

mask_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')



gen_seed=49
seed = gen_seed
image_generator = image_datagen.flow(X_train, batch_size=16, seed=gen_seed)
mask_generator = mask_datagen.flow(Y_train, batch_size=16, seed=gen_seed)
train_dataset = tf.data.Dataset.from_generator(
    lambda: zip(image_generator, mask_generator),
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)


ix = random.randint(0, len(train_ids))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()


# Attention block
def attention_block(input_x, gating_x, inter_filters):
    theta_x = layers.Conv2D(inter_filters, (1, 1), strides=(1, 1), padding='same')(input_x)
    phi_g = layers.Conv2D(inter_filters, (1, 1), strides=(1, 1), padding='same')(gating_x)
    add_xg = layers.Add()([theta_x, phi_g])
    relu_xg = layers.Activation('relu')(add_xg)
    psi = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same')(relu_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    upsample_psi = layers.Lambda(lambda inputs: tf.image.resize(inputs[0], tf.shape(inputs[1])[1:3]))([sigmoid_xg, input_x])

    multiply_xg = layers.Multiply()([input_x, upsample_psi])
    return multiply_xg

# Model blocks
def residual_block(x, filters, kernel_size=(3, 3)):
    shortcut = layers.Conv2D(filters, (1, 1), padding="same")(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

def encoder_block(x, filters, kernel_size):
    x = residual_block(x, filters, kernel_size)
    skip = x
    x = layers.MaxPooling2D((2, 2))(x)
    return x, skip

def decoder_block(x, skip, filters, kernel_size):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
    attn = attention_block(skip, x, filters // 2)
    x = layers.Concatenate()([x, attn])
    x = residual_block(x, filters, kernel_size)
    return x

def CellSegUNet(input_shape, kernel_size):
    inputs = layers.Input(input_shape)
    s = Lambda(lambda x: x / 255)(inputs)
    x1, skip1 = encoder_block(s, 32, kernel_size)
    x2, skip2 = encoder_block(x1, 64, kernel_size)
    x3, skip3 = encoder_block(x2, 128, kernel_size)
    x4, skip4 = encoder_block(x3, 256, kernel_size)
    bottleneck = residual_block(x4, 512, kernel_size)
    x = decoder_block(bottleneck, skip4, 256, kernel_size)
    x = decoder_block(x, skip3, 128, kernel_size)
    x = decoder_block(x, skip2, 64, kernel_size)
    x = decoder_block(x, skip1, 32, kernel_size)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    return Model(inputs, outputs)



FILTERS = [32, 64, 128, 256]


def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / (denominator + tf.keras.backend.epsilon())

def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred - y_true * y_pred)
    return intersection / (union + tf.keras.backend.epsilon())

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = 1 - dice_coefficient(y_true, y_pred)
    return bce + dice



print(pd.DataFrame(results))


from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Lambda

combinations = [((3, 3), 'adam'), ((5, 5), 'adam'), ((7, 7), 'adam'), ((7, 7), 'sgd'), ((3, 3), 'sgd'), ((5, 5), 'sgd'),((3, 3), 'rmsprop'), ((5, 5), 'rmsprop'), ((7, 7), 'rmsprop'), ((3, 3), 'adagrad'), ((5, 5), 'adagrad'), ((7, 7), 'adagrad')]
results = []

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

for kernel_size, optimizer_name in combinations:
    print(f"\nTraining with kernel_size={kernel_size}, optimizer={optimizer_name}")
    model = CellSegUNet(input_shape, kernel_size)
    model.summary()

    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
    elif optimizer_name == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=1e-2)
    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coefficient, iou])
    history = model.fit(train_dataset, steps_per_epoch=len(X_train) // 16, epochs=15, verbose=1)
    results.append({
            "kernel_size": kernel_size,
            "optimizer": optimizer_name,
            "dice": history.history['dice_coefficient'][-1],
            "iou": history.history['iou'][-1]
        })
print(pd.DataFrame(results))

