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

from keras.models import Model, load_model
from keras.layers import Input,Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D,concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


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


ix = random.randint(0, len(train_ids))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()


import tensorflow as tf
from keras import layers, Model
from keras.layers import Lambda

def residual_block(x, filters):
    # Save the input as a shortcut
    shortcut = layers.Conv2D(filters, (1, 1), padding="same")(x)  # Match the number of filters

    # Apply convolutions
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Add shortcut to the output
    x = layers.Add()([shortcut, x])  # Now shapes match
    x = layers.ReLU()(x)
    return x


def attention_block(input_x, gating_x, inter_filters):
    theta_x = layers.Conv2D(inter_filters, (1, 1), strides=(1, 1), padding='same')(input_x)
    phi_g = layers.Conv2D(inter_filters, (1, 1), strides=(1, 1), padding='same')(gating_x)

    add_xg = layers.Add()([theta_x, phi_g])
    relu_xg = layers.Activation('relu')(add_xg)

    psi = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same')(relu_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)

    upsample_psi = layers.UpSampling2D(size=(input_x.shape[1] // sigmoid_xg.shape[1],
                                             input_x.shape[2] // sigmoid_xg.shape[2]))(sigmoid_xg)

    upsample_psi = layers.Reshape(input_x.shape[1:])(upsample_psi)
    multiply_xg = layers.Multiply()([input_x, upsample_psi])

    return multiply_xg

def encoder_block(x, filters):
    x = residual_block(x, filters)
    skip = x
    x = layers.MaxPooling2D((2, 2))(x)
    return x, skip

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = residual_block(x, filters)
    return x

def CellSegUNet(input_shape):
    inputs = layers.Input(input_shape)

    # Scale inputs
    s = Lambda(lambda x: x / 255) (inputs)

    # Encoder
    x1, skip1 = encoder_block(s, 32)
    x2, skip2 = encoder_block(x1, 64)
    x3, skip3 = encoder_block(x2, 128)
    x4, skip4 = encoder_block(x3, 256)

    # Bottleneck
    bottleneck = residual_block(x4, 512)

    # Decoder
    x = decoder_block(bottleneck, skip4, 256)
    x = decoder_block(x, skip3, 128)
    x = decoder_block(x, skip2, 64)
    x = decoder_block(x, skip1, 32)

    # Output Layer
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)

    return Model(inputs, outputs)

# Define the model
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = CellSegUNet(input_shape)
model.summary()






def dice_coefficient(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / (denominator + tf.keras.backend.epsilon())
    
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)
    
def iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred - y_true * y_pred)
    return intersection / (union + tf.keras.backend.epsilon())
Y_train = Y_train.astype('float32')


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=dice_loss,  #  Dice Loss
    metrics=[dice_coefficient, iou]  # Dice Coefficient and IoU
)



earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.keras', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=80, 
                    callbacks=[earlystopper, checkpointer])


import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

# Predict on test data
print("Predicting on test data...")
predictions = model.predict(X_test, verbose=1)

# Apply a threshold to convert probabilities to binary masks
threshold = 0.5
predictions_binary = (predictions > threshold).astype(np.uint8)

# Reshape predictions back to their original sizes
resized_predictions = []
for i, prediction in enumerate(predictions_binary):
    original_size = sizes_test[i]
    resized_pred = resize(prediction.squeeze(), 
                          (original_size[0], original_size[1]), 
                          mode='constant', preserve_range=True)
    resized_predictions.append((resized_pred > threshold).astype(np.uint8))

# Convert the list to a NumPy array


print("Predictions complete!")



# Visualize test image, ground truth, and prediction
for i in range(3):  # Display first 3 test samples
    plt.figure(figsize=(10, 10))
    
    plt.subplot(1, 3, 1)
    plt.title("Test Image")
    plt.imshow(X_test[i])
    
    plt.subplot(1, 3, 2)
    plt.title("Prediction (Resized)")
    plt.imshow(resized_predictions[i], cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Binary Mask Prediction")
    plt.imshow(predictions_binary[i].squeeze(), cmap='gray')
    
    plt.show()



import matplotlib.pyplot as plt

# Access the loss values from the history object
training_loss = results.history['loss']
validation_loss = results.history['val_loss']

# Plot the training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()



import matplotlib.pyplot as plt



# Check if the metrics (e.g., Dice coefficient) are available
if 'dice_coefficient' in results.history:
    training_metric = results.history['dice_coefficient']
    validation_metric = results.history['val_dice_coefficient']
    metric_name = "Dice Coefficient"

else:
    raise ValueError("Metric not found in history. Ensure it was defined during model.compile().")


# Plot training and validation metrics
plt.subplot(1, 2, 2)
plt.plot(training_metric, label=f'Training {metric_name}')
plt.plot(validation_metric, label=f'Validation {metric_name}')
plt.title(f'{metric_name} Over Epochs')
plt.xlabel('Epochs')
plt.ylabel(metric_name)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()



if 'iou' in results.history:
    training_metric = results.history['iou']
    validation_metric = results.history['val_iou']
    metric_name = "IoU"
else:
    raise ValueError("Metric not found in history. Ensure it was defined during model.compile().")
plt.subplot(1, 2, 2)
plt.plot(training_metric, label=f'Training {metric_name}')
plt.plot(validation_metric, label=f'Validation {metric_name}')
plt.title(f'{metric_name} Over Epochs')
plt.xlabel('Epochs')
plt.ylabel(metric_name)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


