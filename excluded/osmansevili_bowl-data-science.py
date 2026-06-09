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
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



import zipfile
TRAIN_ZIP = '/kaggle/input/data-science-bowl-2018/stage1_train.zip'
TEST_ZIP = '/kaggle/input/data-science-bowl-2018/stage1_test.zip'
# Directory where to extract
TRAIN_PATH = '/kaggle/working/stage1_train/'
TEST_PATH = '/kaggle/working/stage1_test/'

# Unzip the train data
with zipfile.ZipFile(TRAIN_ZIP, 'r') as zip_ref:
    zip_ref.extractall(TRAIN_PATH)

# Unzip the test data
with zipfile.ZipFile(TEST_ZIP, 'r') as zip_ref:
    zip_ref.extractall(TEST_PATH)

print("Extraction complete!")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Constants
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


train_ids = next(os.walk(TRAIN_PATH))[1]
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)


print(f'Getting and resizing {len(train_ids)} images and masks ... ')
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    
    # Load and combine all masks for this image
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), 
                                     mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    
    # Normalize mask to 0-1 range
    if mask.max() > 1.0:
        mask = mask / 255.0
    
    Y_train[n] = mask



# Normalize images
X_train = X_train / 255.0


# Split into train and validation
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.15, random_state=SEED
)
print(f'Train set: {X_train.shape}')
print(f'Validation set: {X_val.shape}')


# Input layer
inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))



# Encoder block 1
c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
c1 = layers.Dropout(0.2)(c1)
c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
p1 = layers.MaxPooling2D((2, 2))(c1)

# Encoder block 2
c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
c2 = layers.Dropout(0.2)(c2)
c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
p2 = layers.MaxPooling2D((2, 2))(c2)

# Encoder block 3
c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
c3 = layers.Dropout(0.2)(c3)
c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
p3 = layers.MaxPooling2D((2, 2))(c3)

# Encoder block 4
c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
c4 = layers.Dropout(0.2)(c4)
c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
p4 = layers.MaxPooling2D((2, 2))(c4)

# Bridge
c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
c5 = layers.Dropout(0.2)(c5)
c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

# Decoder block 1
u6 = layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(c5)
u6 = layers.Concatenate()([u6, c4])
c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
c6 = layers.Dropout(0.2)(c6)
c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

# Decoder block 2
u7 = layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(c6)
u7 = layers.Concatenate()([u7, c3])
c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
c7 = layers.Dropout(0.2)(c7)
c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

# Decoder block 3
u8 = layers.Conv2DTranspose(32, (2, 2), strides=2, padding='same')(c7)
u8 = layers.Concatenate()([u8, c2])
c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
c8 = layers.Dropout(0.2)(c8)
c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

# Decoder block 4
u9 = layers.Conv2DTranspose(16, (2, 2), strides=2, padding='same')(c8)
u9 = layers.Concatenate()([u9, c1])
c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
c9 = layers.Dropout(0.2)(c9)
c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

# Output layer
outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)



# Create model
model = models.Model(inputs=[inputs], outputs=[outputs], name='U-Net')


# Dice coefficient metric
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + 
                                           tf.keras.backend.sum(y_pred_f) + smooth)




# Dice loss
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)



# Combined BCE + Dice loss
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


# IoU metric
def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)



# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coefficient, iou_metric, 'accuracy']
)


model.summary()



# Setup callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]



# Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=16,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)



fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Dice Coefficient
axes[0, 1].plot(history.history['dice_coefficient'], label='Train Dice', linewidth=2)
axes[0, 1].plot(history.history['val_dice_coefficient'], label='Val Dice', linewidth=2)
axes[0, 1].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Dice Coefficient')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# IoU
axes[1, 0].plot(history.history['iou_metric'], label='Train IoU', linewidth=2)
axes[1, 0].plot(history.history['val_iou_metric'], label='Val IoU', linewidth=2)
axes[1, 0].set_title('IoU Metric', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('IoU')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Accuracy
axes[1, 1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
axes[1, 1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
axes[1, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()


print("\nGenerating predictions on validation set...")
Y_val_pred = model.predict(X_val, verbose=1)



# Visualize some predictions
print("\nVisualizing predictions...")
num_samples = 5
fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

for i in range(num_samples):
    idx = random.randint(0, len(X_val)-1)
    
    axes[i, 0].imshow(X_val[idx])
    axes[i, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(np.squeeze(Y_val[idx]), cmap='gray')
    axes[i, 1].set_title('True Mask', fontsize=12, fontweight='bold')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(np.squeeze(Y_val_pred[idx]), cmap='gray')
    axes[i, 2].set_title('Predicted Mask', fontsize=12, fontweight='bold')
    axes[i, 2].axis('off')

plt.tight_layout()


test_ids = next(os.walk(TEST_PATH))[1]
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []

print(f'Loading {len(test_ids)} test images...')

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    
# Normalize
X_test = X_test / 255.0

# Predict 
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions - try multiple thresholds
threshold = 0.5
preds_test_t = (preds_test > threshold).astype(np.uint8)




# Run-length encoding function
def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Convert predictions to RLE
new_test_ids = []
rles = []

print("Encoding predictions...")
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    # Get prediction and squeeze to 2D
    pred = preds_test_t[n].squeeze()
    
    # Label each connected component (each nucleus)
    lab_img = label(pred)
    
    # If no nuclei detected, add empty prediction
    if lab_img.max() == 0:
        new_test_ids.append(id_)
        rles.append('')
    else:
        # Encode each nucleus separately
        for i in range(1, lab_img.max() + 1):
            nucleus_mask = (lab_img == i).astype(np.uint8)
            rle = rle_encode(nucleus_mask)
            rles.append(rle)
            new_test_ids.append(id_)

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = rles
sub.to_csv('submission.csv', index=False)




sub




