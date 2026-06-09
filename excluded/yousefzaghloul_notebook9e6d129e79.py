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


import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load train.csv
train_df = pd.read_csv('/kaggle/input/ai-dl-multiclass-segmentation/train.csv')

# Function to decode RLE to mask
def decode_rle_to_mask(rle, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)
    if pd.isna(rle):
        return mask.reshape((height, width))
    rle = list(map(int, rle.split()))
    for i in range(0, len(rle), 2):
        start, length = rle[i], rle[i + 1]
        mask[start:start + length] = 1
    return mask.reshape((height, width))



# Define the number of classes
NUM_CLASSES = 4  # 1: Head, 2: Body, 3: Legs, 4: Tail

# Function to create multi-class mask
def create_multi_class_mask(df, image_name, height, width):
    mask = np.zeros((height, width, NUM_CLASSES), dtype=np.uint8)
    image_df = df[df['ImageName'] == image_name]
    for _, row in image_df.iterrows():
        class_num = row['ClassNumber']
        rle = row['Encoding']
        single_mask = decode_rle_to_mask(rle, height, width)
        mask[:, :, class_num - 1] = single_mask  # Classes 1-4 mapped to 0-3
    return mask

# Example visualization
sample_image = '2008_006280'
sample_height = 375
sample_width = 500
sample_mask = create_multi_class_mask(train_df, sample_image, sample_height, sample_width)

plt.figure(figsize=(10, 10))
for i in range(NUM_CLASSES):
    plt.subplot(2, 2, i+1)
    plt.imshow(sample_mask[:, :, i], cmap='gray')
    plt.title(f'Class {i+1}')
    plt.axis('off')
plt.show()


# Get unique image names
image_names = train_df['ImageName'].unique()

# Split into train and validation
train_images, val_images = train_test_split(image_names, test_size=0.2, random_state=42)

# Verify the split
print(f'Training samples: {len(train_images)}')
print(f'Validation samples: {len(val_images)}')


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define augmentation parameters
data_gen_args = dict(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create ImageDataGenerator instances
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Seed for reproducibility
seed = 1



import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def unet_model(input_size=(500, 500, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(input_size)
    
    # Pretrained ResNet34 as encoder
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Extract encoder layers
    layer_names = [
        'conv1_relu',   # 256x256
        'conv2_block3_out',  # 128x128
        'conv3_block4_out',  # 64x64
        'conv4_block6_out',  # 32x32
        'conv5_block3_out',  # 16x16
    ]
    layers_output = [base_model.get_layer(name).output for name in layer_names]
    
    # Create the encoder
    encoder = models.Model(inputs, layers_output)
    
    # Decoder
    x = encoder.output[-1]
    for i in reversed(range(len(layers_output) - 1)):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, encoder.output[i]])
        x = layers.Conv2D(256 // (2**i), (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256 // (2**i), (3, 3), activation='relu', padding='same')(x)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

# Instantiate the model
model = unet_model()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)])
              
model.summary()






from sklearn.utils import class_weight

# Calculate class weights
# Flatten all masks to compute class weights
all_classes = []
for image in train_images:
    mask = create_multi_class_mask(train_df, image, 375, 500)
    all_classes.extend(mask.flatten())

class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(all_classes),
                                                  y=all_classes)
class_weights_dict = {i: class_weights[i-1] for i in range(1, NUM_CLASSES+1)}
print(class_weights_dict)


# Import Necessary Libraries
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import jaccard_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress TensorFlow warnings for clarity
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Define Constants
NUM_CLASSES = 4  # 1: Head, 2: Body, 3: Legs, 4: Tail
IMAGE_SIZE = (512, 512)  # Updated from (500, 500) to (512, 512)
BATCH_SIZE = 16
EPOCHS = 50
TRAIN_IMAGES_DIR = '/kaggle/input/ai-dl-multiclass-segmentation/TrainImages'
TEST_IMAGES_DIR = '/kaggle/input/ai-dl-multiclass-segmentation/TestImages'
TRAIN_CSV = '/kaggle/input/ai-dl-multiclass-segmentation/train.csv'
TEST_CSV = '/kaggle/input/ai-dl-multiclass-segmentation/test_class.csv'
SUBMISSION_FILE = 'submission.csv'


# 1. Data Preprocessing Functions

def decode_rle_to_mask(rle, height, width):
    """
    Decodes a Run-Length Encoded (RLE) mask into a binary mask.
    
    Parameters:
    - rle (str): RLE string.
    - height (int): Height of the mask.
    - width (int): Width of the mask.
    
    Returns:
    - np.array: Decoded binary mask of shape (height, width).
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    if pd.isna(rle):
        return mask.reshape((height, width))
    rle = list(map(int, rle.split()))
    for i in range(0, len(rle), 2):
        start, length = rle[i], rle[i + 1]
        start -= 1  # Convert to zero-based index
        mask[start:start + length] = 1
    return mask.reshape((height, width))

def create_multi_class_mask(df, image_name, height, width):
    """
    Creates a multi-channel mask for a given image.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing mask information.
    - image_name (str): Name of the image file.
    - height (int): Original height of the image.
    - width (int): Original width of the image.
    
    Returns:
    - np.array: Multi-channel mask of shape (height, width, NUM_CLASSES).
    """
    mask = np.zeros((height, width, NUM_CLASSES), dtype=np.uint8)
    image_df = df[df['ImageName'] == image_name]
    for _, row in image_df.iterrows():
        class_num = row['ClassNumber']
        rle = row['Encoding']
        single_mask = decode_rle_to_mask(rle, height, width)
        mask[:, :, class_num - 1] = single_mask  # Classes 1-4 mapped to 0-3
    return mask

def encode_mask_to_rle(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).
    
    Parameters:
    - mask (np.array): Binary mask of shape (height, width).
    
    Returns:
    - str: RLE string.
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def preprocess_test_image(image_path, target_size=IMAGE_SIZE):
    """
    Preprocesses a test image for prediction.
    
    Parameters:
    - image_path (str): Path to the test image.
    - target_size (tuple): Desired image size.
    
    Returns:
    - np.array: Preprocessed image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return img


# 2. Load and Parse Data

# Load train.csv
train_df = pd.read_csv(TRAIN_CSV)

# Get unique image names
image_names = train_df['ImageName'].unique()

# Split into training and validation sets
train_images, val_images = train_test_split(image_names, test_size=0.2, random_state=42)

print(f'Training samples: {len(train_images)}')
print(f'Validation samples: {len(val_images)}')


# 3. Data Generators

def generate_data(df, image_names, batch_size=BATCH_SIZE):
    """
    Generator that yields batches of images and masks.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing mask information.
    - image_names (list): List of image names.
    - batch_size (int): Number of samples per batch.
    
    Yields:
    - Tuple of (images, masks).
    """
    while True:
        for start in range(0, len(image_names), batch_size):
            end = min(start + batch_size, len(image_names))
            batch_images = image_names[start:end]
            images = []
            masks = []
            for image in batch_images:
                img_path = os.path.join(TRAIN_IMAGES_DIR, f'{image}.jpg')
                img = cv2.imread(img_path)
                if img is None:
                    print(f'Warning: Image {img_path} not found.')
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0  # Normalize

                # Get original image dimensions from DataFrame
                img_df = df[df['ImageName'] == image]
                # Assuming all entries for an image have the same height and width
                orig_height = img_df['ImageHeight'].iloc[0]
                orig_width = img_df['ImageWidth'].iloc[0]

                mask = create_multi_class_mask(df, image, orig_height, orig_width)
                mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
                masks.append(mask)
                images.append(img)

            images = np.array(images)
            masks = np.array(masks)
            masks = to_categorical(masks, num_classes=NUM_CLASSES)
            yield images, masks

# Create training and validation generators
train_gen = generate_data(train_df, train_images, batch_size=BATCH_SIZE)
val_gen = generate_data(train_df, val_images, batch_size=BATCH_SIZE)


# 4. Model Definition

def unet_model(input_size=(512, 512, 3), num_classes=NUM_CLASSES):
    """
    Builds a U-Net model with ResNet50 backbone.
    
    Parameters:
    - input_size (tuple): Shape of the input images.
    - num_classes (int): Number of segmentation classes.
    
    Returns:
    - tf.keras.Model: Compiled U-Net model.
    """
    inputs = layers.Input(input_size)
    
    # Use ResNet50 as the backbone
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Specify the layers for skip connections
    layer_names = [
        'conv1_relu',            # 256x256
        'conv2_block3_out',      # 128x128
        'conv3_block4_out',      # 64x64
        'conv4_block6_out',      # 32x32
        'conv5_block3_out',      # 16x16
    ]
    layers_output = [base_model.get_layer(name).output for name in layer_names]
    
    # Create the encoder model
    encoder = models.Model(inputs, layers_output)
    
    # Decoder
    x = encoder.output[-1]  # Start from the deepest layer
    for i in reversed(range(len(layers_output) - 1)):
        x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        skip_connection = encoder.output[i]
        x = layers.Concatenate()([x, skip_connection])
        x = layers.Conv2D(256 // (2**i), (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256 // (2**i), (3, 3), activation='relu', padding='same')(x)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

# Instantiate the model
model = unet_model(input_size=(512, 512, 3), num_classes=NUM_CLASSES)


# 5. Compile the Model

# Define Dice Loss
def dice_loss(y_true, y_pred, smooth=1):
    """
    Computes Dice Loss.
    
    Parameters:
    - y_true (tf.Tensor): Ground truth one-hot encoded masks.
    - y_pred (tf.Tensor): Predicted masks.
    - smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
    - tf.Tensor: Dice loss.
    """
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) + smooth)

# Define Combined Loss (Categorical Crossentropy + Dice Loss)
def combined_loss(y_true, y_pred):
    """
    Combines Categorical Crossentropy and Dice Loss.
    
    Parameters:
    - y_true (tf.Tensor): Ground truth one-hot encoded masks.
    - y_pred (tf.Tensor): Predicted masks.
    
    Returns:
    - tf.Tensor: Combined loss.
    """
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# Compile the model with Combined Loss
model.compile(optimizer='adam',
              loss=combined_loss,
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)])

# Display Model Summary
model.summary()


# 6. Define Callbacks

checkpoint = ModelCheckpoint('best_model.keras', 
                             monitor='val_mean_io_u', 
                             mode='max', 
                             verbose=1, 
                             save_best_only=True)

early_stopping = EarlyStopping(monitor='val_mean_io_u', 
                               mode='max', 
                               patience=10, 
                               verbose=1, 
                               restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_mean_io_u', 
                              mode='max', 
                              factor=0.5, 
                              patience=5, 
                              verbose=1)

callbacks = [checkpoint, early_stopping, reduce_lr]


# 7. Train the Model

# Calculate steps per epoch
steps_per_epoch = len(train_images) // BATCH_SIZE
validation_steps = len(val_images) // BATCH_SIZE

# Fit the model
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks
)



# 3. Data Preprocessing Functions

def decode_rle_to_mask(rle, height, width):
    """
    Decodes a Run-Length Encoded (RLE) mask into a binary mask.

    Parameters:
    - rle (str): RLE string.
    - height (int): Height of the mask.
    - width (int): Width of the mask.

    Returns:
    - np.array: Decoded binary mask of shape (height, width).
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    if pd.isna(rle):
        return mask.reshape((height, width))
    rle = list(map(int, rle.split()))
    for i in range(0, len(rle), 2):
        start, length = rle[i], rle[i + 1]
        start -= 1  # Convert to zero-based index
        mask[start:start + length] = 1
    return mask.reshape((height, width))

def create_multi_class_mask(df, image_name, height, width):
    """
    Creates a single-channel mask with class IDs for a given image.

    Parameters:
    - df (pd.DataFrame): DataFrame containing mask information.
    - image_name (str): Name of the image file.
    - height (int): Original height of the image.
    - width (int): Original width of the image.

    Returns:
    - np.array: Single-channel mask of shape (height, width) with class IDs.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    image_df = df[df['ImageName'] == image_name]
    for _, row in image_df.iterrows():
        class_num = row['ClassNumber']
        rle = row['Encoding']
        single_mask = decode_rle_to_mask(rle, height, width)
        mask[single_mask == 1] = class_num - 1  # Classes 1-4 mapped to 0-3
    return mask

def encode_mask_to_rle(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Parameters:
    - mask (np.array): Binary mask of shape (height, width).

    Returns:
    - str: RLE string.
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def preprocess_test_image(image_path, target_size=IMAGE_SIZE):
    """
    Preprocesses a test image for prediction.

    Parameters:
    - image_path (str): Path to the test image.
    - target_size (tuple): Desired image size.

    Returns:
    - np.array: Preprocessed image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return img

# 4. Load and Split Data

# Load train.csv
train_df = pd.read_csv(TRAIN_CSV)

# Get unique image names
image_names = train_df['ImageName'].unique()

# Split into training and validation sets
train_images, val_images = train_test_split(image_names, test_size=0.2, random_state=42)

print(f'Training samples: {len(train_images)}')
print(f'Validation samples: {len(val_images)}')


# 5. Data Generators

def generate_data(df, image_names, batch_size=BATCH_SIZE):
    """
    Generator that yields batches of images and masks.

    Parameters:
    - df (pd.DataFrame): DataFrame containing mask information.
    - image_names (list): List of image names.
    - batch_size (int): Number of samples per batch.

    Yields:
    - Tuple of (images, masks).
    """
    while True:
        for start in range(0, len(image_names), batch_size):
            end = min(start + batch_size, len(image_names))
            batch_images = image_names[start:end]
            images = []
            masks = []
            for image in batch_images:
                img_path = os.path.join(TRAIN_IMAGES_DIR, f'{image}.jpg')
                img = cv2.imread(img_path)
                if img is None:
                    print(f'Warning: Image {img_path} not found.')
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0  # Normalize

                # Get original image dimensions from DataFrame
                img_df = df[df['ImageName'] == image]
                # Assuming all entries for an image have the same height and width
                orig_height = img_df['ImageHeight'].iloc[0]
                orig_width = img_df['ImageWidth'].iloc[0]

                mask = create_multi_class_mask(df, image, orig_height, orig_width)
                mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
                masks.append(mask)
                images.append(img)

            images = np.array(images)
            masks = np.array(masks)
            masks = to_categorical(masks, num_classes=NUM_CLASSES)  # Shape: (batch_size, 512, 512, 4)
            yield images, masks

# Create training and validation generators
train_gen = generate_data(train_df, train_images, batch_size=BATCH_SIZE)
val_gen = generate_data(train_df, val_images, batch_size=BATCH_SIZE)


# 6. Model Definition

def unet_model(input_size=(512, 512, 3), num_classes=NUM_CLASSES):
    """
    Builds a U-Net model with ResNet50 backbone.

    Parameters:
    - input_size (tuple): Shape of the input images.
    - num_classes (int): Number of segmentation classes.

    Returns:
    - tf.keras.Model: Compiled U-Net model.
    """
    inputs = layers.Input(input_size)

    # Use ResNet50 as the backbone
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Specify the layers for skip connections
    layer_names = [
        'conv1_relu',            # 256x256
        'conv2_block3_out',      # 128x128
        'conv3_block4_out',      # 64x64
        'conv4_block6_out',      # 32x32
        'conv5_block3_out',      # 16x16
    ]
    layers_output = [base_model.get_layer(name).output for name in layer_names]

    # Create the encoder model
    encoder = models.Model(inputs, layers_output)

    # Decoder
    x = encoder.output[-1]  # Start from the deepest layer
    for i in reversed(range(len(layers_output) - 1)):
        x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        skip_connection = encoder.output[i]
        x = layers.Concatenate()([x, skip_connection])
        x = layers.Conv2D(256 // (2**i), (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256 // (2**i), (3, 3), activation='relu', padding='same')(x)

    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Instantiate the model
model = unet_model(input_size=(512, 512, 3), num_classes=NUM_CLASSES)


# 7. Define Loss Functions

def dice_loss(y_true, y_pred, smooth=1):
    """
    Computes Dice Loss.

    Parameters:
    - y_true (tf.Tensor): Ground truth one-hot encoded masks.
    - y_pred (tf.Tensor): Predicted masks.
    - smooth (float): Smoothing factor to avoid division by zero.

    Returns:
    - tf.Tensor: Dice loss.
    """
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) + smooth)

def combined_loss(y_true, y_pred):
    """
    Combines Categorical Crossentropy and Dice Loss.

    Parameters:
    - y_true (tf.Tensor): Ground truth one-hot encoded masks.
    - y_pred (tf.Tensor): Predicted masks.

    Returns:
    - tf.Tensor: Combined loss.
    """
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# 8. Compile the Model

model.compile(optimizer='adam',
              loss=combined_loss,
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)])

# Display Model Summary
model.summary()

# 9. Define Callbacks

checkpoint = ModelCheckpoint('best_model.keras', 
                             monitor='val_mean_io_u', 
                             mode='max', 
                             verbose=1, 
                             save_best_only=True)

early_stopping = EarlyStopping(monitor='val_mean_io_u', 
                               mode='max', 
                               patience=10, 
                               verbose=1, 
                               restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_mean_io_u', 
                              mode='max', 
                              factor=0.5, 
                              patience=5, 
                              verbose=1)

callbacks = [checkpoint, early_stopping, reduce_lr]


# 10. Train the Model

# Calculate steps per epoch
steps_per_epoch = len(train_images) // BATCH_SIZE
validation_steps = len(val_images) // BATCH_SIZE

# Fit the model
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks
)


model.summary()



def unet_model(input_size=(512, 512, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(input_size)

    # Use ResNet50 as the backbone
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Specify the layers for skip connections
    layer_names = [
        'conv1_relu',            # 256x256
        'conv2_block3_out',      # 128x128
        'conv3_block4_out',      # 64x64
        'conv4_block6_out',      # 32x32
        'conv5_block3_out',      # 16x16
    ]
    layers_output = [base_model.get_layer(name).output for name in layer_names]

    # Create the encoder model
    encoder = models.Model(inputs, layers_output)

    # Decoder
    x = encoder.output[-1]  # Start from the deepest layer
    for i in reversed(range(len(layers_output) - 1)):
        x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        skip_connection = encoder.output[i]
        x = layers.Concatenate()([x, skip_connection])
        x = layers.Conv2D(256 // (2**i), (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256 // (2**i), (3, 3), activation='relu', padding='same')(x)
        
        # Debugging: Print the shape after each upsampling
        print(f'After upsampling and concatenation, x shape: {x.shape}')

    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model



layers.Conv2D(filters, (3, 3), activation='relu', padding='same')



model = unet_model()
model.summary()



# 3. Data Preprocessing Functions

def decode_rle_to_mask(rle, height, width):
    """
    Decodes a Run-Length Encoded (RLE) mask into a binary mask.

    Parameters:
    - rle (str): RLE string.
    - height (int): Height of the mask.
    - width (int): Width of the mask.

    Returns:
    - np.array: Decoded binary mask of shape (height, width).
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    if pd.isna(rle):
        return mask.reshape((height, width))
    rle = list(map(int, rle.split()))
    for i in range(0, len(rle), 2):
        start, length = rle[i], rle[i + 1]
        start -= 1  # Convert to zero-based index
        mask[start:start + length] = 1
    return mask.reshape((height, width))

def create_multi_class_mask(df, image_name, height, width):
    """
    Creates a single-channel mask with class IDs for a given image.

    Parameters:
    - df (pd.DataFrame): DataFrame containing mask information.
    - image_name (str): Name of the image file.
    - height (int): Original height of the image.
    - width (int): Original width of the image.

    Returns:
    - np.array: Single-channel mask of shape (height, width) with class IDs.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    image_df = df[df['ImageName'] == image_name]
    for _, row in image_df.iterrows():
        class_num = row['ClassNumber']
        rle = row['Encoding']
        single_mask = decode_rle_to_mask(rle, height, width)
        mask[single_mask == 1] = class_num - 1  # Classes 1-4 mapped to 0-3
    return mask

def encode_mask_to_rle(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Parameters:
    - mask (np.array): Binary mask of shape (height, width).

    Returns:
    - str: RLE string.
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def preprocess_test_image(image_path, target_size=IMAGE_SIZE):
    """
    Preprocesses a test image for prediction.

    Parameters:
    - image_path (str): Path to the test image.
    - target_size (tuple): Desired image size.

    Returns:
    - np.array: Preprocessed image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return img

# 4. Load and Split Data

# Load train.csv
train_df = pd.read_csv(TRAIN_CSV)

# Get unique image names
image_names = train_df['ImageName'].unique()

# Split into training and validation sets
train_images, val_images = train_test_split(image_names, test_size=0.2, random_state=42)

print(f'Training samples: {len(train_images)}')
print(f'Validation samples: {len(val_images)}')

# 5. Data Generators

def generate_data(df, image_names, batch_size=BATCH_SIZE):
    """
    Generator that yields batches of images and masks.

    Parameters:
    - df (pd.DataFrame): DataFrame containing mask information.
    - image_names (list): List of image names.
    - batch_size (int): Number of samples per batch.

    Yields:
    - Tuple of (images, masks).
    """
    while True:
        for start in range(0, len(image_names), batch_size):
            end = min(start + batch_size, len(image_names))
            batch_images = image_names[start:end]
            images = []
            masks = []
            for image in batch_images:
                img_path = os.path.join(TRAIN_IMAGES_DIR, f'{image}.jpg')
                img = cv2.imread(img_path)
                if img is None:
                    print(f'Warning: Image {img_path} not found.')
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0  # Normalize

                # Get original image dimensions from DataFrame
                img_df = df[df['ImageName'] == image]
                # Assuming all entries for an image have the same height and width
                orig_height = img_df['ImageHeight'].iloc[0]
                orig_width = img_df['ImageWidth'].iloc[0]

                mask = create_multi_class_mask(df, image, orig_height, orig_width)
                mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
                masks.append(mask)
                images.append(img)

            images = np.array(images)
            masks = np.array(masks)
            masks = to_categorical(masks, num_classes=NUM_CLASSES)  # Shape: (batch_size, 512, 512, 4)
            yield images, masks

# Create training and validation generators
train_gen = generate_data(train_df, train_images, batch_size=BATCH_SIZE)
val_gen = generate_data(train_df, val_images, batch_size=BATCH_SIZE)


# 6. Model Definition

def unet_model(input_size=(512, 512, 3), num_classes=NUM_CLASSES):
    """
    Builds a U-Net model with ResNet50 backbone.

    Parameters:
    - input_size (tuple): Shape of the input images.
    - num_classes (int): Number of segmentation classes.

    Returns:
    - tf.keras.Model: Compiled U-Net model.
    """
    inputs = layers.Input(input_size)

    # Use ResNet50 as the backbone
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Specify the layers for skip connections
    layer_names = [
        'conv1_relu',            # 256x256
        'conv2_block3_out',      # 128x128
        'conv3_block4_out',      # 64x64
        'conv4_block6_out',      # 32x32
        'conv5_block3_out',      # 16x16
    ]
    layers_output = [base_model.get_layer(name).output for name in layer_names]

    # Create the encoder model
    encoder = models.Model(inputs, layers_output)

    # Decoder
    x = encoder.output[-1]  # Start from the deepest layer
    for i in reversed(range(len(layers_output) - 1)):
        x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        skip_connection = encoder.output[i]
        x = layers.Concatenate()([x, skip_connection])
        x = layers.Conv2D(256 // (2**i), (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256 // (2**i), (3, 3), activation='relu', padding='same')(x)
        print(f'After upsampling and concatenation, x shape: {x.shape}')

    # Additional upsampling step to reach 512x512
    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    # No skip connection for this step
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    print(f'After final upsampling, x shape: {x.shape}')

    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Instantiate the model
model = unet_model(input_size=(512, 512, 3), num_classes=NUM_CLASSES)


# 7. Define Loss Functions

def dice_loss(y_true, y_pred, smooth=1):
    """
    Computes Dice Loss.

    Parameters:
    - y_true (tf.Tensor): Ground truth one-hot encoded masks.
    - y_pred (tf.Tensor): Predicted masks.
    - smooth (float): Smoothing factor to avoid division by zero.

    Returns:
    - tf.Tensor: Dice loss.
    """
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) + smooth)

def combined_loss(y_true, y_pred):
    """
    Combines Categorical Crossentropy and Dice Loss.

    Parameters:
    - y_true (tf.Tensor): Ground truth one-hot encoded masks.
    - y_pred (tf.Tensor): Predicted masks.

    Returns:
    - tf.Tensor: Combined loss.
    """
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# 8. Compile the Model

model.compile(optimizer='adam',
              loss=combined_loss,
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)])

# Display Model Summary
model.summary()


print(f'y_true shape: {y_true.shape}')
print(f'y_pred shape: {y_pred.shape}')




# 9. Define Callbacks

checkpoint = ModelCheckpoint('best_model.keras', 
                             monitor='val_mean_io_u', 
                             mode='max', 
                             verbose=1, 
                             save_best_only=True)

early_stopping = EarlyStopping(monitor='val_mean_io_u', 
                               mode='max', 
                               patience=10, 
                               verbose=1, 
                               restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_mean_io_u', 
                              mode='max', 
                              factor=0.5, 
                              patience=5, 
                              verbose=1)

callbacks = [checkpoint, early_stopping, reduce_lr]


# 10. Train the Model

# Calculate steps per epoch
steps_per_epoch = len(train_images) // BATCH_SIZE
validation_steps = len(val_images) // BATCH_SIZE

# Fit the model
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks
)


# Import Necessary Libraries
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import jaccard_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress TensorFlow warnings for clarity
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Define Constants
NUM_CLASSES = 4  # 1: Head, 2: Body, 3: Legs, 4: Tail
IMAGE_SIZE = (512, 512)  # Updated from (500, 500) to (512, 512)
BATCH_SIZE = 4
EPOCHS = 50
TRAIN_IMAGES_DIR = '/kaggle/input/ai-dl-multiclass-segmentation/TrainImages'
TEST_IMAGES_DIR = '/kaggle/input/ai-dl-multiclass-segmentation/TestImages'
TRAIN_CSV = '/kaggle/input/ai-dl-multiclass-segmentation/train.csv'
TEST_CSV = '/kaggle/input/ai-dl-multiclass-segmentation/test_class.csv'
SUBMISSION_FILE = 'submission.csv'


# 3. Data Preprocessing Functions

def decode_rle_to_mask(rle, height, width):
    """
    Decodes a Run-Length Encoded (RLE) mask into a binary mask.

    Parameters:
    - rle (str): RLE string.
    - height (int): Height of the mask.
    - width (int): Width of the mask.

    Returns:
    - np.array: Decoded binary mask of shape (height, width).
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    if pd.isna(rle):
        return mask.reshape((height, width))
    rle = list(map(int, rle.split()))
    for i in range(0, len(rle), 2):
        start, length = rle[i], rle[i + 1]
        start -= 1  # Convert to zero-based index
        mask[start:start + length] = 1
    return mask.reshape((height, width))

def create_multi_class_mask(df, image_name, height, width):
    """
    Creates a single-channel mask with class IDs for a given image.

    Parameters:
    - df (pd.DataFrame): DataFrame containing mask information.
    - image_name (str): Name of the image file.
    - height (int): Original height of the image.
    - width (int): Original width of the image.

    Returns:
    - np.array: Single-channel mask of shape (height, width) with class IDs.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    image_df = df[df['ImageName'] == image_name]
    for _, row in image_df.iterrows():
        class_num = row['ClassNumber']
        rle = row['Encoding']
        single_mask = decode_rle_to_mask(rle, height, width)
        mask[single_mask == 1] = class_num - 1  # Classes 1-4 mapped to 0-3
    return mask

def encode_mask_to_rle(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Parameters:
    - mask (np.array): Binary mask of shape (height, width).

    Returns:
    - str: RLE string.
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def preprocess_test_image(image_path, target_size=IMAGE_SIZE):
    """
    Preprocesses a test image for prediction.

    Parameters:
    - image_path (str): Path to the test image.
    - target_size (tuple): Desired image size.

    Returns:
    - np.array: Preprocessed image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return img

# 4. Load and Split Data

# Load train.csv
train_df = pd.read_csv(TRAIN_CSV)

# Get unique image names
image_names = train_df['ImageName'].unique()

# Split into training and validation sets
train_images, val_images = train_test_split(image_names, test_size=0.2, random_state=42)

print(f'Training samples: {len(train_images)}')
print(f'Validation samples: {len(val_images)}')

# 5. Data Generators

def generate_data(df, image_names, batch_size=BATCH_SIZE):
    """
    Generator that yields batches of images and masks.

    Parameters:
    - df (pd.DataFrame): DataFrame containing mask information.
    - image_names (list): List of image names.
    - batch_size (int): Number of samples per batch.

    Yields:
    - Tuple of (images, masks).
    """
    while True:
        for start in range(0, len(image_names), batch_size):
            end = min(start + batch_size, len(image_names))
            batch_images = image_names[start:end]
            images = []
            masks = []
            for image in batch_images:
                img_path = os.path.join(TRAIN_IMAGES_DIR, f'{image}.jpg')
                img = cv2.imread(img_path)
                if img is None:
                    print(f'Warning: Image {img_path} not found.')
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0  # Normalize

                # Get original image dimensions from DataFrame
                img_df = df[df['ImageName'] == image]
                # Assuming all entries for an image have the same height and width
                orig_height = img_df['ImageHeight'].iloc[0]
                orig_width = img_df['ImageWidth'].iloc[0]

                mask = create_multi_class_mask(df, image, orig_height, orig_width)
                mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
                masks.append(mask)
                images.append(img)

            images = np.array(images)
            masks = np.array(masks)
            masks = to_categorical(masks, num_classes=NUM_CLASSES)  # Shape: (batch_size, 512, 512, 4)
            yield images, masks

# Create training and validation generators
train_gen = generate_data(train_df, train_images, batch_size=BATCH_SIZE)
val_gen = generate_data(train_df, val_images, batch_size=BATCH_SIZE)

# 6. Model Definition

def unet_model(input_size=(512, 512, 3), num_classes=NUM_CLASSES):
    """
    Builds a U-Net model with ResNet50 backbone.

    Parameters:
    - input_size (tuple): Shape of the input images.
    - num_classes (int): Number of segmentation classes.

    Returns:
    - tf.keras.Model: Compiled U-Net model.
    """
    inputs = layers.Input(input_size)

    # Use ResNet50 as the backbone
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Specify the layers for skip connections
    layer_names = [
        'conv1_relu',            # 256x256
        'conv2_block3_out',      # 128x128
        'conv3_block4_out',      # 64x64
        'conv4_block6_out',      # 32x32
        'conv5_block3_out',      # 16x16
    ]
    layers_output = [base_model.get_layer(name).output for name in layer_names]

    # Create the encoder model
    encoder = models.Model(inputs, layers_output)

    # Decoder
    x = encoder.output[-1]  # Start from the deepest layer
    for i in reversed(range(len(layers_output) - 1)):
        x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        skip_connection = encoder.output[i]
        x = layers.Concatenate()([x, skip_connection])
        x = layers.Conv2D(256 // (2**i), (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256 // (2**i), (3, 3), activation='relu', padding='same')(x)
        print(f'After upsampling and concatenation, x shape: {x.shape}')

    # Additional upsampling step to reach 512x512
    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    # No skip connection for this step
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    print(f'After final upsampling, x shape: {x.shape}')

    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Instantiate the model
model = unet_model(input_size=(512, 512, 3), num_classes=NUM_CLASSES)

# 7. Define Loss Functions

def dice_loss(y_true, y_pred, smooth=1):
    """
    Computes Dice Loss.

    Parameters:
    - y_true (tf.Tensor): Ground truth one-hot encoded masks.
    - y_pred (tf.Tensor): Predicted masks.
    - smooth (float): Smoothing factor to avoid division by zero.

    Returns:
    - tf.Tensor: Scalar Dice loss.
    """
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) + smooth)
    dice = tf.reduce_mean(dice)  # Aggregate over classes
    return 1 - dice

def combined_loss(y_true, y_pred):
    """
    Combines Categorical Crossentropy and Dice Loss.

    Parameters:
    - y_true (tf.Tensor): Ground truth one-hot encoded masks.
    - y_pred (tf.Tensor): Predicted masks.

    Returns:
    - tf.Tensor: Scalar combined loss.
    """
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    ce_loss = tf.reduce_mean(ce_loss)  # Aggregate over batch and spatial dimensions
    dl = dice_loss(y_true, y_pred)
    return ce_loss + dl

# 8. Compile the Model

model.compile(optimizer='adam',
              loss=combined_loss,
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)])

# Display Model Summary
model.summary()



# 9. Define Callbacks

checkpoint = ModelCheckpoint('best_model.keras', 
                             monitor='val_mean_io_u', 
                             mode='max', 
                             verbose=1, 
                             save_best_only=True)

early_stopping = EarlyStopping(monitor='val_mean_io_u', 
                               mode='max', 
                               patience=10, 
                               verbose=1, 
                               restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_mean_io_u', 
                              mode='max', 
                              factor=0.5, 
                              patience=5, 
                              verbose=1)

callbacks = [checkpoint, early_stopping, reduce_lr]


# 10. Train the Model

# Calculate steps per epoch
steps_per_epoch = len(train_images) // BATCH_SIZE
validation_steps = len(val_images) // BATCH_SIZE

# Perform a forward pass to verify shapes
images, masks = next(train_gen)
preds = model.predict(images)
print(f'Images shape: {images.shape}')  # Expected: (batch_size, 512, 512, 3)
print(f'Masks shape: {masks.shape}')    # Expected: (batch_size, 512, 512, 4)
print(f'Predictions shape: {preds.shape}')  # Expected: (batch_size, 512, 512, 4)

# Fit the model
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks
)


# 11. Evaluate the Model

# Plot Training History
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot IoU
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_io_u'], label='Train IoU', color='blue')
plt.plot(history.history['val_mean_io_u'], label='Validation IoU', color='orange')
plt.title('IoU Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()

plt.tight_layout()
plt.show()


# Initialize list to store IoUs
val_ious = []

# Reset the validation generator
val_gen = generate_data(train_df, val_images, batch_size=BATCH_SIZE)

# Iterate over the validation set
for _ in tqdm(range(validation_steps), desc='Calculating IoU on Validation Set'):
    X_val, y_val = next(val_gen)
    y_pred = model.predict(X_val)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_val, axis=-1)
    
    # Calculate IoU for each class
    for class_num in range(NUM_CLASSES):
        true_mask = (y_true == class_num).astype(np.uint8)
        pred_mask = (y_pred == class_num).astype(np.uint8)
        if np.sum(true_mask) == 0 and np.sum(pred_mask) == 0:
            iou = 1.0  # Perfect match (both empty)
        elif np.sum(true_mask) == 0 or np.sum(pred_mask) == 0:
            iou = 0.0  # One is empty, the other is not
        else:
            iou = jaccard_score(true_mask.flatten(), pred_mask.flatten(), average='binary')
        val_ious.append(iou)

# Convert list to numpy array
val_ious = np.array(val_ious)

# Calculate mean IoU for each class
mean_ious = []
for i in range(NUM_CLASSES):
    class_ious = val_ious[i::NUM_CLASSES]
    mean_iou = np.mean(class_ious)
    mean_ious.append(mean_iou)
    print(f'IoU for Class {i+1}: {mean_iou:.4f}')


# 12. Inference and Submission

# Load Test Image Names
test_image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith('.jpg')]
test_image_names = [os.path.splitext(f)[0] for f in test_image_files]

# Initialize Submission Dictionary
submission_dict = {}

# Iterate over Test Images
for image in tqdm(test_image_names, desc='Processing Test Images'):
    img_path = os.path.join(TEST_IMAGES_DIR, f'{image}.jpg')
    img = preprocess_test_image(img_path, target_size=IMAGE_SIZE)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Predict
    preds = model.predict(img)
    preds = np.argmax(preds, axis=-1)[0]  # Shape: (512, 512)
    
    # Encode masks for each class
    rles = []
    for class_num in range(NUM_CLASSES):
        class_mask = (preds == class_num).astype(np.uint8)
        if class_mask.sum() > 0:
            rle = encode_mask_to_rle(class_mask)
            rles.append(rle)
    
    # Combine RLEs
    submission_dict[image] = ' '.join(rles) if rles else ''

# Prepare Submission DataFrame
submission = pd.DataFrame({
    'ImageName': [f'{img}.jpg' for img in test_image_names],
    'Encoding': [submission_dict[img] for img in test_image_names]
})

# Save to CSV
submission.to_csv(SUBMISSION_FILE, index=False)
print(f'Submission file saved to {SUBMISSION_FILE}')


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def decode_rle(rle_string, height, width):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.
    
    Parameters:
        rle_string (str): The RLE string (e.g., "3 5 10 2 ...").
        height (int): Height of the mask.
        width (int): Width of the mask.
    
    Returns:
        np.ndarray: Decoded binary mask of shape (height, width).
    """
    if pd.isna(rle_string) or rle_string.strip() == '':
        return np.zeros((height, width), dtype=np.uint8)
    
    s = rle_string.strip().split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1  # Convert to zero-based indexing
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for start, end in zip(starts, ends):
        img[start:end] = 1
    return img.reshape((height, width))

def overlay_mask(image, mask, color, alpha=0.4):
    """
    Overlays a single mask on the image with the specified color and transparency.
    
    Parameters:
        image (np.ndarray): The original image in RGB format.
        mask (np.ndarray): Binary mask to overlay.
        color (tuple): RGB color tuple (e.g., (255, 0, 0) for red).
        alpha (float): Transparency factor.
    
    Returns:
        np.ndarray: Image with the mask overlay.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            (1 - alpha) * image[:, :, c] + alpha * color[c],
            image[:, :, c]
        )
    return image



import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define paths
SUBMISSION_CSV = "/kaggle/working/submission.csv"  # Path to your submission CSV
TEST_IMAGES_DIR = "/kaggle/input/ai-dl-multiclass-segmentation/TestImages"      # Directory containing test images
VISUALIZATION_DIR = "Visualiz/kaggle/input/ai-dl-multiclass-segmentation/TestImagesations"  # Directory to save visualization images

# Create visualization directory if it doesn't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Define class names and their corresponding colors (RGB)
custom_class_names = ["head", "body", "legs", "tail"]
custom_colors = {
    "head": (255, 0, 0),    # Red
    "body": (0, 255, 0),    # Green
    "legs": (0, 0, 255),    # Blue
    "tail": (255, 255, 0)   # Yellow
}

# Read the submission CSV
submission_df = pd.read_csv(SUBMISSION_CSV)

def visualize_masks(image_path, rle_encodings, image_name, save_path=None):
    """
    Visualizes masks on the image based on RLE encodings.
    
    Parameters:
        image_path (str): Path to the image file.
        rle_encodings (str): Space-separated RLE strings.
        image_name (str): Name of the image (for titles).
        save_path (str, optional): Path to save the visualization image.
    """
    # Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Warning: Unable to read image {image_path}. Skipping.")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    height, width = image_rgb.shape[:2]
    
    # Split RLE encodings
    rle_list = rle_encodings.strip().split('  ')  # Assuming double space separates RLEs
    if len(rle_list) == 1:
        rle_list = rle_encodings.strip().split(' ')
        # Reconstruct RLEs assuming each mask has even number of numbers
        if len(rle_list) % 2 != 0:
            print(f"Warning: Odd number of RLE elements in image {image_name}.")
            return
        rle_list = [' '.join(rle_list[i:i+2]) for i in range(0, len(rle_list), 2)]
    
    # Initialize overlay image
    overlay = image_rgb.copy()
    
    # Assign classes and colors
    for i, rle in enumerate(rle_list[:4]):  # Ensure max 4 masks
        class_name = custom_class_names[i] if i < len(custom_class_names) else f"class_{i+1}"
        color = custom_colors.get(class_name, (255, 255, 255))  # Default to white if not found
        
        # Decode RLE
        mask = decode_rle(rle, height, width)
        
        # Overlay mask
        overlay = overlay_mask(overlay, mask, color, alpha=0.4)
    
    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title(f"Image: {image_name}")
    plt.axis('off')
    plt.show()
    
    # Save the visualization if a path is provided
    if save_path:
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, overlay_bgr)
        print(f"Saved visualization to {save_path}")

# Iterate over each row in the submission CSV
for idx, row in submission_df.iterrows():
    image_name = row['ImageName']
    encoding = row['Encoding']
    image_path = os.path.join(TEST_IMAGES_DIR, image_name)
    save_image_path = os.path.join(VISUALIZATION_DIR, f"{os.path.splitext(image_name)[0]}_overlay.jpg")
    
    visualize_masks(image_path, encoding, image_name, save_path=save_image_path)





