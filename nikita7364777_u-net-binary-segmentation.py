! pip install protobuf==4.21.0 --upgrade
! pip install tensorrt==8.6.1
! pip install tensorflow[and-cuda]==2.15.0 --upgrade


import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras.utils import load_img, img_to_array
import tensorflow.keras.layers as tfl
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomContrast
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
print("tensorflow" + tf.__version__)


print(tf.config.list_physical_devices('GPU'))


tf.autograph.set_verbosity(0)
stg = tf.distribute.MirroredStrategy() # For U-net model


###########################################################################################
# ----------------------------------------------------------------------------------------
# Get input zip files
# ----------------------------------------------------------------------------------------
##########################################################################################
def getZippedFilePaths():
    zip_file_names = []
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            if filename.split('.')[-1] == 'zip':
                zip_file_names.append((os.path.join(dirname, filename)))
    return zip_file_names

###########################################################################################
# ----------------------------------------------------------------------------------------
# Preprocess images and mask
# ----------------------------------------------------------------------------------------
##########################################################################################
# file_path = train_hq.zip
def preprocess_image(file_path):
    # Load and decode the image
    img = tf.io.read_file(file_path)
    # You can adjust channels based on your images (3 for RGB)
    img = tf.image.decode_jpeg(img, channels = 3) # Returned as uint8
    # Normalize the pixel values to [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize the image to your desired dimensions
    img = tf.image.resize(img, [96, 128], method = 'nearest')
    return img

# file_path = train_masks.zip
def preprocess_target(file_path):
    # Load and decode the image
    mask = tf.io.read_file(file_path)
    # Normalizing to between 0 and 1 (only two classes)
    mask = tf.image.decode_image(mask, expand_animations = False, dtype = tf.float32)
    # Get only one value for the 3rd channel
    mask = tf.math.reduce_max(mask, axis = -1, keepdims = True)
    # Resize the image to your desired dimensions
    mask = tf.image.resize(mask, [96, 128], method = 'nearest')
    return mask

###########################################################################################
# ----------------------------------------------------------------------------------------
# Loss function
# ----------------------------------------------------------------------------------------
##########################################################################################
def dice_coef(y_true, y_pred, smooth=10e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
###########################################################################################
# ----------------------------------------------------------------------------------------
# Display pred results
# ----------------------------------------------------------------------------------------
##########################################################################################
def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# Converting probabilities from *.predict(dataset) to the class index
def create_mask(pred_mask):
    mask = pred_mask[..., -1] >= 0.5
    pred_mask[..., -1] = tf.where(mask, 1, 0)
    # Return only first mask of batch
    return pred_mask[0]

# Predict images visualization
def show_predictions(model, dataset = None, num = 1):
    # Displays the first image of each of the num batches
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])


zip_file_names = getZippedFilePaths()

items_to_remove = ['/kaggle/input/carvana-image-masking-challenge/train.zip', 
                   '/kaggle/input/carvana-image-masking-challenge/test.zip']
     
zip_file_names = [item for item in zip_file_names if item not in items_to_remove]

for zip_file_path in zip_file_names:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall()


zip_file_names


def analyze_mask_values(mask_files, sample_size = 5, title = "Mask Analysis"):
    # Analyzes the values in masks before processing.
        # mask_files: list of mask file paths
        # sample_size: number of masks to analyze
        # title: output header

    print(f"\n{'='*60}")
    print(f"{title} (analysis {min(sample_size, len(mask_files))} mask)")
    print('='*60)
    
    for i, mask_path in enumerate(mask_files[:sample_size]):
        try:
            # Reading and decoding the source file
            mask_bytes = tf.io.read_file(mask_path)
            mask = tf.image.decode_image(mask_bytes, expand_animations=False, dtype=tf.uint8)
            
            print(f"\Mask {i+1}: {os.path.basename(mask_path)}")
            print(f"  Form: {mask.shape}")
            print(f"  Data type: {mask.dtype}")
            
            # Analyzing unique values
            if len(mask.shape) == 3:
                # If the image is multi-channel
                if mask.shape[2] == 1:
                    values = mask.numpy().flatten()
                else:
                    # For RGB, we look at individual channels
                    values_r = mask.numpy()[..., 0].flatten()
                    values_g = mask.numpy()[..., 1].flatten()
                    values_b = mask.numpy()[..., 2].flatten()
                    
                    print(f"  Unique values in the R channel: {np.unique(values_r)}")
                    print(f"  Unique values in the G channel: {np.unique(values_g)}")
                    print(f"  Unique values in the B channel: {np.unique(values_b)}")
                    
                    # We also look at unique RGB combinations.
                    reshaped = mask.numpy().reshape(-1, mask.shape[2])
                    unique_combinations = np.unique(reshaped, axis=0)
                    print(f"  Unique RGB combinations ({len(unique_combinations)}):")
                    for combo in unique_combinations[:10]:  # Showing the first 10
                        print(f"    {combo}")
                    if len(unique_combinations) > 10:
                        print(f"    ... and more {len(unique_combinations) - 10}")
                    values = np.concatenate([values_r, values_g, values_b])
            else:
                values = mask.numpy().flatten()
            
            unique_values = np.unique(values)
            print(f"  All unique values ({len(unique_values)}): {unique_values}")
            print(f"  Range of values: [{values.min()}, {values.max()}]")
            
            # Histogram of values
            plt.figure(figsize=(10, 3))
            plt.subplot(1, 2, 1)
            plt.hist(values, bins = 50, alpha = 0.7, color = 'blue', edgecolor = 'black')
            plt.title(f'Distribution of values (mask {i+1})')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            
            # Mask Visualization
            plt.subplot(1, 2, 2)
            if len(mask.shape) == 3 and mask.shape[2] == 3:
                plt.imshow(mask.numpy())
            else:
                plt.imshow(mask.numpy(), cmap='gray', vmin=0, vmax=255)
            plt.title(f'Mask Visualization {i+1}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error during analysis {mask_path}: {e}")

def analyze_processed_masks(dataset, sample_size = 5, title = "Analysis of processed masks"):
    # Analyzes the values in masks after processing.
        # dataset: tf.data.Dataset with processed masks
        # sample_size: number of masks to analyze
        # title: output header
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    
    # We take the first sample_size elements from the dataset
    iterator = iter(dataset)
    for i in range(min(sample_size, 5)):
        try:
            mask = next(iterator)
            
            print(f"\nProcessed mask {i+1}:")
            print(f"  Form: {mask.shape}")
            print(f"  Data type: {mask.dtype}")
            
            # Analysis of unique values
            values = mask.numpy().flatten()
            unique_values = np.unique(values)
            
            print(f"  Unique values ({len(unique_values)}): {unique_values}")
            print(f"  Range of values: [{values.min()}, {values.max()}]")
            
            # Counting the number of each value
            value_counts = {}
            for val in values:
                value_counts[val] = value_counts.get(val, 0) + 1
            
            print("  Distributing:")
            for val in sorted(unique_values):
                count = value_counts.get(val, 0)
                percentage = (count / len(values)) * 100
                print(f"    {val}: {count} pixel values ({percentage:.2f}%)")
            
            # Visualization
            plt.figure(figsize=(10, 3))
            
            # The histogram
            plt.subplot(1, 2, 1)
            bins = len(unique_values) if len(unique_values) <= 50 else 50
            plt.hist(values, bins = bins, alpha=0.7, color='green', edgecolor='black')
            plt.title(f'Distribution of values (processed {i+1})')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            
            # Image of the mask
            plt.subplot(1, 2, 2)
            if len(mask.shape) == 3 and mask.shape[2] > 1:
                # If multi-channel
                plt.imshow(mask.numpy())
            else:
                # If it is single-channel
                plt.imshow(mask.numpy(), cmap='viridis')
                plt.colorbar()
            plt.title(f'Visualization of the processed mask {i+1}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except StopIteration:
            break
        except Exception as e:
            print(f"Error when analyzing the processed mask {i+1}: {e}")


# Appending all path names to a sorted list
train_hq_dir    = '/kaggle/working/train_hq/'
train_masks_dir = '/kaggle/working/train_masks/'
test_hq_dir     = '/kaggle/working/test_hq/'

X_train_id = sorted([os.path.join(train_hq_dir, filename)    for filename in os.listdir(train_hq_dir)],    key = lambda x: x.split('/')[-1].split('.')[0])
y_train    = sorted([os.path.join(train_masks_dir, filename) for filename in os.listdir(train_masks_dir)], key = lambda x: x.split('/')[-1].split('.')[0])
X_test_id  = sorted([os.path.join(test_hq_dir, filename)     for filename in os.listdir(test_hq_dir)],     key = lambda x: x.split('/')[-1].split('.')[0])

X_train_id = X_train_id[:1000]
y_train    = y_train[:1000]

#----------------------------------------------------------
# 1.) ANALYSIS BEFORE PROCESSING
#----------------------------------------------------------
print("="*80)
print("ANALYSIS OF THE INITIAL MASKS BEFORE PROCESSING")
print("="*80)
analyze_mask_values(y_train, sample_size = 5)
#----------------------------------------------------------
#----------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(X_train_id, y_train, test_size = 0.2, random_state = 42)

# Create Dataset objects from the list of file paths
X_train_ds = tf.data.Dataset.from_tensor_slices(X_train)
y_train_ds = tf.data.Dataset.from_tensor_slices(y_train)

X_val_ds = tf.data.Dataset.from_tensor_slices(X_val)
y_val_ds = tf.data.Dataset.from_tensor_slices(y_val)

X_test_ds = tf.data.Dataset.from_tensor_slices(X_test_id)

img_height = 96
img_width = 128
num_channels = 3
img_size = (img_height, img_width)

# Apply preprocessing
X_train_processed = X_train_ds.map(preprocess_image)
y_train_processed = y_train_ds.map(preprocess_target)

X_val_processed = X_val_ds.map(preprocess_image)
y_val_processed  = y_val_ds.map(preprocess_target)

X_test_processed = X_test_ds.map(preprocess_image)

#----------------------------------------------------------
# 2.) POST-TREATMENT ANALYSIS
#----------------------------------------------------------
print("\n\n" + "="*80)
print("POST-TREATMENT MASK ANALYSIS")
print("="*80)
analyze_processed_masks(y_train_processed, sample_size = 5, title = "Analysis of processed masks")
#----------------------------------------------------------
#----------------------------------------------------------

#----------------------------------------------------------
# 3.) Analysis of all processed masks in the dataset
#----------------------------------------------------------
print("\n\n" + "="*80)
print("ANALYSIS OF ALL PROCESSED MASKS IN THE DATASET")
print("="*80)

# We collect statistics on all processed masks
all_values = []
for mask in y_train_processed.take(100):  # We take 100 for analysis
    all_values.extend(mask.numpy().flatten())

all_values = np.array(all_values)
print(f"\nStatistics on 100 processed masks:")
print(f"  Total pixels: {len(all_values):,}")
print(f"  Unique values: {np.unique(all_values)}")
print(f"  Range: [{all_values.min()}, {all_values.max()}]")
print(f"  The average value: {all_values.mean():.6f}")
print(f"  Standard deviation: {all_values.std():.6f}")

# Distribution of values
value_counts = {}
for val in all_values:
    val_key = f"{val:.6f}"  # To avoid rounding errors with float
    value_counts[val_key] = value_counts.get(val_key, 0) + 1

print(f"\nDistribution:")
for val_str in sorted(value_counts.keys(), key=lambda x: float(x)):
    count = value_counts[val_str]
    percentage = (count / len(all_values)) * 100
    print(f"  {val_str}: {count:,} of pixel values ({percentage:.2f}%)")

# Visualization of the overall distribution
plt.figure(figsize=(10, 5))
plt.hist(all_values, bins=100, alpha=0.7, color='purple', edgecolor='black')
plt.title('The overall distribution of pixel values in the processed masks')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency (logarithmic)')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()
#----------------------------------------------------------
#----------------------------------------------------------

# Adding labels to datasets
train_dataset = tf.data.Dataset.zip((X_train_processed, y_train_processed))
val_dataset = tf.data.Dataset.zip((X_val_processed, y_val_processed))

print("\n\n" + "="*80)
print("DATASETS ARE READY")
print("="*80)
print(f"The size of the training dataset: {len(X_train)}")
print(f"The size of the validation dataset: {len(X_val)}")
print(f"The size of the test dataset: {len(X_test_id)}")

BATCH_SIZE = 32
batched_train_dataset = train_dataset.batch(BATCH_SIZE)
batched_val_dataset   = val_dataset.batch(BATCH_SIZE)
batched_test_dataset  = X_test_processed.batch(BATCH_SIZE)

# Adding autotune for pre-fetching
AUTOTUNE = tf.data.experimental.AUTOTUNE
batched_train_dataset = batched_train_dataset.prefetch(buffer_size = AUTOTUNE)
batched_val_dataset   = batched_val_dataset.prefetch(buffer_size = AUTOTUNE)
batched_test_dataset  = batched_test_dataset.prefetch(buffer_size = AUTOTUNE)


# Control check the count of files:
print("The size of the training/validation dataset", len(os.listdir("/kaggle/working/train_hq")))
print("The size of the testing dataset", len(os.listdir("/kaggle/working/test_hq")))


plt.figure(figsize=(40, 30))

for images, masks in batched_val_dataset.take(1):
    car_number = 0
    for image_slot in range(16):
        ax = plt.subplot(4, 4, image_slot + 1)
        
        if image_slot % 2 == 0:
            plt.imshow((images[car_number]))
            plt.title(f'Image {car_number + 1}', fontsize=22, pad=20)
        else:
            # Mask
            mask_display = masks[car_number]
            if len(mask_display.shape) == 3 and mask_display.shape[2] == 1:
                mask_display = mask_display[:, :, 0]
            
            plt.imshow(mask_display, cmap='gray')
            plt.title(f'Mask {car_number + 1}', fontsize=22, pad=20)
            car_number += 1
        
        plt.axis('off')

plt.suptitle('Dataset for binary segmentation', fontsize=28, y=1.02)
plt.tight_layout()
plt.show()


data_augmentation = tf.keras.Sequential([tfl.RandomFlip(mode="horizontal", seed=42),
                                         tfl.RandomRotation(factor=0.01, seed=42),
                                         tfl.RandomContrast(factor=0.2, seed=42)
                                        ])


def get_model(img_size):
    inputs = Input(shape=img_size + (3,))
    x = data_augmentation(inputs)
    
    # Contracting path
    #-------------------------------------------------------------------------------------------
    x = tfl.Conv2D(64, 3, strides = 2, activation = "relu", 
                                       padding = "same", kernel_initializer = 'he_normal')(x)
    #-------------------------------------------------------------------------------------------
    x = tfl.Conv2D(64, 3,              activation = "relu", 
                                       padding="same", kernel_initializer='he_normal')(x)
    #-------------------------------------------------------------------------------------------
    x = tfl.Conv2D(128,3, strides = 2, activation = "relu", 
                                       padding="same", kernel_initializer='he_normal')(x)
    #-------------------------------------------------------------------------------------------
    x = tfl.Conv2D(128,3,              activation = "relu", 
                                       padding="same", kernel_initializer='he_normal')(x)
    #-------------------------------------------------------------------------------------------
    x = tfl.Conv2D(256,3, strides = 2, activation = "relu", 
                                       padding="same", kernel_initializer='he_normal')(x)
    #-------------------------------------------------------------------------------------------
    x = tfl.Conv2D(256,3,              activation = "relu", 
                                       padding="same", kernel_initializer='he_normal')(x)
    #-------------------------------------------------------------------------------------------
    
    # Expanding path
    #-------------------------------------------------------------------------------------------
    x = tfl.Conv2DTranspose(256, 3, activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = tfl.Conv2DTranspose(256, 3, activation="relu", padding="same", kernel_initializer='he_normal', strides = 2)(x)
    x = tfl.Conv2DTranspose(128, 3, activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = tfl.Conv2DTranspose(128, 3, activation="relu", padding="same", kernel_initializer='he_normal', strides = 2)(x)
    x = tfl.Conv2DTranspose(64,  3, activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = tfl.Conv2DTranspose(64,  3, activation="relu", padding="same", kernel_initializer='he_normal', strides = 2)(x)
    outputs = tfl.Conv2D(1, 3, activation = "sigmoid", padding = "same")(x)
    model = keras.Model(inputs, outputs) 
    return model


custom_model = get_model(img_size=img_size) 
custom_model.summary()


with stg.scope():
    custom_model = get_model(img_size=img_size) 
    custom_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001,
                                                            epsilon       = 1e-06), 
                                                            loss          = [dice_loss], 
                                                            metrics       = [dice_coef])

    callbacks_list = [keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    patience=5,
                                                   ),
                      keras.callbacks.ModelCheckpoint(filepath="best-custom-model",
                                                      monitor="val_loss",
                                                      save_best_only=True,
                                                     )
                     ]

history = custom_model.fit(batched_train_dataset,
                           validation_data = batched_val_dataset,
                           epochs          = 50,
                           callbacks       = callbacks_list,
                          )


custom_model = keras.models.load_model("/kaggle/working/best-custom-model", 
                                       custom_objects = {'dice_coef': dice_coef, 'dice_loss': dice_loss})

show_predictions(model = custom_model, dataset = batched_train_dataset, num = 6)


def contracting_block(inputs = None, n_filters = 64, dropout_prob = 0, max_pooling = True):
    conv = Conv2D(n_filters, 3,activation = 'relu',padding = 'same',kernel_initializer = 'he_normal')(inputs)
    conv = Conv2D(n_filters, 3,activation = 'relu',padding = 'same',kernel_initializer = 'he_normal')(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection

def expanding_block(expansive_input, contractive_input, n_filters = 64):
    up = Conv2DTranspose(n_filters, 3, strides=(2, 2), padding='same', kernel_initializer='he_normal')(expansive_input)
    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis = 3)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    return conv

def Unet_model(input_size=(96, 128, 3), n_filters=64, n_classes=1):
    inputs = Input(input_size)
    inputs = data_augmentation(inputs)

    # Contracting Path (encoding)
    cblock1 = contracting_block(inputs,     n_filters)
    cblock2 = contracting_block(cblock1[0], n_filters*2)
    cblock3 = contracting_block(cblock2[0], n_filters*4)
    cblock4 = contracting_block(cblock3[0], n_filters*8, dropout_prob = 0.3)

    # Bottleneck Layer
    cblock5 = contracting_block(cblock4[0], n_filters*16, dropout_prob = 0.3, max_pooling = False)
    
    # Expanding Path (decoding)
    ublock6 = expanding_block(cblock5[0], cblock4[1],  n_filters*8)
    ublock7 = expanding_block(ublock6,    cblock3[1],  n_filters*4)
    ublock8 = expanding_block(ublock7,    cblock2[1],  n_filters*2)
    ublock9 = expanding_block(ublock8,    cblock1[1],  n_filters)

    conv9  = Conv2D(n_filters, 3, activation = 'relu',    padding = 'same', kernel_initializer='he_normal')(ublock9)
    conv10 = Conv2D(n_classes, 1, activation = "sigmoid", padding = 'same')(conv9)

    model = tf.keras.Model(inputs = inputs, outputs = conv10)

    return model


with stg.scope():
    # Compile the model with the same loss and metric, that we used in custom model
    unet = Unet_model(input_size = (img_height, img_width, num_channels), 
                      n_filters = 64, 
                      n_classes = 1
                     )
    unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001, 
                                                    epsilon = 1e-06
                                                   ),
                 loss    = [dice_loss], 
                 metrics = [dice_coef]
                )

unet.summary()


callbacks_list = [keras.callbacks.EarlyStopping(monitor  = "val_loss",
                                                patience = 5,
                                               ),
                  keras.callbacks.ModelCheckpoint(filepath       = "best-u_net-model.weights.h5",
                                                  monitor        = "val_loss",
                                                  save_best_only = True,
                                                 )
                 ]

history = unet.fit(batched_train_dataset,
                   validation_data = batched_val_dataset,
                   epochs = 50,
                   callbacks = callbacks_list,
                  )


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label = 'Training dice loss')
plt.plot(epochs, val_loss, label = 'Validation dice loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


unet = keras.models.load_model("/kaggle/working/best-u_net-model", 
                               custom_objects={'dice_coef': dice_coef, 'dice_loss': dice_loss}
                              )
show_predictions(model = unet, dataset = batched_train_dataset, num = 6)

