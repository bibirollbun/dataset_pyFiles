import tensorflow as tf
import zipfile
import numpy as np
import random
import os
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


SEED = 42
np.random.seed = SEED

UNZIP_PATH = '../input/data-science-bowl-2018/'
TRAIN_PATH = './train/'
TEST_PATH = './test/'

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


# Unzip data
with zipfile.ZipFile(UNZIP_PATH + 'stage1_train.zip', 'r') as zip_ref:
    zip_ref.extractall('./train')
    
with zipfile.ZipFile(UNZIP_PATH + 'stage1_test.zip', 'r') as zip_ref:
    zip_ref.extractall('./test')


# get list of all subfolders
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
print(train_ids[:10])
print(test_ids[:10])



# define placeholders (also used to replace NaN in images to resize by 0)
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)


# Loop through all train_ids
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):  
    
    # Get the path to the image and mask directories
    path = TRAIN_PATH + id_  
    
    # Read the image and keep the first IMG_CHANNELS channels
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]  
    
    # Resize the image to IMG_HEIGHT x IMG_WIDTH while keeping the pixel values (mode='constant')
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)  
    
    # Store the processed image into the X_train array at index n
    X_train[n] = img 
    
    # Initialize an empty mask of dimensions IMG_HEIGHT x IMG_WIDTH x 1 with boolean values
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)  
    
    # Loop through each file in the masks directory
    for mask_file in next(os.walk(path + '/masks/'))[2]:  
        
        # Read the mask file
        mask_ = imread(path + '/masks/' + mask_file)
        
        # Resize the mask to IMG_HEIGHT x IMG_WIDTH while keeping the pixel values (mode='constant'), and add an extra dimension
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        
        # Merge all the masks associated with the same image by taking the maximum value at each pixel location
        mask = np.maximum(mask, mask_)
    
    # Store the processed mask into the Y_train array at index n
    Y_train[n] = mask


# Create an empty array for test images with dimensions as (number of test images, height, width, channels)
# The datatype for the array is uint8 which can hold values from 0 to 255
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

# Initialize an empty list to store the original sizes of the test images
sizes_test = []

# Loop through all the test_ids
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    
    # Get the path to the test image
    path = TEST_PATH + id_
    
    # Read the image and keep the first IMG_CHANNELS channels
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    
    # Append the original size of the image to sizes_test list
    sizes_test.append([img.shape[0], img.shape[1]])
    
    # Resize the image to IMG_HEIGHT x IMG_WIDTH while keeping the pixel values
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    
    # Store the processed image into the X_test array at index n
    X_test[n] = img


# Create a function for a convolution block
def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation="relu", 
                               kernel_initializer="he_normal", padding="same")(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation="relu", 
                               kernel_initializer="he_normal", padding="same")(x)
    return x

# Create a function for the expanding path
def upsample_block(inputs, conv_prev, num_filters):
    up = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(inputs)
    concat = tf.keras.layers.concatenate([up, conv_prev])
    conv = conv_block(concat, num_filters)
    return conv


# Inputs
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

# Normalization
s = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs) 

# Contraction path
c1 = conv_block(s, 16) #(128, 128, 16)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1) #(64, 64, 16)

c2 = conv_block(p1, 32) # (64, 64, 32)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2) # (32, 32, 32)

c3 = conv_block(p2, 64) # (32, 32, 64)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3) # (16, 16, 64)

c4 = conv_block(p3, 128) # (16, 16, 128)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4) # (8, 8, 128)

c5 = conv_block(p4, 256) # (8, 8, 256) - Bottleneck

# Expansive path
c6 = upsample_block(c5, c4, 128) # (16, 16, 128) + skip connection from c4
c7 = upsample_block(c6, c3, 64)  # (32, 32, 64) + skip connection from c3
c8 = upsample_block(c7, c2, 32)  # (64, 64, 32) + skip connection from c2
c9 = upsample_block(c8, c1, 16)  # (128, 128, 16) + skip connection from c1

# Output layer
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)


# Model
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

# Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary
model.summary()


# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', verbose = 1, save_best_only = True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience = 5, monitor = 'val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir = 'logs')
            ]


results = model.fit(
    X_train, 
    Y_train, 
    validation_split = 0.1, 
    batch_size = 16, 
    epochs = 25, 
    callbacks = callbacks)


# Predictions
idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose = 1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose = 1)
preds_test = model.predict(X_test, verbose = 1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


# Function to display images
def display_images(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Select a random index from training set
ix = random.randint(0, len(preds_train_t))
print(len(preds_train_t))

for i in range(0, 10):
    ix = random.randint(0, len(preds_val_t))
    
    # Note: validation data starts at 90% of training data
    train_ix = int(X_train.shape[0]*0.9) + ix
    
    print("Input Image:")
    display_images(X_train[train_ix])
    
    print("Ground Truth Mask:")
    display_images(np.squeeze(Y_train[train_ix]))
    
    print("Predicted Mask:")
    display_images(np.squeeze(preds_val_t[ix]))




# Select a random index from validation set
ix = random.randint(0, len(preds_val_t))

# Calculate the index from where validation data starts
val_data_start = int(X_train.shape[0] * 0.9)

# Display the validation input image
print("Validation Input Image:")
display_images(X_train[val_data_start:][ix])

# Display the validation ground truth mask
print("Validation Ground Truth Mask:")
display_images(np.squeeze(Y_train[val_data_start:][ix]))

# Display the validation predicted mask
print("Validation Predicted Mask:")
display_images(np.squeeze(preds_val_t[ix]))

