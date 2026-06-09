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
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import shutil
from pathlib import Path 
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras import regularizers
import random
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, img_to_array


df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
df.head()


df.info()


df.shape


df.describe()


df.isna().sum()


df.isnull().sum()


sns.countplot(df, x="label", order=df['label'].value_counts().index)
# Add a title
plt.title('Label Histogram')
proportions = list(df['label'].value_counts(normalize=True))
print(proportions)
# Display the chart
plt.show()


proportions = list(df['label'].value_counts(normalize=True))
print(proportions)


plt.pie(proportions,labels = df['label'].unique(),
        autopct='%1.1f%%',  # Display percentages on slices
        shadow=True,
        startangle=140)

# circular pie chart
plt.axis('equal')
plt.title('Label Distribution')
plt.show()


dfs = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/sample_submission.csv')
dfs.shape


image_directory = '/kaggle/input/histopathologic-cancer-detection/train/'
    
# list of image file paths
image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png', '.jpeg', '.tif'))]


len(image_files)   # 220025


from collections import Counter

def find_duplicates_with_counter(input_list):
    counts = Counter(input_list)
    duplicates = [item for item, count in counts.items() if count > 1]
    return duplicates

duplicate_values = find_duplicates_with_counter(image_files)
print(f"Duplicate values: {duplicate_values}")


image_directory_test = '/kaggle/input/histopathologic-cancer-detection/test/'
    
# list of image file paths
image_files_test = [os.path.join(image_directory_test, f) for f in os.listdir(image_directory_test) if f.endswith(('.jpg', '.png', '.jpeg', '.tif'))]


len(image_files_test)   


duplicate_values = find_duplicates_with_counter(image_files_test)
print(f"Duplicate values: {duplicate_values}")


# Seperate dataframes for label 0 and label 1
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]

df_0_ids = df_0['id'].head(3).tolist()
df_1_ids = df_1['id'].head(3).tolist()

s = '.tif'
path = '/kaggle/input/histopathologic-cancer-detection/train/'

def add_tif(id_list):
    for i in range(len(id_list)):
        id_list[i] = str(path + id_list[i] + s)
    return id_list
    
df_0_ids_tif = add_tif(df_0_ids)
df_1_ids_tif = add_tif(df_1_ids)


fig, axs = plt.subplots(2, 3, figsize=(7, 5))

# Flatten the `axs` array for easier iteration
axs = axs.flatten()

# Combine both image lists into one 
all_image_paths = df_0_ids_tif + df_1_ids_tif
titles = [f'Label 0 ' for i in range(3)] + [f'Label 1 ' for i in range(3)]

# Display each image in its own subplot
for i, img_path in enumerate(all_image_paths):
    img = Image.open(img_path)
    axs[i].imshow(img)
    axs[i].set_title(titles[i])
    axs[i].axis('off')

plt.tight_layout()
plt.show()


#df_0.shape # (130908, 2)
#df_1.shape # (89117, 2)
#len(df_0_ids_tif) #3
def add_tif1(id_list):
    for i in range(len(id_list)):
        id_list[i] = str(id_list[i] + s)
    return id_list
    
df_0_ids_tif = add_tif(df_0_ids)
df_1_ids_tif = add_tif(df_1_ids)

df_0_ids_20k = df_0['id'].sample(n=40000, random_state=42).tolist()
df_1_ids_20k = df_1['id'].sample(n=40000, random_state=42).tolist()
    
df_0_ids_20k_tif = add_tif1(df_0_ids_20k)
df_1_ids_20k_tif = add_tif1(df_1_ids_20k)


import PIL
PIL.Image.open("/kaggle/input/histopathologic-cancer-detection/test/f5be692779144a3ecdaed9b82f9487564edcbccb.tif").size


# Compress Image to array for easy computation
img = PIL.Image.open('/kaggle/input/histopathologic-cancer-detection/test/f5be692779144a3ecdaed9b82f9487564edcbccb.tif').resize((96, 96))

rgb_pixels = np.array(img)
rgb_pixels.shape


# Find the mode of given image : RGB or RGBA
print(img.mode)


grayscale_image = img.convert("L")
grayscale_image


resized_img = img.resize((200, 200))
resized_img


rotated_img = resized_img.rotate(90)
rotated_img


plt.imshow(rgb_pixels)


fig, axs = plt.subplots(1, 3, figsize=(9, 5))

# Flatten the `axs` array 
axs = axs.flatten()
title = ['red channel', 'green channel', 'blue channel']

# Display each image in its subplot
for i in range(0, 3):
    axs[i].imshow(rgb_pixels[:, :, i], cmap='Greys')
    axs[i].set_title(title[i])
    axs[i].axis('off')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5)) 

# Plot the full image 
axes[0].imshow(rgb_pixels)
axes[0].set_title("Full image")

# Plot the left half 
axes[1].imshow(rgb_pixels[0:96, 0:47])
axes[1].set_title("Left half")

# Plot the right half
axes[2].imshow(rgb_pixels[0:96, 47:96])
axes[2].set_title("Right half")

fig.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# getting a sample image
sample_image = mpimg.imread("/kaggle/input/histopathologic-cancer-detection/test/f5be692779144a3ecdaed9b82f9487564edcbccb.tif")

colors = ('red', 'green', 'blue')
for i, color in enumerate(colors):
    plt.hist(sample_image[:, :, i].ravel(), bins=256, color=color, alpha=0.5)
plt.title('Pixel Distribution (RGB Histograms)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()


from PIL import Image, ImageStat
stat = ImageStat.Stat(img)

print(f"Mean pixel value: {stat.mean[0]:.2f}")
print(f"Median pixel value: {stat.median[0]}")
print(f"Min/Max pixel values: {stat.extrema[0]}")
print(f"Standard deviation: {stat.stddev[0]:.2f}")


if sample_image.ndim == 3 :
    red_channel = sample_image[:, :, 0].flatten()
    green_channel = sample_image[:, :, 1].flatten()
    blue_channel = sample_image[:, :, 2].flatten()
    
    data = [red_channel, green_channel, blue_channel]
    labels = ['Red', 'Green', 'Blue']
    title = 'Pixel Distribution by Color Channel'

plt.figure(figsize=(10, 6))
plt.boxplot(data, vert=True, patch_artist=True, labels=labels)
plt.title(title)
plt.ylabel('Pixel Value')
plt.show()


# Define the names of the directories to be created
directory1_name = "0"
directory2_name = "1"

# Construct the full paths for the new directories
path_dir1 = os.path.join("/kaggle/working/", directory1_name)
path_dir2 = os.path.join("/kaggle/working/", directory2_name)

# Create the first directory 
if not os.path.exists(path_dir1):
    os.mkdir(path_dir1)
    print(f"Directory '{directory1_name}' created at {path_dir1}")
else:
    print(f"Directory '{directory1_name}' already exists at {path_dir1}")

# Create the second directory 
if not os.path.exists(path_dir2):
    os.mkdir(path_dir2)
    print(f"Directory '{directory2_name}' created at {path_dir2}")
else:
    print(f"Directory '{directory2_name}' already exists at {path_dir2}")


def transfer_images(source_dir: str, destination_dir: str, image_list: list[str]):
    """
    Copies a list of images from a source directory to a destination directory.

    Args:
        source_dir (str): The path to the source directory.
        destination_dir (str): The path to the destination directory.
        image_list (list[str]): A list of filenames (strings) to be transferred.
    """
    
    # Use Path objects for clean and reliable path joining
    source = Path(source_dir)
    destination = Path(destination_dir)

    # Ensure the destination directory exists; create it if necessary
    destination.mkdir(parents=True, exist_ok=True)
    
    for image_name in image_list:
        source_path = source / image_name
        destination_path = destination / image_name

        if source_path.is_file():
            # shutil.copy2 copies file data and metadata (timestamps, etc.)
            shutil.copy2(source_path, destination_path)

    print("Files copied")


# 1. Define paths and list of files
SOURCE_DIR = "/kaggle/input/histopathologic-cancer-detection/train/"
DEST_DIR = "/kaggle/working/0/"
IMAGES_TO_COPY = df_0_ids_20k_tif

transfer_images(SOURCE_DIR, DEST_DIR, IMAGES_TO_COPY)


DEST_DIR = "/kaggle/working/1/"
IMAGES_TO_COPY = df_1_ids_20k_tif

transfer_images(SOURCE_DIR, DEST_DIR, IMAGES_TO_COPY)


def convert_tif_to_bmp(tif_path, bmp_path):
    """
    Converts a single TIF file to a BMP file losslessly.
    """ 
    # Open the TIF image
    with Image.open(tif_path) as im:
        # Save the image as a BMP file. BMP is lossless by nature.
        im.save(bmp_path, 'BMP')


image_directory_working0 = '/kaggle/working/0/'
image_files_working0 = [f for f in os.listdir(image_directory_working0) if f.endswith(('.tif'))]
source_tif_dir = image_directory_working0
destination_bmp_dir = image_directory_working0 # Destination is same directory

for filename in image_files_working0:
    source_path = os.path.join(source_tif_dir, filename)
    
    # 1. Change the file extension from .tif to .bmp for the destination filename
    base, ext = os.path.splitext(filename)
    destination_filename = base + '.bmp'

    # 2. destination file path
    destination_path = os.path.join(destination_bmp_dir, destination_filename)
    
    # Call  conversion function 
    convert_tif_to_bmp(source_path, destination_path)


# --- Loop 1 ---
image_directory_working1 = '/kaggle/working/1/'
image_files_working1 = [f for f in os.listdir(image_directory_working1) if f.endswith(('.tif'))]
source_tif_dir_1 = image_directory_working1
destination_bmp_dir_1 = image_directory_working1

for filename in image_files_working1:
    source_path = os.path.join(source_tif_dir_1, filename)
    
    # 1. Change the file extension from .tif to .bmp for the destination filename
    base, ext = os.path.splitext(filename)
    destination_filename = base + '.bmp'

    # 2. destination file path
    destination_path = os.path.join(destination_bmp_dir_1, destination_filename)
    
    # Call  conversion function 
    convert_tif_to_bmp(source_path, destination_path)





# This function loads images, and gets labels from the directory structure
train_ds = tf.keras.utils.image_dataset_from_directory(
    '/kaggle/working/',
    labels='inferred',  
    label_mode='binary', 
    image_size=(96, 96),  # The image size for this dataset
    batch_size=32,
    validation_split=0.3,
    subset='training',
    shuffle=True,seed = 42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    '/kaggle/working/',
    labels='inferred',
    label_mode='binary',
    image_size=(96, 96),
    batch_size=32,
    validation_split=0.3,
    subset='validation', # This is the validation set
    shuffle=True,seed = 42
)


input_shape = (96, 96, 3)

# Create a Sequential model with an Input layer
model = Sequential()
# Image Normalization 
tf.keras.layers.Rescaling(1./255, input_shape=(96, 96, 3)), 

# Add the hidden layer
# Conv2D: 32 filters, 3x3 kernel, ReLU activation
# MaxPooling2D: 2x2 pooling to reduce dimensions

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))                  
model.add(BatchNormalization()) # Add Batch Normalization
model.add(MaxPooling2D((2, 2)))

# Conv2D: 64 filters, 3x3 kernel, ReLU activation
# MaxPooling2D: 2x2 pooling to reduce dimensions
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
model.add(BatchNormalization()) # Added Batch Normalization
model.add(MaxPooling2D((2, 2)))

# Flatten the output of the convolutional layers

model.add(Flatten())

# Add the output layer
# Dense: 1 neuron with a sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid', 
                kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0001)))

# Compile the model
# 'binary_crossentropy' for binary classification
sgd_optimizer = keras.optimizers.SGD(learning_rate=0.0001) 
model.compile(optimizer=sgd_optimizer, 
              loss='binary_crossentropy',
              metrics=['accuracy'])

#model's architecture
model.summary()


history = model.fit(train_ds,epochs=150)


train_loss = history.history['loss']
train_accuracy = history.history['accuracy']


history = model.fit(val_ds,epochs=150)


val_loss = history.history['loss']
val_accuracy = history.history['accuracy'] 


#epochs_range = range(100) # First Tuning
epochs_range = range(150) # Second Tuning

# Plot for Accuracy - Train and Validation set
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot for Loss - Train and Validation set
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# adam optimizer
adam_optimizer = keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model
# 'binary_crossentropy'  for binary classification
model.compile(optimizer=adam_optimizer, 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# model's architecture
model.summary()


history = model.fit(train_ds,epochs=50, batch_size = 32)


history = model.fit(train_ds,epochs=75, batch_size = 64)


train_loss = history.history['loss']
train_accuracy = history.history['accuracy']


val_history = model.fit(val_ds,epochs=50, batch_size = 32)


val_history = model.fit(val_ds,epochs=75, batch_size = 64)


val_loss = val_history.history['loss']
val_accuracy = val_history.history['accuracy'] 


#epochs_range = range(50)  # for first tuning
epochs_range = range(75)   # for second tuning

# Plot for Accuracy - Train and Validation set
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot for Loss - Train and Validation set
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Compile the model
# 'binary_crossentropy' for binary classification
rmsprop_optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=rmsprop_optimizer, #'adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#model's architecture
model.summary()


history = model.fit(train_ds,epochs=20, batch_size = 32)


history = model.fit(train_ds,epochs=40, batch_size = 64)


# get the accuracy and loss scores at each epoch
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']


val_history = model.fit(val_ds,epochs=20, batch_size = 32)


val_history = model.fit(val_ds,epochs=40, batch_size = 64)


val_loss = val_history.history['loss']
val_accuracy = val_history.history['accuracy'] 


#epochs_range = range(20) # for first tuning 
epochs_range = range(40) # for second tuning

# Plot for Accuracy - Train and Validation set
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot for Loss - Train and Validation set
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

