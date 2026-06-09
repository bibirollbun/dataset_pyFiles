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


# Import libraries
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

import os
import pandas as pd

from matplotlib import pyplot as plt
import time


# Defining Constants
Image_size = (128, 128)

Img_width = 128
Img_height = 128

batch_size = 32
epochs = 15
channels = 3

# Define image dimensions
num_classes = 7


# Load the datasets
# 1. sample_submission.csv
sample_submission = pd.read_csv("/kaggle/input/image-matching-challenge-2025/sample_submission.csv")
sample_submission.head()


# 2. train_labels.csv
train_labels = pd.read_csv("/kaggle/input/image-matching-challenge-2025/train_labels.csv")
train_labels.head()


# 3. categories.csv
categories = pd.read_csv("/kaggle/input/image-matching-challenge-2025/train_thresholds.csv")
categories.head()


# Plotting
# 1. sample_submission.csv
sample_submission['dataset'].value_counts().plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Vals')
plt.title('Sample Submission')
plt.show()


# Check for missing values in the three datasets
print("Missing values in Sample-Submission:", sample_submission.isnull().sum())

print("Missing values in Train-labels:", train_labels.isnull().sum())

print("Missing values in Categories:", categories.isnull().sum())


# Check for duplicated values in the three datasets
print("Duplicates in Sample-Submission:", sample_submission.duplicated().any())

print("Duplicates in Train-labels:", train_labels.duplicated().any())

print("Duplicates in Categories:", categories.duplicated().any())


# Path to the train and test directories
train_dir = '/kaggle/input/image-matching-challenge-2025/train'
test_dir = '/kaggle/input/image-matching-challenge-2025/test'


# Path for the training images and testing
train_images = tf.data.Dataset.list_files(train_dir + '/*')
test_images = tf.data.Dataset.list_files(test_dir + '/*')


# Filtering the training and testing images to exclude .csv
train_images = train_images.filter(lambda x: not tf.strings.regex_full_match(x, '.*\.csv'))
test_images = test_images.filter(lambda x: not tf.strings.regex_full_match(x, '.*\.csv'))


for file_path in train_images:
    print(file_path.numpy())


for file_path in test_images:
    print(file_path.numpy())


# Shuffing
train_images = train_images.shuffle(500)
test_images = test_images.shuffle(500)


for file_path in train_images:
    print(file_path.numpy())


# Classes in our dataset both of training and testing
train_classes = ["church", "dioscuri", "lizard", "multi-temporal-temple-baalshar", "pond", "transp_obj_glass_cup", "transp_obj_glass_cylinder"]
test_classes = ["church"]


# Finding the count of classes for training
train_images_count = 0
for _ in train_images:
    train_images_count += 1

print(train_images_count)


# Finding the count of classes for testing
test_images_count = 0
for _ in test_images:
    test_images_count += 1

print(test_images_count)


train_size = int(train_images_count + test_images_count * 0.8)

train_ds = train_images.take(train_size)
test_ds = train_images.skip(train_size)


def get_label(file_path):
    return tf.strings.split(file_path, os.path.sep)[-2]


def process_image(file_path):
    label = get_label(file_path)

    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [128, 128])

    return img, label


for t in train_ds.take(4):
    print (t.numpy())


import matplotlib.image as mpimg


def process_image(file_path, label):

    img = tf.io.read_file(file_path)

    img = tf.image.decode_jpeg(img, channels=3)  # Adjust channels if needed

    img = tf.image.resize(img, [img_height, img_width])

    img = img / 255.0
    return img, label


image_path = "/kaggle/input/image-matching-challenge-2025/train/pt_stpeters_stpauls/st_peters_square_35727766_2927321004.png"

# Load the image
image = mpimg.imread(image_path)

# Display the image
plt.imshow(image)
plt.axis('off')  # Turn off axis
plt.show()


image_path = "/kaggle/input/image-matching-challenge-2025/train/pt_stpeters_stpauls/st_peters_square_35727766_2927321004.png"

# Load the image
image = mpimg.imread(image_path)

# Display the image
plt.imshow(image)
plt.axis('off')  # Turn off axis
plt.show()


image_path = "/kaggle/input/image-matching-challenge-2025/train/amy_gardens/peach_0026.png"

# Load the image
image = mpimg.imread(image_path)

# Display the image
plt.imshow(image)
plt.axis('off')  # Turn off axis
plt.show()


def process_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = img / 255.0
    return img, label


train_ds = train_images.take(train_size).map(lambda x: process_image(x, get_label(x)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = train_images.skip(train_size).map(lambda x: process_image(x, get_label(x)), num_parallel_calls=tf.data.experimental.AUTOTUNE)


# Before prefetching
start_time = time.time()

# Define and preprocess the dataset without prefetching
train_ds = train_images.take(train_size).map(lambda x: process_image(x, get_label(x)))


# Apply prefetching
train_ds = train_ds.prefetch(buffer_size=1)

# Calculate time taken
time_after_prefetch1 = time.time() - start_time


print("Time taken after prefetching buffer_size 1:", time_after_prefetch1)


# Apply prefetching
train_ds = train_ds.prefetch(buffer_size=2)

# Calculate time taken
time_after_prefetch2 = time.time() - start_time


print("Time taken after prefetching buffer size 2:", time_after_prefetch2)


# Apply prefetching
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Calculate time taken
time_after_prefetch_AUTOTUNE = time.time() - start_time


print("Time taken after prefetching buffer size AUTOTUNE:", time_after_prefetch_AUTOTUNE)


# Before caching
start_time = time.time()

# Define and preprocess the dataset without prefetching and caching
train_ds_no_cache = train_images.take(train_size).map(lambda x: process_image(x, get_label(x)), num_parallel_calls=tf.data.experimental.AUTOTUNE)


# Calculate time taken
time_before_cache = time.time() - start_time

# Apply caching
train_ds_cached = train_ds_no_cache.cache()


# Calculate time taken
time_after_cache = time.time() - start_time


print("Time taken before caching:", time_before_cache)
print("Time taken after caching:", time_after_cache)


import pandas as pd
import numpy as np
import io

data = """dataset,scene,image,rotation_matrix,translation_vector
dataset1,cluster1,image1.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,cluster1,image2.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,cluster2,image3.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,cluster2,image4.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,outliers,image5.png,nan;nan;nan;nan;nan;nan;nan;nan;nan,nan;nan;nan
dataset2,cluster1,image1.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
"""

df = pd.read_csv(io.StringIO(data))

def parse_matrix(matrix_str):
    try:
        return np.array([float(x) for x in matrix_str.split(';')]).reshape((3, 3))
    except:
        return np.full((3, 3), np.nan)

def parse_vector(vector_str):
    try:
        return np.array([float(x) for x in vector_str.split(';')]).reshape((3,))
    except:
        return np.full((3,), np.nan)

df['rotation_matrix'] = df['rotation_matrix'].apply(parse_matrix)
df['translation_vector'] = df['translation_vector'].apply(parse_vector)

print(df)


import pandas as pd
import numpy as np
import io

data = """dataset,scene,image,rotation_matrix,translation_vector
dataset1,cluster1,image1.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,cluster1,image2.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,cluster2,image3.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,cluster2,image4.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,outliers,image5.png,nan;nan;nan;nan;nan;nan;nan;nan;nan,nan;nan;nan
dataset2,cluster1,image1.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset2,cluster1,image2.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset2,outliers,image3.png,nan;nan;nan;nan;nan;nan;nan;nan;nan,nan;nan;nan
dataset3,unclustered,image1.png,nan;nan;nan;nan;nan;nan;nan;nan;nan,nan;nan;nan
dataset3,unclustered,image2.png,nan;nan;nan;nan;nan;nan;nan;nan;nan,nan;nan;nan
"""

df = pd.read_csv(io.StringIO(data))

def parse_matrix(matrix_str):
    values = matrix_str.split(';')
    try:
        return np.array([float(x) for x in values]).reshape((3, 3))
    except ValueError:
        return np.full((3, 3), np.nan)

def parse_vector(vector_str):
    values = vector_str.split(';')
    try:
        return np.array([float(x) for x in values]).reshape((3,))
    except ValueError:
        return np.full((3,), np.nan)

df['rotation_matrix'] = df['rotation_matrix'].apply(parse_matrix)
df['translation_vector'] = df['translation_vector'].apply(parse_vector)

print(df)

