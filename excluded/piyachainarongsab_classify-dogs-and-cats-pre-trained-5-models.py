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
print(os.listdir("../input"))

import zipfile

with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/test1.zip","r") as z:
    z.extractall(".")
    
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")


import psutil
print(f"Available Memory: {psutil.virtual_memory().available / 1e9:.2f} GB")



!fallocate -l 4G /swapfile
!chmod 600 /swapfile
!mkswap /swapfile
!swapon /swapfile
!free -h  # à¸•à¸£à¸§à¸ˆà¸ªà¸­à¸š Swap à¸—à¸µà¹ˆà¹€à¸�à¸´à¹ˆà¸¡



# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
# define location of dataset
folder = 'train/'
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'dog.' + str(i) + '.jpg'
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()


# plot cat photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
# define location of dataset
folder = 'train/'
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'cat.' + str(i) + '.jpg'
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()


# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
from numpy import asarray
from numpy import save
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


from os import listdir
from numpy import asarray, save
from tensorflow.keras.preprocessing.image import load_img , img_to_array

import random

# define location of dataset
folder = 'train/'
photos, labels = list(), list()

# à¸ªà¸¸à¹ˆà¸¡à¹€à¸¥à¸·à¸­à¸�à¹�à¸„à¹ˆ 50% à¸‚à¸­à¸‡à¹„à¸Ÿà¸¥à¹Œà¸—à¸±à¹‰à¸‡à¸«à¸¡à¸”
files = listdir(folder)
random.shuffle(files)  # à¸ªà¸¸à¹ˆà¸¡à¸¥à¸³à¸”à¸±à¸šà¹„à¸Ÿà¸¥à¹Œ
files = files[:len(files) // 2]  # à¹€à¸­à¸²à¹�à¸„à¹ˆà¸„à¸£à¸¶à¹ˆà¸‡à¹€à¸”à¸µà¸¢à¸§

# enumerate files in the directory
for file in files:
    # determine class
    output = 1.0 if file.startswith('dog') else 0.0

    # load image
    photo = load_img(folder + file, target_size=(100, 100))  # à¸¥à¸”à¸‚à¸™à¸²à¸”à¸£à¸¹à¸›à¸ à¸²à¸�
    photo = img_to_array(photo)

    # store
    photos.append(photo)
    labels.append(output)

# convert to a numpy arrays
photos = asarray(photos, dtype='float16')  # à¹ƒà¸Šà¹‰ float16 à¹€à¸�à¸·à¹ˆà¸­à¸¥à¸”à¸‚à¸™à¸²à¸”
labels = asarray(labels, dtype='uint8')    # uint8 à¹ƒà¸Šà¹‰à¸�à¸·à¹‰à¸™à¸—à¸µà¹ˆà¸™à¹‰à¸­à¸¢à¸�à¸§à¹ˆà¸² float

print(photos.shape, labels.shape)

# save the reshaped photos
save('dogs_vs_cats_photos.npy', photos)
save('dogs_vs_cats_labels.npy', labels)




# # load dogs vs cats dataset, reshape and save to a new file
# from os import listdir
# from numpy import asarray
# from numpy import save
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# # define location of dataset
# folder = 'train/'
# photos, labels = list(), list()
# # enumerate files in the directory
# for file in listdir(folder):
# 	# determine class
# 	output = 0.0
# 	if file.startswith('dog'):
# 		output = 1.0
# 	# load image
# 	photo = load_img(folder + file, target_size=(200, 200))
# 	# convert to numpy array
# 	photo = img_to_array(photo)
# 	# store
# 	photos.append(photo)
# 	labels.append(output)
# # convert to a numpy arrays
# photos = asarray(photos)
# labels = asarray(labels)
# print(photos.shape, labels.shape)
# # save the reshaped photos
# save('dogs_vs_cats_photos.npy', photos)
# save('dogs_vs_cats_labels.npy', labels)


import psutil
print(f"Available Memory: {psutil.virtual_memory().available / 1e9:.2f} GB")



# load and confirm the shape
from numpy import load
photos = load('dogs_vs_cats_photos.npy')
labels = load('dogs_vs_cats_labels.npy')
print(photos.shape, labels.shape)



# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random


# Create directories
import os
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    # Create label subdirectories
    labeldirs = ['dogs/', 'cats/']
    for labldir in labeldirs:
        newdir = os.path.join(dataset_home, subdir, labldir)
        os.makedirs(newdir, exist_ok=True)






# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = 'train/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'train/'
	if random() < val_ratio:
		dst_dir = 'test/'
	if file.startswith('cat'):
		dst = dataset_home + dst_dir + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + dst_dir + 'dogs/'  + file
		copyfile(src, dst)


!ls -l dataset_dogs_vs_cats/train/dogs | wc -l


!ls -l dataset_dogs_vs_cats/train/cats | wc -l


!pip install livelossplot


from livelossplot import PlotLossesKeras


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# create data generator with validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # à¹�à¸šà¹ˆà¸‡ 20% à¹€à¸›à¹‡à¸™ validation
)

# prepare iterators for training, validation, and testing
train_it = datagen.flow_from_directory(
    'dataset_dogs_vs_cats/train/',  # à¹ƒà¸«à¹‰à¹�à¸™à¹ˆà¹ƒà¸ˆà¸§à¹ˆà¸² path à¸•à¸£à¸‡
    class_mode='binary',
    batch_size=64,
    target_size=(224, 224),
    subset="training"  # à¹ƒà¸Šà¹‰à¸ªà¸³à¸«à¸£à¸±à¸š training
)

val_it = datagen.flow_from_directory(
    'dataset_dogs_vs_cats/train/',  # à¹ƒà¸Šà¹‰ dataset à¹€à¸”à¸µà¸¢à¸§à¸�à¸±à¸™
    class_mode='binary',
    batch_size=64,
    target_size=(224, 224),
    subset="validation"  # à¹ƒà¸Šà¹‰à¸ªà¸³à¸«à¸£à¸±à¸š validation
)

# prepare iterator for testing
test_datagen = ImageDataGenerator(rescale=1./255)  # à¹„à¸¡à¹ˆà¹ƒà¸Šà¹‰ validation_split à¸ªà¸³à¸«à¸£à¸±à¸š test
test_it = test_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/test/',  # à¹‚à¸Ÿà¸¥à¹€à¸”à¸­à¸£à¹Œà¸ªà¸³à¸«à¸£à¸±à¸šà¸Šà¸¸à¸”à¸—à¸”à¸ªà¸­à¸š
    class_mode='binary',
    batch_size=64,
    target_size=(224, 224)
)



from tensorflow.keras.applications import VGG19

# à¹‚à¸«à¸¥à¸”à¹‚à¸¡à¹€à¸”à¸¥ VGG19 (à¸•à¸±à¸” Fully Connected Layers à¸­à¸­à¸�)
base_model = VGG19(
    include_top=False,  # à¹„à¸¡à¹ˆà¹‚à¸«à¸¥à¸” Fully Connected Layers
    weights="imagenet", # à¹ƒà¸Šà¹‰ pretrained weights à¸ˆà¸²à¸� ImageNet
    input_shape=(224, 224, 3)
)

print(base_model.summary())


# à¹�à¸Šà¹ˆà¹�à¸‚à¹‡à¸‡ Layers à¸šà¸²à¸‡à¸ªà¹ˆà¸§à¸™ (block1 - block3)
for layer in base_model.layers[:15]:  
    layer.trainable = False


from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# à¹ƒà¸Šà¹‰ GlobalAveragePooling2D à¹�à¸—à¸™ Flatten
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

# à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥à¹ƒà¸«à¸¡à¹ˆ
model = Model(inputs=base_model.input, outputs=output)


from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# à¸„à¸­à¸¡à¹„à¸�à¸¥à¹Œà¹‚à¸¡à¹€à¸”à¸¥à¸�à¹ˆà¸­à¸™à¸�à¸²à¸£à¸—à¸”à¸ªà¸­à¸š
model.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from livelossplot import PlotLossesKeras
import tensorflow as tf

# Callbacks
checkpoint = ModelCheckpoint(
    "best_model_VGG19.keras", monitor="val_loss", save_best_only=True, mode="min", verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
)
csv_logger = CSVLogger("training_log.csv", append=True)

# à¹€à¸—à¸£à¸™à¹‚à¸¡à¹€à¸”à¸¥
history = model.fit(
    train_it,
    epochs=1,
    validation_data=val_it,
    callbacks=[PlotLossesKeras(), checkpoint, early_stopping, reduce_lr, csv_logger]
)


import pandas as pd
import matplotlib.pyplot as plt

# ğŸ”¹ à¹‚à¸«à¸¥à¸” Log CSV
log_df = pd.read_csv("/kaggle/input/log-all-train-model/training_log_VGG19.csv")  # à¹�à¸�à¹‰à¹€à¸›à¹‡à¸™à¸Šà¸·à¹ˆà¸­à¹„à¸Ÿà¸¥à¹Œà¸—à¸µà¹ˆà¸•à¹‰à¸­à¸‡à¸�à¸²à¸£

# ğŸ”¹ à¸ªà¸£à¹‰à¸²à¸‡à¸�à¸£à¸²à¸Ÿ Loss
plt.figure(figsize=(10, 5))
plt.plot(log_df["epoch"], log_df["loss"], label="Training Loss", marker="o")
plt.plot(log_df["epoch"], log_df["val_loss"], label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid()
plt.show()

# ğŸ”¹ à¸ªà¸£à¹‰à¸²à¸‡à¸�à¸£à¸²à¸Ÿ Accuracy
plt.figure(figsize=(10, 5))
plt.plot(log_df["epoch"], log_df["accuracy"], label="Training Accuracy", marker="o")
plt.plot(log_df["epoch"], log_df["val_accuracy"], label="Validation Accuracy", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid()
plt.show()



from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import load_model


# à¹‚à¸«à¸¥à¸”à¸™à¹‰à¸³à¸«à¸™à¸±à¸�à¸ˆà¸²à¸�à¹„à¸Ÿà¸¥à¹Œ .h5 à¸—à¸µà¹ˆà¸¡à¸µ
model_VGG19.load_weights('/kaggle/input/mymodel-pretrained-5-models-dog_cat/keras/default/1/best_model_VGG19.hdf5', by_name=True)


# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
# from tensorflow.keras.models import Model

# # à¹‚à¸¡à¹€à¸”à¸¥ VGG19 à¸—à¸µà¹ˆà¸¡à¸µà¸�à¸²à¸£à¹€à¸�à¸´à¹ˆà¸¡ GlobalAveragePooling2D à¹�à¸¥à¸° Dense layer à¹€à¸�à¸·à¹ˆà¸­à¹ƒà¸«à¹‰à¹„à¸”à¹‰à¸�à¸²à¸£à¸—à¸³à¸™à¸²à¸¢ class
# x = best_model_VGG19.output
# x = GlobalAveragePooling2D()(x)  # à¸—à¸³à¸�à¸²à¸£à¸¥à¸”à¸¡à¸´à¸•à¸´à¸‚à¸­à¸‡ feature map
# x = Dense(10, activation='softmax')(x)  # à¸ˆà¸³à¸™à¸§à¸™à¸„à¸¥à¸²à¸ªà¹€à¸›à¹‡à¸™ 10 (à¹€à¸›à¸¥à¸µà¹ˆà¸¢à¸™à¸•à¸²à¸¡à¸ˆà¸³à¸™à¸§à¸™à¸„à¸¥à¸²à¸ªà¸‚à¸­à¸‡à¸„à¸¸à¸“)
# model = Model(inputs=best_model_VGG19.input, outputs=x)

# # à¸„à¸³à¸™à¸§à¸“à¸�à¸²à¸£à¸�à¸¢à¸²à¸�à¸£à¸“à¹Œ (prediction)
# y_pred = model.predict(test_it)

# # à¸–à¹‰à¸² y_pred à¸¡à¸µà¸¡à¸´à¸•à¸´ (6303, 10) à¸ˆà¸°à¹ƒà¸Šà¹‰ argmax à¹€à¸�à¸·à¹ˆà¸­à¸¥à¸”à¸¡à¸´à¸•à¸´à¹ƒà¸«à¹‰à¹€à¸›à¹‡à¸™à¸„à¸¥à¸²à¸ªà¹€à¸”à¸µà¸¢à¸§
# y_pred = np.argmax(y_pred, axis=-1)

# # à¸•à¸£à¸§à¸ˆà¸ªà¸­à¸šà¸§à¹ˆà¸² y_true à¹€à¸›à¹‡à¸™ label à¸—à¸µà¹ˆà¸¡à¸µà¸„à¹ˆà¸²à¹€à¸›à¹‡à¸™à¸„à¸¥à¸²à¸ªà¹€à¸”à¸µà¸¢à¸§à¸«à¸£à¸·à¸­à¹„à¸¡à¹ˆ
# y_true = np.array(test_it.labels)  # à¹€à¸›à¸¥à¸µà¹ˆà¸¢à¸™à¹ƒà¸«à¹‰à¸•à¸£à¸‡à¸�à¸±à¸šà¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸ˆà¸£à¸´à¸‡à¸‚à¸­à¸‡à¸„à¸¸à¸“

# # à¸–à¹‰à¸² y_true à¹€à¸›à¹‡à¸™ one-hot encoding (à¹€à¸Šà¹ˆà¸™ à¸¡à¸µà¸‚à¸™à¸²à¸” (6303, num_classes)) à¹ƒà¸«à¹‰à¹ƒà¸Šà¹‰ argmax
# if len(y_true.shape) > 1 and y_true.shape[1] > 1:
#     y_true = np.argmax(y_true, axis=-1)

# # à¸•à¸£à¸§à¸ˆà¸ªà¸­à¸šà¸§à¹ˆà¸² y_true à¹�à¸¥à¸° y_pred à¸¡à¸µ shape à¹€à¸”à¸µà¸¢à¸§à¸�à¸±à¸™à¸«à¸£à¸·à¸­à¹„à¸¡à¹ˆ
# assert y_true.shape == y_pred.shape, f"Shape mismatch: y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}"

# # à¸›à¸£à¸°à¹€à¸¡à¸´à¸™à¸œà¸¥à¸�à¸²à¸£à¸—à¸”à¸ªà¸­à¸š
# loss, acc = model.evaluate(test_it, verbose=1)

# # à¹�à¸ªà¸”à¸‡à¸œà¸¥à¸¥à¸±à¸�à¸˜à¹Œ
# print(f"Test loss: {loss}")
# print(f"Test accuracy: {acc}")


from tensorflow.keras.applications import ResNet152
# à¹‚à¸«à¸¥à¸”à¹‚à¸¡à¹€à¸”à¸¥ InceptionV3 (à¸•à¸±à¸” Fully Connected Layers à¸­à¸­à¸�)
base_model = ResNet152(
    include_top=False, 
    weights="imagenet", 
    input_shape=(224, 224, 3))

print(base_model.summary())


# ğŸ”¹ Freeze 70% à¸‚à¸­à¸‡ Layers (à¹„à¸¡à¹ˆà¹ƒà¸«à¹‰ weight à¹€à¸›à¸¥à¸µà¹ˆà¸¢à¸™à¹�à¸›à¸¥à¸‡)
for layer in base_model.layers[:360]:  # ResNet152 à¸¡à¸µ 514 layers
    layer.trainable = False

# à¸•à¸£à¸§à¸ˆà¸ªà¸­à¸šà¸§à¹ˆà¸² layer à¹„à¸«à¸™ trainable à¸šà¹‰à¸²à¸‡
for layer in base_model.layers:
    print(f"{layer.name}: Trainable={layer.trainable}")


from tensorflow.keras.applications import ResNet152
from tensorflow.keras.models import Model  # âœ… Import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout


# ğŸ”¹ à¹€à¸�à¸´à¹ˆà¸¡ Fully Connected Layers à¸”à¹‰à¸²à¸™à¸šà¸™
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)

# ğŸ”¹ à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥à¹ƒà¸«à¸¡à¹ˆ
model = Model(inputs=base_model.input, outputs=x)  # âœ… à¹ƒà¸Šà¹‰ Model à¹„à¸”à¹‰à¹�à¸¥à¹‰à¸§!

# à¹�à¸ªà¸”à¸‡à¹‚à¸„à¸£à¸‡à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥
print(model.summary())


from tensorflow.keras.optimizers import Adam

# ğŸ”¥ à¸•à¸±à¹‰à¸‡à¸„à¹ˆà¸² Adam optimizer à¸�à¸£à¹‰à¸­à¸¡ learning rate
optimizer = Adam(learning_rate=5E-5)  # à¸„à¹ˆà¸²à¹€à¸£à¸´à¹ˆà¸¡à¸•à¹‰à¸™ = 0.001

# ğŸ”¥ Compile à¹‚à¸¡à¹€à¸”à¸¥
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
# from livelossplot import PlotLossesKeras
# import tensorflow as tf

# # Callbacks
# checkpoint = ModelCheckpoint(
#     "best_model_ResNet152.keras", monitor="val_loss", save_best_only=True, mode="min", verbose=1
# )

# early_stopping = EarlyStopping(
#     monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
# )
# reduce_lr = ReduceLROnPlateau(
#     monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
# )
# csv_logger = CSVLogger("training_log.csv", append=True)

# # à¹€à¸—à¸£à¸™à¹‚à¸¡à¹€à¸”à¸¥
# history = model.fit(
#     train_it,
#     # steps_per_epoch=len(train_it),
#     epochs=60,
#     validation_data=val_it,
#     # validation_steps=len(val_it),
#     verbose=1 ,
#     callbacks=[PlotLossesKeras(), checkpoint, early_stopping, reduce_lr, csv_logger]
# )





from tensorflow.keras.applications import InceptionV3

# à¹‚à¸«à¸¥à¸”à¹‚à¸¡à¹€à¸”à¸¥ InceptionV3 (à¸•à¸±à¸” Fully Connected Layers à¸­à¸­à¸�)
base_model = InceptionV3(
    include_top=False,  # à¹„à¸¡à¹ˆà¹‚à¸«à¸¥à¸” Fully Connected Layers
    weights="imagenet",  # à¹ƒà¸Šà¹‰ pretrained weights à¸ˆà¸²à¸� ImageNet
    input_shape=(224, 224, 3)
)

print(base_model.summary())


from tensorflow.keras.models import Model

# à¹‚à¸«à¸¥à¸”à¹‚à¸¡à¹€à¸”à¸¥ 
# ğŸ”¹ Freeze à¹€à¸‰à¸�à¸²à¸° block1 - block3 (à¹„à¸¡à¹ˆà¸­à¸±à¸�à¹€à¸”à¸• weight)
for layer in base_model.layers[:15]:  # à¸«à¸£à¸·à¸­à¹ƒà¸Šà¹‰ base_model.layers[:]
    layer.trainable = False

# ğŸ”¹ Unfreeze à¹€à¸‰à¸�à¸²à¸° block4 - block5 (à¸ªà¸²à¸¡à¸²à¸£à¸–à¹€à¸£à¸µà¸¢à¸™à¸£à¸¹à¹‰à¸•à¹ˆà¸­)
for layer in base_model.layers[15:]:
    layer.trainable = True

# à¸•à¸£à¸§à¸ˆà¸ªà¸­à¸šà¸§à¹ˆà¸² layer à¹„à¸«à¸™ trainable à¸šà¹‰à¸²à¸‡
for layer in base_model.layers:
    print(f"{layer.name}: Trainable={layer.trainable}")


from tensorflow.keras import layers, models

# à¸”à¸¶à¸‡ feature maps à¸ˆà¸²à¸� base_model
x = base_model.output  

# ğŸ”¹ à¹ƒà¸Šà¹‰ GlobalAveragePooling2D à¹�à¸—à¸™ Flatten
x = layers.GlobalAveragePooling2D()(x)  

# ğŸ”¹ Fully Connected Layers
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# ğŸ”¹ Output Layer
output_layer = layers.Dense(1, activation='sigmoid')(x)

# ğŸ”¥ à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥à¹ƒà¸«à¸¡à¹ˆ
model = models.Model(inputs=base_model.input, outputs=output_layer)

# ğŸ”� à¸”à¸¹à¹‚à¸„à¸£à¸‡à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥
model.summary()


# from tensorflow.keras.optimizers import Adam

# # ğŸ”¥ à¸•à¸±à¹‰à¸‡à¸„à¹ˆà¸² Adam optimizer à¸�à¸£à¹‰à¸­à¸¡ learning rate
# optimizer = Adam(learning_rate=0.0001)  # à¸„à¹ˆà¸²à¹€à¸£à¸´à¹ˆà¸¡à¸•à¹‰à¸™ = 0.001

# # ğŸ”¥ Compile à¹‚à¸¡à¹€à¸”à¸¥
# model.compile(
#     optimizer=optimizer,
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )


# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
# from livelossplot import PlotLossesKeras
# import tensorflow as tf

# # Callbacks
# checkpoint = ModelCheckpoint(
#     "best_model_InceptionV3.keras", monitor="val_loss", save_best_only=True, mode="min", verbose=1
# )

# early_stopping = EarlyStopping(
#     monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
# )
# reduce_lr = ReduceLROnPlateau(
#     monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
# )
# csv_logger = CSVLogger("training_log.csv", append=True)

# # à¹€à¸—à¸£à¸™à¹‚à¸¡à¹€à¸”à¸¥
# history = model.fit(
#     train_it,
#     # steps_per_epoch=len(train_it),
#     epochs=50,
#     validation_data=val_it,
#     # validation_steps=len(val_it),
#     verbose=1 ,
#     callbacks=[PlotLossesKeras(), checkpoint, early_stopping, reduce_lr, csv_logger]
# )


from tensorflow.keras.models import load_model
densenet169_model = load_model('/kaggle/input/mymodel-pretrained-5-models-dog_cat/keras/default/1/densenet169_model.hdf5')


# à¸ªà¸³à¸«à¸£à¸±à¸šà¸�à¸²à¸£à¸›à¸£à¸°à¹€à¸¡à¸´à¸™à¸„à¸§à¸²à¸¡à¹�à¸¡à¹ˆà¸™à¸¢à¸³
_, acc = densenet169_model.evaluate(test_it, verbose=1)
print('> Accuracy: %.3f' % (acc * 100.0))


from tensorflow.keras.models import load_model
densenet121_model = load_model('/kaggle/input/mymodel-pretrained-5-models-dog_cat/keras/default/1/densenet121_model.hdf5')


# à¸ªà¸³à¸«à¸£à¸±à¸šà¸�à¸²à¸£à¸›à¸£à¸°à¹€à¸¡à¸´à¸™à¸„à¸§à¸²à¸¡à¹�à¸¡à¹ˆà¸™à¸¢à¸³
_, acc = densenet121_model.evaluate(test_it, verbose=1)
print('> Accuracy: %.3f' % (acc * 100.0))




