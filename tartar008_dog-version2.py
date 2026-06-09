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
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


from os import listdir
from numpy import asarray, save
from keras.preprocessing.image import load_img, img_to_array
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


# à¸™à¸³à¹€à¸‚à¹‰à¸² Keras à¹�à¸¥à¸° TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥ Sequential
model = Sequential()

# block 1
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))

# block 2
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))

# block 3
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))

# à¹�à¸ªà¸”à¸‡à¹‚à¸¡à¹€à¸”à¸¥
model.summary()



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# à¹€à¸£à¸µà¸¢à¸�à¹ƒà¸Šà¹‰à¸Ÿà¸±à¸‡à¸�à¹Œà¸Šà¸±à¸™à¹€à¸�à¸·à¹ˆà¸­à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥
model = define_model()

# à¹�à¸ªà¸”à¸‡ summary à¸‚à¸­à¸‡à¹‚à¸¡à¹€à¸”à¸¥
model.summary()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# create data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# prepare iterator
train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
    class_mode='binary', batch_size=64, target_size=(224, 224))
test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
    class_mode='binary', batch_size=64, target_size=(224, 224))


from livelossplot import PlotLossesKeras

# fit model
history = model.fit(train_it, 
                    steps_per_epoch=len(train_it), 
                    epochs=10, verbose=0, 
                    validation_data=test_it, 
                    validation_steps=len(test_it), 
                    callbacks=[PlotLossesKeras()]
                   )


# evaluate model
_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))



# save model
model.save('final_model.h5')


from keras.models import load_model

# à¹‚à¸«à¸¥à¸”à¹‚à¸¡à¹€à¸”à¸¥à¸ˆà¸²à¸�à¹„à¸Ÿà¸¥à¹Œ
model = load_model('final_model.h5')

# à¹ƒà¸Šà¹‰à¸‡à¸²à¸™à¹‚à¸¡à¹€à¸”à¸¥à¸—à¸µà¹ˆà¹‚à¸«à¸¥à¸”à¸¡à¸²
model.summary()  # à¹�à¸ªà¸”à¸‡à¸£à¸²à¸¢à¸¥à¸°à¹€à¸­à¸µà¸¢à¸”à¸‚à¸­à¸‡à¹‚à¸¡à¹€à¸”à¸¥à¸—à¸µà¹ˆà¹‚à¸«à¸¥à¸”à¸¡à¸²



# Defining the model with a correct input shape (200, 200, 3)
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))  # Adjust the input shape here
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


from matplotlib import pyplot
def summarize_diagnostics(history):
    pyplot.figure(figsize=(8, 6), dpi=150)
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.show()
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.show()
    # save plot to file
    #filename = sys.argv[0].split('/')[-1]
    #pyplot.savefig(filename + '_plot.png')
    pyplot.close()


import matplotlib.pyplot as pyplot


# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generator with validation split
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # 20% à¸ªà¸³à¸«à¸£à¸±à¸š validation
    
    # prepare iterators
    train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
        class_mode='binary', batch_size=64, target_size=(200, 200), subset='training')  # à¹ƒà¸Šà¹‰à¸‚à¹‰à¸­à¸¡à¸¹à¸¥ train
    
    validation_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
        class_mode='binary', batch_size=64, target_size=(200, 200), subset='validation')  # à¹ƒà¸Šà¹‰à¸‚à¹‰à¸­à¸¡à¸¹à¸¥ validation à¸ˆà¸²à¸� train
    
    # fit model
    history = model.fit(train_it, 
                    steps_per_epoch=len(train_it), 
                    epochs=20, 
                    verbose=0, 
                    validation_data=validation_it, 
                    validation_steps=len(validation_it), 
                    callbacks=[PlotLossesKeras()]
                   )
    
    # evaluate model on test data
    test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
        class_mode='binary', batch_size=64, target_size=(200, 200))
    
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))

    # learning curves
    summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()



# save model
model.save('final_model.h5')


import os

dataset_path = "/kaggle/input/"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        print(os.path.join(root, file))  # à¹�à¸ªà¸”à¸‡à¸—à¸µà¹ˆà¸­à¸¢à¸¹à¹ˆà¹„à¸Ÿà¸¥à¹Œà¸—à¸±à¹‰à¸‡à¸«à¸¡à¸”



from keras.models import load_model

model_path = "/kaggle/input/modelsdogandcat/keras/default/1/final_model.h5"
model = load_model(model_path)

model.summary()



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)  # à¹ƒà¸Šà¹‰ learning_rate à¹�à¸—à¸™ lr
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# ğŸ”¥ à¸—à¸”à¸ªà¸­à¸šà¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥
model = define_model()
print(model.summary())  # à¹�à¸ªà¸”à¸‡à¹‚à¸„à¸£à¸‡à¸ªà¸£à¹‰à¸²à¸‡à¸‚à¸­à¸‡à¹‚à¸¡à¹€à¸”à¸¥



import matplotlib.pyplot as pyplot


# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generator with validation split
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # 20% à¸ªà¸³à¸«à¸£à¸±à¸š validation
    
    # prepare iterators
    train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
        class_mode='binary', batch_size=64, target_size=(200, 200), subset='training')  # à¹ƒà¸Šà¹‰à¸‚à¹‰à¸­à¸¡à¸¹à¸¥ train
    
    validation_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
        class_mode='binary', batch_size=64, target_size=(200, 200), subset='validation')  # à¹ƒà¸Šà¹‰à¸‚à¹‰à¸­à¸¡à¸¹à¸¥ validation à¸ˆà¸²à¸� train
    
    # fit model
    history = model.fit(train_it, 
                    steps_per_epoch=len(train_it), 
                    epochs=20, 
                    verbose=0, 
                    validation_data=validation_it, 
                    validation_steps=len(validation_it), 
                    callbacks=[PlotLossesKeras()]
                   )
    
    # evaluate model on test data
    test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
        class_mode='binary', batch_size=64, target_size=(200, 200))
    
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))

# entry point, run the test harness
run_test_harness()



from matplotlib import pyplot
def summarize_diagnostics(history):
    pyplot.figure(figsize=(8, 6), dpi=150)
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.show()
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.show()
    # save plot to file
    #filename = sys.argv[0].split('/')[-1]
    #pyplot.savefig(filename + '_plot.png')
    pyplot.close()


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model with correct parameter name
    opt = SGD(learning_rate=0.001, momentum=0.9)  # à¹ƒà¸Šà¹‰ learning_rate à¹�à¸—à¸™ lr
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model



import matplotlib.pyplot as pyplot
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# define model
model = define_model()
# create data generator with validation split
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # 20% à¸ªà¸³à¸«à¸£à¸±à¸š validation

# prepare iterators
train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
    class_mode='binary', batch_size=64, target_size=(200, 200), subset='training')  # à¹ƒà¸Šà¹‰à¸‚à¹‰à¸­à¸¡à¸¹à¸¥ train

validation_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
    class_mode='binary', batch_size=64, target_size=(200, 200), subset='validation')  # à¹ƒà¸Šà¹‰à¸‚à¹‰à¸­à¸¡à¸¹à¸¥ validation à¸ˆà¸²à¸� train

# fit model
history = model.fit(train_it, 
                steps_per_epoch=len(train_it), 
                epochs=20, 
                verbose=0, 
                validation_data=validation_it, 
                validation_steps=len(validation_it), 
                callbacks=[PlotLossesKeras()]
               )


# evaluate model on test data
test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
    class_mode='binary', batch_size=64, target_size=(200, 200))

_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))


# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)  # à¹ƒà¸Šà¹‰ learning_rate à¹�à¸—à¸™ lr
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


import matplotlib.pyplot as pyplot
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# define model
model = define_model()
# create data generator with validation split
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # 20% à¸ªà¸³à¸«à¸£à¸±à¸š validation

# prepare iterators
train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
    class_mode='binary', batch_size=64, target_size=(200, 200), subset='training')  # à¹ƒà¸Šà¹‰à¸‚à¹‰à¸­à¸¡à¸¹à¸¥ train

validation_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
    class_mode='binary', batch_size=64, target_size=(200, 200), subset='validation')  # à¹ƒà¸Šà¹‰à¸‚à¹‰à¸­à¸¡à¸¹à¸¥ validation à¸ˆà¸²à¸� train

# fit model
history = model.fit(train_it, 
                steps_per_epoch=len(train_it), 
                epochs=20, 
                verbose=0, 
                validation_data=validation_it, 
                validation_steps=len(validation_it), 
                callbacks=[PlotLossesKeras()]
               )


# evaluate model on test data
test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
    class_mode='binary', batch_size=64, target_size=(200, 200))

_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))


from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)  # à¹ƒà¸Šà¹‰ learning_rate à¹�à¸—à¸™ lr
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

model = define_model()


# fit model
history = model.fit(train_it, 
                steps_per_epoch=len(train_it), 
                epochs=20, 
                verbose=0, 
                validation_data=validation_it, 
                validation_steps=len(validation_it), 
                callbacks=[PlotLossesKeras()]
               )


# evaluate model on test data
test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
    class_mode='binary', batch_size=64, target_size=(200, 200))

_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))


import sys
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model



# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generators
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                      width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    train_it = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
                                                 class_mode='binary', batch_size=64, target_size=(200, 200))
    test_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
                                               class_mode='binary', batch_size=64, target_size=(200, 200))

        # fit model
    history = model.fit(train_it, 
                    steps_per_epoch=len(train_it), 
                    epochs=50, 
                    verbose=1, 
                    validation_data=test_it, 
                    validation_steps=len(test_it), 
                    callbacks=[PlotLossesKeras()]
                   )
    
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()



from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# load model
model = VGG16(
    include_top=False,
    input_shape=(224, 224, 3)
    )
print(model.summary())
#model = Xception(include_top=False, input_shape=(299, 299, 3))


# mark loaded layers as not trainable
for index, layer in enumerate(model.layers[1:15]):  # à¸‚à¹‰à¸²à¸¡ InputLayer
    print(index, layer.name, layer.output.shape)
    layer.trainable = False
    
for index, layer in enumerate(model.layers[15:]):
    print(index, layer.name, layer.output.shape)
    layer.trainable = True



model.layers[-1].output


# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = Dense(1, activation='sigmoid')(class1)
# define new model
my_model = Model(inputs=model.inputs, outputs=output)
# compile model
opt = SGD(learning_rate=0.0001, momentum=0.9)
my_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
my_model.summary()



from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

# à¹‚à¸«à¸¥à¸”à¹‚à¸¡à¹€à¸”à¸¥ VGG16 à¹‚à¸”à¸¢à¹„à¸¡à¹ˆà¸£à¸§à¸¡à¹€à¸¥à¹€à¸¢à¸­à¸£à¹Œ fully connected
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

# à¹€à¸­à¸²à¸•à¹Œà¸�à¸¸à¸•à¸ˆà¸²à¸�à¹‚à¸¡à¹€à¸”à¸¥ VGG16
vgg16_output = base_model.output

# à¹€à¸�à¸´à¹ˆà¸¡à¹€à¸¥à¹€à¸¢à¸­à¸£à¹Œà¸�à¸²à¸£à¸ˆà¸³à¹�à¸™à¸�à¹ƒà¸«à¸¡à¹ˆ
flat1 = Flatten()(vgg16_output)  # à¹�à¸Ÿà¸¥à¸•à¹€à¸­à¸²à¸•à¹Œà¸�à¸¸à¸•
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = Dense(1, activation='sigmoid')(class1)

# à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥à¹ƒà¸«à¸¡à¹ˆà¹‚à¸”à¸¢à¹€à¸Šà¸·à¹ˆà¸­à¸¡à¸•à¹ˆà¸­à¸­à¸´à¸™à¸�à¸¸à¸•à¸ˆà¸²à¸� base_model à¹�à¸¥à¸°à¹€à¸­à¸²à¸•à¹Œà¸�à¸¸à¸•à¸—à¸µà¹ˆà¹€à¸£à¸²à¹€à¸�à¸´à¹ˆà¸¡
my_model = Model(inputs=base_model.input, outputs=output)

# à¸„à¸­à¸¡à¹„à¸�à¸¥à¹Œà¹‚à¸¡à¹€à¸”à¸¥
opt = SGD(learning_rate=0.0001, momentum=0.9)
my_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# à¸ªà¸£à¸¸à¸›à¹‚à¸„à¸£à¸‡à¸ªà¸£à¹‰à¸²à¸‡à¸‚à¸­à¸‡à¹‚à¸¡à¹€à¸”à¸¥
my_model.summary()



# fit model
history = my_model.fit(train_it, 
                       validation_data=test_it, 
                       epochs=2, verbose=1,
                       callbacks=[PlotLossesKeras()])

# evaluate model
_, acc = my_model.evaluate(test_it, verbose=1)
print('> %.3f' % (acc * 100.0))

# learning curves
summarize_diagnostics(history)

# Save the best model
best_model = 'vgg16_model.hdf5'
my_model.save(best_model)



# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
# create directories
dataset_home = 'finalize_dogs_vs_cats/'
# create label subdirectories
labeldirs = ['dogs/', 'cats/']
for labldir in labeldirs:
	newdir = dataset_home + labldir
	makedirs(newdir, exist_ok=True)
# copy training dataset images into subdirectories
src_directory = 'train/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	if file.startswith('cat'):
		dst = dataset_home + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + 'dogs/'  + file
		copyfile(src, dst)





# prepare iterator
train_it = datagen.flow_from_directory('finalize_dogs_vs_cats/',
	class_mode='binary', batch_size=64, target_size=(200, 200))


# fit model
my_model.fit(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=2)


# save model
my_model.save('final_model.h5')


import os

# à¸£à¸°à¸šà¸¸à¹‚à¸Ÿà¸¥à¹€à¸”à¸­à¸£à¹Œà¸—à¸µà¹ˆà¸¡à¸µà¸£à¸¹à¸›à¸ à¸²à¸�
folder = 'train/'  # à¹€à¸›à¸¥à¸µà¹ˆà¸¢à¸™à¹€à¸›à¹‡à¸™ path à¸‚à¸­à¸‡ dataset à¸ˆà¸£à¸´à¸‡

# à¸”à¸¶à¸‡à¸£à¸²à¸¢à¸Šà¸·à¹ˆà¸­à¹„à¸Ÿà¸¥à¹Œà¸—à¸±à¹‰à¸‡à¸«à¸¡à¸”à¹ƒà¸™à¹‚à¸Ÿà¸¥à¹€à¸”à¸­à¸£à¹Œ
image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(image_files) == 0:
    print("à¹„à¸¡à¹ˆà¸¡à¸µà¹„à¸Ÿà¸¥à¹Œà¸ à¸²à¸�à¹ƒà¸™à¹‚à¸Ÿà¸¥à¹€à¸”à¸­à¸£à¹Œà¸—à¸µà¹ˆà¸�à¸³à¸«à¸™à¸”")
else:
    # à¹ƒà¸Šà¹‰à¸ à¸²à¸�à¹�à¸£à¸�à¹ƒà¸™à¸¥à¸´à¸ªà¸•à¹Œ (à¸«à¸£à¸·à¸­à¸ˆà¸°à¸ªà¸¸à¹ˆà¸¡à¸�à¹‡à¹„à¸”à¹‰)
    first_image = image_files[0]
    print(f'à¸Šà¸·à¹ˆà¸­à¹„à¸Ÿà¸¥à¹Œà¸ à¸²à¸�à¹�à¸£à¸�: {first_image}')

    # à¸–à¹‰à¸²à¸•à¹‰à¸­à¸‡à¸�à¸²à¸£à¸ªà¸¸à¹ˆà¸¡à¹€à¸¥à¸·à¸­à¸�à¸ à¸²à¸�à¹ƒà¸”à¸�à¹‡à¹„à¸”à¹‰
    import random
    random_image = random.choice(image_files)
    print(f'à¸Šà¸·à¹ˆà¸­à¹„à¸Ÿà¸¥à¹Œà¸—à¸µà¹ˆà¹€à¸¥à¸·à¸­à¸�à¹�à¸šà¸šà¸ªà¸¸à¹ˆà¸¡: {random_image}')



# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(200, 200))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 200, 200, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img
 
# load an image and predict the class
def run_example():
	# load the image
	img = load_image('train/dog.2168.jpg')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	result = model.predict(img)
	print(result[0])
 
# entry point, run the example
run_example()


# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(200, 200))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 200, 200, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img
 
# load an image and predict the class
def run_example():
	# load the image
	img = load_image('train/cat.3599.jpg')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	result = model.predict(img)
	print(result[0])
 
# entry point, run the example
run_example()


import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# à¹‚à¸«à¸¥à¸”à¹�à¸¥à¸°à¹€à¸•à¸£à¸µà¸¢à¸¡à¸ à¸²à¸�
def load_image(filename):
    img = load_img(filename, target_size=(200, 200))  # à¹‚à¸«à¸¥à¸”à¸ à¸²à¸�à¹�à¸¥à¸°à¸›à¸£à¸±à¸šà¸‚à¸™à¸²à¸”
    img_array = img_to_array(img)  # à¹�à¸›à¸¥à¸‡à¹€à¸›à¹‡à¸™ array
    img_array = img_array.reshape(1, 200, 200, 3)  # à¸›à¸£à¸±à¸š shape à¹ƒà¸«à¹‰à¸•à¸£à¸‡à¸�à¸±à¸šà¹‚à¸¡à¹€à¸”à¸¥
    img_array = img_array.astype('float32')  # à¹�à¸›à¸¥à¸‡à¹€à¸›à¹‡à¸™ float
    img_array = img_array - [123.68, 116.779, 103.939]  # Normalize (VGG16)
    return img, img_array

# à¹‚à¸«à¸¥à¸”à¸ à¸²à¸�à¹�à¸¥à¸°à¸�à¸¢à¸²à¸�à¸£à¸“à¹Œà¸„à¸¥à¸²à¸ª
def run_example():
    filename = 'train/cat.11451.jpg'  # à¹€à¸›à¸¥à¸µà¹ˆà¸¢à¸™à¹€à¸›à¹‡à¸™à¸ à¸²à¸�à¸—à¸µà¹ˆà¸•à¹‰à¸­à¸‡à¸�à¸²à¸£
    true_label = "Cat" if "cat" in filename else "Dog"  # à¸«à¸²à¸„à¸³à¸•à¸­à¸šà¸—à¸µà¹ˆà¹�à¸—à¹‰à¸ˆà¸£à¸´à¸‡à¸ˆà¸²à¸�à¸Šà¸·à¹ˆà¸­à¹„à¸Ÿà¸¥à¹Œ

    # à¹‚à¸«à¸¥à¸”à¸ à¸²à¸�
    img, img_array = load_image(filename)
    
    # à¹‚à¸«à¸¥à¸”à¹‚à¸¡à¹€à¸”à¸¥
    model = load_model('final_model.h5')

    # à¸—à¸³à¸™à¸²à¸¢à¸„à¸¥à¸²à¸ª
    result = model.predict(img_array)[0][0]
    predicted_label = "Cat" if result < 0.5 else "Dog"  # à¸•à¸±à¹‰à¸‡à¸„à¹ˆà¸²à¸‚à¸µà¸”à¸ˆà¸³à¸�à¸±à¸” (0.5)

    # à¹�à¸ªà¸”à¸‡à¸ à¸²à¸�à¹�à¸¥à¸°à¸œà¸¥à¸¥à¸±à¸�à¸˜à¹Œ
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label}\nActual: {true_label}\nConfidence: {result:.4f}")
    plt.show()

# à¸£à¸±à¸™à¸•à¸±à¸§à¸­à¸¢à¹ˆà¸²à¸‡
run_example()


