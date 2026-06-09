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


!ls -l


ls test1/ |wc -l


ls train/ |wc -l


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


import tensorflow 
print(tensorflow.__version__)

import keras
print(keras.__version__)

import tensorflow.keras
print(tensorflow.__version__)


# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random


# create directories
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']#test=>validation
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['dogs/', 'cats/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)
        print(newdir)


ls -l dataset_dogs_vs_cats/train/dogs/


ls -l


ls -l dataset_dogs_vs_cats/train


ls -l dataset_dogs_vs_cats/test


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


!pip -q install livelossplot==0.5.5


from livelossplot import PlotLossesKeras


#start tutorial
#Develop a Baseline CNN Model



# baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.show()
    pyplot.close()


# save the final model to file
from keras.applications.mobilenet_v2 import MobileNetV2 
from keras.applications.nasnet import NASNetLarge
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


from keras.preprocessing.image import ImageDataGenerator
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


# load model
model = NASNetLarge(
    include_top=False,
    input_shape=(224, 224, 3)
    )
print(model.summary())
#model = Xception(include_top=False, input_shape=(299, 299, 3))


# mark loaded layers as not trainable
for index, layer in enumerate(model.layers):
    print(index, layer.name, layer.output_shape)
    layer.trainable = False
    
for index, layer in enumerate(model.layers[-30:]):
    print(index, layer.name, layer.output_shape)
    layer.trainable = True


model.layers[-1].output


# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = Dense(1, activation='sigmoid')(class1)
# define new model
my_model = Model(inputs=model.inputs, outputs=output)
# compile model
opt = SGD(lr=0.001, momentum=0.9)
my_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
my_model.summary()


 # fit model
my_model.reset_states()
history = my_model.fit(train_it, 
                    steps_per_epoch=len(train_it),
                    validation_data=test_it, 
                    validation_steps=len(test_it), 
                    epochs=2, verbose=1,
                    callbacks=[PlotLossesKeras()])


# evaluate model
_, acc = my_model.evaluate(test_it, steps=len(test_it), verbose=1)
print('> %.3f' % (acc * 100.0))


# learning curves
summarize_diagnostics(history)


best_model = 'NASNetLarge_model.hdf5'
my_model.save(best_model)


# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model



best_model = load_model(best_model)
best_model.evaluate(test_it, verbose=1)


import matplotlib.pyplot as plt


import random
def predict_random_images(test_dir, model, class_names):
    img_size=(224, 224)
    images = []  # เก็บ path รูปภาพ
    labels = []  # เก็บ label จริง

    # เลือก 5 รูปจากแต่ละ class
    for label in class_names:
        folder = os.path.join(test_dir, label)
        files = random.sample(list(os.listdir(folder)), 5)
        for f in files:
            images.append(os.path.join(folder, f))
            labels.append(label)

    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, (img_path, true_label) in enumerate(zip(images, labels)):
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img) / 255
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0]  # ค่าระหว่าง 0-1
        pred_label = class_names[1] if pred > 0.5 else class_names[0]

        axes[i // 5, i % 5].imshow(img)
        axes[i // 5, i % 5].set_title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=10)
        axes[i // 5, i % 5].axis("off")

    plt.tight_layout()
    plt.show()
predict_random_images("dataset_dogs_vs_cats/test/", best_model, ["cats", "dogs"])





