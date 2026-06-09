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



# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random


# create directories
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['dogs/', 'cats/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
		makedirs(newdir, exist_ok=True)


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


import tensorflow as tf
from keras.callbacks import LearningRateScheduler


# EfficientNetB0 model used for transfer learning on the dogs and cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications import EfficientNetV2L 
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.layers import Flatten
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau

# define cnn model
def define_model():
    # Load VGG16 model
    model = EfficientNetV2L(include_top=False, input_shape=(480, 480, 3)) 
    for layer in model.layers:
        layer.trainable = False
    # Add new classifier layers
        #flat1 = Flatten()(model.layers[-1].output)
        #class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
        #output = Dense(1, activation='sigmoid')(class1)
    x = GlobalAveragePooling2D()(model.output)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    # Define new model
    model = Model(inputs=model.inputs, outputs=output)
    # Compile model with Adam optimizer
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

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
    pyplot.close()

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
initial_learning_rate = 0.001  # ค่าเริ่มต้นของ Learning Rate
epochs = 30
def cosine_decay(epoch):
    return initial_learning_rate * 0.5 * (1 + tf.math.cos((epoch / epochs) * 3.14159))
cosine_scheduler = LearningRateScheduler(cosine_decay, verbose=1)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(
        rotation_range=20,
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
    # initialize PlotLosses callback
    plot_losses = PlotLossesKeras()
    # fit model with PlotLosses callback
    history = model.fit_generator(
        train_it, 
        steps_per_epoch=len(train_it), 
        validation_data=test_it, 
        validation_steps=len(test_it), 
        epochs=20, 
        verbose=1,
        callbacks=[lr_scheduler, PlotLossesKeras()]
    ) 
    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)
    model.save_weights('model_weights.h5')

# entry point, run the test harness
run_test_harness()



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# โหลดโมเดลที่ถูกฝึกไว้แล้ว
model = define_model()  # สร้างโมเดล
model.load_weights('model_weights.h5')  # โหลดน้ำหนักของโมเดลที่ถูกฝึกแล้ว

# ใช้ ImageDataGenerator พร้อม EfficientNetV2 preprocessing
datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input)

# เตรียมชุดข้อมูลทดสอบ
test_it = datagen.flow_from_directory(
    'dataset_dogs_vs_cats/test/',
    class_mode='binary', 
    batch_size=64, 
    target_size=(480, 480)  # ✅ แก้ไขเป็น 480x480 ให้ตรงกับ EfficientNetV2L
)

# ฟังก์ชันสุ่มเลือกรูปภาพจากชุดทดสอบ
def random_images_from_test_set(test_it, num_images=30):
    images, labels = next(test_it)  # ดึง batch แรกออกมา
    num_samples = len(images)

    if num_samples < num_images:
        num_images = num_samples  # ป้องกันกรณีรูปใน batch ไม่พอ

    indices = np.random.choice(num_samples, num_images, replace=False)
    random_images = images[indices]
    random_labels = labels[indices]

    return random_images, random_labels

# ฟังก์ชันทำนายภาพที่สุ่ม
def predict_random_images(model, test_it, num_images=30):
    random_images, random_labels = random_images_from_test_set(test_it, num_images)

    # ทำนายผล
    predictions = model.predict(random_images)

    # แสดงผล
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(5, 6, i + 1)  # 5 แถว 6 คอลัมน์
        img = random_images[i]
        true_label = 'Dog' if random_labels[i] == 1 else 'Cat'
        pred_label = 'Dog' if predictions[i] > 0.5 else 'Cat'

        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
        plt.axis('off')

    plt.show()

# เรียกใช้งาน
predict_random_images(model, test_it)





