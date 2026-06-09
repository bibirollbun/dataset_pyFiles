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
dataset_home = 'dataset_dogs_vs_cats1/'
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


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


# สร้าง Data Generators
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


train_it = datagen.flow_from_directory(
    'dataset_dogs_vs_cats1/train/',
    class_mode='binary', batch_size=64, target_size=(224, 224)
)
test_it = datagen.flow_from_directory(
    'dataset_dogs_vs_cats1/test/',
    class_mode='binary', batch_size=64, target_size=(224, 224)
)


from keras.applications import DenseNet169
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam


# โหลด DenseNet169 pre-trained model
base_model = DenseNet169(
    weights='imagenet',
    include_top=False,  # ไม่เอาเลเยอร์ Top เดิม (fully connected layers)
    input_shape=(224, 224, 3)  # ขนาด Input
)



# เพิ่ม GlobalAveragePooling2D และ Dense layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # ลดขนาดให้เหลือ 1D
x = Dense(128, activation='relu')(x)  # Fully connected layer
output = Dense(1, activation='sigmoid')(x)  # สำหรับ Binary classification



densenet169_model = Model(inputs=base_model.input, outputs=output)



# Freeze เลเยอร์ของ base_model
for layer in base_model.layers:
    layer.trainable = False


# Compile โมเดล
densenet169_model.compile(optimizer=Adam(learning_rate=0.001), 
                       loss='binary_crossentropy', 
                       metrics=['accuracy'])


densenet169_model.summary()


csv_logger_densenet169 = CSVLogger("training_densenet169_log.csv", append=True)


early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)


# Train
history_densenet169 = densenet169_model.fit(
    train_it,
    validation_data=test_it,
    epochs=10,
    verbose=1,
    callbacks=[PlotLossesKeras(), csv_logger_densenet169, early_stopping]
)

# Evaluate
_, acc = densenet169_model.evaluate(test_it, verbose=1)
print('DenseNet121 Accuracy: %.3f' % (acc * 100.0))


# Save Model
densenet169_model.save('densenet169_model.hdf5')


# import pandas as pd
# import csv
# # อ่านข้อมูลจาก CSV
# epochs, accuracy, loss, val_accuracy, val_loss = [], [], [], [], []
# with open('/kaggle/input/training-densenet169-log/training_densenet169_log.csv', 'r') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         epochs.append(int(row['epoch']))
#         accuracy.append(float(row['accuracy']))
#         loss.append(float(row['loss']))
#         val_accuracy.append(float(row['val_accuracy']))
#         val_loss.append(float(row['val_loss']))


# # แสดงข้อมูลในแต่ละ epoch
# print(f"{'Epoch':<5} {'Accuracy':<10} {'Loss':<10} {'Val_Accuracy':<15} {'Val_Loss':<10}")
# print("=" * 50)
# for i in range(len(epochs)):
#     print(f"{epochs[i]:<5} {accuracy[i]:<10.6f} {loss[i]:<10.6f} {val_accuracy[i]:<15.6f} {val_loss[i]:<10.6f}")


# import pandas as pd

# # โหลด log จากไฟล์ CSV
# log_df = pd.read_csv('/kaggle/input/training-densenet169-log/training_densenet169_log.csv')


# # Plot Training and Validation Loss
# plt.figure(figsize=(10, 6))
# plt.plot(log_df['epoch'], log_df['loss'], label='Train Loss')
# plt.plot(log_df['epoch'], log_df['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()
# plt.show()


# # Plot Training and Validation Accuracy
# plt.figure(figsize=(10, 6))
# plt.plot(log_df['epoch'], log_df['accuracy'], label='Train Accuracy')
# plt.plot(log_df['epoch'], log_df['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid()
# plt.show()


from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# โหลด DenseNet121
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,  # ไม่ใช้ Fully Connected Layer เดิม
    input_shape=(224, 224, 3)
)


# เพิ่มชั้น Classifier
x = Flatten()(base_model.output)
x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
output = Dense(1, activation='sigmoid')(x)


# สร้างโมเดลใหม่
model_densenet = Model(inputs=base_model.input, outputs=output)


# Compile
opt = SGD(learning_rate=0.0001, momentum=0.9)
model_densenet.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# แสดงโครงสร้างโมเดล
model_densenet.summary()


csv_logger_densenet121 = CSVLogger("training_densenet121_log.csv", append=True)


early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)


# Train
history_densenet = model_densenet.fit(train_it,
                                      validation_data=test_it,
                                      epochs=1, verbose=1,
                                     callbacks=[PlotLossesKeras(), csv_logger_densenet121, early_stopping]
                                     )

# Evaluate
_, acc = model_densenet.evaluate(test_it, verbose=1)
print('DenseNet121 Accuracy: %.3f' % (acc * 100.0))


# Save Model
model_densenet.save('densenet121_model.hdf5')

