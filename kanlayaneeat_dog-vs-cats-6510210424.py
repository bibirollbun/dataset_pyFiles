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


# save the final model to file
from tensorflow.keras.applications import InceptionV3
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # แบ่งข้อมูล 20% สำหรับ validation
)

train_it = datagen.flow_from_directory(
    'dataset_dogs_vs_cats/train/',
    class_mode='binary',
    batch_size=64,
    target_size=(224, 224),
    subset='training'
)

val_it = datagen.flow_from_directory(
    'dataset_dogs_vs_cats/train/',
    class_mode='binary',
    batch_size=64,
    target_size=(224, 224),
    subset='validation'
)



# load model
model = InceptionV3(
    include_top=False,
    input_shape=(224, 224, 3)
    )
print(model.summary())
#model = Xception(include_top=False, input_shape=(299, 299, 3))




import keras 
from keras import layers
import tensorflow as tf
from tensorflow import data as tf_data
import matplotlib.pyplot as plt


# Assuming 'model' is already defined and compiled

# Mark loaded layers as not trainable
for index, layer in enumerate(model.layers[:15]):
    # Check if it's not an InputLayer
    if not isinstance(layer, tf.keras.layers.InputLayer):  
        # Ensure that the model is built before accessing output_shape
        print(index, layer.name, layer.output.shape)  # Use .output.shape instead of .output_shape
        layer.trainable = False

for index, layer in enumerate(model.layers[15:]):
    print(index, layer.name, layer.output.shape)  # Use .output.shape here as well
    layer.trainable = True




model.layers[-1].output


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# เพิ่ม Fully Connected Layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(512, activation='relu', kernel_initializer='he_uniform')(flat1)
class1 = BatchNormalization()(class1)  # ช่วยให้ Training เสถียรขึ้น
class1 = Dropout(0.5)(class1)  # ลด Overfitting

class2 = Dense(256, activation='relu', kernel_initializer='he_uniform')(class1)
class2 = BatchNormalization()(class2)
class2 = Dropout(0.5)(class2)

output = Dense(1, activation='sigmoid')(class2)  # Output สำหรับ Binary Classification

# สร้างโมเดลใหม่
my_model = Model(inputs=model.inputs, outputs=output)

# ใช้ Adam Optimizer แทน SGD
opt = Adam(learning_rate=0.0001)
my_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

my_model.summary()



# Fit model (ลบบรรทัด reset_states() หากไม่มี stateful RNN)
history = my_model.fit(
    train_it, 
    validation_data=val_it, 
    epochs=20, 
    verbose=1,
    callbacks=[PlotLossesKeras()]
)

# Evaluate model
_, acc = my_model.evaluate(val_it, verbose=1)  # ใช้ evaluate() แทน evaluate_generator()
print(f'> Accuracy: {acc * 100:.3f}%')

# Save model
best_model = 'InceptionV3.keras'
my_model.save(best_model)



from tensorflow.keras.models import load_model
import os

# บันทึกโมเดล
save_path = 'InceptionV3.keras'
my_model.save(save_path)

# ตรวจสอบว่าไฟล์ถูกบันทึกหรือไม่
if os.path.exists(save_path):
    print("Model saved successfully!")

# โหลดโมเดล
best_model = load_model(save_path)

# Evaluate โมเดล
best_model.evaluate(val_it, verbose=1)



# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# โหลดโมเดลที่เทรนเสร็จแล้ว
model = load_model('InceptionV3.keras')  # เปลี่ยนเป็น path ของโมเดลที่เซฟไว้

# เตรียม ImageDataGenerator สำหรับ test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_it = test_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/test/',  # โฟลเดอร์ test set
    class_mode='binary',
    batch_size=1,  # โหลดทีละ 1 รูป
    target_size=(224, 224),
    shuffle=True  # สุ่มลำดับของภาพ
)

# ดึงรูปจาก test set (สุ่ม 10 รูป คลาสละ 5 รูป)
num_samples = 10
selected_images = []
selected_labels = []

while len(selected_images) < num_samples:
    img, label = next(val_it)  # ใช้ next() หรือ test_it.__next__()
    if sum(selected_labels) < 5 or label[0] == 0:  # คุมให้ได้ 5 รูปต่อคลาส
        selected_images.append(img[0])  # ดึงภาพ
        selected_labels.append(label[0])  # ดึง label


selected_images = np.array(selected_images)

# ใช้โมเดล predict
predictions = model.predict(selected_images)
pred_labels = (predictions > 0.5).astype(int)  # แปลงค่า sigmoid เป็น 0/1

# แสดงผลลัพธ์
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(num_samples):
    axes[i].imshow(selected_images[i])
    true_label = "Dog" if selected_labels[i] == 1 else "Cat"
    pred_label = "Dog" if pred_labels[i] == 1 else "Cat"
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()











