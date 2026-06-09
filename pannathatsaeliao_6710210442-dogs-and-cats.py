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




!pip -q install livelossplot



from livelossplot import PlotLossesKeras



# save the final model to file
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator



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


def summarize_diagnostics(history):
    # ตัวอย่างการสร้างกราฟจากประวัติการฝึก
    import matplotlib.pyplot as plt
    
    # Plot training & validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



# load model
model = VGG16(
    include_top=False,
    input_shape=(224, 224, 3)
    )
print(model.summary())
#model = Xception(include_top=False, input_shape=(299, 299, 3))



# mark loaded layers as not trainable
for index, layer in enumerate(model.layers[:15]):
    # ตรวจสอบว่าเลเยอร์มีคุณสมบัติ output_shape ก่อนพิมพ์
    if hasattr(layer, 'output shape'):
        print(index, layer.name, layer.output_shape)
    layer.trainable = False

for index, layer in enumerate(model.layers[15:]):
    # ตรวจสอบว่าเลเยอร์มีคุณสมบัติ output_shape ก่อนพิมพ์
    if hasattr(layer, 'output shape'):
        print(index, layer.name, layer.output_shape)

    layer.trainable = True


model.layers[-1].output



from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model

# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)  # ใช้ output จากเลเยอร์สุดท้าย
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)  # ReLU และ He initializer
output = Dense(1, activation='sigmoid')(class1)  # Layer สุดท้ายสำหรับ binary classification

# define new model
my_model = Model(inputs=model.inputs, outputs=output)

# compile model
opt = SGD(learning_rate=0.0001, momentum=0.9)  # ใช้ learning_rate แทน lr
my_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# display model summary
my_model.summary()



# fit model
history = my_model.fit(
    train_it, 
    validation_data=test_it, 
    epochs=2,  # ปรับเป็นจำนวน epochs ที่ต้องการ
    verbose=1,
    callbacks=[PlotLossesKeras()]  # สามารถตัดออกหากไม่ต้องการการแสดงผลกราฟ
)

# evaluate model
_, acc = my_model.evaluate(test_it, verbose=1)  # ใช้ evaluate แทน evaluate_generator
print('> %.3f' % (acc * 100.0))

# learning curves
summarize_diagnostics(history)

# save model
best_model = 'vgg16_model.hdf5'
my_model.save(best_model)
print(f'Model saved to {best_model}')



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# โหลดโมเดลที่เทรนเสร็จแล้ว
model = load_model('vgg16_model.hdf5')  # เปลี่ยนเป็น path ของโมเดลที่เซฟไว้

# เตรียม ImageDataGenerator สำหรับ test set
test_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูลจากโฟลเดอร์ test
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

# จำนวนรูปที่ต้องการจากแต่ละคลาส
num_dogs = 0
num_cats = 0

# เลือกรูป 10 รูป โดย 5 รูปจากสุนัขและ 5 รูปจากแมว
while len(selected_images) < num_samples:
    img, label = next(test_it)  # ดึงรูปจาก test_it
    if label[0] == 1 and num_dogs < 5:  # เลือกสุนัข
        selected_images.append(img[0])  # ดึงภาพ
        selected_labels.append(label[0])  # ดึง label
        num_dogs += 1
    elif label[0] == 0 and num_cats < 5:  # เลือกแมว
        selected_images.append(img[0])  # ดึงภาพ
        selected_labels.append(label[0])  # ดึง label
        num_cats += 1

# แปลงข้อมูลภาพให้เป็น numpy array
selected_images = np.array(selected_images)

# ใช้โมเดลทำนาย
predictions = model.predict(selected_images)

# เปลี่ยนค่าผลลัพธ์จาก sigmoid ให้เป็น 0 หรือ 1
pred_labels = (predictions > 0.5).astype(int)

# แสดงผลลัพธ์
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(num_samples):
    axes[i].imshow(selected_images[i])  # แสดงรูปภาพ
    true_label = "Dog" if selected_labels[i] == 1 else "Cat"  # ป้ายจริง
    pred_label = "Dog" if pred_labels[i] == 1 else "Cat"  # ป้ายที่ทำนาย
    pred_confidence = predictions[i][0] * 100  # ความมั่นใจของการทำนาย
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({pred_confidence:.2f}%)")  # แสดงผลลัพธ์
    axes[i].axis('off')

plt.tight_layout()  # ปรับการจัดตำแหน่งภาพ
plt.show()



from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from livelossplot import PlotLossesKeras

# โหลดโมเดล Xception
model = Xception(
    include_top=False, 
    input_shape=(299, 299, 3)  # ใช้ขนาด input ที่ Xception ต้องการ
)

# พิมพ์ summary ของโมเดล Xception
print(model.summary())




# mark loaded layers as not trainable (freeze layers)
for index, layer in enumerate(model.layers[:15]):
    if hasattr(layer, 'output_shape'):
        print(index, layer.name, layer.output_shape)
    layer.trainable = False

# unfreeze the remaining layers
for index, layer in enumerate(model.layers[15:]):
    if hasattr(layer, 'output_shape'):
        print(index, layer.name, layer.output_shape)
    layer.trainable = True


# เพิ่ม new classifier layers
flat1 = Flatten()(model.layers[-1].output)  # ใช้ output จากเลเยอร์สุดท้าย
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)  # ReLU และ He initializer
output = Dense(1, activation='sigmoid')(class1)  # Layer สุดท้ายสำหรับ binary classification


# define new model
my_model = Model(inputs=model.inputs, outputs=output)

# compile model
opt = SGD(learning_rate=0.0001, momentum=0.9)  # ใช้ learning_rate แทน lr
my_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# display model summary
my_model.summary()


# เตรียม ImageDataGenerator สำหรับการฝึก
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_it = train_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/train/',  # โฟลเดอร์ train set
    class_mode='binary',
    batch_size=32,  # ปรับ batch size ตามที่ต้องการ
    target_size=(299, 299),  # ปรับขนาดให้ตรงกับที่ Xception ต้องการ
)


test_it = test_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/test/',  # โฟลเดอร์ test set
    class_mode='binary',
    batch_size=32,  # ปรับ batch size ตามที่ต้องการ
    target_size=(299, 299),  # ปรับขนาดให้ตรงกับที่ Xception ต้องการ
)


# fit model
history = my_model.fit(
    train_it, 
    validation_data=test_it, 
    epochs=3,  # ปรับเป็นจำนวน epochs ที่ต้องการ
    verbose=1,
)

# evaluate model
_, acc = my_model.evaluate(test_it, verbose=1)  # ใช้ evaluate แทน evaluate_generator
print('> %.3f' % (acc * 100.0))

# learning curves
summarize_diagnostics(history)

# save model
best_model = 'xception_model.hdf5'  # เปลี่ยนชื่อไฟล์เป็นของ Xception
my_model.save(best_model)
print(f'Model saved to {best_model}')


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# โหลดโมเดล Xception ที่ฝึกเสร็จแล้ว
model = load_model('xception_model.hdf5')  # เปลี่ยนเป็น path ของโมเดลที่เซฟไว้

# เตรียม ImageDataGenerator สำหรับ test set
test_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูลจากโฟลเดอร์ test
test_it = test_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/test/',  # โฟลเดอร์ test set
    class_mode='binary',
    batch_size=1,  # โหลดทีละ 1 รูป
    target_size=(299, 299),  # ปรับขนาดให้ตรงกับที่ Xception ต้องการ
    shuffle=True  # สุ่มลำดับของภาพ
)

# ดึงรูปจาก test set (สุ่ม 10 รูป คลาสละ 5 รูป)
num_samples = 10
selected_images = []
selected_labels = []

# จำนวนรูปที่ต้องการจากแต่ละคลาส
num_dogs = 0
num_cats = 0

# เลือกรูป 10 รูป โดย 5 รูปจากสุนัขและ 5 รูปจากแมว
while len(selected_images) < num_samples:
    img, label = next(test_it)  # ดึงรูปจาก test_it
    if label[0] == 1 and num_dogs < 5:  # เลือกสุนัข
        selected_images.append(img[0])  # ดึงภาพ
        selected_labels.append(label[0])  # ดึง label
        num_dogs += 1
    elif label[0] == 0 and num_cats < 5:  # เลือกแมว
        selected_images.append(img[0])  # ดึงภาพ
        selected_labels.append(label[0])  # ดึง label
        num_cats += 1

# แปลงข้อมูลภาพให้เป็น numpy array
selected_images = np.array(selected_images)

# ใช้โมเดลทำนาย
predictions = model.predict(selected_images)

# เปลี่ยนค่าผลลัพธ์จาก sigmoid ให้เป็น 0 หรือ 1
pred_labels = (predictions > 0.5).astype(int)

# แสดงผลลัพธ์
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(num_samples):
    axes[i].imshow(selected_images[i])  # แสดงรูปภาพ
    true_label = "Dog" if selected_labels[i] == 1 else "Cat"  # ป้ายจริง
    pred_label = "Dog" if pred_labels[i] == 1 else "Cat"  # ป้ายที่ทำนาย
    pred_confidence = predictions[i][0] * 100  # ความมั่นใจของการทำนาย
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({pred_confidence:.2f}%)")  # แสดงผลลัพธ์
    axes[i].axis('off')

plt.tight_layout()  # ปรับการจัดตำแหน่งภาพ
plt.show()



from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from livelossplot import PlotLossesKeras

# โหลดโมเดล InceptionV3
model = InceptionV3(
    include_top=False, 
    input_shape=(299, 299, 3)  # ขนาดที่โมเดล InceptionV3 ต้องการ
)

# พิมพ์ summary ของโมเดล InceptionV3
print(model.summary())



# mark loaded layers as not trainable (freeze layers)
for index, layer in enumerate(model.layers[:15]):
    if hasattr(layer, 'output_shape'):
        print(index, layer.name, layer.output_shape)
    layer.trainable = False

# unfreeze the remaining layers
for index, layer in enumerate(model.layers[15:]):
    if hasattr(layer, 'output_shape'):
        print(index, layer.name, layer.output_shape)
    layer.trainable = True



# เพิ่ม new classifier layers
flat1 = Flatten()(model.layers[-1].output)  # ใช้ output จากเลเยอร์สุดท้าย
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)  # ReLU และ He initializer
output = Dense(1, activation='sigmoid')(class1)  # Layer สุดท้ายสำหรับ binary classification


# define new model
my_model = Model(inputs=model.inputs, outputs=output)

# compile model
opt = SGD(learning_rate=0.0001, momentum=0.9)  # ใช้ learning_rate แทน lr
my_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# display model summary
my_model.summary()


# เตรียม ImageDataGenerator สำหรับการฝึก
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_it = train_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/train/',  # โฟลเดอร์ train set
    class_mode='binary',
    batch_size=32,  # ปรับ batch size ตามที่ต้องการ
    target_size=(299, 299),  # ปรับขนาดให้ตรงกับที่ InceptionV3 ต้องการ
)

test_it = test_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/test/',  # โฟลเดอร์ test set
    class_mode='binary',
    batch_size=32,  # ปรับ batch size ตามที่ต้องการ
    target_size=(299, 299),  # ปรับขนาดให้ตรงกับที่ InceptionV3 ต้องการ
)


# fit model
history = my_model.fit(
    train_it, 
    validation_data=test_it, 
    epochs=3,  # ปรับเป็นจำนวน epochs ที่ต้องการ
    verbose=1,
)

# evaluate model
_, acc = my_model.evaluate(test_it, verbose=1)  # ใช้ evaluate แทน evaluate_generator
print('> %.3f' % (acc * 100.0))

# learning curves
summarize_diagnostics(history)

# save model
best_model = 'inceptionv3_model.hdf5'  # เปลี่ยนชื่อไฟล์เป็นของ InceptionV3
my_model.save(best_model)
print(f'Model saved to {best_model}')


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# โหลดโมเดล InceptionV3 ที่ฝึกเสร็จแล้ว
model = load_model('inceptionv3_model.hdf5')  # เปลี่ยนเป็น path ของโมเดลที่เซฟไว้

# เตรียม ImageDataGenerator สำหรับ test set
test_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูลจากโฟลเดอร์ test
test_it = test_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/test/',  # โฟลเดอร์ test set
    class_mode='binary',
    batch_size=1,  # โหลดทีละ 1 รูป
    target_size=(299, 299),  # ปรับขนาดให้ตรงกับที่ InceptionV3 ต้องการ
    shuffle=True  # สุ่มลำดับของภาพ
)

# ดึงรูปจาก test set (สุ่ม 10 รูป คลาสละ 5 รูป)
num_samples = 10
selected_images = []
selected_labels = []

# จำนวนรูปที่ต้องการจากแต่ละคลาส
num_dogs = 0
num_cats = 0

# เลือกรูป 10 รูป โดย 5 รูปจากสุนัขและ 5 รูปจากแมว
while len(selected_images) < num_samples:
    img, label = next(test_it)  # ดึงรูปจาก test_it
    if label[0] == 1 and num_dogs < 5:  # เลือกสุนัข
        selected_images.append(img[0])  # ดึงภาพ
        selected_labels.append(label[0])  # ดึง label
        num_dogs += 1
    elif label[0] == 0 and num_cats < 5:  # เลือกแมว
        selected_images.append(img[0])  # ดึงภาพ
        selected_labels.append(label[0])  # ดึง label
        num_cats += 1

# แปลงข้อมูลภาพให้เป็น numpy array
selected_images = np.array(selected_images)

# ใช้โมเดลทำนาย
predictions = model.predict(selected_images)

# เปลี่ยนค่าผลลัพธ์จาก sigmoid ให้เป็น 0 หรือ 1
pred_labels = (predictions > 0.5).astype(int)

# แสดงผลลัพธ์
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(num_samples):
    axes[i].imshow(selected_images[i])  # แสดงรูปภาพ
    true_label = "Dog" if selected_labels[i] == 1 else "Cat"  # ป้ายจริง
    pred_label = "Dog" if pred_labels[i] == 1 else "Cat"  # ป้ายที่ทำนาย
    pred_confidence = predictions[i][0] * 100  # ความมั่นใจของการทำนาย
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({pred_confidence:.2f}%)")  # แสดงผลลัพธ์
    axes[i].axis('off')

plt.tight_layout()  # ปรับการจัดตำแหน่งภาพ
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# โหลด MobileNetV2 model โดยไม่รวม top layers
base_model = MobileNetV2(
    include_top=False,  # ลบ fully connected layer ที่ใช้ในการจำแนก
    input_shape=(224, 224, 3)  # ขนาดที่โมเดลต้องการ
)


# ทำให้ layer ของ base model ไม่สามารถฝึกได้ (freeze)
for layer in base_model.layers:
    layer.trainable = False

# เพิ่ม classifier layer สำหรับโมเดลที่ใช้กับงาน binary classification
x = Flatten()(base_model.output)  # แปลงข้อมูลเป็น 1D
x = Dense(128, activation='relu')(x)  # Dense layer แรก
x = Dense(1, activation='sigmoid')(x)  # สุดท้ายเป็น sigmoid สำหรับ binary classification (แมวหรือสุนัข)



# สร้างโมเดลใหม่โดยใช้ base model และ classifier ที่เราเพิ่ม
model = Model(inputs=base_model.input, outputs=x)

# คอมไพล์โมเดล
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])



# เตรียม ImageDataGenerator สำหรับ train และ test sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูลการฝึกและการทดสอบ
train_it = train_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/train/',  # เปลี่ยนเป็น path ของ train set
    class_mode='binary',
    batch_size=32,
    target_size=(224, 224)  # ขนาดที่โมเดลต้องการ
)

test_it = test_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/test/',  # เปลี่ยนเป็น path ของ test set
    class_mode='binary',
    batch_size=32,
    target_size=(224, 224)  # ขนาดที่โมเดลต้องการ
)


# ฝึกโมเดล
history = model.fit(
    train_it,
    validation_data=test_it,
    epochs=10,  # ปรับจำนวน epochs ตามที่ต้องการ
    verbose=1
)

# ประเมินโมเดล
_, acc = model.evaluate(test_it, verbose=1)
print(f'Model Accuracy: {acc * 100:.2f}%')

# แสดงผลการฝึก
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# บันทึกโมเดลที่ฝึกเสร็จแล้ว
model.save('mobilenetv2_model.hdf5')
print("Model saved to 'mobilenetv2_model.hdf5'")


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2

# โหลดโมเดล MobileNetV2 ที่ฝึกเสร็จแล้ว
model = load_model('mobilenetv2_model.hdf5')  # เปลี่ยนเป็น path ของโมเดลที่เซฟไว้

# เตรียม ImageDataGenerator สำหรับ test set
test_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูลจากโฟลเดอร์ test
test_it = test_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/test/',  # โฟลเดอร์ test set
    class_mode='binary',
    batch_size=1,  # โหลดทีละ 1 รูป
    target_size=(224, 224),  # ปรับขนาดให้ตรงกับที่ MobileNetV2 ต้องการ
    shuffle=True  # สุ่มลำดับของภาพ
)

# ดึงรูปจาก test set (สุ่ม 10 รูป คลาสละ 5 รูป)
num_samples = 10
selected_images = []
selected_labels = []

# จำนวนรูปที่ต้องการจากแต่ละคลาส
num_dogs = 0
num_cats = 0

# เลือกรูป 10 รูป โดย 5 รูปจากสุนัขและ 5 รูปจากแมว
while len(selected_images) < num_samples:
    img, label = next(test_it)  # ดึงรูปจาก test_it
    if label[0] == 1 and num_dogs < 5:  # เลือกสุนัข
        selected_images.append(img[0])  # ดึงภาพ
        selected_labels.append(label[0])  # ดึง label
        num_dogs += 1
    elif label[0] == 0 and num_cats < 5:  # เลือกแมว
        selected_images.append(img[0])  # ดึงภาพ
        selected_labels.append(label[0])  # ดึง label
        num_cats += 1

# แปลงข้อมูลภาพให้เป็น numpy array
selected_images = np.array(selected_images)

# ใช้โมเดลทำนาย
predictions = model.predict(selected_images)

# เปลี่ยนค่าผลลัพธ์จาก sigmoid ให้เป็น 0 หรือ 1
pred_labels = (predictions > 0.5).astype(int)

# แสดงผลลัพธ์
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(num_samples):
    axes[i].imshow(selected_images[i])  # แสดงรูปภาพ
    true_label = "Dog" if selected_labels[i] == 1 else "Cat"  # ป้ายจริง
    pred_label = "Dog" if pred_labels[i] == 1 else "Cat"  # ป้ายที่ทำนาย
    pred_confidence = predictions[i][0] * 100  # ความมั่นใจของการทำนาย
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({pred_confidence:.2f}%)")  # แสดงผลลัพธ์
    axes[i].axis('off')

plt.tight_layout()  # ปรับการจัดตำแหน่งภาพ
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# โหลดโมเดล DenseNet121 โดยไม่รวม top layers (จะใช้ classifier ของตัวเอง)
base_model = DenseNet121(
    include_top=False,  # ลบ fully connected layers
    input_shape=(224, 224, 3)  # ขนาดที่โมเดลต้องการ
)


# ทำให้ layer ของ base model ไม่สามารถฝึกได้ (freeze)
for layer in base_model.layers:
    layer.trainable = False

# เพิ่ม classifier layer สำหรับโมเดลที่ใช้กับงาน binary classification
x = Flatten()(base_model.output)  # แปลงข้อมูลเป็น 1D
x = Dense(128, activation='relu')(x)  # Dense layer แรก
x = Dense(1, activation='sigmoid')(x)  # สุดท้ายเป็น sigmoid สำหรับ binary classification (แมวหรือสุนัข)


# สร้างโมเดลใหม่โดยใช้ base model และ classifier ที่เราเพิ่ม
model = Model(inputs=base_model.input, outputs=x)

# คอมไพล์โมเดล
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])



# เตรียม ImageDataGenerator สำหรับ train และ test sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูลการฝึกและการทดสอบ
train_it = train_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/train/',  # เปลี่ยนเป็น path ของ train set
    class_mode='binary',
    batch_size=32,
    target_size=(224, 224)  # ขนาดที่โมเดลต้องการ
)

test_it = test_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/test/',  # เปลี่ยนเป็น path ของ test set
    class_mode='binary',
    batch_size=32,
    target_size=(224, 224)  # ขนาดที่โมเดลต้องการ
)


# ฝึกโมเดล
history = model.fit(
    train_it,
    validation_data=test_it,
    epochs=10,  # ปรับจำนวน epochs ตามที่ต้องการ
    verbose=1
)

# ประเมินโมเดล
_, acc = model.evaluate(test_it, verbose=1)
print(f'Model Accuracy: {acc * 100:.2f}%')

# แสดงผลการฝึก
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# บันทึกโมเดลที่ฝึกเสร็จแล้ว
model.save('densenet121_model.hdf5')
print("Model saved to 'densenet121_model.hdf5'")


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import DenseNet121  # เปลี่ยนจาก MobileNetV2 เป็น DenseNet121

# โหลดโมเดล DenseNet121 ที่ฝึกเสร็จแล้ว
model = load_model('densenet121_model.hdf5')  # เปลี่ยนเป็น path ของโมเดลที่เซฟไว้

# เตรียม ImageDataGenerator สำหรับ test set
test_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูลจากโฟลเดอร์ test
test_it = test_datagen.flow_from_directory(
    'dataset_dogs_vs_cats/test/',  # โฟลเดอร์ test set
    class_mode='binary',
    batch_size=1,  # โหลดทีละ 1 รูป
    target_size=(224, 224),  # ปรับขนาดให้ตรงกับที่ DenseNet121 ต้องการ
    shuffle=True  # สุ่มลำดับของภาพ
)

# ดึงรูปจาก test set (สุ่ม 10 รูป คลาสละ 5 รูป)
num_samples = 10
selected_images = []
selected_labels = []

# จำนวนรูปที่ต้องการจากแต่ละคลาส
num_dogs = 0
num_cats = 0

# เลือกรูป 10 รูป โดย 5 รูปจากสุนัขและ 5 รูปจากแมว
while len(selected_images) < num_samples:
    img, label = next(test_it)  # ดึงรูปจาก test_it
    if label[0] == 1 and num_dogs < 5:  # เลือกสุนัข
        selected_images.append(img[0])  # ดึงภาพ
        selected_labels.append(label[0])  # ดึง label
        num_dogs += 1
    elif label[0] == 0 and num_cats < 5:  # เลือกแมว
        selected_images.append(img[0])  # ดึงภาพ
        selected_labels.append(label[0])  # ดึง label
        num_cats += 1

# แปลงข้อมูลภาพให้เป็น numpy array
selected_images = np.array(selected_images)

# ใช้โมเดลทำนาย
predictions = model.predict(selected_images)

# เปลี่ยนค่าผลลัพธ์จาก sigmoid ให้เป็น 0 หรือ 1
pred_labels = (predictions > 0.5).astype(int)

# แสดงผลลัพธ์
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(num_samples):
    axes[i].imshow(selected_images[i])  # แสดงรูปภาพ
    true_label = "Dog" if selected_labels[i] == 1 else "Cat"  # ป้ายจริง
    pred_label = "Dog" if pred_labels[i] == 1 else "Cat"  # ป้ายที่ทำนาย
    pred_confidence = predictions[i][0] * 100  # ความมั่นใจของการทำนาย
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({pred_confidence:.2f}%)")  # แสดงผลลัพธ์
    axes[i].axis('off')

plt.tight_layout()  # ปรับการจัดตำแหน่งภาพ
plt.show()





