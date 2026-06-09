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
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# à¸•à¸±à¹‰à¸‡à¸„à¹ˆà¸² ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# à¹‚à¸«à¸¥à¸”à¸‚à¹‰à¸­à¸¡à¸¹à¸¥ Train à¹�à¸¥à¸° Validation
train_it = datagen.flow_from_directory(
    dataset_home + "train/",
    class_mode="binary",
    batch_size=32,  # à¸ªà¸²à¸¡à¸²à¸£à¸–à¸¥à¸”à¹€à¸«à¸¥à¸·à¸­ 16 à¸–à¹‰à¸² RAM à¹€à¸•à¹‡à¸¡
    target_size=(200, 200),
    subset="training"
)

test_it = datagen.flow_from_directory(
    dataset_home + "train/",
    class_mode="binary",
    batch_size=32,
    target_size=(200, 200),
    subset="validation"
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model


# define model
model = define_model()


model.summary()


# fit model
history = model.fit(train_it, steps_per_epoch=len(train_it),
	validation_data=test_it, validation_steps=len(test_it), epochs=1, verbose=1)


# evaluate model
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
print('> %.3f' % (acc * 100.0))


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


from keras.preprocessing.image import ImageDataGenerator

# âœ… à¹ƒà¸Šà¹‰ Data Augmentation à¹€à¸”à¸µà¸¢à¸§à¸�à¸±à¸™
datagen = ImageDataGenerator(
    featurewise_center=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# âœ… à¸•à¸±à¹‰à¸‡à¸„à¹ˆà¸² mean à¹ƒà¸«à¹‰à¸•à¸£à¸‡à¸�à¸±à¸š ImageNet
datagen.mean = [123.68, 116.779, 103.939]

# âœ… à¸Ÿà¸±à¸‡à¸�à¹Œà¸Šà¸±à¸™à¹‚à¸«à¸¥à¸” dataset (à¹ƒà¸Šà¹‰ `target_size` à¸•à¸²à¸¡à¹‚à¸¡à¹€à¸”à¸¥à¸—à¸µà¹ˆà¹€à¸£à¸µà¸¢à¸�)
def load_data(target_size):
    train_it = datagen.flow_from_directory(
        'dataset_dogs_vs_cats/train/',
        class_mode='binary', batch_size=64, target_size=target_size
    )
    test_it = datagen.flow_from_directory(
        'dataset_dogs_vs_cats/test/',
        class_mode='binary', batch_size=64, target_size=target_size
    )
    return train_it, test_it



from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# âœ… à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥ VGG16
def define_vgg16():
    model = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    # Fine-Tune à¹€à¸‰à¸�à¸²à¸° 4 à¹€à¸¥à¹€à¸¢à¸­à¸£à¹Œà¸ªà¸¸à¸”à¸—à¹‰à¸²à¸¢
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers[-4:]:
        layer.trainable = True

    # Fully Connected Layers
    flat1 = Flatten()(model.output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)

    model = Model(inputs=model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# âœ… à¹‚à¸«à¸¥à¸”à¸‚à¹‰à¸­à¸¡à¸¹à¸¥
train_it_vgg, test_it_vgg = load_data((224, 224))

# âœ… à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥ VGG16
model_vgg = define_vgg16()

# âœ… Train VGG16
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history_vgg = model_vgg.fit(
    train_it_vgg, steps_per_epoch=len(train_it_vgg),
    validation_data=test_it_vgg, validation_steps=len(test_it_vgg),
    epochs=3, verbose=1, callbacks=[early_stopping]
)

# âœ… Evaluate VGG16
_, acc_vgg = model_vgg.evaluate(test_it_vgg, steps=len(test_it_vgg), verbose=1)
print(f'\nâœ… VGG16 Accuracy: {acc_vgg * 100:.2f}%')

# âœ… Plot à¸œà¸¥à¸¥à¸±à¸�à¸˜à¹Œ
plt.figure(figsize=(10, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_vgg.history['accuracy'], label='train')
plt.plot(history_vgg.history['val_accuracy'], label='val')
plt.title('VGG16 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history_vgg.history['loss'], label='train')
plt.plot(history_vgg.history['val_loss'], label='val')
plt.title('VGG16 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



import matplotlib.pyplot as plt
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# âœ… à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥ Xception
def define_xception():
    model = Xception(include_top=False, input_shape=(299, 299, 3), weights='imagenet')

    # âœ… Fine-Tune 8 à¹€à¸¥à¹€à¸¢à¸­à¸£à¹Œà¸ªà¸¸à¸”à¸—à¹‰à¸²à¸¢ (à¹�à¸—à¸™ 4)
    for layer in model.layers[:-8]:
        layer.trainable = False
    for layer in model.layers[-8:]:
        layer.trainable = True

    # âœ… Fully Connected Layers (à¹€à¸�à¸´à¹ˆà¸¡ Batch Normalization)
    gap = GlobalAveragePooling2D()(model.output)
    gap = BatchNormalization()(gap)  # à¹€à¸�à¸´à¹ˆà¸¡ Batch Normalization à¸¥à¸” NaN
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(gap)
    output = Dense(1, activation='sigmoid')(class1)

    model = Model(inputs=model.input, outputs=output)

    # âœ… à¸¥à¸” Learning Rate à¹�à¸¥à¸°à¹ƒà¸Šà¹‰ clipvalue à¹€à¸�à¸·à¹ˆà¸­à¸¥à¸” NaN
    opt = Adam(learning_rate=0.00005, clipvalue=1.0)  # à¸¥à¸” LR à¸ˆà¸²à¸� 0.0001 â†’ 0.00005
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# âœ… à¹‚à¸«à¸¥à¸”à¸‚à¹‰à¸­à¸¡à¸¹à¸¥ (à¸¥à¸” Batch Size à¹€à¸›à¹‡à¸™ 32 à¹€à¸�à¸·à¹ˆà¸­à¸¥à¸”à¸ à¸²à¸£à¸° GPU)
train_it_xc, test_it_xc = load_data((299, 299))

# âœ… à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥ Xception
model_xc = define_xception()

# âœ… Train Xception
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history_xc = model_xc.fit(
    train_it_xc, steps_per_epoch=len(train_it_xc),
    validation_data=test_it_xc, validation_steps=len(test_it_xc),
    epochs=3, verbose=1, callbacks=[early_stopping]
)

# âœ… Evaluate Xception
_, acc_xc = model_xc.evaluate(test_it_xc, steps=len(test_it_xc), verbose=1)
print(f'\nâœ… Xception Accuracy: {acc_xc * 100:.2f}%')

# âœ… Plot à¸œà¸¥à¸¥à¸±à¸�à¸˜à¹Œ
plt.figure(figsize=(10, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_xc.history['accuracy'], label='Train')
plt.plot(history_xc.history['val_accuracy'], label='Validation')
plt.title('Xception Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history_xc.history['loss'], label='Train')
plt.plot(history_xc.history['val_loss'], label='Validation')
plt.title('Xception Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



from keras.applications.vgg19 import VGG19

# âœ… à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥ VGG19
def define_vgg19():
    model = VGG19(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    # Fine-Tune à¹€à¸‰à¸�à¸²à¸° 4 à¹€à¸¥à¹€à¸¢à¸­à¸£à¹Œà¸ªà¸¸à¸”à¸—à¹‰à¸²à¸¢
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers[-4:]:
        layer.trainable = True

    # Fully Connected Layers
    flat1 = Flatten()(model.output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)

    model = Model(inputs=model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# âœ… à¹‚à¸«à¸¥à¸”à¸‚à¹‰à¸­à¸¡à¸¹à¸¥
train_it_vgg19, test_it_vgg19 = load_data((224, 224))

# âœ… à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥ VGG19
model_vgg19 = define_vgg19()

# âœ… Train VGG19
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history_vgg19 = model_vgg19.fit(
    train_it_vgg19, steps_per_epoch=len(train_it_vgg19),
    validation_data=test_it_vgg19, validation_steps=len(test_it_vgg19),
    epochs=3, verbose=1, callbacks=[early_stopping]
)

# âœ… Evaluate VGG19
_, acc_vgg19 = model_vgg19.evaluate(test_it_vgg19, steps=len(test_it_vgg19), verbose=1)
print(f'\nâœ… VGG19 Accuracy: {acc_vgg19 * 100:.2f}%')

# âœ… Plot à¸œà¸¥à¸¥à¸±à¸�à¸˜à¹Œ
plt.figure(figsize=(10, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_vgg19.history['accuracy'], label='train')
plt.plot(history_vgg19.history['val_accuracy'], label='val')
plt.title('VGG19 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history_vgg19.history['loss'], label='train')
plt.plot(history_vgg19.history['val_loss'], label='val')
plt.title('VGG19 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# âœ… à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥ InceptionV3
def define_inceptionv3():
    model = InceptionV3(include_top=False, input_shape=(299, 299, 3), weights='imagenet')

    # Fine-Tune à¹€à¸‰à¸�à¸²à¸° 4 à¹€à¸¥à¹€à¸¢à¸­à¸£à¹Œà¸ªà¸¸à¸”à¸—à¹‰à¸²à¸¢
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers[-4:]:
        layer.trainable = True

    # Fully Connected Layers
    gap = GlobalAveragePooling2D()(model.output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(gap)
    output = Dense(1, activation='sigmoid')(class1)

    model = Model(inputs=model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# âœ… à¹‚à¸«à¸¥à¸”à¸‚à¹‰à¸­à¸¡à¸¹à¸¥
train_it_inc, test_it_inc = load_data((299, 299))

# âœ… à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥ InceptionV3
model_inc = define_inceptionv3()

# âœ… Train InceptionV3
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history_inc = model_inc.fit(
    train_it_inc, steps_per_epoch=len(train_it_inc),
    validation_data=test_it_inc, validation_steps=len(test_it_inc),
    epochs=3, verbose=1, callbacks=[early_stopping]
)

# âœ… Evaluate InceptionV3
_, acc_inc = model_inc.evaluate(test_it_inc, steps=len(test_it_inc), verbose=1)
print(f'\nâœ… InceptionV3 Accuracy: {acc_inc * 100:.2f}%')

# âœ… Plot à¸œà¸¥à¸¥à¸±à¸�à¸˜à¹Œ
plt.figure(figsize=(10, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_inc.history['accuracy'], label='train')
plt.plot(history_inc.history['val_accuracy'], label='val')
plt.title('InceptionV3 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history_inc.history['loss'], label='train')
plt.plot(history_inc.history['val_loss'], label='val')
plt.title('InceptionV3 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



import matplotlib.pyplot as plt
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# âœ… à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥ MobileNetV2
def define_mobilenetv2():
    model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    # âœ… Fine-Tune 12 à¹€à¸¥à¹€à¸¢à¸­à¸£à¹Œà¸ªà¸¸à¸”à¸—à¹‰à¸²à¸¢
    for layer in model.layers[:-12]:
        layer.trainable = False
    for layer in model.layers[-12:]:
        layer.trainable = True

    # âœ… Fully Connected Layers
    gap = GlobalAveragePooling2D()(model.output)
    gap = BatchNormalization()(gap)  # âœ… à¸›à¹‰à¸­à¸‡à¸�à¸±à¸™ Numerical Instability
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(gap)
    output = Dense(1, activation='sigmoid')(class1)

    model = Model(inputs=model.input, outputs=output)

    # âœ… à¹ƒà¸Šà¹‰ Adam Optimizer à¸�à¸£à¹‰à¸­à¸¡ Gradient Clipping
    opt = Adam(learning_rate=0.00005, clipnorm=1.0)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# âœ… à¹‚à¸«à¸¥à¸”à¸‚à¹‰à¸­à¸¡à¸¹à¸¥ (à¹ƒà¸Šà¹‰à¸‚à¸™à¸²à¸” 224x224)
train_it_mob, test_it_mob = load_data((224, 224))

# âœ… à¸ªà¸£à¹‰à¸²à¸‡à¹‚à¸¡à¹€à¸”à¸¥ MobileNetV2
model_mob = define_mobilenetv2()

# âœ… Train MobileNetV2
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history_mob = model_mob.fit(
    train_it_mob, steps_per_epoch=len(train_it_mob),
    validation_data=test_it_mob, validation_steps=len(test_it_mob),
    epochs=3, verbose=1, callbacks=[early_stopping]
)

# âœ… Evaluate MobileNetV2
_, acc_mob = model_mob.evaluate(test_it_mob, steps=len(test_it_mob), verbose=1)
print(f'\nâœ… MobileNetV2 Accuracy: {acc_mob * 100:.2f}%')

# âœ… Plot à¸œà¸¥à¸¥à¸±à¸�à¸˜à¹Œ
plt.figure(figsize=(10, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_mob.history['accuracy'], label='Train')
plt.plot(history_mob.history['val_accuracy'], label='Validation')
plt.title('MobileNetV2 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history_mob.history['loss'], label='Train')
plt.plot(history_mob.history['val_loss'], label='Validation')
plt.title('MobileNetV2 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input as preprocess_vgg
from keras.applications.xception import preprocess_input as preprocess_xc

# âœ… à¹‚à¸«à¸¥à¸”à¹‚à¸¡à¹€à¸”à¸¥à¸—à¸±à¹‰à¸‡à¸«à¸¡à¸”
models = {
    "VGG16": model_vgg,
    "VGG19": model_vgg19,
    "Xception": model_xc,
    "InceptionV3": model_inc,
    "MobileNetV2": model_mob
}

# âœ… à¸�à¸³à¸«à¸™à¸”à¸‚à¸™à¸²à¸” Input à¸‚à¸­à¸‡à¹�à¸•à¹ˆà¸¥à¸°à¹‚à¸¡à¹€à¸”à¸¥
model_input_sizes = {
    "VGG16": (224, 224),
    "VGG19": (224, 224),
    "Xception": (299, 299),
    "InceptionV3": (299, 299),
    "MobileNetV2": (224, 224)
}

# âœ… à¹€à¸¥à¸·à¸­à¸�à¸Ÿà¸±à¸‡à¸�à¹Œà¸Šà¸±à¸™ preprocess_input à¹ƒà¸«à¹‰à¸•à¸£à¸‡à¸�à¸±à¸šà¹‚à¸¡à¹€à¸”à¸¥
preprocess_functions = {
    "VGG16": preprocess_vgg,
    "VGG19": preprocess_vgg,
    "Xception": preprocess_xc,
    "InceptionV3": preprocess_xc,
    "MobileNetV2": preprocess_vgg
}

# âœ… à¸•à¸±à¹‰à¸‡à¸„à¹ˆà¸²à¸�à¸²à¸˜à¸‚à¸­à¸‡à¸Šà¸¸à¸”à¸—à¸”à¸ªà¸­à¸š
test_dir = "dataset_dogs_vs_cats/test/"
categories = ["cats", "dogs"]

# âœ… à¸ªà¸¸à¹ˆà¸¡à¹€à¸¥à¸·à¸­à¸� 5 à¸£à¸¹à¸›à¸ˆà¸²à¸�à¹�à¸•à¹ˆà¸¥à¸°à¸„à¸¥à¸²à¸ª
num_samples = 5
selected_images = {category: random.sample(os.listdir(os.path.join(test_dir, category)), num_samples) for category in categories}

# âœ… à¸šà¸±à¸™à¸—à¸¶à¸�à¸�à¸²à¸˜à¸‚à¸­à¸‡à¸ à¸²à¸�à¸—à¸µà¹ˆà¸ªà¸¸à¹ˆà¸¡à¹„à¸”à¹‰ (à¹ƒà¸Šà¹‰à¹€à¸›à¹‡à¸™ Image 1 - 10)
image_paths = []
for category in categories:
    for img_name in selected_images[category]:
        img_path = os.path.join(test_dir, category, img_name)
        image_paths.append(img_path)

# âœ… à¸ªà¸£à¹‰à¸²à¸‡ DataFrame à¸ªà¸³à¸«à¸£à¸±à¸šà¹�à¸ªà¸”à¸‡à¸œà¸¥
results = []

# âœ… à¸§à¸™à¸¥à¸¹à¸›à¸�à¸¢à¸²à¸�à¸£à¸“à¹Œà¸ˆà¸²à¸�à¸—à¸¸à¸�à¹‚à¸¡à¹€à¸”à¸¥ (à¹ƒà¸Šà¹‰ Image 1 - 10 à¹€à¸«à¸¡à¸·à¸­à¸™à¸�à¸±à¸™)
for model_name, model in models.items():
    target_size = model_input_sizes[model_name]
    preprocess_fn = preprocess_functions[model_name]

    for img_index, img_path in enumerate(image_paths, start=1):
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_fn(np.expand_dims(img_array, axis=0))

        # âœ… à¸—à¸³à¸�à¸²à¸£à¸�à¸¢à¸²à¸�à¸£à¸“à¹Œ
        prediction = model.predict(img_array)[0][0]
        predicted_label = "Cat ğŸ�±" if prediction > 0.5 else "Dog ğŸ�¶"
        confidence = max(prediction, 1 - prediction)

        # âœ… à¸•à¸£à¸§à¸ˆà¸ªà¸­à¸šà¸§à¹ˆà¸²à¸—à¸²à¸¢à¸–à¸¹à¸�à¸«à¸£à¸·à¸­à¸œà¸´à¸”
        true_label = "Cats" if "cat" in img_path else "Dogs"
        correct_label = "Cat ğŸ�±" if true_label == "Cats" else "Dog ğŸ�¶"
        is_correct = "T âœ…" if predicted_label == correct_label else f"F â�Œ"

        # âœ… à¹€à¸�à¹‡à¸šà¸œà¸¥à¸¥à¸±à¸�à¸˜à¹Œà¸¥à¸‡ DataFrame
        results.append([model_name, true_label, f"Image {img_index}", predicted_label, f"{confidence:.2f}", is_correct])

# âœ… STEP 1: à¹�à¸ªà¸”à¸‡à¸ à¸²à¸�à¸—à¸µà¹ˆà¸ªà¸¸à¹ˆà¸¡à¸¡à¸² (à¸ˆà¸±à¸”à¹€à¸£à¸µà¸¢à¸‡ 2 à¹�à¸–à¸§ à¸�à¸£à¹‰à¸­à¸¡à¸«à¸¡à¸²à¸¢à¹€à¸¥à¸‚à¸£à¸¹à¸› 1 - 10)
fig, axes = plt.subplots(2, num_samples, figsize=(10, 6))  # âœ… 2 à¹�à¸–à¸§: à¸šà¸™ = à¹�à¸¡à¸§, à¸¥à¹ˆà¸²à¸‡ = à¸«à¸¡à¸²

# ğŸ”¹ **à¹�à¸–à¸§à¸šà¸™ = à¹�à¸¡à¸§**
for i, img_path in enumerate(image_paths[:num_samples]):
    img = load_img(img_path)
    axes[0, i].imshow(img)
    axes[0, i].axis("off")
    axes[0, i].set_title(f"Cat ğŸ�± (Image {i+1})", fontsize=10)

# ğŸ”¹ **à¹�à¸–à¸§à¸¥à¹ˆà¸²à¸‡ = à¸«à¸¡à¸²**
for i, img_path in enumerate(image_paths[num_samples:]):
    img = load_img(img_path)
    axes[1, i].imshow(img)
    axes[1, i].axis("off")
    axes[1, i].set_title(f"Dog ğŸ�¶ (Image {i+num_samples+1})", fontsize=10)

plt.tight_layout()
plt.show()

# âœ… STEP 2: à¹�à¸ªà¸”à¸‡à¸œà¸¥à¸�à¸¢à¸²à¸�à¸£à¸“à¹Œà¹�à¸šà¸š DataFrame (à¸•à¸²à¸£à¸²à¸‡)
df_results = pd.DataFrame(results, columns=["Model", "True Label", "Image", "Predicted Label", "Confidence", "Correct"])
from IPython.display import display
display(df_results)


