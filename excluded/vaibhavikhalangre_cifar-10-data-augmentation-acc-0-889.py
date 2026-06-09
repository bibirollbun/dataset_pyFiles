# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import cv2
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import matplotlib.pyplot as plt 
import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()



x_train[0].shape


#plotting 5x5 images from xtrain with y_train:

grid = 7 

fig, ax = plt.subplots(grid,grid,figsize=(12,12))

for i in range(grid*grid):
    imaj = x_train[i]
    imaj = cv2.cvtColor(imaj,cv2.COLOR_BGR2RGB)
    ax[i//grid][i%grid].imshow(imaj)
    ax[i//grid][i%grid].set_title(y_train[i],fontsize = 12)
    ax[i//grid][i%grid].axis('off')


print('input data:',x_train.shape)
print('val data: ',x_test.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization


#normalizing the input is necessary

x_train_norm = x_train/255
x_test_norm = x_test/255


x_train[0]


x_train_norm[0]


# if using categorical cross entropy, then i need to make the output one hot encoding
from tensorflow.keras.utils import to_categorical

print('first element of y_train',y_train[0])

y_train_norm = to_categorical(y_train,num_classes= 10)

print('first element of norm_y_train', y_train_norm[0])

y_test_norm = to_categorical(y_test,num_classes = 10)


x_train[0].shape


x_train_norm.shape[1:]


# I prefer to have frontier layers more so it captures/extract quick information from the pic such as edge, corner, shape etc. 
from tensorflow.keras.layers import ReLU

model = Sequential()

#convolution layer:
model.add(Input(shape = (x_train_norm.shape[1:]))) #32x32x3                  #input layer 

model.add(Conv2D(32,(3,3),activation = 'relu', padding = 'same',kernel_initializer='he_uniform'))      #Layer1 (early layer) 
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation = 'relu', padding='same',kernel_initializer='he_uniform'))        #Layer2 (early layer) 
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation = 'relu', padding='same',kernel_initializer='he_uniform'))        #Layer3 (early layer) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation = 'relu', padding = 'same',kernel_initializer='he_uniform'))      #Layer1 (early layer) 
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation = 'relu', padding='same',kernel_initializer='he_uniform'))        #Layer2 (early layer) 
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation = 'relu', padding='same',kernel_initializer='he_uniform'))        #Layer3 (early layer) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128,(3,3),activation = 'relu',padding='same',kernel_initializer='he_uniform'))                       #Layer4 (mid layer) 
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation = 'relu',padding='same',kernel_initializer='he_uniform'))                       #Layer5 (mid layer)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))                            
model.add(Dropout(0.3))

model.add(Conv2D(256,(3,3),activation = 'relu',padding='same',kernel_initializer='he_uniform'))                       #Layer6 (high layer)
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),activation = 'relu',padding='same',kernel_initializer='he_uniform'))                       #Layer7 (high layer)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))


model.add(Flatten())

#fully connecteed layer:
model.add(Dense(128))                                                  #Layer1(FC)
model.add(BatchNormalization())
model.add(ReLU()) 
model.add(Dropout(0.5))

model.add(Dense(64))                                                  #Layer2(FC)
model.add(BatchNormalization())
model.add(ReLU()) 
model.add(Dropout(0.5))

model.add(Dense(10,activation = 'softmax'))                            #Layer3(FC)

model.summary()


#compiling
from tensorflow.keras.callbacks import EarlyStopping

callback = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=1,
    mode="min",
    restore_best_weights=True,
)

cross_entropy = keras.losses.CategoricalCrossentropy()
adam = keras.optimizers.Adam(learning_rate = 0.00015)

model.compile(loss=cross_entropy,optimizer=adam, metrics=['accuracy'])

history = model.fit(x_train_norm,y_train_norm,
                    epochs = 100,
                    batch_size = 32,
                    validation_data = (x_test_norm,y_test_norm),
                    callbacks = [callback]
                   )


ev = model.evaluate(x_test_norm,y_test_norm)
print('Test Accuracy: {}'.format(ev[1]))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc','train_loss','val_loss'], loc='upper left')


# I prefer to have frontier layers more so it captures/extract quick information from the pic such as edge, corner, shape etc. 
from tensorflow.keras.layers import ReLU

model = Sequential()

#convolution layer:
model.add(Input(shape = (x_train.shape[1:]))) #32x32x3                  #input layer 

model.add(Conv2D(32,(3,3),activation = 'relu', padding = 'same',kernel_initializer='he_uniform'))      #Layer1 (early layer) 
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation = 'relu', padding='same',kernel_initializer='he_uniform'))        #Layer2 (early layer) 
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation = 'relu', padding='same',kernel_initializer='he_uniform'))        #Layer3 (early layer) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(64,(3,3),activation = 'relu', padding = 'same',kernel_initializer='he_uniform'))      #Layer1 (early layer) 
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation = 'relu', padding='same',kernel_initializer='he_uniform'))        #Layer2 (early layer) 
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation = 'relu', padding='same',kernel_initializer='he_uniform'))        #Layer3 (early layer) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation = 'relu',padding='same',kernel_initializer='he_uniform'))                       #Layer4 (mid layer) 
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation = 'relu',padding='same',kernel_initializer='he_uniform'))                       #Layer5 (mid layer)
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation = 'relu',padding='same',kernel_initializer='he_uniform'))                       #Layer5 (mid layer)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))                            
model.add(Dropout(0.3))

model.add(Conv2D(256,(3,3),activation = 'relu',padding='same',kernel_initializer='he_uniform'))                       #Layer6 (high layer)
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),activation = 'relu',padding='same',kernel_initializer='he_uniform'))                       #Layer7 (high layer)
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),activation = 'relu',padding='same',kernel_initializer='he_uniform'))                       #Layer7 (high layer)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))


model.add(Flatten())

#fully connecteed layer:
model.add(Dense(128))                                                  #Layer1(FC)
model.add(BatchNormalization())
model.add(ReLU()) 
model.add(Dropout(0.5))

model.add(Dense(10,activation = 'softmax'))                            #Layer3(FC)

model.summary()


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

train_datagen  = ImageDataGenerator(
    rotation_range=20,         # Random rotation in degrees
    width_shift_range=0.2,     # Random horizontal shift
    height_shift_range=0.2,    # Random vertical shift
    shear_range=0.15,          # Shear angle in counter-clockwise direction
    zoom_range=0.2,            # Random zoom
    horizontal_flip=True,      # Randomly flip images horizontally
    fill_mode='nearest',        # Fill pixels outside boundaries  
    rescale = 1./255           # Normalize pixel values to [0, 1]
)

#no augmentation
val_datagen = ImageDataGenerator(
    rescale = 1./225
) # Here, it ensure not using normalization seperately. it makes it normalized.


cross_entropy = keras.losses.CategoricalCrossentropy()
adam = keras.optimizers.Adam(learning_rate = 0.0001)

model.compile(loss=cross_entropy,optimizer=adam, metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(
                                            monitor="val_loss",
                                            #min_delta=0,
                                            patience=10,
                                            verbose=1,
                                            mode="min",
                                            #baseline=None,
                                            restore_best_weights=True,
                                            #start_from_epoch=0
                                            )

train_genetor = train_datagen.flow(x_train,y_train_norm,batch_size = 32, shuffle = True)
val_generator = val_datagen.flow(x_test, y_test_norm, batch_size=32, shuffle=False)


aug_history = model.fit(train_genetor,
                        validation_data = val_generator,
                        epochs =  100, verbose = 1, callbacks = [callback])


#no augmentation
test_datagen = ImageDataGenerator(
    rescale = 1./225
) # Here, it ensure not using normalization seperately. it makes it normalized.

test_gen = test_datagen.flow(x_test,y_test_norm,shuffle = False)

eval = model.evaluate(test_gen,verbose = 1)
# Print results
print(f"Test Loss: {eval[0]}")
print(f"Test Accuracy: {eval[1]}")


plt.plot(aug_history.history['accuracy'])
plt.plot(aug_history.history['val_accuracy'])
plt.plot(aug_history.history['loss'])
plt.plot(aug_history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc','train_loss','val_loss'], loc='upper left')


pip install py7zr


test_link = '/kaggle/input/cifar-10/test.7z'
mydir = '/kaggle/working/' 

import py7zr

archive = py7zr.SevenZipFile(test_link, mode='r')
archive.extractall(path=mydir)
archive.close()

test_list = []
test_path = '/kaggle/working/test/'

for each in os.listdir(test_path):

    test_list.append(os.path.join(test_path,each))

test_arr = [] #images in np.arr form
test_norm= [] #images in norm form [0] to [1]
test_id = [] #id column in submission excel
for link in test_list:
    test_id.append(os.path.splitext(os.path.basename(link))[0])
    imaj = cv2.imread(link)
    test_arr.append(imaj)
    test_norm.append((imaj/255.0))


#Check the test shape:
(cv2.imread(test_list[0])).shape


test_norm[0].shape


prediction_test = model.predict(test_norm) 


class_object= ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
num_object = [0,1,2,3,4,5,6,7,8,9]
num_to_class = dict(zip(num_object, class_object))
# Convert to class names
class_predictions = [num_to_class[num] for num in prediction_test]
print(len(class_predictions))


# Save to CSV (optional)
import pandas as pd
df = pd.DataFrame({'id': test_id, 'label': class_predictions})
df.to_csv('/kaggle/working/test_prediction_aug.csv', index=False)
print("Predictions saved to test_prediction_aug.csv")

