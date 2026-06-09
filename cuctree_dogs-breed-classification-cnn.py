import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import urllib.request

import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('/kaggle/input/')
os.listdir()


# tensorflow libraries/dependencies
import tensorflow as tf
from tensorflow import keras
## preprocessing
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 

# models, layers, metrics, optimizers and callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, AveragePooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## pre-trained models
from tensorflow.keras.applications import InceptionResNetV2,InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.xception import Xception

print(f"Tensorflow Version: {tf.__version__}")
print(f"Number of GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")


# set constants
SIZE = (224,224)
BATCH_SIZE = 42
SEED = 42


labels = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
labels['id'] = labels['id'].apply(lambda x: x + '.jpg')
labels.head()


d = labels.breed.value_counts()
plt.figure(figsize=(20,5))
plt.bar(d.index,d.values,color=plt.cm.Paired([2]))
plt.xticks(rotation=90, fontsize=9)
plt.show()


# ImageDatagenerator to load the images in batches and perform data augmentation

data_generator = ImageDataGenerator(rescale= 1./255, validation_split=0.3, rotation_range=20,
                                    zoom_range=0.1, width_shift_range=0.2, height_shift_range=0.2,
                                    shear_range=0.1, horizontal_flip=True, fill_mode="nearest")

INPUT_DIR = '/kaggle/input/dog-breed-identification/train'


train_data = data_generator.flow_from_dataframe(dataframe=labels, directory=INPUT_DIR, x_col='id', y_col='breed', 
                                                target_size=SIZE, batch_size=BATCH_SIZE, seed=SEED, shuffle=True, 
                                                class_mode='categorical', subset='training', random_state=1)

val_data = data_generator.flow_from_dataframe(dataframe=labels, directory=INPUT_DIR, x_col='id', y_col='breed', 
                                              target_size=SIZE, batch_size=BATCH_SIZE, seed=SEED, shuffle=True, 
                                              class_mode='categorical', subset='validation', random_state=1)


label_mapper = np.asarray(list(train_data.class_indices.keys()))
label_mapper


os.chdir('/kaggle/working')
np.save('label_map',label_mapper)


# display data
img,label = next(train_data)
plt.subplots(3,4,figsize=(10,8))
for i in range(1,13):
    plt.subplot(3,4,i)
    plt.imshow(img[i])
    plt.axis('off')
    idx = label[i].argmax()
    plt.title(label_mapper[idx])


base_model = Xception(weights='imagenet', include_top=False, classes=120)


# base_model.trainable=False

# inputs = Input(shape = (224,224,3))
# x = base_model(inputs, training = False) 
# x = GlobalAveragePooling2D(name= "global_average_pooling")(x)
# x = Dropout(0.2)(x)
# x = Dense(120, activation="softmax")(x)

# ModelDogBreed = tf.keras.Model(inputs, x) 

# ModelDogBreed.compile(loss = "categorical_crossentropy", 
#                      optimizer = Adam(), 
#                      metrics=["accuracy"]) 

# ModelDogBreed.summary()


# model = AveragePooling2D(pool_size=(4, 4))(base_model.output)
# model = Flatten(name='flatten')(model)
# model = Dense(1024, activation='relu')(model)
# model = Dropout(0.3)(model)
# model = Dense(512, activation='relu')(model)
# model = Dropout(0.3)(model)
# model = Dense(120, activation='softmax')(model)


base_model.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Dense(120, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001, ), loss='categorical_crossentropy', metrics=["accuracy"])

model.summary()


# callbacks
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.1)
es = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=4, min_delta=0.01)
checkpoint = ModelCheckpoint(filepath='/kaggle/working/model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)


logs = model.fit(train_data, validation_data=val_data,
                steps_per_epoch = train_data.samples//BATCH_SIZE,
                validation_steps = val_data.samples//BATCH_SIZE,
                epochs=10, verbose=1, callbacks=[checkpoint,es])


plt.plot(logs.history['loss'], label='loss', linestyle='dotted')
plt.plot(logs.history['accuracy'], label='accuracy', linewidth=2)
plt.plot(logs.history['val_loss'], label='val_loss', linestyle='dotted')
plt.plot(logs.history['val_accuracy'], label='val_accuracy', linewidth=2)
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Training Logs')
plt.show()


saved_model = keras.models.load_model('/kaggle/working/model.h5')
saved_model.evaluate(val_data)


# os.chdir('/kaggle/working')
img = load_img('/kaggle/input/dog-breed-identification/test/00485d47de966a9437ad3b33ac193b6f.jpg', target_size=SIZE)
display(img)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)


pred = saved_model.predict(img_array)
idx = pred.argmax()
print(label_mapper[idx])

