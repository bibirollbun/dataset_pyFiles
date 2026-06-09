import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
import cv2
import os


num_of_class = 2
channels=3
image_resize=(200, 200)
resnet_pooling='avg'
dense_activation='softmax'
object_finction='categorical_crossentropy'
metricss=['accuracy']
num_of_epochs=10
early_stop=4
steps_epoch_train=10
stpes_epoch_test=10
batch_size_train=100
batch_size_validate=100
batch_size_test=1


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


resnet_weights = '/kaggle/input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


model = Sequential()

model.add(ResNet50(include_top=False, pooling=resnet_pooling, weights=resnet_weights))

model.add(Dense(num_of_class, activation=dense_activation))

model.layers[0].trainable=False


model.summary()


from tensorflow.keras.optimizers import Adam  

adam = Adam(lr=0.001)  
model.compile(optimizer=adam, loss=object_finction, metrics=metricss)


from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

image_size = image_resize

data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = data_gen.flow_from_directory('../input/catsdogs-trainvalid-80pc-prepd/trainvalidfull4keras/trainvalidfull4keras/train',
                                        target_size=image_size,
                                        batch_size=batch_size_train,
                                        class_mode='categorical')

valid_gen = data_gen.flow_from_directory('../input/catsdogs-trainvalid-80pc-prepd/trainvalidfull4keras/trainvalidfull4keras/valid',
                                        target_size=image_size,
                                        batch_size=batch_size_validate,
                                        class_mode='categorical')


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = early_stop)
cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')


history = model.fit_generator(train_gen, 
                    steps_per_epoch=steps_epoch_train,
                   epochs=num_of_epochs,
                   validation_data=valid_gen,
                   validation_steps=stpes_epoch_test,
                   callbacks=[cb_early_stopper, cb_checkpointer])


test_gen = data_gen.flow_from_directory('/kaggle/input/test-files-prepd/',
                                       target_size=image_size,
                                        batch_size=batch_size_test,
                                        class_mode=None,
                                       shuffle=False,
                                       seed=123)


test_gen.reset()

pred = model.predict_generator(test_gen, steps = len(test_gen), verbose = 1)

predicted_class_indices = np.argmax(pred, axis = 1)


test_dir = '/kaggle/input/test-files-prepd/'

f, ax = plt.subplots(5, 5, figsize=(15, 15))

start = 50
for idx, i in enumerate(range(start, start + 25)):
    if i >= len(test_gen.filenames):
        break

    imgBGR = cv2.imread(test_dir + test_gen.filenames[i])
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    
    # a if condition else b
    predicted_class = "Dog" if predicted_class_indices[i] else "Cat"

    ax[idx // 5, idx % 5].imshow(imgRGB)
    ax[idx // 5, idx % 5].axis('off')
    ax[idx // 5, idx % 5].set_title(f"Predicted: {predicted_class}")  

plt.show()

