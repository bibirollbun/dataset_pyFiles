#Histopathological Cancer Detection Project
#Copy & Edit from JWO HistoCancerDetect img-sizes select v1, which is from JWO HistoCancerDetect EDA select v1, which is from MWV Final Project Training Model 5


# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()




# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

histopathologic_cancer_detection_path = kagglehub.competition_download('histopathologic-cancer-detection')

print('Data source import complete.')



print(histopathologic_cancer_detection_path)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, datasets


train = pd.read_csv(f'{histopathologic_cancer_detection_path}/train_labels.csv')


train.isnull().sum().T


(train.label.value_counts()/len(train.label)).to_frame().T


#Challenge: The image ids are used as filenames for the images, but the ids are missing the ".tif" extension.
#You will need to add a copy to the DataFrame to store the complete filename rather than just the id
train['filenames'] = train['id']+'.tif'


train.head()


SS = 50000
RS = 10

positives = train[train['label']==1].sample(SS, random_state = RS)
negatives = train[train['label']==0].sample(SS, random_state = SS)

new_train = pd.concat([positives,negatives], axis = 0).reset_index(drop = True)
new_train = shuffle(new_train)


(new_train.label.value_counts()/len(new_train)).to_frame().T


new_train = new_train.sample(frac=0.5)  # use only a fraction of the dataset


train_df, val_df = train_test_split(new_train, test_size = .2, random_state = 10, stratify = new_train.label)

print(train_df.shape)
print(val_df.shape)


train_images_path = f'{histopathologic_cancer_detection_path}/train'


#Challenge: You will need to use an image data generator to load the files from disk.
train_datagen = ImageDataGenerator(rescale = 1/255)
val_datagen = ImageDataGenerator(rescale = 1/255)


train_df['label'] = train_df['label'].astype(str)
val_df['label'] = val_df['label'].astype(str)


%%time
batch_size = 64

train_datagen = ImageDataGenerator(rescale = 1/255)
val_datagen = ImageDataGenerator(rescale = 1/255)

print('Creating training data loader for images with no data augmentation\n')
train_loadern = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory = train_images_path,
    x_col = 'filenames',
    y_col = 'label',
    batch_size = batch_size,
    seed = 10,
    shuffle = True,
    class_mode = 'binary',
    horizontal_flip = False,
    vertical_flip = False,
    height_shift_range = 0,
    width_shift_range = 0,
    rotation_range = 0,
    target_size = (64,64),
    color_mode = 'grayscale'
)

print('Creating training data loader for images with only flips as data augmentation\n')
train_loaderf = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory = train_images_path,
    x_col = 'filenames',
    y_col = 'label',
    batch_size = batch_size,
    seed = 10,
    shuffle = True,
    class_mode = 'binary',
    horizontal_flip = True,
    vertical_flip = True,
    height_shift_range = 0,
    width_shift_range = 0,
    rotation_range = 0,
    target_size = (64,64),
    color_mode = 'grayscale'
)

print('Creating training data loader for images with only shifts as data augmentation\n')
train_loadert = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory = train_images_path,
    x_col = 'filenames',
    y_col = 'label',
    batch_size = batch_size,
    seed = 10,
    shuffle = True,
    class_mode = 'binary',
    horizontal_flip = False,
    vertical_flip = False,
    height_shift_range = 0.12,
    width_shift_range = 0.12,
    rotation_range = 0,
    target_size = (64,64),
    color_mode = 'grayscale'
)

print('Creating training data loader for images with only rotation as data augmentation\n')
train_loaderr = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory = train_images_path,
    x_col = 'filenames',
    y_col = 'label',
    batch_size = batch_size,
    seed = 10,
    shuffle = True,
    class_mode = 'binary',
    horizontal_flip = False,
    vertical_flip = False,
    height_shift_range = 0,
    width_shift_range = 0,
    rotation_range = 21,
    target_size = (64,64),
    color_mode = 'grayscale'
)

print('Creating validation data loader\n')
val_loader = val_datagen.flow_from_dataframe(
    dataframe = val_df,
    directory = train_images_path,
    x_col = 'filenames',
    y_col = 'label',
    batch_size = batch_size,
    seed = 10,
    shuffle = True,
    class_mode = 'binary',
    target_size = (64,64),
    color_mode = 'grayscale'
)


TR_STEPSN = len(train_loadern)
TR_STEPSF = len(train_loaderf)
TR_STEPST = len(train_loadert)
TR_STEPSR = len(train_loaderr)
VAL_STEPS = len(val_loader)

print(TR_STEPSN)
print(TR_STEPSF)
print(TR_STEPST)
print(TR_STEPSR)
print(VAL_STEPS)


# use same batch size, epochs for all models
train_batch_size = 64
num_epochs = 10


# use same callback definitions for all models
from tensorflow import keras

early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_AUC',
    patience=10,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

lr_scheduler_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_AUC',
    factor=0.5,
    patience=5,
    min_lr=1e-8,
    mode='max',
    verbose=1
)

callbacks_list = [early_stopping_callback, lr_scheduler_callback]


modeln = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])


modeln.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['AUC'])


%%time

historyn = modeln.fit(
    x = train_loadern,
    steps_per_epoch = TR_STEPSN,
    epochs = num_epochs,
    validation_data = val_loader,
    validation_steps = VAL_STEPS,
    verbose = 1,
    callbacks = callbacks_list
)


lossn, aucn = modeln.evaluate(val_loader, verbose=0)
print(f"No data aug Test AUC: {aucn:.4f}")


# Predict on new data
# Get a sample batch from the validation loader
sample_batch = next(iter(val_loader))
sample_images, sample_labels = sample_batch
sample = sample_images[0:1] # Take the first image in the batch

predictionn = modeln.predict(sample)
print(f"No data aug predicted probability of class 1: {predictionn[0][0]:.4f}")


modelf = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])


modelf.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['AUC'])


%%time

historyf = modelf.fit(
    x = train_loaderf,
    steps_per_epoch = TR_STEPSF,
    epochs = num_epochs,
    validation_data = val_loader,
    validation_steps = VAL_STEPS,
    verbose = 1,
    callbacks = callbacks_list
)


lossf, aucf = modelf.evaluate(val_loader, verbose=0)
print(f"Flips only Test AUC: {aucf:.4f}")


# Predict on new data
predictionf = modelf.predict(sample)
print(f"Flips only predicted probability of class 1: {predictionf[0][0]:.4f}")


modelt = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])


modelt.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['AUC'])


%%time

historyt = modelt.fit(
    x = train_loadert,
    steps_per_epoch = TR_STEPST,
    epochs = num_epochs,
    validation_data = val_loader,
    validation_steps = VAL_STEPS,
    verbose = 1,
    callbacks = callbacks_list
)


loss, auct = modelt.evaluate(val_loader, verbose=0)
print(f"Shifts only Test AUC: {auct:.4f}")


# Predict on new data
predictiont = modelt.predict(sample)
print(f"Shifts only predicted probability of class 1: {predictiont[0][0]:.4f}")


modelr = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])


modelr.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['AUC'])


%%time

historyr = modelr.fit(
    x = train_loaderr,
    steps_per_epoch = TR_STEPSR,
    epochs = num_epochs,
    validation_data = val_loader,
    validation_steps = VAL_STEPS,
    verbose = 1,
    callbacks = callbacks_list
)


lossr, aucr = modelr.evaluate(val_loader, verbose=0)
print(f"Rotations only Test AUC: {aucr:.4f}")


# Predict on new data
predictionr = modelr.predict(sample)
print(f"Rotations only predicted probability of class 1: {predictionr[0][0]:.4f}")


epoch_range = range(1, len(historyn.history['loss'])+1)
plt.figure(figsize = [12,5]); plt.subplot(1,2,1)
plt.plot(epoch_range, historyn.history['loss'], label = 'Training')
plt.plot(epoch_range, historyn.history['val_loss'], label = 'Validation')
plt.xlabel('Epoch');plt.ylabel('Loss');plt.title("Loss No Data Aug")

plt.subplot(1,2,2)
plt.plot(epoch_range, historyn.history['AUC'], label = 'Training')
plt.plot(epoch_range, historyn.history['val_AUC'], label = 'Validation')
plt.xlabel("Epoch");plt.ylabel("AUC");plt.title("AUC No Data Aug")
plt.legend()
plt.show()

epoch_range = range(1, len(historyf.history['loss'])+1)
plt.figure(figsize = [12,5]); plt.subplot(1,2,1)
plt.plot(epoch_range, historyf.history['loss'], label = 'Training')
plt.plot(epoch_range, historyf.history['val_loss'], label = 'Validation')
plt.xlabel('Epoch');plt.ylabel('Loss');plt.title("Loss Flips Only")

plt.subplot(1,2,2)
plt.plot(epoch_range, historyf.history['AUC'], label = 'Training')
plt.plot(epoch_range, historyf.history['val_AUC'], label = 'Validation')
plt.xlabel("Epoch");plt.ylabel("AUC");plt.title("AUC Flips Only")
plt.legend()
plt.show()

epoch_range = range(1, len(historyt.history['loss'])+1)
plt.figure(figsize = [12,5]); plt.subplot(1,2,1)
plt.plot(epoch_range, historyt.history['loss'], label = 'Training')
plt.plot(epoch_range, historyt.history['val_loss'], label = 'Validation')
plt.xlabel('Epoch');plt.ylabel('Loss');plt.title("Loss Shifts Only")

plt.subplot(1,2,2)
plt.plot(epoch_range, historyt.history['AUC'], label = 'Training')
plt.plot(epoch_range, historyt.history['val_AUC'], label = 'Validation')
plt.xlabel("Epoch");plt.ylabel("AUC");plt.title("AUC Shifts Only")
plt.legend()
plt.show()

epoch_range = range(1, len(historyr.history['loss'])+1)
plt.figure(figsize = [12,5]); plt.subplot(1,2,1)
plt.plot(epoch_range, historyr.history['loss'], label = 'Training')
plt.plot(epoch_range, historyr.history['val_loss'], label = 'Validation')
plt.xlabel('Epoch');plt.ylabel('Loss');plt.title("Loss Rotations Only")

plt.subplot(1,2,2)
plt.plot(epoch_range, historyr.history['AUC'], label = 'Training')
plt.plot(epoch_range, historyr.history['val_AUC'], label = 'Validation')
plt.xlabel("Epoch");plt.ylabel("AUC");plt.title("AUC Rotations Only")
plt.legend()
plt.show()


# save models in Google Drive
from google.colab import drive
drive.mount('/content/drive')
model_save_path = '/content/drive/MyDrive/Cancer_Detection_Models/'


import pickle
import os

# Create the directory if it doesn't exist
os.makedirs(model_save_path, exist_ok=True)

modeln.save(f'{model_save_path}/Cancer_Detection_model_n.keras')
pickle.dump(historyn.history, open(f'{model_save_path}/Cancer_Detection_model_n.pk1', 'wb'))

modelf.save(f'{model_save_path}/Cancer_Detection_model_f.keras')
pickle.dump(historyf.history, open(f'{model_save_path}/Cancer_Detection_model_f.pk1', 'wb'))

modelt.save(f'{model_save_path}/Cancer_Detection_model_t.keras')
pickle.dump(historyt.history, open(f'{model_save_path}/Cancer_Detection_model_t.pk1', 'wb'))

modelr.save(f'{model_save_path}/Cancer_Detection_model_r.keras')
pickle.dump(historyr.history, open(f'{model_save_path}/Cancer_Detection_model_r.pk1', 'wb'))




