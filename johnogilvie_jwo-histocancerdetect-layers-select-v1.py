#Histopathological Cancer Detection Project
#Copy & Edit from JWO HistoCancerDetect EDA transfer v1, which is from JWO HistoCancerDetect EDA select v1, which is from MWV Final Project Training Model 5


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
from tensorflow.keras.activations import swish
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


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


# new_train = new_train.sample(frac=0.05)  # use only a fraction of the dataset


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
batch_size = 32

train_loader = train_datagen.flow_from_dataframe(
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
    height_shift_range = .12,
    width_shift_range = .12,
    rotation_range = 18,
    target_size = (128,128)
)

val_loader = val_datagen.flow_from_dataframe(
    dataframe = val_df,
    directory = train_images_path,
    x_col = 'filenames',
    y_col = 'label',
    batch_size = batch_size,
    seed = 10,
    shuffle = True,
    class_mode = 'binary',
    target_size = (128,128)
)


TR_STEPS = len(train_loader)
VAL_STEPS = len(val_loader)

print(TR_STEPS)
print(VAL_STEPS)


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


# Use same number of epochs in each model
num_epochs = 10


# Define image dimensions and number of channels
img_width, img_height = 128, 128
input_shape = (img_width, img_height, 3) # 3 channels for color images


def display_perf(history, loss_title, auc_title):
  epoch_range = range(1, len(history['loss'])+1)
  plt.figure(figsize = [12,5])
  plt.subplot(1,2,1)
  plt.plot(epoch_range, history['loss'], label = 'Training')
  plt.plot(epoch_range, history['val_loss'], label = 'Validation')
  plt.xlabel('Epoch');plt.ylabel('Loss');plt.title(loss_title)

  plt.subplot(1,2,2)
  plt.plot(epoch_range, history['AUC'], label = 'Training')
  plt.plot(epoch_range, history['val_AUC'], label = 'Validation')
  plt.xlabel("Epoch");plt.ylabel("AUC");plt.title(auc_title)
  plt.legend()
  plt.show()



# Create a Sequential model
model_onelay = Sequential()

# Add a single Convolutional layer
# 32 filters, 3x3 kernel size, ReLU activation, specify input shape
model_onelay.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

# Add a Max Pooling layer
# 2x2 pool size to reduce spatial dimensions
model_onelay.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output of the pooling layer to feed into a dense layer
model_onelay.add(Flatten())

# Add a Dense output layer for binary classification
# 1 neuron with sigmoid activation for probability output (0 to 1)
model_onelay.add(Dense(1, activation='sigmoid'))

# Print a summary of the model architecture
model_onelay.summary()


opt = tf.keras.optimizers.Adam(learning_rate = 1e-5)
model_onelay.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['AUC'])


%%time
history_onelay = model_onelay.fit(
    x = train_loader,
    steps_per_epoch = TR_STEPS,
    epochs = num_epochs,
    validation_data = val_loader,
    validation_steps = VAL_STEPS,
    verbose = 1,
    callbacks = callbacks_list
)


lossone, aucone = model_onelay.evaluate(val_loader, verbose=0)
print(f"Model One Test AUC: {aucone:.4f}")


# Get a sample batch from the validation loader
sample_batch = next(iter(val_loader))
sample_images, sample_labels = sample_batch
sample = sample_images[0:1] # Take the first image in the batch


# Predict on new data

predictionone = model_onelay.predict(sample)
print(f"Model One predicted probability of class 1: {predictionone[0][0]:.4f}")


# Create a Sequential model
model_twolay = Sequential()

# Add a first Convolutional layer
# 32 filters, 3x3 kernel size, ReLU activation, specify input shape
model_twolay.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

# Add a Max Pooling layer
# 2x2 pool size to reduce spatial dimensions
model_twolay.add(MaxPooling2D(pool_size=(2, 2)))

# Add a second Convolutional Layer
model_twolay.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_twolay.add(layers.MaxPooling2D((2, 2)))

# Flatten the output of the pooling layer to feed into a dense layer
model_twolay.add(Flatten())

# Add a Dense output layer for binary classification
# 1 neuron with sigmoid activation for probability output (0 to 1)
model_twolay.add(Dense(1, activation='sigmoid'))

# Print a summary of the model architecture
model_twolay.summary()


opt = tf.keras.optimizers.Adam(learning_rate = 1e-5)
model_twolay.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['AUC'])


%%time
history_twolay = model_twolay.fit(
    x = train_loader,
    steps_per_epoch = TR_STEPS,
    epochs = num_epochs,
    validation_data = val_loader,
    validation_steps = VAL_STEPS,
    verbose = 1,
    callbacks = callbacks_list
)


losstwo, auctwo = model_twolay.evaluate(val_loader, verbose=0)
print(f"Model Two Test AUC: {auctwo:.4f}")


# Predict on new data

predictiontwo = model_twolay.predict(sample)
print(f"Model Two predicted probability of class 1: {predictiontwo[0][0]:.4f}")


# Create a Sequential model
model_threelay = Sequential()

# Add a first Convolutional layer
# 16 filters, 3x3 kernel size, ReLU activation, specify input shape
model_threelay.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))

# Add a Max Pooling layer
# 2x2 pool size to reduce spatial dimensions
model_threelay.add(MaxPooling2D(pool_size=(2, 2)))

# Add a second Convolutional layer
# 32 filters, 3x3 kernel size, ReLU activation, specify input shape
model_threelay.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model_threelay.add(MaxPooling2D(pool_size=(2, 2)))

# Add a third Convolutional Layer
model_threelay.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_threelay.add(layers.MaxPooling2D((2, 2)))

# Flatten the output of the pooling layer to feed into a dense layer
model_threelay.add(Flatten())

# Add a Dense output layer for binary classification
# 1 neuron with sigmoid activation for probability output (0 to 1)
model_threelay.add(Dense(1, activation='sigmoid'))

# Print a summary of the model architecture
model_threelay.summary()


opt = tf.keras.optimizers.Adam(learning_rate = 1e-5)
model_threelay.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['AUC'])


%%time
history_threelay = model_threelay.fit(
    x = train_loader,
    steps_per_epoch = TR_STEPS,
    epochs = num_epochs,
    validation_data = val_loader,
    validation_steps = VAL_STEPS,
    verbose = 1,
    callbacks = callbacks_list
)


lossthree, aucthree = model_threelay.evaluate(val_loader, verbose=0)
print(f"Model Three Test AUC: {aucthree:.4f}")


# Predict on new data

predictionthree = model_threelay.predict(sample)
print(f"Model Three predicted probability of class 1: {predictionthree[0][0]:.4f}")


# Create a Sequential model
model_fourlay = Sequential()

# Add a first Convolutional layer
# 16 filters, 3x3 kernel size, ReLU activation, specify input shape
model_fourlay.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))

# Add a Max Pooling layer
# 2x2 pool size to reduce spatial dimensions
model_fourlay.add(MaxPooling2D(pool_size=(2, 2)))

# Add a second Convolutional layer
# 32 filters, 3x3 kernel size, ReLU activation, specify input shape
model_fourlay.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model_fourlay.add(MaxPooling2D(pool_size=(2, 2)))

# Add a third Convolutional Layer
model_fourlay.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_fourlay.add(layers.MaxPooling2D((2, 2)))

# Add a fourth Convolutional Layer
model_fourlay.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_fourlay.add(layers.MaxPooling2D((2, 2)))

# Flatten the output of the pooling layer to feed into a dense layer
model_fourlay.add(Flatten())

# Add a Dense output layer for binary classification
# 1 neuron with sigmoid activation for probability output (0 to 1)
model_fourlay.add(Dense(1, activation='sigmoid'))

# Print a summary of the model architecture
model_fourlay.summary()


opt = tf.keras.optimizers.Adam(learning_rate = 1e-5)
model_fourlay.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['AUC'])


%%time
history_fourlay = model_fourlay.fit(
    x = train_loader,
    steps_per_epoch = TR_STEPS,
    epochs = num_epochs,
    validation_data = val_loader,
    validation_steps = VAL_STEPS,
    verbose = 1,
    callbacks = callbacks_list
)


lossfour, aucfour = model_fourlay.evaluate(val_loader, verbose=0)
print(f"Model Four Test AUC: {aucfour:.4f}")


# Predict on new data

predictionfour = model_fourlay.predict(sample)
print(f"Model Four predicted probability of class 1: {predictionfour[0][0]:.4f}")


display_perf(history_onelay.history, 'Model One Training Loss', 'Model One Training AUC')
display_perf(history_twolay.history, 'Model Two Training Loss', 'Model Two Training AUC')
display_perf(history_threelay.history, 'Model Three Training Loss', 'Model Three Training AUC')
display_perf(history_fourlay.history, 'Model Four Training Loss', 'Model Four Training AUC')


# save models in Google Drive
from google.colab import drive
drive.mount('/content/drive')
model_save_path = '/content/drive/MyDrive/Cancer_Detection_Models/'


import pickle
import os

# Create the directory if it doesn't exist
os.makedirs(model_save_path, exist_ok=True)

model_onelay.save(f'{model_save_path}/Cancer_Detection_model_onelay.keras')
pickle.dump(history_onelay.history, open(f'{model_save_path}/Cancer_Detection_model_onelay.pk1', 'wb'))

model_twolay.save(f'{model_save_path}/Cancer_Detection_model_twolay.keras')
pickle.dump(history_twolay.history, open(f'{model_save_path}/Cancer_Detection_model_twolay.pk1', 'wb'))

model_threelay.save(f'{model_save_path}/Cancer_Detection_model_threelay.keras')
pickle.dump(history_threelay.history, open(f'{model_save_path}/Cancer_Detection_model_threelay.pk1', 'wb'))

model_fourlay.save(f'{model_save_path}/Cancer_Detection_model_fourlay.keras')
pickle.dump(history_fourlay.history, open(f'{model_save_path}/Cancer_Detection_model_fourlay.pk1', 'wb'))

