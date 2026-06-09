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
    horizontal_flip = False,
    vertical_flip = False,
    height_shift_range = 0,
    width_shift_range = 0,
    rotation_range = 0,
    target_size = (64,64)
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
    target_size = (64,64)
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


num_epochs = 20


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



# Get a sample batch from the validation loader
sample_batch = next(iter(val_loader))
sample_images, sample_labels = sample_batch
sample = sample_images[0:1] # Take the first image in the batch


base_model_den=keras.applications.DenseNet121(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(64,64,3),
    pooling=None,
    classes=2,
    classifier_activation="Swish",
)
base_model_den.trainable = True

for layer in base_model_den.layers[:-30]:
    layer.trainable = True

cnn_model_den = models.Sequential([
    base_model_den,
    layers.GlobalAveragePooling2D(),
    Dense(512, activation = 'swish'),
    Dropout(.4),
    Dense(256, activation = 'swish'),
    Dropout(.3),
    Dense(128, activation = 'swish'),
    Dropout(.2),
    Dense(1, activation = 'sigmoid')
])
cnn_model_den.summary()


opt = tf.keras.optimizers.Adam(learning_rate = 1e-5)
cnn_model_den.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['AUC'])


%%time
history_den = cnn_model_den.fit(
    x = train_loader,
    steps_per_epoch = TR_STEPS,
    epochs = num_epochs,
    validation_data = val_loader,
    validation_steps = VAL_STEPS,
    verbose = 1,
    callbacks = callbacks_list
)


lossden, aucden = cnn_model_den.evaluate(val_loader, verbose=0)
print(f"DenseNet121 Test AUC: {aucden:.4f}")


# Predict on new data

predictionden = cnn_model_den.predict(sample)
print(f"DenseNet121 predicted probability of class 1: {predictionden[0][0]:.4f}")


display_perf(history_den.history, 'DenseNet121 Training Loss', 'DenseNet121 Training AUC')



# save models in Google Drive
from google.colab import drive
drive.mount('/content/drive')
model_save_path = '/content/drive/MyDrive/Cancer_Detection_Models/'


import pickle
import os

# Create the directory if it doesn't exist
os.makedirs(model_save_path, exist_ok=True)

cnn_model_den.save(f'{model_save_path}/Cancer_Detection_model_denbest.keras')
pickle.dump(history_den.history, open(f'{model_save_path}/Cancer_Detection_model_denbest.pk1', 'wb'))


