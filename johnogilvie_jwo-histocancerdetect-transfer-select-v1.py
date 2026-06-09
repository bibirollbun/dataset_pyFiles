#Histopathological Cancer Detection Project
#Copy & Edit from JWO HistoCancerDetect EDA select v1, which is from MWV Final Project Training Model 5


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
    target_size = (200,200)
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
    target_size = (200,200)
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



base_model_eff=keras.applications.EfficientNetV2B2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(200,200,3),
    pooling=None,
    classes=2,
    classifier_activation="Swish",
)
base_model_eff.trainable = True

for layer in base_model_eff.layers[:-30]:
    layer.trainable = True

cnn_model_eff = models.Sequential([
    base_model_eff,
    layers.GlobalAveragePooling2D(),
    Dense(512, activation = 'swish'),
    Dropout(.4),
    Dense(256, activation = 'swish'),
    Dropout(.3),
    Dense(128, activation = 'swish'),
    Dropout(.2),
    Dense(1, activation = 'sigmoid')
])
cnn_model_eff.summary()


opt = tf.keras.optimizers.Adam(learning_rate = 1e-5)
cnn_model_eff.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['AUC'])


%%time
history_eff = cnn_model_eff.fit(
    x = train_loader,
    steps_per_epoch = TR_STEPS,
    epochs = num_epochs,
    validation_data = val_loader,
    validation_steps = VAL_STEPS,
    verbose = 1,
    callbacks = callbacks_list
)


losseff, auceff = cnn_model_eff.evaluate(val_loader, verbose=0)
print(f"EfficientNetV2B3 Test AUC: {auceff:.4f}")


# Get a sample batch from the validation loader
sample_batch = next(iter(val_loader))
sample_images, sample_labels = sample_batch
sample = sample_images[0:1] # Take the first image in the batch


# Predict on new data

predictioneff = cnn_model_eff.predict(sample)
print(f"EfficientNetV2B3 predicted probability of class 1: {predictioneff[0][0]:.4f}")


base_model_res=keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(200,200,3),
    pooling=None,
    classes=2,
    classifier_activation="Swish",
)
base_model_res.trainable = True

for layer in base_model_res.layers[:-30]:
    layer.trainable = True

cnn_model_res = models.Sequential([
    base_model_res,
    layers.GlobalAveragePooling2D(),
    Dense(512, activation = 'swish'),
    Dropout(.4),
    Dense(256, activation = 'swish'),
    Dropout(.3),
    Dense(128, activation = 'swish'),
    Dropout(.2),
    Dense(1, activation = 'sigmoid')
])
cnn_model_res.summary()


opt = tf.keras.optimizers.Adam(learning_rate = 1e-5)
cnn_model_res.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['AUC'])


%%time
history_res = cnn_model_res.fit(
    x = train_loader,
    steps_per_epoch = TR_STEPS,
    epochs = num_epochs,
    validation_data = val_loader,
    validation_steps = VAL_STEPS,
    verbose = 1,
    callbacks = callbacks_list
)


lossres, aucres = cnn_model_res.evaluate(val_loader, verbose=0)
print(f"ResNet50 Test AUC: {aucres:.4f}")


# Predict on new data

predictionres = cnn_model_res.predict(sample)
print(f"ResNet50 predicted probability of class 1: {predictionres[0][0]:.4f}")


base_model_den=keras.applications.DenseNet121(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(200,200,3),
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





base_model_inc=keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(200,200,3),
    pooling=None,
    classes=2,
    classifier_activation="Swish",
)
base_model_inc.trainable = True

for layer in base_model_inc.layers[:-30]:
    layer.trainable = True

cnn_model_inc = models.Sequential([
    base_model_inc,
    layers.GlobalAveragePooling2D(),
    Dense(512, activation = 'swish'),
    Dropout(.4),
    Dense(256, activation = 'swish'),
    Dropout(.3),
    Dense(128, activation = 'swish'),
    Dropout(.2),
    Dense(1, activation = 'sigmoid')
])
cnn_model_inc.summary()


opt = tf.keras.optimizers.Adam(learning_rate = 1e-5)
cnn_model_inc.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['AUC'])


%%time
history_inc = cnn_model_inc.fit(
    x = train_loader,
    steps_per_epoch = TR_STEPS,
    epochs = num_epochs,
    validation_data = val_loader,
    validation_steps = VAL_STEPS,
    verbose = 1,
    callbacks = callbacks_list
)


lossinc, aucinc = cnn_model_inc.evaluate(val_loader, verbose=0)
print(f"InceptionV3 Test AUC: {aucinc:.4f}")


# Predict on new data

predictioninc = cnn_model_inc.predict(sample)
print(f"InceptionV3 predicted probability of class 1: {predictioninc[0][0]:.4f}")


display_perf(history_eff.history, 'EfficientNetV2B3 Training Loss', 'EfficientNetV2B3 Training AUC')
display_perf(history_res.history, 'ResNet50 Training Loss', 'ResNet50 Training AUC')
display_perf(history_den.history, 'DenseNet121 Training Loss', 'DenseNet121 Training AUC')
display_perf(history_inc.history, 'InceptionV3 Training Loss', 'InceptionV3 Training AUC')


# save models in Google Drive
from google.colab import drive
drive.mount('/content/drive')
model_save_path = '/content/drive/MyDrive/Cancer_Detection_Models/'


import pickle
import os

# Create the directory if it doesn't exist
os.makedirs(model_save_path, exist_ok=True)

cnn_model_eff.save(f'{model_save_path}/Cancer_Detection_model_eff.keras')
pickle.dump(history_eff.history, open(f'{model_save_path}/Cancer_Detection_model_eff.pk1', 'wb'))

cnn_model_res.save(f'{model_save_path}/Cancer_Detection_model_res.keras')
pickle.dump(history_res.history, open(f'{model_save_path}/Cancer_Detection_model_res.pk1', 'wb'))

cnn_model_den.save(f'{model_save_path}/Cancer_Detection_model_den.keras')
pickle.dump(history_den.history, open(f'{model_save_path}/Cancer_Detection_model_den.pk1', 'wb'))

cnn_model_inc.save(f'{model_save_path}/Cancer_Detection_model_inc.keras')
pickle.dump(history_inc.history, open(f'{model_save_path}/Cancer_Detection_model_inc.pk1', 'wb'))

