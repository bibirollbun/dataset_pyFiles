import kagglehub
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


train = '/kaggle/input/competencia-02-julio-2025/archive/pizza_steak/train'


im_size = 224
batch_size = 32
#classes = 2


train_df = tf.keras.utils.image_dataset_from_directory(train, validation_split = 0.2, subset = "training", seed = 123, image_size = (im_size, im_size), batch_size = batch_size)


validation_df = tf.keras.utils.image_dataset_from_directory(train, validation_split = 0.2, subset = "validation", seed = 123, image_size = (im_size, im_size), batch_size = batch_size)


AUTOTUNE = tf.data.AUTOTUNE
train_df = train_df.prefetch(buffer_size=AUTOTUNE)
validation_df = validation_df.prefetch(buffer_size=AUTOTUNE)


modelo = EfficientNetB0(include_top = False,
                        weights='imagenet',
                        input_shape=(im_size, im_size, 3)
                        )


modelo.trainable = False


inputs = layers.Input(shape=(im_size, im_size, 3))
x = modelo(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)


modelo = models.Model(inputs, outputs)


modelo.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


modelo.summary()


early_stopping = EarlyStopping(
    monitor = 'val_accuracy',
    patience = 3,
    restore_best_weights = True 
)


reduce = ReduceLROnPlateau(
    monitor = 'val_accuracy',
    factor = 0.2,
    patience = 2,
    min_rl = 1e-6,
    verbose = 1
)


history = modelo.fit(train_df, epochs = 10, validation_data=validation_df, callbacks = [early_stopping, reduce])


modelo.save('/kaggle/working/pizza_vs_steak_Amaury.keras')

