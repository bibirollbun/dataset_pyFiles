# libraries

import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import RMSprop
from keras.datasets import cifar10

import os
import warnings

warnings.filterwarnings('ignore')


# Load data

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# Shape of features

print("Shape of X_train: {}".format(X_train.shape))
print("Shape of X_test: {}".format(X_test.shape))


# some of examples

import random

random_indices = random.sample(range(50000), 100)  

plt.figure(figsize=(10, 10))

for i, idx in enumerate(random_indices):
    plt.subplot(10, 10, i + 1)
    plt.imshow(X_train[idx])
    plt.axis("off")

plt.show()


# convert to one-hot-encoding

numberOfClass = 10

y_train = to_categorical(y_train, numberOfClass)
y_test = to_categorical(y_test, numberOfClass)

print("Shape of y_train: {}".format(y_train.shape))
print("Shape of y_test: {}".format(y_test.shape))


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.80, random_state=42)


# Shape of trains, validations and test data

print("Shape of X_train: {}".format(X_train.shape))
print("Shape of X_val: {}".format(X_val.shape))
print("Shape of X_test: {}".format(X_test.shape))

print("Shape of y_train: {}".format(y_train.shape))
print("Shape of y_val: {}".format(y_val.shape))
print("Shape of y_test: {}".format(y_test.shape))


train_datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg19.preprocess_input,
                                  rotation_range = 10,
                                  zoom_range = 0.1,
                                  width_shift_range = 0.1,
                                  height_shift_range = 0.1,
                                  shear_range = 0.1,
                                  horizontal_flip = True,
                                  vertical_flip = False)

val_datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg19.preprocess_input)


train_datagen.fit(X_train)
val_datagen.fit(X_val)


from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',
                                           patience = 3,
                                           verbose = 1,
                                           factor = 0.5,
                                           min_lr = 0.00001)


vgg = VGG19(include_top = False, weights = "imagenet", input_shape = X_train.shape[1:])

vgg.summary()


# building our model from vgg19

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()


# loss & optimizer

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9)

model.compile(loss = "categorical_crossentropy",
             optimizer = optimizer,
             metrics = ["accuracy"])


hist = model.fit(train_datagen.flow(X_train, y_train, batch_size = 256),
                validation_data = val_datagen.flow(X_val, y_val, batch_size = 256),
                epochs = 25,
                verbose = 1,
                callbacks = [learning_rate_reduction])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss Graph
ax1.plot(hist.history["loss"], label="Train Loss", color='blue')
ax1.plot(hist.history["val_loss"], label="Validation Loss", color='orange')
ax1.set_title("Loss Over Epochs")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

# Accuracy Graph
ax2.plot(hist.history["accuracy"], label="Train Accuracy", color='blue')
ax2.plot(hist.history["val_accuracy"], label="Validation Accuracy", color='orange')
ax2.set_title("Accuracy Over Epochs")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


from sklearn.metrics import accuracy_score

X_test = tf.keras.applications.vgg19.preprocess_input(X_test)

y_pred = np.argmax(model.predict(X_test), axis=1)
y_test = np.argmax(y_test, axis=1)

print("Testing Accuracy:", accuracy_score(y_test, y_pred))



# confussion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize = (10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

