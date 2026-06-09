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


import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import RMSprop
from keras.datasets import cifar10
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import warnings

warnings.filterwarnings('ignore')


(X_train, y_train), (X_test, y_test) = cifar10.load_data()


X_train.shape


y_train.shape


# Set a random seed for reproducibility
random.seed(42)

# Select random indices
random_indices = random.sample(range(len(X_train)), 100)  

# Create a figure for displaying images
plt.figure(figsize=(12, 12))
plt.suptitle('Random Samples from Training Data', fontsize=16)

for i, idx in enumerate(random_indices):
    plt.subplot(10, 10, i + 1)
    plt.imshow(X_train[idx], cmap='gray' if X_train[idx].ndim == 2 else None)
    plt.axis("off")

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the title
plt.show()


# Preprocess the data
def preprocess_data(X, y):
    X = X.astype('float32') / 255.0  # Normalize to [0, 1]
    y = to_categorical(y, num_classes=10)  # One-hot encode labels
    return X, y


# Load and preprocess data
X_train, y_train = preprocess_data(X_train, y_train)


# Create data generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Create validation set
val_size = int(0.1 * len(X_train))  # 10% for validation
train_size = len(X_train) - val_size


# Train the model with data augmentation
model.fit(datagen.flow(X_train[:train_size], y_train[:train_size], batch_size=32), 
          epochs=10,  # Limit epochs
          validation_data=(X_train[train_size:], y_train[train_size:]),
          steps_per_epoch=len(X_train) // 32)


# Preprocess the test data
X_test = X_test.astype('float32') / 255.0  # Normalize to [0, 1]
y_test = to_categorical(y_test, num_classes=10)  # One-hot encode labels


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


# Flatten the true labels for the classification report
y_test_flat = np.argmax(y_test, axis=1)


# Generate classification report
report = classification_report(y_test_flat, y_pred_classes, target_names=[
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
])
print("Classification Report:\n", report)


# Generate confusion matrix
conf_matrix = confusion_matrix(y_test_flat, y_pred_classes)
print("Confusion Matrix:\n", conf_matrix)


model.save('cifar10_model.h5')




