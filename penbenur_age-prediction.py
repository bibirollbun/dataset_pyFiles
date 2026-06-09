# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
import os
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")
from keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D,Dropout,BatchNormalization,Reshape
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn.metrics import mean_absolute_error


dataset_dir='/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age'


df=pd.read_csv("/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/train.csv")


# Load dataset
def load_data(df, dataset_dir):
    images = []
    ages = []
    
    for index, row in df.iterrows():
        img_path = os.path.join(dataset_dir, row['filename'])
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0  # Normalize the image
        images.append(img_array)
        ages.append(row['age'])

    return np.array(images), np.array(ages)


# Load images and labels
X, y = load_data(df, dataset_dir)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])


# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


# Train the model
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)


# Plot training & validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


# Visualize predictions
y_pred = model.predict(X_test)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Ages')
plt.ylabel('Predicted Ages')
plt.title('Actual vs Predicted Ages')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Diagonal line
plt.show()


# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')


# Save the model
model.save('age_prediction_model.h5')




