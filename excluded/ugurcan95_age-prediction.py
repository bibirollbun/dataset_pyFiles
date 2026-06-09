#pip install opencv-python


#!pip install scikit-learn


 # !pip install --upgrade tensorflow


import os
import random
import warnings

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import re
from sklearn.model_selection import train_test_split
#from IPython.display import Image

from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, Reshape, GlobalAveragePooling2D,Activation
from keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

warnings.filterwarnings("ignore", category=DeprecationWarning)


df = pd.read_csv("/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/train.csv")
df.head()


img_path = "/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/"
df['filename'] = df['filename'].apply(lambda x: img_path + x)


df.rename(columns={'filename': 'img'}, inplace=True)


df.img[0]


selected_images = df.groupby('age', as_index=False).apply(lambda x: x.sample(n=1, random_state=2)).reset_index(drop=True)

sns.set(style='whitegrid')
fig, axes = plt.subplots(5, 10, figsize=(15, 8))

axes = axes.flatten()

for ax, (img_path, label) in zip(axes, zip(selected_images['img'], selected_images['age'])):
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(label, fontsize=14)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=10, kde=True, color='skyblue', stat='density')
plt.title('Age Distribution', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(axis='y')
plt.show()


label_values = df['age'].value_counts().reset_index()
label_values.columns = ['Age', 'Count']

# Create a bar plot with a suitable palette
plt.figure(figsize=(15, 6))
sns.barplot(data=label_values, x='Age', y='Count')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.xticks(rotation=90)
plt.show()


img_width, img_height = 128, 128


x = []
y = []

for img_path, label in zip(df['img'], df['age']):
    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    x.append(img)
    y.append(label)


x = np.array(x)
y = np.array(y)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


model = Sequential()

model.add(Input(shape=(img_width, img_height, 3)))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(1) )

model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])


early_stopping = EarlyStopping(monitor='mse', patience=5)

history = model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping])


model.summary()


history_df = pd.DataFrame(history.history)
history_df


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training vs Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.title("Training vs Validation MAE")
plt.legend()

plt.tight_layout()
plt.show()


unique_ages_array_sorted = np.sort(np.array(df['age'].unique()))


model.save('age_prediction.keras')


img_path= '/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/'
df_test = pd.read_csv('/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/test.csv')
df_test.head()


df_test['filename'] = df_test['filename'].apply(lambda x: img_path + x)


df_test.rename(columns={'filename': 'img'}, inplace=True)


df_test.head()


img_width, img_height = 128, 128


x = []

for img_path in df_test['img']:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        continue

    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    x.append(img)


x_test = np.array(x)


predictions=model.predict(x_test)
predictions=np.round(predictions).astype(int)
predictions=predictions.flatten()


submission=pd.DataFrame({
    'id':df_test['id'],
    'age':predictions
})


submission.head()


submission.to_csv('submission.csv',index=False)

