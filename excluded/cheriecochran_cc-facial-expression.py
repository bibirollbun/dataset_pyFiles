import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import cv2

from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Input, Add, GlobalAveragePooling2D, SpatialDropout2D, Activation
from sklearn.model_selection import train_test_split


train = pd.read_csv("/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv", dtype=str)
print(train.shape)


emotions = {
    0: 'Angry', 
    1: 'Disgust', 
    2: 'Fear', 
    3: 'Happy', 
    4: 'Sad', 
    5: 'Surprise', 
    6: 'Neutral'
}


# Display the head of the train DataFrame. 
train.head()


train['emotion'].value_counts()


plt.figure(figsize=(9, 8))


sns.countplot(x=train.emotion)
_ = plt.title('Label Distribution')
_ = plt.xticks(ticks=range(0, 7), labels=[emotions[i] for i in range(0, 7)], )


def parse_pixels(pixel_string):
    # Split the string by spaces and convert to integers
    pixels = np.array(pixel_string.split(' ')).astype(int)
    # Reshape the 1D array into a 2D array representing the 48x48 image
    return pixels.reshape(48, 48)

train['image'] = train['pixels'].apply(parse_pixels)
print(train['image'][0]) # Access the first image as a NumPy array


sample = train.sample(n=16).reset_index()

plt.figure(figsize=(6, 6))

for i, row in sample.iterrows():
    label = int(row.emotion)

    plt.subplot(4,4,i+1)
    plt.imshow(row['image'], cmap='gray')
    plt.text(0, -5, f"{emotions[label]} ({label})", color='k')
        
    plt.axis('off')

plt.tight_layout()
plt.show()


train_df, valid_df = train_test_split(train, test_size=0.2, random_state=1, stratify=train.emotion)


X_train = np.stack(train_df['image']).reshape(-1, 48, 48, 1)
print(f"X_train shape: {np.stack(X_train).shape}")

y_train = to_categorical(train_df['emotion'], num_classes = len(emotions))
print(f"y_train shape: {y_train.shape}")

X_validation = np.stack(valid_df['image']).reshape(-1, 48, 48, 1)
print(f"X_validation shape:  {np.stack(X_validation).shape}")

y_validation = to_categorical(valid_df['emotion'], num_classes = len(emotions))
print(f"y_validation shape: {y_validation.shape}")


print(y_train)


from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.utils import to_categorical


model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))


model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()


history = model.fit(X_train, y_train,
                    validation_data=(X_validation, y_validation),
                    epochs=12,
                    batch_size=64)


train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy'] # or 'acc' depending on Keras version/metric name
val_accuracy = history.history['val_accuracy'] # or 'val_acc'
epochs = range(1, len(train_loss) + 1)


# Plot training & validation loss values

plt.figure(figsize=(5, 3))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(5, 3))
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


test = pd.read_csv("/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/test.csv", dtype=str)
print(test.shape)


test['image'] = test['pixels'].apply(parse_pixels)

plt.imshow(test['image'][0], cmap='gray')


X_test = np.stack(test['image']).reshape(-1, 48, 48, 1)
print(f"X_test shape:  {np.stack(X_test).shape}")


test_pred = model.predict(X_test)


test_pred.shape


print(test_pred[2])


y_pred_classes = np.argmax(test_pred, axis = 1)


print(y_pred_classes)


# Create submission DataFrame
submission = pd.DataFrame({'label': y_pred_classes})

submission.head()


# Write submission DataFrame to csv
submission.to_csv('cc_FER_submission.csv', header=True, index=False)

