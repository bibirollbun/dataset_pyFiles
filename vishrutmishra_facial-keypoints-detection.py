# import necessary library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers
import tensorflow as tf


# load train data

train_data = pd.read_csv('/kaggle/input/facial-key-points-dataset/training.csv')
train_data.head(2)


# check missing values in train data

(train_data.isnull().mean()) * 100


# fill missing values using forward fill

train_data = train_data.ffill()


# check null values again

train_data.isnull().mean()


# utility to convert an image from string

def get_image(img_str):
  img = img_str.split(' ')
  img = [int(i) for i in img]
  img = np.array(img)
  img = img.reshape((96, 96))
  return img


# create X and y

X = []
y = np.array(train_data[train_data.columns.drop('Image')])

for i in range(0, train_data.shape[0]):
  img = get_image(train_data['Image'].iloc[i])
  X.append(img)

X = np.array(X)


# visuals of train data

img = X[0]
left_side = [y[0][i] for i in range(0, 30, 2)]
right_side = [y[0][i] for i in range(1, 30, 2)]
plt.imshow(img, cmap='gray')
plt.scatter(left_side, right_side, c='red', marker='o', s=20)
plt.show()


# build a CNN model

model = keras.Sequential()

model.add(layers.Input(shape=(96, 96, 1)))
model.add(layers.Rescaling(1.0/255))


model.add(layers.Conv2D(256, (2,2), activation='relu'))
model.add(layers.Conv2D(256, (2,2), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Conv2D(128, (2,2), activation='relu'))
model.add(layers.Conv2D(128, (2,2), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Conv2D(64, (2,2), activation='relu'))
model.add(layers.Conv2D(64, (2,2), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Conv2D(32, (2,2), activation='relu'))
model.add(layers.Conv2D(32, (2,2), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(30))
model.summary()


# compile the model

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])


# fit the model

model.fit(X, y, epochs = 30, batch_size = 128, validation_split = 0.05)


# load test data

test_data = pd.read_csv('/kaggle/input/facial-key-points-dataset/test.csv')


# load other data files

id_lookup = pd.read_csv('/kaggle/input/facial-key-points-dataset/IdLookupTable.csv')


# prepare test data

X_test = []

for i in range(0, test_data.shape[0]):
  img = get_image(test_data['Image'].iloc[i])
  X_test.append(img)

X_test = np.array(X_test)


# get model predictions

preds = model.predict(X_test, batch_size=64)


# visualise model predictions

img = X_test[1]
left_side = [preds[1][i] for i in range(0, 30, 2)]
right_side = [preds[1][i] for i in range(1, 30, 2)]
plt.imshow(img, cmap='gray')
plt.scatter(left_side, right_side, c='red', marker='o', s=20)
plt.show()


# prepare submission file

import warnings
warnings.filterwarnings('ignore')

feature_idx = {k : v for v, k in enumerate(train_data.columns)}

for i in range(0, id_lookup.shape[0]):
  img_idx = id_lookup['ImageId'].iloc[i] - 1
  pred_idx = feature_idx[id_lookup['FeatureName'].iloc[i]]
  id_lookup['Location'].iloc[i] = preds[img_idx][pred_idx]


# submission data

submission = id_lookup[['RowId', 'Location']]
submission.head()


# create submission file

submission.to_csv('/kaggle/working/submission.csv', index=False)

