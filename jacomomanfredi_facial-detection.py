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


import zipfile
import os

import seaborn
from matplotlib import pyplot as plt
%matplotlib inline


facial_keypoints_detection_path = '/kaggle/input/facial-keypoints-detection/'

facial_keypoints_detection_path_out = '/kaggle/working/'


test_zip_path = os.path.join(facial_keypoints_detection_path, 'test.zip')
train_zip_path = os.path.join(facial_keypoints_detection_path, 'training.zip')

# Create directories to extract the files
test_extracted_path = os.path.join(facial_keypoints_detection_path_out, 'test')
train_extracted_path = os.path.join(facial_keypoints_detection_path_out, 'training')

os.makedirs(test_extracted_path, exist_ok=True)
os.makedirs(train_extracted_path, exist_ok=True)

# Unzip the files
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(test_extracted_path)

with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(train_extracted_path)

print("Extraction complete.")


train = pd.read_csv(os.path.join(train_extracted_path, 'training.csv'))
test = pd.read_csv(os.path.join(test_extracted_path, 'test.csv'))



train.shape


test.shape


train.isna().sum()


test.isna().sum()


train.tail(5)


train.ffill(inplace=True)


train.isna().sum()


def process_img(data):
    images = []
    for idx, row in train.iterrows():
        image = np.array(row.Image.split(' '), dtype=int)
        image = np.reshape(image, (96,96,1))
        images.append(image)
    images = np.array(images)/255
    return images


plt.figure(figsize=(12,8))

plt.imshow(process_img(train)[21], cmap='gray')


def get_keypoints(data):
    keypoints = data.drop('Image', axis=1)
    keypoint_features = []
    for idx, samp_keypoints in keypoints.iterrows():
        keypoint_features.append(samp_keypoints)
    keypoint_features = np.array(keypoint_features, dtype='float')
    return keypoint_features


# From train dataset
x_train = process_img(train)
y_train = get_keypoints(train)


plt.figure(figsize=(12,8))

plt.imshow(x_train[2], cmap='gray')


def img_with_keypoints(df, index):
    image = plt.imshow(x_train[index], cmap='gray')
    l = []
    for i in range(1,31,2):
        l.append(plt.plot(df[index][i-1], df[index][i],'r*'))
    return image, l
    


img_with_keypoints(y_train, 3996)


import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, Input, BatchNormalization
from sklearn.model_selection import train_test_split as tts


x_train.shape


model = Sequential()

model.add(Conv2D(32, kernel_size=2, strides=1, activation='relu', padding='same', input_shape=(96,96,1)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(64, kernel_size=3, strides=2, padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(30))


model.summary()


model.compile(loss='mae', optimizer='adam', metrics=['mse'])


preds = model.fit(x_train, y_train, batch_size=32, validation_split=0.2, epochs=4, verbose=1)


X_test = process_img(test)
y_pred = model.predict(X_test)


def pred_keypoints(df, index):
    image = plt.imshow(X_test[index], cmap='gray')
    l = []
    for i in range(1,31,2):
        l.append(plt.plot(df[index][i-1], df[index][i], 'r*'))
    return image, l 


pred_keypoints(y_pred, 900)


lookup_df = pd.read_csv("/kaggle/input/facial-keypoints-detection/IdLookupTable.csv")

lookup_df


feature_names = train.columns[:-1]

prediction_df = pd.DataFrame(y_pred, columns=feature_names)


submission = lookup_df.copy()

submission['Location'] = submission.apply(
    lambda row: prediction_df.loc[row.ImageId, row.FeatureName], axis=1
)
final_submission = submission[['RowId', 'Location']]


final_submission


#Create a dataframe of y_preds with column names from the test dataframe
final_submission.to_csv('submission.csv', index=False)
print("ðŸŽ¯ Submission File Created!")

