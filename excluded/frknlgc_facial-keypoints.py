import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers

from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import zipfile

zip_path = "/kaggle/input/facial-keypoints-detection/training.zip"
extract_path = "/kaggle/working/"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


def  parseData(csvName,trainFlag):
    df = pd.read_csv(csvName)
    numFeatures = df.columns.size -1
    numSamples = df.shape[0]
    y = np.zeros([numSamples,numFeatures])
    X = np.zeros([numSamples,96,96])
  

    for i in range(numSamples):
        anImageRaw = df['Image'][i]
        anImage = list(map(int,anImageRaw.split()))
        anImage = np.array(anImage).reshape(96,96)
        X[i,:,:] = anImage/255
        
    
    if trainFlag == 1:
        i = 0 
        for col in [c for c in df.columns if c != 'Image']:
            y[:,i] = df[col].fillna(df[col].mean())
            i = i+1
    else:
        y = 0

    X = X[..., np.newaxis]
    return [X,y/96]


[X,y] = parseData('training.csv',1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='linear'))
model.add(layers.Dense(30))  # number of features


model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))


import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

