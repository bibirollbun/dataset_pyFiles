import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


train_dir = '/kaggle/input/dog-breed-identification/train/'
test_dir = '/kaggle/input/dog-breed-identification/test/'
labels_file = '/kaggle/input/dog-breed-identification/labels.csv'

labels = pd.read_csv(labels_file)
print(labels.head())



from os.path import join

for dirname, _, filenames in os.walk(train_dir):
    for filename in filenames:
        os.path.join(dirname, filename)



train_data = labels.assign(img_path = lambda x : train_dir + x['id'] + '.jpg')
train_data.head()


img=cv2.imread('/kaggle/input/dog-breed-identification/test/000621fb3cbb32d8935728e48679680e.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
H=128
W=128
C=3

X = np.array([img_to_array(load_img(img,target_size = (H,W))) for img in train_data['img_path'].values.tolist()])
print(X.shape)
Y = pd.get_dummies(train_data['breed'])
print(Y.shape)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test  = train_test_split(X,Y,test_size = 0.25)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


from tensorflow import keras
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,Activation


model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(H,W,C)))
model.add(LeakyReLU(alpha=0.01))
# model.add(Activation('relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3)))
model.add(LeakyReLU(alpha=0.01))
# model.add(Activation('relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3)))
model.add(LeakyReLU(alpha=0.01))
# model.add(Activation('relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3)))
model.add(LeakyReLU(alpha=0.01))
# model.add(Activation('relu'))

model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512))
model.add(LeakyReLU(alpha=0.01))
# model.add(Activation('relu'))

model.add(Dense(Y.shape[1]))
model.add(Activation('softmax'))

model.summary()


batch=32


model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['accuracy'])


trained_model=model.fit(X_train,Y_train,
         epochs=20,
         batch_size=batch,
         steps_per_epoch=X_train.shape[0]//batch,
         validation_steps=X_test.shape[0]//batch,
         validation_data=(X_test,Y_test),
         verbose=2)


test_datagen = ImageDataGenerator()

test_set = test_datagen.flow_from_directory(
    '/kaggle/input/dog-breed-identification',
    target_size = (128,128),
    classes=['test']
)
y_pred = model.predict(test_set)


import re
file_list = test_set.filenames
id_list = []
for name in file_list:
    m = re.sub('test/', '', name)
    m = re.sub('.jpg', '', m)
    id_list.append(m)



train_datagen = ImageDataGenerator()

train_set = train_datagen.flow_from_directory(
    '/kaggle/input/dog-breed-identification',
    target_size = (128,128),
    classes=['train']
)
y_pred_train = model.predict(train_set)


from sklearn.metrics import log_loss
log_loss(Y, y_pred_train)


submission = pd.read_csv('/kaggle/input/dog-breed-identification/sample_submission.csv')
submission.head()
submission['id'] = id_list
submission.iloc[:,1:] =y_pred
submission.head()

df = pd.DataFrame(submission)
df = submission.set_index('id')
df.to_csv('submission.csv')
# print(df)


print(df)

