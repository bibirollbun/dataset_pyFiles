# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv')
df.head()


df['label'].value_counts()


import os
os.mkdir('/kaggle/working/ValidationData')
os.mkdir('/kaggle/working/ValidationData/Naeimi')
os.mkdir('/kaggle/working/ValidationData/Goat')
os.mkdir('/kaggle/working/ValidationData/Sawakni')
os.mkdir('/kaggle/working/ValidationData/Roman')
os.mkdir('/kaggle/working/ValidationData/Najdi')
os.mkdir('/kaggle/working/ValidationData/Harri')
os.mkdir('/kaggle/working/ValidationData/Barbari')
os.mkdir('/kaggle/working/TrainingData')
os.mkdir('/kaggle/working/TrainingData/Naeimi')
os.mkdir('/kaggle/working/TrainingData/Goat')
os.mkdir('/kaggle/working/TrainingData/Sawakni')
os.mkdir('/kaggle/working/TrainingData/Roman')
os.mkdir('/kaggle/working/TrainingData/Najdi')
os.mkdir('/kaggle/working/TrainingData/Harri')
os.mkdir('/kaggle/working/TrainingData/Barbari')



df[df['filename']=='fd69bce2.jpg']['label'].iloc[0]


import shutil
from os import listdir

classes=['Naeimi','Goat','Sawakni','Roman','Najdi','Harri','Barbari']
data={}
for i in listdir('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'):
    src_prefix=f'/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train/{i}'


    label=df[df['filename']==i]['label'].iloc[0]
    dest_prefix=f'/kaggle/working/TrainingData/{label}/{i}'
    #files=os.listdir(src_prefix+i)
    #data.update({i:len(files)})
    
    shutil.copy(src_prefix,dest_prefix)



for i in classes:
    src_prefix='/kaggle/working/TrainingData/'
    dest_prefix='/kaggle/working/ValidationData/'
    files=os.listdir(src_prefix+i)
    size=len(files)*0.2
    counter=0
    for path in files:
        if counter>size:
            break
        else:
            shutil.move(src_prefix+i+'/'+path,dest_prefix+i)
        counter+=1


from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_width = 512
image_height= 512
train_generator= ImageDataGenerator(preprocessing_function=preprocess_input,
    #rotation_range=10,  # Rotate images randomly by up to 20 degrees
    #width_shift_range=0.2,  # Shift images horizontally by up to 20% of the width
    #height_shift_range=0.2,  # Shift images vertically by up to 20% of the height
    zoom_range=0.1,  # Zoom images randomly by up to 20%
    horizontal_flip=True,  # Flip images horizontally
    vertical_flip=True  # Don't flip images vertically

)
valid_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_generator.flow_from_directory(
        '/kaggle/working/TrainingData',
        target_size=(image_width, image_height),
        batch_size=28,
        class_mode='categorical',
        shuffle=True
)

validation_generator = valid_generator.flow_from_directory(
        '/kaggle/working/ValidationData',
        target_size=(image_width, image_height),
        batch_size=10,
        class_mode='categorical',
        shuffle='False'
)


from tensorflow.keras.applications import EfficientNetV2B1
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten,BatchNormalization,Dropout
from tensorflow.keras.regularizers import l2


num_classes = 4
base=EfficientNetV2B1(include_top=False, pooling='avg', weights="imagenet")
model = Sequential()
model.add(base)
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu', kernel_regularizer=l2(0.1)))
model.add(Dropout(0.5))
#improved_res.add(Dense(16,activation='relu',kernel_regularizer=l2(0.1)))
model.add(Dense(7, activation='softmax'))


model.layers[0].trainable = True

model.summary()



for layer in model.layers:
    print(layer, layer.trainable)


import tensorflow as tf

def f1_score(y_true, y_pred):
    # Calculate Precision and Recall for batches
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))

    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())

    # Calculate F1 score
    f1_val = 2*(precision*recall)/(precision+recall+tf.keras.backend.epsilon())
    return f1_val


from tensorflow.keras.optimizers import Adam
from tensorflow import keras
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=[keras.metrics.AUC(),f1_score])


history=model.fit(
        train_generator,
        epochs=6,
        steps_per_epoch=6,
        validation_data=validation_generator,
        validation_steps=5)


import matplotlib.pyplot as plt

plt.plot(history.history['f1_score'],color='red',label='train')
plt.plot(history.history['val_f1_score'],color='blue',label='validation')
plt.legend()
plt.show()

