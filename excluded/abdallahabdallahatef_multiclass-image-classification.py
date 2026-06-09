import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

# deep learning libraries
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image

import cv2
import os

import warnings
warnings.filterwarnings('ignore')


labels=pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
labels.head(10)


path1=os.listdir('/kaggle/input/dog-breed-identification')
path1


train_path=os.path.join('/kaggle/input/dog-breed-identification',path1[2])
train_path


test_path=os.path.join('/kaggle/input/dog-breed-identification',path1[1])
test_path


train = []
for root, dirs, files in os.walk(train_path):
    for file in files:
        full_path = os.path.join(root, file)
        train.append(full_path)



len(train)


train


img=cv2.imread(train[0])
plt.imshow(img)


imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(imgRGB)


len(labels)


test=[]
for root,dirs,files in os.walk(test_path):
    for file in files:
        full_path=os.path.join(root,file)
        test.append(full_path)


test


img=cv2.imread(test[0])
plt.imshow(img)


img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)


def lab(ID):
    return ID+'.jpg'


labels['id']=labels['id'].apply(lab)


labels


sample=pd.read_csv('/kaggle/input/dog-breed-identification/sample_submission.csv')
len(sample)


len(test)


sample


labels.columns


gen = ImageDataGenerator(
    rescale=1./255.,         
    horizontal_flip=True,     
    validation_split=0.2      
)


train_generator = gen.flow_from_dataframe(
    dataframe=labels,         
    directory=train_path,     
    x_col='id',                
    y_col='breed',            
    subset="training",         
    color_mode="rgb",
    target_size=(331,331),     
    class_mode="categorical",  
    batch_size=32,
    shuffle=True,
    seed=42
)



val_generator=gen.flow_from_dataframe(
    labels,
    directory=train_path,
    x_col='id',
    y_col='breed',
    subset="validation",
    class_mode='categorical',
    batch_size=32,
    color_mode="rgb",
    shuffle=True,
    seed=42,
    target_size=(331,331)
    
    
)


x,y = next(train_generator)
print(x.shape)
print(y.shape)


image_name=train_generator.class_indices


image_name


class_name=list(image_name.keys())
class_name


def plot_image(image,label):
    plt.figure(figsize=(12,12))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(image[i])
        plt.title(class_name[np.argmax(label[i])])
        plt.axis('off')
plot_image(x,y)


base_model = tf.keras.applications.InceptionResNetV2(
                     include_top=False,
                     weights='imagenet',
                     input_shape=(331,331,3)
                     )


base_model.trainable=False


input_tensor = tf.keras.Input(shape=(331,331,3))
output_tensor = base_model(input_tensor)
model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(120, activation='softmax')
    ])



model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


model.summary()


early = tf.keras.callbacks.EarlyStopping( patience=10,
                                          min_delta=0.001,
                                          restore_best_weights=True)



batch_size=32
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = val_generator.n//val_generator.batch_size


history = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=25,
                    callbacks=[early])




# store results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# plot results
# accuracy
plt.figure(figsize=(10, 16))
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.facecolor'] = 'white'
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title(f'\nTraining and Validation Accuracy. \nTrain Accuracy: {str(acc[-1])}\nValidation Accuracy: {str(val_acc[-1])}')


model.save('1.h5')




