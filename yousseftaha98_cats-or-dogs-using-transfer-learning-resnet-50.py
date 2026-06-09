import os
import cv2
from tqdm import tqdm
import zipfile
import random
from random import shuffle
import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50


train_dir_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip" 
test_dir_zip =  "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"

train_dir = "train_data"
test_dir = "test_data"

with zipfile.ZipFile(train_dir_zip,'r') as zip_ref:
    zip_ref.extractall(train_dir)
    
with zipfile.ZipFile(test_dir_zip,'r') as zip_ref:
    zip_ref.extractall(test_dir)


train_path = os.listdir("/kaggle/working/train_data/train")
test_path = os.listdir("/kaggle/working/test_data/test")
TRAIN_FOLDER = '/kaggle/working/train_data/train'
TEST_FOLDER = '/kaggle/working/test_data/test'
IMG_SIZE = 224
TEST_SIZE = 0.3
RANDOM_STATE = 2018
BATCH_SIZE = 64
NO_EPOCHS = 20
NUM_CLASSES = 2
RESNET_WEIGHTS_PATH = '/kaggle/input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def label_img_ohe(img):
    # dog.93.png
    labels = img.split(".")[-3]
    if labels == "cat":
        return [1,0]
    elif labels == "dog":
        return [0,1]


def process_data(data_image_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in tqdm(data_image_list):
        path = os.path.join(DATA_FOLDER,img)
        if(isTrain):
            label = label_img_ohe(img)
        else:
            label = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data_df.append([np.array(img),np.array(label)])
    shuffle(data_df)
    return data_df


def plot_image_list_count(data_image_list):
    labels = []
    for img in data_image_list:
        labels.append(img.split('.')[-3])
    sns.countplot(x=labels)
    plt.title('Cats and Dogs')
    
plot_image_list_count(train_path)


plot_image_list_count(os.listdir("/kaggle/working/train_data/train"))


train = process_data(train_path,TRAIN_FOLDER)


def show_images(data, isTest=False):
    f, ax = plt.subplots(5,5, figsize=(15,15))
    for i,data in enumerate(data[:25]):
        img_num = data[1]
        img_data = data[0]
        label = np.argmax(img_num)
        if label  == 1: 
            str_label='Dog'
        elif label == 0: 
            str_label='Cat'
        if(isTest):
            str_label="None"
        ax[i//5, i%5].imshow(img_data)
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_title("{}".format(str_label))
    plt.show()

show_images(train)


test = process_data(test_path, TEST_FOLDER, False)


show_images(test,True)


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)     # shape : (20000,224,224,3)
y = np.array([i[1] for i in train])        # shape : (20000, 2)


model = Sequential()
model.add(ResNet50(include_top=False, pooling='max', weights='imagenet', input_shape=(224, 224, 3))) 
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.layers[0].trainable = True


model.summary()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])


X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = TEST_SIZE,random_state = RANDOM_STATE)


Train_history = model.fit(
    X_train,y_train,
    batch_size=BATCH_SIZE,
    epochs = 10,
    verbose = 1,
    validation_data = (X_val,y_val),
)


def plot_accuracy_and_loss(train_model):
    hist = Train_history.history
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(len(acc))
    f, ax = plt.subplots(1,2, figsize=(14,6))
    ax[0].plot(epochs, acc, 'g', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[1].plot(epochs, loss, 'g', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.show()
    
plot_accuracy_and_loss(Train_history)


score = model.evaluate(X_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])


model.save('resnet_model_clf.h5')


# get the predictions for the test data
predicted_classes = model.predict(X_val)
predicted_classes = np.argmax(predicted_classes, axis=1)

# get the indices to be plotted
y_true = np.argmax(y_val,axis=1)


print(classification_report(
    y_true, predicted_classes, 
    target_names=["Cat [1]","Dog [0]"]
))


f, ax = plt.subplots(5,5, figsize=(15,15))
for i,data in enumerate(test[:25]):
    img_num = data[1]
    img_data = data[0]
    orig = img_data
    data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: 
        str_predicted='Dog'
    else: 
        str_predicted='Cat'
    ax[i//5, i%5].imshow(orig)
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title("Predicted:{}".format(str_predicted))    
plt.show()


pred_list = []
img_list = []
for img in tqdm(test):
    img_data = img[0]
    img_idx = img[1]
    data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)
    predicted = model.predict([data])[0]
    img_list.append(img_idx)
    pred_list.append(predicted[1])


submission = pd.DataFrame({'id':img_list , 'label':pred_list})
submission.head()
submission.to_csv("submission.csv", index=False)

