# References
# https://www.kaggle.com/code/seoyunje/keras-segmentation-unet-linknet/notebook
# https://www.kaggle.com/code/ateplyuk/keras-starter-segmentation


!pip install -U -q segmentation-models
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"


import os
import json
import gc

import cv2
import keras
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, Conv2DTranspose, MaxPool2D, Concatenate
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


train_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
train_df = train_df.dropna()


print(len(train_df))


def rle2mask(rle, imgshape):
    width = imgshape[0]
    height= imgshape[1]
    
    mask= np.zeros(width*height).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1

    return mask.reshape((width, height), order='F')


import cv2


fig=plt.figure(figsize=(20,100))
columns = 2
rows = 50
for i in range(1, 100+1):
    fig.add_subplot(rows, columns, i)    
    image_id = train_df['ImageId'].iloc[i]
    class_id = train_df['ClassId'].iloc[i]
    img = cv2.imread('../input/severstal-steel-defect-detection/train_images/'+ image_id)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = rle2mask(train_df['EncodedPixels'].iloc[i], img.shape  )
    img[mask==1,0] = 255
    
    plt.imshow(img)
    
plt.show()


print(np.shape(img))


import tensorflow.keras.backend as K
def dice_metric(y_true, y_pred, smooth=1):
    true_sum = K.sum(y_true)
    pred_sum = K.sum(y_pred)
    intersect = K.sum(y_true*y_pred)
    
    dice = (2*intersect) /(true_sum + pred_sum + smooth)
    return dice


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, shuffle=False, mode='train', transform=False, preprocess=None):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.mode = mode
        self.transform = transform 
        self.on_epoch_end()
        self.preprocess = preprocess
    
    def __len__(self):
        return int(np.ceil(len(self.data)/self.batch_size))
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle: 
            np.random.shuffle(self.indexes)
    
    def __getitem__(self,index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X,y

    def __data_generation(self,indexes):
        # X = np.zeros(len(indexes))
        images = np.zeros((self.batch_size, 256, 1600, 3), dtype=np.uint8)
        masks = np.zeros((self.batch_size, 256, 1600, 1), dtype=np.float32)
        
        for j,i in enumerate(indexes):
            fn = self.data['ImageId'].iloc[i].split('_')[0]
            img = cv2.imread( '../input/severstal-steel-defect-detection/train_images/'+fn )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
            mask = rle2mask(self.data['EncodedPixels'].iloc[i], img.shape)                                    
            images[j,] = img 
            masks[j,:,:,0] = mask
        return images, masks


train_gen = DataGenerator(train_df.iloc[0:5000,:], shuffle=True, batch_size=32, transform=True)
valid_gen = DataGenerator(train_df.iloc[5000:, :], shuffle=False, batch_size=32, mode='valid')


from segmentation_models import Unet
import segmentation_models

unet_model = Unet('resnet18', input_shape=(256,1600,3), classes=1, activation='sigmoid')
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_metric])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_dice_metric', mode='max', patience=3)
checkpoint = ModelCheckpoint(f"Unet.weights.h5",monitor="val_dice_metric",save_best_only=True,save_weights_only=True,mode="auto",verbose=1)

# There are LearningRate Scheduler,ReduceLROnPlateau in tf.kears.callbacks
reduce_lr = ReduceLROnPlateau(monitor = 'val_dice_metric', factor = 0.1, patience = 1, min_delta = 0.01,
                              mode='auto',verbose=1)


K.clear_session()
history = unet_model.fit(train_gen, 
                         verbose=1,
                         validation_data = valid_gen, 
                         epochs=5, 
                         callbacks=[es,checkpoint,reduce_lr])


print(len(train_df))


fig=plt.figure(figsize=(20,100))
columns = 2
rows = 32
img_list = []
for i in range(1, 31+1):
    fig.add_subplot(rows, columns, i)    
    image_id = train_df['ImageId'].iloc[4999+i]
    class_id = train_df['ClassId'].iloc[4999+i]
    img = cv2.imread('../input/severstal-steel-defect-detection/train_images/'+ image_id)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = rle2mask(train_df['EncodedPixels'].iloc[4999+i], img.shape)
    img_list.append(img.copy())
    img[mask==1,0] = 255
    
    plt.imshow(img)
    
plt.show()


fig=plt.figure(figsize=(20,100))
columns = 2
rows = 32
img_list = []
for i in range(1, 32):
    fig.add_subplot(rows, columns, i)    
    image_id = train_df['ImageId'].iloc[4999+i]
    class_id = train_df['ClassId'].iloc[4999+i]
    img = cv2.imread('../input/severstal-steel-defect-detection/train_images/'+ image_id)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = rle2mask(train_df['EncodedPixels'].iloc[4999+i], img.shape)
    
    img[mask==1,0] = 255    
    plt.imshow(img)
    
plt.show()



predict = unet_model.predict(np.array(img_list))



plt.imshow(img_list[0])


plt.title('Predicted segments')
plt.imshow(predict[1])
plt.show()

plt.title('Actual image with segments')
img = img_list[1]
mask = rle2mask(train_df['EncodedPixels'].iloc[4999+2], img.shape)
img[mask==1,0] = 255
plt.imshow(img)
plt.show()



