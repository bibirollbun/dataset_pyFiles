import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model,Input
from sklearn.metrics import classification_report
from keras.utils import plot_model
import cv2
from sklearn.metrics import classification_report

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


data = pd.read_csv('../input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
data.head()


data = data.rename(columns={
    ' Usage' : 'Usage',
    ' pixels': 'pixels'})
index_to_label = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
label_to_index = {v: k for k,v in index_to_label.items()}

data['emotion'] = data['emotion'].map(index_to_label)


sns.countplot(data,x='emotion')


data_new = pd.read_csv('../input/fer2013-new/fer2013new.csv')
data_new.head()


data_new['pixels'] = data['pixels']
data = data_new
data.head()


data.shape


data.loc[:, 'label'] = np.argmax(data.iloc[:, 2:12].values, axis=-1)
data.head(5)


data = data[data['label'] < 8]
data.shape


sns.countplot(data,x='Usage')


all_image = [list(map(int, pixel.split(' '))) for pixel in data['pixels'].values]


all_image = np.array(all_image).reshape(-1,48,48)
all_image.shape


label_num = data.iloc[:,2:10].columns
label_num


fig,ax=plt.subplots(2,10,figsize=(12,4))
ax = ax.flatten()

for ax_i in ax:
    index = np.random.randint(0,len(data))
    ax_i.imshow(all_image[index],cmap='gray')
    ax_i.set_title(label_num[data['label'].loc[index]])
    ax_i.axis('off')

plt.tight_layout()


train = data[data['Usage']=='Training']
valid = data[data['Usage']=='PublicTest']
test = data[data['Usage']=='PrivateTest']


index_train = len(train)
index_valid = len(train) + len(valid)

x_train,y_train = all_image[:index_train],train.iloc[:len(train),-1]
x_valid,y_valid = all_image[index_train:index_valid],valid.iloc[:len(valid),-1]
x_test,y_test = all_image[index_valid:],test.iloc[:len(test),-1]

print(x_train.shape,y_train.shape)
print(x_valid.shape,y_valid.shape)
print(x_test.shape,y_test.shape)


x_train = x_train/255.
x_valid = x_valid/255.
x_test = x_test/255.


x_train = np.expand_dims(x_train,axis=-1)
x_valid = np.expand_dims(x_valid,axis=-1)
x_test = np.expand_dims(x_test,axis=-1)

print(x_train.shape,y_train.shape)
print(x_valid.shape,y_valid.shape)
print(x_test.shape,y_test.shape)


datagen = ImageDataGenerator(
    rotation_range=40,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2,
    zoom_range=0.2,  
    horizontal_flip=True,  
    fill_mode='nearest'  
)


def residual_block(input_tensor, out_channel, kernel_size, stride):
        x = layers.Conv2D(out_channel, kernel_size, kernel_initializer='he_normal', strides=stride, padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(out_channel, kernel_size, kernel_initializer='he_normal', strides=1, padding='same')(x) 
        x = layers.BatchNormalization()(x)
        if stride == 1:
            shortcut = input_tensor
        else:
            shortcut = layers.Conv2D(out_channel,1,strides=stride)(input_tensor)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

def ResNet(num_class):
        input = keras.Input(shape=(48,48,1))
        x = layers.Conv2D(32, 5, kernel_initializer='he_normal', padding='same')(input)
        x = layers.BatchNormalization()(x)            
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=3, strides=2)(x)
            
        x = residual_block(x,64,3,2)
        x = residual_block(x,64,3,1)
        x = residual_block(x,128,3,2)
        x = residual_block(x,128,3,1)
        x = residual_block(x,256,3,2)            
        x = residual_block(x,256,3,1)
        x = residual_block(x,512,3,2)            
        x = residual_block(x,512,3,1)
    
        x = layers.MaxPooling2D(2)(x)
        x = layers.Flatten()(x)
        output = layers.Dense(num_class, activation = 'softmax')(x)
        model = keras.Model(input,output,name='ResNet')
        return model


model = ResNet(num_class=8)
model(np.random.rand(32,48,48,1))
model.summary()


learning_rate = 0.001
batch_size = 32
epochs = 100
factor = 0.2

model.compile(
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
    optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor= factor, patience=3, min_lr=1e-7)

H = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    validation_data = (x_valid,y_valid),
    steps_per_epoch = x_train.shape[0] // batch_size,
    epochs = epochs,
    callbacks = [early_stopping , reduce_lr]
)


fig , ax = plt.subplots(1,2, figsize=(10,6))

train_acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
train_loss = H.history['loss']
val_loss = H.history['val_loss']

num_epoch = len(train_acc)
ax[0].plot(range(1,num_epoch+1) , train_acc , label = 'Train')
ax[0].plot(range(1,num_epoch+1) , val_acc, label = 'Val')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

ax[1].plot(range(1,num_epoch+1) , train_loss , label = 'Train')
ax[1].plot(range(1,num_epoch+1) , val_loss, label = 'Val')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend()


model.evaluate(x_test,y_test)


class CNN(keras.Model):
    def __init__(self,num_class):
        super().__init__()
        self.conv1 = self.block_conv2d(32,32)
        self.conv2 = self.block_conv2d(64,64)
        self.conv3 = self.block_conv2d(128,128)

        self.fc = keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation = 'relu'),
            layers.Dense(256, activation = 'relu'),
            layers.Dense(num_class, activation = 'softmax')
        ])

    def block_conv2d(self,in_channel,out_channel):
        return keras.Sequential([
            layers.Conv2D(in_channel,3,kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(out_channel,3,kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(2)
        ])

    def call(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


cnn = CNN(num_class=8)
cnn(np.random.rand(32,48,48,1))
cnn.summary()


learning_rate = 0.001
batch_size = 32
epochs = 50
factor = 0.2

cnn.compile(
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
    optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor= factor, patience=3, min_lr=1e-7)

H = cnn.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    validation_data = (x_valid,y_valid),
    steps_per_epoch = x_train.shape[0] // batch_size,
    epochs = epochs,
    callbacks = [early_stopping , reduce_lr]
)


fig , ax = plt.subplots(1,2, figsize=(10,6))

train_acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
train_loss = H.history['loss']
val_loss = H.history['val_loss']

num_epoch = len(train_acc)
ax[0].plot(range(1,num_epoch+1) , train_acc , label = 'Train')
ax[0].plot(range(1,num_epoch+1) , val_acc, label = 'Val')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

ax[1].plot(range(1,num_epoch+1) , train_loss , label = 'Train')
ax[1].plot(range(1,num_epoch+1) , val_loss, label = 'Val')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend()


cnn.evaluate(x_test,y_test)


#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True,dpi=50)


y_pred = model.predict(x_test)
y_pred= np.argmax(y_pred,axis=-1)
report_resnet = classification_report(y_test,y_pred,target_names = train.iloc[:,2:10].columns)
print(report_resnet)


y_pred = cnn.predict(x_test)
y_pred= np.argmax(y_pred,axis=-1)
report_resnet = classification_report(y_test,y_pred,target_names = train.iloc[:,2:10].columns)
print(report_resnet)

