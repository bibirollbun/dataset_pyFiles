import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import cv2
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_curve

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,ResNet50
from tensorflow.keras import layers
from tensorflow.keras import Input,Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


data = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
data.head()


image_path = []

pattern = '../input/siim-isic-melanoma-classification/jpeg/train'
for i in data['image_name'].values:
    path = os.path.join(pattern,i)
    path += '.jpg'
    image_path.append(path)


data['image_path'] = image_path


sns.countplot(data,x='target')


data['target'].value_counts()


data_class_0 = data[data.target==0].sample(3000,random_state=42)
data_class_1 = data[data.target==1]
new_data = pd.concat([data_class_0,data_class_1])
sns.countplot(new_data,x='target')


new_data['target'].value_counts()


class Data(Sequence):
    def __init__(self, image_path, target, batch_size, target_size=(224, 224), aug=None, shuffle=True, seed=42,**kwargs):
        super().__init__()
        self.image_path = np.array(image_path)
        self.target = np.array(target)
        self.batch_size = batch_size
        self.target_size = target_size
        self.aug = aug
        self.shuffle = shuffle
        self.seed = seed
        np.random.seed(self.seed) 
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.image_path) / self.batch_size))  

    def __getitem__(self, item):
        batch_indices = self.indices[item * self.batch_size: (item + 1) * self.batch_size]
        image_path_batch = [self.image_path[i] for i in batch_indices]
        label_batch = [self.target[i] for i in batch_indices]
        images = [self.load_data(i) for i in image_path_batch]

        images = np.array(images)
        label_batch = np.array(label_batch)
        
        if self.aug:
            auged = self.aug.flow(images, label_batch, shuffle=False)  
            images, label_batch = next(auged)

        return images, label_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_path))
        if self.shuffle:
            np.random.shuffle(self.indices)  

    def load_data(self, image_path):
        img = image.load_img(image_path,target_size=self.target_size) 
        img = image.img_to_array(img)
        img /= 255.0
        return img


def load_and_preprocess_train(image_path, label, target_size=(224, 224)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.image.random_flip_left_right(img) 
    img = tf.image.random_flip_up_down(img)    
    img = tf.image.random_crop(img, size=[target_size[0], target_size[1], 3])
    img = img - [123.68, 116.78, 103.94] 
    img = img / [58.40, 57.12, 57.37] 
    
    return img, label

def load_and_preprocess_valid(image_path, label, target_size=(224, 224)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img - [123.68, 116.78, 103.94]  
    img = img / [58.40, 57.12, 57.37] 
    
    return img, label


datagen = ImageDataGenerator(
    rotation_range=30,  
    width_shift_range=0.2,  
    height_shift_range=0.2, 
    shear_range=0.2,     
    zoom_range=0.2,  
    horizontal_flip=True,
    fill_mode='nearest'
)

dataset = Data(new_data['image_path'].values,new_data['target'].values,batch_size=32,target_size=(224,224),aug=datagen)


for images,labels in dataset:
    fig, ax = plt.subplots(4,8,figsize=(12,6))
    ax = ax.flatten()

    for value,ax_i in enumerate(ax):
        ax_i.imshow(images[value])
        ax_i.set_title(labels[value])
        ax_i.axis('off')
    break


# def binary_focal_loss(alpha=0.25, gamma=2):
#     def loss(y_true, y_pred):
#         bce = K.binary_crossentropy(y_true, y_pred)
#         p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
#         alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
#         modulating_factor = K.pow(1 - p_t, gamma)
#         focal_loss = alpha_factor * modulating_factor * bce
#         return K.mean(focal_loss)
#     return loss

def create_model(lr=0.0001):
    base_model = VGG16(input_shape=(224,224,3),include_top = False)

    for layer in base_model.layers:
        layer.trainable = False
    
    input = base_model.layers[-1].output
    x = layers.GlobalAveragePooling2D()(input)
    x = layers.Dense(512, activation = 'relu')(x)
    output = layers.Dense(1, activation = 'sigmoid')(x)

    model = Model(base_model.input,output)
    print(model.summary())
    model.compile(
        #loss = 'binary_crossentropy',
        loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25,gamma=2.0),
        #binary_focal_loss(alpha=0.2,gamma=2),
        #metrics=[keras.metrics.Recall()],
        metrics = ['acc'],
        optimizer = keras.optimizers.Adam(learning_rate = lr),
    )
    return model

model = create_model()


x_train,x_valid,y_train,y_valid = train_test_split(new_data['image_path'],new_data['target'].values,
                                                  test_size=0.1,
                                                  random_state=42,
                                                  stratify=new_data['target'].values)

print(x_train.shape,y_train.shape)
print(x_valid.shape,y_valid.shape)


#train_generator = Data(x_train, y_train, batch_size=32, aug=datagen, shuffle=True)
#valid_generator = Data(x_valid, y_valid, batch_size=32, aug=None, shuffle=False)


batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(lambda x, y: load_and_preprocess_train(x, y, target_size=(224, 224)), num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
valid_dataset = valid_dataset.map(lambda x, y: load_and_preprocess_valid(x, y, target_size=(224, 224)), num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class_0_weight = len(y_train) / (2 * np.bincount(y_train)[0])
class_1_weight = len(y_train) / (2 * np.bincount(y_train)[1])
class_weight = {0: class_0_weight,1:class_1_weight}
print(class_weight)


epochs = 100
factor = 0.2

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor= factor, patience=3, min_lr=1e-7)

H = model.fit(
    train_dataset,#class_weight = class_weight,
    validation_data = valid_dataset,
    epochs = epochs,
    callbacks = [early_stopping , reduce_lr]
)


fig , ax = plt.subplots(1,2, figsize=(10,6))

train_acc = H.history['acc']
val_acc = H.history['val_acc']
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


y_pred = model.predict(valid_dataset)
y_pred = np.where(y_pred>0.2,1,0)

report = classification_report(y_valid,y_pred)
print(report)


data_test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
data_test.head()


image_path = []

pattern = '../input/siim-isic-melanoma-classification/jpeg/test'
for i in data_test['image_name'].values:
    path = os.path.join(pattern,i)
    path += '.jpg'
    image_path.append(path)

data_test['image_path'] = image_path


valid_dataset = tf.data.Dataset.from_tensor_slices((data_test['image_path'].values, None))
valid_dataset = valid_dataset.map(lambda x, _: load_and_preprocess_valid(x, _, target_size=(224, 224)), num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


prediction = model.predict(valid_dataset)


prediction_ = [i[0] for i in prediction]


submit = pd.DataFrame({
    'image_name' : data_test['image_name'],
    'target' : prediction_
})


submit.to_csv('submission.csv',index=False)

