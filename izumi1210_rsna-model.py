import os
import time
import random 
import pathlib     
import itertools 
from glob import *  
from tqdm import *

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')

import matplotlib.pyplot as plt
%matplotlib inline

from skimage.color import *          
from skimage.morphology import *     
from skimage.transform import *       
from sklearn.model_selection import * 
from skimage.io import *         
from sklearn.metrics import *         
from sklearn.utils import *         
from sklearn.preprocessing import *

import tensorflow as tf                  
from tensorflow import keras              
from tensorflow.keras import *               
from tensorflow.keras import backend as K    
from tensorflow.keras.models import *     
from tensorflow.keras.preprocessing.image import * 
from tensorflow.keras.optimizers import *    
from tensorflow.keras.callbacks import *   
from tensorflow.keras.layers import *      
from tensorflow.keras.applications import *  
from tensorflow.keras.utils import * 


import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


!rm -rf /kaggle/working/*


RSNA_512_path = '/kaggle/input/rsna-breast-cancer-512-pngs'


df_train = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/train.csv')


#Lọc dữ liệu
DF_train = df_train[df_train['biopsy'] == 1].reset_index(drop = True)
print(len(DF_train))
DF_train.sample(20)


#Cân bằng dữ liệu
DF_train = DF_train.groupby(['cancer']).apply(lambda x: x.sample(1500, replace = True)).reset_index(drop = True)
print('New Data Size:', DF_train.shape[0])


DF_train['cancer'].value_counts()


# Create the path to each image.
for i in range(len(DF_train)):
    DF_train.loc[i, 'path'] = os.path.join(RSNA_512_path + '/' + str(DF_train.loc[i, 'patient_id']) + '_' + str(DF_train.loc[i, 'image_id']) + '.png')


train_datagen = ImageDataGenerator(
    rescale = 1./255.,
    rotation_range=10,       # xoay nhẹ
    width_shift_range=0.05,  # dịch ngang
    height_shift_range=0.05, # dịch dọc
    zoom_range=0.1,          # phóng to/thu nhỏ nhẹ
    horizontal_flip=True,    # lật ảnh (nếu hợp lý)
    fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale = 1./255.,)


train_df, val_df = train_test_split(DF_train, 
                                 test_size = 0.1, 
                                 stratify = DF_train[['cancer']])
#Tạo df chứa bình thường
train_df_normal = train_df[train_df['cancer'] == 0].reset_index(drop = True)
val_df_normal = val_df[val_df['cancer'] == 0].reset_index(drop = True)
#Tạo df chứa ut
train_df_cancer = train_df[train_df['cancer'] == 1].reset_index(drop = True)
val_df_cancer = val_df[val_df['cancer'] == 1].reset_index(drop = True)


import shutil
# Khai báo destination directory.
destination_dir = '/kaggle/working/train'
destination_dir_sub = '/kaggle/working/train/normal'

# Tạo nếu destination directory chưa tồn tại.
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

if not os.path.exists(destination_dir_sub):
    os.makedirs(destination_dir_sub)   
    
# Copy ảnh vào destination directory.
for path in train_df_normal['path']:
    shutil.copy2(path, destination_dir_sub)


destination_dir = '/kaggle/working/train'
destination_dir_sub = '/kaggle/working/train/cancer'

# Create the destination directory if it doesn't exist.
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

if not os.path.exists(destination_dir_sub):
    os.makedirs(destination_dir_sub)   
    
# Copy the images to the destination directory.
for path in train_df_cancer['path']:
    shutil.copy2(path, destination_dir_sub)


# Define the destination directory.
destination_dir = '/kaggle/working/val'
destination_dir_sub = '/kaggle/working/val/normal'

# Create the destination directory if it doesn't exist.
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

if not os.path.exists(destination_dir_sub):
    os.makedirs(destination_dir_sub)   
    
# Copy the images to the destination directory.
for path in val_df_normal['path']:
    shutil.copy2(path, destination_dir_sub)


# Define the destination directory.
destination_dir = '/kaggle/working/val'
destination_dir_sub = '/kaggle/working/val/cancer'

# Create the destination directory if it doesn't exist.
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

if not os.path.exists(destination_dir_sub):
    os.makedirs(destination_dir_sub)   
    
# Copy the images to the destination directory.
for path in val_df_cancer['path']:
    shutil.copy2(path, destination_dir_sub)


train_path = '/kaggle/working/train'
val_path = '/kaggle/working/val'

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size = (256, 256),
    batch_size = 16,
    class_mode = 'binary',
    shuffle = False
)
validation_generator = val_datagen.flow_from_directory(
    val_path,
    target_size = (256, 256),
    batch_size = 16,
    class_mode = 'binary',
    shuffle = False
)

train_directory_iterator = train_generator
validation_directory_iterator = validation_generator

def segmentation_generator(directory_iterator):
    while True:
        images, labels = next(directory_iterator)
        masks = np.ones((labels.shape[0], IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
        masks *= labels[:, None, None, None]
        yield images, masks

train_steps_per_epoch = len(train_directory_iterator)
val_steps_per_epoch = len(validation_directory_iterator)

train_generator = segmentation_generator(train_directory_iterator)
validation_generator = segmentation_generator(validation_directory_iterator)


def mean_iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 0, 1)
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection

    iou = tf.math.divide_no_nan(intersection, union)
    return tf.reduce_mean(iou)


def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))


from tensorflow.keras.losses import BinaryFocalCrossentropy
focal_loss = BinaryFocalCrossentropy(gamma=2.0)

def dice_focal_loss(y_true, y_pred):
    # Dice Loss
    smooth = 1e-6
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    dice_loss = 1 - dice
    
    # Focal Loss
    focal = focal_loss(y_true, y_pred)
    
    # Tổng hợp 2 loss (có thể điều chỉnh trọng số)
    return 0.5 * dice_loss + 0.5 * focal



def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


# Build U-Net model
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

def unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(shape=input_shape)
    s = Lambda(lambda x: x / 255.0)(inputs)

    c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Activation('elu')(c1)
    c1 = Dropout(0.05)(c1)
    c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('elu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('elu')(c2)
    c2 = Dropout(0.05)(c2)
    c2 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('elu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('elu')(c3)
    c3 = Dropout(0.05)(c3)
    c3 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('elu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('elu')(c4)
    c4 = Dropout(0.05)(c4)
    c4 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('elu')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation('elu')(c5)
    c5 = Dropout(0.05)(c5)
    c5 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('elu')(c5)

    # --- Decoder ---
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Dropout(0.05)(c6)
    c6 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Dropout(0.05)(c7)
    c7 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    c8 = Dropout(0.05)(c8)
    c8 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('elu')(c9)
    c9 = Dropout(0.05)(c9)
    c9 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('elu')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            metrics.AUC(name='auc'),
            metrics.Recall(name='recall'),
            metrics.Precision(name='precision'),
            mean_iou,
            dice_coef
        ]
    )
    return model


IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_CHANNELS = 3
num_labels = 1  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
batch_size = 16


# def initial_block(inputs, num_filters):
#     conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.Activation("relu")(conv)
#     return conv

# def encoder_block(inputs, num_filters):
#     shortcut = inputs
#     conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.Activation("relu")(conv)
#     conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(conv)
#     conv = tf.keras.layers.BatchNormalization()(conv)

#     shortcut = tf.keras.layers.Conv2D(num_filters, 1, padding="same")(shortcut)
#     shortcut = tf.keras.layers.BatchNormalization()(shortcut)

#     conv = tf.keras.layers.Add()([conv, shortcut])
#     conv = tf.keras.layers.Activation("relu")(conv)

#     return conv

# def downsampling_block(inputs, num_filters):
#     shortcut = inputs
#     conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same", strides=2)(inputs)
#     conv = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.Activation("relu")(conv)
#     conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(conv)
#     conv = tf.keras.layers.BatchNormalization()(conv)

#     shortcut = tf.keras.layers.Conv2D(num_filters, 1, padding="same", strides=2)(shortcut)
#     shortcut = tf.keras.layers.BatchNormalization()(shortcut)

#     conv = tf.keras.layers.Add()([conv, shortcut])
#     conv = tf.keras.layers.Activation("relu")(conv)

#     return conv

# def decoder_block(inputs, skip, num_filters):
#     up_conv = tf.keras.layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(inputs)
#     conv = tf.keras.layers.Concatenate()([up_conv, skip])
#     conv = encoder_block(conv, num_filters)
#     return conv

# def LinkNet(input_shape, num_classes=1):
#     inputs = tf.keras.layers.Input(input_shape)

#     initial = initial_block(inputs, 64)

#     # Encoder
#     down1 = downsampling_block(initial, 64)
#     down2 = downsampling_block(down1, 128)
#     down3 = downsampling_block(down2, 256)
#     down4 = downsampling_block(down3, 512)

#     # Decoder
#     up4 = decoder_block(down4, down3, 256)
#     up3 = decoder_block(up4, down2, 128)
#     up2 = decoder_block(up3, down1, 64)
#     up1 = decoder_block(up2, initial, 64)

#     outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="sigmoid")(up1)

#     model = tf.keras.models.Model(inputs, outputs, name="LinkNet")
#     return model


def conv_block(x, filter_size, size, dropout, batch_norm=False):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv


def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn


def Attention_UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="Attention_UNet")
    return model


model = Attention_UNet(input_shape)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        metrics.AUC(name='auc'),
        metrics.Recall(name='recall'),
        metrics.Precision(name='precision'),
        mean_iou,
        dice_coef
    ]
)


# Create U-Net base model with correct parameters
#model = unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

#for layer in model.layers:
#    layer.trainable = False


# model = LinkNet((256,256,3))
# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss='binary_crossentropy',
#     metrics=[
#         'accuracy',
#         metrics.AUC(name='auc'),
#         metrics.Recall(name='recall'),
#         metrics.Precision(name='precision'),
#         mean_iou,
#         dice_coef
#     ]
# )



model.summary()


callbacks = [
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_mean_iou',
        mode='max',
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor='val_mean_iou',
        mode='max',
        patience=10,
        verbose=1,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

history = model.fit(
    x=train_generator,
    validation_data=validation_generator,
    steps_per_epoch=train_steps_per_epoch,
    validation_steps=val_steps_per_epoch,
    epochs=50,
    callbacks=callbacks
)


auc = history.history['auc']
val_auc = history.history['val_auc']

recall = history.history['recall']
val_recall = history.history['val_recall']

mean_iou = history.history['mean_iou']
val_mean_iou = history.history['val_mean_iou']

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize = (15,10))

plt.subplot(2, 4, 1)
plt.plot(auc, label = "Training AUC")
plt.plot(val_auc, label = "Validation AUC")
plt.ylim(0, 1)
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.title("Training vs Validation AUC")
plt.xlabel('epoch')
plt.ylabel('AUC')

plt.subplot(2, 4, 2)
plt.plot(recall, label = "Training Recall")
plt.plot(val_recall, label = "Validation Recall")
plt.ylim(0, 1)
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.title("Training vs Validation Recall")
plt.xlabel('epoch')
plt.ylabel('AUC')

plt.subplot(2, 4, 3)
plt.plot(mean_iou, label = "Training Mean IOU")
plt.plot(val_mean_iou, label = "Validation Mean IOU")
plt.ylim(0, 1)
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.title("Training vs Validation Mean IOU")
plt.xlabel('epoch')
plt.ylabel('Mean IOU')

plt.subplot(2, 4, 4)
plt.plot(loss, label = "Training Loss")
plt.plot(val_loss, label = "Validation Loss")
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.title("Training vs Validation Loss")
plt.xlabel('epoch')
plt.ylabel('loss')


from tensorflow.keras.models import load_model
model = load_model('/kaggle/working/best_model.keras', safe_mode=False, custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef})


pred = model.predict(validation_generator, steps=val_steps_per_epoch, verbose=1)


print(pred.shape)
pred


print("Pred min:", pred.min())
print("Pred max:", pred.max())
print("Pred mean:", pred.mean())


binary_pred = (pred >= 0.5).astype(np.uint8)


print(binary_pred)
print(binary_pred.shape)


pixel_fraction = binary_pred.mean(axis=(1, 2, 3))
y_pred = (pixel_fraction > 0.01).astype(np.uint8)

print(y_pred.shape)
print(np.unique(y_pred, return_counts=True))
print(y_pred)


pd.Series(y_pred).value_counts()


y_true = validation_directory_iterator.classes


print(y_true)
print(y_true.shape)


pd.Series(y_true).value_counts()


cm = confusion_matrix(y_true, y_pred)


# Define the class names.
class_names = ['Cancer', 'Normal']

# Create the heatmap with class names as tick labels.
ax = sns.heatmap(cm, annot = True, fmt = '.0f', cmap = "Blues", annot_kws = {"size": 16},\
           xticklabels = class_names, yticklabels = class_names)

# Set the axis labels.
ax.set_xlabel("Prediction")
ax.set_ylabel("Truth")


model.save('mammography_pred_model_finetuning.h5')

