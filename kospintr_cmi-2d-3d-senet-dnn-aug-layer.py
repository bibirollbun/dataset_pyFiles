## For TPU environment (install missing packages / reinstall tensorflow to solve NaN topic during training / restart kernel)

import IPython
import tensorflow as tf

if len(tf.config.experimental.list_logical_devices('TPU'))>0:
    !pip install -q tensorflow-tpu -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force-reinstall
    !pip install -q pydot
    !pip install -q -U keras-tuner
    !pip install xgboost
    IPython.Application.instance().kernel.do_shutdown(True)


## Import packages

import os
import time
import polars as pl
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow.keras.regularizers import L2
print('Tensorflow version: ' + tf.__version__)


## Detect hardware (CPU/GPU/TPU), setup environment and return appropriate distribution strategy

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu='local') # set tpu is local as it should be available in the VM
    print('✅ Running on TPU ', tpu.master())
except:
    print('❌ Using CPU/GPU')
    tpu = None

if tpu:
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)


## Read csv files

path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/'
trainval = pd.read_csv(path + "train.csv")
trainval_demographics = pd.read_csv(path + "train_demographics.csv")
test = pd.read_csv(path + "test.csv")
test_demographics = pd.read_csv(path + "test_demographics.csv")
#trainval = trainval[:50000] # To save time during debugging/expermenting


## Spliting the train data into train and val

cv = StratifiedGroupKFold(n_splits=10)
for fold, (train_idx, val_idx) in enumerate(cv.split(trainval, trainval['gesture'],
                                                     groups=np.array(trainval['sequence_id'], dtype='<U10'))):
    print(train_idx)
    print(val_idx)
    train, val = trainval.iloc[train_idx], trainval.iloc[val_idx]
    print(f"✅ Fold {fold}: Train size = {len(train_idx)}, Val size = {len(val_idx)}")
    break  # Use only the first fold for now


## Generate transformation dictionaries for classes

class2bfrb = {'Cheek - pinch skin': 1, 'Forehead - pull hairline': 1, 'Write name on leg': 0,
              'Feel around in tray and pull out an object': 0, 'Neck - scratch': 1,
              'Neck - pinch skin': 1, 'Eyelash - pull hair': 1, 'Eyebrow - pull hair': 1,
              'Forehead - scratch': 1, 'Above ear - pull hair': 1, 'Wave hello': 0, 'Write name in air': 0,
              'Text on phone': 0, 'Pull air toward your face': 0, 'Pinch knee/leg skin': 0,
              'Scratch knee/leg skin': 0, 'Drink from bottle/cup': 0, 'Glasses on/off': 0}
class2id = {key: i for i, key in enumerate(class2bfrb.keys())}
id2class = {i: value for i, value in enumerate(class2bfrb.keys())}
id2bfrb = {i: value for i, value in enumerate(class2bfrb.values())}


## Extract sensor related data

scalers = {'scaler_acc' : MinMaxScaler().fit([[-50],[50]]), 'scaler_rot' : MinMaxScaler().fit([[-1],[1]]),
           'scaler_tof' : MinMaxScaler().fit([[-1],[249]]), 'scaler_thm' : MinMaxScaler().fit([[-1],[40]])}
seq_len=80

def extract_time_series_data(data, training):
    start_time = time.time()
    poss_tensor_padded_list = []
    for i in range(1, 6):
       globals()[f'tof_thm_tensor_padded_list{i}'] = []
    
    for s, sequence_id in enumerate(data['sequence_id'].unique()):
        if s%1000 == 0: # Show progress
            start_time2 = time.time()
        
        # Filter data for one sequence and truncate if longer than seq_len
        data_seq = data[data['sequence_id'] == sequence_id].ffill().bfill().fillna(0)
        if len(data_seq.index) > seq_len:
            data_seq = data_seq[-seq_len:]
            
        # Extract position/orientation related data
        pos_tensor_padded_list = []
        for column in ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']:
            pos_array  = pd.DataFrame(data_seq, columns=[column]).to_numpy()
            if column in ['acc_x', 'acc_y', 'acc_z']:
                pos_array = scalers['scaler_acc'].transform(pos_array)
            pos_tensor = tf.convert_to_tensor(pos_array.reshape(1,1,-1,1), dtype=tf.float32)
            pad_length = seq_len - pos_tensor.shape[2] # Max length in train dataset is 71 min length is 3 after removing transitions
            paddings_2d = tf.constant([[0, 0,], [0, 0], [pad_length, 0], [0, 0]])
            pos_tensor_padded = tf.pad(pos_tensor, paddings_2d, "CONSTANT")  # Shape (1,1,seq_len,1)
            pos_tensor_padded_list.append(pos_tensor_padded)
        poss_tensor_padded = tf.concat(pos_tensor_padded_list, axis=1) # Shape (1,7,seq_len,1)
        poss_tensor_padded_list.append(poss_tensor_padded)

        # Extract ToF/THM related data
        for i in range(1, 6):
            # Extract ToF data
            image_arrays  = pd.DataFrame(data_seq, columns=[f'tof_{i}_v{j}' for j in range(64)]).to_numpy()
            image_arrays = (image_arrays + 1)/250
            image3d_tensor = tf.convert_to_tensor(image_arrays.reshape(1,8,8,-1,1), dtype=tf.float32)
            paddings_3d = tf.constant([[0, 0,], [0, 0], [0, 0], [pad_length, 0], [0, 0]])
            image3d_tensor_padded = tf.pad(image3d_tensor, paddings_3d, "CONSTANT")  # Shape (1,8,8,seq_len,1)
            # Extract THM data
            temp_array  = pd.DataFrame(data_seq, columns=[f'thm_{i}']).to_numpy()
            temp_array = scalers['scaler_thm'].transform(temp_array)
            temp_tensor = tf.convert_to_tensor(temp_array.reshape(1,1,1,-1,1), dtype=tf.float32)
            multiply_factor = tf.constant([1,8,8,1,1], tf.int32)
            temp_tensor = tf.tile(temp_tensor, multiply_factor)
            temp_tensor_padded = tf.pad(temp_tensor, paddings_3d, "CONSTANT") # Shape (1,8,8,seq_len,1)
            tof_thm_tensor_padded = tf.concat([image3d_tensor_padded, temp_tensor_padded], axis=4) # Shape (1,8,8,seq_len,2)
            globals()[f'tof_thm_tensor_padded_list{i}'].append(tof_thm_tensor_padded)
        if s%1000 == 0: print("Time taken for 1000 seqs: %.2fs" % (time.time() - start_time2))
    
    data_tensor_0 = tf.concat(poss_tensor_padded_list, axis=0)
    for i in range(1, 6):
        globals()[f'data_tensor_{i}'] = tf.concat(globals()[f'tof_thm_tensor_padded_list{i}'], axis=0)
    print("Time taken: %.2fs" % (time.time() - start_time))
    return data_tensor_0, data_tensor_1, data_tensor_2, data_tensor_3, data_tensor_4, data_tensor_5

train_tensor_0, train_tensor_1, train_tensor_2, train_tensor_3, train_tensor_4, train_tensor_5 = extract_time_series_data(train, True)
val_tensor_0, val_tensor_1, val_tensor_2, val_tensor_3, val_tensor_4, val_tensor_5 = extract_time_series_data(val, False)
test_tensor_0, test_tensor_1, test_tensor_2, test_tensor_3, test_tensor_4, test_tensor_5 = extract_time_series_data(test, False)


## Extract Subject related data

scalers_sub = {'scaler_age' : StandardScaler(), 'scaler_height_cm' : StandardScaler(),
               'scaler_shoulder_to_wrist_cm' : StandardScaler(), 'scaler_elbow_to_wrist_cm' : StandardScaler()}

def extract_subject_related_data(data, data_demographics, scalers, training):
    # Filter subject related data that every sequence id apperas only once
    data_fil = data[['sequence_id', 'subject']].drop_duplicates(subset=['sequence_id'])
    data_fil = data_fil.merge(data_demographics, how='left', left_on='subject', right_on='subject')
    
    # Scaling numeric values
    data_fil[['adult_child', 'sex', 'handedness']] = data_fil[['adult_child', 'sex', 'handedness']].astype(dtype=np.float32)
    if training:
        scalers['scaler_age'].fit(data_fil[['age']])
        scalers['scaler_height_cm'].fit(data_fil[['height_cm']])
        scalers['scaler_shoulder_to_wrist_cm'].fit(data_fil[['shoulder_to_wrist_cm']])
        scalers['scaler_elbow_to_wrist_cm'].fit(data_fil[['elbow_to_wrist_cm']])
    data_fil[['age']] = scalers['scaler_age'].transform(data_fil[['age']])
    data_fil[['height_cm']] = scalers['scaler_height_cm'].transform(data_fil[['height_cm']])
    data_fil[['shoulder_to_wrist_cm']] = scalers['scaler_shoulder_to_wrist_cm'].transform(data_fil[['shoulder_to_wrist_cm']])
    data_fil[['elbow_to_wrist_cm']] = scalers['scaler_elbow_to_wrist_cm'].transform(data_fil[['elbow_to_wrist_cm']])
    data_fil.drop(columns=['sequence_id', 'subject'], inplace=True)
    data_tensor = tf.convert_to_tensor(data_fil, dtype=np.float32)
    return data_tensor, scalers # Shape (Non 7)

train_tensor_6, scalers_sub = extract_subject_related_data(train, trainval_demographics, scalers_sub, True)
val_tensor_6, _ = extract_subject_related_data(val, trainval_demographics, scalers_sub, False)
test_tensor_6, _ = extract_subject_related_data(test, test_demographics, scalers_sub, False)


## Extract labels

def extract_labels(data):
    data_fil = data[['sequence_id', 'gesture']]
    data_fil = data_fil.drop_duplicates(subset=['sequence_id'])
    data_fil['gesture'] = data_fil['gesture'].map(class2id)
    data_fil.drop(columns=['sequence_id'], inplace=True)
    labels_tensor = tf.convert_to_tensor(data_fil, dtype=np.int32)
    labels_tensor = tf.squeeze(labels_tensor)
    return labels_tensor

labels_train = extract_labels(train)
labels_val = extract_labels(val)


## Explore tensors for shape/min/max/nan values

print('train_tensor_0 shape is: '+ str(train_tensor_0.shape))
print('train_tensor_1 shape is: '+ str(train_tensor_1.shape))
print('train_tensor_2 shape is: '+ str(train_tensor_2.shape))
print('train_tensor_3 shape is: '+ str(train_tensor_3.shape))
print('train_tensor_4 shape is: '+ str(train_tensor_4.shape))
print('train_tensor_5 shape is: '+ str(train_tensor_5.shape))
print('train_tensor_6 shape is: '+ str(train_tensor_6.shape))
print('labels_train shape is: '+ str(labels_train.shape))

for i in range(0, 7):
    test_t = globals()[f'train_tensor_{i}']
    if tf.math.reduce_any(tf.math.is_nan(globals()[f'train_tensor_{i}'])):
        print(f'train_tensor_{i} is NaN')
    else:
        print(f'train_tensor_{i} is NOT NaN')
    print('Its min value is: '+ str(tf.reduce_min(test_t)))
    print('Its max value is: '+ str(tf.reduce_max(test_t)))


# scale = 2
# data = tf.clip_by_value(tf.cast(tf.random.normal(shape=[10000], mean=80,  stddev=10*scale, dtype=tf.float32), dtype=tf.int64),
#                  round(10), round(80*scale)).numpy()
# plt.hist(data, bins=150)


## Data augmentation layer for all 2D/3D tensors during training or as pre-processing

build_in_augmentation = True

class TranslationLayer(tf.keras.layers.Layer):
    def __init__(self, max_roll, max_acc_scale, max_t_scale, name='augmentation'):
        super(TranslationLayer, self).__init__(name=name)
        self.max_roll = max_roll
        self.max_acc_scale = max_acc_scale
        self.max_t_scale = max_t_scale
        #with tf.device('/GPU:0'): # Workaround to solve error in GPU environment
        self.roll = tf.Variable(0, name='roll', dtype=tf.int64, trainable=False)
        self.scale = tf.Variable(1.0, name='scale', dtype=tf.float32, trainable=False)
        self.t_scale = tf.Variable(80, name='t_scale', dtype=tf.int64, trainable=False)
    
    def call(self, tensors, training=None):
        if training:
            if self.max_roll > 0:
                self.roll.assign(tf.random.uniform(shape=(), minval=-self.max_roll, maxval=0, dtype=tf.int64))
            self.scale.assign(tf.random.uniform(shape=(), minval=1/self.max_acc_scale, maxval=self.max_acc_scale, dtype=tf.float32))
            self.t_scale.assign(tf.random.uniform(shape=(), minval=round(80/self.max_t_scale),
                                                  maxval=round(80*self.max_t_scale), dtype=tf.int64))
            # self.t_scale.assign(tf.clip_by_value(tf.cast(
            #     tf.random.normal(shape=(), mean=80,  stddev=10*self.max_t_scale, dtype=tf.float32), dtype=tf.int64),
            #                                      round(40/self.max_t_scale), round(80*self.max_t_scale)))
            pass
        else:
            self.roll.assign(0)
            self.scale.assign(1)
            self.t_scale.assign(80)
            pass
        tensors_trans = []
        for tensor in tensors:
            tensor_rev = tf.reverse(tensor, axis=[-2])
            tensor_ext = tf.concat([tensor_rev, tensor, tensor_rev, tensor, tensor_rev], axis=-2)
            if tensor.ndim == 5:
                tensor_trans = tf.roll(tensor_ext, self.roll, axis=-2)
                tensor_trans = tf.reshape(tensor_trans, (-1, 64, 400, 2))[:, :, -(self.t_scale+seq_len):-seq_len, :]
                tensor_trans = tf.image.resize(tensor_trans, [64, 80])
                tensor_trans = tf.reshape(tensor_trans, (-1, 8, 8, 80, 2))
            else:

                tensor_trans = tf.roll(tensor_ext, self.roll, axis=-2)
                tensor_trans = tensor_trans[:, :, -(self.t_scale+seq_len):-seq_len, :]
                tensor_trans = tf.image.resize(tensor_trans, [7, 80])              
                #tensor_trans_acc, tensor_trans_rot = tf.split(tensor_trans, num_or_size_splits=[3,4], axis=1)
                #tensor_trans_acc = tensor_trans_acc*self.scale
                #tensor_trans = tf.concat([tensor_trans_acc, tensor_trans_rot], axis=1)
            tensors_trans.append(tensor_trans)
        #tf.print(self.roll)
        #tf.print(self.scale)
        #tf.print(self.t_scale)
        return tuple(tensors_trans)


## Data augmentation as pre-processing for experimental purposes (image.resize is not working on GPU/TPU fit or tuner))

if not build_in_augmentation:
    max_roll_layer = 0
    max_acc_scale = 1
    max_t_scale = 2
    train_tensor_00_aug_list = []
    train_tensor_0_aug_list = []
    train_tensor_1_aug_list = []
    train_tensor_2_aug_list = []
    train_tensor_3_aug_list = []
    train_tensor_4_aug_list = []
    train_tensor_5_aug_list = []
    for i in range(train_tensor_0.shape[0]):
        input_00a = TranslationLayer(max_roll_layer, max_acc_scale, max_t_scale)(
            [train_tensor_0[i:i+1]], training=True)
        input_0a, input_1a, input_2a, input_3a, input_4a, input_5a = TranslationLayer(max_roll_layer, max_acc_scale, max_t_scale)(
            [train_tensor_0[i:i+1], train_tensor_1[i:i+1], train_tensor_2[i:i+1],
             train_tensor_3[i:i+1], train_tensor_4[i:i+1], train_tensor_5[i:i+1]], training=True)
        train_tensor_00_aug_list.append(input_00a)
        train_tensor_0_aug_list.append(input_0a)
        train_tensor_1_aug_list.append(input_1a)
        train_tensor_2_aug_list.append(input_2a)
        train_tensor_3_aug_list.append(input_3a)
        train_tensor_4_aug_list.append(input_4a)
        train_tensor_5_aug_list.append(input_5a)
    train_tensor_00_aug = tf.concat(train_tensor_00_aug_list, axis=0)
    train_tensor_00_aug = tf.squeeze(train_tensor_00_aug, axis=[1])
    train_tensor_0_aug = tf.concat(train_tensor_0_aug_list, axis=0)
    train_tensor_1_aug = tf.concat(train_tensor_1_aug_list, axis=0)
    train_tensor_2_aug = tf.concat(train_tensor_2_aug_list, axis=0)
    train_tensor_3_aug = tf.concat(train_tensor_3_aug_list, axis=0)
    train_tensor_4_aug = tf.concat(train_tensor_4_aug_list, axis=0)
    train_tensor_5_aug = tf.concat(train_tensor_5_aug_list, axis=0)


## Create datasets with original data and only IMU data (as in hidden test data)

SEED=33
batch_size=64
batch_size_val=64

# Create empty ToF/THM tensors for data augmentation
train_tensor_e = tf.fill(dims=train_tensor_1.shape, value=0.0)
val_tensor_e = tf.fill(dims=val_tensor_1.shape, value=0.0)

# Create original dataset and same dataset with no ToF/THM sensor values for train and validation
dataset_train_orig = tf.data.Dataset.from_tensor_slices(({"input_0": train_tensor_0, "input_1": train_tensor_1,
                                                          "input_2": train_tensor_2, "input_3": train_tensor_3,
                                                          "input_4": train_tensor_4, "input_5": train_tensor_5,
                                                          "input_6": train_tensor_6}, labels_train))

dataset_train_e = tf.data.Dataset.from_tensor_slices(({"input_0": train_tensor_0, "input_1": train_tensor_e,
                                                       "input_2": train_tensor_e, "input_3": train_tensor_e,
                                                       "input_4": train_tensor_e, "input_5": train_tensor_e,
                                                       "input_6": train_tensor_6}, labels_train))

if not build_in_augmentation:
    dataset_train_e2 = tf.data.Dataset.from_tensor_slices(({"input_0": train_tensor_0_aug, "input_1": train_tensor_1_aug,
                                                            "input_2": train_tensor_2_aug, "input_3": train_tensor_3_aug,
                                                            "input_4": train_tensor_4_aug, "input_5": train_tensor_5_aug,
                                                            "input_6": train_tensor_6}, labels_train))
    dataset_train_e = dataset_train_e.concatenate(dataset_train_e2)

dataset_val_orig = tf.data.Dataset.from_tensor_slices(({"input_0": val_tensor_0, "input_1": val_tensor_1,
                                                        "input_2": val_tensor_2, "input_3": val_tensor_3,
                                                        "input_4": val_tensor_4, "input_5": val_tensor_5,
                                                        "input_6": val_tensor_6}, labels_val))

dataset_val_e = tf.data.Dataset.from_tensor_slices(({"input_0": val_tensor_0, "input_1": val_tensor_e,
                                                     "input_2": val_tensor_e, "input_3": val_tensor_e,
                                                     "input_4": val_tensor_e, "input_5": val_tensor_e,
                                                     "input_6": val_tensor_6}, labels_val))

# Create dataset for testing (dummy)
dataset_test = tf.data.Dataset.from_tensor_slices(({"input_0": test_tensor_0, "input_1": test_tensor_1,
                                                    "input_2": test_tensor_2, "input_3": test_tensor_3,
                                                    "input_4": test_tensor_4, "input_5": test_tensor_5,
                                                    "input_6": test_tensor_6}))
dataset_test = dataset_test.batch(batch_size=1)

# Concatenate original and augmented (wo ToF/THM data) datasets for train and validation
dataset_train = dataset_train_orig.concatenate(dataset_train_e).shuffle(len(dataset_train_orig)*2,seed=SEED).batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
dataset_val = dataset_val_orig.concatenate(dataset_val_e).batch(batch_size=1).prefetch(tf.data.AUTOTUNE)
dataset_valbatch = dataset_val_orig.concatenate(dataset_val_e).batch(batch_size=batch_size_val, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

print('Size of train dataset: '+ str(len(dataset_train_orig)*2))
print('Steps in train dataset per batch: '+ str(len(dataset_train)) + f' (in {round(len(dataset_train_orig)*2/len(dataset_train))} batches)')
print('Size of validation dataset: '+ str(len(dataset_val)))
print('Steps in validation dataset per batch: '+ str(len(dataset_valbatch)) + f' (in {round(len(dataset_val)/len(dataset_valbatch))} batches)')
print('Size of test dataset: '+ str(len(dataset_test)))


## Define model branaches for 1D/2D/3D data

def SE_block(input_se, filters, name):
    x = input_se
    if input_se.ndim == 5:
        x = tf.keras.layers.GlobalAveragePooling3D(name=name+'_se_gap3d')(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D(name=name+'_se_gap2d')(x)
    x = tf.keras.layers.Dense(units=int(filters/16), activation="relu", kernel_initializer='he_normal', name=name+'_se_den')(x)
    x = tf.keras.layers.Dense(units=filters, activation="sigmoid", name=name+'_se_fin')(x)
    if input_se.ndim == 5:
        x = tf.keras.layers.Reshape((1, 1, 1, filters))(x)
    else:
        x = tf.keras.layers.Reshape((1, 1, filters))(x)
    return x

# Define 3D CNN model for ToF and THM sensor data
def Conv3D_model(input_conv3d, blocks_conv3d, filters_base_conv3d, name):
    reg = None #L2(1e-4)
    do = 0.15
    x = input_conv3d
    for i, filters in enumerate([filters_base_conv3d*(2**j) for j in range(blocks_conv3d)]):
        if i < 3:
            kernel = (3,3,6)
            padding = 'same'
            pool = (1,1,2)
        elif i < 5:
            kernel = (2,2,3)
            padding = 'same'
            pool = (2,2,2)
        else:
            kernel = (2,2,2)
            padding = 'same'
            pool = (2,2,2)
        x_sc = x
        x = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel, padding=padding,
                                   activation=None, kernel_regularizer=reg, name=name+f'lay{i+1}1')(x)
        x = tf.keras.layers.BatchNormalization(name=name+f'bn{i+1}1')(x)
        x = tf.keras.layers.ReLU(name=name+f'relu{i+1}1')(x)
        x = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel, padding=padding,
                                   activation=None, kernel_regularizer=reg, name=name+f'lay{i+1}2')(x)
        x = tf.keras.layers.BatchNormalization(name=name+f'bn{i+1}2')(x)
        x_sc = tf.keras.layers.Conv3D(filters=filters, kernel_size=1, padding=padding,
                                      activation=None, kernel_regularizer=reg, name=name+f'laysc{i+1}')(x_sc)
        x_sc = tf.keras.layers.BatchNormalization(name=name+f'bnsc{i+1}')(x_sc)
        x_se = SE_block(x, filters, name=name+f'layse{i+1}')
        
        x = tf.keras.layers.Multiply()([x, x_se])
        x = tf.keras.layers.Add(name=name+f'add{i+1}')([x, x_sc])
        x = tf.keras.layers.ReLU(name=name+f'relu{i+1}f')(x)
        x = tf.keras.layers.MaxPool3D(pool_size=pool, name=name+f'mp{i+1}')(x)
        x = tf.keras.layers.Dropout(do, name=name+f'do{i+1}')(x)
    out = tf.keras.layers.GlobalAveragePooling3D(name=name+'gap')(x)
    model = tf.keras.Model(inputs=input_conv3d, outputs=out, name=name)
    return model

# Define 2D CNN model for position and orientation sensor data
def Conv2D_model(input_conv2d, blocks_conv2d, filters_base_conv2d, name):
    reg = None #L2(1e-4)
    do = 0.15
    x = input_conv2d
    for i, filters in enumerate([filters_base_conv2d*(2**j) for j in range(blocks_conv2d)]):
        if i < 3:
            kernel = (3,6)
            padding = 'same'
            pool = (1,2)
        elif i < 5:
            kernel = (2,2)
            padding = 'same'
            pool = (2,2)
        else:
            kernel = (1,2)
            padding = 'same'
            pool = (1,2)
        x_sc = x
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, padding=padding,
                                   activation=None, kernel_regularizer=reg, name=name+f'lay{i+1}1')(x)
        x = tf.keras.layers.BatchNormalization(name=name+f'bn{i+1}1')(x)
        x = tf.keras.layers.ReLU(name=name+f'relu{i+1}1')(x)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, padding=padding,
                                   activation=None, kernel_regularizer=reg, name=name+f'lay{i+1}2')(x)
        x = tf.keras.layers.BatchNormalization(name=name+f'bn{i+1}2')(x)
        x_sc = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding=padding,
                                      activation=None, kernel_regularizer=reg, name=name+f'laysc{i+1}')(x_sc)
        x_sc = tf.keras.layers.BatchNormalization(name=name+f'bnsc{i+1}')(x_sc)
        x_se = SE_block(x, filters, name=name+f'layse{i+1}')
        
        x = tf.keras.layers.Multiply()([x, x_se])
        x = tf.keras.layers.Add(name=name+f'add{i+1}')([x, x_sc])
        x = tf.keras.layers.ReLU(name=name+f'relu{i+1}f')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=pool, name=name+f'mp{i+1}')(x)
        #x = tf.keras.layers.Dropout(do, name=name+f'do{i+1}')(x)
    out = tf.keras.layers.GlobalAveragePooling2D(name=name+'gap')(x)
    model = tf.keras.Model(inputs=input_conv2d, outputs=out, name=name)
    return model

# Define 1D DNN model for subject related data
def DNN_model(input_dnn, layers_dnn, units_dnn, name):
    reg = None #L2(1e-4)
    do = 0.3
    x = input_dnn
    for i, units in enumerate([units_dnn for j in range(layers_dnn)]):
        x = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg,
                                  kernel_initializer='he_normal', name=name+f'fc{i+1}')(x)
        x = tf.keras.layers.BatchNormalization(name=name+f'bn{i+1}')(x)
        x = tf.keras.layers.Dropout(do, name=name+f'do{i+1}')(x)
    out = x
    model = tf.keras.Model(inputs=input_dnn, outputs=out, name=name)
    return model


## Build 1D/2D/3D CNN network

def build_network(hp):
    # define the sets of inputs
    input_0 = tf.keras.Input(shape=(7, seq_len,1), name='input_0')
    input_1 = tf.keras.Input(shape=(8, 8, seq_len, 2), name='input_1')
    input_2 = tf.keras.Input(shape=(8, 8, seq_len, 2), name='input_2')
    input_3 = tf.keras.Input(shape=(8, 8, seq_len, 2), name='input_3')
    input_4 = tf.keras.Input(shape=(8, 8, seq_len, 2), name='input_4')
    input_5 = tf.keras.Input(shape=(8, 8, seq_len, 2), name='input_5')
    input_6 = tf.keras.Input(shape=(7,), name='input_6')

    # data augmentation layer (shift time series data along time axis)
    max_roll_layer = 0 # hp.Int(name='max_roll_layer', min_value=30, max_value=40, step=5, default=30)
    max_acc_scale = 1 # hp.Float(name='max_acc_scale', min_value=1, max_value=4, step=0.5, sampling="linear", default=3)
    max_t_scale = 2 # hp.Float(name='max_t_scale', min_value=1, max_value=4, step=0.5, sampling="linear", default=3)
    if build_in_augmentation:
        input_0a, input_1a, input_2a, input_3a, input_4a, input_5a = TranslationLayer(max_roll_layer, max_acc_scale, max_t_scale)(
            [input_0, input_1, input_2, input_3, input_4, input_5])
    else:
        input_0a, input_1a, input_2a, input_3a, input_4a, input_5a = input_0, input_1, input_2, input_3, input_4, input_5
        
    # define the sets of branches for Conv3D tensors (ToF and THM sensor data)
    blocks_conv3d = 6
    filters_base_conv3d = 16
    conv3d_model_1_out = Conv3D_model(input_1, blocks_conv3d, filters_base_conv3d, name='conv3d_model_1')(input_1a)
    conv3d_model_2_out = Conv3D_model(input_2, blocks_conv3d, filters_base_conv3d, name='conv3d_model_2')(input_2a)
    conv3d_model_3_out = Conv3D_model(input_3, blocks_conv3d, filters_base_conv3d, name='conv3d_model_3')(input_3a)
    conv3d_model_4_out = Conv3D_model(input_4, blocks_conv3d, filters_base_conv3d, name='conv3d_model_4')(input_4a)
    conv3d_model_5_out = Conv3D_model(input_5, blocks_conv3d, filters_base_conv3d, name='conv3d_model_5')(input_5a)
    
    # define branch for 2D tensor (IMU sensor data)
    blocks_conv2d = 6
    filters_base_conv2d = 32
    data2d_model_out = Conv2D_model(input_0, blocks_conv2d, filters_base_conv2d, name='data2d_model')(input_0a)
    
    # define branch for 1D tensor (subject related data)
    layers_dnn = 3
    units_dnn = 256
    data1d_model_out = DNN_model(input_6, layers_dnn, units_dnn, name='data1d_model')(input_6)
    
    # model_3ddummy = Conv3D_model(input_1, blocks_conv3d, filters_base_conv3d, name='conv3d_model_1')
    # model_2ddummy = Conv2D_model(input_0, blocks_conv2d, filters_base_conv2d, name='data2d_model')
    # model_3ddummy.summary()
    # model_2ddummy.summary()
    
    # combine the output of the branches
    z = tf.keras.layers.Concatenate(name='concat_all')([conv3d_model_1_out, conv3d_model_2_out, conv3d_model_3_out,
                                                        conv3d_model_4_out, conv3d_model_5_out, data2d_model_out,
                                                        data1d_model_out])
    
    # apply FC layer(s) and then a regression prediction on the combined outputs
    reg = None #L2(1e-4)
    do = 0.30
    layers_final = 2
    units_final = 512
    for i, units in enumerate([units_final for j in range(layers_final)]):
        z = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg,
                                  kernel_initializer='he_normal', name=f'final_fc{i+1}')(z)
        z = tf.keras.layers.BatchNormalization(name=f'final_bn{i+1}')(z)
        z = tf.keras.layers.Dropout(do, name=f'final_do{i+1}')(z)
    out = tf.keras.layers.Dense(len(class2bfrb.keys()), activation="softmax", kernel_regularizer=reg, name='prediction')(z)
    model = tf.keras.Model(inputs=[input_0, input_1, input_2, input_3, input_4, input_5, input_6], outputs=out)

    lr_tune = 5e-4 # hp.Float(name='learning_rate', min_value=5e-5, max_value=5e-3, sampling='log', default=5e-4)
    optimizer = tf.keras.optimizers.Adam(lr_tune)
    loss_fn = tf.keras.losses.sparse_categorical_crossentropy
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'], run_eagerly=False)
    return model

with strategy.scope():
    model = build_network(kt.HyperParameters())


## Explore model architecture

model.summary()
tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_dtype=False,
                          show_layer_names=True, show_layer_activations=True, show_trainable=False)


## Training parameters

epochs = 400
TUNING = False
TRAINING = False
CUSTOM_TRAINING = False
FINETUNING = False

## Callback functions
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=8, verbose=1, monitor='val_accuracy')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=16, verbose=1, monitor='val_accuracy', restore_best_weights=True)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 20)) # Find starting learning


## Load pre-trained weights

if not TRAINING and not TUNING and not CUSTOM_TRAINING:
    model.load_weights('/kaggle/input/cmi-59x/CMI_5_10_2.weights.h5')
    print('Model weights have been loaded!')


## Train model

if TRAINING or FINETUNING:
    history = model.fit(dataset_train, validation_data=dataset_valbatch, epochs=epochs, callbacks=[lr_scheduler, early_stopping_cb])


## Tuner

i_TunerTyp = 1 # Choose desired tuner type: {1: 'grid', 2: 'random', 3: 'hyper'}
TunerStr = {1: 'grid', 2: 'random', 3: 'hyper'}

tuner_grid = kt.GridSearch(hypermodel=build_network, objective='val_accuracy',
                           max_trials=10, max_consecutive_failed_trials=1,
                           overwrite=True, directory="tuner", project_name="CMI", distribution_strategy = strategy)

tuner_random = kt.RandomSearch(hypermodel=build_network, objective='val_accuracy',
                               max_trials=10, executions_per_trial=1,
                               overwrite=True, directory="tuner", project_name="CMI", distribution_strategy = strategy)

tuner_hyper = kt.Hyperband(hypermodel=build_network, objective='val_accuracy',
                           max_epochs=90, factor=5, hyperband_iterations=1,
                           overwrite=True, directory="tuner", project_name="CMI", distribution_strategy = strategy)

tuner = globals()[f'tuner_{TunerStr[i_TunerTyp]}']
tuner.search_space_summary()


## Train model by Tuner

if TUNING:
    tuner.search(dataset_train, validation_data=dataset_valbatch, epochs=epochs, callbacks=[lr_scheduler, early_stopping_cb])

    best_models = tuner.get_best_models(num_models=2)
    model = best_models[0]
    model.summary()
    tuner.results_summary()


## Custom training loop

if CUSTOM_TRAINING:
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    #loss_fn = tf.keras.losses.sparse_categorical_crossentropy
    #optimizer = tf.keras.optimizers.Adam(0.0005)
    loss_fn = model.loss
    optimizer = model.optimizer
    history = tf.keras.callbacks.History()
    history.set_model(model)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)
    return val_loss_value

def custom_train():
    epochs = 200
    patience_early = 11
    patience_lr = 8
    wait_early = 0
    wait_lr = 0
    best_acc_early = float(0.0)
    best_acc_lr = float(0.0)
    
    for epoch in range(epochs): # len(dataset_train)
        progBar = tf.keras.utils.Progbar(229, stateful_metrics=['train_loss', 'train_acc', 'val_loss', 'val_acc'])
        history.on_train_begin(logs={'loss': 10.0, 'accuracy': 0.0,
                                     'val_loss': 10.0, 'val_accuracy': 0.0,
                                     'learning_rate': optimizer.learning_rate.numpy()})
        
        # Iterate over the batches of the dataset
        loss_value_accum = 0
        for step, (x_batch_train, y_batch_train) in enumerate(dataset_train):
            loss_value_accum += train_step(x_batch_train, y_batch_train)
            loss_value = loss_value_accum/(step+1)
            train_acc = train_acc_metric.result()
            # Update progress bar with training loss and accuracy
            progBar.update(step, values=[('train_loss', loss_value.numpy().mean()), ('train_acc', train_acc.numpy())])
    
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in dataset_valbatch:
            val_loss_value = test_step(x_batch_val, y_batch_val)
        val_acc = val_acc_metric.result()
        
        # Update progress bar with validation loss and accuracy
        progBar.update(step+1, values=[('train_loss', loss_value.numpy().mean()), ('train_acc', train_acc.numpy()),
                                       ('val_loss', val_loss_value.numpy().mean()), ('val_acc', val_acc.numpy()),
                                       ('learning_rate', optimizer.learning_rate.numpy()), ('epoch', int(epoch+1)),
                                       ('wait', wait_early)], finalize=True)
        history.on_epoch_end(int(epoch+1), logs={'loss': loss_value.numpy().mean(), 'accuracy': train_acc.numpy(),
                                                 'val_loss': val_loss_value.numpy().mean(), 'val_accuracy': val_acc.numpy(),
                                                 'learning_rate': optimizer.learning_rate.numpy()})

        # Reset metrics at the end of each epoch
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()
        
        # The early stopping strategy: stop the training if `val_acc` does not decrease over a certain number of epochs.
        wait_early += 1
        if val_acc > best_acc_early:
            best_acc_early = val_acc
            model.save_weights('cp_best.weights.h5')
            best_epoch = epoch+1
            wait_early = 0
        if wait_early >= patience_early:
            model.load_weights('cp_best.weights.h5')
            print(f'Early stopping. Best model is restored from epoch {best_epoch}')
            break
        
        wait_lr += 1
        if val_acc > best_acc_lr:
            best_acc_lr = val_acc
            wait_lr = 0
        if wait_lr >= patience_lr:
            model.load_weights('cp_best.weights.h5')
            optimizer.learning_rate = optimizer.learning_rate * 0.1
            wait_lr = 0

if CUSTOM_TRAINING:
    custom_train()


## Save weights of model

if TRAINING or TUNING or FINETUNING or CUSTOM_TRAINING:
    model.save_weights('CMI_5_10_4.weights.h5')


## Plot learning curves

if TRAINING or FINETUNING or CUSTOM_TRAINING:
    history_fil = {key: history.history[key] for key in ['accuracy', 'val_accuracy']}
    history_fil2 = {key: history.history[key] for key in ['loss', 'val_loss']}
    history_fil3 = {key: history.history[key] for key in ['learning_rate']}
    
    pd.DataFrame(history_fil).plot()
    plt.ylabel("Accuracy")
    plt.xlabel("epochs")
    pd.DataFrame(history_fil2).plot()
    plt.ylabel("Loss")
    plt.xlabel("epochs")
    plt.axis([10, len(history_fil2['val_loss']), 0, history_fil2['val_loss'][10]+0.1*history_fil2['val_loss'][10]])
    pd.DataFrame(history_fil3).plot()
    plt.ylabel("Learning rate")
    plt.xlabel("epochs")


## Calculate competition specific score

from cmi_2025_metric_copy_for_import import CompetitionMetric

# Get predicted labels for the validation set
images, labels = tuple(zip(*dataset_val))

print(model.evaluate(dataset_val))
probabilities = model.predict(dataset_val)
y_val_pred = np.argmax(probabilities, axis=-1)
y_val_true = np.squeeze(np.array(labels))

# Map integer labels back to gesture strings
val_pred_labels = pd.Series(y_val_pred).map(lambda i: id2class[i])
val_true_labels = pd.Series(y_val_true).map(lambda i: id2class[i])

# Build DataFrames for the metric
val_submission = pd.DataFrame({'gesture': val_pred_labels})
val_solution = pd.DataFrame({'gesture': val_true_labels})

# Run competition metric
metric = CompetitionMetric()
score = metric.calculate_hierarchical_f1(val_solution, val_submission)
print(f"Estimated leaderboard (val) score: {score:.4f}")

print(val_pred_labels)
print(val_true_labels)


## Init server

import kaggle_evaluation.cmi_inference_server

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    test = sequence.to_pandas()
    if len(test.index) > seq_len:
        print(len(test.index))
        test = test[-seq_len:]
    #print(len(test.index))

    test_demographics = demographics.to_pandas()
    test_tensor_0, test_tensor_1, test_tensor_2, test_tensor_3, test_tensor_4, test_tensor_5 = extract_time_series_data(test, False)
    test_tensor_6, _ = extract_subject_related_data(test, test_demographics, scalers_sub, False)
    dataset_test = tf.data.Dataset.from_tensor_slices(({"input_0": test_tensor_0,
                                                        "input_1": test_tensor_1, "input_2": test_tensor_2,
                                                        "input_3": test_tensor_3, "input_4": test_tensor_4,
                                                        "input_5": test_tensor_5, "input_6": test_tensor_6}))
    dataset_test = dataset_test.batch(batch_size=1)
    
    probabilities = model.predict(dataset_test, verbose=0)
    predictions = np.argmax(probabilities, axis=-1)
    prediction = np.argmax(probabilities, axis=-1)[0]
    return id2class[prediction]

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)


## Run server

if not TRAINING and not TUNING and not CUSTOM_TRAINING:
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            data_paths=(
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
            )
        )


# ## Explore max/min/average length of sequences
# data_check = train[train['phase']=='Gesture'] # Max: 71
# data_check = train # Max: 700, Min: 29, Avg: 71 ('SEQ_015261')
# min_length = 1000
# max_length = 0
# for s, sequence_id in enumerate(data_check['sequence_id'].unique()):
#     if s%1000 == 0: print(s) # Show progress
#     data_check_fil = data_check[data_check['sequence_id'] == sequence_id]
#     seq_length = len(data_check_fil.index)
#     if seq_length < min_length:
#         min_length = seq_length
#         min_seq_id = sequence_id
#     if seq_length > max_length:
#         max_length = seq_length
#         max_seq_id = sequence_id


# ## Plot learning curves for definition of start leraning rate
# lrs = 1e-5 * (10 ** (np.arange(100) / 20)) # Define the learning rate array
# plt.figure(figsize=(10, 6)) # Set the figure size
# plt.grid(True) # Set the grid
# plt.semilogx(lrs, history.history["loss"]) # Plot the loss in log scale
# plt.tick_params('both', length=10, width=1, which='both') # Increase the tickmarks size
# plt.axis([1e-5, 1e-0, 0, 10]) # Set the plot boundaries


# start = time.time()
# TranslationLayer(20, 3.25, 1.5)([train_tensor_0 , train_tensor_1, train_tensor_2, train_tensor_3, train_tensor_4, train_tensor_5],
#                               training=True)
# end = time.time()
# print(end - start)

