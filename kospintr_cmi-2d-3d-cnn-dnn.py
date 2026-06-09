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
for fold, (train_idx, val_idx) in enumerate(cv.split(trainval, trainval['gesture'], groups=np.array(trainval['sequence_id'], dtype='<U10'))):
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

# scalers = {'scaler_acc' : MinMaxScaler(feature_range=(-1, 1)).fit([[-15],[15]]),
#            'scaler_rot_w' : MinMaxScaler().fit([[0],[1]]), 'scaler_rot' : MinMaxScaler().fit([[-1],[1]]),
#            'scaler_tof' : MinMaxScaler().fit([[-1],[249]]), 'scaler_thm' : MinMaxScaler().fit([[21],[35]])}
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


## Data augmentation function

max_roll_fn = 20
def translate_tensor(tensors, training=True):
    tensor_trans_list = []
    for i in range(train_tensor_0.shape[0]):
        tensor = tensors[i]
        if training:
            roll = random.randint(-max_roll_fn, 0)
        else:
            roll = 0
        tensor_rev = tf.reverse(tensor, axis=[-2])
        tensor_ext = tf.concat([tensor_rev, tensor, tensor_rev], axis=-2)
        if tensor_ext.ndim == 3:
            tensor_trans = tf.roll(tensor_ext, roll, axis=-2)[:, seq_len:2*seq_len, :]
        else:
            tensor_trans = tf.roll(tensor_ext, roll, axis=-2)[:, :, seq_len:2*seq_len, :]
        tensor_trans = tf.expand_dims(tensor_trans, axis=0)
        tensor_trans_list.append(tensor_trans)
    tensor_trans = tf.concat(tensor_trans_list, axis=0)
    return tensor_trans


## Create datasets

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

dataset_train_e = tf.data.Dataset.from_tensor_slices(({"input_0": translate_tensor(train_tensor_0), "input_1": train_tensor_e,
                                                       "input_2": train_tensor_e, "input_3": train_tensor_e,
                                                       "input_4": train_tensor_e, "input_5": train_tensor_e,
                                                       "input_6": train_tensor_6}, labels_train))

dataset_train_e2 = tf.data.Dataset.from_tensor_slices(({"input_0": translate_tensor(train_tensor_0),
                                                        "input_1": translate_tensor(train_tensor_1),
                                                        "input_2": translate_tensor(train_tensor_2),
                                                        "input_3": translate_tensor(train_tensor_3),
                                                        "input_4": translate_tensor(train_tensor_4),
                                                        "input_5": translate_tensor(train_tensor_5),
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


## Data augmentation layer for all 2D/3D tensors during training (experimental)

max_roll_layer = 0
class TranslationLayer(tf.keras.layers.Layer):
    def __init__(self, max_roll, name='augmentation'):
        super(TranslationLayer, self).__init__(name=name)
        self.max_roll = max_roll
    
    def call(self, tensors, training=True):
        if training:
            roll = random.randint(-self.max_roll, 0)
        else:
            roll = 0
        tensors_trans = []
        for tensor in tensors:
            tensor_rev = tf.reverse(tensor, axis=[-2])
            tensor_ext = tf.concat([tensor_rev, tensor, tensor_rev], axis=-2)
            if tensor.ndim == 5:
                tensor_trans = tf.roll(tensor_ext, roll, axis=-2)[:, :, :, seq_len:2*seq_len, :]
            else:
                tensor_trans = tf.roll(tensor_ext, roll, axis=-2)[:, :, seq_len:2*seq_len, :]
            tensors_trans.append(tensor_trans)
        return tuple(tensors_trans)


## Define model branaches for 1D/2D/3D data

# Define 3D CNN model for ToF and THM sensor data
def Conv3D_model(input_conv3d, name):
    reg = L2(1e-4)
    do = 0.15
    x1 = tf.keras.layers.Conv3D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                activation="relu", kernel_regularizer=reg, name=name+'lay1')(input_conv3d)
    x2 = tf.keras.layers.MaxPool3D(name=name+'mp1')(x1)
    x3 = tf.keras.layers.BatchNormalization(name=name+'bn1')(x2)
    x4 = tf.keras.layers.Dropout(do, name=name+'do1')(x3)
    x5 = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                activation="relu", kernel_regularizer=reg, name=name+'lay2')(x4)
    x6 = tf.keras.layers.MaxPool3D(name=name+'mp2')(x5)
    x7 = tf.keras.layers.BatchNormalization(name=name+'bn2')(x6)
    x8 = tf.keras.layers.Dropout(do, name=name+'do2')(x7)
    x9 = tf.keras.layers.Conv3D(filters=256, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                activation="relu", kernel_regularizer=reg, name=name+'lay3')(x8)
    x10 = tf.keras.layers.MaxPool3D(name=name+'mp3')(x9)
    x11 = tf.keras.layers.BatchNormalization(name=name+'bn3')(x10)
    x12 = tf.keras.layers.Dropout(do, name=name+'do3')(x11)
    x13 = tf.keras.layers.Conv3D(filters=512, kernel_size=(1,1,3), padding='same', kernel_initializer='he_normal',
                                 activation="relu", kernel_regularizer=reg, name=name+'lay4')(x12)
    x14 = tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2), name=name+'mp4')(x13)
    x15 = tf.keras.layers.BatchNormalization(name=name+'bn4')(x14)
    x16 = tf.keras.layers.Dropout(do, name=name+'do4')(x15)
    x17 = tf.keras.layers.Conv3D(filters=1024, kernel_size=(1,1,2), padding='valid', kernel_initializer='he_normal',
                                 activation="relu", kernel_regularizer=reg, name=name+'lay5')(x16)
    x18 = tf.keras.layers.MaxPool3D(pool_size=(1, 1, 4), name=name+'mp5')(x17)
    out = tf.keras.layers.GlobalAveragePooling3D(name=name+'gap')(x18)
    model = tf.keras.Model(inputs=input_conv3d, outputs=out, name=name)
    return model

# Define 2D CNN model for position and orientation sensor data
def Conv2D_model(input_conv2d, name):
    reg = L2(1e-4)
    do = 0.15
    x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',
                                activation='relu', kernel_regularizer=reg, name=name+'lay1')(input_conv2d)
    x2 = tf.keras.layers.MaxPool2D(pool_size=(1,2), name=name+'mp1')(x1)
    x3 = tf.keras.layers.BatchNormalization(name=name+'bn1')(x2)
    x4 = tf.keras.layers.Dropout(do, name=name+'do1')(x3)
    x5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',
                                activation='relu', kernel_regularizer=reg, name=name+'lay2')(x4)
    x6 = tf.keras.layers.MaxPool2D(pool_size=(1,2), name=name+'mp2')(x5)
    x7 = tf.keras.layers.BatchNormalization(name=name+'bn2')(x6)
    x8 = tf.keras.layers.Dropout(do, name=name+'do2')(x7)
    x9 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',
                                activation='relu', kernel_regularizer=reg, name=name+'lay3')(x8)
    x10 = tf.keras.layers.MaxPool2D(pool_size=(1,2), name=name+'mp3')(x9)
    x11 = tf.keras.layers.BatchNormalization(name=name+'bn3')(x10)
    x12 = tf.keras.layers.Dropout(do, name=name+'do3')(x11)
    x13 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',
                                 activation='relu', kernel_regularizer=reg, name=name+'lay4')(x12)
    x14 = tf.keras.layers.MaxPool2D(pool_size=(1,2), name=name+'mp4')(x13)
    x15 = tf.keras.layers.BatchNormalization(name=name+'bn4')(x14)
    x16 = tf.keras.layers.Dropout(do, name=name+'do4')(x15)
    x17 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,2), padding='valid', kernel_initializer='he_normal',
                                 activation='relu', kernel_regularizer=reg, name=name+'lay5')(x16)
    x18 = tf.keras.layers.MaxPool2D(pool_size=(1,2), name=name+'mp5')(x17)
    x19 = tf.keras.layers.BatchNormalization(name=name+'bn5')(x18)
    x20 = tf.keras.layers.Dropout(do, name=name+'do5')(x19)
    x21 = tf.keras.layers.Conv2D(filters=2048, kernel_size=(3,2), padding='valid', kernel_initializer='he_normal',
                                 activation='relu', kernel_regularizer=reg, name=name+'lay6')(x20)
    out = tf.keras.layers.GlobalAveragePooling2D(name=name+'gap')(x21)
    model = tf.keras.Model(inputs=input_conv2d, outputs=out, name=name)
    return model

# Alternative 2D CNN model for further experiments (optional)
def Conv2D_model2(input_conv2d, name):
    reg = L2(1e-4)
    do = 0.15
    x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,10), padding='same', kernel_initializer='he_normal',
                                activation='relu', kernel_regularizer=reg, name=name+'lay1')(input_conv2d)
    x2 = tf.keras.layers.MaxPool2D(pool_size=(1,2), name=name+'mp1')(x1)
    x3 = tf.keras.layers.BatchNormalization(name=name+'bn1')(x2)
    x4 = tf.keras.layers.Dropout(do, name=name+'do1')(x3)
    x5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,5), padding='same', kernel_initializer='he_normal',
                                activation='relu', kernel_regularizer=reg, name=name+'lay2')(x4)
    x6 = tf.keras.layers.MaxPool2D(pool_size=(1,2), name=name+'mp2')(x5)
    x7 = tf.keras.layers.BatchNormalization(name=name+'bn2')(x6)
    x8 = tf.keras.layers.Dropout(do, name=name+'do2')(x7)
    x9 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1,3), padding='same', kernel_initializer='he_normal',
                                activation='relu', kernel_regularizer=reg, name=name+'lay3')(x8)
    x10 = tf.keras.layers.MaxPool2D(pool_size=(1,2), name=name+'mp3')(x9)
    x11 = tf.keras.layers.BatchNormalization(name=name+'bn3')(x10)
    x12 = tf.keras.layers.Dropout(do, name=name+'do3')(x11)
    x13 = tf.keras.layers.Conv2D(filters=512, kernel_size=(1,3), padding='same', kernel_initializer='he_normal',
                                 activation='relu', kernel_regularizer=reg, name=name+'lay4')(x12)
    x14 = tf.keras.layers.MaxPool2D(pool_size=(1,2), name=name+'mp4')(x13)
    x15 = tf.keras.layers.BatchNormalization(name=name+'bn4')(x14)
    x16 = tf.keras.layers.Dropout(do, name=name+'do4')(x15)
    x17 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,2), padding='valid', kernel_initializer='he_normal',
                                 activation='relu', kernel_regularizer=reg, name=name+'lay5')(x16)
    x18 = tf.keras.layers.MaxPool2D(pool_size=(1,2), name=name+'mp5')(x17)
    x19 = tf.keras.layers.BatchNormalization(name=name+'bn5')(x18)
    x20 = tf.keras.layers.Dropout(do, name=name+'do5')(x19)
    x21 = tf.keras.layers.Conv2D(filters=2048, kernel_size=(7,2), padding='valid', kernel_initializer='he_normal',
                                 activation='relu', kernel_regularizer=reg, name=name+'lay6')(x20)
    out = tf.keras.layers.GlobalAveragePooling2D(name=name+'gap')(x21)
    model = tf.keras.Model(inputs=input_conv2d, outputs=out, name=name)
    return model

# Define 1D DNN model for subject related data
def DNN_model(input_dnn, name):
    reg = L2(1e-4)
    do = 0.3
    x1 = tf.keras.layers.Dense(256, activation="relu", kernel_initializer='he_normal', 
                               kernel_regularizer=reg, name=name+'fc1')(input_dnn)
    x2 = tf.keras.layers.BatchNormalization(name=name+'bn1')(x1)
    x3 = tf.keras.layers.Dropout(do, name=name+'do1')(x2)
    x4 = tf.keras.layers.Dense(256, activation="relu", kernel_initializer='he_normal', 
                               kernel_regularizer=reg, name=name+'fc2')(x3)
    x5 = tf.keras.layers.BatchNormalization(name=name+'bn2')(x4)
    x6 = tf.keras.layers.Dropout(do, name=name+'do2')(x5)
    out = tf.keras.layers.Dense(256, activation="relu", kernel_initializer='he_normal', 
                                kernel_regularizer=reg, name=name+'fc3')(x6)
    model = tf.keras.Model(inputs=input_dnn, outputs=out, name=name)
    return model


## Build 1D/2D/3D CNN network

def build_network():
    # define the sets of inputs
    input_0 = tf.keras.Input(shape=(7,seq_len,1), name='input_0')
    input_1 = tf.keras.Input(shape=(8, 8, seq_len, 2), name='input_1')
    input_2 = tf.keras.Input(shape=(8, 8, seq_len, 2), name='input_2')
    input_3 = tf.keras.Input(shape=(8, 8, seq_len, 2), name='input_3')
    input_4 = tf.keras.Input(shape=(8, 8, seq_len, 2), name='input_4')
    input_5 = tf.keras.Input(shape=(8, 8, seq_len, 2), name='input_5')
    input_6 = tf.keras.Input(shape=(7,), name='input_6')

    # data augmentation layer (shift time series data along time axis)
    input_0a, input_1a, input_2a, input_3a, input_4a, input_5a = TranslationLayer(max_roll_layer)(
        [input_0, input_1, input_2, input_3, input_4, input_5])
    
    # define the sets of branches for Conv3D tensors (ToF and THM sensor data)
    conv3d_model_1_out = Conv3D_model(input_1, name='conv3d_model_1')(input_1a)
    conv3d_model_2_out = Conv3D_model(input_2, name='conv3d_model_2')(input_2a)
    conv3d_model_3_out = Conv3D_model(input_3, name='conv3d_model_3')(input_3a)
    conv3d_model_4_out = Conv3D_model(input_4, name='conv3d_model_4')(input_4a)
    conv3d_model_5_out = Conv3D_model(input_5, name='conv3d_model_5')(input_5a)
    
    # define branch for 2D tensor (IMU sensor data) and for 1D tensor (subject related data)
    data2d_model_out = Conv2D_model(input_0, name='data2d_model')(input_0a)
    data1d_model_out = DNN_model(input_6, name='data1d_model')(input_6)
    
    # combine the output of the branches
    combined_3d = tf.keras.layers.Concatenate(name='concat_3d')([conv3d_model_1_out, conv3d_model_2_out,
                                                                 conv3d_model_3_out, conv3d_model_4_out,
                                                                 conv3d_model_5_out])
    combined_3d = tf.keras.layers.BatchNormalization(name='concat_3d_bn')(combined_3d)
    combined_3d = tf.keras.layers.Dropout(0.8, name='concat_3d_do')(combined_3d)
    combined_1d_2d = tf.keras.layers.Concatenate(name='concat_1d_2d')([data1d_model_out, data2d_model_out])
    combined_1d_2d = tf.keras.layers.BatchNormalization(name='concat_1d_2d_bn')(combined_1d_2d)
    combined_1d_2d = tf.keras.layers.Dropout(0.65, name='concat_1d_2d_do')(combined_1d_2d)
    combined = tf.keras.layers.Concatenate(name='concat_all')([combined_3d, combined_1d_2d])
    
    # apply a FC layer and then a regression prediction on the combined outputs
    reg = L2(1e-4)
    do = 0.4
    z1 = tf.keras.layers.Dense(512, activation="relu", kernel_initializer='he_normal', kernel_regularizer=reg, name='fc1')(combined)
    z2 = tf.keras.layers.BatchNormalization(name='bn1')(z1)
    z3 = tf.keras.layers.Dropout(do, name='do1')(z2)
    z4 = tf.keras.layers.Dense(512, activation="relu", kernel_initializer='he_normal', kernel_regularizer=reg, name='fc2')(z3)
    z5 = tf.keras.layers.BatchNormalization(name='bn2')(z4)
    z6 = tf.keras.layers.Dropout(do, name='do2')(z5)
    z7 = tf.keras.layers.Dense(512, activation="relu", kernel_initializer='he_normal', kernel_regularizer=reg, name='fc3')(z6)
    z8 = tf.keras.layers.BatchNormalization(name='bn3')(z7)
    z9 = tf.keras.layers.Dropout(do, name='do3')(z8)
    z10 = tf.keras.layers.Dense(len(class2bfrb.keys()), activation="softmax", kernel_regularizer=reg, name='prediction')(z9)
    model = tf.keras.Model(inputs=[input_0, input_1, input_2, input_3, input_4, input_5, input_6], outputs=z10)

    steps_per_epoch = len(dataset_train_orig)*2//batch_size
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(5e-4, first_decay_steps=15 * steps_per_epoch)
    optimizer = tf.keras.optimizers.Adam(0.0005) # SGD(0.001) Adam(0.0001)
    loss_fn = tf.keras.losses.sparse_categorical_crossentropy #tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model

with strategy.scope():
    model = build_network()


## Explore model architecture

model.summary()
tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_dtype=False,
                          show_layer_names=True, show_layer_activations=True, show_trainable=False)


## Training parameters

epochs = 200
TUNING = False
TRAINING = False
FINETUNING = False
steps_per_epoch = len(dataset_train_orig)*2//batch_size
validation_steps = len(dataset_val)//batch_size_val

## Callback functions
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=8, verbose=1, monitor='val_accuracy')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=16, verbose=1,
                                                     monitor='val_accuracy', restore_best_weights=True)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 20)) # Find starting learning 


## Load pre-trained weights

if not TRAINING and not TUNING:
    model.load_weights('/kaggle/input/cmi-422/CMI_5_6_19.weights.h5')
    print('Model weights have been loaded!')


## Train model

if TRAINING or FINETUNING:
    history = model.fit(dataset_train, validation_data=dataset_valbatch, epochs=epochs,
                        callbacks=[lr_scheduler, early_stopping_cb])


## Tuner

if TUNING:
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
    tuner.search(dataset_train, validation_data=dataset_valbatch, epochs=epochs,
                     callbacks=[lr_scheduler, early_stopping_cb])

    best_models = tuner.get_best_models(num_models=2)
    model = best_models[0]
    model.summary()
    tuner.results_summary()


## Save weights of model

if TRAINING or TUNING or FINETUNING:
    model.save_weights('CMI_5_9_33.weights.h5')


## Plot learning curves

if TRAINING or FINETUNING:
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

if not TRAINING and not TUNING and not FINETUNING:
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


# ## Explore split distribution comparing Dataset.shuffle with StratifiedGroupKFold
# SEED = 33
# train_split = 0.9
# labels = extract_labels(trainval)
# dataset_label = tf.data.Dataset.from_tensor_slices(labels)
# dataset_label_train = dataset_label.shuffle(8200,seed=SEED).take(round(len(labels)*train_split))
# dataset_label_val = dataset_label.shuffle(8200,seed=SEED).skip(round(len(labels)*train_split))

# labels_train = extract_labels(train)
# labels_val = extract_labels(val)
# fold_label_train = tf.data.Dataset.from_tensor_slices(labels_train)
# fold_label_val = tf.data.Dataset.from_tensor_slices(labels_val)

# def count_classes(labels):
#     label_reconstructed = []
#     for i, label in enumerate(labels):
#         label_reconstructed.append(label)
#     label_reconstructed = tf.convert_to_tensor(label_reconstructed, dtype=np.int32)
#     y, idx, count = tf.unique_with_counts(label_reconstructed)
#     count = count/len(label_reconstructed)
#     return y, idx, count

# y, idx, count = count_classes(dataset_label)
# distribution = pd.DataFrame(data = count, index = y, columns=['Orig'])
# y, idx, count = count_classes(dataset_label_train)
# distribution = distribution.merge(pd.DataFrame(data = count, index = y, columns=['DS_Train']), how='left', left_index=True, right_index=True)
# y, idx, count = count_classes(dataset_label_val)
# distribution = distribution.merge(pd.DataFrame(data = count, index = y, columns=['DS_Val']), how='left', left_index=True, right_index=True)
# y, idx, count = count_classes(fold_label_train)
# distribution = distribution.merge(pd.DataFrame(data = count, index = y, columns=['Fold_Train']), how='left', left_index=True, right_index=True)
# y, idx, count = count_classes(fold_label_val)
# distribution = distribution.merge(pd.DataFrame(data = count, index = y, columns=['Fold_Val']), how='left', left_index=True, right_index=True)
# distribution


# ## Plot learning curves for definition of start leraning rate
# lrs = 1e-5 * (10 ** (np.arange(100) / 20)) # Define the learning rate array
# plt.figure(figsize=(10, 6)) # Set the figure size
# plt.grid(True) # Set the grid
# plt.semilogx(lrs, history.history["loss"]) # Plot the loss in log scale
# plt.tick_params('both', length=10, width=1, which='both') # Increase the tickmarks size
# plt.axis([1e-5, 1e-0, 0, 10]) # Set the plot boundaries


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from ipywidgets import interact
# #%matplotlib inline

# def interactive_plot(image3d):
#     # only for example, use your grid
#     z = np.linspace(0, 8, 8)
#     x = np.linspace(0, 8, 8)
#     y = np.linspace(0, 8, 8)
    
#     X, Y, Z = np.meshgrid(x, y, z)
    
#     # Your 4dimension, only for example (use yours)
#     U = image3d_tof_1[0].numpy().reshape(8, 8, 100)
#     U = U[:, :, :8]
#     # Creating figure
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.view_init(elev=-84, azim=90, roll=0)
#     #ax = Axes3D(fig)
#     #ax = plt.axes(projection="3d")
    
#     # Creating plot
#     scatter = ax.scatter3D(X, Y, Z, c=U, s=200, marker='o', alpha=0.5, cmap='PRGn')
#     #fig.colorbar(scatter, ax=ax)
#     plt.show()

# interact(interactive_plot(image3d_tof_1[0]))


# # Explore sequence order of different grouping strategies
# seq_gp = train.groupby('sequence_id')
# groups = np.array([seq_id for seq_id, _ in seq_gp])

# groups2 = train['sequence_id'].unique()

# train_fil = train[['sequence_id', 'subject']].drop_duplicates(subset=['sequence_id'])
# train_fil = train_fil.merge(trainval_demographics, how='left', left_on='subject', right_on='subject')
# groups3 = train_fil['sequence_id'].to_numpy()

# print(groups)
# print(groups2)
# print(groups3)
# if np.array_equal(groups, groups2) and np.array_equal(groups, groups3):
#     print("Equal")
# else:
#     print("Not Equal")

