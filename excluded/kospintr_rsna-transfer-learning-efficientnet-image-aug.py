## For TPU environment (install missing packages / reinstall tensorflow to solve NaN topic during training / restart kernel)

import IPython
import tensorflow as tf

if len(tf.config.experimental.list_logical_devices('TPU')) > 0:
    !pip install -q tensorflow-tpu -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force-reinstall
    !pip install -q pydot
    !pip install -q -U keras-tuner
    !pip install -q polars
    !pip install -q pydicom
    !pip install -q protobuf==5.29.5 # to solve tuner compatibility issue
    IPython.Application.instance().kernel.do_shutdown(True)


## Import packages

# General purpose modules
import os
import shutil
from collections import defaultdict
import re
import math
from tqdm import tqdm
import time

# Data handling and visualization modules
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.patches import Circle
import pydicom

# Skikit-learn preprocessing modules
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold

# Tensorflow modules
import tensorflow as tf
import keras_tuner as kt

# Custom specific evaluation module
import kaggle_evaluation.rsna_inference_server


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


## Read csv files and merge them into a single dataframe

path = '/kaggle/input/rsna-intracranial-aneurysm-detection/'
trainval = pd.read_csv(path + "train.csv")
trainval_localizers = pd.read_csv(path + "train_localizers.csv")
trainval = trainval.merge(trainval_localizers, on='SeriesInstanceUID', how='outer')


## Spliting trainval data into train and validation data with StratifiedGroupKFold

# Create a column with multiclass label for StratifiedGroupKFold splitting
label_columns = trainval.columns[trainval.columns.str.contains('Artery|Tip|Other|Present', case=True)]
label2class = {}
trainval['class'] = 0
for i, col in enumerate(label_columns[:]):
    label2class[col] = i + 1
    if i < 13:
        trainval['class'] = trainval['class'] + trainval[col] * (i + 1)

# Shuffle and split trainval data into train and validation data
skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42) # Baseline 42
for fold, (train_idx, val_idx) in enumerate(skf.split(trainval, y=trainval['class'], groups=trainval['SeriesInstanceUID'])):
    train, val = trainval.iloc[train_idx], trainval.iloc[val_idx]
    print(f"✅ Fold {fold}: Train size = {len(train_idx)}, Val size = {len(val_idx)}")
    break  # Use only the first fold for now


## Preprocessing functions

image_size = 300 # input image size fo neural network model

# Remove black frame around images and adjust coordinates accordingly
def crop_image(image, x, y, tol = 0.05, crop=True):
    img = image[0, :, :, 0]
    mask = img > tol
    if crop and mask.sum()>1000:
        masked_idx = np.ix_(mask.any(1), mask.any(0))
        image = img[masked_idx]
        image = image.reshape(1, image.shape[0], image.shape[1], 1)
    if crop and x >= 0 and mask.sum()>1000: # if valid labels are present rescale coordinates according masking
        coor = np.zeros((img.shape), dtype=float)
        coor[round(y), round(x)] = 1
        coor_masked = coor[masked_idx]
        row, col = np.where(coor_masked == 1)
        y, x = row[0], col[0]
    return image, x, y

# Pad and resize images (while retaining aspect ratio) and adjust coordinates accordingly
def pad_and_resize(image, x, y):
    _, image_size_rows, image_size_cols, _ = image.shape
    pad_size = max(image_size_rows, image_size_cols)
    image_padded = tf.image.resize_with_crop_or_pad(image, pad_size, pad_size)
    image_resized = tf.image.resize(image_padded, [image_size, image_size], method=tf.image.ResizeMethod.BICUBIC)
    if x >= 0:
        coor = np.zeros((image.shape), dtype=float)
        coor[:, round(y), round(x), :] = 1
        coor_padded = tf.image.resize_with_crop_or_pad(coor, pad_size, pad_size)
        coor_resized = tf.image.resize(coor_padded, [image_size, image_size], method=tf.image.ResizeMethod.AREA)
        _, row, col, _ = np.where(coor_resized.numpy() == coor_resized.numpy().max())
        y, x = row[0], col[0]
    return image_resized, x, y

# Zoom/rotate/translate images and adjust coordinates accordingly
def image_augmentation(image, x, y, augmentation=True):
    if augmentation:
        coor = np.zeros((image.shape), dtype=float)
        zoom_fac = np.random.uniform(0.0, 0.0)
        rot_fac = np.random.uniform(-0.1, 0.1)
        trans_fac = np.random.uniform(-0.05, 0.05)
        z = tf.keras.layers.RandomZoom(height_factor=(zoom_fac, zoom_fac), fill_mode='constant', name='auglay1')(image)
        z = tf.keras.layers.RandomRotation(factor=(rot_fac, rot_fac), fill_mode='constant', name='auglay2')(z)
        image = tf.keras.layers.RandomTranslation(height_factor=(trans_fac, trans_fac), width_factor=(trans_fac, trans_fac),
                                                  interpolation='nearest', fill_mode='constant', name='auglay3')(z)
        if x >= 0:
            coor[:, round(y), round(x), :] = 1
            coor = tf.convert_to_tensor(coor)
            z = tf.keras.layers.RandomZoom(height_factor=(zoom_fac, zoom_fac), fill_mode='constant', name='auglay1')(coor)
            z = tf.keras.layers.RandomRotation(factor=(rot_fac, rot_fac), fill_mode='constant', name='auglay2')(z)
            coor = tf.keras.layers.RandomTranslation(height_factor=(trans_fac, trans_fac), width_factor=(trans_fac, trans_fac),
                                                  interpolation='nearest', fill_mode='constant', name='auglay3')(z)
            _, row, col, _ = np.where(coor.numpy() == coor.numpy().max())
            y, x = row[0], col[0]
    return image, x, y

def preprocess_images(image, x, y, crop, augmentation):
    image_scaled = (MinMaxScaler().fit_transform(image.reshape(-1, 1))).reshape(1, image.shape[0], image.shape[1], 1).astype(dtype=np.float32)
    try:
        image_croped, x_croped, y_croped = crop_image(image_scaled, x, y, crop=crop)
    except:
        image_croped, x_croped, y_croped = crop_image(image_scaled, x, y, crop=False)
    image_aug, x_aug, y_aug = image_augmentation(image_croped, x_croped, y_croped, augmentation=augmentation)
    image_resized, x_resized, y_resized = pad_and_resize(image_aug, x_aug, y_aug)
    image_resized = tf.cast(image_resized*255, dtype=tf.uint8)
    if x >= 0:
        x_resized_scaled, y_resized_scaled = x_resized/image_size, y_resized/image_size
    else:
        x_resized_scaled, y_resized_scaled = -1, -1
    return image_resized, x_resized_scaled, y_resized_scaled


## Display croped 2D image(s) with bounding circle around aneurysm center for choosen modalities

modality = 'MRI T2' # Choose modality to show from: CTA/MRA/MRI T2/MRI T1post
aneurysm_present = train[train['Aneurysm Present'] == 1]
aneurysm_present_mod_fil = aneurysm_present[aneurysm_present['Modality'] == modality] 
rows = 3 # Number of images in a row
columns = 4 # Number of images in a column
fig,ax = plt.subplots(rows, columns, figsize=(12, 12))

for i, index in enumerate(aneurysm_present_mod_fil.index[:rows*columns]):
    # Extract raw data for selected sample
    data_slice = aneurysm_present_mod_fil.loc[index]
    SI_UID, SOPI_UID, = data_slice['SeriesInstanceUID'], data_slice['SOPInstanceUID'], 
    sex, age = data_slice['PatientSex'], data_slice['PatientAge']
    x, y = eval(data_slice['coordinates'])['x'], eval(data_slice['coordinates'])['y']
    image_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/series/' + SI_UID + '/' + SOPI_UID + '.dcm'
    image = pydicom.dcmread(image_path).pixel_array
    if len(image.shape) == 3:
        image = image[0, :, :]

    # Preprocess raw data (scaling, croping, padding, resizing)
    image_resized, x_resized, y_resized = preprocess_images(image, x, y, crop=True, augmentation=True)
    
    # Plot processed images
    row = i % rows
    col = i // rows
    ax[row,col].imshow(image_resized[0], cmap=plt.cm.hot)
    ax[row,col].set_title(sex + ' ' + str(age))
    ax[row,col].add_patch(Circle((x_resized*image_size, y_resized*image_size), 20, fill=False, ec='cyan'))


## Preprocess train and validation data

SUBMISSIONING = False # configuration flag to save runtime during submissioning
iter_tot_nr = 6 # number of iteration loops over train.csv entries resulting in iter_tot_nr * 4263 samples
root_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'
encoder_mod = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit([['MR'], ['CT']])

def preprocess_step(data, iter_nr):
    image_list = []
    modality_list = []
    label_list = []
    coordinates_list = []
    
    # Loop over data lines and append image, label and coordinates to the corresponding lists
    for i, index in enumerate(tqdm(data.index)):
        data_slice = data.loc[index]
        subfolder = data_slice['SeriesInstanceUID']
        subfolder_path = root_path + '/' + subfolder
        
        if not data_slice[['SOPInstanceUID']].isna().values[0]:
            file_name = data_slice['SOPInstanceUID'] + '.dcm'
        else:
            file_list = os.listdir(subfolder_path)
            file_tot_nr = len(file_list) # number of frames
            file_nr_list = np.rint((np.linspace(0, file_tot_nr-1, iter_tot_nr)))
            file_nr = int(file_nr_list[iter_nr])
            file_name = file_list[file_nr]

        # Preprocess image data and coordinates
        aneurysm_present = (data_slice['Aneurysm Present'] == 1)
        augmentation = (iter_nr > 0) and aneurysm_present
        dcm = pydicom.dcmread(os.path.join(subfolder_path, file_name))
        try: # Set frame_nr for multiframe dcm (either from csv label if aneurysm is present or iteration number)
            frames_tot_nr = int(dcm.NumberOfFrames)
            if aneurysm_present:
                frame_nr = int(eval(data_slice['coordinates'])['f'])
            else:
                frame_nr_list = np.rint((np.linspace(0, frames_tot_nr-1, iter_tot_nr)))
                frame_nr = int(frame_nr_list[iter_nr])
        except: # Exception if dcm file is not multiframe dcm
            frame_nr = 0
        image = pydicom.pixels.pixel_array(dcm, index=frame_nr)
        mod = encoder_mod.transform([[dcm.Modality]])
        if data_slice[['coordinates']].isna().values[0]:
            x, y = -1, -1
        else:
            x, y = eval(data_slice['coordinates'])['x'], eval(data_slice['coordinates'])['y']
        image_resized, x_resized, y_resized = preprocess_images(image, x, y, crop=True, augmentation=augmentation)
        coordinates_tensor = tf.expand_dims(tf.convert_to_tensor([x_resized, y_resized], dtype=np.float32), 0)
        image_list.append(image_resized)
        modality_list.append(mod)
        coordinates_list.append(coordinates_tensor)
        
        # Preprocess labels
        labels = data_slice[label_columns]
        label_tensor = tf.expand_dims(tf.convert_to_tensor(labels, dtype=np.float32), 0)
        label_list.append(label_tensor)
            
    # Concat list of sample tensors
    images = tf.concat(image_list, axis=0)
    modalities = tf.concat(modality_list, axis=0)
    labels = tf.concat(label_list, axis=0)
    coordinates = tf.concat(coordinates_list, axis=0)
    return images, labels, coordinates, modalities

def preprocess_loop(data, iter_tot_nr):
    data_images_list = []
    data_modalities_list = []
    data_labels_list = []
    data_coordinates_list = []
    for iter_nr in range(iter_tot_nr):
        data_images, data_labels, data_coordinates, data_modalities = preprocess_step(data, iter_nr)
        data_images_list.append(data_images)
        data_modalities_list.append(data_modalities)
        data_labels_list.append(data_labels)
        data_coordinates_list.append(data_coordinates)
    data_images = tf.concat(data_images_list, axis=0)
    data_modalities = tf.concat(data_modalities_list, axis=0)
    data_labels = tf.concat(data_labels_list, axis=0)
    data_coordinates = tf.concat(data_coordinates_list, axis=0)
    return data_images, data_labels, data_coordinates, data_modalities

if not SUBMISSIONING:
    train_images, train_labels, train_coordinates, train_modalities = preprocess_loop(train[:], iter_tot_nr)
    val_images, val_labels, val_coordinates, val_modalities = preprocess_loop(val[:], iter_tot_nr)
else:
    train_images, train_labels, train_coordinates, train_modalities = preprocess_loop(train[:64], iter_tot_nr)
    val_images, val_labels, val_coordinates, val_modalities = preprocess_loop(val[:64], iter_tot_nr)


## Calculate the total memory size of train images in Gigabytes

num_elements = tf.size(train_images).numpy() # Get the number of elements in the tensor
element_size = train_images.dtype.size # Get the size of each element in bytes
total_memory = num_elements * element_size

print(f"Number of elements in train images: {num_elements}")
print(f"Size of each element in train images: {element_size} bytes")
print(f"Total memory size of train images: {total_memory/1024**3} Gigabytes")


## Create train and validation datasets

SEED=42
batch_size=32
batch_size_val=32

train_ds = tf.data.Dataset.from_tensor_slices(({"input_img": train_images, "input_mod": train_modalities},
                                               {"class": train_labels, "reg": train_coordinates}))
train_ds = train_ds.shuffle(len(train_labels), seed=SEED).repeat().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices(({"input_img": val_images, "input_mod": val_modalities},
                                             {"class": val_labels, "reg": val_coordinates}))
val_ds = val_ds.batch(batch_size_val, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

print('Size of train dataset: '+ str(len(train_labels)))
print('Number of batches in train dataset: '+ f'{len(train_labels)//batch_size}')
print('Size of validation dataset: '+ str(len(val_labels)))
print('Number of batches in val dataset: '+ f'{len(val_labels)//batch_size_val}')


## Check validation dataset batch dimensions

for X, y in val_ds.take(1):
    print(X['input_img'].shape)
    print(X['input_mod'].shape)
    print(y['class'].shape)
    print(y['reg'].shape)


## Custom BCE class and AUC function to handle imbalanced label classes

label_weights = tf.constant([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 13], dtype=tf.float32)/26

@tf.keras.utils.register_keras_serializable()
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, name="weighted_bce_loss", reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
        self.weight_positive = 1 - tf.reduce_sum(train_labels, axis=0)/len(train_labels)
        self.weight_negative = tf.reduce_sum(train_labels, axis=0)/len(train_labels)
        self.label_weights = label_weights
        
    def call(self, y_true, y_pred):
        # Clip predictions to avoid log(0) errors
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        # Compute the class (0/1) weighted binary cross-entropy losses
        bce_loss = -(self.weight_positive * y_true * tf.math.log(y_pred) +
                     self.weight_negative * (1 - y_true) * tf.math.log(1 - y_pred))
        # Compute the multilabel weighted binary cross-entropy losses
        bce_loss = bce_loss*self.label_weights
        return tf.reduce_mean(bce_loss)


## Network configuration and preprocessing layer

# Network configurations
base_network = {4: "enb0",     # EfficientNetB0
                5: "enb1",     # EfficientNetB1
                6: "enb2",     # EfficientNetB2
                7: "enb3",     # EfficientNetB3
                8: "enb4",     # EfficientNetB4
                9: "env2b0",   # EfficientNetV2B0
                10: "env2b1",  # EfficientNetV2B1
                11: "env2b2",  # EfficientNetV2B2
                12: "env2b3",  # EfficientNetV2B3
                13: "env2s"}   # EfficientNetV2S

# Custom layer for preprocessing
@tf.keras.utils.register_keras_serializable()
class Rescale(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Rescale, self).__init__(**kwargs)
    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)/255
        x = tf.image.grayscale_to_rgb(x)
        return x

@tf.keras.utils.register_keras_serializable()
class PreProcess(tf.keras.layers.Layer):
    def __init__(self, base_network_type, **kwargs):
        super(PreProcess, self).__init__(**kwargs)
        if base_network_type < 9: self.preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        elif base_network_type >= 9: self.preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
        else: print('Wrong base network number have been choosen!!!')
    def call(self, inputs):
        return self.preprocess_input(inputs*255.0)


## Build Squeeze-and-Excitation network (SENet)

def build_network(hp):
    # Select base model and corresponding preprocessing
    base_network_type = 7 #hp.Int(name='base_network_type', min_value=4, max_value=13, step=1, default=7) # Choose Pretrained Network
    if base_network_type == 4: base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, input_shape=[image_size, image_size, 3], weights='imagenet')
    elif base_network_type == 5: base_model = tf.keras.applications.EfficientNetB1(
        include_top=False, input_shape=[image_size, image_size, 3], weights='imagenet')
    elif base_network_type == 6: base_model = tf.keras.applications.EfficientNetB2(
        include_top=False, input_shape=[image_size, image_size, 3], weights='imagenet')
    elif base_network_type == 7: base_model = tf.keras.applications.EfficientNetB3(
        include_top=False, input_shape=[image_size, image_size, 3], weights='imagenet')
    elif base_network_type == 8: base_model = tf.keras.applications.EfficientNetB4(
        include_top=False, input_shape=[image_size, image_size, 3], weights='imagenet')
    elif base_network_type == 9: base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False, input_shape=[image_size, image_size, 3], weights='imagenet')
    elif base_network_type == 10: base_model = tf.keras.applications.EfficientNetV2B1(
        include_top=False, input_shape=[image_size, image_size, 3], weights='imagenet')
    elif base_network_type == 11: base_model = tf.keras.applications.EfficientNetV2B2(
        include_top=False, input_shape=[image_size, image_size, 3], weights='imagenet')
    elif base_network_type == 12: base_model = tf.keras.applications.EfficientNetV2B3(
        include_top=False, input_shape=[image_size, image_size, 3], weights='imagenet')
    elif base_network_type == 13: base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False, input_shape=[image_size, image_size, 3], weights='imagenet')
    else: print('Wrong base network number have been choosen!!!')
    prepocessing = PreProcess(base_network_type=base_network_type, name='preprocessing')

    # Choose base model layers to be trained
    base_model.trainable = False # freeze base model layers
    max_layer_nr = len(base_model.layers)
    # (B0): all:238 / 2ab+:220 / 3ab+:191 / 4abc+:162 / 5abc+:118 / 6abcd+:75 / 7a+:16
    # (B3): all:385 / 2ab+:355 / 3ab+:311 / 4abc+:267 / 5abc+:193 / 6abcd+:120 / 7a+:31
    layer_id = 355 #hp.Choice(name='layer_id', values=[355, max_layer_nr, 311, 267, 193, 120, 31]) # layer number from the network shall be trained
    print('Unfreeze base model layers from layer ' + str(base_model.layers[-layer_id]))
    for layer in base_model.layers[-layer_id:]: # unfreeze choosen layers
        layer.trainable = True

    # define the sets of inputs
    input_img = tf.keras.Input(shape=(image_size, image_size, 1), name='input_img')
    input_mod = tf.keras.Input(shape=(2,), name='input_mod')

    # cast and rescale image tensors
    x = Rescale(name='rescaling')(input_img)

    # define pretrained EfficientNet layers
    gap_do = hp.Float(name='do_dnn', min_value=0, max_value=0.4, step=0.05, default=0.1)
    x = prepocessing(x)
    x_base = base_model(x)

    # prepare the output of EfficentNet and combine with the modality input for final classification dnn layer(s)
    x_class = tf.keras.layers.GlobalAveragePooling2D(name=f'gap2d')(x_base)
    x_class = tf.keras.layers.BatchNormalization(name=f'gap2d_bn')(x_class)
    x_class = tf.keras.layers.Dropout(gap_do, name=f'gap2d_do')(x_class)
    x_class = tf.keras.layers.Concatenate(name='concat_gap')([x_class, input_mod])

    # prepare the output of EfficentNet and combine with the modality input for final regression dnn layer(s) (optional)
    fil_1x1conv = hp.Int(name='fil_1x1conv', min_value=16, max_value=64, step=16, default=16)
    x_reg = tf.keras.layers.Conv2D(fil_1x1conv, (1,1), name=f'1x1conv2d')(x_base)
    x_reg = tf.keras.layers.Flatten(name=f'1x1conv2d_flatten')(x_reg)
    x_reg = tf.keras.layers.BatchNormalization(name=f'1x1conv2d_bn')(x_reg)
    x_reg = tf.keras.layers.Dropout(gap_do, name=f'1x1conv2d_do')(x_reg)
    x_reg = tf.keras.layers.Concatenate(name='concat_1x1conv2d_reg')([x_reg, input_mod])
            
    # define DNN layer(s) for classification
    x_class_fin = x_class
    reg_dnn = None #tf.keras.regularizers.L2(1e-4)
    ki_dnn = 'he_uniform'
    do_dnn = gap_do
    layers_final = 1 #hp.Int(name='layers_final', min_value=0, max_value=2, step=1, default=1)
    units_final = 512 #hp.Int(name='units_final', min_value=128, max_value=512, step=128, default=512)
    for i, units in enumerate([units_final for j in range(layers_final)]):
        x_class_fin = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg_dnn,
                                            kernel_initializer=ki_dnn, name=f'final_fc{i+1}')(x_class_fin)
        x_class_fin = tf.keras.layers.BatchNormalization(name=f'final_bn{i+1}')(x_class_fin)
        x_class_fin = tf.keras.layers.Dropout(do_dnn, name=f'final_do{i+1}')(x_class_fin)
    output_lab = tf.keras.layers.Dense(len(label2class.keys()), activation="sigmoid", kernel_regularizer=reg_dnn, name='class')(x_class_fin)
    
    # define DNN layer(s) for regression
    x_reg_fin = x_reg
    reg_dnn_reg = None #tf.keras.regularizers.L2(1e-4)
    ki_dnn_reg = 'he_uniform'
    do_dnn_reg = gap_do
    layers_final_reg = 1 #hp.Int(name='layers_final_reg', min_value=0, max_value=3, step=1, default=1)
    units_final_reg = 256 #hp.Int(name='units_final_reg', min_value=128, max_value=512, step=128, default=256)
    for i, units in enumerate([units_final_reg for j in range(layers_final_reg)]):
        x_reg_fin = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg_dnn_reg,
                                          kernel_initializer=ki_dnn_reg, name=f'final_fc_reg{i+1}')(x_reg_fin)
        #x_reg_fin = tf.keras.layers.BatchNormalization(name=f'final_bn_reg{i+1}')(x_reg_fin)
        x_reg_fin = tf.keras.layers.Dropout(do_dnn_reg, name=f'final_do_reg{i+1}')(x_reg_fin)
    output_coord = tf.keras.layers.Dense(2, activation="linear", kernel_regularizer=reg_dnn_reg, name='reg')(x_reg_fin)

    # define model
    model = tf.keras.Model(inputs=[input_img, input_mod], outputs=[output_lab, output_coord], name='RSNA_Class')

    # define optimizer/loss and compile model
    lr_tune = 1e-3 #hp.Float(name='learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
    optimizer = tf.keras.optimizers.Adam(lr_tune)
    loss_class = WeightedBinaryCrossentropy()
    loss = {"reg": "mean_squared_error", "class": loss_class}
    loss_weights = {"reg": 0.01, "class": 0.99} #{"reg": 0.01, "class": 0.99}
    auc = tf.keras.metrics.AUC(multi_label=True, label_weights=label_weights, name='auc')
    metrics = {"reg": ["mae"], "class": [auc]}

    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics, run_eagerly=False)
    return model

if not SUBMISSIONING:
    with strategy.scope():
        model = build_network(kt.HyperParameters())
else:
    # Load pre-trained model for submission
    model = tf.keras.models.load_model('/kaggle/input/rsna-3xx/rsna_3_11_0.h5')
    print('Model weights have been loaded!')


## Explore model architecture

model.summary(line_length=110)
# tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_dtype=False,
#                           show_layer_names=True, show_layer_activations=True, show_trainable=False)


## Training parameters

epochs = 100
steps_per_epoch = len(train_labels)//batch_size
TUNING = False and not SUBMISSIONING
TRAINING = True and not SUBMISSIONING
FINETUNING = False and not SUBMISSIONING


## Tuner configurations

if TUNING:
    i_TunerTyp = 1 # Choose desired tuner type: {1: 'grid', 2: 'random', 3: 'hyper'}
    TunerStr = {1: 'grid', 2: 'random', 3: 'hyper'}
    
    tuner_grid = kt.GridSearch(hypermodel=build_network, objective=kt.Objective("val_class_auc", direction="max"),
                               max_trials=15, max_consecutive_failed_trials=1,
                               overwrite=True, directory="tuner", project_name="RSNA", distribution_strategy = strategy)
    
    tuner_random = kt.RandomSearch(hypermodel=build_network, objective=kt.Objective("val_class_auc", direction="max"),
                                   max_trials=10, executions_per_trial=1,
                                   overwrite=True, directory="tuner", project_name="RSNA", distribution_strategy = strategy)
    
    tuner_hyper = kt.Hyperband(hypermodel=build_network, objective=kt.Objective("val_class_auc", direction="max"),
                               max_epochs=60, factor=4, hyperband_iterations=1,
                               overwrite=True, directory="tuner", project_name="RSNA", distribution_strategy = strategy)
    
    tuner = globals()[f'tuner_{TunerStr[i_TunerTyp]}']
    tuner.search_space_summary()


## Train or tune model

# Callback functions
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, verbose=1, monitor='val_class_auc', mode='max')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1, monitor='val_class_auc', mode='max', restore_best_weights=True)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 10)) # Find starting learning

# Training
if TRAINING or FINETUNING:
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        callbacks=[lr_scheduler, early_stopping_cb])

# Tuning
if TUNING:
    tuner.search(train_ds, validation_data=val_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
                 callbacks=[lr_scheduler, early_stopping_cb])
    best_models = tuner.get_best_models(num_models=2)
    model = best_models[0]
    model.summary()
    tuner.results_summary()


## Save weights of model after training/tuning/finetuning

if TRAINING or TUNING or FINETUNING:
    model.save('rsna_3_11_0.h5', include_optimizer=False)
    print('Model weights have been saved!')


## Plot learning curves

if TRAINING or FINETUNING:
    history_fil = {key: history.history[key] for key in ['class_auc', 'val_class_auc']}
    history_fil2 = {key: history.history[key] for key in ['class_loss', 'val_class_loss']}
    history_fil3 = {key: history.history[key] for key in ['learning_rate']}
    
    pd.DataFrame(history_fil).plot()
    plt.ylabel("AUC")
    plt.xlabel("epochs")
    pd.DataFrame(history_fil2).plot()
    plt.ylabel("Loss")
    plt.xlabel("epochs")
    #plt.axis([10, len(history_fil2['val_loss']), 0, history_fil2['val_loss'][10]+0.1*history_fil2['val_loss'][10]])
    pd.DataFrame(history_fil3).plot()
    plt.ylabel("Learning rate")
    plt.xlabel("epochs")


## Compare true and predicted values for the first 10 samples from validation set

val_vis = val[['class']][:10].rename(columns={'class': 'class_true'})
val_prob, val_coord = model.predict((val_images[:10], val_modalities[:10]), verbose=0)
prob_vis = pd.DataFrame(val_prob, val_vis.index)
true_coord_vis = pd.DataFrame(val_coordinates[:10], val_vis.index, columns=['x_true', 'y_true'])
val_coord_vis = pd.DataFrame(val_coord, val_vis.index, columns=['x_pred', 'y_pred'])
pd.concat([val_vis, prob_vis, true_coord_vis, val_coord_vis], axis=1)


## Make prediction and run submission script

ID_COL = 'SeriesInstanceUID'

LABEL_COLS = ['Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
              'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
              'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery', 'Anterior Communicating Artery',
              'Left Anterior Cerebral Artery', 'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
              'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation', 'Aneurysm Present']

# All tags (other than PixelData and SeriesInstanceUID) that may be in a test set dcm file
DICOM_TAG_ALLOWLIST = ['BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit', 'ImageOrientationPatient',
                       'ImagePositionPatient', 'InstanceNumber', 'Modality', 'PatientID', 'PhotometricInterpretation',
                       'PixelRepresentation', 'PixelSpacing', 'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope',
                       'RescaleType', 'Rows', 'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness',
                       'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID']

def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""

    # Collect file paths of test series
    series_id = os.path.basename(series_path)
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()
    
    # Preprocess test series data
    image_list = []
    mod_list = []
    for image_path in all_filepaths:
        dcm = pydicom.dcmread(image_path)
        image = dcm.pixel_array #pydicom.pixels.pixel_array(dcm, index=iter_nr)
        mod = encoder_mod.transform([[dcm.Modality]])
        if len(image.shape) == 3: # Multiframe dcm
            for frame in image:
                image_resized, x_resized, y_resized = preprocess_images(frame, -1, -1, crop=True, augmentation=False)
                image_list.append(tf.cast(image_resized, tf.float32))
                mod_list.append(tf.cast(mod, tf.float32))
        else: # Single frame dcm
            image_resized, x_resized, y_resized = preprocess_images(image, -1, -1, crop=True, augmentation=False)
            image_list.append(tf.cast(image_resized, tf.float32))
            mod_list.append(tf.cast(mod, tf.float32))
    test_images = tf.concat(image_list, axis=0)
    test_mods = tf.concat(mod_list, axis=0)

    # Make prediction
    lab, coor = model.predict((test_images, test_mods), verbose=0)
    prob_lab = np.max(lab, axis=0)
    predictions_list = prob_lab.astype(dtype='float').tolist()

    # Prepare submisstion output format
    predictions = pl.DataFrame(data=[[series_id] + predictions_list],
                               schema=[ID_COL, *LABEL_COLS], orient='row')

    # ----------------------------- IMPORTANT ------------------------------
    # You MUST have the following code in your `predict` function
    # to prevent "out of disk space" errors. This is a temporary workaround
    # as we implement improvements to our evaluation system.
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    # ----------------------------------------------------------------------
    
    return predictions.drop(ID_COL)


## Run server

if not TRAINING and not TUNING and not FINETUNING:
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
    
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway()
        display(pl.read_parquet('/kaggle/working/submission.parquet'))


if TUNING:
    shutil.rmtree('/kaggle/working/tuner')


# ## Plot learning curves for definition of start leraning rate
# lrs = 1e-5 * (10 ** (np.arange(len(history.history["loss"])) / 10)) # Define the learning rate array
# plt.figure(figsize=(10, 6)) # Set the figure size
# plt.grid(True) # Set the grid
# plt.semilogx(lrs, history.history["loss"]) # Plot the loss in log scale
# plt.tick_params('both', length=10, width=1, which='both') # Increase the tickmarks size
# #plt.axis([1e-5, 1e-0, 0, 10]) # Set the plot boundaries


# ## Visualisation in 3D

# def load_dicom_series(folder_path): # Function to load DICOM series (multiple slices)
#     import os
#     slices = []
#     for file_name in sorted(os.listdir(folder_path)):
#         if file_name.endswith(".dcm"):
#             dicom_file = pydicom.dcmread(os.path.join(folder_path, file_name))
#             NoF = int(dicom_file.NumberOfFrames)
#     return dicom_file.pixel_array

# def plot_3d(image_array, threshold=0.5):
#     x, y, z = np.where(image_array > (image_array.max()-image_array.min())* threshold + image_array.min())
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(projection='3d')
#     colors = image_array[x,y,z]
#     ax.scatter3D(x, y, z, c=colors, vmin=colors.min(), vmax=colors.max(),
#                cmap='gist_rainbow', alpha=0.02, marker='o', s=0.1)
#     #ax.view_init(15, 15, 0)
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     ax.set_zlabel('Z-axis')
#     plt.show()

# folder_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10134365079002163886508836892471866754'
# dicom_volume = load_dicom_series(folder_path)
# plot_3d(dicom_volume, threshold=0.02)
# x, y, z = np.where(dicom_volume > (dicom_volume.max()-dicom_volume.min())* 0.5 + dicom_volume.min())
# dicom_volume[x, y, z].max()

