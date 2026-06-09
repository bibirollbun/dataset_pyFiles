## For TPU environment (install missing packages / reinstall tensorflow to solve NaN topic during training / restart kernel)

import IPython
import tensorflow as tf
print(tf.__version__)

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
from mpl_toolkits.mplot3d import Axes3D

# Skikit-learn preprocessing modules
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
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
skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(trainval, y=trainval['class'], groups=trainval['SeriesInstanceUID'])):
    train, val = trainval.iloc[train_idx], trainval.iloc[val_idx]
    print(f"✅ Fold {fold}: Train size = {len(train_idx)}, Val size = {len(val_idx)}")
    break  # Use only the first fold for now


## Preprocessing functions

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

image_size = 256 # input image size fo neural network model
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

SUBMISSIONING = True # configuration flag to save runtime during submissioning
iteration_nr = 8 # number of iteration loops over train.csv entries resulting in iteration_nr * 4263 samples
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
        if len(os.listdir(subfolder_path)) > iter_nr:
            file_name = os.listdir(subfolder_path)[iter_nr]
        else:
            file_name = os.listdir(subfolder_path)[-1]
        
        if not data_slice[['SOPInstanceUID']].isna().values[0]:
            file_name = data_slice['SOPInstanceUID'] + '.dcm'

        # Preprocess image data and coordinates
        aneurysm_present = (data_slice['Aneurysm Present'] == 1)
        augmentation = (iter_nr > 0) and aneurysm_present
        dcm = pydicom.dcmread(os.path.join(subfolder_path, file_name))
        try: # Set frame_nr for multiframe dcm (either from csv label if aneurysm is present or iteration number)
            nr_frames = int(dcm.NumberOfFrames)
            if aneurysm_present:
                frame_nr = int(eval(data_slice['coordinates'])['f'])
            else:
                frame_nr = min(iter_nr, nr_frames-1)
        except: # Exception if dcm file is not multiframe dcm
            frame_nr = 0
        image = pydicom.pixels.pixel_array(dcm, index=frame_nr)
        mod = encoder_mod.transform([[dcm.Modality]])
        if data_slice[['coordinates']].isna().values[0]:
            x, y = -1, -1
        else:
            x, y = eval(data_slice['coordinates'])['x'], eval(data_slice['coordinates'])['y']
        image_resized, x_resized, y_resized = preprocess_images(image, x, y, crop=False, augmentation=augmentation)
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

def preprocess_loop(data, iteration_nr):
    data_images_list = []
    data_modalities_list = []
    data_labels_list = []
    data_coordinates_list = []
    for i in range(iteration_nr):
        data_images, data_labels, data_coordinates, data_modalities = preprocess_step(data, i)
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
    train_images, train_labels, train_coordinates, train_modalities = preprocess_loop(train[:], iteration_nr)
    val_images, val_labels, val_coordinates, val_modalities = preprocess_loop(val[:], iteration_nr)
else:
    train_images, train_labels, train_coordinates, train_modalities = preprocess_loop(train[:64], iteration_nr)
    val_images, val_labels, val_coordinates, val_modalities = preprocess_loop(val[:64], iteration_nr)


## Calculate the total memory size of train images in Gigabytes

num_elements = tf.size(train_images).numpy() # Get the number of elements in the tensor
element_size = train_images.dtype.size # Get the size of each element in bytes
total_memory = num_elements * element_size

print(f"Number of elements in train images: {num_elements}")
print(f"Size of each element in train images: {element_size} bytes")
print(f"Total memory size of train images: {total_memory/1024**3} Gigabytes")


## Create train and validation datasets

SEED=42
batch_size=64
batch_size_val=64

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


## Custom BCE class to handle imbalanced label classes
label_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 13]

class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, name="weighted_bce_loss"):
        super().__init__(name=name)
        self.weight_positive = 1 - tf.reduce_sum(train_labels, axis=0)/len(train_labels)
        self.weight_negative = tf.reduce_sum(train_labels, axis=0)/len(train_labels)
        self.label_weights = tf.constant([label_weights], dtype=np.float32)/26
        
    def call(self, y_true, y_pred):
        # Clip predictions to avoid log(0) errors
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        # Compute the class (0/1) weighted binary cross-entropy losses
        bce_loss = -(self.weight_positive * y_true * tf.math.log(y_pred) +
                     self.weight_negative * (1 - y_true) * tf.math.log(1 - y_pred))
        # Compute the multilabel weighted binary cross-entropy losses
        bce_loss = bce_loss*self.label_weights
        return tf.reduce_mean(bce_loss)


## Squeeze-and-Excitation block

def SE_block(input_se, filters, name, ki):
    x = input_se
    x = tf.keras.layers.GlobalAveragePooling2D(name=name+'_se_gap2d')(x)
    x = tf.keras.layers.Dense(units=int(filters/16), activation="relu", kernel_initializer=ki, name=name+'_se_den')(x)
    x = tf.keras.layers.Dense(units=filters, activation="sigmoid", name=name+'_se_fin')(x)
    x = tf.keras.layers.Reshape((1, 1, filters))(x)
    return x


## Squeeze-and-Excitation network block

def SENet_model(input_conv2d, blocks_conv2d, filters_base_conv2d, reg_cnn, do_cnn, ki_cnn):
    x = input_conv2d

    for i, filters in enumerate([filters_base_conv2d*(2**j) for j in range(blocks_conv2d)]):
        x_sc = x
        ks = 3
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=ks, padding='same', activation=None,
                                   kernel_regularizer=reg_cnn, kernel_initializer=ki_cnn, name=f'conv2d_lay{i+1}1')(x)
        x = tf.keras.layers.BatchNormalization(name=f'bn{i+1}1')(x)
        x = tf.keras.layers.ReLU(name=f'relu{i+1}')(x)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=ks, padding='same', activation=None,
                                   kernel_regularizer=reg_cnn, kernel_initializer=ki_cnn,  name=f'conv2d_lay{i+1}2')(x)
        x = tf.keras.layers.BatchNormalization(name=f'bn{i+1}2')(x)
        x_sc = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same', activation=None,
                                      kernel_regularizer=reg_cnn, kernel_initializer=ki_cnn,  name=f'laysc{i+1}')(x_sc)
        x_sc = tf.keras.layers.BatchNormalization(name=f'bnsc{i+1}')(x_sc)
        x_se = SE_block(x, filters, name=f'layse{i+1}', ki=ki_cnn)
        
        x = tf.keras.layers.Multiply()([x, x_se])
        x = tf.keras.layers.Add(name=f'add{i+1}')([x, x_sc])
        x = tf.keras.layers.ReLU(name=f'relu{i+1}f')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2), name=f'mp{i+1}')(x)
        x = tf.keras.layers.Dropout(do_cnn, name=f'do{i+1}')(x)
    gap2d = tf.keras.layers.GlobalAveragePooling2D(name=f'gap2d')(x)
    flatten = tf.keras.layers.Flatten(name=f'flatten_SE')(x)
    return gap2d, flatten


## Build Squeeze-and-Excitation network (SENet)

def build_network(hp):
    # define the sets of inputs
    input_img = tf.keras.Input(shape=(image_size, image_size, 1), name='input_img')
    input_mod = tf.keras.Input(shape=(2,), name='input_mod')

    # cast and rescale image tensors
    x = tf.keras.layers.Lambda(lambda x: tf.cast(x, dtype=tf.float32)/255)(input_img)

    # define SENet for processing the features
    reg_cnn = None #tf.keras.regularizers.L2(1e-4)
    ki_cnn = 'he_uniform'
    do_cnn = hp.Float(name='do_cnn', min_value=0, max_value=0.4, step=0.05, default=0.05)
    blocks_conv2d =  hp.Int(name='blocks_conv2d', min_value=1, max_value=5, step=1, default=7)
    filters_base_conv2d = hp.Int(name='filters_base_conv2d', min_value=32, max_value=128, step=32, default=16)
    x, flatten = SENet_model(x, blocks_conv2d, filters_base_conv2d, reg_cnn, do_cnn, ki_cnn)

    # combine the output of SENet with the modality input
    x = tf.keras.layers.Concatenate(name='concat_gap')([x, input_mod])
    flatten = tf.keras.layers.Concatenate(name='concat_flatten')([flatten, input_mod]) # optional input for dnn layers
            
    # define DNN layer(s) for classification
    y=x
    reg_dnn = None #tf.keras.regularizers.L2(1e-4)
    ki_dnn = 'he_uniform'
    do_dnn = hp.Float(name='do_dnn', min_value=0, max_value=0.5, step=0.05, default=0.1)
    layers_final = hp.Int(name='layers_final', min_value=0, max_value=2, step=1, default=1)
    units_final = hp.Int(name='units_final', min_value=128, max_value=512, step=128, default=512) #512 ab 1.5.20
    for i, units in enumerate([units_final for j in range(layers_final)]):
        y = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg_dnn,
                                  kernel_initializer=ki_dnn, name=f'final_fc{i+1}')(y)
        y = tf.keras.layers.BatchNormalization(name=f'final_bn{i+1}')(y)
        y = tf.keras.layers.Dropout(do_dnn, name=f'final_do{i+1}')(y)
    output_lab = tf.keras.layers.Dense(len(label2class.keys()), activation="sigmoid", kernel_regularizer=reg_dnn, name='class')(y)
    
    # define DNN layer(s) for regression
    reg_dnn_reg = None #tf.keras.regularizers.L2(1e-4)
    ki_dnn_reg = 'he_uniform'
    do_dnn_reg = do_dnn
    layers_final_reg = hp.Int(name='layers_final_reg', min_value=0, max_value=3, step=1, default=1)
    units_final_reg = hp.Int(name='units_final_reg', min_value=128, max_value=512, step=128, default=256)
    z=x
    for i, units in enumerate([units_final_reg for j in range(layers_final_reg)]):
        z = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg_dnn_reg,
                                  kernel_initializer=ki_dnn_reg, name=f'final_fc_reg{i+1}')(z)
        #z = tf.keras.layers.BatchNormalization(name=f'final_bn_reg{i+1}')(z)
        z = tf.keras.layers.Dropout(do_dnn_reg, name=f'final_do_reg{i+1}')(z)
    output_coord = tf.keras.layers.Dense(2, activation="linear", kernel_regularizer=reg_dnn_reg,
                                         name='reg')(z)

    # define model
    model = tf.keras.Model(inputs=[input_img, input_mod], outputs=[output_lab, output_coord], name='MITSUI_Reg')

    # define optimizer/loss and compile model
    lr_tune = hp.Float(name='learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
    auc = tf.keras.metrics.AUC(multi_label=True, label_weights=np.array(label_weights)/26, name='auc')
    optimizer = tf.keras.optimizers.Adam(lr_tune)
    loss_class = WeightedBinaryCrossentropy()
    loss = {"reg": "mean_squared_error", "class": loss_class}
    loss_weights = {"reg": 0.01, "class": 0.99}
    metrics = {"reg": ["mae"], "class": [auc]}

    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics, run_eagerly=False)
    return model

with strategy.scope():
    model = build_network(kt.HyperParameters())


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

i_TunerTyp = 3 # Choose desired tuner type: {1: 'grid', 2: 'random', 3: 'hyper'}
TunerStr = {1: 'grid', 2: 'random', 3: 'hyper'}

tuner_grid = kt.GridSearch(hypermodel=build_network, objective=kt.Objective("val_class_auc", direction="max"),
                           max_trials=10, max_consecutive_failed_trials=1,
                           overwrite=True, directory="tuner", project_name="RSNA", distribution_strategy = strategy)

tuner_random = kt.RandomSearch(hypermodel=build_network, objective=kt.Objective("val_class_auc", direction="max"),
                               max_trials=10, executions_per_trial=1,
                               overwrite=True, directory="tuner", project_name="RSNA", distribution_strategy = strategy)

tuner_hyper = kt.Hyperband(hypermodel=build_network, objective=kt.Objective("val_class_auc", direction="max"),
                           max_epochs=60, factor=4, hyperband_iterations=1,
                           overwrite=True, directory="tuner", project_name="RSNA", distribution_strategy = strategy)

tuner = globals()[f'tuner_{TunerStr[i_TunerTyp]}']
tuner.search_space_summary()


## Load pre-trained weights for submission or finetuning

if not TRAINING and not TUNING:
    model.load_weights('/kaggle/input/rsna-1xx/rsna_1_5_24.weights.h5')
    print('Model weights have been loaded!')


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
    model.save_weights('rsna_1_5_25.weights.h5')
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
                image_resized, x_resized, y_resized = preprocess_images(frame, -1, -1, crop=False, augmentation=False)
                image_list.append(tf.cast(image_resized, tf.float32))
                mod_list.append(tf.cast(mod, tf.float32))
        else: # Single frame dcm
            image_resized, x_resized, y_resized = preprocess_images(image, -1, -1, crop=False, augmentation=False)
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

if not TRAINING and not TUNING and not FINETUNING: # or True:
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
    
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway()
        display(pl.read_parquet('/kaggle/working/submission.parquet'))


if TUNING:
    shutil.rmtree('/kaggle/working/tuner')


# subfolder_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10134365079002163886508836892471866754'
# file_name = '1.2.826.0.1.3680043.8.498.12524552726742591936607820382077304797.dcm'

# #file_name = '1.2.826.0.1.3680043.8.498.75206494637570575939256404615022232157.dcm'

# image_path = os.path.join(subfolder_path, file_name)
# dcm = pydicom.dcmread(image_path)
# image = pydicom.pixels.pixel_array(dcm, index=149)
# imshow(image)


# ## Analyse pixel value range of images from different modalities

# train2 = train[train['Modality'] == 'MRA'][:100] # CTA/MRA/MRI T2/MRI T1post
# train_images2, train_labels2 = preprocess(train2)
# train_images_reshape2 = tf.reshape(train_images2, [-1])
# values = train_images_reshape2.numpy()
# values = values[values<2000]

# # Plot histogram
# plt.hist(values, bins=300, alpha=0.7, color='blue')
# print(values.max())
# print(values.min())


# ## Plot learning curves for definition of start leraning rate
# lrs = 1e-5 * (10 ** (np.arange(len(history.history["loss"])) / 20)) # Define the learning rate array
# plt.figure(figsize=(10, 6)) # Set the figure size
# plt.grid(True) # Set the grid
# plt.semilogx(lrs, history.history["loss"]) # Plot the loss in log scale
# plt.tick_params('both', length=10, width=1, which='both') # Increase the tickmarks size
# plt.axis([1e-5, 1e-0, 0, 10]) # Set the plot boundaries


# ## Visualisation in 3D

# def load_dicom_series(folder_path): # Function to load DICOM series (multiple slices)
#     import os
#     slices = []
#     for file_name in sorted(os.listdir(folder_path)):
#         if file_name.endswith(".dcm"):
#             dicom_file = pydicom.dcmread(os.path.join(folder_path, file_name))
#             slices.append(dicom_file.pixel_array)
#     return np.stack(slices)

# def plot_3d(image_array, threshold=0.5):
#     x, y, z = np.where(image_array > (image_array.max()-image_array.min())* threshold + image_array.min())
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(projection='3d')
#     colors = image_array[x,y,z]
#     ax.scatter3D(x, y, z, c=colors, vmin=colors.min(), vmax=colors.max(),
#                cmap='gist_rainbow', alpha=0.02, marker='o', s=0.1)
#     ax.view_init(15, 15, 0)
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     ax.set_zlabel('Z-axis')
#     plt.show()

# folder_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317'
# folder_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10935907012185032169927418164924236382'
# dicom_volume = load_dicom_series(folder_path)
# # plot_3d(dicom_volume, threshold=0.04)
# # x, y, z = np.where(dicom_volume > (dicom_volume.max()-dicom_volume.min())* 0.5 + dicom_volume.min())
# # dicom_volume[x, y, z].max()


# # Collect all file names
# folder_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'
# image_list = []
# label_list = []

# start_time = time.time()
# for root, dirs, files in os.walk(folder_path):
#     image_list.extend(files)
# end_time = time.time()

# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")

