# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



!pip install -qU ../input/for-pydicom/python_gdcm-3.0.22-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl ../input/for-pydicom/pylibjpeg-1.4.0-py3-none-any.whl --find-links frozen_packages --no-index


import pandas as pd
import numpy as np
import pydicom as dicom
import glob
import nibabel as nib
import os
import re
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import InceptionV3, DenseNet121, InceptionResNetV2

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical, plot_model
from keras.preprocessing import image

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
pd.set_option('display.max_columns', None)


base_dir = r'/kaggle/input/rsna-2022-cervical-spine-fracture-detection'
train_images = os.path.join(base_dir,'train_images')
test_images = os.path.join(base_dir,'test_images')
segmentation_data = r'/kaggle/input/rsna-cervical-fracture-segmentations-npy/npy_segmentations'
train_data = pd.read_csv(os.path.join(base_dir,'train.csv'))
segmentation_meta_data = pd.read_csv(r'/kaggle/input/rsna-cervical-fracture-segmentation-metadata/meta_segmentation.csv')



segmentation_meta_data.shape


segmentation_meta_data.columns


segmentation_meta_data['PhotometricInterpretation'].value_counts()


columns = ['StudyInstanceUID','SOPInstanceUID','C1','C2','C3','C4','C5','C6','C7']


seg_labels = segmentation_meta_data[columns]


seg_labels.head(2)


#Get Slice instance number
seg_labels.loc[:,'slice'] = seg_labels['SOPInstanceUID'].apply(lambda x:x.split('.')[-1:][0])


seg_labels


# Function to load DICOM images
def load_scan(dcm_paths):  
    patient_scan = [dicom.dcmread(paths) for paths in dcm_paths]
    return patient_scan

def get_pixels_hu(img):
    image = cv2.resize(img.pixel_array,(128, 128),interpolation = cv2.INTER_NEAREST)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0, the intercept is usually -1024, so air is approximately 0
    image[image <= -1000] = 0
    # Convert to Hounsfield units (HU)    
    intercept = np.array(img.RescaleIntercept)
    slope = np.array(img.RescaleSlope)
    image= (slope * image.astype("float64")) + intercept
#     plt.imshow(image.astype("int16"), cmap='bone') 
    return image.astype("int16")


list(seg_labels['StudyInstanceUID'].unique())


def get_image(study_instance):
    path = '/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images'
    # study_instances = list(seg_labels['StudyInstanceUID'].unique())
    patient_slices = []
    org_images = []
#     study_instance = '1.2.826.0.1.3680043.1868'
    # for study_instance in study_instances:
    slices = list(seg_labels[seg_labels['StudyInstanceUID']==study_instance]['slice'])
    dcm_paths = [path+'/'+study_instance+'/'+ slic + '.dcm' for slic in slices]
    image = load_scan(dcm_paths)
    org_images.append(image)
    slices_p = [dicom.read_file(dcm_path) for dcm_path in dcm_paths]
    patient_slice = [get_pixels_hu(slic) for slic in slices_p]
    patient_slices.append(patient_slice)
    
    return org_images, patient_slices, slices


study_instance = '1.2.826.0.1.3680043.1868'
org_images, patient_slices, slices = get_image(study_instance = study_instance)


dat  = seg_labels[seg_labels['StudyInstanceUID']==study_instance][['SOPInstanceUID', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']]
dat


plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)

start = 0
img = 67
label = dat[dat['SOPInstanceUID']==study_instance+'.1.'+slices[start]][['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].to_string(index=False).split('\n')
plt.figure(figsize=(11, 8))
# Ploting pixel array
plt.subplot(2, 2, 1)
plt.imshow(org_images[start][img].pixel_array,cmap='bone', aspect='auto')
plt.title('Original image')
plt.axis("off")

# Ploting pixel array distribution
plt.subplot(2, 2, 2)
plt.hist(org_images[start][img].pixel_array.flatten(),color="b",bins=50)
# plt.title('Pixel array distribution')
plt.xlabel("Pixel Values")
plt.ylabel("Fequency")

#Ploting HU array
plt.subplot(2, 2, 3)
plt.imshow(get_pixels_hu(org_images[start][img]),cmap='bone', aspect='auto')
plt.title('Processed image')
plt.axis("off")

# Ploting HU distribution
plt.subplot(2, 2, 4)
plt.hist(patient_slices[start][img].flatten(),color="b",bins=50)
# plt.title('HU distribution')
plt.xlabel("HU Values")
plt.ylabel("Fequency")
plt.suptitle(f"{label[0]} \n {label[1].strip()}", y=0.98, fontsize=12)
plt.show()


label, label[1].split()


def load_dicom(path):
    '''
    Function to load and transform DICOM images.
    
    Parameters:
    path(string): Path to the DICOM images

    Returns:
    Transformed and resized image.
    
    '''
    img=dicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
#     data=img.pixel_array
    data=get_pixels_hu(img)
    data=data-np.min(data)
    if np.max(data) != 0:
        data=data/np.max(data)
    data=(data*255).astype(np.uint8)        
    return cv2.cvtColor(data.reshape(128, 128), cv2.COLOR_GRAY2RGB)

def ImgDataGenerator(train_df,base_path):
    '''
    Function to read dicom image path and store the images as numpy arrays.

    Parameters:
    train_df: Pandas dataframe.
    base_path: Python list containing image filepaths.

    Returns:
    [Train image dataset, Train image labels]

    '''
    trainset = []
    trainlabel = []
    for i in tqdm(range(len(train_df))):
        study_id = train_df.loc[i,'StudyInstanceUID']
        slice_id = train_df.loc[i,'slice']+'.dcm'
        study_path = study_id+'/'+slice_id

        path = os.path.join(base_path, study_path)

        img = load_dicom(path)
        img = cv2.resize(img, (128 , 128))
        image = img_to_array(img)
        image = image / 255.0
        trainset += [image]
        cur_label = [train_df.loc[i,f'C{j}'] for j in range(1,8)]
        trainlabel += [cur_label]

    return np.array(trainset), np.array(trainlabel)
    
def metrics(y_test, y_pred_binary):
    '''
    Function to display accuracy, precision, recall and f1-score for the classification task.
    
    Parameters:
    y_test: True labels.
    y_pred_binary: Predicted binary labels.

    Returns:
    Pandas dataframe containing class-wise Sensitivity, Specificity, and F1-score.
    
    '''
    classes = np.array(seg_labels.columns[2:-1])
    df_res = []
    precision_per_class = precision_score(y_test, y_pred_binary, average=None)
    recall_per_class = recall_score(y_test, y_pred_binary, average=None)
    f1_per_class = f1_score(y_test, y_pred_binary, average=None)

    for i in range(len(classes)):
        df_res.append([classes[i], recall_per_class[i], precision_per_class[i], f1_per_class[i]])
    df_res = pd.DataFrame(df_res, columns = ['Class','Sensitivity','Specificity', 'F1-score'])
    return df_res

def plot_history(history):
    '''
    Function to plot the train and validation accuracy and loss.
    
    Parameters:
    history: model train history

    Returns:
    None.
    
    '''
    hist = history.history
    plt.figure(figsize=(8, 4));
    plt.suptitle(f"Performance Metrics", fontsize=12)

    # Actual and validation losses
    plt.subplot(1, 2, 1);
    plt.plot(hist['loss'], label='train')
    plt.plot(hist['val_loss'], label='validation')
    plt.title('Train and val loss curve', fontsize=8)
    plt.legend()

    # Actual and validation accuracy
    plt.subplot(1, 2, 2);
    plt.plot(hist['binary_accuracy'], label='train')
    plt.plot(hist['val_binary_accuracy'], label='validation')
    plt.title('Train and val accuracy curve', fontsize=8)
    plt.legend();
    
def callback(model_name, patience=5): 
    '''
    Function to define callback for model training.
    
    Parameters:
    model_name(string): Name for the saved model with `.h5` extension.
    patience: Patience for early stopping. Usually, the value lies between 5-11.

    Returns:
    [Early Stopping Callback, Model Checkpoint Callback]
    
    '''
    early_stopping = callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    model_checkpoint = callbacks.ModelCheckpoint(model_name, save_best_only=True)
    learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                                        patience=2, 
                                                        verbose=1, 
                                                        factor=0.5, 
                                                        min_lr=0.00001)
    return [early_stopping, model_checkpoint, learning_rate_reduction]


# Convert train images of segmented studyids to array
X_seg, y_seg = ImgDataGenerator(seg_labels,train_images)
X_seg.shape,y_seg.shape


# Divide train and test data
X_train, X_test, y_train, y_test = train_test_split(X_seg, y_seg, random_state=42, test_size=0.1)
y_train, y_test = y_train.astype('float32'), y_test.astype('float32')
X_train.shape, y_train.shape, X_test.shape, y_test.shape


labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']


plt.figure(figsize=(7, 5))
plt.bar(x=labels, height=np.mean(y_train, axis=0))
plt.title("Frequency of Each Class")
plt.show()


# Prepare data batches for training images
X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
train_dataset = tf.data.Dataset.zip((X_train_tensor, y_train_tensor)).batch(16).prefetch(tf.data.AUTOTUNE)
# Prepare data batches for validation images
X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)
val_dataset = tf.data.Dataset.zip((X_test_tensor, y_test_tensor)).batch(16).prefetch(tf.data.AUTOTUNE)


def conv_block(input, num_filters):
    '''
    Function for convolution block unit.
    
    Parameters:
    input(keras layer): Input layer.
    num_filters(int): Number of filters in the Conv2D layer. 

    Returns:
    Final convoluted and activated output layer.
    
    '''
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    '''
    Function for convolution block unit.
    
    Parameters:
    input(keras layer): Input layer.
    num_filters(int): Number of filters in the Conv2D layer. 

    Returns:
    Final convoluted and activated output layer.
    
    '''
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_inception_resnetv2_unet(input_shape):
    # Input layer
    inputs = layers.Input(input_shape)
    
    # Pre-trained transfer learning model
    encoder = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=inputs)
#     encoder.trainable = False
    # Encoder
    s1 = encoder.get_layer("input_1").output           ## (512 x 512)
    s2 = encoder.get_layer("activation").output        ## (255 x 255)
    s2 = layers.ZeroPadding2D(((1, 0), (1, 0)))(s2)         ## (256 x 256)
    s3 = encoder.get_layer("activation_3").output      ## (126 x 126)
    s3 = layers.ZeroPadding2D((1, 1))(s3)                     ## (128 x 128)
    s4 = encoder.get_layer("activation_74").output      ## (61 x 61)
    s4 = layers.ZeroPadding2D(((2, 1), (2, 1)))(s4)           ## (64 x 64)

    # Bridge
    b1 = encoder.get_layer("activation_161").output     ## (30 x 30)
    b1 = layers.ZeroPadding2D((1, 1))(b1)                      ## (32 x 32)

    # Decoder layer
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)
    
    # Output layer
    gap = layers.GlobalAveragePooling2D()(d4)
    dropout = layers.Dropout(0.2)(gap)
    outputs = layers.Dense(7, activation="sigmoid")(dropout)
    # Build the model
    model = Model(inputs, outputs, name="InceptionResNetV2-UNet")
    
    return encoder, model

inception_resnet, model = build_inception_resnetv2_unet(input_shape = (128, 128, 3))


# inception_resnet.summary() #none:3, 3:13, 74:266, 161:606
for i in range(len(inception_resnet.layers)):
    print (i, inception_resnet.layers[i].name)


plot_model(inception_resnet, to_file='model_plot_inception_resnet.png', show_shapes=True, show_layer_names=True)


model_0 = model


plot_model(model_0, to_file='model_plot_inception_resnet_unet.png', show_shapes=True, show_layer_names=True)


for i in range(len(model_0.layers)):
    print (i, model_0.layers[i].name)
    
for layer in model_0.layers[274:]:
    layer.trainable=True
for layer in model_0.layers[0:274]:
    layer.trainable=False


model_0.summary()


# Compile the model
model_0.compile(loss="binary_crossentropy",
              optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

# Train the model
history_model_0 = model_0.fit(train_dataset, 
                              epochs=100, 
                              validation_data=val_dataset,
                              steps_per_epoch=int(len(train_dataset)/32),
                              validation_steps=int(len(val_dataset)),
                              callbacks=[callback(patience=15, model_name="model_0_InceptionResNetV2.h5")])

