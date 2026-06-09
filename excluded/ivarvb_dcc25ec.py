__author__ = "Ivar Vargas Belizario"
__copyright__ = "Copyright 2025"
__credits__ = ["Ivar Vargas Belizario"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Ivar Vargas Belizario"
__email__ = "ivargasbelizario@gmail.com"
__status__ = "development"


import shutil
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import time
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import RMSprop

import random
import numpy as np
import sys
import os

# import tensorflow_addons as tfa
from sklearn import metrics

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as ker
#from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import DenseNet121
#from tensorflow.keras.backend import set_session
from tensorflow.python.keras.backend import set_session 
from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix 
import seaborn as sns
import matplotlib.ticker as mticker
import pandas as pd

print("tf.__version__", tf.__version__)
# print("tf.keras.__version__", tf.keras.__version__)

input_dir = "/kaggle/input/"
output_dir = "/kaggle/working/"
# en la lectura de imagenes
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).


def readData(ds):
    df_train = pd.read_csv('/kaggle/input/'+ds+'/train.csv')
    df_test = pd.read_csv('/kaggle/input/'+ds+'/test.csv')
    df_submission = pd.read_csv('/kaggle/input/'+ds+'/sample_submission.csv')
    return df_train, df_test, df_submission
    
def getdata(ds, ipw, iph, ipd):    
    train_data_dir = os.path.join(input_dir, ds, 'train')
    # validation_data_dir = os.path.join(input_dir, ds, 'valid')
    test_data_dir = os.path.join(input_dir, ds, 'test')
        
    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
    # To save augmentations un-comment save lines and add to your flow parameters.
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       #brightness_range=[0.2,0.4],
                                       #brightness_range=[0.5,1.0],
                                       # brightness_range=[0.5, 0.7],
                                       #zoom_range=[0.1,0.2],
                                       #shear_range=transformation_ratio,
                                       #zoom_range=transformation_ratio,
                                       #cval=transformation_ratio,
                                       #rotation_range=45,
                                       #fill_mode='nearest',
                                       #horizontal_flip=True,
                                       #vertical_flip=True
                                       #preprocessing_function=add_noise,
                                       validation_split=0.2
                                      )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    test_datagen = ImageDataGenerator(rescale=1/255.)
    # https://keras.io/api/data_loading/image/
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        #target_size=(ipw, iph),
                                                        target_size=(ipw, iph),
                                                        batch_size=batch_size,
                                                        #image_size=(64, 64),
                                                        interpolation="bilinear",
                                                        shuffle=True,
                                                        class_mode='categorical',
                                                        
                                                        subset='training',

                                                       )
    
    
    
    validation_generator = train_datagen.flow_from_directory(
                                                        train_data_dir,
                                                        #target_size=(ipw, iph),
                                                        target_size=(ipw, iph),
                                                        batch_size=batch_size,
                                                        #image_size=(64, 64),
                                                        interpolation="bilinear",
                                                        shuffle=False,
                                                        class_mode='categorical',

                                                        subset='validation'

                                                                 ) 
                                                                     
    
    test_generator = test_datagen.flow_from_directory(
                                                        test_data_dir,
                                                        target_size=(ipw, iph),
                                                        batch_size=batch_size,
                                                        #shuffle=False,
                                                        interpolation="bilinear",
                                                        shuffle=False,
                                                        class_mode='categorical')
    
    #recuperar el nombre de las clases
    ilabelsclass = {v: k for k, v in test_generator.class_indices.items()}
    
    print("ilabelsclass", ilabelsclass)

    # retorna la configuración de lectura de imagenes
    return ilabelsclass, train_generator, validation_generator, test_generator

    

if __name__ == '__main__':

    # (I) DATOS ESTRUCTURADOS 
    
    print("Binary Prediction of Smoker Status using Bio-Signals")
    df_train, df_test, df_submission = readData("playground-series-s3e24")
    print(df_train, df_test, df_submission)


    print("Exploring Mental Health Data")
    df_train, df_test, df_submission = readData("playground-series-s4e11")
    print(df_train, df_test, df_submission)

    
    #  (II) DATOS NO ESTRUCTURADOS 
    models = [ 
                {"model":"MobileNet","ds":"melanomads","classes":2,"w":32,"h":32,"d":3},

                {"model":"MobileNet","ds":"5birds","classes":5,"w":32,"h":32,"d":3},
            
    ]

    for m in models:
        print("Model", m["model"], m["ds"])        
        ilabelsclass, train_generator, validation_generator, test_generator = getdata(m["ds"], m["w"], m["h"], m["d"])


### INCLUIR EL CODIGO DESDE LOS SIGUIENTES NOTEBOOKS (PRACTICAS EN CLASE)
### https://www.kaggle.com/code/ivarvb/dcc25pc1
### https://www.kaggle.com/code/ivarvb/dcc25pc2

