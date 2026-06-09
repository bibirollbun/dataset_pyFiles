import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import tifffile as tiff
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator


with open('/kaggle/input/model-5/Cancer_Detection_model_5.pk1', 'rb') as file: 
    history = pickle.load(file)
    
cnn = load_model('/kaggle/input/model-5/Cancer_Detection_cnn_model_5.h5')


test_datagen = ImageDataGenerator(rescale=1/255)


test = pd.read_csv("/kaggle/input/histopathologic-cancer-detection/sample_submission.csv")
test['id'] = test['id'].apply(lambda x:f'{x}.tif')

test_generator = test_datagen.flow_from_dataframe(
    dataframe = test,
    directory = '/kaggle/input/histopathologic-cancer-detection/test',
    x_col = "id",
    y_col = None,
    batch_size = 100,
    seed = 1,
    shuffle = False,
    class_mode = None,
    target_size = (260,260)
)


test_pred = np.argmax(cnn.predict(test_generator), axis=-1)


Submission = test
Submission['label'] = test_pred
Submission['id'] = Submission['id'].str.replace(r'.tif$', '',regex=True)


Submission


Submission.to_csv('submission.csv',index=False,header=True)

