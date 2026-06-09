import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, datasets


test_df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/sample_submission.csv')
test_df.head()


print(test_df.shape)


test_df['filenames'] = test_df['id'] + '.tif'


test_df.head()


test_ref = '/kaggle/input/histopathologic-cancer-detection/test'
print(f'Test Images Shape {len(test_ref)}')


batch_size = 25
test_datagen = ImageDataGenerator(rescale = 1/255)

test_loader_1 = test_datagen.flow_from_dataframe(
    dataframe = test_df,
    directory = test_ref,
    x_col = 'filenames',
    batch_size = batch_size,
    shuffle = False,
    class_mode = None,
    target_size = (260,260)
)

test_loader_2 = test_datagen.flow_from_dataframe(
    dataframe = test_df,
    directory = test_ref,
    x_col = 'filenames',
    batch_size = batch_size,
    shuffle = False,
    class_mode = None,
    target_size = (96,96)
)


cnn_model_1 = keras.models.load_model('/kaggle/input/cnn-model-5/Cancer_Detection_cnn_model_5.h5')
cnn_model_1.summary()


cnn_model_2 = keras.models.load_model('/kaggle/input/mwv-final-project-training-model-3/Cancer_Detection_cnn_model_2.h5')
cnn_model_2.summary()


test_pred_1 = cnn_model_1.predict(test_loader_1)
print(test_pred_1)


test_pred_2 = cnn_model_2.predict(test_loader_2)
print(test_pred_2)


print(test_pred_1[:10].round(2))
print(test_pred_1)


print(test_pred_2[:10].round(2))





ensemble_prediction = (test_pred_1+test_pred_2)/2




cancer_submission = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/sample_submission.csv')
cancer_submission.label = ensemble_prediction
cancer_submission.head(25)


cancer_submission.to_csv('cancer_detection_submission_ensemble.csv', index = False, header = True)


https://www.kaggle.com/code/mattvierheller/mwv-final-project-training-model-5
https://www.kaggle.com/code/mattvierheller/mwv-final-project-training-model-3


