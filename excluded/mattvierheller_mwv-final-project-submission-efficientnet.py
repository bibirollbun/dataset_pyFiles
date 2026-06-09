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

test_loader = test_datagen.flow_from_dataframe(
    dataframe = test_df,
    directory = test_ref,
    x_col = 'filenames',
    batch_size = batch_size,
    shuffle = False,
    class_mode = None,
    target_size = (260,260)
)


cnn_model = keras.models.load_model('/kaggle/input/cnn-model-5/Cancer_Detection_cnn_model_5.h5')
cnn_model.summary()


test_pred = cnn_model.predict(test_loader)
print(test_pred)


print(test_pred[:10].round(2))


cancer_submission = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/sample_submission.csv')
cancer_submission.label = test_pred
cancer_submission.head(25)


cancer_submission.to_csv('cancer_detection_submission_tl_efficientnet.csv', index = False, header = True)




