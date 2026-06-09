# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path
import os.path
import warnings
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = '/kaggle/input/happy-whale-and-dolphin/train_images'


train_csv = pd.read_csv('/kaggle/input/happy-whale-and-dolphin/train.csv')


train_csv['image']  = train_csv['image'].apply(lambda x : train + '/'+ x)


train_csv['species'] = train_csv['species'].replace({
    'false_killer_whale' : 'killer_whale',
    'bottlenose_dolpin' : 'bottlenose_dolphin',
    'kiler_whale' : 'killer_whale',
    'short_finned_pilot_whale' : 'pilot_whale',
    'long_finned_pilot_whale' :  'pilot_whale',
    'pygmy_killer_whale' : 'killer_whale'
})


train_csv['species'].value_counts()


len(train_csv['species'].value_counts())


import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def adjust_brightness_contrast(image, alpha=1.5, beta=50):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def equalize_histogram(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:  # Color image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
    return image
def sharpen_image(image):
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    return cv2.filter2D(image, -1, sharpening_kernel)

def remove_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)
def custom_preprocessing(image):
    # Convert image from range [0, 1] to [0, 255]
    image = image * 255.0
    image = image.astype(np.uint8)
    
    # Apply preprocessing steps
    image = remove_noise(image)
    image = adjust_brightness_contrast(image)
    image = equalize_histogram(image)
    image = sharpen_image(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    # Convert image back to range [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image


train_df , test_df = train_test_split(train_csv, test_size = 0.20, shuffle = True, random_state = 42)


train_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = custom_preprocessing,validation_split = 0.2)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = custom_preprocessing)


#K = 2
#skf = StratifiedKFold(n_splits=K, random_state=SEED, shuffle=True)

#DISASTER = df_train['target'] == 1
# print('Whole Training Set Shape = {}'.format(df_train.shape))
# print('Whole Training Set Unique keyword Count = {}'.format(df_train['keyword'].nunique()))
# print('Whole Training Set Target Rate (Disaster) {}/{} (Not Disaster)'.format(df_train[DISASTER]['target_relabeled'].count(), df_train[~DISASTER]['target_relabeled'].count()))

# for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train['text_cleaned'], df_train['target']), 1):
#     print('\nFold {} Training Set Shape = {} - Validation Set Shape = {}'.format(fold, df_train.loc[trn_idx, 'text_cleaned'].shape, df_train.loc[val_idx, 'text_cleaned'].shape))
#     print('Fold {} Training Set Unique keyword Count = {} - Validation Set Unique keyword Count = {}'.format(fold, df_train.loc[trn_idx, 'keyword'].nunique(), df_train.loc[val_idx, 'keyword'].nunique()))


train_image = train_gen.flow_from_dataframe(
    dataframe = train_df,
    x_col = 'image',
    y_col = 'species',
    target_size = (224, 224),
    color_mode ='rgb',
    class_mode = 'categorical',
    batch_size = 32,
    shuffle = True,
    seed = 42,
    subset = 'training'
)
val_image = train_gen.flow_from_dataframe(
    dataframe = train_df,
    x_col = 'image',
    y_col = 'species',
    target_size = (224, 224),
    color_mode ='rgb',
    class_mode = 'categorical',
    batch_size = 32,
    shuffle = True,
    seed = 42,
    subset = 'validation'
)
test_image = test_gen.flow_from_dataframe(
    dataframe = test_df,
    x_col = 'image',
    y_col = 'species',
    target_size = (224, 224),
    color_mode ='rgb',
    class_mode = 'categorical',
    batch_size = 32,
    shuffle = False
)


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices, 'GPU')


model = Sequential()
model.add(Input(shape=(224,224,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.27)))
model.add(Flatten())
model.add(Dense(24, kernel_initializer = 'normal', activation='softmax'))

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy',"mean_squared_error"]
)

history = model.fit(
    train_image,
    validation_data = val_image,
    epochs = 100,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 3,
            restore_best_weights = True
        )
    ]
)

