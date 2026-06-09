# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('../input/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv')

pd.set_option('display.max_columns', None)

train.tail(2)


train.info()


test = pd.read_csv('../input/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Test.csv')
#pd.set_option('display.max_columns', None)

test.tail(2)


test.shape


val = pd.read_csv('../input/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Val.csv')

#pd.set_option('display.max_columns', None)

val.tail(2)


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

#Importing (Not mine) Beautiful functions for flexibility
!wget https://raw.githubusercontent.com/elinteerie/helper/main/helper_functions.py

from helper_functions import *


dir = '../input/fungi-clef-2025/images/FungiTastic-FewShot'


train_dir = '../input/fungi-clef-2025/images/FungiTastic-FewShot/train' # Data is already splitted test/validation


test_dir = '../input/fungi-clef-2025/images/FungiTastic-FewShot/test'


val_dir = '../input/fungi-clef-2025/images/FungiTastic-FewShot/val'


#Code by Igwe Ugochukwu Sylvester Michael https://www.kaggle.com/code/elinteerie/nigeria-food-ai-model-with-kaggle/notebook

IMG_SIZE = (224, 224)
train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                 labels ='inferred',
                                                                 label_mode="int",
                                                                 image_size=IMG_SIZE,
                                                                 #validation_split=0.15,
                                                                 #subset='training',
                                                                 seed =42)

test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                labels ='inferred',
                                                                 label_mode="int",
                                                                 image_size=IMG_SIZE,
                                                                 #validation_split=0.15,
                                                                 #subset='testing',
                                                                seed =42)

val_data = tf.keras.preprocessing.image_dataset_from_directory(val_dir,
                                                                labels ='inferred',
                                                                 label_mode="int",
                                                                 image_size=IMG_SIZE,
                                                                 #validation_split=0.15,
                                                                 #subset='validation',
                                                                seed =42)


#Code by Igwe Ugochukwu Sylvester Michael https://www.kaggle.com/code/elinteerie/nigeria-food-ai-model-with-kaggle/notebook

ngclass_names = train_data.class_names

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(ngclass_names[labels[i].numpy()])
    plt.axis("off")


#Code by Igwe Ugochukwu Sylvester Michael https://www.kaggle.com/code/elinteerie/nigeria-food-ai-model-with-kaggle/notebook

###Setting Up Data Argumentation
from tensorflow.keras import Sequential

data_augmentation = Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.2),
  layers.RandomHeight(0.2),
  layers.RandomWidth(0.2),
  layers.RandomZoom(0.2),
  # preprocessing.Rescaling(1/255.) # rescale inputs of images to between 0 & 1, required for models like ResNet50 but i am using EfficientNetX
], name="data_augmentation")


#Code by Igwe Ugochukwu Sylvester Michael https://www.kaggle.com/code/elinteerie/nigeria-food-ai-model-with-kaggle/notebook

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)


train_data


#Code by Igwe Ugochukwu Sylvester Michael https://www.kaggle.com/code/elinteerie/nigeria-food-ai-model-with-kaggle/notebook

# Setup the base model and freeze its layers (this will extract features)
base_model = tf.keras.applications.EfficientNetB4(include_top=False)
base_model.trainable = False

# Setup model architecture with trainable top layers
inputs = layers.Input(shape=(224, 224, 3), name="input_layer")
x = data_augmentation(inputs) # augment images (only happens during training phase)
x = base_model(x, training=False) # put the base model in inference mode so weights which need to stay frozen, stay frozen
x = layers.GlobalAveragePooling2D(name="global_avg_pool_layer")(x)
outputs = layers.Dense(14, activation="softmax", name="output_layer")(x)
model = tf.keras.Model(inputs, outputs)


model.summary()


#Code by Igwe Ugochukwu Sylvester Michael https://www.kaggle.com/code/elinteerie/nigeria-food-ai-model-with-kaggle/notebook

# Compile
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])


# Fit
#history_model = model.fit(train_data,epochs=3, # fit for 3 epochs to keep experiments quick
                                           #validation_data=val_data,
                                           #validation_steps=int(len(val_data)), # validate on only 15% of test data during training
                                           #callbacks=[create_tensorboard_callback(dir_name = 'TensorBoard', experiment_name ='model_1b3'),
                                                      #create_model_checkpoint('Checkpoint')])


print(train['family'].unique())


print("Total:\t\t", train['family'].count())
print("")
print(train['family'].value_counts())


fig = plt.figure(figsize=(16, 5))
sns.set_theme(font_scale=2,palette="Set2")
sns.countplot(x=train['family'],
            order=train['family'].value_counts().head(10).index).set(title='Count of Fungi by Families')
plt.xticks(rotation=20);


print("Total:\t\t", train['genus'].count())
print("")
print(train['genus'].value_counts())


print("Total:\t\t", train['habitat'].count())
print("")
print(train['habitat'].value_counts())


print("Total:\t\t", train['substrate'].count())
print("")
print(train['substrate'].value_counts())


#By Paulo Junqueira https://www.kaggle.com/code/paulojunqueira/pew-pew-overview-birdclef-2023/notebook

import plotly.graph_objects as go
from scipy.interpolate import interp1d 


#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'


df_plot = train.groupby(['poisonous','latitude', 'longitude']).count().reset_index()[['poisonous','scientificName','latitude', 'longitude']].rename(columns = {'scientificName':'count'})
meta_2 = train.merge(df_plot, on = ['poisonous','latitude', 'longitude'], how = 'left').dropna(subset = ['count'])
meta_2['count'] = meta_2['count'].astype('int')

values_list = meta_2['count'].values.tolist()

interpolation = interp1d([1, max(values_list)], [3,20])
radius = interpolation(values_list)
fig = go.Figure(go.Densitymapbox(lat =meta_2['latitude'],lon = meta_2['longitude'], radius = radius,z = meta_2['count']))

fig.update_layout(mapbox_style="open-street-map",height = 800,
                  mapbox = {
                          'center': {'lat': 0, 
                          'lon': 0},
                      'zoom':0
                  })
fig.show()


print("Total:\t\t", train['poisonous'].count())
print("")
print(train['poisonous'].value_counts())


#Boolean selection of fungi using the .loc operator

train.loc[(train["poisonous"] == 1)]


train.sort_values('year')


#Groupby Biogeographical Region and Year
#Calculate and sort_values to show on Top the Biogeographical Regions.

train.groupby(['biogeographicalRegion','year']).size().reset_index(name='count').sort_values(by='count', ascending=False) 


#By Ben Jenkins https://www.kaggle.com/code/benjenkins96/identify-eastern-african-bird-species-by-sound

# Set up a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot a histogram of the latitude values
train['year'].hist(bins=50, ax=axs[0, 0])
axs[0, 0].set_title('Distribution of Year', color='red')
axs[0, 0].set_xlabel('Year', color='red')
axs[0, 0].set_ylabel('Count', color='red')
# Plot a histogram of the longitude values
train['poisonous'].hist(bins=50, ax=axs[0, 1])
axs[0, 1].set_title('Distribution of Poisonous', color='red')
axs[0, 1].set_xlabel('Poisonous', color='red')
axs[0, 1].set_ylabel('Count', color='red')

# Plot a scatterplot of the latitude and longitude values
train.plot.scatter(x='longitude', y='latitude', alpha=0.1, ax=axs[1, 0])
axs[1, 0].set_title('Geographic Distribution of Fungi', color='red')
axs[1, 0].set_xlabel('Longitude', color='red')
axs[1, 0].set_ylabel('Latitude', color='red')

# Print the top 10 authors with the most recordings
train['species'].value_counts().nlargest(10).plot.barh(ax=axs[1, 1])
axs[1, 1].set_title('Top 10 Fungi Species', color='red')
axs[1, 1].set_xlabel('Count', color='red')
axs[1, 1].set_ylabel('Author', color='red')

# Adjust the layout of the subplots
plt.tight_layout()

