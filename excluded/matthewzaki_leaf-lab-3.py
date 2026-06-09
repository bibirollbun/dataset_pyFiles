# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



print("TensorFlow version:", tf.__version__)



data_dir = "/kaggle/input/plant-pathology-2020-fgvc7"
train_df = pd.read_csv(f"{data_dir}/train.csv")
train_df.head()



from PIL import Image

img_path = f"/kaggle/input/plant-pathology-2020-fgvc7/images/{train_df.iloc[0,0]}.jpg"
label = train_df.iloc[0,1]

print("Image label:", label)
Image.open(img_path)
 


# Convert one-hot encoded columns into a single label column
label_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
train_df['label'] = train_df[label_cols].idxmax(axis=1)

# Keep only image_id and label
train_df = train_df[['image_id', 'label']]

print(train_df.head())
print(train_df['label'].value_counts())



# Remove double .jpg if exists
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

train_df['image_id'] = train_df['image_id'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')
val_df['image_id'] = val_df['image_id'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

# Double-check the first few rows
print(train_df.head())
print(val_df.head())

# Create ImageDataGenerators again
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)



# Recreate generators
train_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory=f"{data_dir}/images",
    x_col='image_id',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    directory=f"{data_dir}/images",
    x_col='image_id',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE
)



from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout,Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential


def build_basic_cnn(input_shape=(128, 128, 3), num_classes=4):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

   
    for layer in base_model.layers:
        layer.trainable = False
        
    print("Unfreezing layers in block5...")
    for layer in base_model.layers:
        if layer.name.startswith('block5'):
            layer.trainable = True
            print(f"  - Unfrozen: {layer.name}")
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  
    ])
    return model

model = build_basic_cnn()
model.summary()


from tensorflow.keras.optimizers import Adam

fine_tune_lr = 1e-5 

model.compile(
    optimizer=Adam(learning_rate=fine_tune_lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)



# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()

plt.show()


