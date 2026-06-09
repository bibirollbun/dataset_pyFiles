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



label_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
train_df['label'] = train_df[label_cols].idxmax(axis=1)

train_df = train_df[['image_id', 'label']]

print(train_df.head())
print(train_df['label'].value_counts())



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

train_df['image_id'] = train_df['image_id'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')
val_df['image_id'] = val_df['image_id'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

print(train_df.head())
print(val_df.head())

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)
BATCH_SIZE = 16

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



from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam




num_classes = 4

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)




# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,  
    verbose=1
)



predictions = model.predict(val_df, verbose=1)



predicted_classes = predictions.argmax(axis=1)



labels = list(train_gen.class_indices.keys())
predicted_labels = [labels[i] for i in predicted_classes]



filenames = test_gen.filenames
df = pd.DataFrame({
    'id': [f.split('/')[-1] for f in filenames],  # image names
    'label': predicted_labels
})


df.to_csv('submission.csv', index=False)


