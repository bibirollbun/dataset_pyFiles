# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# Set paths (update these with your actual file paths)
train_csv_path = '/kaggle/input/siim-isic-melanoma-classification/train.csv'
train_image_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train'

# Load data
train_df = pd.read_csv(train_csv_path)
train_df['image_path'] = train_df['image_name'].apply(lambda x: os.path.join(train_image_path, f'{x}.jpg'))
train_df['target'] = train_df['target'].astype(str)  # Convert target to string for ImageDataGenerator

# Image preprocessing parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
K_FOLDS = 5

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Define model
def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# K-Fold Cross-Validation
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_no = 1
val_scores = []

for train_index, val_index in kf.split(train_df):
    print(f'Training fold {fold_no}...')
    
    # Split data
    train_data = train_df.iloc[train_index]
    val_data = train_df.iloc[val_index]
    
    # Create data generators
    train_generator = datagen.flow_from_dataframe(
        train_data,
        x_col='image_path',
        y_col='target',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    val_generator = datagen.flow_from_dataframe(
        val_data,
        x_col='image_path',
        y_col='target',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    # Create and train model
    model = create_model()
    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        verbose=1
    )
    
    # Evaluate model
    val_score = model.evaluate(val_generator)[1]  # Get accuracy

