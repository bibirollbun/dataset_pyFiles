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


# ðŸ“Œ Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import applications, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ðŸ“Œ Constants
IMG_SIZE = (224, 224)  # ConvNeXtXLarge supports 224x224 input
BATCH_SIZE = 32


# ðŸ“Œ Load dataset
df = pd.read_csv('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv')
print(df.head())
print(df['label'].value_counts())


# ðŸ“Œ Encode labels
le = LabelEncoder()
df['encoded_label'] = le.fit_transform(df['label'])
NUM_CLASSES = len(le.classes_)
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))


# ðŸ“Œ Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory='/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train',
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='categorical',
    subset='training',
    batch_size=BATCH_SIZE
)

val_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory='/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train',
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='categorical',
    subset='validation',
    batch_size=BATCH_SIZE
)



# ðŸ“Œ Create ConvNeXtXLarge model
def create_model():
    base_model = applications.ConvNeXtXLarge(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False  # Freeze base

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model = create_model()
model.summary()



# ðŸ“Œ Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3)
]



# ðŸ“Œ Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)


# ðŸ“Œ Prepare test data
test_df = pd.DataFrame({'filename': os.listdir('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test')})

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    dataframe=test_df,
    directory='/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test',
    x_col='filename',
    y_col=None,
    target_size=IMG_SIZE,
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ðŸ“Œ Predict
preds = model.predict(test_gen)
pred_classes = le.inverse_transform(np.argmax(preds, axis=1))



# ðŸ“Œ Save submission
test_df['label'] = pred_classes
test_df[['filename', 'label']].to_csv('submission.csv', index=False)
print("âœ… submission.csv saved!")

