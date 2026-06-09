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


import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "/kaggle/input/cassava-leaf-disease-classification"
train_df = pd.read_csv(os.path.join(DATASET_PATH, "train.csv"))
train_df['label'] = train_df['label'].astype(str)

# Chia train/val
train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['label'], random_state=42)
print(train_df.shape, val_df.shape)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory=os.path.join(DATASET_PATH, "train_images"),
    x_col="image_id",
    y_col="label",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    directory=os.path.join(DATASET_PATH, "train_images"),
    x_col="image_id",
    y_col="label",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE
)


from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])


from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=1e-2),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_dir = os.path.join(DATASET_PATH, "test_images")
test_images = sorted(os.listdir(test_dir))

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    pd.DataFrame({"image_id": test_images}),
    directory=test_dir,
    x_col="image_id",
    y_col=None,
    target_size=IMAGE_SIZE,
    class_mode=None,
    shuffle=False,
    batch_size=BATCH_SIZE
)

preds = model.predict(test_gen)
pred_labels = np.argmax(preds, axis=1)

submission = pd.DataFrame({
    "image_id": test_images,
    "label": pred_labels
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv saved!")

