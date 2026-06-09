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


!unzip -oq /kaggle/input/leaf-classification/train.csv.zip -d .
!unzip -oq /kaggle/input/leaf-classification/test.csv.zip -d .
!unzip -oq /kaggle/input/leaf-classification/sample_submission.csv.zip -d .
!unzip -oq /kaggle/input/leaf-classification/images.zip -d images/


train = pd.read_csv('/kaggle/working/train.csv')
test = pd.read_csv('/kaggle/working/test.csv')

print(train.shape, test.shape)
train.head()


import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Tạo cột đường dẫn đến ảnh
train["file_path"] = "/kaggle/working/images/images/" + train["id"].astype(str) + ".jpg"
train_shuffled = train.sample(frac=1, random_state=42).reset_index(drop=True)

# Tạo generator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_dataframe(
    dataframe=train_shuffled,
    x_col="file_path",
    y_col="species",
    target_size=(128, 128),
    batch_size=64,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_dataframe(
    dataframe=train_shuffled,
    x_col="file_path",
    y_col="species",
    target_size=(128, 128),
    batch_size=64,
    class_mode="categorical",
    subset="validation"
)

class_indices = train_gen.class_indices
labels_map = dict((v, k) for k, v in class_indices.items())


from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    
    layers.Conv2D(32, (3,3), padding='same'), 
    layers.ReLU(),
    
    layers.Conv2D(64, (3,3), padding='same'),
    layers.ReLU(),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128, (3,3), padding='same'),
    layers.ReLU(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(256, (3,3), padding='same'),
    layers.ReLU(),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(512, (3,3), padding='same'),
    layers.ReLU(),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.000001)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    callbacks=[reduce_lr]
)


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()


test["file_path"] = "/kaggle/working/images/images/" + test["id"].astype(str) + ".jpg"
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test,
    x_col="file_path",
    y_col=None,
    target_size=(128, 128),
    class_mode=None,
    shuffle=False
)


preds = model.predict(test_gen, verbose=1)

predicted_class_indices = np.argmax(preds, axis=1)

predicted_labels = [labels_map[i] for i in predicted_class_indices]


class_names = list(train_gen.class_indices.keys())

submission = pd.DataFrame(preds, columns=class_names)

submission.insert(0, "id", test["id"])
submission.to_csv("submission.csv", index=False)

