# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
        # print(os.path.join(dirname, filename))
print("/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv\n"
      "/kaggle/input/siim-isic-melanoma-classification/train.csv\n"
      "/kaggle/input/siim-isic-melanoma-classification/test.csv\n"
      "/kaggle/input/siim-isic-melanoma-classification/jpeg/test/ISIC_2417927.jpg\n"
      "/kaggle/input/siim-isic-melanoma-classification/test/ISIC_7770700.dcm\n"
      "/kaggle/input/siim-isic-melanoma-classification/train/ISIC_9691303.dcm")


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pathlib,shutil
from PIL import Image


import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

import logging
tf.get_logger().setLevel(logging.ERROR)


data_dirc=pathlib.Path('/kaggle/input/siim-isic-melanoma-classification/jpeg')
test_dirc=data_dirc / 'test'
train_dirc=data_dirc / 'train'

train_images_count=len(list(train_dirc.glob("*.jpg")))
test_images_count=len(list(test_dirc.glob("*.jpg")))
print(train_images_count)
print(test_images_count)


print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))


strategy = tf.distribute.MirroredStrategy()
print("✅ Using GPU:", tf.config.experimental.list_physical_devices('GPU'))


batch_size=32
img_h=150
img_w=150


img=Image.open('/kaggle/input/siim-isic-melanoma-classification/jpeg/test/ISIC_2417927.jpg')
print(f"og image size{img.size}")


csv_path='/kaggle/input/siim-isic-melanoma-classification/train.csv'
df=pd.read_csv(csv_path)
df['image_path'] = df['image_name'].apply(lambda x: os.path.join(train_dirc, f"{x}.jpg"))
df['target'] = df['target'].astype(str)

csv_path_test='/kaggle/input/siim-isic-melanoma-classification/test.csv'
test_df=pd.read_csv(csv_path_test)
test_df['image_path']=test_df['image_name'].apply(lambda x: os.path.join(test_dirc, f"{x}.jpg"))



train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=None,
    x_col='image_path',
    y_col='target',
    target_size=(img_h, img_w),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)




test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    target_size=(img_h, img_w),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=None,
    x_col='image_path',
    y_col='target',
    target_size=(img_h, img_w),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)




with strategy.scope():
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(150,150,3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
    
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
    
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
    
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),  # Slightly reduced dropout
    
        layers.Dense(64, activation='relu'),  # Reduced from 128 to 64
        layers.Dropout(0.4),
        
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

model.summary()


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    batch_size=batch_size
)
!nvidia-smi


from IPython.display import FileLink

model.save("/kaggle/working/model.h5")




from tensorflow.keras.models import load_model

model_path = "/kaggle/input/melanoma-detection/keras/default/1/skin_cancer_model.h5"
model = load_model(model_path)
print("✅ Model Loaded Successfully!")


print(test_generator)



print(f"Total samples: {test_generator.n}")
print(f"Class mode: {test_generator.class_mode}")



predictions = model.predict(test_generator)
print(predictions[:5])


filenames = test_generator.filenames  
filenames = [fname.split("/")[-1].replace(".jpg", "") for fname in test_generator.filenames]

test_predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

submission_df = pd.DataFrame({
    "id": filenames,  
    "label": test_predictions.flatten()
})

submission_df.to_csv("submission.csv", index=False)
submission_df.head()

