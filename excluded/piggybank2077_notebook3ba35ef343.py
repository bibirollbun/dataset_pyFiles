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


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd

df = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')
print(df.head())


import zipfile


with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    
    file_list = zip_ref.namelist()
    print(file_list[:10])  


from PIL import Image
import zipfile
import io
import matplotlib.pyplot as plt

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    file_list = zip_ref.namelist()
    
    sample_file = file_list[2]
    
    with zip_ref.open(sample_file) as file:
        img = Image.open(file)
        plt.imshow(img)
        plt.title(sample_file)
        plt.axis('off')
        plt.show()


import zipfile
import os
import shutil

# 解凍先ディレクトリ
base_dir = '/kaggle/working/dogs-vs-cats'
train_zip = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'

# 解凍
with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall(base_dir)

# 画像を cat / dog フォルダに分ける
train_dir = os.path.join(base_dir, 'train')
cat_dir = os.path.join(base_dir, 'cats')
dog_dir = os.path.join(base_dir, 'dogs')
os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

for fname in os.listdir(train_dir):
    if fname.startswith('cat'):
        shutil.move(os.path.join(train_dir, fname), os.path.join(cat_dir, fname))
    elif fname.startswith('dog'):
        shutil.move(os.path.join(train_dir, fname), os.path.join(dog_dir, fname))

# train_dir は不要になったので削除
os.rmdir(train_dir)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# データジェネレータ
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 検証用に20%使う
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=15
)

# トレーニングとバリデーション用ジェネレータ
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False 

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') 
])


model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)

