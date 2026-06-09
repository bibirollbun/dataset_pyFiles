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
import shutil
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam


# データの解凍
shutil.unpack_archive('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', '/kaggle/working')
shutil.unpack_archive('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', '/kaggle/working')
shutil.copyfile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv','/kaggle/working/sample_submission.csv')


# 犬猫のフォルダを作って振り分け
dog_dir = '/kaggle/working/train/dog'
cat_dir = '/kaggle/working/train/cat'
os.makedirs(dog_dir, exist_ok=True)
os.makedirs(cat_dir, exist_ok=True)

files = glob.glob('/kaggle/working/train/*.jpg')
for file in files:
    file_name = os.path.basename(file)
    if 'cat' in file_name:
        shutil.move(file, os.path.join(cat_dir, file_name))
    else:
        shutil.move(file, os.path.join(dog_dir, file_name))


# 設定
img_width, img_height = 224, 224
batch_size = 32
epochs = 15
train_dir = '/kaggle/working/train'


# データ拡張（Augmentation）
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)


# モデル構築（VGG16ベース）
vgg_base = VGG16(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
vgg_base.trainable = False  # 全層凍結

model = Sequential([
    vgg_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 2クラス分類
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


# コールバック設定
checkpoint_cb = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
early_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)


# 学習
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint_cb, early_cb]
)


# グラフ表示
plt.plot(history.history['loss'], label='loss', marker='o')
plt.plot(history.history['val_loss'], label='val_loss', marker='x')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(); plt.show()

plt.plot(history.history['accuracy'], label='accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='val_accuracy', marker='x')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(); plt.show()


# テスト画像から予測（ファイル名にID含む前提）
test_dir = '/kaggle/working/test'
test_files = sorted(glob.glob(os.path.join(test_dir, '*.jpg')))

model = load_model('best_model.h5')

results = []
for file in test_files:
    img = load_img(file, target_size=(img_height, img_width))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    pred = model.predict(x)[0][0]
    image_id = int(os.path.basename(file).split('.')[0])
    results.append([image_id, pred])


# CSVで保存
results_df = pd.DataFrame(results, columns=['id', 'label'])
results_df.sort_values('id', inplace=True)
results_df.to_csv('/kaggle/working/submission.csv', index=False)


print("最終訓練精度:", history.history['accuracy'][-1])
print("最終検証精度:", history.history['val_accuracy'][-1])

