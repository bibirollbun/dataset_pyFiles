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


from tensorflow import keras
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import shutil
import os
import glob


shutil.unpack_archive('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', '/kaggle/working')
shutil.unpack_archive('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', '/kaggle/working')


shutil.copyfile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv','/kaggle/working/sample_submission.csv')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
files = glob.glob('/kaggle/working/*')
for file in files:
    print(file)


img_width, img_height = 150, 150
train_dir = '/kaggle/working/train'
test_dir = '/kaggle/working/test'


epoch = 20


classes = ['dog','cat']
nb_classes = len(classes)


vgg_model = VGG16(include_top = False, weights = 'imagenet', input_shape = (img_height, img_width, 3))


model = Sequential()
model.add(vgg_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))


for layer in vgg_model.layers[:15]:
 layer.trainable = False


model.summary()


model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])


dog_dir = '/kaggle/working/train/dog'
cat_dir = '/kaggle/working/train/cat'

os.makedirs(dog_dir, exist_ok = True)
os.makedirs(cat_dir, exist_ok = True)


files = glob.glob('/kaggle/working/train/*.jpg')
for file in files:
    file_name = os.path.basename(file)
    if 'cat' in file:
        shutil.move(file,'/kaggle/working/train/cat/' + file_name)
    else:
        shutil.move(file,'/kaggle/working/train/dog/' + file_name)


train_datagen = ImageDataGenerator(rescale = 1.0 / 255, rotation_range = 90, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True,
                vertical_flip = True, fill_mode = 'reflect', validation_split = 0.2)


test_datagen = ImageDataGenerator(rescale = 1.0 / 255)


train_generator = train_datagen.flow_from_directory(train_dir, target_size = (img_height, img_width), classes = classes, batch_size = 32, class_mode = 'categorical', subset = "training")


test_generator = train_datagen.flow_from_directory(train_dir, target_size = (img_height, img_width), classes = classes, batch_size = 32, class_mode = 'categorical', subset = "validation")


mc_cb = ModelCheckpoint(filepath = 'finetuning.h5', monitor = 'vol_loss', verbose = 1, save_best_only = True)


history = model.fit(train_generator, epochs=epoch, validation_data = test_generator, callbacks = [mc_cb])

model.save('finetuning.h5')


plt.plot(range(len(history.history['loss'])), history.history['loss'], marker='o', color = 'black', label='loss')
plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], marker='v', linestyle='--', color='red', label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()


plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'], marker='o', color = 'black', label='acc')
plt.plot(range(len(history.history['val_accuracy'])), history.history['val_accuracy'], marker='v', linestyle='--', color = 'red', label='val_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()


import sys
import glob
import numpy as np
from keras.preprocessing import image
from keras.models import load_model


img_height, img_width = 150, 150


classes = ['dog','cat']
nb_classes = len(classes)


model = load_model('finetuning.h5')


filename = glob.glob('/content/drive/MyDrive/data/test/*/*.jpg')


for i in range(len(filename)):
 print('input:', filename[i])

# 入力画像のロード、4次元テンソルへ変換 
 img = image.load_img(filename[i], target_size = (img_height, img_width))
 x = image.img_to_array(img)
 x = np.expand_dims(x, axis = 0)

# 入力データへの正規化
 x = x / 255.0

# 分類クラスを予測
 pred = model.predict(x)[0]

#予測結果を予測確率が上位10件分、クラス名と予測確率を出力
 top_n = 10
 top_indices = pred.argsort()[-top_n:][::-1]
 result = [(classes[i], pred[i]) for i in top_indices]
 for x in result:
  print(x)

