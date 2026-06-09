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


import warnings
warnings.filterwarnings("ignore")


train=pd.read_csv('/kaggle/input/image-classifications/train.csv')
test=pd.read_csv('/kaggle/input/image-classifications/test.csv')
submiision=pd.read_csv('/kaggle/input/image-classifications/example_submission.csv')


train.head()


train.shape,test.shape


import numpy as np
import matplotlib.pyplot as plt

image_pixels = test.iloc[0, :-1].values
image_size = 256
image = np.reshape(image_pixels, (image_size, image_size))

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

image_pixels = test.iloc[20, :-1].values
image_size = 256
image = np.reshape(image_pixels, (image_size, image_size))

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()



test=test.drop(['ID'],axis=1)
import numpy as np
test=np.array(test)


test_data = test.reshape(200, 256, 256, 1)


y_train=train.Class
train=train.drop(['Class'],axis=1)


train=train.drop(['ID'],axis=1)
train.shape


train=np.array(train)

train_data = train.reshape(1020, 256, 256, 1)
train_data.shape,y_train.shape


train_data=train_data.astype('float32')/255
test_data=test_data.astype('float32')/255


import tensorflow as tf

train_data = tf.image.resize(train_data, [128, 128]).numpy()
test_data = tf.image.resize(test_data, [128, 128]).numpy()


train_data.shape,y_train.shape


from tensorflow.keras.preprocessing.image import ImageDataGenerator #veriyi arttır

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
    fill_mode='nearest')

augmented_data = datagen.flow(train_data, y_train, batch_size=750)

batch_x, batch_y = next(augmented_data)


train_data = np.concatenate((train_data, batch_x))
y_train = np.concatenate((y_train, batch_y))
train_data.shape,y_train.shape


train_data.shape


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    train_data,
    y_train,
    test_size=0.2,
    random_state=42,
)


y_train=pd.DataFrame(y_train)
y_train.value_counts()


x_train.shape, x_val.shape, y_train.shape, y_val.shape


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

l2_reg = 0.001  # biraz daha düşük L2 ile daha esnek öğrenme

model = Sequential([
    layers.Conv2D(32, kernel_size=3, input_shape=(128, 128, 1), activation='relu', padding='same',
                  kernel_regularizer=l2(l2_reg)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(64, kernel_size=3, activation='relu', padding='same',
                  kernel_regularizer=l2(l2_reg)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, kernel_size=3, activation='relu', padding='same',
                  kernel_regularizer=l2(l2_reg)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_reg)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_reg))
])

optimizer = Adam(learning_rate=0.0002)  # biraz daha yüksek öğrenme oranı
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])



model.summary()


print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)



from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    epochs=100,  # artık 350’ye gerek yok
    validation_data=(x_val, y_val),
    batch_size=64,  # 300 çok büyük, 64 veya 128 daha verimli öğrenir
    callbacks=[early_stop]
)



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()



test_data.shape


pred=model.predict(test_data)


pred


y_pred = (pred > 0.5).astype(int)


submiision.head()


submiision.Class=y_pred


submiision.shape


submiision.to_csv('submission.csv', index=False)




