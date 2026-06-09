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


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Conv2D, Dense, MaxPooling2D, BatchNormalization
from keras.models import Sequential


df = pd.read_csv('train.csv')


df.head()


(x_train, y_train) = ( df['pixels'], df['emotion'])


y_train.value_counts()


pip install --upgrade scikit-learn imbalanced-learn


from imblearn.over_sampling import SMOTE, ADASYN
sm = SMOTE(random_state=2)
ad = ADASYN(random_state=2)
# Конвертуємо рядки з числами у 2D масив numpy
x_train_array = np.array([list(map(int, row.split())) for row in x_train], dtype=np.uint8)

x_train_smote, y_train_smote = ad.fit_resample(x_train_array, y_train)


import numpy as np
import matplotlib.pyplot as plt

x_train_image = np.vstack(x_train_smote).reshape(-1, 48, 48)

print(x_train_image.shape)

plt.imshow(x_train_image[0], cmap='gray')
plt.show()



x_train_image.shape


x_train_image = np.expand_dims(x_train_image, axis=-1)


x_train_image.shape


y_train_smote.value_counts()


y_train_smote.shape


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train_image, y_train_smote, test_size=0.2, random_state=42)


x_train.shape


training_gen = ImageDataGenerator(
    rescale = 1./255

)


train_generator = training_gen.flow(
    x_train,  
    y_train,
    batch_size=64,
    shuffle=True
)


val_gen = ImageDataGenerator(
    rescale = 1./255
)


val_gen = training_gen.flow(
    x_val,
    y_val,
    batch_size=64,
    shuffle=True
)


from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

def model(filters, kernel_size, hidden_layers, hidden_neurons, learning_rate):
    model = Sequential()

    model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), input_shape=(48, 48, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    for _ in range(hidden_layers):
        model.add(Dense(hidden_neurons, activation='relu'))
        model.add(BatchNormalization())

    model.add(Dense(7, activation='softmax'))  # 7 emotions
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model



pip install scikit-learn==1.3.0


from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

model = model(16,3, 3, 216, 0.01)
models = KerasClassifier(model=model, epochs=10, batch_size=32, verbose=1)
results = cross_val_score(models, x_train, y_train, cv=5, scoring='accuracy')

print("Cross-validation results:", results)



model = model(16,3, 3, 216, 0.01)


from tensorflow.keras.callbacks import EarlyStopping

model.fit(x=train_generator, verbose=2, validation_data=val_gen, epochs=30, callbacks=EarlyStopping(patience=5))

