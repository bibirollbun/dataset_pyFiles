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


train=pd.read_csv("/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/train.csv")
test=pd.read_csv("/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/test.csv")


train.head()


test.head()


import seaborn as sns


sns.histplot(train,x='age')


train.shape


test.shape


import pandas as pd
import cv2
import os

images_dir = '/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/'
csv_path = '/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/train.csv'

df = pd.read_csv(csv_path)

data = []
labels = []


for index, row in df.iterrows():
    
    img_path = os.path.join(images_dir, str(row['filename']))
   
    image = cv2.imread(img_path)
    image = cv2.resize(image, (128, 128))
    
    data.append(image)
    
    labels.append(row['age'])

print(f'Total images: {len(data)}')
print(f'Total labels: {len(labels)}')


data_arr=np.array(data)
label_arr=np.array(labels)


y=labels


x=data_arr/255


x


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=13)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

l2_reg = 0.001  # biraz daha düşük L2 ile daha esnek öğrenme

model = Sequential([
    layers.Conv2D(32, kernel_size=3, input_shape=(128, 128, 3), activation='relu', padding='same',
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

    layers.Dense(1)
])

optimizer = Adam(learning_rate=0.0002)  # biraz daha yüksek öğrenme oranı
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])



y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)
y_train=y_train.values
y_test=y_test.values


model.summary()


history = model.fit(
    x_train, y_train,
    epochs=50,  # artık 350’ye gerek yok
    batch_size=64,  # 300 çok büyük, 64 veya 128 daha verimli öğrenir
    validation_data=(x_test,y_test)
)


import matplotlib.pyplot as plt

acc = history.history['mae']
val_acc = history.history['val_mae']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training mae')
plt.plot(epochs, val_acc, 'b', label='Validation mae')
plt.title('Training and validation mae')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()


plt.show()


import pandas as pd
import cv2
import os

images_dir = '/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/'
csv_path = '/kaggle/input/applications-of-deep-learning-wustl-spring-2024/faces-age/test.csv'

test_df = pd.read_csv(csv_path)

test_data = []
labels = []


for index, row in test_df.iterrows():
    
    img_path = os.path.join(images_dir, str(row['filename']))
   
    image = cv2.imread(img_path)
    image = cv2.resize(image, (128, 128))
    
    test_data.append(image)

print(f'Total images: {len(test_data)}')


test_array=np.array(test_data)


x_test=test_array/255


x_test.shape


pred=model.predict(x_test)


pred


pred=np.round(pred.flatten(),1)


pred


submission= pd.DataFrame({
    "id": test_df["id"],
    "age": np.round(pred,1)
})


submission.head()


submission.to_csv("submission.csv",index=False,float_format="%.1f")




