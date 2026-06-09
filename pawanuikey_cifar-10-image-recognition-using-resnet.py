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


pip install py7zr


import py7zr


archive =py7zr.SevenZipFile("/kaggle/input/cifar-10/train.7z",mode='r')
archive.extractall()
archive.close()


import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split


filenames=os.listdir("/kaggle/working/train")


type(filenames)


len(filenames)


print(filenames[0:5])
print(filenames[-5:])


labeled_df =pd.read_csv("/kaggle/input/cifar-10/trainLabels.csv")


labeled_df.shape


labeled_df.head()


labeled_df[labeled_df['id']==7796]


labeled_df.tail(10)


labeled_df['label'].value_counts()


labeled_dictionary = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

labels = [labeled_dictionary[i] for i in labeled_df['label']]


labels[0:5]
labels[-5:]


import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('/kaggle/working/train/7756.png')

# Check if the image is loaded correctly
if img is None:
    print("Image not found or the path is incorrect.")
else:
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axis
    plt.show()



import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('/kaggle/working/train/7156.png')

# Check if the image is loaded correctly
if img is None:
    print("Image not found or the path is incorrect.")
else:
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axis
    plt.show()



id_list=list(labeled_df['id'])


id_list[-5:]


train_data_folder='/kaggle/working/train/'
data=[]
for id in id_list:
    image =Image.open(train_data_folder+str(id)+'.png')
    image=np.array(image)
    data.append(image)


len(data)


type(data[0])


data[0].shape





data[0]


X =np.array(data)
Y=np.array(labels)


print(X.shape)
print(Y.shape)


X_train,X_test,y_train,y_test =train_test_split(X,Y,test_size=0.2,random_state=2)


X_train.shape


X_test.shape


x_train_scaled =X_train/255
x_test_scaled=X_test/255


x_train_scaled


import tensorflow as tf
from tensorflow import keras


EPOCHS=40


from tensorflow.keras import Sequential, models, layers
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import optimizers
     

convolutional_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3))
convolutional_base.summary()


model2 = models.Sequential()
model2.add(layers.UpSampling2D((2,2)))
model2.add(layers.UpSampling2D((2,2)))
model2.add(layers.UpSampling2D((2,2)))
model2.add(convolutional_base)
model2.add(layers.Flatten())
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(128, activation='relu'))
model2.add(layers.Dropout(0.5))
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dropout(0.5))
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(10, activation='softmax'))


from tensorflow.keras import optimizers

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model2.compile(
    optimizer=optimizer, 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)



from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    verbose=1, 
    min_lr=1e-6
)




from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model2.fit(
    x_train_scaled, y_train,
    validation_split=0.1,
    epochs=40,
    batch_size=32,

)



test_loss, test_acc = model2.evaluate(x_test_scaled, y_test)
print("Test Accuracy:", test_acc)


model2.save('/kaggle/working/Resnet_model.h5')


history.history.keys()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


EPOCHS=40


plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


y_pred = model2.predict(x_test_scaled)
predicted_categories = tf.argmax(y_pred, axis=1)


Y_true = y_test 


predicted_categories


from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(Y_true,predicted_categories)


class_name = list(labeled_dictionary.keys())


print(classification_report(Y_true,predicted_categories,target_names=class_name))


import seaborn as sns
plt.figure(figsize=(12, 12))
sns.heatmap(cm,annot=True,annot_kws={"size": 10})

plt.xlabel('Predicted Class',fontsize = 20)
plt.ylabel('Actual Class',fontsize = 20)
plt.title('Image Recognition Confusion Matrix',fontsize = 25)
plt.show()




