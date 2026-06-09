import numpy as np 
import pandas as pd 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import zipfile
train_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
test_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'
file_path = '/kaggle/working/'

with zipfile.ZipFile(train_path, 'r') as zipp:
    zipp.extractall(file_path)


with zipfile.ZipFile(test_path, 'r') as zipp:
    zipp.extractall(file_path)


import os
image_dir = '/kaggle/working/train'
filenames = os.listdir(image_dir)
labels = [x.split('.')[0] for x in filenames]
data = pd.DataFrame({'filename': filenames, 'label': labels})

data.head()


import matplotlib.pyplot as plt
import matplotlib.image as imread
plt.figure(figsize = (20, 20))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    filename = 'train/dog.' + str(i) + '.jpg'
    image = plt.imread(filename)
    plt.imshow(image)
    plt.title('dog')
    plt.axis('off')


plt.figure(figsize = (20, 20))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    filename = 'train/cat.' + str(i) + '.jpg'
    image = plt.imread(filename)
    plt.imshow(image)
    plt.title('cat')
    plt.axis('off')


from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(data, stratify = data.label, random_state = 42)


image_size = 128 #define size of new images
bat_size = 32
channel = 3


from tensorflow.keras.preprocessing.image import ImageDataGenerator #Procedure to form new image based on training set
train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            channel_shift_range=0.2,
            fill_mode='nearest',
            horizontal_flip=True,
            rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(X_train,
                                                    directory = 'train/',
                                                    x_col= 'filename',
                                                    y_col= 'label',
                                                    batch_size = bat_size,
                                                    target_size = (image_size,image_size)
                                                   )
val_generator = test_datagen.flow_from_dataframe(X_val, 
                                                 directory = 'train/',
                                                 x_col= 'filename',
                                                 y_col= 'label',
                                                 batch_size = bat_size,
                                                 target_size = (image_size,image_size),
                                                 shuffle=False
                                                )


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dense
model = Sequential()

# Input Layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape = (image_size,image_size,channel))) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Bloack 1 
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
# Block 2
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
# Block 3
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Fully Connected layers 
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Output layer
model.add(Dense(2,activation='softmax'))

model.summary()


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
learning_rate_reduction = ReduceLROnPlateau(
    monitor = 'val_accuracy',
    patience = 2,
    factor = 0.5,
    min_lr = 0.0001,
    verbose = 1
)
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = 'True', verbose = 0)


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


cat_dog = model.fit(train_generator, validation_data = val_generator, 
                    callbacks = [early_stopping, learning_rate_reduction],
                    epochs = 5)


cat_dog.history


import seaborn as sns
error = pd.DataFrame(cat_dog.history)

plt.figure(figsize=(18,5),dpi=200)
sns.set_style('darkgrid')

plt.subplot(121)
plt.title('Cross Entropy Loss',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.plot(error['loss'])
plt.plot(error['val_loss'])

plt.subplot(122)
plt.title('Classification Accuracy',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.plot(error['accuracy'])
plt.plot(error['val_accuracy'])

plt.show()


# Image generator (only rescaling, no augmentation for test)
test_datagen = ImageDataGenerator(rescale=1./255)
test_dir = "/kaggle/working/test"
filenames = os.listdir(test_dir)

# Flow from directory (no labels since it's test data)
test_data = pd.DataFrame({"filename":filenames})
test_data['label'] = 'unknown'

test_generator = test_datagen.flow_from_dataframe(
    test_data,
    directory = "test/",         # parent dir
    x_col = "filename",
    y_col = "label",       # only the test folder
    target_size=(image_size, image_size),      # use same size as training
    batch_size=bat_size,
    class_mode=None,             # no labels
    shuffle=False                # IMPORTANT to keep order
)



test_predict = model.predict(test_generator,verbose = 0)
test_predict_argmax = np.argmax(test_predict, axis=1)
y_test_pred = test_predict_argmax

test_data['label'] = y_test_pred

# mapping
label_mapping = {0: 'cat', 1: 'dog'}
test_data['label'] = test_data['label'].map(label_mapping)



test_data.head()


# csv file output for submission


sub = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv',index_col='id')

sub['label'] = y_test_pred

sub.to_csv('submission.csv',index=True)

