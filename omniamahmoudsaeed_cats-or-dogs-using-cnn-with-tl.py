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
import zipfile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam




TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
NO_EPOCHS = 20
NUM_CLASSES = 2
SAMPLE_SIZE = 20000
IMG_SIZE = 128

TRAIN_FOLDER = "/kaggle/working/train"
TEST_FOLDER  = "/kaggle/working/test"
PATH_TRAIN = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
PATH_TEST  = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'





with zipfile.ZipFile(PATH_TRAIN, 'r') as zip_ref:
    zip_ref.extractall(TRAIN_FOLDER)
with zipfile.ZipFile(PATH_TEST, 'r') as zip_ref:
    zip_ref.extractall(TEST_FOLDER)

train_inner_folder = os.path.join(TRAIN_FOLDER, 'train')
test_inner_folder = os.path.join(TEST_FOLDER, 'test')

train_image_list = os.listdir(train_inner_folder)[:SAMPLE_SIZE]
test_image_list = os.listdir(test_inner_folder)

print("Found train images:", len(train_image_list))
print("Using SAMPLE_SIZE:", len(train_image_list))
print("Found test images:", len(test_image_list))



def label_pet_image_one_hot_encoder(img_filename):
    pet = img_filename.split('.')[0]
    if pet == 'cat': return [1,0]
    elif pet == 'dog': return [0,1]
    return [1,0]


def process_data(data_image_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in data_image_list:
        path = os.path.join(DATA_FOLDER, img)
        if isTrain:
            label = label_pet_image_one_hot_encoder(img)
        else:
            label = img
        img_array = cv2.imread(path)
        if img_array is None:
            continue
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data_df.append([np.array(img_array), np.array(label)])
    shuffle(data_df)
    return data_df


def plot_image_list_count(data_image_list):
    labels = []
    for img in data_image_list:
        labels.append(img.split('.')[0])   
    plt.figure(figsize=(6,4))
    sns.countplot(x=labels)
    plt.title('Cats and Dogs')
    plt.show()


plot_image_list_count(train_image_list)


train = process_data(train_image_list, train_inner_folder, True)


def show_images(data, isTest=False):
    f, ax = plt.subplots(5,5, figsize=(12,12))
    for i,item in enumerate(data[:25]):
        img_data = item[0]
        img_num = item[1]
        if isTest:
            str_label = "None"
        else:
            label = np.argmax(img_num)
            str_label = "Dog" if label == 1 else "Cat"
        ax[i//5, i%5].imshow(img_data.astype(np.uint8))
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_title("Label: {}".format(str_label))
    plt.tight_layout()
    plt.show()

show_images(train)


test = process_data(test_image_list, test_inner_folder, False)
show_images(test, True)


X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array([i[1] for i in train])
print("X shape:", X.shape, "y shape:", y.shape)


base = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)
for layer in base.layers[:-10]:
    layer.trainable = False
for layer in base.layers[-10:]:
    layer.trainable = True


model = Sequential()
model.add(base)
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

datagen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = datagen.flow(X, y, batch_size=BATCH_SIZE, subset='training')
val_generator = datagen.flow(X, y, batch_size=BATCH_SIZE, subset='validation')


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

train_model = model.fit(
    train_generator,
    epochs=NO_EPOCHS,
    validation_data=val_generator,
    verbose=1,
    callbacks=[early_stop]
)



def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['accuracy'] if 'accuracy' in hist else hist.get('acc', [])
    val_acc = hist['val_accuracy'] if 'val_accuracy' in hist else hist.get('val_acc', [])
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(len(acc))
    f, ax = plt.subplots(1,2, figsize=(14,6))
    ax[0].plot(epochs, acc, 'g', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[1].plot(epochs, loss, 'g', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.show()


plot_accuracy_and_loss(train_model)


X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
test_filenames = [str(i[1]) for i in test]


y_pred_proba = model.predict(X_test)
predicted_classes = np.argmax(y_pred_proba, axis=1)

test_ids = [int(os.path.splitext(f)[0]) for f in test_filenames]

submission = pd.DataFrame({
    'id': test_ids,
    'label': predicted_classes
})

submission.to_csv('submission.csv', index=False)


submission.head()


submission.shape




