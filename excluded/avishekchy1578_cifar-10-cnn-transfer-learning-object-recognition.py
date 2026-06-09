# from google.colab import drive
# drive.mount("/content/drive")


!pip install py7zr


# Fetching Dataset
import pandas as pd
import py7zr

# Reading Image
import os
import cv2 as cv
import matplotlib.pyplot as plt

# Scaling Features
from sklearn.preprocessing import MinMaxScaler

# Binarizing Output
from sklearn.preprocessing import LabelBinarizer

# Spliting Train & Test Set
from sklearn.model_selection import train_test_split

# Image Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Creating and Training Model
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50, ResNet152

# Evaluating Model
from sklearn.metrics import (classification_report, multilabel_confusion_matrix, confusion_matrix)
import seaborn as sb
import matplotlib.pyplot as plt

# Making Predictions
import numpy as np


dataset_archive_dir = '/kaggle/input/cifar-10/'
dataset_train = pd.read_csv(dataset_archive_dir + "trainLabels.csv")
dataset_test = pd.read_csv(dataset_archive_dir + "sampleSubmission.csv")


train_archive_dir = dataset_archive_dir + 'train.7z'
test_archive_dir = dataset_archive_dir + 'test.7z'

dataset_output_dir = '/kaggle/working/cifar/'
train_dataset_dir = dataset_output_dir + 'train/'
test_dataset_dir = dataset_output_dir + 'test/'

# os.makedirs(train_dataset_dir, exist_ok=True)
# os.makedirs(test_dataset_dir, exist_ok=True)
os.makedirs(dataset_output_dir, exist_ok=True)

with py7zr.SevenZipFile(train_archive_dir, mode='r') as archive:
    archive.extractall(path=dataset_output_dir)

print(f"Successfully extracted '{train_archive_dir}' to '{dataset_output_dir}'")

with py7zr.SevenZipFile(test_archive_dir, mode='r') as archive:
    archive.extractall(path=dataset_output_dir)

print(f"Successfully extracted '{test_archive_dir}' to '{dataset_output_dir}'")


dataset_train


dataset_test


X_train = dataset_train.loc[:, 'id'].values
X_test = dataset_test.loc[:, 'id'].values
y_train = dataset_train.loc[:, 'label'].values
y_test = dataset_test.loc[:, 'label'].values


X_train


X_test


y_train


y_test


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


IMG_SIZE = 32 # Same Height and Width is being used
x_train_data = np.empty((len(X_train), IMG_SIZE, IMG_SIZE, 3), dtype = np.int64)
x_test_data = np.empty((len(X_test), IMG_SIZE, IMG_SIZE, 3), dtype = np.int64)


print(x_train_data.shape)
print(x_test_data.shape)


for i, img_path in enumerate(X_train):
    img = cv.imread(train_dataset_dir + str(img_path) + '.png')
    img_rs = cv.resize(img, dsize = (IMG_SIZE, IMG_SIZE))
    # img_gray = cv.cvtColor(img_rs, cv.COLOR_BGR2GRAY)
    # ret, candidate_threshold = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    x_train_data[i] = img_rs


for i, img_path in enumerate(X_test):
    img = cv.imread(test_dataset_dir + str(img_path) + '.png')
    img_rs = cv.resize(img, dsize = (IMG_SIZE, IMG_SIZE))
    # img_gray = cv.cvtColor(img_rs, cv.COLOR_BGR2GRAY)
    # ret, candidate_threshold = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    x_test_data[i] = img_rs


plt.imshow(x_train_data[1])


plt.imshow(x_test_data[1])


len(x_train_data[0][0])


len(x_test_data[0][0])


X_train = np.reshape(x_train_data, (X_train.shape[0], IMG_SIZE * IMG_SIZE, 3))
X_test = np.reshape(x_test_data, (X_test.shape[0], IMG_SIZE * IMG_SIZE, 3))


print(X_train.shape)
print(X_test.shape)


# sc = MinMaxScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


X_train


X_test


lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)


y_train


y_test


print(y_train.shape)
print(y_test.shape)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = 0.9, random_state = 45, stratify = y_train)


print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


plt.imshow(X_train[45])


# Define ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range = 10,
    zoom_range = 0.1,
    shear_range = 0.5,
    cval = 0.0,
    fill_mode = 'constant')


# Generate augmented data
batch_size = 32
train_generator = datagen.flow(X_train, y_train, batch_size = batch_size, shuffle = False)


len(train_generator)


# Load the pre-trained model
resnet152 = ResNet152(weights='imagenet', input_shape=(IMG_SIZE*5, IMG_SIZE*5, 3), include_top=False)

# Freeze the weights of the pre-trained layers
for layer in resnet152.layers:
    layer.trainable = False

# Add your custom layers to the top of the pre-trained model
model = models.Sequential(name = 'CIFAR-10-Object-Recognition')
model.add(layers.Input((IMG_SIZE, IMG_SIZE, 3)))
model.add(layers.UpSampling2D(size=(5, 5)))
model.add(resnet152)
model.add(layers.Flatten())
model.add(layers.Dense(units=1024, activation='relu'))
model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))


model.summary()


epochs = 100
acc_callback = EarlyStopping(monitor = 'val_accuracy', verbose = 1, patience = 10, mode = 'max', restore_best_weights = True, start_from_epoch = 15)
loss_callback = EarlyStopping(monitor = 'val_loss', verbose = 1, patience = 10, mode = 'min', restore_best_weights = True, start_from_epoch = 15)


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


history  = model.fit(train_generator, validation_data = (X_val, y_val), epochs = epochs, callbacks = [acc_callback, loss_callback])


history_test = model.evaluate(X_val, y_val)


y_predicted = model.predict(X_test)


y_predicted


# y_test_labels = lb.inverse_transform(y_test)
y_predicted_labels = lb.inverse_transform(y_predicted)


y_predicted_labels


dataset_test['label'] = y_predicted_labels
dataset_test


!rm -rf /kaggle/working/*


dataset_test.to_csv('submission.csv', index = False)

