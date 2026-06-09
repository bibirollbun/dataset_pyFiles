from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import cv2
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# libraries for EDA and Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Importing Tensorflow Libraries

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


os.getcwd()


os.listdir(os.getcwd())


# Get the directories and data
os.listdir('../input/')


os.listdir('../input/histopathologic-cancer-detection')


dataset = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
train_path = '/kaggle/input/train/'
test_path = '/kaggle/input/test/'
# quick look at the label stats - there are two: 0 (not cancerous) and 1 (cancerous)
print(dataset['label'].value_counts())


'''
Get dataset columns' names
There are two: the images' ids and the labels (0 for non-canceerous images, 1 otherwise)
'''
dataset.columns


print(dataset.shape)


dataset.head()


dataset.tail()


# Take a look at the labels in the training data; there should be all unique id values
dataset['label'].unique()


# Get all value counts
dataset.value_counts()


# Check if there are any null values in the dataset; there should be none
dataset.isnull().sum()


# Get percentages of which labels are cancerous or not
zeros = len(dataset[dataset['label'] == 0])
ones = len(dataset[dataset['label'] == 1])
total = dataset.shape[0]

print("Value is 0:", zeros, 'รท', total, '=', zeros / total)
print("Value is 1:", ones, 'รท', total, '=', ones / total)


# Plot the distributions
# Credit can be attributed to the following notebook: 
# https://www.kaggle.com/code/alexandermaitken/cnn-cancer-detection?scriptVersionId=240762737&cellId=10
# 
# These should show the values from the previous code block

plt.bar('0', len(dataset[dataset['label'] == 0]))
plt.bar('1', len(dataset[dataset['label'] == 1]))
plt.xlabel("Label Value")
plt.ylabel("Number of Occurrences")
plt.title("Training Data Labels Count")
plt.legend(['Not Cancerous', 'Cancerous'])
plt.show()


dataset.describe()


# Credit can be attributed to the following cell in this notebook to show sample images from the test and train folders
# https://www.kaggle.com/code/alexandermaitken/cnn-cancer-detection?scriptVersionId=240762737&cellId=12

# Will need to alter base_path for training data since it is being used online via Kaggle
# Also shows the image of any one shape - or the number of pixels on any one image
base_path='/kaggle/input/histopathologic-cancer-detection/train'
def show_samples(df, label, n=10):
    samples = df[df['label'] == label].sample(n)
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    for img_id, ax in zip(samples['id'], axes):
        # img = get_image(img_id)
        path = os.path.join(base_path, f"{img_id}.tif")
        img = cv2.imread(path)
        print(img.shape)

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    plt.suptitle(f"Label: {label}")
    plt.show()

# Show example images of non-cancerous images along with their respective shapes
# in pixels and colors type
show_samples(dataset, label=0)
plt.show()


# Show example images of cancerous images along with their respective shapes
# in pixels and colors type
show_samples(dataset, label=1)
plt.show()


print("Total image count in TRAIN dataset: ", len(os.listdir("/kaggle/input/histopathologic-cancer-detection/train")))
print("Total image count in TEST dataset: ", len(os.listdir("/kaggle/input/histopathologic-cancer-detection/test")))


# Check for duplicates - there should be none - again check for uniques
print("\nDuplicate rows in training labels:")
print(dataset[dataset.duplicated(keep=False)])


# Set sample size to 50,000
SAMPLE_SIZE=50000
# Set random states to 72
RANDOM_STATE=72


# Create the train and test sets with the sample size
dset0=dataset[dataset['label']==0].sample(SAMPLE_SIZE,random_state=RANDOM_STATE)
dset1=dataset[dataset['label']==1].sample(SAMPLE_SIZE,random_state=RANDOM_STATE)

# Put the dataset dataframes together
# update the dataset variable appropriately
dataset = pd.concat([dset0, dset1], axis=0).reset_index(drop=True)
# shuffle
dataset = shuffle(dataset)

# Check that both the label counts are 50,000 each
dataset['label'].value_counts()


dataset


# Create the 80-20 split for the image data labels

# The validation/actual labels - get the labels of either 0 or 1
y = dataset['label']

# Split the train data and test labels
train_data, train_labels = train_test_split(dataset, test_size=0.2, 
                                            random_state=RANDOM_STATE, stratify=y)

# Print out the shapes to verify sizes of the train and test datasets
print(train_data.shape, train_labels.shape)


# Get a sample of train_data and train_labels
train_data.head()


train_labels.head()


train_data.columns, train_labels.columns


base_dir = '/kaggle/working/base_dir'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')


# Use ImageDataGenerator and create a base folder called 'base' to test out model
# and determine what images are to be sorted as cancerous or not cancerous

# First, create the 'base' directory
# Should be removed and recreated if already there

import shutil

base_dir = '/kaggle/working/base_dir'
os.makedirs(base_dir, exist_ok=True)


# Create a path to 'base' to make the two directories inside of base
# Make the directory train_dir
train_dir = os.path.join(base_dir, 'train_imgs')
os.makedirs(train_dir, exist_ok=True)

# Make the directory val_dir
val_dir = os.path.join(base_dir, 'val_imgs')
os.makedirs(val_dir, exist_ok=True)



# Create the subdirectories inside the train directories that were just created
negative = os.path.join(train_dir, 'negative')
os.makedirs(negative, exist_ok=True)
positive = os.path.join(train_dir, 'positive')
os.makedirs(positive, exist_ok=True)


# create new folders inside value directories that were just created
negative = os.path.join(val_dir, 'negative')
os.makedirs(negative, exist_ok=True)
positive = os.path.join(val_dir, 'positive')
os.makedirs(positive, exist_ok=True)


# check that the directories and subdirectories have been created
print(os.listdir('base_dir/train_imgs'))
print(os.listdir('base_dir/val_imgs'))


print(os.listdir('/kaggle/working/base_dir/train_imgs'))
print(os.listdir('/kaggle/working/base_dir/val_imgs'))


# dataset id set as index
dataset.set_index('id', inplace=True)


# Get the train data and train values in list form

list_train_data = list(train_data['id'])
list_train_values = list(train_labels['id'])


data_path_pull = "/kaggle/input/histopathologic-cancer-detection"


# Transfer the train images

for image in list_train_data:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    # get the label for a certain image
    target = dataset.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = 'negative'
    else:
        label = 'positive'
    
    # source path to image data path
    src = os.path.join(data_path_pull, 'train', fname)
    # destination path to image
    dst = os.path.join(train_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# Transfer the val images
for image in list_train_values:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    # get the label for a certain image
    target = dataset.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = 'negative'
    else:
        label = 'positive'
    

    # source path to image
    src = os.path.join(data_path_pull, 'train', fname)
    # destination path to image
    dst = os.path.join(val_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# check how many train images there are in each directory
print(len(os.listdir('base_dir/train_imgs/negative')))
print(len(os.listdir('base_dir/train_imgs/positive')))



# Set up the generators and the appropriate directories/paths
train_path = 'base_dir/train_imgs'
valid_path = 'base_dir/val_imgs'
test_path = '/kaggle/input/histopathologic-cancer-detection/test'

# Splits were titled train_data (80), train_labels (20)

num_train_samples = len(train_data)
num_val_samples = len(train_labels)
train_batch_size = 10
val_batch_size = 10


train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)



# Check the types on the train_steps and val_steps
type(train_steps), type(val_steps)


# Since the model only takes integers, set the train_steps and val_steps to type int and then check it. 
train_steps = train_steps.astype(int)
val_steps= val_steps.astype(int)


# Check once again
print(train_steps, train_steps.dtype)
print(val_steps, val_steps.dtype)


IMAGE_SIZE = 96

datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)


kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3


model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()


model.compile(Adam(learning_rate=0.0001), loss='binary_crossentropy', 
              metrics=['accuracy'])


print(val_gen.class_indices)


# Creating the model
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

# Set the model to 15 epochs
history = model.fit(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=15, verbose=1,
                   callbacks=callbacks_list)


model.metrics_names


val_loss= model.evaluate(test_gen, steps=len(train_labels))

print('val_loss:', val_loss)



# Get loss and accuracy rates
print('val_loss:', val_loss[0])
print('val_accuracy', val_loss[1])


# Get the loss and accuracy visuals
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show()


# make a prediction
predictions = model.predict(test_gen)


predictions.shape



data_preds = pd.DataFrame(predictions, columns=['negative', 'positive'])

data_preds.head()


data_preds.shape


len(test_gen.filenames)


# Get the true labels
y_true = test_gen.classes

# Get the predicted labels as probabilities
y_pred = data_preds['positive']


from sklearn.metrics import roc_auc_score
# Get ROC Score
roc_auc_score(y_true, y_pred)


test_labels = test_gen.classes
test_labels.shape


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Get max value in row using argmax
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
# Print the label associated with each class
test_gen.class_indices


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()
plt.show()


# Remove base_dir and create test_dir

shutil.rmtree('base_dir')



# create test_dir
test_dir = 'test_dir'
os.makedirs(test_dir, exist_ok=True)

test_imgs = os.path.join(test_dir, 'test_imgs')
os.makedirs(test_imgs, exist_ok=True)

os.listdir('test_dir')


# Transfer the test images into image_dir

test_list = os.listdir('../input/histopathologic-cancer-detection/test')

for image in test_list:
    
    fname = image
    
    # source path to image
    src = os.path.join('../input/histopathologic-cancer-detection/test', fname)
    # destination path to image
    dst = os.path.join(test_imgs, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)
# check that the images are now in the test_images
# Total is 57458 images in the test_imgs folder
len(os.listdir('test_dir/test_imgs'))


test_path ='test_dir'
# Adjust the path to put images into the test_imgs directory.
test_gen = datagen.flow_from_directory(test_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)


test_imgs_ct = 57458
test_predictions = model.predict(test_gen, steps=test_imgs_ct, verbose=1)


len(test_predictions)


# Create dataframe with all test's predictions
test_preds= pd.DataFrame(test_predictions, columns=['negative', 'positive'])
test_preds.head()


# Get all filenames and add to the test_preds dataframe
test_fnames = test_gen.filenames

test_preds['fnames'] = test_fnames
test_preds.head()


test_preds.fnames


''' Similar to the sample submission, create an id column
Get rid of the 'test_imgs/' portion of each of the file names
to get just the file name as well as rid the '.tif' at the end of 
each cell in fname column
'''


def get_id(x):
    
    # split into a list
    a = x.split('/')
    # split into a list
    b = a[1].split('.')
    extracted_id = b[0]
    
    return extracted_id

test_preds['id'] = test_preds['fnames'].apply(get_id)

test_preds.head()




# Sample of a fnames cell
test_preds['fnames'][0]


# fnames column without the 'test_imgs/' at beginning or the trailing '.tif' at end
test_preds['id']


# Get predicted labels with cancerous/positive images and their respective ids
y_pred = test_preds['positive']
img_ids = test_preds['id']


# Create submission file dataframe
submission = pd.DataFrame({'id':img_ids, 
                           'label':y_pred, 
                          }).set_index('id')

# Create CSV file to be submitted
submission.to_csv('patch_preds.csv', columns=['label']) 
submission.head()



# Remove all contents in test_dir
shutil.rmtree('test_dir')


cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
# Print the label associated with each class
test_gen.class_indices
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()
plt.show()


# Compare subplots from previous confusion matrix. 
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix')
plt.show()




