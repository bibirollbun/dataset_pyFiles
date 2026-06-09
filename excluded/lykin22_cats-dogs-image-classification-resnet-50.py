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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import cv2
from tqdm import tqdm          #Used for the purpose of showing iterations getting loaded in bar kind of form
from random import shuffle
from zipfile import ZipFile
import glob
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


zip_file = glob.glob('../input/dogs-vs-cats-redux-kernels-edition/*.zip')  #return any files with .zip extension
print(zip_file)

#extract file into a temp folder
def extract_zip(file):
    with ZipFile(file,'r') as zip_ref:
        zip_ref.extractall('.')
        
#extract both train and test1 zip
for files in zip_file:
    extract_zip(files)


def get_path(directory):
    path = []
    for files in os.listdir(directory):
        path.append(files)
    return path

train_path = get_path('../working/train')
test_path = get_path('../working/test')


#this labelling function is just for the visualization, we'll have separate one for preprocessing.
def label_img(img):
    category = img.split('.')[-3]
    if category == 'cat': return [1,0]
    elif category == 'dog': return [0,1]

#Process the data, here we're converting images into numpy array. This function takes image data, image directory, a boolean as an argument.
def process_data(img_data, data_dir, isTrain=True):
    data_df = []
    for img in tqdm(img_data):
        path = os.path.join(data_dir,img)         #Assigning path to images by concatenating directory and images
        if(isTrain):
            label = label_img(img)            #Calling label_img to assign labels to image present in training directory
        else:
            label = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224,224))
        data_df.append([np.array(img),np.array(label)])          #append image and labels as numpy array in data_df list
    shuffle(data_df)
    return data_df


#We will plot the images of dogs and cats and display the assigned label above image
def show_images(data, isTest=False):
    f, ax = plt.subplots(nrows = 5, ncols = 5, figsize = (15,15))
    for i,data in enumerate(data[:25]):              #enumerate helps in keeping track of count of iterations
        img_num = data[1]
        img_data = data[0]
        label = np.argmax(img_num)                    #to get maximum indices of an array
        if label  == 1: 
            str_label='Dog'
        elif label == 0: 
            str_label='Cat'
        if(isTest):
            str_label="None"
        ax[i//5, i%5].imshow(img_data)
        ax[i//5, i%5].axis('off')                 #removing axis for better look
        ax[i//5, i%5].set_title("Label: {}".format(str_label))
    plt.show()


train = process_data(train_path, './train/')
show_images(train)


test = process_data(test_path, './test/', False)
show_images(test,True)


#We will use below function to label data
def get_label(directory):
    label = []
    for file in os.listdir(directory):
        if (file.split('.')[0] == 'dog'):
            label.append(str(1))
        elif (file.split('.')[0] == 'cat'):
            label.append(str(0))
    return label

label = get_label('../working/train')

#In case of train_test_split, allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes. Convert the images and labels into dataframe.
df = pd.DataFrame({'filename': train_path, 'label': label})
df.head()


#split data into train and valid set
train_df, valid_df = train_test_split(df, test_size = 0.2, stratify = df['label'], random_state = 123)
print(train_df.shape)
print(valid_df.shape)


#We'll perform individually on train and validation set.
train_datagen = ImageDataGenerator(rotation_range = 10, zoom_range = 0.1, horizontal_flip = True, fill_mode = 'nearest', 
                                   width_shift_range = 0.1, height_shift_range = 0.1, preprocessing_function = preprocess_input)

#flow_from_dataframe() method will accept dataframe with filenames as x_column and labels as y_column to generate mini-batches
train_gen = train_datagen.flow_from_dataframe(train_df, directory = '../working/train', x_col = 'filename', y_col = 'label', target_size = (224,224),
                                              batch_size = 64, class_mode='binary')

#we do not augment validation data.
valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

valid_gen = valid_datagen.flow_from_dataframe(valid_df, directory = '../working/train', x_col = 'filename', y_col = 'label', target_size = (224,224),
                                              batch_size = 64, class_mode='binary')


model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'max', weights = 'imagenet'))
model.add(Dense(1, activation = 'sigmoid'))

model.layers[0].trainable = False 


model.summary()


model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'binary_crossentropy')


from tensorflow.keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(
    filepath='dogcat.weights.best.weights.h5',
    save_best_only=True,
    save_weights_only=True
)



model.fit(train_gen, epochs=10, validation_data=valid_gen, callbacks=[checkpointer])



loss = pd.DataFrame(model.history.history)
loss[['loss', 'val_loss']].plot()
loss[['accuracy', 'val_accuracy']].plot()


#load the best weights saved using checkpointer
model.load_weights('dogcat.weights.best.weights.h5')


test_df = pd.DataFrame({'filename': test_path})    #test_path from function get_path


test_sample = test_df.sample(n = 25, random_state = 123)    #Randomly sample 25 images from test directory for visualizing predictions

label_pred = []                         #list for predicted labels
for path in test_sample['filename'].to_numpy():
    full_path = '../working/test/'+path
    x = load_img(full_path, target_size=(224,224))
    img_array = img_to_array(x)
    img_array = np.expand_dims(img_array, axis=0)
    
    out =  model.predict(img_array)    
    out = 'Dog' if float(out) >0.5 else 'Cat'
    label_pred.append(out)


test_array = test_sample['filename'].to_numpy()            # convert dataframe to numpy array 

plt.figure(figsize=(15, 15))
for i in range(0, 25):
    plt.subplot(5, 5, i+1)
    img = test_array[i]
    path = '../working/test/' + img
    image = load_img(path, target_size=(256,256))
    
    plt.title('Predicted: {}'.format(label_pred[i]), color='green',fontsize= 10, bbox=dict(facecolor='black', alpha=0.7))
    plt.imshow(image)
    plt.axis('off')

plt.tight_layout()


#Generate mini batches of test data
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

test_gen = test_datagen.flow_from_dataframe(test_df, directory = '../working/test', batch_size = 64, x_col = 'filename', y_col = None, class_mode = None, shuffle = False,
                                            img_size = (224, 224))


# Performing predictions
prediction = model.predict(test_gen)
prediction = prediction.clip(min=0.005, max=0.995)



submission_df = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')

for i, fname in enumerate(test_path):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    submission_df.at[index-1, 'label'] = prediction[i]
submission_df.to_csv('Cats&DogsSubmission.csv', index=False)


submission_df.head()

