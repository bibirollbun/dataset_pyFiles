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


import zipfile
import os
import math
import imagehash
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from glob import glob

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.callbacks import EarlyStopping

from IPython.display import FileLink

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


 #create a new folder inside /kaggle/working --> Full path /kaggle/working/data/train/train
## the names of the images in train folder comes with the description if it is a cat or dog, but in the test folder we have just a numeric name 

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip','r') as zip_ext:
    zip_ext.extractall('data/train')  

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip','r') as zip_ext:
    zip_ext.extractall('data/test')


train_dir = '/kaggle/working/data/train/train'
files = os.listdir(train_dir)

data_train=[]
for i in files:
    label = i.split('.')[0]  #dog or cat
    path = os.path.join(train_dir,i)
    data_train.append((path,label))

train_df = pd.DataFrame(data_train, columns=['filepath','label'])
train_df.head()


train_df['label'].value_counts()


train_df.iloc[2]['filepath']


## Lets retrieve the size of the fist 5 files

for i in range(5):
    path = train_df.iloc[2]['filepath']
    with Image.open(path) as img:
        print(f'The size of the image is: {img.size}')  #(width,height)


def get_size(path):
    try:
        with Image.open(path) as img:
            return img.size
    except:
        raise ValueError('Unexpected error in path: ' +path)


train_df['size'] = train_df['filepath'].apply(get_size)


train_df['size'].value_counts().sort_index()


train_df[train_df['label'] == 'cat'].sample(10)


def show_random_img(df_subset, n=8):
    cols = 4  # we can adjust this of course
    rows = math.ceil (n/cols)  #this is to have a flexible parameter in case that we prefer to check more images and not assuming that the row will be 1
    
    sample_df = df_subset.sample(min(n,len(df_subset)))  #retrieve n random rows, but if the df has fewer than n rows, just retrieve how many it has.
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols,4*rows))
    ## in case that we got nxm it will be a 2D array, but the loop expect 1D, that is the reason why we need to flatten first
    axs = axs.flatten() if n>1 else [axs] 

    for i, (idx,row) in enumerate(sample_df.iterrows()):
        img = Image.open(row['filepath'])
        axs[i].imshow(img)
        axs[i].set_title(f"{idx} |{row['label']}")   #we add the index, because in some display results, it could be usefull to know the index in case that there is any weird picture

    plt.tight_layout()
    plt.show()        


## GRAYSCALE check

def is_grayscale(path):
    ## returns true is the image is not RGB
    try:
        with Image.open(path) as img:
            return img.mode != 'RGB'  #check https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    except:
        raise ValueError('Unexpected error in path: ' +path)


## BLURRINESS CHECK --> For this, we will use the variance of Laplacian method, so low variance = blurry.

def is_blurry(path, threshold=100):  #https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
    try:
        img= cv2.imread(path, cv2.IMREAD_GRAYSCALE)   #convert to grayscale for edge detection
        if img is None:
            return False
        variance = cv2.Laplacian(img,cv2.CV_64F).var()   #64-bit float
        return variance < threshold
    except:
        raise ValueError(f'Error checking blur: {path}')


### Duplicate images with hash library

def get_duplicate(path):
    try:
        with Image.open(path) as img:
            return str(imagehash.average_hash(img))
    except:
        return None


def plot_metrics(val_generator, model): 
    y_true = val_generator.classes
    y_pred_probs = model.predict(val_generator)
    y_pred = (y_pred_probs > 0.5).astype(int)  #we need to binarize the predictions. 

    report_dic = classification_report(
    y_true,
    y_pred,
    target_names=['cat','dog'],
    output_dict=True
    )
    
    report_df = pd.DataFrame(report_dic).transpose()
    report_df = report_df.round(2)  #rounds values for clarity
    report_df.style.background_gradient(cmap='Blues').format({'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"})

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,  #using labels,
        display_labels=['cat','dog'],
        cmap='Blues',
        normalize=None
    )

    plt.title('Normalized Confusion Matrix')
    plt.xticks(rotation=45)
    plt.show()

    return report_df.style.background_gradient(cmap='Blues').format({'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"})



show_random_img(train_df[train_df['label'] =='cat'],n=6)


show_random_img(train_df[train_df['label'] =='dog'],n=6)


train_df['is_grayscale'] = train_df['filepath'].apply(is_grayscale)

train_df['is_grayscale'].value_counts()


train_df['is_blurry'] = train_df['filepath'].apply(is_blurry)

train_df['is_blurry'].value_counts()


show_random_img(train_df[train_df['is_blurry'] == True],n=12)


train_df['img_hash'] = train_df['filepath'].apply(get_duplicate)


duplicates = train_df[train_df.duplicated('img_hash', keep=False)]


train_df = train_df.drop([12224,14890])


show_random_img(duplicates,n=12)


# Proceed to delete duplicate files!

train_df = train_df.drop_duplicates(subset='img_hash', keep='first')


## Before jumping with the model, it could be nice to save a cleaned version of our dataset; although this is a small dataset compared with other, it is a good practice to include in each work.

train_df.to_pickle('cleaned_train_df.pkl')
FileLink('cleaned_train_df.pkl')


train_imagedata = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,  #randomly rotates images up to +-15º (helps model to be rotation-tolerant)
    zoom_range=0.1,   #simulates camera zoom variation
    horizontal_flip=True   #randomly flig image left <-> right || learn symmetrical patterns   
)


train_generator = train_imagedata.flow_from_dataframe(
    dataframe = train_df,
    x_col = 'filepath',   #column w full path of images
    y_col='label',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='training',  #use the training part of the split
    shuffle=True,
    seed=42   
)


### As we created a train_generator should do the same for val_generator to use it later in .fit

val_generator = train_imagedata.flow_from_dataframe(
    dataframe = train_df,
    x_col='filepath',
    y_col='label',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=42    
)


### Remember that train_generator is a generator object, not a shape

model= Sequential([
    Conv2D(32,(3,3), activation = 'relu', padding='same', input_shape=train_generator.image_shape),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation = 'relu',padding='same'),
    MaxPooling2D(2,2),
    Conv2D(128,(3,3), activation = 'relu',padding='same'),
    MaxPooling2D(3,3),
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',Precision(),Recall()])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_generator,epochs=10,validation_data=val_generator,callbacks=[early_stop])
model.evaluate(val_generator)

## We save the model to reuse it later, so we dont have to re run everything again.
model.save('cat_vs_dogs.keras')
FileLink('cat_vs_dogs.keras')


model = load_model("cats_vs_dogs_model.keras")


train_datagen_resnet = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,  #randomly rotates images up to +-15º (helps model to be rotation-tolerant)
    zoom_range=0.1,   #simulates camera zoom variation
    horizontal_flip=True   #randomly flig image left <-> right || learn symmetrical patterns   
)

train_generator = train_datagen_resnet.flow_from_dataframe(
    dataframe = train_df,
    x_col = 'filepath',   #column w full path of images
    y_col='label',
    target_size=(256,256),
    batch_size=32,
    class_mode='binary',
    subset='training',  #use the training part of the split
    shuffle=True,
    seed=42   
)

val_generator = train_datagen_resnet.flow_from_dataframe(
    dataframe = train_df,
    x_col='filepath',
    y_col='label',
    target_size=(256,256),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=42    
)


## Load Resnet

resnet_base = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(256,256,3)
)

resnet_base.trainable = False  #freeze weights so we dont retrain Imagenet layers

model_res = Sequential([
    resnet_base,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])


model_res.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',Precision(),Recall()])
early_stop = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001, restore_best_weights=True)
history = model_res.fit(train_generator,epochs=10,validation_data=val_generator,callbacks=[early_stop])
model_res.evaluate(val_generator)

## We save the model to reuse it later, so we dont have to re run everything again.
model_res.save('cat_vs_dogs_res.keras')
FileLink('cat_vs_dogs_res.keras')


plot_metrics(val_generator,model)


plot_metrics(val_generator,model_res)


test_dir = '/kaggle/working/data/test/test1'
test_paths = sorted(glob(os.path.join(test_dir,"*.jpg")))

test_df = pd.DataFrame({'filename': test_paths})


test_datagen_resnet = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen_resnet.flow_from_dataframe(
    test_df,
    x_col='filename',
    class_mode=None,
    target_size=(256,256),
    shuffle=False,
    batch_size=32
)


y_pred = model_res.predict(test_generator)

#np.save('y_pred_resnet',y_pred)
#FileLink('y_pred_resnet')


y_pred_loaded = np.load("y_pred_resnet.npy")


test_labels = (y_pred > 0.5).astype(int).flatten()
ids = [int(os.path.basename(path).split('.')[0]) for path in test_generator.filenames]
submission_df = pd.DataFrame({
    'id': ids,
    'label': test_labels
}).sort_values('id')  

submission_df.to_csv("submission.csv", index=False)

