# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import glob, pylab, pandas as pd
import pydicom

import warnings
warnings.simplefilter(action='ignore', category=Warning)


import cv2

import glob
import time
import random

import nibabel as nib
from glob import glob
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut

pydicom.__version__, nib.__version__


test = pd.read_csv('../input/rsna-intracranial-aneurysm-detection/kaggle_evaluation/test.csv')
test.head()


localizers = pd.read_csv('../input/rsna-intracranial-aneurysm-detection/train_localizers.csv')
localizers.head(2)


#By Pedro Andrade https://www.kaggle.com/code/pbizil/datahackers-managers-radiografia-dos-gestores

loc_counts = localizers["location"].value_counts().head(13)#Try different values of head
sns.set(style="white")
plt.figure(figsize=(8, 6))
#x=loc_counts.index, y=loc_counts.values
ax = sns.barplot(x=loc_counts.index, y=loc_counts.values, color=sns.color_palette("Reds", n_colors=5)[3])
plt.title("Distribution of IA Location", fontsize=16)
plt.xlabel("Tags", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)
sns.despine()

#+2 is good if chart is vertical. +20 worked for horizontal
for i, v in enumerate(loc_counts.values):
    ax.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


train['Modality'].value_counts()


train = pd.read_csv('../input/rsna-intracranial-aneurysm-detection/train.csv')
train.tail()


train.info()


train.describe().loc[['mean','min','max']].T


numerical_cols =['PatientAge',
 'Left Infraclinoid Internal Carotid Artery',
 'Right Infraclinoid Internal Carotid Artery',
 'Left Supraclinoid Internal Carotid Artery',
 'Right Supraclinoid Internal Carotid Artery',
 'Left Middle Cerebral Artery',
 'Right Middle Cerebral Artery',
 'Anterior Communicating Artery',
 'Left Anterior Cerebral Artery',
 'Right Anterior Cerebral Artery',
 'Left Posterior Communicating Artery',
 'Right Posterior Communicating Artery',
 'Basilar Tip',
 'Other Posterior Circulation',
 'Aneurysm Present']


# OutlierPandas https://www.kaggle.com/code/abhyudaya456/s5e6-eda-for-predicting-optimal-fertilizers/notebook 
plt.figure(figsize=(16,8))
sns.heatmap(train[numerical_cols].corr(), annot=True, cmap='summer')
plt.title("Correlation Between Numerical Features")
plt.show()


#Original figsize 15,10
train[numerical_cols].hist(figsize=(20,15), bins=30, color='Green', edgecolor='black')
plt.suptitle("Histogram of Numeric Features")
plt.show()


#By Shivam811 https://www.kaggle.com/code/shivams811/sms-spam-detection-97-67-acc-1-0-ps/notebook

sns.histplot(train[train['Aneurysm Present'] == 0])
sns.histplot(train[train['Aneurysm Present'] == 1], color='red')
plt.title("Presence of IA Distribution");


#By Pedro Andrade https://www.kaggle.com/code/pbizil/datahackers-managers-radiografia-dos-gestores

mod_counts = train["Modality"].value_counts().head(13)#Try different values of head
sns.set(style="white")
plt.figure(figsize=(8, 6))
#x=loc_counts.index, y=loc_counts.values
ax = sns.barplot(x=mod_counts.index, y=mod_counts.values, color=sns.color_palette("Greens", n_colors=5)[3])
plt.title("Medical Imaging Series Modalities", fontsize=16)
plt.xlabel("Tags", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)
sns.despine()

#+2 is good if chart is vertical. +20 worked for horizontal
for i, v in enumerate(mod_counts.values):
    ax.text(i, v + 20, str(v), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


#Code by Abdul Basit https://www.kaggle.com/code/abdulbasitniazi/enetb7-explained-98-fine-tuning-eda

train_images = glob.glob('../input/rsna-intracranial-aneurysm-detection/series/**/*.dcm')
print("Total number of images: ", len(train_images))

train_images = pd.Series(train_images)


#By Marco Vasquez E https://www.kaggle.com/code/marcovasquez/basic-eda-data-visualization/notebook

fig=plt.figure(figsize=(15, 10))
columns = 5; rows = 4
for i in range(1, columns*rows +1):
    ds = pydicom.dcmread(train_images[i])#Original was dcmread(train_images_dir + train_images[i])
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot


print(ds) # this is file type of image


im = ds.pixel_array
print(type(im))
print(im.dtype)
print(im.shape)


#Marco Vasquez E https://www.kaggle.com/code/marcovasquez/basic-eda-data-visualization/notebook

pylab.imshow(im, cmap=pylab.cm.gist_gray)
pylab.axis('on');


#By David Roberts https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427795

import matplotlib.pylab as plt
import nibabel as nib
import shutil

# Copy a random label file to /kaggle/working directory (without .nii file extension)
src = '/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381.nii'
dst = '/kaggle/working/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381.nii'

shutil.copyfile(src, dst);

# Check the shape
img = nib.load(dst).get_fdata()
print(img.shape)

# Plot a single frame from the middle of the stack
plt.imshow(img[:,:,10]) #Original was 150
plt.show()


#By Innat https://www.kaggle.com/code/ipythonx/cervical-spine-fracture-detection-quick-eda

mrscans = sorted(
    glob(
        '../input/rsna-intracranial-aneurysm-detection/segmentations/*/*'
    )
)

# num of segmentation mask
len(mrscans)


#Innat https://www.kaggle.com/code/ipythonx/cervical-spine-fracture-detection-quick-eda

def read_nibabel(path):
    return nib.load(path).get_fdata()


#Innat https://www.kaggle.com/code/ipythonx/cervical-spine-fracture-detection-quick-eda

def multi_dim_plot(multi_dim_array, id, num_slices=49): #Original was 64
    fig = plt.figure(figsize=(30, 30))
    plt.title(
        f'Plotting first {num_slices} slices of {id}', 
        fontdict = {'fontsize' : 20}
    )
    plt.yticks([])
    plt.xticks([])
    
    xy = int(np.sqrt(num_slices))
    for i in range(num_slices):
        ax = fig.add_subplot(xy, xy, i + 1)
        plt.imshow(multi_dim_array[..., :num_slices][..., i])
        plt.axis("off")
    plt.show()


#Innat https://www.kaggle.com/code/ipythonx/cervical-spine-fracture-detection-quick-eda

from random import sample
import nibabel as nib

for msk in sample(mrscans, 2):
    m = read_nibabel(msk)
    multi_dim_plot(m, id=msk.split('/')[-1])


#Innat https://www.kaggle.com/code/ipythonx/cervical-spine-fracture-detection-quick-eda

def multi_dim_plot(multi_dim_array, id, num_slices=25): #Original was 64
    fig = plt.figure(figsize=(30, 30))
    plt.title(
        f'Plotting first {num_slices} slices of {id}', 
        fontdict = {'fontsize' : 20}
    )
    plt.yticks([])
    plt.xticks([])
    
    xy = int(np.sqrt(num_slices))
    for i in range(num_slices):
        ax = fig.add_subplot(xy, xy, i + 1)
        plt.imshow(multi_dim_array[..., :num_slices][..., i])
        plt.axis("off")
    plt.show()


#Innat https://www.kaggle.com/code/ipythonx/cervical-spine-fracture-detection-quick-eda

from random import sample
import nibabel as nib

for msk in sample(mrscans, 2):
    m = read_nibabel(msk)
    multi_dim_plot(m, id=msk.split('/')[-1])

