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


#Code by Sumeet Sagar https://www.kaggle.com/code/ssagar012/siim-covid-19-novice-notebook-eda-box-detection

# Installing dcm library to read Dicom Images
!pip install python-gdcm
print("Installation Complete")
!pip install tensorflow-io
print(" TF - io Installed Successfully")


#Code by Sumeet Sagar https://www.kaggle.com/code/ssagar012/siim-covid-19-novice-notebook-eda-box-detection

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast #helps to process trees of the Python abstract syntax grammar.
import pydicom # for working with DICOM files such as medical images, reports, and radiotherapy objects.
import matplotlib.pyplot as plt
%matplotlib inline
import PIL # Python Imaging Library
from PIL import Image, ImageDraw, ImageFont #Python Imaging Library
import tensorflow as tf

import tensorflow_hub as hub
import wandb # experiment tracking, dataset versioning, and model management
import seaborn as sns
import tqdm # visualise progress
import cv2 #convert dicom to png


#Code by Sumeet Sagar https://www.kaggle.com/code/ssagar012/siim-covid-19-novice-notebook-eda-box-detection

# Importing the training files names
t_image_fnames = []
path = "/kaggle/input/rsna-breast-cancer-detection/train_images/"
import os
len(os.listdir(path))
for root, dirs, filenames in os.walk(path):
    for fname in filenames:
        t_image_fnames.append(os.path.join(root,fname))

#train_image_level = pd.read_csv("/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv")
train_study_level = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")    
len(t_image_fnames)


#Code by Sumeet Sagar https://www.kaggle.com/code/ssagar012/siim-covid-19-novice-notebook-eda-box-detection

# Crosschecking that the number of image file paths is same as the number of image IDs
if len(train_study_level.image_id) == len(t_image_fnames):
    print("length is almost the same")
    
else:
    print("holy moly")
    
train_study_level.head()


print("There are ",train_study_level.image_id.duplicated().sum()," Images that refer to duplicated study IDs")


X = t_image_fnames
y = train_study_level["image_id"]


#Code by Sumeet Sagar https://www.kaggle.com/code/ssagar012/siim-covid-19-novice-notebook-eda-box-detection

import matplotlib.pyplot as plt

import pydicom
%matplotlib inline
plt.figure(figsize = (10,8))
image = pydicom.dcmread(X[21])
plt.imshow(image.pixel_array,cmap=plt.cm.bone);


#Code by Sumeet Sagar https://www.kaggle.com/code/ssagar012/siim-covid-19-novice-notebook-eda-box-detection

## Function to Display 25 Images
def show_25_images(images):
    """
    Displays a plot of 25 images and their labes for training images
    """
    
    # setup the figure
    plt.figure(figsize = (10,10))
    
    # loop through 25 files to display 25 images
    for i in range(25):
        # Create subplots ( 5 rows , 5 columns)
        ax = plt.subplot(5,5,i+1)
        # display an image
        image = pydicom.dcmread(images[i])
        plt.imshow(image.pixel_array,cmap = plt.cm.bone)
        plt.axis("off")


show_25_images(X[20:])


#Code by _lev_lipinski https://www.kaggle.com/code/leventelippenszky/rsna-eda-dicom-segmentations-bboxes-3d-plot

import glob
import pydicom
import nibabel as nib
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.patches as patches


#Code by _lev_lipinski https://www.kaggle.com/code/leventelippenszky/rsna-eda-dicom-segmentations-bboxes-3d-plot

DATA_DIR = "../input/rsna-breast-cancer-detection"
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
#SEGM_DIR = os.path.join(DATA_DIR, "segmentations")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print(train_df.shape)
train_df.head()


#Code by _lev_lipinski https://www.kaggle.com/code/leventelippenszky/rsna-eda-dicom-segmentations-bboxes-3d-plot

example = "../input/rsna-breast-cancer-detection/train_images/10706/763186195.dcm"
example_ds = pydicom.dcmread(example)
example_ds


# source: https://www.kaggle.com/code/allunia/rsna-csf-cervical-spine-fracture-eda/notebook
def rescale_img_to_hu(dcm_ds):
    """Rescales the image to Hounsfield unit.
    """
    return dcm_ds.pixel_array * dcm_ds.RescaleSlope + dcm_ds.RescaleIntercept


#Code by _lev_lipinski https://www.kaggle.com/code/leventelippenszky/rsna-eda-dicom-segmentations-bboxes-3d-plot

# original image
fig, axs = plt.subplots(2, 2, figsize=(24, 12))
axs[0, 0].imshow(example_ds.pixel_array, cmap="bone")
axs[0, 0].axis("off")
sns.histplot(example_ds.pixel_array.flatten(), ax=axs[0, 1])

# rescaled image
rescaled_img = rescale_img_to_hu(example_ds)
axs[1, 0].imshow(rescaled_img, cmap="bone")
axs[1, 0].axis("off")
sns.histplot(rescaled_img.flatten(), ax=axs[1, 1]);


!pip install PyPDF2


# importing module to read PDF
import PyPDF2
  
# creating a pdf file object
pdfFileObj = open('../input/cusersmarildownloadsijerph1909756pdf/ijerph-19-09756.pdf', 'rb')


# creating a pdf reader object
PdfReader = PyPDF2.PdfReader(pdfFileObj)
  
# printing number of pages in pdf file
print (len(PdfReader.pages))
  
# creating a page object
pageObj = PdfReader.pages


for page in PdfReader.pages:
    text = page.extract_text()  # Użyj extract_text(), NIE extractText()
    print(text)


# creating a page object
pageObj = (PdfReader.pages[8])

# extracting text from page
print(pageObj.extract_text())


# closing the pdf file object
pdfFileObj.close()

