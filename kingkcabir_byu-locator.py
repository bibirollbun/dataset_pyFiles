#FORKED FROM MARILIA PRATA NO CHANGES MADE


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

import plotly.io as pio
pio.renderers.default = 'iframe'

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('../input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
train.head()


sub = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/sample_submission.csv')
sub.tail()


train.info()


#Code by Anmorgul https://www.kaggle.com/anmorgul/strange-pattern-cottonwood-willow
#https://www.kaggle.com/code/mpwolke/roosevelt-forest-of-northern-colorado-charts

for i in range(4,5):
    fig = px.scatter_3d(train, x='Motor axis 0', y='Motor axis 1', z='Motor axis 2',
                  color='Number of motors', size_max=6, width=800, height=600, opacity=0.9, template="plotly_dark")
    fig.update_layout(
        font_size=8,
        legend_font_size=16,)
    fig.show()


for i in range(4,5):
    fig = px.scatter_3d(train, x='Array shape (axis 0)', y='Array shape (axis 1)', z='Array shape (axis 2)',
                  color='Number of motors', size_max=6, width=800, height=600, opacity=0.9, template="ggplot2")
    fig.update_layout(
        font_size=8,
        legend_font_size=16,)
    fig.show()


train.describe().loc[['mean','min','max']].T


#https://stackoverflow.com/questions/64791405/log-scale-for-multiple-subplot-histograms-in-pandas

# no need to initiate `fig,ax` to avoid the warning
axes = train.hist(bins=25, figsize=(8,6), layout=(-1, 4), edgecolor="black")
plt.tight_layout()


#Lucas Dat Artist https://www.kaggle.com/code/lucasdataartist/eda-prediction-of-obesity-risk

# correlation matrix
plt.figure(figsize = (8, 4), facecolor = "white")

# plotting
sns.heatmap(
    data = train.corr(numeric_only = True),
    cmap = "summer",
    vmin = -1, vmax = 1,
    linecolor = "white", linewidth = 0.5,
    annot = True,
    fmt = ".2f"
)

plt.title('Correlation Heatmap')
plt.show()


%matplotlib inline
from PIL import Image
from glob import glob
import cv2

import math
import random


def plotImages(motor,directory):
    print(motor)
    multipleImages = glob(directory)
    plt.rcParams['figure.figsize'] = (15, 15)
    plt.subplots_adjust(wspace=0, hspace=0)
    i_ = 0
    for l in multipleImages[:25]:
        im = cv2.imread(l)
        im = cv2.resize(im, (128, 128)) 
        plt.subplot(5, 5, i_+1) #.set_title(l)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
        i_ += 1
        
        
plotImages("Bacterial Flagellar Motors train images","../input/byu-locating-bacterial-flagellar-motors-2025/train/***/**")


def plotImages(motor,directory):
    print(motor)
    multipleImages = glob(directory)
    plt.rcParams['figure.figsize'] = (15, 15)
    plt.subplots_adjust(wspace=0, hspace=0)
    i_ = 0
    for l in multipleImages[:25]:
        im = cv2.imread(l)
        im = cv2.resize(im, (128, 128)) 
        plt.subplot(5, 5, i_+1) #.set_title(l)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
        i_ += 1
        
        
plotImages("Bacterial Flagellar Motors test images","../input/byu-locating-bacterial-flagellar-motors-2025/test/***/**")


#By Yaroslav Isaienkov https://www.kaggle.com/ihelon/monet-eda-and-visualization-techniques

def visualize_images(path, n_images, is_random=True, figsize=(16, 16)):
    plt.figure(figsize=figsize)
    w = int(n_images ** .5)
    h = math.ceil(n_images / w)
    
    all_names = os.listdir(path)
    image_names = all_names[:n_images]   
    if is_random:
        image_names = random.sample(all_names, n_images)
            
    for ind, image_name in enumerate(image_names):
        img = cv2.imread(os.path.join(path, image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        plt.subplot(h, w, ind + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    
    plt.show()


tomo08bf73_JPG_PATH = '../input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_08bf73'


visualize_images(tomo08bf73_JPG_PATH, 9)


tomo05f919_JPG_PATH = '../input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_05f919'


visualize_images(tomo05f919_JPG_PATH, 9)

