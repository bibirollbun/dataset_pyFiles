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


#JSON
import json


df= pd.read_json('../input/fathomnet-2025/dataset_train.json', typ="series")

df.tail()


#df = pd.read_json('../input/cusersmarildownloadsdata-json/data.json', encoding = 'utf-8-sig')

#input_df = pd.read_json('../input/traffic-violations/data.json', lines=True, orient="columns")

#data= pd.read_json('../input/traffic-violations/data.json', lines=True)

#data= pd.read_json('../input/traffic-violations/data.json', lines=True, orient='records')

#data= pd.read_json('../input/traffic-violations/data.json', orient=str)

#data= pd.read_json('../input/traffic-violations/data.json', typ="series")

#df = pd.read_json(path_or_buf='/kaggle/input/en-to-ko/EN_TO_KO.json')


#Code by Ventakumar R https://www.kaggle.com/venkatkumar001/hfp-2-eda-tensorflow/notebook

import json, codecs

with codecs.open("../input/fathomnet-2025/dataset_train.json", 'r',
                 encoding='utf-8', errors='ignore') as f:
    train_meta = json.load(f)
    
with codecs.open("../input/fathomnet-2025/dataset_test.json", 'r',
                 encoding='utf-8', errors='ignore') as f:
    test_meta = json.load(f)


#Code by Ventakumar R https://www.kaggle.com/venkatkumar001/hfp-2-eda-tensorflow/notebook

display(train_meta.keys())


#Code by Ventakumar R https://www.kaggle.com/venkatkumar001/hfp-2-eda-tensorflow/notebook

display(test_meta.keys())


#Code by Ventakumar R https://www.kaggle.com/venkatkumar001/hfp-2-eda-tensorflow/notebook

fath = pd.DataFrame(train_meta['categories'])

display(fath)


fath_list = fath.get('name').tolist()
print(fath_list)


print(fath.name.to_string(index=True))


rubescens = fath[(fath['name']=='Octopus rubescens')].reset_index(drop=True)
rubescens.head()


#Code by Ventakumar R https://www.kaggle.com/venkatkumar001/hfp-2-eda-tensorflow/notebook

fath1 = pd.DataFrame(train_meta['images'])
display(fath1)


# read JSON file
f = open('../input/fathomnet-2025/dataset_train.json')
dict_train = json.load(f)


# info section
dict_train['info']


# convert image section to data frame
df_train_images = pd.json_normalize(dict_train['images'])
# we can remove "license" as this is always 0
df_train_images.drop(['license'], axis=1, inplace=True)
# show preview
df_train_images.head(5)


# convert annotations section to data frame
df_train_annots = pd.json_normalize(dict_train['annotations'])
# show preview
df_train_annots.head(10)


# remove columns w/o info
df_train_annots.drop(['segmentation','iscrowd'], axis=1, inplace=True)


# aux function (remove suffix ".png" from file name)
def drop_suffix(i_str):
    length = len(i_str)
    return i_str[0:(length-4)]

# add filename-based id to image table in order to be able to join with df_train
df_train_images['id_file'] = df_train_images.file_name.apply(drop_suffix)


# aux function (remove suffix ".png" from file name)
def drop_suffix(i_str):
    length = len(i_str)
    return i_str[0:(length-4)]

# add filename-based id to image table in order to be able to join with df_train
df_train_images['id_file'] = df_train_images.file_name.apply(drop_suffix)


##Code by Arian Ghasemi https://www.kaggle.com/code/arianghasemi/display-images/notebook
#https://www.kaggle.com/code/mpwolke/backpacks-image-url

from IPython.core.display import HTML

def path_to_image_html(path):
    return '<img src="'+ path + '" width="60" >'
HTML(fath1[0:7].to_html(escape=False,formatters=dict(coco_url=path_to_image_html))) 


def path_to_image_html1(path1):
    return '<img src="'+ path1 + '" width="60" >'
HTML(fath1[0:5].to_html(escape=False,formatters=dict(flickr_url=path_to_image_html1)))


samp = df_train_images[(df_train_images['id']==46)].reset_index(drop=True)
samp.head()


samp['coco_url'][0]


# get an example image
#my_row = 3
#my_url = df_train_with_labels.coco_url[my_row]
#my_url

my_url = 'https://fathomnet.org/static/m3/framegrabs/Doc%20Ricketts/images/0617/01_43_19_24.png'
my_url


my_url = 'https://database.fathomnet.org/static/m3/framegrabs/Ventana/images/4194/20190530T164104Z-60628e6f-6398-4267-9757-288078efec28.png'
my_url


# image
from PIL import Image
import requests
from io import BytesIO


from urllib.request import urlopen
from PIL import Image

img = Image.open(urlopen(my_url))
img


my_database_url = 'https://database.fathomnet.org/static/m3/framegrabs/Doc%20Ricketts/images/1376/20210814T162558.344Z--ca59deda-72cb-49c5-a159-98f13741a7b3.png'
my_database_url


im = Image.open(requests.get(my_database_url, stream=True).raw)
im


my_url_data = 'https://database.fathomnet.org/static/m3/framegrabs/Doc%20Ricketts/images/1000/00_49_08_07.png'
my_url_data


# load and display image (credits: https://stackoverflow.com/questions/7391945/how-do-i-read-image-data-from-a-url-in-python)
response = requests.get(my_url_data)
my_img_train = Image.open(BytesIO(response.content))
my_img_train


#https://database.fathomnet.org/fathomnet/#/imagedetails/b7a3c3c8-8380-45bd-a6d8-fec45a23d251

import IPython
url = 'https://database.fathomnet.org/static/m3/framegrabs/Doc%20Ricketts/images/0617/01_43_19_24.png'
IPython.display.Image(url, width = 250)


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://database.fathomnet.org/static/m3/framegrabs/Doc%20Ricketts/images/0573/01_05_29_17.png',width=400,height=400)

