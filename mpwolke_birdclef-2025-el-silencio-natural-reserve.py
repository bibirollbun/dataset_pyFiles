# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#By Paulo Junqueira https://www.kaggle.com/code/paulojunqueira/pew-pew-overview-birdclef-2023/notebook

import re
import librosa
import librosa.display

import IPython.display as ipd
from urllib.request import urlopen
from datetime import datetime, timedelta

import plotly.graph_objects as go
from scipy.interpolate import interp1d 
from bs4 import BeautifulSoup as bs
import librosa
import librosa.display
import IPython.display as ipd
# import noisereduce as nr

from tqdm.notebook import tqdm
# Pytorch
import torch
import torchaudio
import requests
from PIL import Image


meta = pd.read_csv('../input/birdclef-2025/train.csv')
meta['secondary_labels'] = meta['secondary_labels'].apply(lambda x: re.findall(r"'(\w+)'", x))
meta['len_sec_labels'] = meta['secondary_labels'].map(len)
meta.head(2)


#By Paulo Junqueira https://www.kaggle.com/code/paulojunqueira/pew-pew-overview-birdclef-2023/notebook

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'


df_plot = meta.groupby(['primary_label','latitude', 'longitude']).count().reset_index()[['primary_label','scientific_name','latitude', 'longitude']].rename(columns = {'scientific_name':'count'})
meta_2 = meta.merge(df_plot, on = ['primary_label','latitude', 'longitude'], how = 'left').dropna(subset = ['count'])
meta_2['count'] = meta_2['count'].astype('int')

values_list = meta_2['count'].values.tolist()

interpolation = interp1d([1, max(values_list)], [3,20])
radius = interpolation(values_list)
fig = go.Figure(go.Densitymapbox(lat =meta_2['latitude'],lon = meta_2['longitude'], radius = radius,z = meta_2['count']))

fig.update_layout(mapbox_style="open-street-map",height = 800,
                  mapbox = {
                          'center': {'lat': 0, 
                          'lon': 0},
                      'zoom':0
                  })
fig.show()


#By Ben Jenkins https://www.kaggle.com/code/benjenkins96/identify-eastern-african-bird-species-by-sound

# Set up a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot a histogram of the latitude values
meta['latitude'].hist(bins=50, ax=axs[0, 0])
axs[0, 0].set_title('Distribution of Latitude', color='red')
axs[0, 0].set_xlabel('Latitude', color='red')
axs[0, 0].set_ylabel('Count', color='red')
# Plot a histogram of the longitude values
meta['longitude'].hist(bins=50, ax=axs[0, 1])
axs[0, 1].set_title('Distribution of Longitude', color='red')
axs[0, 1].set_xlabel('Longitude', color='red')
axs[0, 1].set_ylabel('Count', color='red')

# Plot a scatterplot of the latitude and longitude values
meta.plot.scatter(x='longitude', y='latitude', alpha=0.1, ax=axs[1, 0])
axs[1, 0].set_title('Geographic Distribution of Recordings', color='red')
axs[1, 0].set_xlabel('Longitude', color='red')
axs[1, 0].set_ylabel('Latitude', color='red')

# Print the top 10 authors with the most recordings
meta['author'].value_counts().nlargest(10).plot.barh(ax=axs[1, 1])
axs[1, 1].set_title('Top 10 Authors with the Most Recordings', color='red')
axs[1, 1].set_xlabel('Count', color='red')
axs[1, 1].set_ylabel('Author', color='red')

# Adjust the layout of the subplots
plt.tight_layout()


taxonomy = pd.read_csv("/kaggle/input/birdclef-2025/taxonomy.csv")
pd.set_option('display.max_columns', None)
taxonomy.head()


ax = taxonomy['class_name'].value_counts()[:20].plot.barh(figsize=(16, 8), color='green')
ax.set_title('Colombian Animals Class names', size=18, color='orange')
ax.set_ylabel('class_name', size=10)
ax.set_xlabel('Count', size=10);


taxonomy.groupby(['scientific_name','class_name']).size().reset_index(name='count')


#Eunji Goo https://www.kaggle.com/code/quantum09/is-it-going-to-rain

class_proportion = taxonomy['class_name'].value_counts()/taxonomy['class_name'].value_counts().sum()
colormap = plt.cm.tab10(range(0, len(class_proportion)))
labels = class_proportion.index
values = class_proportion.values

bars = plt.barh(labels, values)

#plt.xlabel("Frequency") #Não alterou nada

#plt.legend(title='Forest Animals Class names' , bbox_to_anchor=(1.0, 1), loc='lower right')#HORRÌVEL!

bar_plot = class_proportion.plot.barh(color=colormap)

# Add titles, labels, invert y-axis

bar_plot.set_title("Colombian Forest Animals by Class names")
bar_plot.set_ylabel("Class Names")

total = values.sum()
for bar, count in zip(bars, values):
    width = bar.get_width()
    pct = count / total * 100
    plt.text(width, bar.get_y() + bar.get_height()/2,
             f"{count}\n({pct:.1f}%)",
             ha='left', va='center')

#Invert the axis to have the descending order
bar_plot.invert_yaxis()
plt.show(bar_plot)


#Code by Sayantan Mazumdar https://www.kaggle.com/swaralipibose/converting-audio-to-spectogram-noise-image-data/notebook

!pip install soundfile -q


#Code by Sayantan Mazumdar https://www.kaggle.com/swaralipibose/converting-audio-to-spectogram-noise-image-data/notebook

!pip install noisereduce


#Code by Sayantan Mazumdar https://www.kaggle.com/swaralipibose/converting-audio-to-spectogram-noise-image-data/notebook

#Convert Audio to Frequency

#import soundfile as sf
#freq,rate=sf.read('../input/birdclef-2025/train_audio/1139490/CSA36389.ogg')
#import plotly.express as px
#import numpy as np
#px.line(x=np.array(list(range(len(freq)))),y=freq)


#Code by Sayantan Mazumdar https://www.kaggle.com/swaralipibose/converting-audio-to-spectogram-noise-image-data/notebook

import IPython
IPython.display.Audio("../input/birdclef-2025/train_audio/1139490/CSA36389.ogg")

