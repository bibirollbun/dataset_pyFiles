# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import re
import librosa
import librosa.display
import matplotlib.pyplot as plt
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


taxonomy = pd.read_csv("/kaggle/input/birdclef-2025/taxonomy.csv")
pd.set_option('display.max_columns', None)
taxonomy.head()


ax = taxonomy['class_name'].value_counts()[:20].plot.barh(figsize=(16, 8), color='blue')
ax.set_title('Colombian Animals Class names', size=18, color='red')
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

bar_plot = class_proportion.plot.barh(color= colormap)

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


    import os
    import librosa
    import numpy as np
    import pandas as pd
    
    # Set seed
    np.random.seed(2)
    
    # Class labels from train audio
    class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
    
    # List of test soundscapes (only visible during submission)
    test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes/'
    test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]
    
    # Open each soundscape and make predictions for 5-second segments
    # Use pandas df with 'row_id' plus class labels as columns
    predictions = pd.DataFrame(columns=['row_id'] + class_labels)
    for soundscape in test_soundscapes:
    
        # Load audio
        sig, rate = librosa.load(path=soundscape, sr=None)
    
        # Split into 5-second chunks
        chunks = []
        for i in range(0, len(sig), rate*5):
            chunk = sig[i:i+rate*5]
            chunks.append(chunk)
            
        # Make predictions for each chunk
        for i, chunk in enumerate(chunks):
            
            # Get row id  (soundscape id + end time of 5s chunk)      
            row_id = os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}'
            
            # Make prediction (let's use random scores for now)
            # scores = model.predict...
            scores = np.random.rand(len(class_labels))
            
            # Append to predictions as new row
            new_row = pd.DataFrame([[row_id] + list(scores)], columns=['row_id'] + class_labels)
            predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)
            
    # Save prediction as csv
    predictions.to_csv('submission.csv', index=False)
    predictions.head()


print(predictions)

