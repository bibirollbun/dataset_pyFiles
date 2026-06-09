import pandas as pd
import numpy as np
import os
from datetime import datetime
from datetime import time

#VisualizaÃ§Ã£o
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

#Audio
import torchaudio
from IPython.display import Audio

'''
silero-vad -- biblioteca para reconhecimento de voz humana nos Ã¡udios
librosa    -- biblioteca para carregar/manipular arquivos de audio
'''


root = '/kaggle/input/birdclef-2025'
trainAudios = os.path.join(root, 'train_audio')
trainSoundscapes = os.path.join(root, 'train_soundscapes')
test_soundscapes = os.path.join(root, 'test_soundscapes')

df = pd.read_csv(os.path.join(root, 'train.csv'))
recording_location = os.path.join(root, 'recording_location.txt')
taxonomy = pd.read_csv(os.path.join(root, 'taxonomy.csv'))


df['class_name'] = df['primary_label'].map(taxonomy.set_index('primary_label')['class_name'])
df.head(3)


data, rate = torchaudio.load(os.path.join(trainAudios, df['filename'][28000]))

display(Audio(data[0, :rate*5], rate = rate))
px.line(y=data[0, :rate*5], title=df.common_name[28000])


fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='primary_label', 
                        hover_name='primary_label', hover_data=['latitude', 'longitude', 'common_name', 'class_name'], 
                        title='Geographical Distribution of Species',
                        zoom=1, height=600)

fig.update_layout(mapbox_style="open-street-map")
fig.show()


df.groupby('class_name')['collection'].count()/len(df)


# DistribuiÃ§Ã£o das classes no dataset
plt.figure(figsize=(12, 6))
sns.countplot(x='class_name', data=df, order=df['class_name'].value_counts().index)
plt.title('Distribution of Classes')
plt.xlabel('Class Name')
plt.ylabel('Count')
plt.show()


df.loc[(df['secondary_labels'] != "['']") & (df['collection'] == 'XC')].head()
# Apenas os audios do xeno canto possuem secondary_labels, e ainda sÃ£o a minoria deles


# Fazendo um dataframe com os dados sem rÃ³tulos fornecidos pela competiÃ§Ã£o

soundscapes = pd.DataFrame(columns=['file','site', 'date', 'local_time'])

for f in os.listdir(trainSoundscapes):

    aux = os.path.basename(f).replace('.ogg', '').split('_')
    site = aux[0]
    date = datetime.strptime(aux[1], "%Y%m%d").strftime("%Y/%m/%d")
    local_time = time(int(aux[2][:2]), int(aux[2][2:4]), int(aux[2][4:6]))

    new_row = pd.DataFrame([[f, site, date, local_time]], columns=['file', 'site', 'date', 'local_time'])
    soundscapes = pd.concat([soundscapes, new_row], axis=0, ignore_index=True)

soundscapes


soundscapes['site'].value_counts()/len(soundscapes)



fig, axs = plt.subplots(1, 1, figsize=(10,5))
axs.hist(soundscapes['local_time'].map(lambda x: x.hour), histtype='bar', rwidth=0.8)
axs.set_title('DistribuiÃ§Ã£o das amostras pelo tempo')
plt.show()




