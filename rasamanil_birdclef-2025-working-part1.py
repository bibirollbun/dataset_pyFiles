# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
   # for filename in filenames:
     #   print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd



from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal=plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle=cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


audio_files=glob('/kaggle/input/birdclef-2025/train_audio/*/*.ogg')


# Play audio file
ipd.Audio(audio_files[0])


y,sr=librosa.load(audio_files[0])


print(f'y: {y[10]}')
print(f'shape y: {y.shape}')
print(f'sr: {sr}')


pd.Series(y).plot(figsize=(10,5), lw=1,title='Raw Audio Example',
                                       color=color_pal[0])
                  
plt.show()


y_trimmed, _ =librosa.effects.trim(y,top_db=20)
pd.Series(y_trimmed).plot(figsize=(10,5), lw=1,title='Raw Audio Trimmed Example',
                                       color=color_pal[1])
                  
plt.show()


pd.Series(y[400000:400100]).plot(figsize=(10,5), lw=1,title='Raw Audio Zoomed In Example',
                                       color=color_pal[2])
                  
plt.show()


D = librosa.stft(y)
s_db=librosa.amplitude_to_db(np.abs(D), ref=np.max)
s_db.shape


# Plot the trasformed audio data
fig,ax = plt.subplots(figsize = (10,5))
img= librosa.display.specshow(s_db,
                              x_axis='time',
                              y_axis='log',
                              ax=ax)
ax.set_title('Spectogram Example',fontsize=20)
fig.colorbar(img,ax=ax,format=f'%0.2f')
plt.show()


S = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128,)
s_db_mel=librosa.amplitude_to_db(S, ref=np.max)

S.shape


# Plot the mel spectrogram
fig,ax = plt.subplots(figsize = (15,5))
img= librosa.display.specshow(s_db_mel,
                              x_axis='time',
                              y_axis='log',
                              ax=ax)
ax.set_title('Mel Spectogram Example',fontsize=20)
fig.colorbar(img,ax=ax,format=f'%0.2f')
plt.show()


s_db_mel

