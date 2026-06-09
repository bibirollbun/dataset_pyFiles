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


train = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')





train['secondary_labels'].value_counts()


train.head()


import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import os


len(next(os.walk('/kaggle/input/birdclef-2025/train_audio/'))[1])


train_audio_sample, sr = librosa.load('/kaggle/input/birdclef-2025/train_audio/1139490/CSA36389.ogg', sr=32000)


fig, ax = plt.subplots(nrows=2, sharex=True)
S = librosa.feature.melspectrogram(y=train_audio_sample, sr=sr, n_mels=128,
                                   fmax=8000)
mfccs = librosa.feature.mfcc(y=train_audio_sample, sr=sr, n_mfcc=40)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])

fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')


Audio(data=train_audio_sample, rate=sr)




