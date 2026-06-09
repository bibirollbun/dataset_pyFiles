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


# Code source: Brian McFee
# License: ISC

##################
# Standard imports
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display


# Code source: Brian McFee
# License: ISC

y, sr = librosa.load('../input/birdclef-2025/train_audio/1139490/CSA36389.ogg', duration=120)


# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))


# Code source: Brian McFee
# License: ISC

idx = slice(*librosa.time_to_frames([30, 35], sr=sr))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.colorbar()
plt.tight_layout()


# Code source: Brian McFee
# License: ISC

# We'll compare frames using cosine similarity, and aggregate similar frames
# by taking their (per-frequency) median value.
#
# To avoid being biased by local continuity, we constrain similar frames to be
# separated by at least 2 seconds.
#
# This suppresses sparse/non-repetetitive deviations from the average spectrum,
# and works well to discard vocal elements.

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

# The output of the filter shouldn't be greater than the input
# if we assume signals are additive.  Taking the pointwise minimium
# with the input spectrum forces this.
S_filter = np.minimum(S_full, S_filter)


# Code source: Brian McFee
# License: ISC

# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
# Note: the margins need not be equal for foreground and background separation
margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full


# Code source: Brian McFee
# License: ISC

# sphinx_gallery_thumbnail_number = 2

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Full spectrum')
plt.colorbar()

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Background')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Foreground')
plt.colorbar()
plt.tight_layout()
plt.show()


#https://stackoverflow.com/questions/36458214/split-speech-audio-file-on-words-in-python

from pydub import AudioSegment
from pydub.silence import split_on_silence

sound_file = AudioSegment.from_ogg("../input/birdclef-2025/train_audio/1139490/CSA36389.ogg")
audio_chunks = split_on_silence(sound_file, 
    # must be silent for at least half a second
    min_silence_len=500,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-16
)

for i, chunk in enumerate(audio_chunks):

    out_file = ".//splitAudio//chunk{0}.ogg".format(i)
    print ("exporting"), out_file
    chunk.export(out_file, format="ogg")


#Code by Sayantan Mazumdar https://www.kaggle.com/swaralipibose/converting-audio-to-spectogram-noise-image-data/notebook

import tensorflow as tf


#Code by Sayantan Mazumdar https://www.kaggle.com/swaralipibose/converting-audio-to-spectogram-noise-image-data/notebook

!pip install noisereduce


#Code by Sayantan Mazumdar https://www.kaggle.com/swaralipibose/converting-audio-to-spectogram-noise-image-data/notebook

import IPython
IPython.display.Audio("../input/birdclef-2025/train_audio/cocwoo1/XC11212.ogg")


#Code by Sayantan Mazumdar https://www.kaggle.com/swaralipibose/converting-audio-to-spectogram-noise-image-data/notebook

!pip install tensorflow-io


#Code by Sayantan Mazumdar https://www.kaggle.com/swaralipibose/converting-audio-to-spectogram-noise-image-data/notebook

import tensorflow_io as tfio
import tensorflow as tf


import soundfile as sf
freq,rate=sf.read('../input/birdclef-2025/train_audio/ywcpar/XC115515.ogg')


import noisereduce as nr
reduced_noise=nr.reduce_noise(y=freq,sr=rate)


import IPython
IPython.display.Audio("../input/birdclef-2025/train_audio/ywcpar/XC115515.ogg")


#Code by Sayantan Mazumdar https://www.kaggle.com/swaralipibose/converting-audio-to-spectogram-noise-image-data/notebook

def read(pth,reduce_noise):
    freq,rate=sf.read(pth)
    if reduce_noise:
        freq=nr.reduce_noise(y=freq,sr=rate)
    # Convert to spectrogram
    spectrogram = tfio.audio.spectrogram(
    reduced_noise, nfft=3600, window=256, stride=256)
    return tf.math.log(spectrogram).numpy()
plt.imshow(read('../input/birdclef-2025/train_audio/ywcpar/XC115515.ogg',True));

