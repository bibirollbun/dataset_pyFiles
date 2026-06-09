import numpy as np
import pandas as pd
import os
import soundfile as sf
import librosa
from pathlib import Path
from matplotlib import pyplot as plt
import IPython.display as ipd


# Reading in the competition material
train =  pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
taxonomy =  pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')


print(f"This dataset has {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"There are {train.isna().sum().sum()} NA's in the dataset.")

# quick look at the data
train.head(3)


# a closer look reveals that our NAs are confined only to the latitude & longitude columns
train[['latitude','longitude']].isna().sum()


print(f"There are {train['primary_label'].nunique()} primary labels corresponding to {train['scientific_name'].nunique()} scientific names.")


print(f"There are {train['filename'].nunique()} unique filenames in train.")
# This 28564 matches the number of rows in train.csv

# iterate through /train_audio file structure and count .ogg files
train_ogg_files = []
for dirname, _, filenames in os.walk('/kaggle/input/birdclef-2025/train_audio'):
    for filename in filenames:
        train_ogg_files.append(os.path.join(dirname, filename))
print(f"There are {len(train_ogg_files)} .ogg files in train_audio")

# So there is one file in /train_audio that corresponds to the filename in train.csv


train.groupby('primary_label')['filename'].count()

# We can see we have a vastly different number of sound samples per species, with the max being 990!


print(f"This dataset has {taxonomy.shape[0]} rows and {taxonomy.shape[1]} columns.")
print(f"There are {taxonomy.isna().sum().sum()} NA's in the dataset.")

# quick look at the data
taxonomy.head(3)


taxonomy.groupby('class_name')['primary_label'].count().plot.pie(x='class_name',figsize=(5,5),title='Taxonomy class_name breakout',ylabel='')

# We can see from this pie chart that the majority of the sound samples involve Aves


print(f"There are {len(train_ogg_files)} .ogg files in train_audio")


def print_plot_play(x, Fs, text=''):
    """
    1. Prints information about an audio singal, 
    2. plots the waveform
    3. Creates player
    """
    print('%s Fs = %d, x.shape = %s, x.dtype = %s' % (text, Fs, x.shape, x.dtype))
    plt.figure(figsize=(8, 2))
    plt.plot(x, color='gray')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    ipd.display(ipd.Audio(data=x, rate=Fs))

def print_spectral_centroids(x, Fs, text=''):
    """
    1. Prints spectral centroid of audio signal
    """
    print('%s Fs = %d, x.shape = %s, x.dtype = %s' % (text, Fs, x.shape, x.dtype))
    plt.plot(librosa.feature.spectral_centroid(y=x, sr=Fs)[0])
    plt.xlabel('Frame number')
    plt.ylabel('frequency (Hz)')
    plt.title('Spectral centroids')
    plt.show()


# load the audio
x, Fs = librosa.load('/kaggle/input/birdclef-2025/train_audio/126247/iNat1109254.ogg', sr=None)

# print spectral centroid
print_spectral_centroids(x,Fs,'OGG file:')

# print waveform & play audio
print_plot_play(x=x, Fs=Fs, text='OGG file: ')


# load the audio
x, Fs = librosa.load('/kaggle/input/birdclef-2025/train_audio/1139490/CSA36385.ogg', sr=None)

# print spectral centroid
print_spectral_centroids(x,Fs,'OGG file:')

# print waveform & play audio
print_plot_play(x=x, Fs=Fs, text='OGG file: ')


soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes/'
soundscapes = [os.path.join(soundscape_path, afile) for afile in sorted(os.listdir(soundscape_path)) if afile.endswith('.ogg')]
if len(soundscapes) == 0:
    # not submission
    soundscape_path = '/kaggle/input/birdclef-2025/train_soundscapes/'
    soundscapes = [os.path.join(soundscape_path, afile) for afile in sorted(os.listdir(soundscape_path)) if afile.endswith('.ogg')]
    soundscapes = soundscapes[0:700] # simulate processing time of test if this is not a submission
    
print(f"There are {len(soundscapes)} ogg files in soundscapes.")


%%time

# Class labels from train audio
class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
cols = ["row_id"] + class_labels

submission = pd.DataFrame(columns=['row_id'] + class_labels)

for soundscape in soundscapes:

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
        
        # Different from the starter notebook referenced above, we will fill predictions with '.5' for now
        ### Placeholder for inference ###
        scores = [.5]*(len(class_labels))
        
        # Append to predictions as new row
        new_row = pd.DataFrame([[row_id] + (scores)], columns=['row_id'] + class_labels)
        submission = pd.concat([submission, new_row], axis=0, ignore_index=True)


sample_submission =  pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
print(f"This dataset has {sample_submission.shape[0]} rows and {sample_submission.shape[1]} columns.")

# quick look at the data
sample_submission.head(3)

# Most importantly is we need all 207 columns, and need to build our rows in 5 second increments for each file in the *_soundscapes directory


print(f"Sample submission shape is {sample_submission.shape}")
print(f"Submission shape is {submission.shape}")


submission.head(3)


submission.to_csv('submission.csv',index=False)

