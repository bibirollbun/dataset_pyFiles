import plotly.express as px


import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# Map 1 library
import plotly.express as px

# Map 2 libraries
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Librosa Libraries
import librosa
import librosa.display
import IPython.display as ipd

import sklearn

import warnings
warnings.filterwarnings('ignore')


# Import data
train_csv = pd.read_csv("../input/birdsong-recognition/train.csv")
test_csv = pd.read_csv("../input/birdsong-recognition/test.csv")

# Create some time features
train_csv['year'] = train_csv['date'].apply(lambda x: x.split('-')[0])
train_csv['month'] = train_csv['date'].apply(lambda x: x.split('-')[1])
train_csv['day_of_month'] = train_csv['date'].apply(lambda x: x.split('-')[2])

print("There are {:,} unique bird species in the dataset.".format(len(train_csv['species'].unique())))


# Inspect text_csv before checking train data
test_csv




plt.figure(figsize=(16, 6))
ax = sns.countplot(train_csv['year'], palette="hls")

plt.title("Audio Files Registration per Year Made", fontsize=16)
plt.xticks(rotation=90, fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel("Frequency", fontsize=14)
plt.xlabel("")




plt.figure(figsize=(20, 6))
ax = sns.countplot(train_csv['month'], palette="Blues_d")


plt.title("Audio Files Registration per Month Made", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel("Frequency", fontsize=14)
plt.xlabel("");



plt.figure(figsize=(20, 6))
ax = sns.countplot(train_csv['pitch'], palette="hls", order = train_csv['pitch'].value_counts().index)

plt.title("Pitch (quality of sound - how high/low the tone is)", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel("Frequency", fontsize=14)
plt.xlabel("");


# Create a new variable type by exploding all the values
adjusted_type = train_csv['type'].apply(lambda x: x.split(',')).reset_index().explode("type")

# Strip of white spaces and convert to lower chars
adjusted_type = adjusted_type['type'].apply(lambda x: x.strip().lower()).reset_index()
adjusted_type['type'] = adjusted_type['type'].replace('calls', 'call')

# Create Top 15 list with song types
top_15 = list(adjusted_type['type'].value_counts().head(15).reset_index()['index'])
data = adjusted_type[adjusted_type['type'].isin(top_15)]

# === PLOT ===

plt.figure(figsize=(20, 6))
ax = sns.countplot(data['type'], palette="hls", order = data['type'].value_counts().index)

plt.title("Top 15 Song Types", fontsize=16)
plt.ylabel("Frequency", fontsize=14)
plt.yticks(fontsize=13)
plt.xticks(rotation=45, fontsize=13)
plt.xlabel("");


# Top 15 most common elevations
top_15 = list(train_csv['elevation'].value_counts().head(15).reset_index()['index'])
data = train_csv[train_csv['elevation'].isin(top_15)]

# === PLOT ===

plt.figure(figsize=(20, 6))
ax = sns.countplot(data['elevation'], palette="Blues_d", order = data['elevation'].value_counts().index)

plt.title("Top 15 Elevation Types", fontsize=16)
plt.ylabel("Frequency", fontsize=14)
plt.yticks(fontsize=13)
plt.xticks(rotation=45, fontsize=13)
plt.xlabel("");


# Create data
data = train_csv['bird_seen'].value_counts().reset_index()

# === PLOT ===

plt.figure(figsize=(20, 6))
ax = sns.barplot(x = 'bird_seen', y = 'index', data = data, palette="hls")

plt.title("Song was Heard, but was Bird Seen?", fontsize=16)
plt.ylabel("Frequency", fontsize=14)
plt.yticks(fontsize=13)
plt.xticks(rotation=45, fontsize=13)
plt.xlabel("");


# Top 15 most common elevations
top_15 = list(train_csv['country'].value_counts().head(15).reset_index()['index'])
data = train_csv[train_csv['country'].isin(top_15)]

# === PLOT ===

plt.figure(figsize=(20, 6))
ax = sns.countplot(data['country'], palette='hls', order = data['country'].value_counts().index)

plt.title("Top 15 Countries with most Recordings", fontsize=16)
plt.ylabel("Frequency", fontsize=14)
plt.yticks(fontsize=13)
plt.xticks(rotation=45, fontsize=13)
plt.xlabel("");


import pandas as pd
import plotly.express as px

# Simulate your data
train_csv = pd.DataFrame({
    'country': ['India', 'United States', 'Brazil', 'India', 'Brazil'],
    'species': ['bird1', 'bird2', 'bird3', 'bird4', 'bird5']
})

# Get ISO codes from gapminder
df = px.data.gapminder().query("year==2007")[["country", "iso_alpha"]]

# Merge and count
data = pd.merge(train_csv, df, on="country", how="inner")
data = data.groupby(["country", "iso_alpha"]).count().reset_index()

# Plot
fig = px.choropleth(data,
                    locations="iso_alpha",
                    color="species",
                    hover_name="country",
                    color_continuous_scale=px.colors.sequential.Teal,
                    title="Species Count per Country")

# Save as HTML
fig.write_html("choropleth_map.html")



import IPython.display as display
display.IFrame('choropleth_map.html', width=1000, height=600)



print("Columns in train_csv:", train_csv.columns.tolist())
print("Columns in gapminder:", px.data.gapminder().columns.tolist())
print(train_csv.columns.tolist())



print(train_csv.columns)
print(train_csv.sample(5))



# SHP file
world_map = gpd.read_file("../input/world-shapefile/world_shapefile.shp")

# Coordinate reference system
crs = {"init" : "epsg:4326"}

# Lat and Long need to be of type float, not object
data = train_csv[train_csv["latitude"] != "Not specified"]
data["latitude"] = data["latitude"].astype(float)
data["longitude"] = data["longitude"].astype(float)

# Create geometry
geometry = [Point(xy) for xy in zip(data["longitude"], data["latitude"])]

# Geo Dataframe
geo_df = gpd.GeoDataFrame(data, crs=crs, geometry=geometry)

# Create ID for species
species_id = geo_df["species"].value_counts().reset_index()
species_id.insert(0, 'ID', range(0, 0 + len(species_id)))

species_id.columns = ["ID", "species", "count"]

# Add ID to geo_df
geo_df = pd.merge(geo_df, species_id, how="left", on="species")

# === PLOT ===
fig, ax = plt.subplots(figsize = (16, 10))
world_map.plot(ax=ax, alpha=0.4, color="grey")

palette = iter(sns.hls_palette(len(species_id)))

for i in range(264):
    geo_df[geo_df["ID"] == i].plot(ax=ax, markersize=20, color=next(palette), marker="o", label = "test");


# Creating Interval for *duration* variable
train_csv['duration_interval'] = ">500"
train_csv.loc[train_csv['duration'] <= 100, 'duration_interval'] = "<=100"
train_csv.loc[(train_csv['duration'] > 100) & (train_csv['duration'] <= 200), 'duration_interval'] = "100-200"
train_csv.loc[(train_csv['duration'] > 200) & (train_csv['duration'] <= 300), 'duration_interval'] = "200-300"
train_csv.loc[(train_csv['duration'] > 300) & (train_csv['duration'] <= 400), 'duration_interval'] = "300-400"
train_csv.loc[(train_csv['duration'] > 400) & (train_csv['duration'] <= 500), 'duration_interval'] = "400-500"


plt.figure(figsize=(20, 6))
ax = sns.countplot(train_csv['duration_interval'], palette="hls")

plt.title("Distribution of Recordings Duration", fontsize=16)
plt.ylabel("Frequency", fontsize=14)
plt.yticks(fontsize=13)
plt.xticks(rotation=45, fontsize=13)
plt.xlabel("");


def show_values_on_bars(axs, h_v="v", space=0.4):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# Create Full Path so we can access data more easily
base_dir = '../input/birdsong-recognition/train_audio/'
train_csv['full_path'] = base_dir + train_csv['ebird_code'] + '/' + train_csv['filename']

# Now let's sample a fiew audio files
amered = train_csv[train_csv['ebird_code'] == "amered"].sample(1, random_state = 33)['full_path'].values[0]
cangoo = train_csv[train_csv['ebird_code'] == "cangoo"].sample(1, random_state = 33)['full_path'].values[0]
haiwoo = train_csv[train_csv['ebird_code'] == "haiwoo"].sample(1, random_state = 33)['full_path'].values[0]
pingro = train_csv[train_csv['ebird_code'] == "pingro"].sample(1, random_state = 33)['full_path'].values[0]
vesspa = train_csv[train_csv['ebird_code'] == "vesspa"].sample(1, random_state = 33)['full_path'].values[0]

bird_sample_list = ["amered", "cangoo", "haiwoo", "pingro", "vesspa"]


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2
from IPython.display import display, Audio

def viz_from_path(filepath):
    # Load the audio file
    data, sr = librosa.load(filepath, sr=22050)

    # Generate the mel spectrogram
    melsp_feature = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
    melsp_db = librosa.power_to_db(melsp_feature, ref=np.max)

    # Create signal (you can modify this to split data into chunks if needed)
    signals = [data]

    # Set up the plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Plot signal
    ax[0].plot(data, color='crimson')
    ax[0].set_title("Signal", fontsize=16)

    # Resize and plot mel spectrogram
    mel_resized = cv2.resize(melsp_db, (4096, 1024))
    ax[1].imshow(mel_resized, aspect='auto', origin='lower')
    ax[1].set_title("Melspectrogram", fontsize=16)

    # Display audio player
    display(Audio(filepath))

    # Display the plot
    plt.tight_layout()
    plt.show()

# Sample bird audio file
cangoo = train_csv[train_csv['ebird_code'] == "cangoo"].sample(1, random_state=33)['full_path'].values[0]

# Visualize the audio file and its features
viz_from_path(cangoo)



viz_from_path(amered)



# Haiwoo
viz_from_path(haiwoo)


# Pingro
viz_from_path(pingro)


# Vesspa
viz_from_path(vesspa)


# Importing 1 file
y, sr = librosa.load(vesspa)

print('y:', y, '\n')
print('y shape:', np.shape(y), '\n')
print('Sample Rate (KHz):', sr, '\n')

# Verify length of the audio
print('Check Len of Audio:', np.shape(y)[0]/sr)


# Trim leading and trailing silence from an audio signal (silence before and after the actual audio)
audio_file, _ = librosa.effects.trim(y)

# the result is an numpy ndarray
print('Audio File:', audio_file, '\n')
print('Audio File shape:', np.shape(audio_file))


# Importing the 5 files
y_amered, sr_amered = librosa.load(amered)
audio_amered, _ = librosa.effects.trim(y_amered)

y_cangoo, sr_cangoo = librosa.load(cangoo)
audio_cangoo, _ = librosa.effects.trim(y_cangoo)

y_haiwoo, sr_haiwoo = librosa.load(haiwoo)
audio_haiwoo, _ = librosa.effects.trim(y_haiwoo)

y_pingro, sr_pingro = librosa.load(pingro)
audio_pingro, _ = librosa.effects.trim(y_pingro)

y_vesspa, sr_vesspa = librosa.load(vesspa)
audio_vesspa, _ = librosa.effects.trim(y_vesspa)


fig, ax = plt.subplots(5, figsize = (16, 9))
fig.suptitle('Sound Waves', fontsize=16)

librosa.display.waveplot(y = audio_amered, sr = sr_amered, color = "#A300F9", ax=ax[0])
librosa.display.waveplot(y = audio_cangoo, sr = sr_cangoo, color = "#4300FF", ax=ax[1])
librosa.display.waveplot(y = audio_haiwoo, sr = sr_haiwoo, color = "#009DFF", ax=ax[2])
librosa.display.waveplot(y = audio_pingro, sr = sr_pingro, color = "#00FFB0", ax=ax[3])
librosa.display.waveplot(y = audio_vesspa, sr = sr_vesspa, color = "#D9FF00", ax=ax[4]);

for i, name in zip(range(5), bird_sample_list):
    ax[i].set_ylabel(name, fontsize=13)


# Default FFT window size
n_fft = 2048 # FFT window size
hop_length = 512 # number audio of frames between STFT columns (looks like a good default)

# Short-time Fourier transform (STFT)
D_amered = np.abs(librosa.stft(audio_amered, n_fft = n_fft, hop_length = hop_length))
D_cangoo = np.abs(librosa.stft(audio_cangoo, n_fft = n_fft, hop_length = hop_length))
D_haiwoo = np.abs(librosa.stft(audio_haiwoo, n_fft = n_fft, hop_length = hop_length))
D_pingro = np.abs(librosa.stft(audio_pingro, n_fft = n_fft, hop_length = hop_length))
D_vesspa = np.abs(librosa.stft(audio_vesspa, n_fft = n_fft, hop_length = hop_length))


print('Shape of D object:', np.shape(D_amered))


# Convert an amplitude spectrogram to Decibels-scaled spectrogram.
DB_amered = librosa.amplitude_to_db(D_amered, ref = np.max)
DB_cangoo = librosa.amplitude_to_db(D_cangoo, ref = np.max)
DB_haiwoo = librosa.amplitude_to_db(D_haiwoo, ref = np.max)
DB_pingro = librosa.amplitude_to_db(D_pingro, ref = np.max)
DB_vesspa = librosa.amplitude_to_db(D_vesspa, ref = np.max)

# === PLOT ===
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Spectrogram', fontsize=16)
fig.delaxes(ax[1, 2])

librosa.display.specshow(DB_amered, sr = sr_amered, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'cool', ax=ax[0, 0])
librosa.display.specshow(DB_cangoo, sr = sr_cangoo, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'cool', ax=ax[0, 1])
librosa.display.specshow(DB_haiwoo, sr = sr_haiwoo, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'cool', ax=ax[0, 2])
librosa.display.specshow(DB_pingro, sr = sr_pingro, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'cool', ax=ax[1, 0])
librosa.display.specshow(DB_vesspa, sr = sr_vesspa, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'cool', ax=ax[1, 1]);

for i, name in zip(range(0, 2*3), bird_sample_list):
    x = i // 3
    y = i % 3
    ax[x, y].set_title(name, fontsize=13) 


# Create the Mel Spectrograms
S_amered = librosa.feature.melspectrogram(y_amered, sr=sr_amered)
S_DB_amered = librosa.amplitude_to_db(S_amered, ref=np.max)

S_cangoo = librosa.feature.melspectrogram(y_cangoo, sr=sr_cangoo)
S_DB_cangoo = librosa.amplitude_to_db(S_cangoo, ref=np.max)

S_haiwoo = librosa.feature.melspectrogram(y_haiwoo, sr=sr_haiwoo)
S_DB_haiwoo = librosa.amplitude_to_db(S_haiwoo, ref=np.max)

S_pingro = librosa.feature.melspectrogram(y_pingro, sr=sr_pingro)
S_DB_pingro = librosa.amplitude_to_db(S_pingro, ref=np.max)

S_vesspa = librosa.feature.melspectrogram(y_vesspa, sr=sr_vesspa)
S_DB_vesspa = librosa.amplitude_to_db(S_vesspa, ref=np.max)

# === PLOT ====
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Mel Spectrogram', fontsize=16)
fig.delaxes(ax[1, 2])

librosa.display.specshow(S_DB_amered, sr = sr_amered, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'rainbow', ax=ax[0, 0])
librosa.display.specshow(S_DB_cangoo, sr = sr_cangoo, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'rainbow', ax=ax[0, 1])
librosa.display.specshow(S_DB_haiwoo, sr = sr_haiwoo, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'rainbow', ax=ax[0, 2])
librosa.display.specshow(S_DB_pingro, sr = sr_pingro, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'rainbow', ax=ax[1, 0])
librosa.display.specshow(S_DB_vesspa, sr = sr_vesspa, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'rainbow', ax=ax[1, 1]);

for i, name in zip(range(0, 2*3), bird_sample_list):
    x = i // 3
    y = i % 3
    ax[x, y].set_title(name, fontsize=13)


# Total zero_crossings in our 1 song
zero_amered = librosa.zero_crossings(audio_amered, pad=False)
zero_cangoo = librosa.zero_crossings(audio_cangoo, pad=False)
zero_haiwoo = librosa.zero_crossings(audio_haiwoo, pad=False)
zero_pingro = librosa.zero_crossings(audio_pingro, pad=False)
zero_vesspa = librosa.zero_crossings(audio_vesspa, pad=False)

zero_birds_list = [zero_amered, zero_cangoo, zero_haiwoo, zero_pingro, zero_vesspa]

for bird, name in zip(zero_birds_list, bird_sample_list):
    print("{} change rate is {:,}".format(name, sum(bird)))


y_harm_haiwoo, y_perc_haiwoo = librosa.effects.hpss(audio_haiwoo)

plt.figure(figsize = (16, 6))
plt.plot(y_perc_haiwoo, color = '#FFB100')
plt.plot(y_harm_haiwoo, color = '#A300F9')
plt.legend(("Perceptrual", "Harmonics"))
plt.title("Harmonics and Perceptrual : Haiwoo Bird", fontsize=16);


# Calculate the Spectral Centroids
spectral_centroids = librosa.feature.spectral_centroid(audio_cangoo, sr=sr)[0]

# Shape is a vector
print('Centroids:', spectral_centroids, '\n')
print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')

# Computing the time variable for visualization
frames = range(len(spectral_centroids))

# Converts frame counts to time (seconds)
t = librosa.frames_to_time(frames)

print('frames:', frames, '\n')
print('t:', t)

# Function that normalizes the Sound Data
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


#Plotting the Spectral Centroid along the waveform
plt.figure(figsize = (16, 6))
librosa.display.waveplot(audio_cangoo, sr=sr, alpha=0.4, color = '#A300F9', lw=3)
plt.plot(t, normalize(spectral_centroids), color='#FFB100', lw=2)
plt.legend(["Spectral Centroid", "Wave"])
plt.title("Spectral Centroid: Cangoo Bird", fontsize=16);


# Increase or decrease hop_length to change how granular you want your data to be
hop_length = 5000

# Chromogram Vesspa
chromagram = librosa.feature.chroma_stft(audio_vesspa, sr=sr_vesspa, hop_length=hop_length)
print('Chromogram Vesspa shape:', chromagram.shape)

plt.figure(figsize=(16, 6))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='twilight')

plt.title("Chromogram: Vesspa", fontsize=16);


# Create Tempo BPM variable
tempo_amered, _ = librosa.beat.beat_track(y_amered, sr = sr_amered)
tempo_cangoo, _ = librosa.beat.beat_track(y_cangoo, sr = sr_cangoo)
tempo_haiwoo, _ = librosa.beat.beat_track(y_haiwoo, sr = sr_haiwoo)
tempo_pingro, _ = librosa.beat.beat_track(y_pingro, sr = sr_pingro)
tempo_vesspa, _ = librosa.beat.beat_track(y_vesspa, sr = sr_vesspa)

data = pd.DataFrame({"Type": bird_sample_list , 
                     "BPM": [tempo_amered, tempo_cangoo, tempo_haiwoo, tempo_pingro, tempo_vesspa] })


# Plot
plt.figure(figsize = (20, 6))
ax = sns.barplot(y = data["BPM"], x = data["Type"], palette="hls")

plt.ylabel("BPM", fontsize=14)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.xlabel("")
plt.title("BPM for 5 Different Bird Species", fontsize=16);


# Spectral RollOff Vector
spectral_rolloff = librosa.feature.spectral_rolloff(audio_amered, sr=sr_amered)[0]

# Computing the time variable for visualization
frames = range(len(spectral_rolloff))
# Converts frame counts to time (seconds)
t = librosa.frames_to_time(frames)

# The plot
plt.figure(figsize = (16, 6))
librosa.display.waveplot(audio_amered, sr=sr_amered, alpha=0.4, color = '#A300F9', lw=3)
plt.plot(t, normalize(spectral_rolloff), color='#FFB100', lw=3)
plt.legend(["Spectral Rolloff", "Wave"])
plt.title("Spectral Rolloff: Amered Bird", fontsize=16);


# Bird Call Recognition - Model Implementation

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical




# ------------------- Feature Extraction Function -------------------

def extract_features(file_path, max_pad_len=300):
    """
    Extract features from audio file:
    - MFCC (Mel-frequency cepstral coefficients)
    - Zero Crossing Rate
    - Spectral Centroid
    - Spectral Rolloff
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        zcr = librosa.feature.zero_crossing_rate(audio)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        
        # Pad or trim to ensure consistent dimensions
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            zcr = np.pad(zcr, pad_width=((0, 0), (0, pad_width)), mode='constant')
            spectral_centroid = np.pad(spectral_centroid, pad_width=((0, 0), (0, pad_width)), mode='constant')
            spectral_rolloff = np.pad(spectral_rolloff, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
            zcr = zcr[:, :max_pad_len]
            spectral_centroid = spectral_centroid[:, :max_pad_len]
            spectral_rolloff = spectral_rolloff[:, :max_pad_len]
            
        # Stack all features
        features = np.vstack((mfccs, zcr, spectral_centroid, spectral_rolloff))
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None




# ------------------- Data Loading & Preprocessing -------------------

# Define base directory (update this to your actual path)
base_dir = '../input/birdsong-recognition/train_audio/'

# Load the training CSV
train_csv = pd.read_csv("../input/birdsong-recognition/train.csv")

# Add full path for audio files
train_csv['full_path'] = base_dir + train_csv['ebird_code'] + '/' + train_csv['filename']

# Limit to a subset of species to make the initial model building faster
# You can increase this number or remove this filter for a more comprehensive model
top_species = train_csv['species'].value_counts().head(10).index.tolist()
filtered_df = train_csv[train_csv['species'].isin(top_species)].copy()

print(f"Working with {len(filtered_df)} audio files from {len(top_species)} species")

# Encode labels
label_encoder = LabelEncoder()
filtered_df['encoded_species'] = label_encoder.fit_transform(filtered_df['species'])

# Sample a smaller dataset for quick model building
# Increase this number or remove this sampling for better results
sample_size = min(1000, len(filtered_df))
sampled_df = filtered_df.sample(sample_size, random_state=42)




# ------------------- Feature Extraction -------------------

print("Extracting features from audio files...")
features = []
labels = []

for i, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
    file_path = row['full_path']
    feature = extract_features(file_path)
    
    if feature is not None:
        features.append(feature)
        labels.append(row['encoded_species'])

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Reshape for CNN: (samples, height, width, channels)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# One-hot encode the labels
y = to_categorical(y, num_classes=len(top_species))

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")
print(f"Input shape: {X_train.shape}")




# ------------------- Model Building -------------------

# Create CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(top_species), activation='softmax')
])




# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)




# Summary of model
model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=15,  # Increase for better results
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)




# ------------------- Evaluation -------------------

# Evaluate the model
test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# Get predictions
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_true, 
    y_pred_classes, 
    target_names=[label_encoder.inverse_transform([i])[0] for i in range(len(top_species))]
))




# Confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=[label_encoder.inverse_transform([i])[0] for i in range(len(top_species))],
    yticklabels=[label_encoder.inverse_transform([i])[0] for i in range(len(top_species))]
)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()




# Plot training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()




# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()




# ------------------- Predict on Test Data -------------------

# Function to predict on test data
def predict_test(test_csv, model, label_encoder):
    # Create prediction dataframe
    predictions_df = pd.DataFrame()
    predictions_df['row_id'] = test_csv['row_id']
    
    # Extract features and predict for each test file
    for i, row in tqdm(test_csv.iterrows(), total=len(test_csv)):
        file_path = f"../input/birdsong-recognition/test_audio/{row['audio_id']}.mp3"
        
        if os.path.exists(file_path):
            feature = extract_features(file_path)
            
            if feature is not None:
                # Reshape for prediction
                feature = feature.reshape(1, feature.shape[0], feature.shape[1], 1)
                
                # Predict
                pred = model.predict(feature)
                pred_class = np.argmax(pred, axis=1)[0]
                pred_species = label_encoder.inverse_transform([pred_class])[0]
                
                # Add to dataframe
                predictions_df.loc[i, 'birds'] = pred_species
            else:
                predictions_df.loc[i, 'birds'] = 'nocall'  # Default if feature extraction fails
        else:
            predictions_df.loc[i, 'birds'] = 'nocall'  # Default if file doesn't exist
    
    return predictions_df

# Load test CSV (this would be used for test predictions if we had test data)
# test_csv = pd.read_csv("../input/birdsong-recognition/test.csv")
# predictions_df = predict_test(test_csv, model, label_encoder)
# predictions_df.to_csv('submission.csv', index=False)

print("Model training and evaluation complete!")




