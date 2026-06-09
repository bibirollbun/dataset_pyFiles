# packages

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import folium

import librosa
import librosa.display
from IPython.display import Audio


# file overview
!ls -l '../input/birdclef-2025'


# read train data file
df = pd.read_csv('../input/birdclef-2025/train.csv')

# read taxonomy file
df_taxo = pd.read_csv('../input/birdclef-2025/taxonomy.csv')


# add taxonomy info to data
df = pd.merge(left=df, right=df_taxo[['primary_label', 'inat_taxon_id', 'class_name']], how='left', on='primary_label')


# preview of data
df.head()


df.latitude.value_counts()


df.longitude.value_counts()


# convert lat/lon to numeric after removing string entries
df.latitude = pd.to_numeric(df.latitude.replace(to_replace='None', value=np.nan), errors='coerce')
df.longitude = pd.to_numeric(df.longitude.replace(to_replace='None', value=np.nan), errors='coerce')


# structure details
df.info()


# eval frequencies - primary labels
prim_freq = df.primary_label.value_counts()
prim_freq


# secondary labels
df.secondary_labels.value_counts()


# distribution of class names
df.class_name.value_counts()


# collections
df.collection.value_counts()


# ratings
plt.figure(figsize=(10,4))
df.rating.value_counts().sort_index().plot(kind='bar', color='darkblue')
plt.title('Ratings')
plt.grid()
plt.show()


# first simple plot of locations
plt.figure(figsize=(12,6))
sns.scatterplot(data=df, x='longitude', y='latitude', 
                color='darkblue')
plt.grid()
plt.show()


# select first 10 categories and plot in color
df_select = df[df.primary_label.isin(prim_freq[0:9+1].index)]
plt.figure(figsize=(12,6))
sns.scatterplot(x='longitude', y='latitude', hue='primary_label', data=df_select, palette='colorblind')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # move legend out of the plot area
plt.grid()
plt.show()


# select next 10 categories and plot in color
df_select = df[df.primary_label.isin(prim_freq[10:19+1].index)]
plt.figure(figsize=(12,6))
sns.scatterplot(x='longitude', y='latitude', hue='primary_label', data=df_select, palette='colorblind')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid()
plt.show()


# select next 10 categories and plot in color
df_select = df[df.primary_label.isin(prim_freq[20:29+1].index)]
plt.figure(figsize=(12,6))
sns.scatterplot(x='longitude', y='latitude', hue='primary_label', data=df_select, palette='colorblind')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid()
plt.show()


# select next 10 categories and plot in color
df_select = df[df.primary_label.isin(prim_freq[30:39+1].index)]
plt.figure(figsize=(12,6))
sns.scatterplot(x='longitude', y='latitude', hue='primary_label', data=df_select, palette='colorblind')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid()
plt.show()


classes = df.class_name.value_counts().index.tolist()
print(classes)


for c in classes:
    df_select = df[df.class_name==c]
    plt.figure(figsize=(12,6))
    sns.scatterplot(data=df_select, x='longitude', y='latitude', 
                    color='darkblue')
    plt.xlim(-180,180)
    plt.ylim(-70,70)
    plt.title(c)
    plt.grid()
    plt.show()


my_bird = 'grekis'
df_example = df[df.primary_label.isin([my_bird])]
df_example.shape


# check for missing coordinates
print('Missing latitudes:', df_example.latitude.isna().sum())
print('Missing longitudes:', df_example.longitude.isna().sum())


# remove rows with missings
df_example = df_example.dropna(axis=0, subset=['latitude','longitude'])
df_example.shape


# interactive map
zoom_factor = 1.9
my_map_1 = folium.Map(location=[0,0], zoom_start=zoom_factor)

for i in range(0,df_example.shape[0]):
    folium.Circle(
        location=[df_example.iloc[i]['latitude'], df_example.iloc[i]['longitude']],
        radius=np.sqrt(df_example.iloc[i]['rating'])*25000,
        color='blue',
        weight=1,
        popup='label: ' + df_example.iloc[i]['primary_label'] + '<br>' +
              'sec_labels: ' + df_example.iloc[i]['secondary_labels'] + '<br>' +
              'type: ' + df_example.iloc[i]['type'] + '<br>' +
              'URL: ' + df_example.iloc[i]['url'],
        fill=True,
        fill_color='blue').add_to(my_map_1)

my_map_1 # display


my_bird = 'banana'
df_example = df[df.primary_label.isin([my_bird])]
df_example = df_example.dropna(axis=0, subset=['latitude','longitude'])

# interactive map
zoom_factor = 1.9
my_map_2 = folium.Map(location=[0,0], zoom_start=zoom_factor)

for i in range(0,df_example.shape[0]):
    folium.Circle(
        location=[df_example.iloc[i]['latitude'], df_example.iloc[i]['longitude']],
        radius=np.sqrt(df_example.iloc[i]['rating'])*25000,
        color='red',
        weight=1,
        popup='label: ' + df_example.iloc[i]['primary_label'] + '<br>' +
              'sec_labels: ' + df_example.iloc[i]['secondary_labels'] + '<br>' +
              'type: ' + df_example.iloc[i]['type'] + '<br>' +
              'URL: ' + df_example.iloc[i]['url'],
        fill=True,
        fill_color='red').add_to(my_map_1)

my_map_1 # display


# look in an example path
!ls -l '../input/birdclef-2025/train_audio/banana'


# load audio file
filename='XC112602.ogg'
y, sr = librosa.load('../input/birdclef-2025/train_audio/banana/' + filename)


# show wave data
plt.figure(figsize=(14,5))
plt.plot(y, color='darkblue')
plt.grid()
plt.show()


# play sound
Audio(y, rate=sr)


# fourier transform + amplitudes in dB scale
ft = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(ft), ref=np.max)


# plot spectrogram
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7))
img = librosa.display.specshow(S_db, y_axis='log', sr=sr, 
                         x_axis='time', ax=ax)
ax.set(title='Spectrogram - log Frequency')
ax.label_outer()
plt.show()

