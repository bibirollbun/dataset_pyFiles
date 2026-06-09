# Essential Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import ipywidgets as widgets
from IPython.display import display
import warnings

# Niche Imports
import IPython
import folium
from folium.plugins import FastMarkerCluster
import librosa
import librosa.display


# Settings
sns.set_style("whitegrid")
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
warnings.filterwarnings("ignore")


# View files

def get_comp_files_and_dirs(input_dir):
    file_list = []
    dir_list = []
    try:
        for comp_dir in os.listdir(input_dir):
            comp_path = '/'.join([input_dir, comp_dir])
            print(f"Competition Directory: {comp_path}")
            print("Contains:")
            with os.scandir(comp_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        print(f"- (File) {entry.name}, Size: {entry.stat().st_size} bytes")
                        file_list.append(os.path.join(input_dir, comp_dir, entry))
                    elif entry.is_dir():
                        print(f"- (Folder) {entry.name}")
                        dir_list.append(os.path.join(input_dir, comp_dir, entry))

    except FileNotFoundError:
        print(f"The specified directory '{directory}' does not exist.")
    except PermissionError:
        print(f"Permission error accessing directory '{directory}'.")
    return file_list, dir_list
    
input_dir = '/kaggle/input'
file_list, dir_list = get_comp_files_and_dirs(input_dir)


comp_dir = '/kaggle/input/birdclef-2025'
ss_df = pd.read_csv(os.path.join(comp_dir, 'sample_submission.csv'))
tax_df = pd.read_csv(os.path.join(comp_dir, 'taxonomy.csv'))
train_df = pd.read_csv(os.path.join(comp_dir, 'train.csv'))
with open(os.path.join(comp_dir, "recording_location.txt"), "r") as file:
    recording_location = file.read()



print(recording_location)


ss_df.head()


train_df.head()


# How many rows are there?
len(train_df)


tax_df.head(-1)


train_df = train_df.merge(tax_df[['primary_label', 'inat_taxon_id', 'class_name']], on='primary_label', how='left')


count = train_df['common_name'].value_counts(ascending=True)
plt.figure(figsize=(12, 30))
count.plot(kind='barh')

plt.xlabel('Count')
plt.ylabel('Label')
plt.title('Count of Primary Species in Recordings')
plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)

plt.tight_layout()
plt.show()



directory = "../input/birdclef-2025/train_audio"

folders_list = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
primary_label_to_name = dict(zip(tax_df['primary_label'], tax_df['common_name']))
name_to_primary_label = dict(zip(tax_df['common_name'], tax_df['primary_label']))

def audio_gui():
    # Create widgets
    animal_select = widgets.Select(
        options=sorted([primary_label_to_name.get(item, item) for item in folders_list]),
        description='Animals:',
        rows=10
    )
    
    audio_select = widgets.Select(
        options=[],
        description='Audio:',
        rows=10
    )

    audio_output = widgets.Output()

    # Function to update audio list when an animal is selected
    def on_animal_change(change):
        audio_output.clear_output()
        selected_animal = animal_select.value
        if selected_animal:
            folder_name = name_to_primary_label[selected_animal]
            audio_list = [f for f in os.listdir(os.path.join(directory, folder_name)) if f.endswith('.ogg')]
            audio_select.options = sorted(audio_list)  # Update the dropdown options

    # Function to play selected audio
    def on_audio_change(change):
        audio_output.clear_output()
        selected_animal = animal_select.value
        selected_audio = audio_select.value
        if selected_animal and selected_audio:
            audio_path = os.path.join(directory, name_to_primary_label[selected_animal], selected_audio)
            with audio_output:
                display(IPython.display.Audio(audio_path))
                y, sr = librosa.load(audio_path, sr=None)  # Load with original sample rate
                
                # Compute the spectrogram (Short-Time Fourier Transform)
                D = librosa.stft(y)
                DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert amplitude to dB
                
                # Plot the spectrogram
                plt.figure(figsize=(20, 8))
                librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
                plt.colorbar(label='dB')
                plt.title(f'Spectrogram of {animal_select.value} - {audio_select.value}')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.show()

    # Attach event listeners
    animal_select.observe(on_animal_change, names='value')
    audio_select.observe(on_audio_change, names='value')

    # Display widgets
    display(animal_select, audio_select, audio_output)

# Run the function
audio_gui()





directory = "../input/birdclef-2025/train_soundscapes"

files_list = [f for f in os.listdir(directory) if f.endswith('.ogg')]

def soundscape_gui():
    # Create widgets
    soundscape_select = widgets.Select(
        options=sorted(files_list),
        description='Soundscape:',
        rows=10,
        style={'description_width': 'auto'}
    )
    soundscape_output = widgets.Output()

    # Function to play selected audio
    def on_soundscape_change(change):
        soundscape_output.clear_output()
        if soundscape_select:
            soundscape_audio_path = os.path.join(directory, soundscape_select.value)
            with soundscape_output:
                display(IPython.display.Audio(soundscape_audio_path))
                y, sr = librosa.load(soundscape_audio_path, sr=None)  # Load with original sample rate

                # Compute the spectrogram (Short-Time Fourier Transform)
                D = librosa.stft(y)
                DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert amplitude to dB
                
                # Plot the spectrogram
                plt.figure(figsize=(20, 8))
                librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
                plt.colorbar(label='dB')
                plt.title(f'Spectrogram of {soundscape_select.value}')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.show()

    # Attach event listeners
    soundscape_select.observe(on_soundscape_change, names='value')

    # Display widgets
    display(soundscape_select, soundscape_output)
    soundscape_select.layout = widgets.Layout(width='300px')

# Run the function
soundscape_gui()


data = train_df.dropna(subset=['latitude', 'longitude'])
# Create a map centered around the mean of the data
m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=3)

# Convert DataFrame to a list of (lat, lon) tuples
locations = list(zip(data.latitude, data.longitude))

# Add FastMarkerCluster to the map
FastMarkerCluster(locations).add_to(m)

# Display the map (work
m


# View per animal
data = train_df.dropna(subset=['latitude', 'longitude'])
def animal_map_gui():
    animal_map_select = widgets.Select(
        options=tax_df['common_name'].unique(),
        description='Animals:',
        rows=10
    )
    map_output = widgets.Output()
    display(animal_map_select)
    def on_change(change):
        map_output.clear_output()
        with map_output:
            animal_id = tax_df[tax_df['common_name']==animal_map_select.value]['primary_label'].iloc[0]
            print(f'Name: {animal_map_select.value}')
            print(f'ID: {animal_id}')
            filtered_data = data[data['primary_label']==animal_id]
            print(f'Total Recordings: {len(filtered_data)}')
            m = folium.Map(location=[filtered_data['latitude'].mean(), filtered_data['longitude'].mean()], zoom_start=3)
            
            # Convert DataFrame to a list of (lat, lon) tuples
            locations = list(zip(filtered_data.latitude, filtered_data.longitude))
            FastMarkerCluster(locations).add_to(m)
            display(m)
    animal_map_select.observe(on_change, names='value')
    display(map_output)
            
animal_map_gui()

