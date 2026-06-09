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


import pandas as pd
taxonomy_file_path = '/kaggle/input/birdclef-2025/taxonomy.csv'  # Update this with the actual path to your file
taxonomy_data = pd.read_csv(taxonomy_file_path)
print(taxonomy_data.head(20))



scientific_names = taxonomy_data['scientific_name']  
common_names = taxonomy_data['common_name'] 
class_name=taxonomy_data['class_name']
for sci_name, com_name, cl_name in zip(scientific_names[:5], common_names[:5],class_name[:5]):
    print(f"Scientific Name: {sci_name}, Common Name: {com_name},Class Name:{cl_name}")


import os

directory_path = '/kaggle/input/birdclef-2025/train_soundscapes'

# List files inside the directory
files = os.listdir(directory_path)

# Check the files to see if it contains audio files
print(files)


import os
import librosa

# Path to the train_soundscapes directory
train_soundscapes_dir = '/kaggle/input/birdclef-2025/train_soundscapes'

# List all files in the directory
files = os.listdir(train_soundscapes_dir)

# Filter to keep only audio files (e.g., .wav files, adjust the extension if needed)
audio_files = [f for f in files if f.endswith('.wav')]  # Change this to match your file format

# Iterate through the audio files and load them using librosa
for audio_file in audio_files:
    file_path = os.path.join(train_soundscapes_dir, audio_file)
    
    try:
        # Load the audio file (using librosa, for example)
        audio, sr = librosa.load(file_path, sr=22050)
        print(f"Successfully loaded {audio_file} with sample rate {sr}")
        
        # Process the audio here (e.g., feature extraction, classification)
        # Extract features, make predictions, etc.
        
    except Exception as e:
        print(f"Error loading {audio_file}: {e}")


# List all files without filtering by extension
files = os.listdir(train_soundscapes_dir)
print(files)  # To check what files are in the directory


# Iterate through the directory and only process files, skip directories
for file in os.listdir(train_soundscapes_dir):
    file_path = os.path.join(train_soundscapes_dir, file)
    if os.path.isfile(file_path):  # Skip directories
        try:
            # Process the audio file
            audio, sr = librosa.load(file_path, sr=22050)
            print(f"Successfully loaded {file} with sample rate {sr}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

