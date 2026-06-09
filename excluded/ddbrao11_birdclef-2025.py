# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from collections import Counter, defaultdict

# for audio files
!pip install -q mutagen
from mutagen.oggvorbis import OggVorbis
from mutagen import File

# play audio
from IPython.display import Audio

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


print(os.getcwd())


# dataset name and paths
dataset_name = 'birdclef-2025'
path = f'/kaggle/input/{dataset_name}'
print('dataset_name: ',dataset_name)
print('path: ',path)

train_audio_name = 'train_audio'
train_soundscapes_name = 'train_soundscapes'

train_audio_path = f'/kaggle/input/{dataset_name}/{train_audio_name}'
train_soundscapes_path = f'/kaggle/input/{dataset_name}/{train_soundscapes_name}'
# test_path = f'/kaggle/input/{dataset_name}/test'

print('\ntrain_audio_path: ',train_audio_path)
print('train_soundscapes_path: ',train_soundscapes_path)

sample_submission_name = 'sample_submission.csv'
train_name = 'train.csv'
taxonomy_name = 'taxonomy.csv'
recording_location_name = 'recording_location.txt'
test_soundscapes_name = 'test_soundscapes/readme.txt'

sample_submission_path = f'/kaggle/input/{dataset_name}/{sample_submission_name}'
train_path = f'/kaggle/input/{dataset_name}/{train_name}'
taxonomy_path = f'/kaggle/input/{dataset_name}/{taxonomy_name}'

recording_location_path = f'/kaggle/input/{dataset_name}/recording_location.txt'
readme_path = f'/kaggle/input/{dataset_name}/{test_soundscapes_name}'

print('\nsample_submission_path: ',sample_submission_path)
print('train_path: ',train_path)
print('taxonomy_path: ',taxonomy_path)
print('\nrecording_location_path: ',recording_location_path)
print('readme_path: ',readme_path)


df_sample_submission = pd.read_csv(f'{path}/{sample_submission_name}')
print('df_sample_submission shape: ',df_sample_submission.shape)
df_sample_submission.head()


df_train = pd.read_csv(f'{path}/{train_name}')
print('df_train shape: ',df_train.shape)
df_train.head()


df_taxonomy_name = pd.read_csv(f'{path}/{taxonomy_name}')
print('df_taxonomy_name shape: ',df_taxonomy_name.shape)
df_taxonomy_name.head()


# test soundscapes
with open(readme_path, 'r') as f:
    content = f.read()

# Print the content
print(content)


# train soundscapes
with open(recording_location_path, 'r') as f:
    content = f.read()

# Print the content
print(content)


# folder summary function

def analyze_folder_detailed(folder_path):
    total_file_count = 0
    extension_counter_global = Counter()
    subdir_stats = []

    for root, dirs, files in os.walk(folder_path):
        # Determine relative subdir name
        subdir_name = os.path.relpath(root, folder_path)
        if subdir_name == ".":
            subdir_name = "[root]"

        file_count = len(files)
        total_file_count += file_count

        # Count extensions in this directory
        extension_counter_local = Counter()
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            extension_counter_local[ext] += 1
            extension_counter_global[ext] += 1

        # Record stats for this folder only if it contains files
        if file_count > 0:
            row = {
                "subdirectory": subdir_name,
                "total_files": file_count,
            }
            row.update(extension_counter_local)
            subdir_stats.append(row)

    # Create DataFrame
    df = pd.DataFrame(subdir_stats)
    if not df.empty:
        df.fillna(0, inplace=True)
        df = df.astype({col: int for col in df.columns if col != 'subdirectory'})

    # Summary Info
    num_subdirs_with_files = df[df["subdirectory"] != "[root]"].shape[0]
    has_root_files = "[root]" in df["subdirectory"].values

    print(f"Analyzing folder: {folder_path}")
    print(f"Number of subdirectories with files: {num_subdirs_with_files}")
    if has_root_files:
        print("Root folder also contains files.")
    print(f"Total number of files: {total_file_count}")
    print(f"Unique file extensions: {len(extension_counter_global)}\n")

    print("File Extension Counts (Global):")
    for ext, count in extension_counter_global.items():
        print(f"  {ext or '[No Extension]'}: {count}")

    return df




 # folder summary- train_audio
df_train_metadata = analyze_folder_detailed(train_audio_path)
print('df_train_metadata: ',df_train_metadata.shape)
column_name = 'total_files'
max_value = df_train_metadata[column_name].max()
min_value = df_train_metadata[column_name].min()

print(f"Maximum value in '{column_name}': {max_value}")
print(f"Minimum value in '{column_name}': {min_value}")
df_train_metadata.sort_values(by = column_name,ascending=False).head()


 # folder summary- train_soundscapes
df_train_soundscapes_metadata = analyze_folder_detailed(train_soundscapes_path)
print('df_train_soundscapes_metadata: ',df_train_soundscapes_metadata.shape)
column_name = 'total_files'
max_value = df_train_soundscapes_metadata[column_name].max()
min_value = df_train_soundscapes_metadata[column_name].min()

print(f"Maximum value in '{column_name}': {max_value}")
print(f"Minimum value in '{column_name}': {min_value}")
df_train_soundscapes_metadata.sort_values(by = column_name,ascending=False).head()


# Load and print metadata of the audio file
def audio_metadata(audio_path):
    audio_file = File(audio_path)
    print("Metadata:")
    for key, value in audio_file.items():
        print(f"{key}: {value}")
        print("Duration (s):", audio_file.info.length)
        print("Bitrate (bps):", audio_file.info.bitrate)


# metadata of the audio file - train_audio
audio_path1 = f'/kaggle/input/{dataset_name}/train_audio/1192948/CSA36388.ogg'
audio_metadata(audio_path1)

# Play audio
Audio(filename=audio_path1)


# metadata of the audio file - train_soundscapes
audio_path2 = f'/kaggle/input/{dataset_name}/train_soundscapes/H02_20230420_074000.ogg'
audio_metadata(audio_path2)
# Play audio
Audio(filename=audio_path2)

