import os
import shutil
from tqdm import tqdm
import math
import librosa
import pandas as pd


df_train = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
filenames = []
labels = []
segment_idxes = []
for _, row in df_train.iterrows():
    label = row['primary_label']
    filename = row['filename']
    file_path = f'/kaggle/input/birdclef-2025/train_audio/{filename}'
    duration = librosa.get_duration(path=file_path)
    segments = math.ceil(duration / 5)
    for idx in range(segments):
        filenames.append(filename)
        labels.append(label)
        segment_idxes.append(idx)

convert_train_df = pd.DataFrame({
    'filename': filenames, 
    'segment_idx': segment_idxes, 
    'label': labels, 
})
convert_train_df.to_csv('convert_train.csv', index=False)


convert_train_df.head()


source_folder = "/kaggle/input/birdclef-2025"
destination_folder = "/kaggle/working/"

shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)




