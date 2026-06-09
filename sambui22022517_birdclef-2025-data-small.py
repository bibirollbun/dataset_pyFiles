import os
import shutil
from tqdm import tqdm
import math
import librosa
import pandas as pd


df_train = pd.read_csv('/kaggle/input/birdclef-2025-source-data/convert_train.csv')
df_sampled = df_train.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.4, random_state=42))
df_sampled.to_csv('convert_train_small.csv', index=False)


df_sampled.head()


source_folder = "/kaggle/input/birdclef-2025-source-data"
destination_folder = "/kaggle/working/"

shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)


df_train = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
df_train.to_csv('sample_submission.csv', index=False)

