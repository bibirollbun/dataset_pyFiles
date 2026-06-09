import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import IPython.display as ipd
import soundfile as sf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import xgboost as xgb 
import lightgbm as lgb  

import warnings
warnings.filterwarnings("ignore")


# Define paths
INPUT_PATH = '/kaggle/input/birdclef-2025/'
TRAIN_AUDIO_PATH = os.path.join(INPUT_PATH, 'train_audio')
TEST_SOUNDSCAPES_PATH = os.path.join(INPUT_PATH, 'test_soundscapes')
TRAIN_SOUNDSCAPES_PATH = os.path.join(INPUT_PATH, 'train_soundscapes')

# Load data
taxonomy = pd.read_csv(os.path.join(INPUT_PATH, 'taxonomy.csv'))
train_meta = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
sample_submission = pd.read_csv(os.path.join(INPUT_PATH, 'sample_submission.csv'))
recording_locations = pd.read_csv(os.path.join(INPUT_PATH, 'recording_location.txt'))


# Data Preprocessing
def preprocess_train_meta(df):
    """Preprocesses the training metadata."""
    df['secondary_labels'] = df['secondary_labels'].apply(lambda x: re.findall(r"'(\w+)'", x))
    df['len_sec_labels'] = df['secondary_labels'].map(len)
    df['file_path'] = df.apply(lambda row: os.path.join(TRAIN_AUDIO_PATH, row['filename']), axis=1)
    return df

train_meta = preprocess_train_meta(train_meta)


# Print train_meta shape
print("Train Meta Shape:", train_meta.shape)


# Print train_meta head with gradient background
print("Train Meta Head:")
display(train_meta.head().style.background_gradient(cmap='YlOrBr'))


# Print taxonomy head with gradient background
print("Taxonomy Head:")
display(taxonomy.head().style.background_gradient(cmap='plasma'))


# Print recording_locations head with gradient background
print("Recording Locations Head:")
display(recording_locations.head().style.background_gradient(cmap='plasma'))

