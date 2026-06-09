import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from IPython.display import Audio
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import xgboost as xgb 
import lightgbm as lgb  

import warnings
warnings.filterwarnings("ignore")


# Load data
INPUT_PATH = '/kaggle/input/birdclef-2025/'
TRAIN_AUDIO_PATH = os.path.join(INPUT_PATH, 'train_audio')
TEST_SOUNDSCAPES_PATH = os.path.join(INPUT_PATH, 'test_soundscapes')
TRAIN_SOUNDSCAPES_PATH = os.path.join(INPUT_PATH, 'train_soundscapes')

# Load data
taxonomy = pd.read_csv(os.path.join(INPUT_PATH, 'taxonomy.csv'))
train_meta = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
# recording_locations = pd.read_csv(os.path.join(INPUT_PATH, 'recording_location.txt'))


print(train_meta.dtypes)
train_meta.head()


print(taxonomy.dtypes)
taxonomy.head() # we see that taxonomy allow us to match 'primarry_lables' of train.csv to the species



plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.countplot(data=taxonomy, y='class_name', order=taxonomy['class_name'].value_counts().index)
plt.title('Distribution of Class Names')
plt.xlabel('Count')
plt.ylabel('Class Name')

plt.subplot(1, 2, 2)
sns.countplot(data=taxonomy, y='common_name', order=taxonomy['common_name'].value_counts().index)
plt.title('Distribution of Common Names')
plt.xlabel('Count')
plt.ylabel('Common Name')

plt.tight_layout()
plt.show()

# Check if the two columns are identical
identical = taxonomy['inat_taxon_id'].equals(taxonomy['primary_label'])
print(f"Are 'class_name' and 'common_name' identical? {identical}")

# Show rows where they differ, if any
diff_rows = taxonomy[taxonomy['class_name'] != taxonomy['common_name']]
diff_rows


# check for duplicate rows
print(f"number of duplicate rows: {train_meta.duplicated().sum()}")

# check if secondary_label and type are empty
print("secondary_labels is empty: " + str((train_meta['secondary_labels'] == '[\'\']').all()))
print("type is empty: " + str((train_meta['type'] == "[\'\']").all()))

# check for differences in columns
print(f"scientific_name and common_name are identical: {(train_meta['scientific_name'] == train_meta['common_name']).all()}")
print(f"primary_label and inat_taxon_id are identical: {(taxonomy['primary_label'] == taxonomy['inat_taxon_id']).all()}")
print(f"collection is always CSA: {(train_meta['collection'] == 'CSA').all()}")

# remove irrelevant columns
train_meta.drop(['rating', 'latitude', 'longitude', 'author', 'license'], axis=1, inplace=True)

# link the filenames to our dataset and turn secondary_labels and columns to lists
def preprocess_train_meta(df):
    """Preprocesses the training metadata."""
    df['secondary_labels'] = df['secondary_labels'].apply(lambda x: re.findall(r"'(\w+)'", x))
    df['type'] = df['type'].apply(lambda x: re.findall(r"'(\w+)'", x))
    df['file_path'] = df.apply(lambda row: os.path.join(TRAIN_AUDIO_PATH, row['filename']), axis=1)
    return df

train_meta = preprocess_train_meta(train_meta)

train_meta.head(100)


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.countplot(data=taxonomy, y='class_name', order=taxonomy['class_name'].value_counts().index)
plt.title('Distribution of Class Names')
plt.xlabel('Count')
plt.ylabel('Class Name')

plt.subplot(1, 2, 2)
sns.countplot(data=taxonomy, y='common_name', order=taxonomy['common_name'].value_counts().index)
plt.title('Distribution of Common Names')
plt.xlabel('Count')
plt.ylabel('Common Name')

plt.tight_layout()
plt.show()

# Check if the two columns are identical
identical = taxonomy['inat_taxon_id'].equals(taxonomy['primary_label'])
print(f"Are 'class_name' and 'common_name' identical? {identical}")

# Show rows where they differ, if any
diff_rows = taxonomy[taxonomy['class_name'] != taxonomy['common_name']]
diff_rows


# lets explore an audio file
audio_examples = train_meta.sample(1)
file_path = audio_examples['file_path'].tolist()[0]
title = audio_examples['primary_label'].tolist()[0]


Audio(file_path)
y, sr = librosa.load(file_path)
# Plot waveform
plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform: " + title )

# Plot spectrogram
plt.subplot(1, 2, 2)
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.title(f'Spectrogram: {title}')
plt.show()


# Feature Extraction (Simple MFCC)
def extract_mfcc(file_path, sr=22050, n_mfcc=20):
    """Extracts MFCC features from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)  # Average across time
    except Exception as e:
        # Comment out this line to suppress error messages
        # print(f"Error processing {file_path}: {e}")
        return None
    return mfccs_processed

# Example MFCC extraction
example_file = train_meta['file_path'].iloc[0]
mfccs = extract_mfcc(example_file)
print("\nMFCC Features Example:", mfccs)


# Model Training (XGBoost or LightGBM)
def train_model(train_meta, model_type='xgboost', n_samples=500, n_mfcc=20):
    """Trains a XGBoost or LightGBM model."""
    # Sample a subset of data for faster training
    train_subset = train_meta.sample(n_samples, random_state=42)

    # Extract MFCC features
    features = []
    labels = []
    for index, row in train_subset.iterrows():
        mfccs = extract_mfcc(row['file_path'], n_mfcc=n_mfcc)
        if mfccs is not None:
            features.append(mfccs)
            labels.append(row['primary_label'])

    X = np.array(features)
    y = np.array(labels)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    if model_type == 'xgboost':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42) #added eval_metric as it is needed
        model.fit(X_train, y_train)

    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(random_state=42)
        model.fit(X_train, y_train)

    else:
        raise ValueError("Invalid model_type. Choose 'xgboost' or 'lightgbm'.")

    return model, X_test, y_test, y_train

#Choose between 'xgboost' or 'lightgbm'
model, X_test, y_test, y_train = train_model(train_meta, model_type='lightgbm') #or 'xgboost'




# model evaluations

fpr, tpr, thresholds = roc_curve(y_true, y_score)



