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


!pip install librosa --quiet


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
import plotly.graph_objects as go
import torch
import torchaudio
import requests
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from urllib.request import urlopen
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from bs4 import BeautifulSoup as bs
from tqdm.notebook import tqdm
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


# Define dataset paths
INPUT_PATH = '/kaggle/input/birdclef-2025/'
TRAIN_AUDIO_PATH = os.path.join(INPUT_PATH, 'train_audio')
TEST_SOUNDSCAPES_PATH = os.path.join(INPUT_PATH, 'test_soundscapes')

# Load necessary data files
biodiversity_df = pd.read_csv(os.path.join(INPUT_PATH, 'taxonomy.csv'))
train = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
sample_submission_df = pd.read_csv(os.path.join(INPUT_PATH, 'sample_submission.csv'))
recording_locations_df = pd.read_csv(os.path.join(INPUT_PATH, 'recording_location.txt'))

# Process secondary_labels in training data
train = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
train['secondary_labels'] = train['secondary_labels'].apply(lambda x: re.findall(r"'(\w+)'", x))

# Add column for count of secondary labels
train['secondary_labels_count'] = train['secondary_labels'].apply(len)

# Display the first few rows of training data
train.head()


# Prepare data for visualization
grouped_df = train.groupby(['primary_label', 'latitude', 'longitude']).size().reset_index(name='count')
train_plot_df = train.merge(grouped_df, on=['primary_label', 'latitude', 'longitude'], how='left').dropna(subset=['count'])
train_plot_df['count'] = train_plot_df['count'].astype(int)

# Radius scaling interpolation
counts = train_plot_df['count'].tolist()
radius_scale = interp1d([min(counts), max(counts)], [3, 20])
radius_values = radius_scale(counts)

# Create interactive map visualization
fig = go.Figure(go.Densitymapbox(
    lat=train_plot_df['latitude'],
    lon=train_plot_df['longitude'],
    z=train_plot_df['count'],
    radius=radius_values,
    colorscale="Rainbow",
    opacity=0.7,
    colorbar=dict(title="Observation Count")
))

# Map layout adjustments
fig.update_layout(
    title="Geographic Distribution of Bird Observations",
    title_x=0.5,
    mapbox_style="carto-positron",
    height=800,
    mapbox=dict(
        zoom=2,
        center=dict(lat=train_plot_df['latitude'].mean(), lon=train_plot_df['longitude'].mean()),
    ),
    margin=dict(r=0, l=0, t=50, b=0),
)

fig.show()


# Set up an optimized figure layout for data visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram for Latitude distribution
train['latitude'].hist(bins=50, color='skyblue', ax=axes[0, 0])
axes[0, 0].set_title('Latitude Distribution', fontsize=14)
axes[0, 0].set_xlabel('Latitude', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)

# Histogram for Longitude distribution
train['longitude'].hist(bins=50, color='lightgreen', ax=axes[0, 1])
axes[0, 1].set_title('Longitude Distribution', fontsize=14)
axes[0, 1].set_xlabel('Longitude', fontsize=12)
axes[0, 1].set_ylabel('Count', fontsize=12)

# Scatter plot showing geographic distribution of recordings
train.plot.scatter(x='longitude', y='latitude', alpha=0.3, color='coral', ax=axes[1, 0])
axes[1, 0].set_title('Geographic Distribution of Recordings', fontsize=14)
axes[1, 0].set_xlabel('Longitude', fontsize=12)
axes[1, 0].set_ylabel('Latitude', fontsize=12)

# Horizontal bar chart of top 10 authors by number of recordings
train['author'].value_counts().nlargest(10).plot.barh(color='orchid', ax=axes[1, 1])
axes[1, 1].set_title('Top 10 Authors by Recordings', fontsize=14)
axes[1, 1].set_xlabel('Number of Recordings', fontsize=12)
axes[1, 1].set_ylabel('Author', fontsize=12)

# Adjust layout to prevent overlap and improve readability
plt.tight_layout()
plt.show()


# Load biodiversity taxonomy data
biodiversity_df = pd.read_csv("/kaggle/input/birdclef-2025/taxonomy.csv")

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# Display initial rows of the dataset
display(biodiversity_df.head())


# Visualize top 20 animal classes
class_counts = biodiversity_df['class_name'].value_counts().head(20)

plt.figure(figsize=(16, 8))
class_counts.plot.barh(color='forestgreen')

# Customize plot aesthetics
plt.title('Top 20 Animal Classes in Magdalena Valley', fontsize=18, color='darkorange')
plt.xlabel('Occurrences', fontsize=14)
plt.ylabel('Animal Class', fontsize=14)

# Invert y-axis for readability
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# Group biodiversity data by scientific and class names
animal_classification = biodiversity_df.groupby(['scientific_name', 'class_name']).size().reset_index(name='total_count')

# Calculate proportions of each animal class
class_counts = biodiversity_df['class_name'].value_counts()
class_proportions = class_counts / class_counts.sum()

# Plot distribution of animal classes
plt.figure(figsize=(12, 8))
plt.barh(class_proportions.index, class_proportions.values, color=plt.cm.coolwarm(np.linspace(0, 1, len(class_proportions))))

# Enhance plot aesthetics
plt.title('Distribution of Animal Classes in Magdalena Valley', fontsize=16)
plt.xlabel('Proportion of Total Observations', fontsize=12)
plt.ylabel('Animal Class', fontsize=12)

# Annotate each bar with percentage values
for i, (value, label) in enumerate(zip(class_proportions.values, class_proportions.index)):
    plt.text(value, i, f'{value:.1%}', va='center', ha='left', fontsize=10)

plt.gca().invert_yaxis()  # Highest proportions at the top
plt.tight_layout()
plt.show()


TRAIN_AUDIO_PATH = '/kaggle/input/birdclef-2025/train_audio/'
train['file_path'] = train['filename'].apply(
    lambda x: os.path.join(TRAIN_AUDIO_PATH, x)
)


import librosa
import numpy as np

def extract_audio_features(file_paths, n_mfcc=20, max_duration=5, sr=32000):
    """
    Extract MFCC features from a list of audio file paths.
    
    Args:
        file_paths (list or pd.Series): Paths to audio files.
        n_mfcc (int): Number of MFCC features to extract.
        max_duration (int): Maximum duration (in seconds) of audio to process.
        sr (int): Sampling rate for audio loading.
        
    Returns:
        np.ndarray: 2D array of shape (num_files, num_features)
    """
    features = []
    
    for fp in file_paths:
        # Load audio file with librosa, limit duration to avoid memory issues
        audio, _ = librosa.load(fp, sr=sr, duration=max_duration)

        # Ensure consistent length by padding or trimming
        required_length = sr * max_duration
        if len(audio) < required_length:
            audio = np.pad(audio, (0, required_length - len(audio)))
        else:
            audio = audio[:required_length]

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Compute statistics over MFCC features (mean, std)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Concatenate statistics into a single feature vector
        feature_vector = np.concatenate([mfcc_mean, mfcc_std])

        features.append(feature_vector)
    
    return np.array(features)


# Feature extraction using MFCC
def extract_mfcc_features(audio_path, sample_rate=22050, n_mfcc=20):
    """Extracts MFCC features from an audio file."""
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        return mfcc_mean
    except Exception:
        return None

# Model training function using XGBoost or LightGBM
def train_audio_classifier(data, model_type='lightgbm', sample_size=500, n_mfcc=20):
    """Trains an audio classifier using XGBoost or LightGBM with MFCC features."""
    data_sample = data.sample(sample_size, random_state=42)
    
    # Extract features
    X, y = [], []
    for _, row in data_sample.iterrows():
        features = extract_mfcc_features(row['file_path'], n_mfcc=n_mfcc)
        if features is not None:
            X.append(features)
            y.append(row['primary_label'])

    X = np.array(X)
    y = np.array(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'xgboost':
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False)
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(random_state=42)
    else:
        raise ValueError("Model type must be either 'xgboost' or 'lightgbm'")

    model.fit(X_train, y_train)

    return model, X_test, y_test

# Example usage
sample_audio_path = train.iloc[0]['file_path']
sample_mfcc = extract_mfcc_features(sample_audio_path)

print("Extracted MFCC Features:", sample_mfcc)


def evaluate_model(model, X_test, y_test, labels, y_train):
    """Evaluates the model performance using macro-averaged ROC-AUC."""
    y_pred_proba = model.predict_proba(X_test)

    # Binarize labels for multi-class ROC evaluation
    mlb = MultiLabelBinarizer(classes=labels)
    mlb.fit([[lbl] for lbl in np.concatenate([y_train, y_test])])
    y_test_bin = mlb.transform([[lbl] for lbl in y_test])

    roc_auc_scores = []
    fprs, tprs = [], []

    for idx, label in enumerate(labels):
        try:
            label_idx = list(mlb.classes_).index(label)

            # Check if ROC AUC can be calculated
            if len(np.unique(y_test_bin[:, label_idx])) <= 1:
                raise ValueError(f"Only one class present in y_true for {label}.")

            roc_auc = roc_auc_score(y_test_bin[:, label_idx], y_pred_proba[:, label_idx])
            fpr, tpr, _ = roc_curve(y_test_bin[:, label_idx], y_pred_proba[:, label_idx])

            roc_auc_scores.append(roc_auc)
            fprs.append(fpr)
            tprs.append(tpr)

        except (ValueError, IndexError) as e:
            print(f"Skipped label {label}: {e}")
            roc_auc_scores.append(np.nan)
            fprs.append(None)
            tprs.append(None)

    # Compute macro-averaged ROC-AUC
    macro_roc_auc = np.nanmean(roc_auc_scores)
    print(f"Macro-Averaged ROC AUC: {macro_roc_auc:.4f}")

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for idx, label in enumerate(labels):
        if fprs[idx] is not None and tprs[idx] is not None:
            plt.plot(fprs[idx], tprs[idx], label=f'{label} (AUC: {roc_auc_scores[idx]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    return macro_roc_auc


def train_audio_classifier(train_df, model_type='lightgbm'):
    # Example placeholder feature extraction
    X = extract_audio_features(train_df['file_path'])
    y = train_df['primary_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == 'lightgbm':
        model = lgb.LGBMClassifier(objective='multiclass', random_state=42)
        model.fit(X_train, y_train)
    else:
        raise ValueError("Currently only 'lightgbm' is supported")

    return model, X_test, y_test, y_train


# Train your model
model, X_test, y_test, y_train = train_audio_classifier(train, model_type='lightgbm')

# Extract labels for evaluation
labels = train['primary_label'].unique()

# Evaluate your trained model
macro_roc_auc = evaluate_model(model, X_test, y_test, labels, y_train)


# Prediction and Submission Function
def predict_and_submit(model, sample_submission_df, train, labels, n_mfcc=20):
    predictions = {}

    # Map file_id to file_path for efficiency
    file_id_to_path = {
        row['filename'].replace('.ogg', ''): row['file_path']
        for _, row in train.iterrows()
    }

    for _, row in sample_submission_df.iterrows():
        row_id = row['row_id']
        file_id = row_id.split('_')[1]
        audio_path = os.path.join(TEST_SOUNDSCAPES_PATH, f"{file_id}.ogg")

        try:
            mfccs = extract_mfcc_features(audio_path, n_mfcc=n_mfcc)

            if mfccs is None:
                prediction_probs = [0.01] * len(labels)
            else:
                prediction_probs = model.predict_proba([mfccs])[0]

            label_predictions = dict(zip(labels, prediction_probs))

        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: {e}, assigning default probabilities for {row_id}.")
            label_predictions = {label: 0.01 for label in labels}

        predictions[row_id] = label_predictions
        
    # Create and save submission DataFrame
    submission_df = pd.DataFrame.from_dict(predictions, orient='index')
    submission_df.index.name = 'row_id'
    submission_df.reset_index(inplace=True)

    submission_df.to_csv('submission.csv', index=False)
    print("Submission file created successfully!")

    return submission_df

# Generate the submission
submission_df = predict_and_submit(model, sample_submission_df, train, labels)

# Display submission file head
print("Submission File Preview:")
print(submission_df.head())

