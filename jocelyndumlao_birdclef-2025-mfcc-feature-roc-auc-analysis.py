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

print("Train Meta Shape:", train_meta.shape)


print("Train Meta Head:")
train_meta.head().style.background_gradient(cmap='YlOrBr')


print("Taxonomy Head:")
taxonomy.head().style.background_gradient(cmap='plasma')


print("Recording Locations Head:")
recording_locations.head().style.background_gradient(cmap='plasma')


# Audio Visualization (Train Data Examples)
def visualize_audio(file_paths, titles):
    """Visualizes audio waveforms and spectrograms."""
    plt.figure(figsize=(15, 4 * len(file_paths)))
    for i, file_path in enumerate(file_paths):
        try:
            y, sr = librosa.load(file_path)
            plt.subplot(len(file_paths), 2, 2 * i + 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title(f'Waveform: {titles[i]}')

            plt.subplot(len(file_paths), 2, 2 * i + 2)
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
            plt.title(f'Spectrogram: {titles[i]}')
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    plt.tight_layout()
    plt.show()

# Select 4 audio examples
audio_examples = train_meta.sample(4)
file_paths = audio_examples['file_path'].tolist()
titles = audio_examples['primary_label'].tolist()

print("\nVisualizing Train Audio Examples:")
visualize_audio(file_paths, titles)

# Soundscapes Visualization
def visualize_soundscape(file_paths, titles):
    """Visualizes soundscape audio waveforms and spectrograms."""
    plt.figure(figsize=(15, 4 * len(file_paths)))
    for i, file_path in enumerate(file_paths):
        try:
            y, sr = librosa.load(file_path)
            plt.subplot(len(file_paths), 2, 2 * i + 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title(f'Soundscape Waveform: {titles[i]}')

            plt.subplot(len(file_paths), 2, 2 * i + 2)
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
            plt.title(f'Soundscape Spectrogram: {titles[i]}')
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    plt.tight_layout()
    plt.show()

# Select 4 diverse soundscape examples
def get_available_soundscapes(soundscapes_path):
    """Returns a list of available soundscape files."""
    try:
        soundscape_files = [f for f in os.listdir(soundscapes_path) if f.endswith('.ogg')]
        soundscape_files = [os.path.join(soundscapes_path, f) for f in soundscape_files]
        return soundscape_files
    except FileNotFoundError:
        print(f"Error: The directory {soundscapes_path} was not found.")
        return []
    except Exception as e:
        print(f"Error listing soundscape files: {e}")
        return []

available_soundscapes = get_available_soundscapes(TRAIN_SOUNDSCAPES_PATH)

if len(available_soundscapes) >= 4:
    soundscape_files = available_soundscapes[:4]  # Select the first 4
    soundscape_titles = [f'Soundscape {i+1}' for i in range(4)]

    print("\nVisualizing Soundscapes:")
    visualize_soundscape(soundscape_files, soundscape_titles)
else:
    print("\nNot enough soundscapes available to visualize 4 examples.")

# Recording Location Map
import folium

def create_location_map(train_meta, n_samples=200):
    """Creates an interactive map of recording locations using Folium."""
    location_df = train_meta[['latitude', 'longitude']].dropna().sample(n_samples)
    latitudes = location_df['latitude'].values.tolist()
    longitudes = location_df['longitude'].values.tolist()

    # Calculate the average location to center the map
    avg_lat = sum(latitudes) / len(latitudes)
    avg_lon = sum(longitudes) / len(longitudes)

    # Create the map
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6)

    # Add markers for each location
    for lat, lon in zip(latitudes, longitudes):
        folium.CircleMarker(location=[lat, lon], radius=5, color='blue', fill=True, fill_color='blue').add_to(m)

    return m

print("\nCreating Recording Location Map:")
map_obj = create_location_map(train_meta)
map_obj  # Display the map

# Taxonomy Visualization
def plot_taxonomy_distribution(taxonomy, column_name, top_n=10):
    """Plots the distribution of taxonomy categories."""
    plt.figure(figsize=(12, 6))
    counts = taxonomy[column_name].value_counts().nlargest(top_n)
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title(f'Top {top_n} {column_name} Distribution')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

print("\nVisualizing Taxonomy Distributions:")
plot_taxonomy_distribution(taxonomy, 'class_name')
plot_taxonomy_distribution(taxonomy, 'common_name')


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



# Macro-averaged ROC-AUC Evaluation
def evaluate_model(model, X_test, y_test, labels, y_train):
    """Evaluates the model using macro-averaged ROC-AUC."""

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)

    # Binarize the true labels (fitting on both train and test labels to ensure all classes are present)
    mlb = MultiLabelBinarizer(classes=labels)
    mlb.fit([[label] for label in np.concatenate([y_train, y_test])])
    y_test_bin = mlb.transform([[label] for label in y_test])

    # Calculate ROC AUC for each class
    roc_auc_scores = []
    fprs, tprs, thresholds = [], [], []  # Store ROC curve data

    for i, label in enumerate(labels):
        try:
            # Get the index of the label in the MultiLabelBinarizer's classes_
            label_index = list(mlb.classes_).index(label)

            # Check if there's only one class present in the true labels
            if len(np.unique(y_test_bin[:, label_index])) <= 1:
                raise ValueError(f"Only one class present in y_true for label {label}. ROC AUC score is not defined in that case.")

            # Calculate ROC AUC
            roc_auc = roc_auc_score(y_test_bin[:, label_index], y_pred_proba[:, list(mlb.classes_).index(label)])
            roc_auc_scores.append(roc_auc)

            # Calculate ROC curve
            fpr, tpr, threshold = roc_curve(y_test_bin[:, label_index], y_pred_proba[:, list(mlb.classes_).index(label)])
            fprs.append(fpr)
            tprs.append(tpr)
            thresholds.append(threshold)

        except ValueError as e:
            #Remove this line if you want to see the error messages
            #print(f"ValueError for label {label}: {e}")  # Added error message
            roc_auc_scores.append(np.nan)
            fprs.append(None)
            tprs.append(None)
            thresholds.append(None)

        except IndexError as e:
            #Remove this line if you want to see the error messages
            #print(f"IndexError for label {label}: {e}")
            roc_auc_scores.append(np.nan)
            fprs.append(None)
            tprs.append(None)
            thresholds.append(None)

    # Calculate the macro-averaged ROC AUC, ignoring NaN values
    macro_roc_auc = np.nanmean(roc_auc_scores)
    print(f"Macro-Averaged ROC AUC: {macro_roc_auc:.4f}")

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        if fprs[i] is not None and tprs[i] is not None:
            plt.plot(fprs[i], tprs[i], label=f'{label} (AUC = {roc_auc_scores[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc='lower right')
    plt.show()

    return macro_roc_auc

# Prepare labels for MultiLabelBinarizer
labels = train_meta['primary_label'].unique()

# Evaluate the model
macro_roc_auc = evaluate_model(model, X_test, y_test, labels, y_train)



# Prediction and Submission
def predict_and_submit(model, sample_submission, train_meta, labels, n_mfcc=20):
    """Predicts bird presence and creates a submission file."""
    predictions = {}  # Store predictions

    # Create a mapping from file_id to audio file path in train_meta
    file_id_to_path = {row['filename'].replace('.ogg', ''): row['file_path'] for _, row in train_meta.iterrows()}

    for index, row in sample_submission.iterrows():
        try:
            # Extract file_id from row_id (required format is soundscape_{file_id}_{time})
            file_id = row['row_id'].split('_')[1]

            # Construct the audio path using file_id and the mapping
            audio_path = os.path.join(TEST_SOUNDSCAPES_PATH, file_id + '.ogg')

            # Or if the audio file is available in the train_meta data
            # audio_path = file_id_to_path.get(file_id, None)  # Try to fetch from train_meta
            # if audio_path is None:
            #     audio_path = os.path.join(TEST_SOUNDSCAPES_PATH, file_id + '.ogg')  # Fallback to TEST_SOUNDSCAPES

            mfccs = extract_mfcc(audio_path, n_mfcc=n_mfcc)

            if mfccs is None:
                # Handle missing or corrupted audio files
                prediction_values = [0.01] * len(labels)  # Give small default probability
            else:
                # Get the predicted probabilities for each class
                prediction_values = model.predict_proba(mfccs.reshape(1, -1))[0]  # Predict probabilities

            # Create a dictionary mapping labels to prediction probabilities
            label_predictions = dict(zip(labels, prediction_values))
            predictions[row['row_id']] = label_predictions

        except KeyError as e:
            print(f"KeyError in predict_and_submit: {e}. Skipping row {row['row_id']}")
            # Create a dictionary with default probabilities for all labels
            label_predictions = {label: 0.01 for label in labels} # small probability to all labels
            predictions[row['row_id']] = label_predictions
        except FileNotFoundError as e:
            print(f"FileNotFoundError in predict_and_submit: {e}. Skipping row {row['row_id']}")
            # Create a dictionary with default probabilities for all labels
            label_predictions = {label: 0.01 for label in labels} # small probability to all labels
            predictions[row['row_id']] = label_predictions


    # Create submission DataFrame
    submission_data = []
    for row_id, label_predictions in predictions.items():
        row_data = {'row_id': row_id}
        row_data.update(label_predictions)
        submission_data.append(row_data)

    submission_df = pd.DataFrame(submission_data)
    submission_df = submission_df.set_index('row_id')

    # Ensure that the order of columns in the submission matches the sample_submission
    cols = sample_submission.columns[1:]
    submission_df = submission_df[cols]

    submission_df.to_csv('submission.csv')
    print("Submission file created successfully!")
    return submission_df

# Predict and submit
submission_df = predict_and_submit(model, sample_submission, train_meta, labels)

# Print the head of the created submission file
submission = pd.read_csv('submission.csv')
print("Submission File Head:")
submission.head()


import os
import re
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = 'iframe'

# Define Paths 
INPUT_PATH = './' 
TRAIN_AUDIO_PATH = './'  

# Load the training metadata
train_meta = pd.read_csv(os.path.join(INPUT_PATH, '/kaggle/input/birdclef-2025/train.csv'))


# Data Preprocessing
def preprocess_train_meta(df):
    """Preprocesses the training metadata."""
    df['secondary_labels'] = df['secondary_labels'].apply(lambda x: re.findall(r"'(\w+)'", x))
    df['len_sec_labels'] = df['secondary_labels'].map(len)
    df['file_path'] = df.apply(lambda row: os.path.join(TRAIN_AUDIO_PATH, row['filename']), axis=1)
    return df


train_meta = preprocess_train_meta(train_meta)

# Data aggregation and preparation
train_meta_grouped = train_meta.groupby(['primary_label', 'latitude', 'longitude']).count().reset_index()[
    ['primary_label', 'scientific_name', 'latitude', 'longitude']].rename(columns={'scientific_name': 'count'})


df = train_meta.merge(train_meta_grouped, on=['primary_label', 'latitude', 'longitude'], how='left').dropna(
    subset=['count'])
df['count'] = df['count'].astype('int')

values_list = df['count'].values.tolist()

# Radius scaling using interpolation
interpolation = interp1d([1, max(values_list)], [3, 20])  # Adjust range [min_radius, max_radius] as needed
radius = interpolation(values_list)

# Color scale selection (using Plotly's built-in options for better aesthetics)
color_scale = "Rainbow"  # "Hot", "Viridis", "Plasma", "Cividis", "Rainbow"

# Densitymapbox plot with enhanced aesthetics
fig = go.Figure(go.Densitymapbox(
    lat=df['latitude'],
    lon=df['longitude'],
    z=df['count'],
    radius=radius,
    colorscale=color_scale,  # Use chosen colorscale
    zmin=min(df['count']),  # Explicitly set zmin and zmax for consistent color mapping
    zmax=max(df['count']),
    opacity=0.7,  # Adjust opacity for better visualization
    colorbar=dict(title="Observation Count")  # Add a title to the colorbar
))

# Map layout customization
fig.update_layout(
    title="Geographic Distribution of Primary Labels",  # Adding a descriptive title
    title_x=0.5,  # Center the title
    mapbox_style="carto-positron",  # Choose a suitable basemap style: 'open-street-map', 'carto-positron', 'stamen-terrain', 'white-bg'
    height=800,
    mapbox={
        'center': {'lat': df['latitude'].mean(), 'lon': df['longitude'].mean()},  # Center on the data
        'zoom': 2,  # Adjust initial zoom level
        'accesstoken': "pk.eyJ1IjoiZXJpY21mYXJsaW5nIiwiYSI6ImNqOHB2eTMxOTAza2EzMm1xeDFiaG9zNnoifQ.xLWC_nwyxG2WbNlWr33oIA"
    },
    margin={"r": 0, "t": 50, "l": 0, "b": 0},  # Adjust margins for better appearance
    template="plotly_white"  # Use a clean template
)

fig.show()




