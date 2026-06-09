!pip install mlflow


!pip install mutagen
!pip install contextily
!pip install silero_vad


!pip install sweetviz


######## Common ########
import pandas as pd
import os
import copy
import numpy as np
import copy
from collections import defaultdict, Counter

######## Data Processing ########
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

######## Sound ########
import librosa
from mutagen.oggvorbis import OggVorbis

# Human Voices Detection
from silero_vad import get_speech_timestamps, save_audio

######## Machine Learning ########
import torch
import torchaudio
import xgboost as xgb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
######## Visualization ########
# Ploting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# Geographical Maps
import geopandas as gpd
import contextily as ctx

######## Logging ########

import mlflow
import mlflow.sklearn
import tempfile


######## EDA ########
import sweetviz as sv





# Load CSV and TXT metadata
data_df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
taxonomy_df = pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')
submission_df = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
location_df = pd.read_csv('/kaggle/input/birdclef-2025/recording_location.txt', delimiter='\t')

# Quick overview
print("Train shape:", data_df.shape)
print("Taxonomy shape:", taxonomy_df.shape)
data_df.head()


# Generate SweetViz report for data_df
report_data = sv.analyze(data_df)
report_data.show_html('SweetViz_report_data_df.html')  # This will generate an HTML file

# Generate SweetViz report for taxonomy_df
report_taxonomy = sv.analyze(taxonomy_df)
report_taxonomy.show_html('SweetViz_report_taxonomy_df.html')


report_data.show_html('SweetViz_report_data_df.html')


data_df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
# Label Mapping
labels = data_df['primary_label'].unique()
# Sort for consistency
sorted_labels = sorted(labels)

# Creating dictionaries
label_to_common = taxonomy_df.set_index("primary_label")["common_name"].to_dict()
label_to_class = taxonomy_df.set_index("primary_label")["class_name"].to_dict()

print("Shape:", data_df.shape)
data_df.info()
data_df.head()



data_df.drop(columns=['author', 'license', 'url', 'latitude', 'longitude', 'type'],inplace=True)


data_df.info()


# Remove data with secondary labels
data_df = data_df[data_df['secondary_labels'] == "['']"]


# Apply dictionaries to create new columns
data_df["common_name"] = data_df["primary_label"].map(label_to_common)
data_df["animal_class"] = data_df["primary_label"].map(label_to_class)


data_df


report_data = sv.analyze(data_df)
report_data.show_html('SweetViz_report_data_df.html')  # This will generate an HTML file


num_of_top_most_common = 4
insecta_species = data_df[data_df["animal_class"] == "Insecta"]["common_name"].value_counts().nlargest(num_of_top_most_common).index
mammalia_species = data_df[data_df["animal_class"] == "Mammalia"]["common_name"].value_counts().nlargest(num_of_top_most_common).index
amphibia_species = data_df[data_df["animal_class"] == "Amphibia"]["common_name"].value_counts().nlargest(num_of_top_most_common).index
aves_species = data_df[data_df["animal_class"] == "Aves"]["common_name"].value_counts().nlargest(num_of_top_most_common).index

# Filter train_df to include only the top species from each animal class
filtered_df = data_df[
    (data_df["common_name"].isin(insecta_species)) |
    (data_df["common_name"].isin(mammalia_species)) |
    (data_df["common_name"].isin(amphibia_species)) |
    (data_df["common_name"].isin(aves_species))
]


max_samples_per_animal = 33

# Downsample each species to at most 33 samples
downsampled_df = filtered_df.groupby("common_name", group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), max_samples_per_animal), random_state=42)
).reset_index(drop=True)


# Compare original and cleaned
compare_report = sv.compare([data_df, "Original"], [downsampled_df, "Downsampled"])
compare_report.show_html("SweetViz_comparison.html")


def show_class_and_animal_balance(train_df):
    
    # Create dataframe with animal name, animal class, and count
    animal_counts_df = train_df.groupby(["common_name", "animal_class"]).size().reset_index(name="count")
    
    for c in ["Aves","Mammalia","Amphibia","Insecta"]:
        df = animal_counts_df[animal_counts_df['animal_class']==c]
    
        df.sort_values(by="count", ascending=True, inplace=True)
        
        # Plot horizontal bar chart
        plt.figure(figsize=(3, 2))  # Adjust height for better visualization
        ax = sns.barplot(x="count", y="common_name", data=df, hue="common_name", palette="viridis")
        if ax.legend_:
            ax.legend_.remove()
        
        
        # Add count numbers to each bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=5)
        
        # Adjust x-axis limit for spacing
        max_count = df["count"].max()
        plt.xlim(0, max_count * 1.2)  # Extend the max value by 15%
        
        plt.xlabel("Number of Recordings")
        plt.ylabel("common_name")
        plt.title(f"Number of Recordings per {c} Animal")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.show()

show_class_and_animal_balance(downsampled_df)


# Stratified train/validation/test split
def stratified_split(df, stratify_col, train_size=0.8, val_size=0.1, test_size=0.1):
    train_df, temp_df = train_test_split(df, train_size=train_size, stratify=df[stratify_col], random_state=42)
    val_df, test_df = train_test_split(temp_df, train_size=val_size / (val_size + test_size), stratify=temp_df[stratify_col], random_state=42)
    return train_df, val_df, test_df

train_df, val_df, test_df = stratified_split(downsampled_df, stratify_col=["animal_class", "primary_label"])


train_df


show_class_and_animal_balance(test_df)


# Compare original and cleaned
compare_report = sv.compare([train_df, "Train"], [test_df, "Test"])
compare_report.show_html("Train_test_split.html")
compare_report_val = sv.compare([train_df, "Train"], [val_df, "Validation"])
compare_report_val.show_html("Train_val_split.html")
compare_report_val_test = sv.compare([test_df, "Test"], [val_df, "Validation"])
compare_report_val_test.show_html("Test_val_split.html")


from tqdm import tqdm
import torchaudio

def seconds_to_mmss(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def extract_speech_timestamps_for_df(df, audio_root, threshold=0.17):
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, trust_repo=True)
    get_speech_timestamps, *_ = utils
    
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        file_path = os.path.join(audio_root, filename)

        try:
            wav, sr = torchaudio.load(file_path)
            raw_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr, threshold=threshold)

            # Convert samples to readable time (mm:ss)
            timestamps_mmss = []
            for segment in raw_timestamps:
                start_sec = segment['start'] / sr
                end_sec = segment['end'] / sr
                timestamps_mmss.append({
                    'start': seconds_to_mmss(start_sec),
                    'end': seconds_to_mmss(end_sec)
                })

            result = {
                'filename': filename,
                'speech_timestamps': timestamps_mmss
            }

            # Add all original metadata
            for col in df.columns:
                result[col] = row[col]

            results.append(result)

        except Exception as e:
            print(f"[ERROR] Could not process {filename}: {e}")
            continue

    return results



# Define the audio path on your local/Kaggle environment
audio_path = "/kaggle/input/birdclef-2025/train_audio"

# Extract timestamps for training set
speech_info_train = extract_speech_timestamps_for_df(train_df, audio_root=audio_path)

# Example output per item:
# {
#     'filename': '48124/CSA35116.ogg',
#     'speech_timestamps': [{'start': 1000, 'end': 32000}, ...],
#     'primary_label': 'compau',
#     ...
# }

# Convert to DataFrame if needed
speech_df_train = pd.DataFrame(speech_info_train)
speech_df_train.head()



# Extract timestamps for training set
speech_info_val = extract_speech_timestamps_for_df(val_df, audio_root=audio_path)

# Example output per item:
# {
#     'filename': '48124/CSA35116.ogg',
#     'speech_timestamps': [{'start': 1000, 'end': 32000}, ...],
#     'primary_label': 'compau',
#     ...
# }

# Convert to DataFrame if needed
speech_df_val = pd.DataFrame(speech_info_val)
speech_df_val.head()


# Extract timestamps for training set
speech_info_test = extract_speech_timestamps_for_df(test_df, audio_root=audio_path)

# Example output per item:
# {
#     'filename': '48124/CSA35116.ogg',
#     'speech_timestamps': [{'start': 1000, 'end': 32000}, ...],
#     'primary_label': 'compau',
#     ...
# }

# Convert to DataFrame if needed
speech_df_test = pd.DataFrame(speech_info_test)
speech_df_test.head()


import os
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm

def normalize_feature(X):
    return (X - X.min()) / (X.max() - X.min() + 1e-8)

def mmss_to_seconds(mmss: str):
    """Convert 'mm:ss' format to total seconds"""
    mins, secs = map(int, mmss.split(":"))
    return mins * 60 + secs

def extract_features_for_df_with_speech_removal(df, audio_root, chunk_duration=5, speech_df=None):
    all_rows = []

    # Convert speech_df to dictionary: filename â†’ list of timestamps
    speech_map = {}
    if speech_df is not None:
        for row in speech_df:
            speech_map[row['filename']] = row['speech_timestamps']

    for i, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        file_path = os.path.join(audio_root, filename)

        try:
            y, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        duration = len(y) / sr  # in seconds

        # === Remove speech if present ===
        if filename in speech_map:
            speech = speech_map[filename]
            if speech:  # Check non-empty list
                speech_starts = [mmss_to_seconds(seg['start']) for seg in speech]
                speech_ends = [mmss_to_seconds(seg['end']) for seg in speech]

                earliest = min(speech_starts)
                latest = max(speech_ends)

                if latest < duration / 2:
                    # Speech is at the beginning â†’ remove from start to ceil(latest/5)*5
                    trim_sec = int(np.ceil(latest / chunk_duration) * chunk_duration)
                    y = y[int(trim_sec * sr):]
                elif earliest > duration / 2:
                    # Speech is at the end â†’ remove from floor(earliest/5)*5 to end
                    trim_sec = int(np.floor(earliest / chunk_duration) * chunk_duration)
                    y = y[:int(trim_sec * sr)]
                else:
                    pass

        samples_per_chunk = int(sr * chunk_duration)
        n_chunks = int(len(y) / samples_per_chunk)

        if n_chunks == 0:
            pass


        for c in range(n_chunks):
            start = c * samples_per_chunk
            end = start + samples_per_chunk
            y_chunk = y[start:end]

            # === Extract Features ===
            mfcc = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=20)
            chroma = librosa.feature.chroma_stft(y=y_chunk, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y_chunk, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y_chunk)
            rms = librosa.feature.rms(y=y_chunk)
            centroid = librosa.feature.spectral_centroid(y=y_chunk, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=y_chunk, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y_chunk, sr=sr)

            # === Normalize ===
            mfcc = normalize_feature(mfcc)
            chroma = normalize_feature(chroma)
            contrast = normalize_feature(contrast)
            zcr = normalize_feature(zcr)
            rms = normalize_feature(rms)
            centroid = normalize_feature(centroid)
            bandwidth = normalize_feature(bandwidth)
            rolloff = normalize_feature(rolloff)

            # === Average features ===
            chunk_features = {
                f'mfcc_{i}': mfcc[i].mean() for i in range(mfcc.shape[0])
            }
            chunk_features.update({
                f'chroma_{i}': chroma[i].mean() for i in range(chroma.shape[0])
            })
            chunk_features.update({
                f'contrast_{i}': contrast[i].mean() for i in range(contrast.shape[0])
            })
            chunk_features.update({
                'zcr': zcr.mean(),
                'rms': rms.mean(),
                'centroid': centroid.mean(),
                'bandwidth': bandwidth.mean(),
                'rolloff': rolloff.mean(),
                'chunk_start_time': c * chunk_duration  # Add timestamp
            })

            for col in df.columns:
                chunk_features[col] = row[col]

            all_rows.append(chunk_features)

    return pd.DataFrame(all_rows)



train_features = extract_features_for_df_with_speech_removal(train_df, audio_root='/kaggle/input/birdclef-2025/train_audio', speech_df=speech_info_train)
val_features   = extract_features_for_df_with_speech_removal(val_df, audio_root='/kaggle/input/birdclef-2025/train_audio', speech_df=speech_info_val)
test_features  = extract_features_for_df_with_speech_removal(test_df, audio_root='/kaggle/input/birdclef-2025/train_audio', speech_df=speech_info_test)


train_features.head()



test_features


speech_df_train.to_csv("speech_df_train.csv", index=False)
speech_df_val.to_csv("speech_df_val.csv", index=False)
speech_df_test.to_csv("speech_df_test.csv", index=False)



taxonomy_df = pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')
print("Shape:", taxonomy_df.shape)
taxonomy_df.info()
taxonomy_df.head()



def prepare_xgb_data(train_df, val_df):
    drop_cols = ['filename', 'primary_label', 'common_name', 'scientific_name', 'collection', 'secondary_labels', 'rating']
    features = [col for col in train_df.columns if col not in drop_cols + ['animal_class']]
    
    le = LabelEncoder()
    train_labels = le.fit_transform(train_df['animal_class'])
    val_labels = le.transform(val_df['animal_class'])
    
    train_X = train_df[features]
    val_X = val_df[features]
    
    return train_X, train_labels, val_X, val_labels, le

train_X, train_y, val_X, val_y, label_encoder = prepare_xgb_data(train_features, val_features)

# MLflow Experiment Logging
with mlflow.start_run(run_name="XGBoost_BirdCLEF"):
    # Log parameters
    mlflow.log_param("num_class", 4)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)

    # Train model
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    model.fit(train_X, train_y)

    # Validation evaluation
    val_preds = model.predict(val_X)
    val_acc = accuracy_score(val_y, val_preds)

    print("Accuracy:", val_acc)
    print("Classification Report:\n", classification_report(val_y, val_preds, target_names=label_encoder.classes_))

    # Log accuracy
    mlflow.log_metric("val_accuracy", val_acc)

    # Log confusion matrix
    cm = confusion_matrix(val_y, val_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name)
        mlflow.log_artifact(tmp.name, artifact_path="val_confusion_matrix")

    # Log model
    mlflow.sklearn.log_model(model, "xgb_model")


def prepare_test_data(test_df, label_encoder, features):
    test_X = test_df[features]
    test_y = label_encoder.transform(test_df['animal_class'])
    return test_X, test_y

used_features = train_X.columns.tolist()
test_X, test_y = prepare_test_data(test_features, label_encoder, used_features)
test_preds = model.predict(test_X)

test_acc = accuracy_score(test_y, test_preds)
print("âœ… Test Set Accuracy:", test_acc)
print("âœ… Test Set Classification Report:\n", classification_report(test_y, test_preds, target_names=label_encoder.classes_))
mlflow.log_metric("test_accuracy", test_acc)

# Test confusion matrix
cm = confusion_matrix(test_y, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.title("Test Set Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    plt.savefig(tmp.name)
    mlflow.log_artifact(tmp.name, artifact_path="test_confusion_matrix")


# -----------------------------
# File-Level Aggregation
# -----------------------------
test_preds_proba = model.predict_proba(test_X)
test_features['pred'] = test_preds
test_features['true'] = test_y

file_votes = defaultdict(list)
file_true = {}

for _, row in test_features.iterrows():
    fname = row['filename']
    file_votes[fname].append(row['pred'])
    file_true[fname] = row['true']

final_preds, final_labels = [], []

for fname, preds in file_votes.items():
    vote = Counter(preds).most_common(1)[0][0]
    final_preds.append(vote)
    final_labels.append(file_true[fname])

file_level_acc = accuracy_score(final_labels, final_preds)
print("âœ… File-level Accuracy (Majority Vote):", file_level_acc)
print("âœ… File-level Classification Report:\n", classification_report(final_labels, final_preds, target_names=label_encoder.classes_))

mlflow.log_metric("file_level_accuracy", file_level_acc)

# File-level confusion matrix
cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("File-level Confusion Matrix (XGBoost)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    plt.savefig(tmp.name)
    mlflow.log_artifact(tmp.name, artifact_path="file_level_confusion_matrix")


import os
import librosa
import numpy as np
import cv2
from tqdm import tqdm

def mmss_to_seconds(mmss: str):
    mins, secs = map(int, mmss.split(":"))
    return mins * 60 + secs

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def resize_feat(feat, target_shape):
    from cv2 import resize, INTER_LINEAR
    return resize(feat, target_shape, interpolation=INTER_LINEAR)

def extract_rgb_features(y, sr, img_size=(224, 224)):
    try:
        # Compute features
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        delta = librosa.feature.delta(log_mel)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        # Resize features
        log_mel_resized = resize_feat(log_mel, img_size)
        delta_resized = resize_feat(delta, img_size)
        chroma_resized = resize_feat(chroma, img_size)

        # Normalize
        log_mel_norm = normalize(log_mel_resized)
        delta_norm = normalize(delta_resized)
        chroma_norm = normalize(chroma_resized)

        # Stack to form RGB image
        rgb_image = np.stack([log_mel_norm, delta_norm, chroma_norm], axis=-1)
        rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

        return rgb_image_uint8

    except Exception as e:
        print(f"[ERROR in feature extraction]: {e}")
        return None

def save_chunked_rgb_images_trimmed(df, speech_info, audio_root, output_dir, chunk_duration=5, img_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    speech_map = {entry['filename']: entry['speech_timestamps'] for entry in speech_info}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        animal_class = row['animal_class']
        input_path = os.path.join(audio_root, filename)

        try:
            y, sr = librosa.load(input_path, sr=None)
        except Exception as e:
            print(f"[ERROR] loading {filename}: {e}")
            continue

        duration_sec = len(y) / sr

        # Trim human speech if found
        if filename in speech_map:
            speech = speech_map[filename]
            if speech:
                speech_starts = [mmss_to_seconds(seg['start']) for seg in speech]
                speech_ends = [mmss_to_seconds(seg['end']) for seg in speech]
                earliest = min(speech_starts)
                latest = max(speech_ends)

                if latest < duration_sec / 2:
                    trim_sec = int(np.ceil(latest / chunk_duration) * chunk_duration)
                    y = y[int(trim_sec * sr):]
                elif earliest > duration_sec / 2:
                    trim_sec = int(np.floor(earliest / chunk_duration) * chunk_duration)
                    y = y[:int(trim_sec * sr)]
                else:
                    # Mid-speech â†’ skip
                    pass

        # 5-second chunking
        samples_per_chunk = int(sr * chunk_duration)
        n_chunks = int(len(y) / samples_per_chunk)
        if n_chunks == 0:
            pass

        base_name = os.path.splitext(os.path.basename(filename))[0]
        class_dir = os.path.join(output_dir, animal_class)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(n_chunks):
            start = i * samples_per_chunk
            end = start + samples_per_chunk
            chunk = y[start:end]

            rgb_img = extract_rgb_features(chunk, sr, img_size=img_size)
            if rgb_img is None:
                continue

            out_name = f"{base_name}_clip_{i}.png"
            out_path = os.path.join(class_dir, out_name)

            try:
                cv2.imwrite(out_path, rgb_img)
            except Exception as e:
                print(f"[ERROR] saving {out_path}: {e}")



save_chunked_rgb_images_trimmed(train_df, speech_info_train, audio_root='/kaggle/input/birdclef-2025/train_audio', output_dir='/kaggle/working/spectrogram_chunks/train_chunked')
save_chunked_rgb_images_trimmed(val_df, speech_info_val, audio_root='/kaggle/input/birdclef-2025/train_audio', output_dir='/kaggle/working/spectrogram_chunks/val_chunked')
save_chunked_rgb_images_trimmed(test_df, speech_info_test, audio_root='/kaggle/input/birdclef-2025/train_audio', output_dir='/kaggle/working/spectrogram_chunks/test_chunked')



import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Transforms
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Datasets
train_ds = ImageFolder("/kaggle/working/spectrogram_chunks/train_chunked", transform=transform)
val_ds   = ImageFolder("/kaggle/working/spectrogram_chunks/val_chunked", transform=transform)
test_ds  = ImageFolder("/kaggle/working/spectrogram_chunks/test_chunked", transform=transform)

# Dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)



import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
class RegNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # RegNet setup
        self.model = models.regnet_y_400mf(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # For manual tracking
        self.val_losses = []
        self.val_accuracies = []
        self._val_loss_batches = []
        self._val_acc_batches = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Save to temporary lists for epoch-end aggregation
        self._val_loss_batches.append(loss)
        self._val_acc_batches.append(acc)

        # Log per batch for trainer bar
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        # Average batch metrics for this epoch
        avg_loss = torch.stack(self._val_loss_batches).mean().item()
        avg_acc = torch.stack(self._val_acc_batches).mean().item()

        # Store for plotting
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        # Log epoch-level values
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", avg_acc, prog_bar=True)

        # Clear lists for next epoch
        self._val_loss_batches.clear()
        self._val_acc_batches.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.8)
        return [optimizer], [scheduler]




from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

mlf_logger = MLFlowLogger(
    experiment_name="BirdCLEF-RegNet",
    tracking_uri="file:/kaggle/working/mlruns"  # or another location if remote
)

checkpoint = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)

model = RegNetClassifier(num_classes=4, lr=1e-3)

trainer = Trainer(
    max_epochs=10,
    accelerator="auto",
    callbacks=[checkpoint],
    logger=mlf_logger
)

trainer.fit(model, train_loader, val_loader)



# Convert to arrays for smooth plotting
val_losses = model.val_losses
val_accuracies = model.val_accuracies
val_losses = model.val_losses[1:]
val_accuracies = model.val_accuracies[1:]

epochs = range(1, len(val_losses) + 1)

plt.figure(figsize=(12, 5))

# Plot Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, val_losses, 'o-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss over Epochs')
plt.grid(True)
plt.legend()

# Plot Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies, 'o-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


mlflow.end_run()


import mlflow
import tempfile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.stats import mode

with mlflow.start_run(run_name="RegNet_Eval"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(model.device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=test_ds.classes)

    print("âœ… Test Accuracy:", acc)
    print(report)

    mlflow.log_metric("test_accuracy", acc)

    # Log confusion matrix image
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_ds.classes, yticklabels=test_ds.classes, cmap="Greens")
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name)
        mlflow.log_artifact(tmp.name, artifact_path="test_confusion_matrix")


file_probs_model = defaultdict(list)
file_targets = {}
all_image_paths = [s[0] for s in test_loader.dataset.samples]
sample_idx = 0

with torch.no_grad():
    for batch, labels in test_loader:
        batch = batch.to(model.device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        for j in range(len(batch)):
            img_path = all_image_paths[sample_idx]
            true_class = test_loader.dataset.samples[sample_idx][1]
            base_filename = os.path.basename(img_path).split("_clip")[0]

            file_probs_model[base_filename].append(probs[j])
            file_targets[base_filename] = true_class
            sample_idx += 1

final_preds, final_labels = [], []

for fname, prob_list in file_probs_model.items():
    votes = [np.argmax(p) for p in prob_list]
    pred = mode(votes, keepdims=True).mode[0]
    final_preds.append(pred)
    final_labels.append(file_targets[fname])

file_acc = accuracy_score(final_labels, final_preds)
print("âœ… File-level Accuracy:", file_acc)
print("âœ… File-level Classification Report:")
print(classification_report(final_labels, final_preds, target_names=test_ds.classes))

mlflow.log_metric("file_level_accuracy", file_acc)

# File-level confusion matrix
cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_ds.classes,
            yticklabels=test_ds.classes)
plt.title("File-level Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    plt.savefig(tmp.name)
    mlflow.log_artifact(tmp.name, artifact_path="file_level_confusion_matrix")


mlflow.end_run()


import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class EfficientNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained EfficientNet 
        self.model = models.efficientnet_b3(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        # Manual metric tracking
        self.val_losses = []
        self.val_accuracies = []
        self._val_loss_batches = []
        self._val_acc_batches = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Store per batch
        self._val_loss_batches.append(loss)
        self._val_acc_batches.append(acc)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self._val_loss_batches).mean().item()
        avg_acc = torch.stack(self._val_acc_batches).mean().item()

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", avg_acc, prog_bar=True)

        self._val_loss_batches.clear()
        self._val_acc_batches.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.8)
        return [optimizer], [scheduler]




from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

mlf_logger = MLFlowLogger(
    experiment_name="BirdCLEF-EfficientNet",
    tracking_uri="file:/kaggle/working/mlruns"  # or another location if remote
)

checkpoint = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)

eff_net_model = EfficientNetClassifier(num_classes=4, lr=1e-3)

trainer = Trainer(
    max_epochs=10,
    accelerator="auto",
    callbacks=[checkpoint],
    logger=mlf_logger
)

trainer.fit(eff_net_model, train_loader, val_loader)



# Convert to arrays for smooth plotting
val_losses = eff_net_model.val_losses
val_accuracies = eff_net_model.val_accuracies
val_losses = eff_net_model.val_losses[1:]
val_accuracies = eff_net_model.val_accuracies[1:]

epochs = range(1, len(val_losses) + 1)

plt.figure(figsize=(12, 5))

# Plot Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, val_losses, 'o-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss over Epochs')
plt.grid(True)
plt.legend()

# Plot Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies, 'o-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


import mlflow
import tempfile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.stats import mode

with mlflow.start_run(run_name="EfficientNet_Eval"):
    eff_net_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(eff_net_model.device)
            logits = eff_net_model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=test_ds.classes)

    print("âœ… Test Accuracy:", acc)
    print(report)

    mlflow.log_metric("test_accuracy", acc)

    # Log confusion matrix image
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_ds.classes, yticklabels=test_ds.classes, cmap="Greens")
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name)
        mlflow.log_artifact(tmp.name, artifact_path="test_confusion_matrix")


file_probs_eff = defaultdict(list)
file_targets = {}
all_image_paths = [s[0] for s in test_loader.dataset.samples]
sample_idx = 0

with torch.no_grad():
    for batch, labels in test_loader:
        batch = batch.to(eff_net_model.device)
        logits = eff_net_model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        for j in range(len(batch)):
            img_path = all_image_paths[sample_idx]
            true_class = test_loader.dataset.samples[sample_idx][1]
            base_filename = os.path.basename(img_path).split("_clip")[0]

            file_probs_eff[base_filename].append(probs[j])
            file_targets[base_filename] = true_class
            sample_idx += 1

final_preds, final_labels = [], []

for fname, prob_list in file_probs_eff.items():
    votes = [np.argmax(p) for p in prob_list]
    pred = mode(votes, keepdims=True).mode[0]
    final_preds.append(pred)
    final_labels.append(file_targets[fname])

file_acc = accuracy_score(final_labels, final_preds)
print("âœ… File-level Accuracy:", file_acc)
print("âœ… File-level Classification Report:")
print(classification_report(final_labels, final_preds, target_names=test_ds.classes))

mlflow.log_metric("file_level_accuracy", file_acc)

# File-level confusion matrix
cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_ds.classes,
            yticklabels=test_ds.classes)
plt.title("File-level Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    plt.savefig(tmp.name)
    mlflow.log_artifact(tmp.name, artifact_path="file_level_confusion_matrix")


mlflow.end_run()


import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class SpectrogramContextDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.filenames = []

        for class_name in sorted(os.listdir(image_dir)):
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file in sorted(os.listdir(class_dir)):  # ensures chronological chunk order
                if file.endswith('.png'):
                    self.image_paths.append(os.path.join(class_dir, file))
                    self.labels.append(class_name)
                    self.filenames.append(file.split("_clip")[0])  # e.g., iNat1122209

        self.classes = sorted(set(self.labels))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.samples = list(zip(self.image_paths, self.labels))

    def __len__(self):
        return len(self.image_paths)

    def _load_half(self, path, side='left'):
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        h, w, c = img.shape
        if side == 'left':
            half = img[:, :w // 2, :]
        else:
            half = img[:, w // 2:, :]
        return half  # shape: (224, 112, 3)

    def __getitem__(self, idx):
        center_path = self.image_paths[idx]
        center_img = Image.open(center_path).convert("RGB")
        center_arr = np.array(center_img)  # shape: (224, 224, 3)
        fname = self.filenames[idx]

        # Default paddings
        left_half = np.zeros((224, 112, 3), dtype=np.uint8)
        right_half = np.zeros((224, 112, 3), dtype=np.uint8)

        # LEFT: prev image from same file
        if idx > 0 and self.filenames[idx - 1] == fname:
            left_half = self._load_half(self.image_paths[idx - 1], side='right')

        # RIGHT: next image from same file
        if idx < len(self.image_paths) - 1 and self.filenames[idx + 1] == fname:
            right_half = self._load_half(self.image_paths[idx + 1], side='left')

        # Combine horizontally: (224, 448, 3)
        full_img = np.concatenate([left_half, center_arr, right_half], axis=1)
        full_img = Image.fromarray(full_img.astype(np.uint8))

        if self.transform:
            full_img = self.transform(full_img)

        label = self.class_to_idx[self.labels[idx]]
        return full_img, label



from torchvision import transforms
from torch.utils.data import DataLoader

# Path to your context-aware spectrogram folders
train_dir = "/kaggle/working/spectrogram_chunks/train_chunked"
val_dir   = "/kaggle/working/spectrogram_chunks/val_chunked"
test_dir  = "/kaggle/working/spectrogram_chunks/test_chunked"

# Shared transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Context-aware datasets
train_ds = SpectrogramContextDataset(train_dir, transform=transform)
val_ds   = SpectrogramContextDataset(val_dir, transform=transform)
test_ds  = SpectrogramContextDataset(test_dir, transform=transform)

# Dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)



import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class RegNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Load RegNet with wide support
        self.model = models.regnet_y_400mf(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        # Manual metric tracking
        self.val_losses = []
        self.val_accuracies = []
        self._val_loss_batches = []
        self._val_acc_batches = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self._val_loss_batches.append(loss)
        self._val_acc_batches.append(acc)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self._val_loss_batches).mean().item()
        avg_acc = torch.stack(self._val_acc_batches).mean().item()

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", avg_acc, prog_bar=True)

        self._val_loss_batches.clear()
        self._val_acc_batches.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.8)
        return [optimizer], [scheduler]




from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_cb = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)

model_regnet_extra = RegNetClassifier(num_classes=4)

trainer = Trainer(
    max_epochs=10,
    accelerator="auto",
    callbacks=[checkpoint_cb]
)

trainer.fit(model_regnet_extra, train_loader, val_loader)



# Convert to arrays for smooth plotting
val_losses = model_regnet_extra.val_losses
val_accuracies = model_regnet_extra.val_accuracies
val_losses = model_regnet_extra.val_losses[1:]
val_accuracies = model_regnet_extra.val_accuracies[1:]

epochs = range(1, len(val_losses) + 1)

plt.figure(figsize=(12, 5))

# Plot Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, val_losses, 'o-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss over Epochs')
plt.grid(True)
plt.legend()

# Plot Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies, 'o-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


import mlflow
import tempfile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.stats import mode

with mlflow.start_run(run_name="RegNet_Eval_extra_context"):
    model_regnet_extra.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(model_regnet_extra.device)
            logits = model_regnet_extra(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=test_ds.classes)

    print("âœ… Test Accuracy:", acc)
    print(report)

    mlflow.log_metric("test_accuracy", acc)

    # Log confusion matrix image
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_ds.classes, yticklabels=test_ds.classes, cmap="Greens")
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name)
        mlflow.log_artifact(tmp.name, artifact_path="test_confusion_matrix")


# Map of RegNet predictions per file
file_probs_extra = defaultdict(list)
file_targets = {}

# Get filepaths in order
all_image_paths = [s[0] for s in test_loader.dataset.samples]
sample_idx = 0

# Mapping from class index to label
idx_to_class = {i: c for i, c in enumerate(test_ds.classes)}

with torch.no_grad():
    for batch, labels in test_loader:
        batch = batch.to(model_regnet_extra.device)
        logits = model_regnet_extra(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        for j in range(len(batch)):
            img_path = all_image_paths[sample_idx]
            class_idx = test_loader.dataset.samples[sample_idx][1]  # already integer
            fname = os.path.basename(img_path).split("_clip")[0]

            file_probs_extra[fname].append(probs[j])
            file_targets[fname] = class_idx  # always integer
            sample_idx += 1

# Aggregate per file
final_preds, final_labels = [], []

for fname, prob_list in file_probs_extra.items():
    vote_indices = [np.argmax(p) for p in prob_list]
    majority_vote = mode(vote_indices, keepdims=True).mode[0]
    final_preds.append(majority_vote)
    final_labels.append(file_targets[fname])

# Accuracy
file_acc = accuracy_score(final_labels, final_preds)
print("âœ… File-level Accuracy:", file_acc)

# Classification Report
if isinstance(final_labels[0], str):
    class_to_idx = test_ds.class_to_idx
    final_labels = [class_to_idx[l] for l in final_labels]

if isinstance(final_preds[0], str):
    class_to_idx = test_ds.class_to_idx
    final_preds = [class_to_idx[p] for p in final_preds]

print("âœ… File-level Classification Report:")
print(classification_report(final_labels, final_preds, target_names=test_ds.classes))

mlflow.log_metric("file_level_accuracy", file_acc)

# Confusion Matrix
cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_ds.classes,
            yticklabels=test_ds.classes)
plt.title("File-level Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

# Log CM to MLflow
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    plt.savefig(tmp.name)
    mlflow.log_artifact(tmp.name, artifact_path="file_level_confusion_matrix")


from collections import defaultdict, Counter
import numpy as np
from scipy.stats import mode

# Final maps from previous models:
# file_probs_model, file_probs_eff, file_probs_extra
# file_targets: {filename: int}

# Voting function across models
final_preds, final_labels = [], []

for fname in file_targets.keys():
    # Gather predictions from all 3 models
    preds_model = [np.argmax(p) for p in file_probs_model[fname]]
    preds_eff   = [np.argmax(p) for p in file_probs_eff[fname]]
    preds_extra = [np.argmax(p) for p in file_probs_extra[fname]]

    # Majority vote per chunk list
    vote = Counter(preds_model + preds_eff + preds_extra).most_common(1)[0][0]
    
    final_preds.append(vote)
    final_labels.append(file_targets[fname])



from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Create and fit label encoder using true class names
label_encoder = LabelEncoder()
label_encoder.fit(final_labels)  # uses ['Aves', 'Mammalia', ...]

# Convert int predictions â†’ string predictions using inverse_transform
final_preds_str = label_encoder.inverse_transform(final_preds)

# Now both predictions and labels are strings
acc = accuracy_score(final_labels, final_preds_str)
print("âœ… Ensemble File-Level Accuracy:", acc)

# Classification report
print(classification_report(final_labels, final_preds_str))

# Confusion matrix (normalized optional)
cm = confusion_matrix(final_labels, final_preds_str, labels=label_encoder.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cmap='Blues')
plt.title("Ensemble Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()


