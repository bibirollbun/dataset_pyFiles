# General
import os
import gc
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import ast
import time
import json

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display

# Audio processing
import librosa
import soundfile as sf

# Machine Learning / Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Model and Optimizer
import timm
from torch.optim import Adam

# Metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Set display options
pd.set_option('display.max_columns', 100)


class CFG:
    # Paths
    TRAIN_AUDIO_DIR = "/kaggle/input/birdclef-2025/train_audio"
    TRAIN_SOUNDSCAPE_DIR = "/kaggle/input/birdclef-2025/train_soundscapes"
    TEST_SOUNDSCAPE_DIR = "/kaggle/input/birdclef-2025/test_soundscapes"
    TRAIN_CSV = "/kaggle/input/birdclef-2025/train.csv"
    TAXONOMY_CSV = "/kaggle/input/birdclef-2025/taxonomy.csv"
    SAMPLE_SUBMISSION_CSV = "/kaggle/input/birdclef-2025/sample_submission.csv"
    RECORDING_LOCATION = "/kaggle/input/birdclef-2025/recording_location.txt"

    # Audio Parameters
    SR = 32000               
    DURATION = 5             
    HOP_LENGTH = 512        
    N_MELS = 128             
    FMIN = 20                
    FMAX = SR // 2           

    # Model Parameters
    MODEL_NAME = "tf_efficientnet_b0"  
    PRETRAINED = True
    NUM_CLASSES = 206                  

    # Training Parameters
    EPOCHS = 1
    BATCH_SIZE = 3
    LR = 1e-4
    SEED = 42
    NUM_WORKERS = 2

    # Inference
    THRESHOLD = 0.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug Mode
    DEBUG = False

cfg = CFG()


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.SEED)


df_taxonomy = pd.read_csv(cfg.TAXONOMY_CSV)
df_train = pd.read_csv(cfg.TRAIN_CSV)


df_train


# Convert stringified lists to actual Python lists
for col in ['secondary_labels', 'type']:
    df_train[col] = df_train[col].apply(lambda x: ast.literal_eval(x))

# Add full path to audio files
df_train['filepath'] = df_train['filename'].apply(lambda x: os.path.join(cfg.TRAIN_AUDIO_DIR, x))

# Preview
df_train.sample(5)


# Check shape and column details
print("Shape of training data:", df_train.shape)
print("Columns:", df_train.columns.tolist())
df_train.info()


# Check missing values
df_train.isnull().sum()


# How many unique bird labels?
print("Number of unique bird species:", df_train['primary_label'].nunique())

# Most common species
df_train['primary_label'].value_counts().head(10)


plt.figure(figsize=(12, 8))
sns.countplot(y='primary_label', data=df_train,
              order=df_train['primary_label'].value_counts().iloc[:30].index)
plt.title("Top 30 Most Frequent Bird Species")
plt.xlabel("Frequency")
plt.ylabel("Bird Species")
plt.tight_layout()
plt.show()


# Count unique values in 'rating'
rating_counts = df_train['rating'].value_counts().sort_index()
print(rating_counts)


plt.figure(figsize=(8, 5))
sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis")

plt.title("Distribution of Ratings in Training Data")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Plot class distribution of primary labels
plt.figure(figsize=(16, 6))
df_train["primary_label"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Distribution of Primary Labels in Train Set")
plt.xlabel("Bird Species (primary_label)")
plt.ylabel("Number of Samples")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# Plot distribution of primary_label for each rating (0 to 5)
for r in range(6):
    df_rated = df_train[df_train["rating"] == r]
    
    if df_rated.empty:
        print(f"No data found for rating = {r}")
        continue
    
    plt.figure(figsize=(16, 5))
    df_rated["primary_label"].value_counts().plot(kind="bar", color="coral")
    plt.title(f"Distribution of Primary Labels (Rating = {r})")
    plt.xlabel("Bird Species (primary_label)")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# Compute duration for each audio file
def get_duration(path):
    f = sf.SoundFile(path)
    return len(f) / f.samplerate

# Apply to all audio files
tqdm.pandas()
df_train["duration"] = df_train["filepath"].progress_apply(get_duration)

# Plot the distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_train["duration"], bins=50, kde=True, color="teal")
plt.title("Distribution of Audio Durations")
plt.xlabel("Duration (seconds)")
plt.ylabel("Number of Files")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# Plot Mel Spectrogram
def plot_mel_spectrogram(path, sr=cfg.SR, n_mels=cfg.N_MELS, fmin=cfg.FMIN, fmax=cfg.FMAX, hop_length=cfg.HOP_LENGTH):
    y, sr = librosa.load(path, sr=sr)
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", fmax=fmax
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.show()


# Pick a random audio file
sample_path = df_train.sample(1)["filepath"].values[0]
plot_mel_spectrogram(sample_path)


# Plot mel spectrograms for 5 random samples
for i, row in df_train.sample(5).iterrows():
    print(f"Sample {i+1} - Primary Label: {row['primary_label']}")
    plot_mel_spectrogram(row['filepath'])


def preprocess_audio(path, sr=cfg.SR, duration=cfg.DURATION, n_mels=cfg.N_MELS,
                     fmin=cfg.FMIN, fmax=cfg.FMAX, hop_length=cfg.HOP_LENGTH):
    
    y, sr = librosa.load(path, sr=sr, duration=duration)
    
    # Pad if audio is too short
    expected_length = sr * duration
    if len(y) < expected_length:
        y = np.pad(y, (0, expected_length - len(y)))
    else:
        y = y[:expected_length]
    
    # Create mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                         fmin=fmin, fmax=fmax, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to 0-1
    mel_db -= mel_db.min()
    mel_db /= mel_db.max()

    return mel_db.astype(np.float32)  # shape: [n_mels, time]


class BirdClefDataset(Dataset):
    def __init__(self, df, label2id, transform=None):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label = row["primary_label"]

        # Preprocess audio
        mel = preprocess_audio(filepath)  # shape: [n_mels, time]

        if self.transform:
            mel = self.transform(mel)

        # Convert to tensor and add channel dimension
        mel = torch.tensor(mel).unsqueeze(0)  # shape: [1, n_mels, time]

        # Convert label to index
        label_idx = self.label2id[label]

        return mel, torch.tensor(label_idx, dtype=torch.long)


# Create label-to-index and index-to-label mappings
unique_labels = sorted(df_train["primary_label"].unique())
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}


# Create a small sample dataset
sample_df = df_train.sample(5).reset_index(drop=True)

# Instantiate the dataset
dataset = BirdClefDataset(sample_df, label2id=label2id)

# Test the first sample
mel_tensor, label = dataset[0]

print(f"Mel spectrogram shape: {mel_tensor.shape}")  # Expected: [1, n_mels, time]
print(f"Label index: {label} → {list(label2id.keys())[list(label2id.values()).index(label.item())]}")


plt.figure(figsize=(10, 4))
plt.imshow(mel_tensor.squeeze(0).numpy(), aspect='auto', origin='lower')
plt.title("Mel Spectrogram Tensor")
plt.xlabel("Time")
plt.ylabel("Mel Bins")
plt.colorbar(format="%+2.0f")
plt.show()


class BirdCLEFModel(nn.Module):
    def __init__(self, model_name=cfg.MODEL_NAME, num_classes=cfg.NUM_CLASSES, pretrained=cfg.PRETRAINED):
        super(BirdCLEFModel, self).__init__()
        
        # Use timm backbone with no classifier, include pooling
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool='avg'  # this handles pooling internally
        )
        
        # Add classifier
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)      # shape: (B, features)
        x = self.classifier(x)    # shape: (B, num_classes)
        return x


# Create sample dataset and dataloader
sample_df = df_train.sample(8).reset_index(drop=True)
dataset = BirdClefDataset(sample_df, label2id=label2id)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Instantiate model
model = BirdCLEFModel().to(cfg.DEVICE)

# Get a batch of data
batch = next(iter(dataloader))
inputs, targets = batch
inputs = inputs.to(cfg.DEVICE)

# Forward pass
outputs = model(inputs)

# Display shapes
print(f"Input shape      : {inputs.shape}")   # [B, 1, 128, time_steps]
print(f"Output shape     : {outputs.shape}")  # [B, num_classes]
print(f"Target labels     : {targets}")


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


# Training function with checkpoint saving
def train_model(model, train_loader, val_loader, epochs=cfg.EPOCHS, lr=cfg.LR, resume=False, checkpoint_path=None):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    start_epoch = 0

    # Load from checkpoint if resuming
    if resume and checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        print(f"Resumed from checkpoint: {checkpoint_path} | Starting at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, epochs):
        print(f"\n Epoch {epoch+1}/{epochs}")

        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        end = time.time()

        print(f" Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f" Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
        print(f" Time: {(end - start):.2f}s")

        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, f"/kaggle/working/checkpoint_epoch_{epoch+1}.pth")

        # Save best model and label2id mapping
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Save model
            torch.save(model.state_dict(), "/kaggle/working/best_model.pth")
            print("Best model saved.")

            # Save label mapping
            with open("/kaggle/working/label2id.json", "w") as f:
                json.dump(label2id, f)
            print("label2id mapping saved.")

    print(f"\n Best Validation Accuracy: {best_val_acc:.4f}")


# Spliting training data
train_df, val_df = train_test_split(df_train, test_size=0.2, stratify=df_train['primary_label'], random_state=cfg.SEED)

train_dataset = BirdClefDataset(train_df.reset_index(drop=True), label2id)
val_dataset = BirdClefDataset(val_df.reset_index(drop=True), label2id)

train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

# Initialize model
model = BirdCLEFModel().to(cfg.DEVICE)

# Launch training (can resume later with resume=True)
train_model(model, train_loader, val_loader, resume=False)



train_model(model, train_loader, val_loader, resume=True, checkpoint_path="checkpoint_epoch_1.pth")




