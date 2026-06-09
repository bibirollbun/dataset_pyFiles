import pandas as pd
import numpy as np

import os
from glob import glob
import sys
import ast 
import random 
from tqdm import tqdm
import time 
import json
import soundfile as sf
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Audio

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


import timm 
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader



class CONFIG:
    TRAIN = '/kaggle/input/birdclef-2025/train_audio'
    OUTPUT = '/kaggle/working/'
    TRAIN_CSV = '/kaggle/input/birdclef-2025/train.csv'
    TAXONOMY = '/kaggle/input/birdclef-2025/taxonomy.csv'
    SR = 32000
    DURATION = 10
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 20
    FMAX = SR//2
    # Model Parameters
    MODEL_NAME = "tf_efficientnet_b0"  
    PRETRAINED = True
    NUM_CLASSES = 4                  

    # Training Parameters
    EPOCHS = 1
    BATCH_SIZE = 3
    LR = 1e-4
    SEED = 42
    NUM_WORKERS = 0

    # Inference
    THRESHOLD = 0.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug Mode
    DEBUG = False


    


train = pd.read_csv(CONFIG.TRAIN_CSV)
taxonomy = pd.read_csv(CONFIG.TAXONOMY)

taxonomy.head()


train.head()


data_path = pd.merge(train,taxonomy[['scientific_name','class_name']],how='left',on='scientific_name')



# Convert stringified lists to actual Python lists
for col in ['secondary_labels', 'type']:
    data_path[col] = data_path[col].apply(lambda x: ast.literal_eval(x))

# Add full path to audio files
data_path['filepath'] = data_path['filename'].apply(lambda x: os.path.join(CONFIG.TRAIN, x))

# Preview
data_path.sample(5)



print("Shape of training data:", data_path.shape)
print("Columns:", data_path.columns.tolist())
data_path.info()






plt.figure(figsize=(12, 8))
sns.countplot(y='primary_label', data=data_path,
              order=data_path['primary_label'].value_counts().iloc[:30].index)
plt.title("Top 30 Most Frequent Bird Species")
plt.xlabel("Frequency")
plt.ylabel("Bird Species")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.barplot(x=data_path['rating'].value_counts().sort_index().index, y=data_path['rating'].value_counts().sort_index().values, palette="viridis")

plt.title("Distribution of Ratings in Training Data")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


plt.title('Count of Bird Classes', size=16)
sns.countplot(data = data_path, y ='class_name')
plt.ylabel('Count', size=12)
plt.xlabel('Classes', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()
## so we have 6 emotions, almost equally distributed



data_path = pd.read_csv('/kaggle/input/dataset-with-recordings/record-length.csv')


# Waveplots - Waveplots let us know the loudness of the audio at a given time.
# Spectograms - A spectrogram is a visual representation of the spectrum of frequencies of sound or other signals 
# as they vary with time. It’s a representation of frequencies changing with respect to time for given audio/music signals.

def create_waveplot(sample_file, e=None):
    data, sr = librosa.load(sample_file)

    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for {} class'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def create_spectrogram(sample_file, bird_class=None):
    data, sr = librosa.load(sample_file)

    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    if bird_class is not None:
        plt.title('Spectrogram for {} class'.format(bird_class), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')   
    plt.colorbar()
    plt.show()




sample_file , bird = data_path.sample(1)[['filepath','class_name']].values[0]
create_waveplot(sample_file, bird)
create_spectrogram(sample_file, bird)
Audio(sample_file)


for i, row in data_path.sample(5).iterrows():
    print(f"Sample {i+1} - Primary Label: {row['primary_label']}")
    create_spectrogram(row['filepath'])
    Audio(row['filepath'])


data_path['duration'].sort_values(ascending=True)


def preprocess_audio(path, sr=CONFIG.SR, duration=CONFIG.DURATION, n_mels=CONFIG.N_MELS,
                     fmin=CONFIG.FMIN, fmax=CONFIG.FMAX, hop_length=CONFIG.HOP_LENGTH):
    
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


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG.SEED)



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
unique_labels = sorted(data_path["primary_label"].unique())
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}


sample_df = data_path.sample(5).reset_index(drop=True)

# Instantiate the dataset
dataset = BirdClefDataset(sample_df, label2id=label2id)

# Test the first sample
mel_tensor, label = dataset[1]

print(f"Mel spectrogram shape: {mel_tensor.shape}")  # Expected: [1, n_mels, time]
print(f"Label index: {label} → {list(label2id.keys())[list(label2id.values()).index(label.item())]}")
plt.figure(figsize=(10,4))
plt.imshow(mel_tensor.squeeze(0).numpy(), aspect='auto', origin='lower')
plt.title('mel spectogram tensor')

plt.xlabel('time')
plt.ylabel('mel bins')

plt.show()


class BirdCLEFModel(nn.Module):
    def __init__(self, model_name=CONFIG.MODEL_NAME, num_classes=CONFIG.NUM_CLASSES, pretrained=CONFIG.PRETRAINED):
        super(BirdCLEFModel, self).__init__()
        
        # Use timm backbone with no classifier, include pooling
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes,
            global_pool='avg'  # this handles pooling internally
        )
        
        # Add classifier
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)      # shape: (B, features)
        x = self.classifier(x)    # shape: (B, num_classes)
        return x


# Create sample dataset and dataloader
sample_df = data_path.sample(8).reset_index(drop=True)
dataset = BirdClefDataset(sample_df, label2id=label2id)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Instantiate model
model = BirdCLEFModel().to(CONFIG.DEVICE)

# Get a batch of data
batch = next(iter(dataloader))
inputs, targets = batch
inputs = inputs.to(CONFIG.DEVICE)

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
        inputs, labels = inputs.to(CONFIG.DEVICE), labels.to(CONFIG.DEVICE)

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
            inputs, labels = inputs.to(CONFIG.DEVICE), labels.to(CONFIG.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


# Training function with checkpoint saving
def train_model(model, train_loader, val_loader, epochs=CONFIG.EPOCHS, lr=CONFIG.LR, resume=False, checkpoint_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
train_df, val_df = train_test_split(data_path, test_size=0.2, stratify=data_path['primary_label'], random_state=CONFIG.SEED)

train_dataset = BirdClefDataset(train_df.reset_index(drop=True), label2id)
val_dataset = BirdClefDataset(val_df.reset_index(drop=True), label2id)

train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=CONFIG.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=CONFIG.NUM_WORKERS)

# Initialize model
model = BirdCLEFModel().to(CONFIG.DEVICE)

# Launch training (can resume later with resume=True)
train_model(model, train_loader, val_loader, resume=False)





