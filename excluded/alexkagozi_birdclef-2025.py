# !pip install --upgrade timm


import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
import math, random
import torch
import torchaudio
import torchaudio.transforms as T
from IPython.display import Audio
import numpy as np
import librosa
import librosa.display
import IPython.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import IPython.display as ipd

import matplotlib as mpl
import matplotlib.pylab as ply
import ipywidgets as widgets
import seaborn as sns


from itertools import cycle
# Set interactive backend
%matplotlib inline


cmap = mpl.cm.get_cmap('coolwarm')
sns.set_theme(style="white", palette=None)
color_pal = ply.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(ply.rcParams["axes.prop_cycle"].by_key()["color"])


DATASET_PATH = '/kaggle/input/birdclef-2025'


## To handle our settings  and configurations, let's create a class
class Config:
    seed = 42
    # Input image size and batch size
    img_size = [128, 384]
    
    # Audio duration, sample rate, and length
    duration = 5 # second
    sample_rate = 32000
    audio_len = duration*sample_rate
    
    # STFT parameters
    nfft = 2028
    window = 2048
    hop_length = audio_len // (img_size[1] - 1)
    fmin = 20
    fmax = 16000
    
    #model name
    preset = 'efficientnetv2_b2_imagenet'
    class_names = sorted(os.listdir(f'{DATASET_PATH}/train_audio/'))
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v:k for k,v in label2name.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(Config.seed)


## Print out the first 5 items in the label2name and name2label dictionaries
print(f"Number of classes: {Config.num_classes}")
print({k: Config.label2name[k] for k in list(Config.label2name)[:5]})


df = pd.read_csv(f'{DATASET_PATH}/train.csv')
df['filepath'] = DATASET_PATH + '/train_audio/' + df.filename
df['target'] = df.primary_label.map(Config.name2label)
df['filename'] = df.filepath.map(lambda x: x.split('/')[-1])
df['xc_id'] = df.filepath.map(lambda x: x.split('/')[-1].split('.')[0])

## display a few rows of the dataframe from columns ['scientific_name', 'scientific_name',  'filepath']
df = df.sample(frac=1, random_state=Config.seed)
df.head(5)


## Display the number of samples per class and save the result in a dictionary
class_counts = df.primary_label.value_counts()
class_counts = class_counts.sort_index()
class_counts
# ## Save to a csv file
class_counts_csv = pd.DataFrame(class_counts.items(), columns=['class', 'count'])
## Show the largest and smallest classes with the corresponding counts
# Find the minimum and maximum counts
min_count = class_counts.min()
max_count = class_counts.max()
 
print(f"Smallest class: {class_counts_csv['class'][class_counts_csv['count'].idxmin()]} {min_count}")
print(f"Largest class: {class_counts_csv['class'][class_counts_csv['count'].idxmax()]} {max_count}")


# Store the sampling rate as `sr`
def load_audio(filepath):
    audio, sr = librosa.load(filepath)
    return audio, sr


import random
for i in range(2):
    random_index = random.randint(0, df.shape[0])
    ipd.Audio(df['filepath'].iloc[random_index])
    audio, sr = load_audio(df['filepath'].iloc[random_index])
    plt.figure(figsize=(10, 3))
    pd.Series(audio).plot(figsize=(10, 5),
                    lw=1,
                    title=f"{df['scientific_name'].iloc[random_index]}",
                    color=color_pal[0])
    ## Zoomed in sample to view waves better:
    plt.show()


#### Understanding the audio data
for i in range(2):
    random_index = random.randint(0, df.shape[0])
    ipd.Audio(df['filepath'].iloc[random_index])
    audio, sr = load_audio(df['filepath'].iloc[random_index])
    print(f"Audio: {audio}")
    print(f"Shape of the audio: {audio.shape}")
## The audio file is a numpy array. However, the size of the arrays are different hence we need to pad/trim the arrays to make them the same size



"""
 function to preview simple spectrograms in decibels. A decibel is a logarithmic unit that expresses 
 the ratio of two values of a physical quantity, often power or intensity.
 """
def audio_to_spectrogram(audio):
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print(S_db.shape)

    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(S_db,
                                x_axis='time',
                                y_axis='log',
                                ax=ax)
    ax.set_title(f"{df['scientific_name'].iloc[random_index]} Audio Spectogram", fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.show()

for i in range(2):
    random_index = random.randint(0, df.shape[0])
    audio, sr = load_audio(df['filepath'].iloc[random_index])
    audio_to_spectrogram(audio)


"""
While a regular spectrogram uses a linear frequency scale, 
a Mel spectrogram uses the Mel scale, which is designed to better reflect how humans perceive sound.
"""
def audio_to_melspectrogram(audio, sr):
    S = librosa.feature.melspectrogram(y=audio,
                                   sr=sr,
                                   n_mels=128 * 2,)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    print(S_db_mel.shape)
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot the mel spectogram
    img = librosa.display.specshow(S_db_mel,
                                x_axis='time',
                                y_axis='log',
                                ax=ax)
    ax.set_title('Mel Spectogram Example', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.show()

for i in range(2):
    random_index = random.randint(0, df.shape[0])
    audio, sr = load_audio(df['filepath'].iloc[random_index])
    audio_to_melspectrogram(audio, sr)


import random

class AudioUtil:
    @staticmethod
    def open(audio_file, target_sample_rate=Config.sample_rate):
        waveform, sr = torchaudio.load(audio_file)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != target_sample_rate:
            resampler = T.Resample(sr, target_sample_rate)
            waveform = resampler(waveform)
        return waveform, target_sample_rate

    @staticmethod
    def pad_truncate(waveform, max_len=Config.audio_len):
        if waveform.shape[1] > max_len:
            waveform = waveform[:, :max_len]
        elif waveform.shape[1] < max_len:
            pad_size = max_len - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_size))
        return waveform

    @staticmethod
    def spectrogram(waveform, n_fft=Config.nfft, hop_length=Config.hop_length,
                    n_mels=Config.img_size[0], f_min=Config.fmin, f_max=Config.fmax):
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=Config.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )(waveform)
        mel_db = T.AmplitudeToDB()(mel_spectrogram)
        return mel_db

    @staticmethod
    def normalize(tensor):
        min_val, max_val = tensor.min(), tensor.max()
        tensor = (tensor - min_val) / (max_val - min_val + 1e-6)
        return tensor

    @staticmethod
    def to_rgb(tensor):
        return tensor.repeat(3, 1, 1)

    # ------------------- AUGMENTATIONS -------------------
    @staticmethod
    def augment_waveform(waveform, sr):
        """Apply random augmentations directly on the waveform."""
        # Time shifting
        if random.random() < 0.5:
            shift = int(random.uniform(-0.1, 0.1) * waveform.shape[1])
            waveform = torch.roll(waveform, shifts=shift, dims=1)

        # Add Gaussian noise
        if random.random() < 0.3:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise

        # Pitch shift (fix: use keyword args for librosa >=0.10)
        if random.random() < 0.3:
            n_steps = random.choice([-2, -1, 1, 2])  # semitones
            waveform_np = waveform.squeeze().numpy()
            shifted = librosa.effects.pitch_shift(y=waveform_np, sr=sr, n_steps=n_steps)
            waveform = torch.tensor(shifted).unsqueeze(0)

        return waveform


    @staticmethod
    def augment_spectrogram(spec):
        """Apply random augmentations on the spectrogram (SpecAugment-style)."""
        if random.random() < 0.5:
            spec = T.FrequencyMasking(freq_mask_param=15)(spec)
        if random.random() < 0.5:
            spec = T.TimeMasking(time_mask_param=30)(spec)
        return spec

    # ------------------- MAIN PIPELINE -------------------
    @staticmethod
    def process(audio_file, with_label=False, label=None, augment=False):
        waveform, sr = AudioUtil.open(audio_file)
        waveform = AudioUtil.pad_truncate(waveform)

        # Apply waveform augmentations
        if augment:
            waveform = AudioUtil.augment_waveform(waveform, sr)

        spec = AudioUtil.spectrogram(waveform)
        
        # Apply spectrogram augmentations
        if augment:
            spec = AudioUtil.augment_spectrogram(spec)

        spec = AudioUtil.normalize(spec)
        spec = AudioUtil.to_rgb(spec)

        if with_label and label is not None:
            target = torch.tensor(label, dtype=torch.long)
            return spec, target
        else:
            return spec



from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

class BirdClefDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec, label = AudioUtil.process(
            row.filepath,
            with_label=True,
            label=row.target,
            augment=self.augment
        )
        return spec, label

def balance_classes_for_cv(df, n_splits=5, min_count=3):
    """
    Hybrid strategy for handling rare classes:
    - Drop classes with < min_count samples
    - Oversample classes with count < n_splits
    - Keep others unchanged
    """
    dfs = []
    for label, group in df.groupby("target"):
        count = len(group)
        if count < min_count:
            # Drop ultra-rare classes
            continue
        elif count < n_splits:
            # Oversample to reach n_splits
            repeat_times = -(-n_splits // count)  # ceil division
            group = pd.concat([group] * repeat_times, ignore_index=True)
        dfs.append(group)
    
    return pd.concat(dfs, ignore_index=True).reset_index(drop=True)


def build_dataloaders_cv(df, n_splits=5, fold_idx=0, batch_size=32, num_workers=2):
    # Balance dataset first
    df_balanced = balance_classes_for_cv(df, n_splits=n_splits)

    # Stratified split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.seed)
    splits = list(skf.split(df_balanced, df_balanced["target"]))
    train_idx, val_idx = splits[fold_idx]
    train_df = df_balanced.iloc[train_idx].reset_index(drop=True)
    val_df = df_balanced.iloc[val_idx].reset_index(drop=True)

    # Build datasets
    train_ds = BirdClefDataset(train_df, augment=True)
    val_ds = BirdClefDataset(val_df, augment=False)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_df, val_df


# def build_dataloaders_cv(df, n_splits=5, batch_size=32, num_workers=2, fold_idx=0):
#     """
#     Build stratified train/val dataloaders for a given fold.
    
#     Args:
#         df (DataFrame): full metadata dataframe with 'target' column
#         n_splits (int): number of folds
#         batch_size (int): batch size
#         num_workers (int): dataloader workers
#         fold_idx (int): which fold to use (0 .. n_splits-1)
#     """
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.seed)

#     # Split indices
#     splits = list(skf.split(df, df["target"]))
#     train_idx, val_idx = splits[fold_idx]

#     train_df = df.iloc[train_idx].reset_index(drop=True)
#     val_df   = df.iloc[val_idx].reset_index(drop=True)

#     # Datasets
#     train_dataset = BirdClefDataset(train_df, augment=True)
#     val_dataset   = BirdClefDataset(val_df, augment=False)

#     # DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                               num_workers=num_workers, pin_memory=True)
#     val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                               num_workers=num_workers, pin_memory=True)

#     return train_loader, val_loader, train_df, val_df



def filter_rare_classes(df, n_splits=5):
    counts = df["target"].value_counts()
    valid_classes = counts[counts >= n_splits].index
    return df[df["target"].isin(valid_classes)].reset_index(drop=True)


# Example: 5-fold cross validation
for fold in range(5):
    print(f"\n===== Fold {fold+1} =====")
    train_loader, val_loader, train_df, val_df = build_dataloaders_cv(df, n_splits=5, fold_idx=0)
    print("Train size:", len(train_df), " Val size:", len(val_df))

    # Inspect one batch
    specs, labels = next(iter(train_loader))
    print("Batch shape:", specs.shape, " Labels shape:", labels.shape)

    # You can now train your model on this fold



import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ------------------- Model -------------------
# def create_model(num_classes=Config.num_classes, pretrained=True):
#     model = timm.create_model('efficientnetv2', pretrained=pretrained)
#     in_features = model.classifier.in_features
#     model.classifier = nn.Linear(in_features, num_classes)
#     return model.to(Config.device)
# Choose one of these specific EfficientNetV2 variants:
def create_model(num_classes=Config.num_classes, pretrained=True):
    # Common EfficientNetV2 variants:
    # 'efficientnetv2_rw_t' - Tiny
    # 'efficientnetv2_rw_s' - Small  
    # 'efficientnetv2_rw_m' - Medium
    # 'efficientnetv2_rw_l' - Large
    
    model = timm.create_model('efficientnetv2_rw_s', pretrained=pretrained)  # Example: Small variant
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model.to(Config.device)

# ------------------- Optimizer + Scheduler -------------------
def create_optimizer(model, lr=1e-4, weight_decay=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def create_scheduler(optimizer, train_loader, epochs=10):
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
        three_phase=False
    )
    return scheduler

# ------------------- Training Loop -------------------
def train_one_epoch(model, train_loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for specs, labels in tqdm(train_loader, desc="Training", leave=False):
        specs, labels = specs.to(Config.device), labels.to(Config.device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(specs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * specs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for specs, labels in tqdm(val_loader, desc="Validation", leave=False):
            specs, labels = specs.to(Config.device), labels.to(Config.device)
            outputs = model(specs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * specs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ------------------- Full Training -------------------
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, lr)
    scheduler = create_scheduler(optimizer, train_loader, epochs)
    scaler = GradScaler()

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model âœ…")

    print("Training complete. Best Val Acc:", best_val_acc)






# Create model
model = create_model()

# Build DataLoaders (example fold 0)
train_loader, val_loader, train_df, val_df = build_dataloaders_cv(df, n_splits=5, fold_idx=0, batch_size=32)

# Train
train_model(model, train_loader, val_loader, epochs=10, lr=1e-4)





