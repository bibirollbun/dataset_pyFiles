# Standard Libraries
import os
import gc
import time
import math
import random
import warnings
import logging
from pathlib import Path
from glob import glob
from typing import Union
import copy
import concurrent.futures

# Data Handling
import numpy as np
import pandas as pd
import joblib
import pickle
import collections
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report

# Audio Processing
import librosa
import librosa.display
import soundfile as sf
from soundfile import SoundFile
import torchaudio

# Machine Learning & PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import timm

# Visualization
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Progress Bars
from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm

# Logging and Warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# Check versions
print(f"librosa version : {librosa.__version__}")
print(f"librosa files : {librosa.__file__}")
print("âœ… All libraries imported in the environment.")




class Config:
    # ===== General Settings =====
    seed = 42
    print_freq = 100
    num_workers = 4

    # ===== Audio Settings =====
    FS = 32000  # Sampling rate
    N_FFT = 1024
    HOP_LENGTH = 512
    FMIN = 50
    FMAX = 14000
    N_MELS = 128
    TARGET_DURATION = 10.0  # duration in seconds for full input
    TARGET_DURATION_TRAIN = 10
    TARGET_DURATION_TEST = 5

    # ===== Image / Mel Spectrogram Settings =====
    MEL_SHAPE = (256, 256)         # (height, width)
    TARGET_SHAPE = (3, 256, 256)   # RGB Image Shape (C, H, W)

    
    # ===== File Paths =====
    test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes"
    submission_csv = "/kaggle/input/birdclef-2025/sample_submission.csv"
    model_path = "/kaggle/input/birdcleft-clean-and-vad-filtered-data/best_model_187.pth"
    backbone_weights = "/kaggle/input/birdcleft-clean-and-vad-filtered-data/seresnext_backbone_weights.pth"
    master_labels = "/kaggle/input/birdcleft-clean-and-vad-filtered-data/valid_labels.pkl"

    # ===== Model Settings =====
    model_name = 'seresnext26t_32x4d'
    pretrained = False
    in_channels = 1

    # ===== Derived Attributes (initialized later) =====
    master_labels = None
    NUM_CLASSES = None


# Instantiate and initialize
config = Config()

print(f"âœ… Loaded master label list. Total number of classes: {config.NUM_CLASSES}")
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
species_ids = pd.read_csv(config.submission_csv).columns[1:].tolist()


# Load the master label list and set NUM_CLASSES
with open("/kaggle/input/birdcleft-clean-and-vad-filtered-data/valid_labels.pkl", "rb") as f:
    master_labels = pickle.load(f)
NUM_CLASSES = len(master_labels)  # should be 206


print(f"total number of labels in full data : {len(master_labels)}")  # Should print 206


def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config.seed)



# Your input directory
mel_dir = "/kaggle/input/birdclef-2025-mel-spectrogram"
csv_path = "/kaggle/input/birdcleft-clean-and-vad-filtered-data/train_audio_10sec_chunks_VAD_filtered.csv"

# Read only clean train dataframe
clean_train_df = pd.read_csv(csv_path)
print("âœ… CSV loaded:", clean_train_df.shape)

# Filter only clean_train mel batches
clean_train_files = sorted([
    f for f in os.listdir(mel_dir)
    if f.startswith("clean_train") and f.endswith(".npz")
])
print(f"ğŸŸ¢ Found {len(clean_train_files)} clean_train .npz files")

# Build key-to-file index for clean_train only
key_to_file = {}

print("ğŸ”� Indexing clean_train mel keys...")
for batch_file in tqdm(clean_train_files):
    path = os.path.join(mel_dir, batch_file)
    try:
        npz = np.load(path)
        for key in npz.files:
            key_to_file[key] = batch_file
    except Exception as e:
        print(f"âš ï¸� Error reading {batch_file}: {e}")

print(f"âœ… Total keys indexed: {len(key_to_file)}")



# Master label filtering before any split
#label_counts = clean_train_df['primary_label'].value_counts()
#valid_labels = label_counts[label_counts >= 2].index



#with open("valid_labels.pkl", "wb") as f:
    #pickle.dump(valid_labels.tolist(), f)
# i have saved it once and then we will be loading it directly in dataset class


# Pick 3 random chunk_ids from the clean train CSV
sample_keys = random.sample(clean_train_df['chunk_id'].tolist(), 3)

def load_clean_train_mel(chunk_id):
    if chunk_id not in key_to_file:
        raise ValueError(f"â�Œ Key {chunk_id} not found.")
    file = key_to_file[chunk_id]
    path = os.path.join(mel_dir, file)
    with np.load(path) as npz:
        return npz[chunk_id]

for key in sample_keys:
    mel = load_clean_train_mel(key)
    print(f"\nChunk ID: {key}")
    print(f"Shape: {mel.shape}")
    print(f"Min: {mel.min():.4f}, Max: {mel.max():.4f}, Mean: {mel.mean():.4f}")



class BirdMelNPZDataset(Dataset):
    def __init__(self, df, mel_dir, key_to_file, labels=None, augment=False, sample_rate=32000):
        self.df = df.reset_index(drop=True)
        self.mel_dir = mel_dir
        self.key_to_file = key_to_file
        self.augment = augment
        self.sample_rate = sample_rate

        if labels is None:
            with open("/kaggle/input/birdcleft-clean-and-vad-filtered-data/valid_labels.pkl", "rb") as f:
                self.labels = pickle.load(f)
        else:
            self.labels = labels

        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
        self.num_classes = len(self.labels)

        self.df['chunk_id_clean'] = self.df['chunk_id'].str.replace('/', '-', regex=False)

        # Cache loaded npz files to avoid multiple reloads in an epoch
        self.loaded_batches = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key = row['chunk_id_clean']

        batch_file = self.key_to_file.get(key, None)
        if batch_file is None:
            # Key not found in index
            mel = np.zeros((3, 256, 256), dtype=np.float32)  # assuming shape
        else:
            # Load batch file if not already loaded
            if batch_file not in self.loaded_batches:
                path = os.path.join(self.mel_dir, batch_file)
                self.loaded_batches[batch_file] = np.load(path)

            mel = self.loaded_batches[batch_file][key]

        if self.augment:
            mel = self.apply_augmentation(mel)

        mel_tensor = torch.tensor(mel, dtype=torch.float32)  # (3, 256, 256)

        label_vec = np.zeros(self.num_classes, dtype=np.float32)
        label_vec[self.label2idx[row['primary_label']]] = 1.0

        # Handling secondary labels if present
        if 'secondary_labels' in row and isinstance(row['secondary_labels'], list):
            for sec in row['secondary_labels']:
                if sec in self.label2idx:
                    label_vec[self.label2idx[sec]] = 1.0

        return mel_tensor, torch.tensor(label_vec)

    # (Augmentation methods )


    def apply_augmentation(self, mel):
        mel = self.time_mask(mel, T=30)
        mel = self.freq_mask(mel, F=15)
        mel = self.add_noise(mel, noise_level=0.01)
        return mel

    def time_mask(self, mel, T=30):
        t = mel.shape[2]  # time dimension
        t0 = np.random.randint(0, max(1, t - T))
        mel[:, :, t0:t0 + T] = 0
        return mel

    def freq_mask(self, mel, F=15):
        f = mel.shape[1]  # frequency dimension
        f0 = np.random.randint(0, max(1, f - F))
        mel[:, f0:f0 + F, :] = 0
        return mel

    def add_noise(self, mel, noise_level=0.01):
        noise = np.random.randn(*mel.shape) * noise_level
        return mel + noise





# collate function to pad mel spectrograms on time dimension (if varying length)
def collate_pad_mel(batch):
    mel_specs = [item[0] for item in batch]  # each shape (C, n_mels, time)
    targets = [item[1] for item in batch]

    max_time = max([m.shape[-1] for m in mel_specs])

    padded_mels = []
    for m in mel_specs:
        pad_len = max_time - m.shape[-1]
        padded = F.pad(m, (0, pad_len))
        padded_mels.append(padded)

    mel_specs_padded = torch.stack(padded_mels)  # (B, C, n_mels, max_time)
    targets = torch.stack(targets)

    return mel_specs_padded, targets



def get_balanced_sampler(df, label_col='primary_label'):
    """
    Create a WeightedRandomSampler to balance class frequencies.

    Args:
        df: pandas DataFrame with a 'primary_label' column
        label_col: name of the column containing class labels

    Returns:
        torch.utils.data.WeightedRandomSampler
    """
    # Count samples per class
    class_counts = df[label_col].value_counts()
    class_weights = 1. / class_counts

    # Assign each sample its weight
    sample_weights = df[label_col].map(class_weights).values
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# Count instances per class
label_counts = clean_train_df['primary_label'].value_counts()

# Keep only those classes with at least 2 samples
valid_labels = label_counts[label_counts >= 2].index
clean_train_df= clean_train_df[clean_train_df['primary_label'].isin(valid_labels)].reset_index(drop=True)
valid_labels.nunique()


# Assume mel_dir and key_to_file already defined as:
mel_dir = "/kaggle/input/birdclef-2025-mel-spectrogram"

# Split dataframe into train and validation stratified on primary_label
train_idx, val_idx = train_test_split(
    range(len(clean_train_df)),
    test_size=0.2,
    stratify=clean_train_df['primary_label'],
    random_state=42
)

train_df = clean_train_df.iloc[train_idx].reset_index(drop=True)
val_df = clean_train_df.iloc[val_idx].reset_index(drop=True)

#alag alag datset create 
train_dataset = BirdMelNPZDataset(
    train_df,
    mel_dir=mel_dir,
    key_to_file=key_to_file,
    augment=True  # Enable augmentations in training
)

val_dataset = BirdMelNPZDataset(
    val_df,
    mel_dir=mel_dir,
    key_to_file=key_to_file,
    augment=False
)

# ğŸ”„ Balanced Sampler for training (if you have implemented `get_balanced_sampler`)
train_sampler = get_balanced_sampler(train_df, label_col='primary_label')

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    sampler=train_sampler,
    collate_fn=collate_pad_mel
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_pad_mel
)

def check_dataset(dataset, name="Dataset"):
    mel_tensor,  label_tensor = dataset[0]
    print(f"=== {name.upper()} ===")
    print(f"Mel Spectrogram shape: {mel_tensor.shape}")         # Expected: (128, Time)
    #print(f"YAMNet Embedding shape: {embedding_tensor.shape}")  # Expected: (1024,)
    print(f"Label shape: {label_tensor.shape}")                 # Expected: (206,)

# Check train dataset
check_dataset(train_dataset, "Train Dataset")

sample_idx = 0  # koi b sample index

# Train dataset ka sample
print("=== TRAIN DATASET SAMPLE ===")
print(type(train_dataset))
print(train_dataset[sample_idx])

# Validation dataset ka sample
print("=== VALIDATION DATASET SAMPLE ===")
print(type(val_dataset))
print(val_dataset[sample_idx])




def plot_multiple_mels(dataset, indices, title_prefix='Sample Mel Spectrogram'):
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))  # 1 row, n columns

    for i, idx in enumerate(indices):
        mel_tensor, _ = dataset[idx]
        mel = mel_tensor[0].numpy()  # first channel

        ax = axes[i] if n > 1 else axes
        im = ax.imshow(mel, aspect='auto', origin='lower')
        ax.set_title(f"{title_prefix} #{idx}")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Bands")
        ax.label_outer()  # Only show outer labels for clean look

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.show()

# Example usage:
print("Train Samples:")
plot_multiple_mels(train_dataset, indices=list(range(5)), title_prefix='Train Sample Mel')

print("Validation Samples:")
plot_multiple_mels(val_dataset, indices=list(range(5)), title_prefix='Validation Sample Mel')



sampler = get_balanced_sampler(train_df)

print("Unique labels:", train_df['primary_label'].nunique())
print("Sampler length:", len(sampler))
sample_weights = [sampler.weights[i].item() for i in range(10)]
print("First 10 sample weights:", sample_weights)

sampled_labels = []
for i, (mels, labels) in enumerate(train_loader):
    sampled_labels.extend(labels.argmax(dim=1).tolist())  # One-hot to class index
    if i > 50:  # Analyze only first 50 batches
        break

print("Sampled class distribution in first 50 batches:", collections.Counter(sampled_labels))
label_counts = dict(collections.Counter(sampled_labels))

plt.figure(figsize=(14, 6))
sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
plt.title("Sampled Label Distribution (First 50 Batches)")
plt.xlabel("Class Index")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



#model download kr k weights save kr lety hain (inference me intenet allowed nhi hai to it would be helpful that time)
#model = timm.create_model(
#    "hf_hub:timm/seresnext26t_32x4d.bt_in1k",
#    pretrained=True,
#    in_chans=3,
#    num_classes=0)

# Save the backbone weights locally
#torch.save(model.state_dict(), "seresnext_backbone_weights.pth")



class ImprovedBirdCLEFModel(nn.Module):
    def __init__(self, num_classes=187, pretrained=True):
        super().__init__()
        
        self.backbone = timm.create_model(
            "seresnext26t_32x4d",  # â¬…ï¸� hf_hub hata diya downloaded weights use kry gy
            pretrained=False,  
            in_chans=3,
            num_classes=0
        )

        if pretrained:
            state_dict = torch.load("/kaggle/input/birdcleft-clean-and-vad-filtered-data/seresnext_backbone_weights.pth", map_location=device)  # local path
            self.backbone.load_state_dict(state_dict, strict=False)  # load weights manually download kiye hain

        
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)             # Already [B, 2048]
        #print("After backbone:", x.shape)
        x = self.classifier(x)           # [B, num_classes]
        return x


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False

def unfreeze_last_blocks(model, num_blocks=2):
    # Unfreezes last few blocks of the backbone
    children = list(model.backbone.children())
    for block in children[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImprovedBirdCLEFModel(num_classes=NUM_CLASSES, pretrained=True).to(device)

# Step 1: Freeze full backbone
freeze_backbone(model)

# Step 2: Unfreeze last N blocks (e.g., last 3 blocks)
unfreeze_last_blocks(model, num_blocks=3)

# âœ… Print trainable parameters (for sanity check)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")



# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load model with classifier
model = ImprovedBirdCLEFModel(num_classes=NUM_CLASSES, pretrained=True).to(device)

# Step 2: Freeze backbone
freeze_backbone(model)

# Step 3: Unfreeze last N blocks for fine-tuning
unfreeze_last_blocks(model, num_blocks=3)

# Step 4: Define loss function
criterion = nn.BCEWithLogitsLoss()

# Step 5: Define optimizer (AdamW for better regularization)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),  # only trainable params
    lr=1e-4,
    weight_decay=1e-4
)

# Step 6: Define LR scheduler (optional, you can add in training loop)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# âœ… Print summary
print("ğŸ”¥ Model compiled and ready for training!")
print(f"Loss: {criterion.__class__.__name__}")
print(f"Optimizer: {optimizer.__class__.__name__} | LR: {optimizer.param_groups[0]['lr']}")



# Mixup helper functions
def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# EarlyStopping Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric):
        if self.best_score is None or metric > self.best_score + self.min_delta:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Your kaggle metric function you shared:
def kaggle_macro_roc_auc_ignoring_empty(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    valid_classes = np.where(y_true.sum(axis=0) > 0)[0]
    if len(valid_classes) == 0:
        return float('nan')
    return roc_auc_score(y_true[:, valid_classes], y_probs[:, valid_classes], average='macro')


# Training Loop
def train_one_epoch(model, optimizer, criterion, dataloader, device, alpha=0.4):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        inputs, targets = batch  # <- tuple unpacking here
        inputs = inputs.to(device)
        targets = targets.to(device)

        if alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha)
            inputs, targets_a, targets_b = inputs.to(device), targets_a.to(device), targets_b.to(device)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

# Validation function (using your Kaggle ROC metric)
def validate_model(model, dataloader, device, kaggle_macro_roc_auc_ignoring_empty):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch  # âœ… fixed here
            inputs = inputs.to(device)
            targets = targets.cpu().numpy()

            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    score = kaggle_macro_roc_auc_ignoring_empty(all_targets, all_preds)
    return score



#training setup
num_epochs = 30
early_stopping = EarlyStopping(patience=2, min_delta=0.0005)
best_model_wts = copy.deepcopy(model.state_dict())
best_score = -np.inf

history = {'train_loss': [], 'val_score': []}

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device, alpha=0.4)
    val_score = validate_model(model, val_loader, device, kaggle_macro_roc_auc_ignoring_empty)

    history['train_loss'].append(train_loss)
    history['val_score'].append(val_score)

    print(f"Train Loss: {train_loss:.4f} | Validation Kaggle ROC-AUC: {val_score:.5f}")

    # Save best model
    if val_score > best_score:
        best_score = val_score
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "best_model_phase1.pth")
        print("=> Saved Best Model!")

    # Early stopping check
    early_stopping(val_score)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

# Load best model weights after training
model.load_state_dict(best_model_wts)


def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12,5))

    # Train loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Validation Kaggle Macro ROC-AUC plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_score'], 'g-', label='Validation Kaggle Macro ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.title('Validation Score Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Use this after training:
plot_training_history(history)



# ===================== MODEL =====================
class ImprovedBirdCLEFModel(nn.Module):
    def __init__(self, num_classes=187, backbone_weights=None, device=device):
        super().__init__()
        self.backbone = timm.create_model(
            "seresnext26t_32x4d",
            pretrained=False,
            in_chans=3,
            num_classes=0
        )
        if backbone_weights:
            state_dict = torch.load(backbone_weights, map_location=device)
            self.backbone.load_state_dict(state_dict, strict=False)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

def load_model(model_path, device, num_classes, backbone_weights):
    model = ImprovedBirdCLEFModel(num_classes=num_classes, backbone_weights=backbone_weights, device=device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
    
model = load_model(
    model_path=config.model_path,
    device=device,
    num_classes=187,
    backbone_weights=config.backbone_weights  # Pass it as a keyword argument
)
print("âœ… Model loaded!")



# Validate model on validation dataloader
val_score = validate_model(model, val_loader, device, kaggle_macro_roc_auc_ignoring_empty)
print(f"ğŸ“Š Validation Macro ROC-AUC: {val_score:.4f}")


def debug_validate_model(model, dataloader, device, species_ids, num_batches=1):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.cpu().numpy()

            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

            # ğŸ–¨ï¸� Print predictions
            for i in range(len(preds)):
                top5_idx = preds[i].argsort()[-5:][::-1]  # Top 5 predictions
                top5_probs = preds[i][top5_idx]
                true_indices = np.where(targets[i] == 1)[0]

                print(f"\nğŸ”Š Sample {i+1}:")
                print(f"âœ… True Labels: {[species_ids[j] for j in true_indices]}")
                print(f"ğŸ”® Top Predictions:")
                for rank, (cls_idx, prob) in enumerate(zip(top5_idx, top5_probs), start=1):
                    print(f"  {rank}. {species_ids[cls_idx]} ({prob:.3f})")

    return


debug_validate_model(
    model=model,
    dataloader=val_loader,
    device=device,
    species_ids=species_ids,  # a list like ['species_1', ..., 'species_187']
    num_batches=2  # check first 2 batches only
)


























