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


import librosa
import torch
import torchaudio
import matplotlib.pyplot as plt
import ast
import numpy as np
import timm
import shutil
import random

from torch import nn
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# Constants
SAMPLE_RATE = 32000
DURATION = 5  # seconds
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
NUM_CLASSES = 206
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_audio(file_path, duration=DURATION, sr=SAMPLE_RATE):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    if len(y) < sr * duration:
        y = np.pad(y, (0, sr * duration - len(y)))
    else:
        y = y[:sr * duration]
    return y

def audio_to_logmel(y, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel)
   
    logmel = (logmel - logmel.mean(axis=1, keepdims=True)) / (logmel.std(axis=1, keepdims=True) + 1e-6)
    return logmel

def spec_augment(mel_spectrogram, time_mask_param=40, freq_mask_param=15, num_masks=1):
    """
    Apply SpecAugment-style time and frequency masking to a mel-spectrogram.
    Input:
        mel_spectrogram: np.array of shape (n_mels, time_steps)
    Output:
        Augmented spectrogram (same shape)
    """
    augmented = mel_spectrogram.copy()
    n_mels, time_steps = augmented.shape

    # Frequency masking
    for _ in range(num_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(1, n_mels - f))
        augmented[f0:f0+f, :] = 0.0

    # Time masking
    for _ in range(num_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(1, time_steps - t))
        augmented[:, t0:t0+t] = 0.0

    return augmented




class BirdDataset(Dataset):
    def __init__(self, df, mlb, augment=False, mixup=False, noise_std=0.0):
        self.df = df.reset_index(drop=True)
        self.mlb = mlb
        self.augment = augment
        self.mixup = mixup 
        self.noise_std = noise_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Select correct path depending on source
        if row['source'] == 'train':
            file_path = os.path.join(AUDIO_PATH_LABELED, row['filename'])
        else:
            file_path = os.path.join(AUDIO_PATH_PSEUDO, row['filename'])

        y = load_audio(file_path)
        melspec = audio_to_logmel(y)  # (128, T)

        # Normalize before augmentation
        melspec = (melspec - melspec.mean(axis=1, keepdims=True)) / (melspec.std(axis=1, keepdims=True) + 1e-6)

        # Apply SpecAugment if enabled
        if self.augment:
            melspec = spec_augment(melspec)

        melspec = torch.tensor(melspec).unsqueeze(0).float()  # (1, 128, T)

        # Build label vector
        label_list = row['labels']
        label_vec = np.zeros(len(self.mlb.classes_), dtype=np.float32)

        if row.get("source", "train") == "pseudo":
            # Pseudo-labeled: uniform weight summing to 0.5
            weight = 0.5 / len(label_list)
            for lbl in label_list:
                label_vec[self.mlb.classes_.tolist().index(lbl)] = weight
        elif len(label_list) == 1:
            # Single label: full weight
            label_vec[self.mlb.classes_.tolist().index(label_list[0])] = 1.0
        else:
            # Primary: 0.5, others: shared 0.5
            primary = label_list[0]
            secondaries = label_list[1:]
            label_vec[self.mlb.classes_.tolist().index(primary)] = 0.5
            for lbl in secondaries:
                label_vec[self.mlb.classes_.tolist().index(lbl)] = 0.5 / len(secondaries)
        # Adding gaussian Noice        
        if self.noise_std > 0:
            melspec += np.random.normal(0, self.noise_std, size=melspec.shape)

        # MIXUP LOGIC
        if self.mixup and random.random() < 0.5:
            # Sample a second example randomly
            idx2 = random.randint(0, len(self.df) - 1)
            row2 = self.df.iloc[idx2]
            if row2['source'] == 'train':
                file_path2 = os.path.join(AUDIO_PATH_LABELED, row2['filename'])
            else:
                file_path2 = os.path.join(AUDIO_PATH_PSEUDO, row2['filename'])

            y2 = load_audio(file_path2)
            mel2 = audio_to_logmel(y2)
            mel2 = (mel2 - mel2.mean(axis=1, keepdims=True)) / (mel2.std(axis=1, keepdims=True) + 1e-6)

            if self.augment:
                mel2 = spec_augment(mel2)

            # Build label for second sample
            label_vec2 = np.zeros(len(self.mlb.classes_), dtype=np.float32)
            label_list2 = row2['labels']
            if row2.get("source", "train") == "pseudo":
                weight = 0.5 / len(label_list2)
                for lbl in label_list2:
                    label_vec2[self.mlb.classes_.tolist().index(lbl)] = weight
            elif len(label_list2) == 1:
                label_vec2[self.mlb.classes_.tolist().index(label_list2[0])] = 1.0
            else:
                primary = label_list2[0]
                secondaries = label_list2[1:]
                label_vec2[self.mlb.classes_.tolist().index(primary)] = 0.5
                for lbl in secondaries:
                    label_vec2[self.mlb.classes_.tolist().index(lbl)] = 0.5 / len(secondaries)

            # Mix the two examples
            lam = np.random.beta(0.4, 0.4)  # Mixup strength
            melspec = lam * melspec + (1 - lam) * mel2
            label_vec = lam * label_vec + (1 - lam) * label_vec2

        # Convert to tensor
        # Ensure melspec is shape [1, 128, T] before passing to model
        if melspec.ndim == 2:
            melspec = torch.tensor(melspec).unsqueeze(0).float()
        else:
            melspec = torch.tensor(melspec).float()

        label_vec = torch.tensor(label_vec).float()
        return melspec, label_vec



class EfficientNetFrozen(nn.Module):
    """
    EfficientNetâ€�B0 backbone where:
      - All layers up through blocks.5 are frozen.
      - Only backbone.blocks.6 and backbone.bn2 are trainable.
      - The classifier head is trainable.
    """
    def __init__(self, model_name="efficientnet_b0", n_classes=206, unfreeze_blocks=["blocks.6"]):
        super().__init__()
        # 1) Load pretrained EfficientNetâ€�B0, no head:
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=1,
            num_classes=0
        )
        # 2) First, freeze everything:
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 3) Unfreeze only the specified blocks (e.g. "blocks.6") and final batchnorm ("bn2"):
        #    If you want to unfreeze more, add them by name in unfreeze_blocks.
        for name, param in self.backbone.named_parameters():
            # â€œblocks.6.â€� is the final MBConv block in B0; â€œbn2â€� is the last batchnorm
            if any([name.startswith(block_name) for block_name in unfreeze_blocks]) \
               or name.startswith("bn2"):
                param.requires_grad = True

        # 4) Build a new classifier head on top of pooled features:
        num_features = self.backbone.num_features  # should be 1280 for B0
        self.classifier = nn.Linear(num_features, n_classes)
        # Ensure classifier is trainable:
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)     # â†’ (B, 1280)
        logits = self.classifier(feats)  # â†’ (B, 206)
        return logits



def criterion(logits, targets):
    return sigmoid_focal_loss(inputs=logits, targets=targets, alpha=0.25, gamma=2.0, reduction="mean")

def train_model(model, train_dl, val_dl, optimizer, criterion, num_epochs=10, patience=3, fold=0):
    model.to(DEVICE)

    best_auc = 0
    epochs_since_improvement = 0
    best_model_state = None

    # sanity check: Print trainable layers (debugging)
    print("âœ… Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch+1} - Training"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in tqdm(val_dl, desc=f"Epoch {epoch+1} - Validation"):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        aucs = []
        binarized_targets = (all_targets >= 0.5).astype(int)

        for i in range(binarized_targets.shape[1]):
            if np.sum(binarized_targets[:, i]) > 0:
                try:
                    auc = roc_auc_score(binarized_targets[:, i], all_preds[:, i])
                    aucs.append(auc)
                except ValueError:
                    continue

        macro_auc = np.mean(aucs) if aucs else 0.0

        print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_dl):.4f} | Val Loss = {val_loss/len(val_dl):.4f} | Val Macro ROC-AUC = {macro_auc:.4f}")

        # Early stopping logic
        if macro_auc > best_auc:
            best_auc = macro_auc
            epochs_since_improvement = 0
            best_model_path = f"efficientnet_b0_frozen_fold{fold}_best.pth"
            torch.save(model.state_dict(), best_model_path)
            best_model_state = model.state_dict()
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"â�¹ï¸� Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    return best_auc


# === Combine Primary + Secondary Labels ===
def combine_labels(row):
    if pd.isna(row['secondary_labels']):
        secondary = []
    else:
        secondary = ast.literal_eval(row['secondary_labels'])
        # remove empty strings from parsed list
        secondary = [s for s in secondary if s.strip() != '']
    return [row['primary_label']] + secondary

# ===Convert one-hot encoded columns to label indices ===
def extract_labels(row):
    return [f"class_{i}" for i, v in enumerate(row[1:].values) if v > 0.9]

# === Setup Paths ===
DATA_PATH = "/kaggle/input/birdclef-2025"
AUDIO_PATH_LABELED = "/kaggle/input/birdclef-2025/train_audio"
AUDIO_PATH_PSEUDO  = "/kaggle/input/birdclef-2025/train_soundscapes"


# === Load Data ===
df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))

df['labels'] = df.apply(combine_labels, axis=1)
df['source'] = 'train'

pseudo_df = pd.read_csv("/kaggle/input/pseudo-labels-new/pseudo_labels.csv")

pseudo_df['labels'] = pseudo_df.apply(extract_labels, axis=1)
pseudo_df['source'] = 'pseudo'

# Add dummy filename column to satisfy dataset requirements
pseudo_df['filename'] = pseudo_df['row_id'].apply(lambda x: x.split('_')[0] + '.ogg')

# Retain only columns needed by BirdDataset
pseudo_df = pseudo_df[['filename', 'labels', 'source']]

# Match schema with df
df = pd.concat([df, pseudo_df], ignore_index=True)

# Collect labeled files in format: '1139490/CSA36385.ogg'
labeled_files = set()
for species_folder in os.listdir(AUDIO_PATH_LABELED):
    species_path = os.path.join(AUDIO_PATH_LABELED, species_folder)
    if os.path.isdir(species_path):
        for file in os.listdir(species_path):
            labeled_files.add(f"{species_folder}/{file}")

pseudo_files = set(os.listdir(AUDIO_PATH_PSEUDO))

def is_valid_file(row):
    if row['source'] == 'train':
        return row['filename'] in labeled_files
    else:
        return row['filename'] in pseudo_files

df = df[df.apply(is_valid_file, axis=1)].reset_index(drop=True)
print("âœ… Filtered df shape:", df.shape)
print("ğŸ“Š Remaining sources:", df['source'].value_counts())

# === Fit MultiLabelBinarizer ===
mlb = MultiLabelBinarizer()
mlb.fit(df['labels'])  # ğŸ”¥ Only once, on full label set

NUM_CLASSES = len(mlb.classes_)

NUM_FOLDS = 2
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

df_labeled = df[(df['source'] == 'train') & (df['primary_label'].notna())].copy()
if df_labeled.empty:
    raise ValueError("â�Œ No labeled data left after filtering. Check your audio folders and filenames.")

# Get all pseudo-labeled rows (added after split)
df_pseudo = df[df['source'] == 'pseudo'].copy()
fold_aucs = []


fold_aucs = []
best_fold_index = -1
best_auc_across_folds = -1.0

for fold, (train_idx, val_idx) in enumerate(skf.split(df_labeled, df_labeled['primary_label'])):
    print(f"\n===== Fold {fold + 1} / {NUM_FOLDS} =====")

    # Build fold splits from labeled data
    train_df = df_labeled.iloc[train_idx].reset_index(drop=True)
    val_df   = df_labeled.iloc[val_idx].reset_index(drop=True)

    # Add pseudo-labeled data to training set
    train_df = pd.concat([train_df, df_pseudo], ignore_index=True)

    # Datasets and loaders
    train_ds = BirdDataset(train_df, mlb, augment = False, mixup = True, noise_std = 0.1)
    val_ds   = BirdDataset(val_df, mlb, augment = False, mixup = False, noise_std = 0.0)

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    # Model and optimizer
    model = EfficientNetFrozen(model_name='efficientnet_b0', n_classes=NUM_CLASSES, unfreeze_blocks=["blocks.5", "blocks.6"]).to(DEVICE)

    trainable_params = list(model.classifier.parameters()) + [
        p for n, p in model.backbone.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-5)

    # Train model with early stopping and save best checkpoint
    fold_auc = train_model(
        model, train_dl, val_dl,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=10,
        patience=3,
        fold=fold
    )

    fold_aucs.append(fold_auc)

    # Track the best model across folds
    if fold_auc > best_auc_across_folds:
        best_auc_across_folds = fold_auc
        best_fold_index = fold

# === After all folds complete ===

print(f"\nâœ… Fold AUCs: {fold_aucs}")
print(f"ğŸ“Š Average ROC-AUC across folds: {np.mean(fold_aucs):.4f}")

# Copy the best model to a general name
src_path = f"efficientnet_b0_frozen_fold{best_fold_index}_best.pth"
dst_path = "efficientnet_b0_frozen_overall_best.pth"
shutil.copy(src_path, dst_path)

print(f"ğŸ�† Best model was from fold {best_fold_index} with AUC = {best_auc_across_folds:.4f}")
print(f"ğŸ“¦ Saved as: {dst_path}")


