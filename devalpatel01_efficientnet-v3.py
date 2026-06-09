# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install timm

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import timm

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

from torch import nn
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



class EfficientNetFrozen(nn.Module):
    def __init__(self, model_name='efficientnet_b0', n_classes=206):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=1, num_classes=0)
        
        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.backbone.num_features, n_classes)

    def forward(self, x):
        with torch.no_grad():  # Optional: safer if frozen
            features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def train_model(model, train_dl, val_dl, optimizer, criterion, num_epochs=10, patience=3, fold=0):
    model.to(DEVICE)

    best_auc = 0
    epochs_since_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch+1} - Training"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in tqdm(val_dl, desc=f"Epoch {epoch+1} - Validation"):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        aucs = []
        for i in range(all_targets.shape[1]):
            if np.sum(all_targets[:, i]) > 0:
                auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
                aucs.append(auc)
        macro_auc = np.mean(aucs)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_dl):.4f} | Val Loss = {val_loss/len(val_dl):.4f} | Val Macro ROC-AUC = {macro_auc:.4f}")

        # Early stopping logic
        if macro_auc > best_auc:
            best_auc = macro_auc
            epochs_since_improvement = 0
            best_model_path = f"efficientnet_b0_frozen_fold{fold}_best.pth"
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"â�¹ï¸� Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    return best_auc



# === Setup Paths ===
DATA_PATH = "/kaggle/input/birdclef-2025"
AUDIO_PATH = os.path.join(DATA_PATH, "train_audio")

# === Load Data ===
df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))

# === Combine Primary + Secondary Labels ===
def combine_labels(row):
    if pd.isna(row['secondary_labels']):
        secondary = []
    else:
        secondary = ast.literal_eval(row['secondary_labels'])
        # remove empty strings from parsed list
        secondary = [s for s in secondary if s.strip() != '']
    return [row['primary_label']] + secondary

df['labels'] = df.apply(combine_labels, axis=1)

# === Fit MultiLabelBinarizer ===
mlb = MultiLabelBinarizer()
mlb.fit(df['labels'])  # ğŸ”¥ Only once, on full label set

NUM_CLASSES = len(mlb.classes_)

# === Define Dataset ===
class BirdDataset(Dataset):
    def __init__(self, df, audio_dir, mlb):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.mlb = mlb

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = load_audio(os.path.join(self.audio_dir, row['filename']))
        melspec = audio_to_logmel(y)
        melspec = torch.tensor(melspec).unsqueeze(0).float()

        label_list = row['labels']  # Already combined
        label_vec = self.mlb.transform([label_list])[0]

        # âœ… Debug: Confirm shape
        if idx == 0:
            print("Label list:", label_list)
            print("Transformed label shape:", label_vec.shape)

        return melspec, torch.tensor(label_vec).float()
NUM_FOLDS = 3
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

fold_aucs = []



for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
    print(f"\n===== Fold {fold + 1} / {NUM_FOLDS} =====")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_ds = BirdDataset(train_df, AUDIO_PATH, mlb)
    val_ds = BirdDataset(val_df, AUDIO_PATH, mlb)

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    model = EfficientNetFrozen(model_name='efficientnet_b0', n_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
 
    fold_auc = train_model(model, train_dl, val_dl, optimizer, criterion, num_epochs=10, patience=3, fold=fold)
    fold_aucs.append(fold_auc)

    torch.save(model.state_dict(), f"efficientnet_b0_frozen_fold{fold}.pth")

print(f"\nâœ… Average ROC-AUC across {NUM_FOLDS} folds: {np.mean(fold_aucs):.4f}")

