# importing libraries
import librosa
import numpy as np
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import glob
import torch
import ast

import random
from torchvision import models
from sklearn.model_selection import StratifiedKFold

from warnings import filterwarnings
filterwarnings('ignore')

import seaborn as sns
from ast import literal_eval
import os
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import timm
import torch.nn.functional as F
import gc
from torchvision import transforms
import geopandas as gpd
from shapely.geometry import Point
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import joblib
pd.set_option('display.max_colwidth', None)
from concurrent.futures import ProcessPoolExecutor, as_completed

# create the config class
class config:
    train_audio = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    sample_solution = '/kaggle/input/birdclef-2025/sample_submission.csv'
    test_soundscape = '/kaggle/input/birdclef-2025/test_soundscapes'
    save_path = '/kaggle/working/'

    sampling_rate = 32000
    num_classes = 206
    n_mels = 128
    fmin = 20
    fmax = 16000
    chunk_length = 15  # seconds
    n_fft = 2048
    hop_length = 512
    seed = 42

    batch_size = 64
    epochs = 10
    learning_rate = 2e-3
    num_folds = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
def set_seed(seed):
    # see in config see I don't want to return anything cool
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"All done! Set seed: {seed}")

set_seed(config.seed)  # <-- Add this line

train_df = pd.read_csv(config.train_csv)


# deal with secondary_label , type and filenames columns
for col in ('secondary_labels', 'type'):
    train_df[col] = train_df[col].apply(lambda x: "###".join(literal_eval(x)))

train_df['filename'] = train_df['filename'].apply(lambda x: config.train_audio + "/" + x)

def load_audio(path, duration=config.chunk_length, sr=config.sampling_rate):
    y, _ = librosa.load(path, sr=sr, mono=True)
    length = sr * duration

    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    else:
        y = y[:length]

    return y

def audio_to_melspec(audio):
    melspec = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sampling_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        pad_mode="reflect",
        norm='slaney',
        htk=True,
        center=True,
    )
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    return melspec_db



class BirdDataset(Dataset):
    def __init__(self, df, label_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(config.train_audio, row['filename'])
        audio = load_audio(path)
        melspec = audio_to_melspec(audio)

        # Normalize to fixed shape
        expected_shape = (config.n_mels, int(config.chunk_length * config.sampling_rate / config.hop_length))
        if melspec.shape[1] < expected_shape[1]:
            pad_width = expected_shape[1] - melspec.shape[1]
            melspec = np.pad(melspec, ((0, 0), (0, pad_width)))
        else:
            melspec = melspec[:, :expected_shape[1]]

        # Convert to 3-channel
        melspec_rgb = np.stack([melspec, melspec, melspec])
        melspec_tensor = torch.tensor(melspec_rgb).float()

        if self.transform:
            melspec_tensor = self.transform(melspec_tensor)

        label = torch.zeros(config.num_classes)
        label[row['primary_label']] = 1.0

        if isinstance(row['secondary_labels'], str):
            for sec in row['secondary_labels'].split("###"):
                if sec in self.label_map:
                    label[self.label_map[sec]] = 0.5

        return melspec_tensor, label
def get_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(p=0.5),
        transforms.Normalize([0.485]*3, [0.229]*3)
    ])


class AttBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='sigmoid'):
        super().__init__()
        self.att = nn.Conv1d(in_features, out_features, 1, bias=True)
        self.cla = nn.Conv1d(in_features, out_features, 1, bias=True)
        self.activation = activation

    def forward(self, x):
        # x: (batch, features, time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        if self.activation == 'sigmoid':
            cla = torch.sigmoid(self.cla(x))
        else:
            cla = self.cla(x)
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att

class EfficientNetSED(nn.Module):
    def __init__(self, num_classes, n_mels=128):
        super().__init__()
        base = models.efficientnet_b0(pretrained=True)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))  # keep time dim
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(base.classifier[1].in_features, 256)
        self.att_block = AttBlock(256, num_classes, activation='sigmoid')

    def forward(self, x):
        # x: (batch, 3, n_mels, time)
        x = self.features(x)
        x = self.avgpool(x)  # (batch, features, 1, time)
        x = x.squeeze(2)     # (batch, features, time)
        x = x.permute(0, 2, 1)  # (batch, time, features)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1)  # (batch, features, time)
        clipwise_output, _ = self.att_block(x)
        return clipwise_output

def get_model():
    return EfficientNetSED(num_classes=config.num_classes, n_mels=config.n_mels)

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0

    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(config.device), targets.to(config.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


from sklearn.metrics import roc_auc_score

def calculate_auc(targets, outputs):
    num_classes = targets.shape[1]
    aucs = []
    probs = 1 / (1 + np.exp(-outputs))
    for i in range(num_classes):
        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)
    return np.mean(aucs) if aucs else 0.0

def validate_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    auc = calculate_auc(all_targets, all_outputs)
    return avg_loss, auc
def run_training(df):
    label_map = {label: i for i, label in enumerate(sorted(df['primary_label'].unique()))}
    df['primary_label'] = df['primary_label'].map(label_map)

    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)

    for fold, (train_idx, _) in enumerate(skf.split(df, df['primary_label'])):
        print(f"\n=== Fold {fold} ===")
        train_df = df.iloc[train_idx]

        train_dataset = BirdDataset(train_df, label_map, transform=get_transforms())
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

        model = get_model().to(config.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

        for epoch in range(config.epochs):
            print(f"\nEpoch {epoch+1}/{config.epochs}")
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            print(f"Train Loss: {train_loss:.4f}")

        torch.save(model.state_dict(), f"{config.save_path}/sed_fold{fold}.pth")
        print("Model saved!")



run_training(train_df)

