import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from glob import glob
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
import random
import json


DATA_DIR = "/kaggle/input/freesound-audio-tagging/"   
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "audio_train")
TEST_AUDIO_DIR = os.path.join(DATA_DIR, "audio_test")
META_DIR = os.path.join(DATA_DIR, "FSDKaggle2018.meta")

TRAIN_CSV = os.path.join(DATA_DIR, "train_post_competition.csv")   
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")     

print("Paths:")
print("TRAIN_AUDIO_DIR", TRAIN_AUDIO_DIR)
print("TEST_AUDIO_DIR", TEST_AUDIO_DIR)
print("TRAIN_CSV", TRAIN_CSV)
print("SAMPLE_SUB", SAMPLE_SUB)



# dataset_freesound.py

import torch
import numpy as np
import os
from torch.utils.data import Dataset
import librosa

SR = 44100
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
MAX_SECONDS = 10

def load_audio(path, sr=SR, max_secs=MAX_SECONDS):
    y, sr = librosa.load(path, sr=sr, mono=True, duration=max_secs)
    target_len = int(sr * max_secs)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr

def log_mel_spectrogram(y, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)

class FreesoundDataset(Dataset):
    def __init__(self, df, audio_dir, mode='train', spec_augment=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.mode = mode
        self.spec_augment = spec_augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        path = os.path.join(self.audio_dir, row["fname"])
        y, sr = load_audio(path)
        mel = log_mel_spectrogram(y, sr)

        if self.mode == 'train' and self.spec_augment:
            t0 = np.random.randint(0, mel.shape[1]//2)
            t1 = np.random.randint(mel.shape[1]//2, mel.shape[1])
            mel[:, t0:t1] = 0
            f0 = np.random.randint(0, mel.shape[0]//2)
            f1 = np.random.randint(mel.shape[0]//2, mel.shape[0])
            mel[f0:f1, :] = 0

        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        mel = np.expand_dims(mel, 0)
        return torch.tensor(mel, dtype=torch.float32), torch.tensor(row["label_idx"])


train_df = pd.read_csv(TRAIN_CSV)
train_df.head()



labels = sorted(train_df['label'].unique())
label2idx = {l:i for i,l in enumerate(labels)}
idx2label = {i:l for l,i in label2idx.items()}

train_df['label_idx'] = train_df['label'].map(label2idx)



class ImprovedCNN(nn.Module):
    def __init__(self, n_classes=len(labels)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2,2)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
class HybridAudioModel(nn.Module):
    def __init__(self, n_classes=len(labels)):
        super().__init__()
        self.efficientnet = efficientnet_b0(weights="IMAGENET1K_V1")
        self.efficientnet.classifier = nn.Identity() 
        self.cnn = ImprovedCNN() 
        self.fc = None
        self.n_classes = n_classes

    def forward(self, x):
        x_eff = x.repeat(1, 3, 1, 1)  
        eff_feat = self.efficientnet(x_eff)
        cnn_feat = self.cnn(x)  
        feat = torch.cat([eff_feat, cnn_feat], dim=1)

        if self.fc is None:
            self.fc = nn.Linear(feat.size(1), self.n_classes).to(feat.device)

        out = self.fc(feat)
        return out


import numpy as np

def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
            break
    return score

def mapk(actuals, predicted_lists, k=3):
    return np.mean([apk(a, p, k) for a,p in zip(actuals, predicted_lists)])




if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = HybridAudioModel(n_classes=len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(train_df, train_df['label_idx']))

    train_ds = FreesoundDataset(train_df.loc[train_idx], TRAIN_AUDIO_DIR, mode='train', spec_augment=False)
    val_ds = FreesoundDataset(train_df.loc[val_idx], TRAIN_AUDIO_DIR, mode='val')

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    EPOCHS = 5
    best_map3 = -1

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Train epoch {epoch+1}"):
            x, y = x.to(device).float(), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device).float()
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                top3 = np.argsort(probs, axis=1)[:, ::-1][:, :3]
                for t, p in zip(y.numpy(), top3):
                    trues.append(idx2label[t])
                    preds.append([idx2label[i] for i in p])

        current_map3 = mapk(trues, preds, k=3)
        print(f"Epoch {epoch+1} loss {epoch_loss:.4f} MAP@3 {current_map3:.4f}")

        if current_map3 > best_map3:
            best_map3 = current_map3
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")


model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
TEST_AUDIO_DIR = os.path.join(DATA_DIR, "audio_test")
sample_df = pd.read_csv(SAMPLE_SUB)
test_files = sample_df['fname'].values
def predict_file(fname, audio_dir=TEST_AUDIO_DIR):
    path = os.path.join(audio_dir, fname)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return "unknown"   
    y, sr = load_audio(path, sr=SR, max_secs=MAX_SECONDS)

    mel = log_mel_spectrogram(y, sr=sr)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    mel = np.expand_dims(mel, axis=0)
    mel = np.expand_dims(mel, axis=0)
    x = torch.tensor(mel).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top3 = np.argsort(probs)[::-1][:3]
    return " ".join([idx2label[i] for i in top3])

preds = []
for f in tqdm(test_files, desc="Predicting test set"):
    preds.append(predict_file(f))

sample_df['label'] = preds
sample_df.to_csv("submission.csv", index=False)
print("Saved submission.csv")


