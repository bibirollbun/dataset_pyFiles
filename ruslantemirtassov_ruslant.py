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


import warnings
warnings.filterwarnings('ignore')


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import timm
from tqdm.auto import tqdm


# --- Настройки ---
SEED = 42
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATA_DIR = "/kaggle/input/rfcx-species-audio-detection/train/"
CSV_PATH = "/kaggle/input/rfcx-species-audio-detection/train_tp.csv"


# --- Загрузка CSV ---
df = pd.read_csv(CSV_PATH)
df = df[['recording_id', 'species_id']].drop_duplicates()

# --- Словари классов ---
LABELS = sorted(df['species_id'].unique())
LABEL2ID = {label: i for i, label in enumerate(LABELS)} 
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# train/valid split
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['species_id'])

num_classes = len(LABEL2ID)
print(f"Найдено классов: {num_classes}")


# --- SpecAugment + загрузка аудио ---
def spec_augment(mel, freq_mask_param=15, time_mask_param=25):
    mel = mel.copy()
    # freq mask
    num_mel_channels = mel.shape[0]
    f = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, num_mel_channels - f)
    mel[f0:f0+f, :] = 0
    # time mask
    num_time_steps = mel.shape[1]
    t = np.random.randint(0, time_mask_param)
    t0 = np.random.randint(0, num_time_steps - t)
    mel[:, t0:t0+t] = 0
    return mel

def load_audio(fp, sr=32000, n_mels=128, duration=5, augment=False):
    y, _ = librosa.load(fp, sr=sr)
    y = np.asfortranarray(y) 
    y = librosa.util.fix_length(y, size=sr * duration)
    y = y / (np.max(np.abs(y)) + 1e-6)

    # --- Аугментации ---
    if augment:
        if np.random.rand() < 0.5:
            y = np.flip(y)
        if np.random.rand() < 0.5:
            n_steps = np.random.uniform(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        if np.random.rand() < 0.5:
            rate = np.random.uniform(0.8, 1.2)
            try:
                y = librosa.effects.time_stretch(y=y, rate=rate)
            except Exception as e:
                print(f"time_stretch error ({fp}): {e}")
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.005, y.shape)
            y = y + noise

    # --- Преобразование в мел-спектрограмму ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    # --- Паддинг по времени ---
    max_len = sr // 512 * duration
    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0, 0), (0, max_len - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]

    return torch.tensor(mel_db).unsqueeze(0).float()


class BirdFrogDataset(Dataset):
    def __init__(self, df, data_dir, num_classes, augment=False):
        self.df = df
        self.data_dir = data_dir
        self.augment = augment
        self.num_classes = num_classes
        self.labels = df.groupby("recording_id")["species_id"].apply(list).to_dict()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rec_id = list(self.labels.keys())[idx]
        fp = os.path.join(self.data_dir, rec_id + ".flac")
        x = load_audio(fp, augment=self.augment)

        y = torch.zeros(self.num_classes)
        for sid in self.labels[rec_id]:
            y[sid] = 1.0
        return x, y


train_ds = BirdFrogDataset(train_df, DATA_DIR, num_classes, augment=True)
valid_ds = BirdFrogDataset(valid_df, DATA_DIR, num_classes, augment=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=0)


# --- Модель ---
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, in_chans=1, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

model = AudioClassifier(num_classes).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
criterion = nn.BCEWithLogitsLoss()


def accuracy_multi(preds, y, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    correct = (preds == y).float().mean().item()
    return correct


# --- Train / Eval ---
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    correct, count = 0, 0
    for x, y in tqdm(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += accuracy_multi(preds, y)
        count += 1
    return total_loss / count, correct / count

def eval_epoch(model, loader):
    model.eval()
    total_loss = 0
    correct, count = 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item() * len(y)
            correct += accuracy_multi(preds, y)
            count += 1
    return total_loss / count, correct / count


# --- Обучение с ранней остановкой ---
best_val = 0
patience, wait = 3, 0
num_epochs = 3

for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

    if epoch == 1:
        for param in model.backbone.parameters():
            param.requires_grad = True

    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = eval_epoch(model, valid_loader)
    scheduler.step()

    print(f"Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
          f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_model_singlelabel.pth")
        print(f"New best val_acc={best_val:.3f} — модель сохранена")
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            break

