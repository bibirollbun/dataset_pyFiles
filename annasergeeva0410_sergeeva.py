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


!pip install --no-build-isolation -q audiomentations


import os, numpy as np, pandas as pd, librosa, torch, torch.nn as nn, torch.optim as optim
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

AUDIO_DIR = "/kaggle/input/itmo-acoustic-event-detectin-2025/audio_train/train"
TEST_DIR = "/kaggle/input/itmo-acoustic-event-detectin-2025/audio_test/test"
CSV_PATH = "/kaggle/input/itmo-acoustic-event-detectin-2025/train.csv"
SAMPLE_RATE, N_MELS, DURATION = 11025, 64, 2
MAX_LEN = SAMPLE_RATE * DURATION
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10

# Загрузка и кодировка 
df = pd.read_csv(CSV_PATH)
le = LabelEncoder()
df["label_idx"] = le.fit_transform(df["label"])
NUM_CLASSES = len(le.classes_)

class_weights = compute_class_weight('balanced', classes=np.unique(df["label_idx"]), y=df["label_idx"])
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Аугментации вручную 
def apply_augmentations(y, sr):
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.005, y.shape)
        y += noise
    if np.random.rand() < 0.3:
        rate = np.random.uniform(0.8, 1.2)
        try:
            y = librosa.effects.time_stretch(y, rate)
        except:
            pass
    if np.random.rand() < 0.4:
        steps = np.random.randint(-2, 3)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)  # ← исправление
    if np.random.rand() < 0.3:
        shift = int(np.random.uniform(-0.2, 0.2) * len(y))
        y = np.roll(y, shift)
    return y


TIME_MASK = T.TimeMasking(time_mask_param=15)
FREQ_MASK = T.FrequencyMasking(freq_mask_param=8)

# Dataset 
class AudioDataset(Dataset):
    def __init__(self, df, augment=False, path=AUDIO_DIR):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.path = path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, sr = librosa.load(os.path.join(self.path, row["fname"]), sr=SAMPLE_RATE)
        if len(y) < MAX_LEN:
            y = np.pad(y, (0, MAX_LEN - len(y)))
        else:
            y = y[:MAX_LEN]
        if self.augment:
            y = apply_augmentations(y, sr)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel)
        log_mel = torch.tensor(log_mel).unsqueeze(0)

        if self.augment:
            log_mel = TIME_MASK(log_mel)
            log_mel = FREQ_MASK(log_mel)

        return log_mel.float(), row["label_idx"]

    def __len__(self): return len(self.df)

class TestDataset(Dataset):
    def __init__(self, files, path=TEST_DIR):
        self.files = sorted(files)
        self.path = path

    def __getitem__(self, idx):
        y, sr = librosa.load(os.path.join(self.path, self.files[idx]), sr=SAMPLE_RATE)
        if len(y) < MAX_LEN: y = np.pad(y, (0, MAX_LEN - len(y)))
        else: y = y[:MAX_LEN]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel)
        return torch.tensor(log_mel).unsqueeze(0).float(), self.files[idx]

    def __len__(self): return len(self.files)

# Mixup 
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[index], y, y[index], lam

# Label Smoothing 
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, x, target):
        logprobs = nn.functional.log_softmax(x, dim=-1)
        true_dist = torch.zeros_like(logprobs).fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))

# Модель 
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveMaxPool2d((4, 4)),
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x): return self.net(x)

# Обучение 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = AudioDataset(df, augment=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

model = CNNClassifier(NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = LabelSmoothingLoss(NUM_CLASSES, smoothing=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

best_f1, patience = 0, 0

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        x_mix, y1, y2, lam = mixup(xb, yb)
        out = model(x_mix)
        loss = lam * criterion(out, y1) + (1 - lam) * criterion(out, y2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Быстрая оценка (сэмпл из train)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(device)
            preds += model(xb).argmax(1).cpu().tolist()
            targets += yb.tolist()
            if len(preds) > 2000: break

    f1 = f1_score(targets, preds[:len(targets)], average="macro")
    print(f"Epoch {epoch+1}/{EPOCHS} — F1: {f1:.4f}")
    scheduler.step(f1)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_model.pth")
        patience = 0
    else:
        patience += 1
        if patience >= PATIENCE:
            print("Early stopping")
            break

# ============ Сабмит ============
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

test_ds = TestDataset(os.listdir(TEST_DIR))
test_loader = DataLoader(test_ds, batch_size=32)
all_preds, all_names = [], []

with torch.no_grad():
    for xb, names in test_loader:
        xb = xb.to(device)
        preds = model(xb).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_names.extend(names)

decoded_preds = le.inverse_transform(all_preds)
submission = pd.DataFrame({"fname": all_names, "label": decoded_preds})
submission.to_csv("submission.csv", index=False)

