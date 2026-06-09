# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





import pandas as pd
import ast

metadata = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
metadata.head()



def parse_secondary_labels(val):
    if isinstance(val, str):
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [s for s in parsed if s] 
    return [] 


metadata["secondary_labels_list"] = metadata["secondary_labels"].apply(parse_secondary_labels)


import pandas as pd

taxonomy = pd.read_csv("/kaggle/input/birdclef-2025/taxonomy.csv")

label_list = taxonomy['primary_label'].tolist()
label_to_idx = {label: idx for idx, label in enumerate(label_list)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

print("Total classes:", len(label_list))


metadata["parsed_labels"] = metadata.apply(
    lambda row: list(set([row["primary_label"]] + row["secondary_labels_list"])),
    axis=1
)


import os
import torch
import pandas as pd
from pathlib import Path
import numpy as np

import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


T = 1 + int((32000 * 5 - 2048) / 512)
T


class BirdCLEFDatasetOnTheFly(Dataset):
    def __init__(self, df, label_to_idx, audio_root, sr=32000, n_mels=128, target_len=309, random_crop=False):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.audio_root = audio_root
        self.sr = sr
        self.n_mels = n_mels
        self.target_len = target_len
        self.random_crop = random_crop

    def __len__(self):
        return len(self.df)


    def audio_to_logmelspec(self, path):
        y, _ = librosa.load(path, sr=self.sr, mono=True)

        if self.random_crop:
            segment_samples = int(self.sr * 5.0)
            if len(y) > segment_samples:
                max_start = len(y) - segment_samples
                start = np.random.randint(0, max_start)
                y = y[start:start + segment_samples]
            else:
                y = np.pad(y, (0, max(0, segment_samples - len(y))))

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=2048,
            hop_length=512,
            n_mels=self.n_mels,
            fmin=20,
            fmax=16000
        )
        logmel = librosa.power_to_db(mel).astype(np.float32)
        return logmel

    def pad_or_crop(self, logmel):
        _, t = logmel.shape
        if t < self.target_len:
            pad_width = self.target_len - t
            logmel = np.pad(logmel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            logmel = logmel[:, :self.target_len]
        return logmel

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # ĞŸÑƒÑ‚ÑŒ Ğº Ğ°ÑƒĞ´Ğ¸Ğ¾Ñ„Ğ°Ğ¹Ğ»Ñƒ
        audio_path = os.path.join(self.audio_root, row["filename"])
        
        # Ğ“ĞµĞ½ĞµÑ€Ğ°Ñ†Ğ¸Ñ� log-mel Ğ½Ğ° Ğ»ĞµÑ‚Ñƒ
        logmel = self.audio_to_logmelspec(audio_path)
        logmel = self.pad_or_crop(logmel)
        mel_tensor = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)  # [1, 128, T]

        # Multi-hot Ğ²ĞµĞºÑ‚Ğ¾Ñ€
        label_vec = torch.zeros(len(self.label_to_idx))
        for label in row["parsed_labels"]:
            if label in self.label_to_idx:
                label_vec[self.label_to_idx[label]] = 1.0

        return mel_tensor, label_vec


# x, y = dataset[17]
# print(x.shape)  # [1, 128, 1312]
# print(y.sum())  # >= 1


!pip install -q iterative-stratification


from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd

def multilabel_stratified_train_test_split(df, label_to_idx, test_size=0.2, random_state=42):
    """
    Ğ�Ğ½Ğ°Ğ»Ğ¾Ğ³ train_test_split, Ğ½Ğ¾ Ñ� ÑƒÑ‡Ñ‘Ñ‚Ğ¾Ğ¼ Ğ¼ÑƒĞ»ÑŒÑ‚Ğ¸-ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ²Ğ¾Ğ¹ Ñ€Ğ°Ğ·Ğ¼ĞµÑ‚ĞºĞ¸.
    
    Ğ�Ñ€Ğ³ÑƒĞ¼ĞµĞ½Ñ‚Ñ‹:
        df (pd.DataFrame): Ğ´Ğ°Ñ‚Ğ°Ñ„Ñ€ĞµĞ¹Ğ¼, Ñ�Ğ¾Ğ´ĞµÑ€Ğ¶Ğ°Ñ‰Ğ¸Ğ¹ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºÑƒ 'parsed_labels'
        label_to_idx (dict): Ñ�Ğ»Ğ¾Ğ²Ğ°Ñ€ÑŒ Ñ�Ğ¾Ğ¾Ñ‚Ğ²ĞµÑ‚Ñ�Ñ‚Ğ²Ğ¸Ñ� Ğ¼ĞµÑ‚Ğ¾Ğº â†’ Ğ¸Ğ½Ğ´ĞµĞºÑ�Ñ‹
        test_size (float): Ğ´Ğ¾Ğ»Ñ� Ğ¾Ñ‚ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ¸ Ğ² Ñ‚ĞµÑ�Ñ‚ (Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ�)
        random_state (int): seed Ğ´Ğ»Ñ� Ğ²Ğ¾Ñ�Ğ¿Ñ€Ğ¾Ğ¸Ğ·Ğ²Ğ¾Ğ´Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸

    Ğ’Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµÑ‚:
        train_df, val_df â€” Ñ�Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ¿Ğ¾Ğ´Ğ¼Ğ½Ğ¾Ğ¶ĞµÑ�Ñ‚Ğ²Ğ°
    """
    # Ğ¡Ğ¾Ğ·Ğ´Ğ°Ñ‘Ğ¼ multi-hot Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ñƒ
    num_samples = len(df)
    num_classes = len(label_to_idx)
    y_multihot = np.zeros((num_samples, num_classes))

    for i, label_list in enumerate(df["parsed_labels"]):
        for label in label_list:
            if label in label_to_idx:
                y_multihot[i, label_to_idx[label]] = 1

    # Ğ’Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ñ�ĞµĞ¼ Ğ½ÑƒĞ¶Ğ½Ğ¾Ğµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ„Ğ¾Ğ»Ğ´Ğ¾Ğ², Ñ‡Ñ‚Ğ¾Ğ±Ñ‹ Ğ¿Ğ¾Ğ»ÑƒÑ‡Ğ¸Ñ‚ÑŒ Ğ½ÑƒĞ¶Ğ½Ñ‹Ğ¹ test_size
    n_splits = int(1 / test_size)
    if not 0 < test_size < 1:
        raise ValueError("test_size Ğ´Ğ¾Ğ»Ğ¶ĞµĞ½ Ğ±Ñ‹Ñ‚ÑŒ Ğ¼ĞµĞ¶Ğ´Ñƒ 0 Ğ¸ 1")

    # Ğ Ğ°Ğ·Ğ±Ğ¸Ğ²Ğ°ĞµĞ¼ Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ğ¾Ğ´Ğ¸Ğ½ Ñ€Ğ°Ğ· (fold==0)
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(mskf.split(df, y_multihot))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df


# subset_df = metadata.sample(frac=0.1, random_state=42).reset_index(drop=True)


train_df, val_df = multilabel_stratified_train_test_split(metadata, label_to_idx, test_size=0.2)


train_filenames = set(train_df["filename"])
val_filenames = set(val_df["filename"])

intersection = train_filenames & val_filenames
print(f"ğŸ”� ĞŸĞµÑ€ĞµÑ�ĞµÑ‡ĞµĞ½Ğ¸Ğ¹ Ğ¿Ğ¾ filename: {len(intersection)}")
if intersection:
    print("ĞŸÑ€Ğ¸Ğ¼ĞµÑ€Ñ‹ Ñ�Ğ¾Ğ²Ğ¿Ğ°Ğ´ĞµĞ½Ğ¸Ğ¹:", list(intersection)[:5])


cols_to_check = ["collection", "author", "latitude", "longitude"]

for col in cols_to_check:
    if col in train_df.columns:
        train_vals = set(train_df[col].dropna())
        val_vals = set(val_df[col].dropna())
        common = train_vals & val_vals
        print(f"ğŸ”� {col}: {len(common)} Ğ¾Ğ±Ñ‰Ğ¸Ñ… Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğ¹")


from collections import Counter

def get_label_counts(df):
    c = Counter()
    for labels in df["parsed_labels"]:
        c.update(labels)
    return c

train_counts = get_label_counts(train_df)
val_counts = get_label_counts(val_df)

print(f"Ğ£Ğ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ñ… ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ² Ğ² train: {len(train_counts)}")
print(f"Ğ£Ğ½Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ñ… ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ² Ğ² val: {len(val_counts)}")


train_dataset = BirdCLEFDatasetOnTheFly(
    df=train_df,
    label_to_idx=label_to_idx,
    audio_root="/kaggle/input/birdclef-2025/train_audio",
    random_crop=True
)

val_dataset = BirdCLEFDatasetOnTheFly(
    df=val_df,
    label_to_idx=label_to_idx,
    audio_root="/kaggle/input/birdclef-2025/train_audio",
    random_crop=True
)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=0)


import timm
import torch.nn as nn

class BirdCLEFNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s", pretrained=True, in_chans=1)
        self.backbone.global_pool = nn.Identity()  # ÑƒĞ±Ğ¸Ñ€Ğ°ĞµĞ¼ pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)  # [B, C, H, W]
        x = self.pooling(x).squeeze(-1).squeeze(-1)  # [B, C]
        x = self.classifier(x)  # [B, num_classes]
        return x


import torch
import torch.nn.functional as F
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BirdCLEFNet(num_classes=len(label_to_idx)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()


device


from tqdm import tqdm


import numpy as np
from sklearn.metrics import roc_auc_score

def safe_macro_auc(y_true, y_pred):
    """Compute macro-averaged ROC AUC, skipping invalid classes."""
    scores = []
    for i in range(y_true.shape[1]):
        y_col = y_true[:, i]
        p_col = y_pred[:, i]
        if (y_col == 1).sum() > 0 and (y_col == 0).sum() > 0:
            try:
                score = roc_auc_score(y_col, p_col)
                scores.append(score)
            except:
                continue
    return np.mean(scores) if scores else float("nan")


import csv

LOG_PATH = "/kaggle/working/train_log.csv"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Ğ—Ğ°Ğ¿Ğ¸Ñ�Ñ‹Ğ²Ğ°ĞµĞ¼ ÑˆĞ°Ğ¿ĞºÑƒ Ğ´Ğ»Ñ� Ğ»Ğ¾Ğ³Ğ¾Ğ²
with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_auc"])

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(dataloader, desc=f"[Epoch {epoch}] Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_one_epoch(model, dataloader, loss_fn, device, epoch):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=f"[Epoch {epoch}] Val", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            running_loss += loss.item()
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)
    val_auc = safe_macro_auc(y_true, y_pred)
    return running_loss / len(dataloader), val_auc


NUM_EPOCHS = 10
best_auc = -1.0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
    val_loss, val_auc = validate_one_epoch(model, val_loader, loss_fn, device, epoch)

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ROC AUC: {val_auc:.5f}")

    # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ² Ğ»Ğ¾Ğ³-Ñ„Ğ°Ğ¹Ğ»
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss, val_auc])

    # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ, ĞµÑ�Ğ»Ğ¸ ÑƒĞ»ÑƒÑ‡ÑˆĞ¸Ğ»Ğ°Ñ�ÑŒ
    if val_auc > best_auc:
        best_auc = val_auc
        model_filename = f"baseline2_randomcrop_epoch{epoch}_auc{val_auc:.5f}.pth"
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/{model_filename}")
        print(f"âœ… New best AUC: {best_auc:.5f} â€” saved as {model_filename}")













