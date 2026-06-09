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


DATA_DIR = "../input/cassava-leaf-disease-classification/train_images/"  # path to images
CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
OUTPUT_DIR = "./prepared_data/"


import pandas as pd
df = pd.read_csv(CSV_PATH)
df['file_path'] = df['image_id'].apply(lambda x: DATA_DIR + x)
print(df.shape)
print(df['label'].value_counts())
df.head()



from PIL import Image
import os

missing = []
corrupt = []
for fp in df['file_path'].tolist():
    if not os.path.exists(fp):
        missing.append(fp)
    else:
        try:
            with Image.open(fp) as im:
                im.verify()   # PIL verify for corruption
        except Exception as e:
            corrupt.append((fp, str(e)))

print("missing:", len(missing))
print("corrupt:", len(corrupt))
# Optionally print examples
corrupt[:5]



import matplotlib.pyplot as plt
counts = df['label'].value_counts().sort_index()
plt.figure(figsize=(6,4))
plt.bar(counts.index.astype(str), counts.values)
plt.xlabel("label")
plt.ylabel("count")
plt.title("Class distribution")
plt.show()



import matplotlib.pyplot as plt
from PIL import Image
import random

def show_samples(df, n=5, size=(3,3)):
    labels = sorted(df['label'].unique())
    plt.figure(figsize=(n*2, len(labels)*2))
    row = 0
    for lbl in labels:
        samples = df[df['label']==lbl]['file_path'].sample(n, random_state=42).tolist()
        for i, p in enumerate(samples):
            im = Image.open(p).convert('RGB')
            plt.subplot(len(labels), n, row*n + i + 1)
            plt.imshow(im); plt.axis('off')
            if i == 0:
                plt.ylabel(f"label {lbl}", rotation=0, labelpad=40)
        row += 1
    plt.tight_layout()
show_samples(df, n=5)



from PIL import Image
import numpy as np
ws, hs = [], []
for fp in df['file_path'].sample(1000, random_state=1):  # sample for speed
    with Image.open(fp) as im:
        w,h = im.size
    ws.append(w); hs.append(h)

import seaborn as sns
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(ws, bins=30); plt.title("width")
plt.subplot(1,2,2)
sns.histplot(hs, bins=30); plt.title("height")
plt.show()



# pip install albumentations timm
import os
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd

# Example: ensure df exists and has 'file_path' and 'label' columns from earlier steps
# df = pd.read_csv("...")  # make sure this is defined

train_transform = A.Compose([
    # pass size as a tuple (height, width) OR use named args height=..., width=...
    A.RandomResizedCrop((384, 384), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])
# If 'fold' is missing, create stratified folds
if 'fold' not in df.columns:
    print("Creating stratified folds...")
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df['label'])):
        df.loc[val_idx, 'fold'] = fold
    print("Fold counts:\n", df['fold'].value_counts())
else:
    print("'fold' already present. Fold counts:\n", df['fold'].value_counts())

class CassavaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = row['file_path']
        # safety: check file exists
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Image path not found: {fp}")
        img_bgr = cv2.imread(fp)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image (cv2.imread returned None): {fp}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Ensure dtype is uint8 for albumentations
        if img.dtype != 'uint8':
            img = (img * 255).astype('uint8')

        if self.transform:
            img = self.transform(image=img)['image']
        label = int(row['label'])
        return img, label

# debug dataloader
# make sure df and df['fold'] exist; here we assume it's prepared earlier
train_df = df[df['fold'] != 0].reset_index(drop=True)
ds = CassavaDataset(train_df, transform=train_transform)
dl = DataLoader(ds, batch_size=8, num_workers=4, shuffle=True, pin_memory=True)

# quick sanity check
images, labels = next(iter(dl))
print("images:", images.shape, images.dtype)   # expect (B, C, H, W), float32
print("labels:", labels.shape, labels.dtype)   # expect (B,), torch.long



import numpy as np
import torch
import torch.nn.functional as F

# one-hot helper
def one_hot(labels, num_classes, device):
    return torch.zeros(labels.size(0), num_classes, device=device).scatter_(1, labels.view(-1,1), 1)

# sample lambda from Beta
def rand_beta(alpha):
    return np.random.beta(alpha, alpha) if alpha > 0 else 1.0

# CutMix batch function
def cutmix_batch(images, labels, alpha=1.0, prob=0.5, num_classes=5):
    """
    images: Tensor (B, C, H, W)
    labels: Tensor (B,) int
    returns: mixed_images, mixed_labels (soft), lam
    """
    if np.random.rand() > prob:
        device = images.device
        return images, one_hot(labels, num_classes, device), 1.0

    lam = rand_beta(alpha)
    batch_size, _, H, W = images.size()
    index = torch.randperm(batch_size, device=images.device)
    labels2 = labels[index]

    # box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    new_images = images.clone()
    new_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

    # adjust lambda based on actual area
    area = (bbx2 - bbx1) * (bby2 - bby1)
    lam_adjusted = 1 - area / (H * W)

    device = images.device
    y1_hot = one_hot(labels, num_classes, device)
    y2_hot = one_hot(labels2, num_classes, device)
    y_mixed = lam_adjusted * y1_hot + (1 - lam_adjusted) * y2_hot

    return new_images, y_mixed, lam_adjusted

# soft cross entropy
def soft_cross_entropy(pred_logits, soft_targets):
    log_probs = F.log_softmax(pred_logits, dim=1)
    loss = - (soft_targets * log_probs).sum(dim=1).mean()
    return loss



import timm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 5
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
model = model.to(device)

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=10)  # tune T_max per epochs
scaler = torch.cuda.amp.GradScaler()  # for AMP



from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def validate(model, val_loader):
    model.eval()
    preds = []
    truths = []
    running_loss = 0.0
    n = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # forward
            logits = model(images)
            loss = F.cross_entropy(logits, labels)  # integer labels for val
            running_loss += loss.item() * images.size(0)
            n += images.size(0)
            ps = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(ps)
            truths.append(labels.cpu().numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='macro')
    return running_loss / n, acc, f1





