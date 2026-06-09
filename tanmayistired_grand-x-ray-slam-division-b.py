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


import numpy as np
import pandas as pd
train_df = pd.read_csv('/kaggle/input/grand-xray-slam-division-b/train2.csv')
train_df.head(10)


train_df.info()


# Summarize key metrics
total_images = len(train_df)
total_patients = train_df['Patient_ID'].nunique()
total_studies = train_df['Study'].nunique()
print(f"Total Images: {total_images}")
print(f"Total Patients: {total_patients}")
print(f"Total Studies: {total_studies}")


train_df.isnull().sum()


train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Sex'] = train_df['Sex'].fillna('Unknown')


train_df.isnull().sum()


# Define the 14 condition columns
label_columns = ['No Finding', 'Lung Opacity', 'Support Devices', 'Atelectasis',
                 'Cardiomegaly', 'Pleural Effusion', 'Enlarged Cardiomediastinum',
                 'Edema', 'Consolidation', 'Pneumonia', 'Fracture', 'Lung Lesion',
                 'Pneumothorax', 'Pleural Other']

# Calculate counts and percentages for each condition
label_counts = train_df[label_columns].sum()
label_percentages = (label_counts / total_images * 100).round(2)
prevalence_df = pd.DataFrame({
    'Condition': label_counts.index,
    'Count': label_counts.values,
    'Percent (%)': label_percentages.values
})

# Display prevalence table
print("Label Prevalence:")
print(prevalence_df)


# Check for duplicate Image_Names
duplicate_images = train_df['Image_name'].duplicated().sum()
print(f"Duplicated Image_Name entries: {duplicate_images}")

# Check for duplicate Patient_IDs (expected due to multiple images per patient)
duplicate_patients = total_images - total_patients
print(f"Duplicated Patient_ID entries: {duplicate_patients}")

# Check for invalid Age values
invalid_ages = train_df['Age'].dropna()
invalid_ages = invalid_ages[invalid_ages < 0].count()
print(f"Invalid Age values (<0): {invalid_ages}")


import os, random
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit

# Important: split by Patient_ID so same patient never leaks into train+val
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(train_df, groups=train_df["Patient_ID"]))

train_df_split = train_df.iloc[train_idx].reset_index(drop=True)
val_df_split   = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Train size: {len(train_df_split)} | Val size: {len(val_df_split)}")



from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import torch
import os

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, label_columns=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_columns = label_columns

        # Pre-store labels if provided
        if self.label_columns is not None:
            self.targets = self.df[self.label_columns].values.astype("float32")
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "Image_name"]
        img_path = os.path.join(self.img_dir, img_name)

        # ---- Load image safely ----
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"âš ï¸� Corrupted image skipped: {img_path} ({e})")
            # Return next sample instead of crashing
            return self.__getitem__((idx + 1) % len(self.df))

        if self.transform:
            image = self.transform(image)

        # ---- Labels ----
        if self.targets is not None:
            labels = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image, labels
        else:
            return image, img_name



from torch.utils.data import DataLoader
from torchvision import transforms

# ---- Transforms ----
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.08, contrast=0.08),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# ---- Datasets ----
train_ds = ChestXrayDataset(train_df_split,
                            "/kaggle/input/grand-xray-slam-division-b/train2",
                            transform=train_transform,
                            label_columns=label_columns)

val_ds   = ChestXrayDataset(val_df_split,
                            "/kaggle/input/grand-xray-slam-division-b/train2",
                            transform=val_transform,
                            label_columns=label_columns)

# ---- Dataloaders ----
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# ---- Class imbalance for BCEWithLogitsLoss ----
pos_counts = train_df_split[label_columns].sum().values
neg_counts = len(train_df_split) - pos_counts
pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)

# ---- Device ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("CUDA device:", torch.cuda.get_device_name(0))

# Move pos_weight to same device
pos_weight = pos_weight.to(DEVICE)



import os
import time
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Hyperparameters ----
HEAD_EPOCHS   = 5          # train classifier head first
FT_EPOCHS     = 8         # then fine-tune backbone
BATCH_SIZE    = 32         # don't leave None, set explicitly
LR_HEAD       = 1e-3       # higher LR for head
LR_FT         = 1e-4       # lower LR for fine-tuning
WEIGHT_DECAY  = 1e-4       # regularization
PATIENCE      = 3          # early stopping patience
MIN_LR        = 1e-7       # minimum learning rate for scheduler

def build_model(num_classes=14, dropout=0.3):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    
    # Better head: hidden layer + ReLU + Dropout
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, in_features // 2),
        nn.ReLU(),
        nn.Dropout(dropout/2),
        nn.Linear(in_features // 2, num_classes)
    )
    return model
# Build model
model = build_model(num_classes=14, dropout=0.3).to(DEVICE)
# Loss with pos_weight for multi-label
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))  # pos_weight must match labels shape### LOSS

# Stage 1: freeze backbone except final fc
for name, param in model.named_parameters():
    param.requires_grad = False
for name, param in model.fc.named_parameters():
    param.requires_grad = True

opt_head = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    opt_head, mode='min', factor=0.5, patience=PATIENCE, min_lr=MIN_LR
)

def run_epoch(model, loader, optimizer=None, train=False, device=DEVICE, use_amp=False):
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    all_labels = []
    all_probs = []
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    loop = tqdm(loader, desc='Train' if train else 'EVAL', leave= False)
    for imgs, labels in loop:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.set_grad_enabled(train):
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)

            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())

            if train:
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        losses.append(loss.item())
        loop.set_postfix(loss=np.mean(losses))
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    return np.mean(losses), auc

### STAGE - 1 HEAD TRAINING
for epoch in range(1,HEAD_EPOCHS+1):
    t0 = time.time()
    train_loss, train_auc = run_epoch(model, train_loader, optimizer=opt_head, train=True, use_amp=False)
    val_loss, val_auc = run_epoch(model, val_loader, train=False, use_amp=False)
    scheduler.step(val_loss)
    print(f"Epoch {epoch}/{HEAD_EPOCHS}  train_loss={train_loss:.4f} train_auc={train_auc:.4f}  val_loss={val_loss:.4f} val_auc={val_auc:.4f}  time={(time.time()-t0):.1f}s")

## UNFREEZE BACKBONE AND FINE TUNEEEEEE

for param in model.parameters():
    param.requires_grad = True

opt_ft = optim.AdamW(model.parameters(), lr=LR_FT, weight_decay=WEIGHT_DECAY)
scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(
    opt_ft, mode='min', factor=0.5, patience=PATIENCE, min_lr=MIN_LR
)

for epoch in range(1,FT_EPOCHS+1):
    t0 = time.time()
    train_loss, train_auc = run_epoch(model, train_loader, optimizer=opt_ft, train=True, use_amp=False)
    val_loss, val_auc = run_epoch(model, val_loader, train=False, use_amp=False)
    scheduler_ft.step(val_loss)
    print(f"FT Epoch {epoch}/{FT_EPOCHS}  train_loss={train_loss:.4f} train_auc={train_auc:.4f}  val_loss={val_loss:.4f} val_auc={val_auc:.4f}  time={(time.time()-t0):.1f}s")


import os, json, torch
from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi, HfFolder, upload_folder

# ---- Load HF Token from Kaggle Secrets ----
user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")

HF_REPO_ID = "TanmayTomar/chest-xray-resnet50"

# Save token so huggingface_hub can pick it up
HfFolder.save_token(HF_TOKEN)

# Init API with token
api = HfApi(token=HF_TOKEN)

# Create repo if not exists
try:
    api.create_repo(repo_id=HF_REPO_ID, private=False)
except Exception as e:
    print("Repo already exists:", e)

# ---- After fine-tuning ----
SAVE_DIR = "./resnet50_chest_xray"
os.makedirs(SAVE_DIR, exist_ok=True)

# save model weights
torch.save(model.state_dict(), f"{SAVE_DIR}/pytorch_model.bin")

# save config
config = {"architecture": "resnet50", "num_classes": 14, "dropout": 0.3}
with open(f"{SAVE_DIR}/config.json", "w") as f:
    json.dump(config, f)

# push to HF
upload_folder(
    repo_id=HF_REPO_ID,
    folder_path=SAVE_DIR,
    commit_message="Upload fine-tuned ResNet50 after FT training",
    token=HF_TOKEN,
)

print(f"Model pushed to https://huggingface.co/{HF_REPO_ID}")



import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from torchvision import models
import json
from tqdm import tqdm

# ---- Load config from HF ----
config_path = hf_hub_download(
    repo_id="TanmayTomar/chest-xray-resnet50",
    filename="config.json"
)
with open(config_path, "r") as f:
    config = json.load(f)

NUM_CLASSES = config["num_classes"]
DROPOUT = config["dropout"]

# ---- Load model weights ----
weights_path = hf_hub_download(
    repo_id="TanmayTomar/chest-xray-resnet50",
    filename="pytorch_model.bin"
)

# Build the model EXACTLY like in training
model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(in_features, in_features // 2),
    nn.ReLU(),
    nn.Dropout(DROPOUT / 2),
    nn.Linear(in_features // 2, NUM_CLASSES)
)

# Load weights
state_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# ---- Load sample submission ----
sub_df = pd.read_csv("/kaggle/input/grand-xray-slam-division-b/sample_submission_2.csv")

# Ensure label columns (skip first col if it's ID)
label_columns = sub_df.columns.tolist()[1:]

# ---- Test dataset ----
test_transform = val_transform  # reuse validation transform

test_ds = ChestXrayDataset(
    dataframe=sub_df,
    img_dir="/kaggle/input/grand-xray-slam-division-b/test2",
    transform=test_transform,
    label_columns=None
)

test_loader = DataLoader(
    test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# ---- Inference ----
all_preds = []
with torch.no_grad():
    for imgs, img_names in tqdm(test_loader, desc="Predicting"):
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)

all_preds = np.vstack(all_preds)

# ---- Fill & Save submission ----
sub_df[label_columns] = all_preds
output_path = "submission.csv"
sub_df.to_csv(output_path, index=False)

print(f"Submission file saved at {output_path}")
display(sub_df.head())





