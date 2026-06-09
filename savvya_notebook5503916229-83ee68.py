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


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import os

root = "/kaggle/input"
for path, dirs, files in os.walk(root):
    print(path)



import os

for dirname, _, filenames in os.walk('/kaggle/input/UBC-OCEAN'):
    print(dirname)
    for filename in filenames:
        print("  ", filename)



!ls /kaggle/input


!ls /kaggle/input/UBC-OCEAN


import pandas as pd

base = "/kaggle/input/UBC-OCEAN"

train_df = pd.read_csv(f"{base}/train.csv")
test_df  = pd.read_csv(f"{base}/test.csv")
sample_df = pd.read_csv(f"{base}/sample_submission.csv")

train_df.head(), test_df.head()


import pandas as pd

# Correct paths
train_df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
test_df  = pd.read_csv('/kaggle/input/UBC-OCEAN/test.csv')

train_df.head()



import os

base = "/kaggle/input/UBC-OCEAN/train_images"

for root, dirs, files in os.walk(base):
    print(root, "->", len(files), "files")
    break



!ls /kaggle/input/UBC-OCEAN/train_images | head



image_id = train_df.iloc[0]["image_id"]
path = "/kaggle/input/UBC-OCEAN/train_images"

print("Searching for:", image_id)

matches = []
for root, dirs, files in os.walk(path):
    for f in files:
        if f.startswith(str(image_id)):
            matches.append(os.path.join(root, f))

matches[:10], len(matches)



import pandas as pd

# Correct paths for the CSVs
train_df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
test_df = pd.read_csv('/kaggle/input/UBC-OCEAN/test.csv')

train_df.head()



import cv2
import os

# Get an example image_id from the train CSV
image_id = train_df.iloc[0]["image_id"]  # e.g., 4, 4608, etc.

# Build correct path
image_path = f"/kaggle/input/UBC-OCEAN/train_images/{image_id}.png"

# Load the image
img = cv2.imread(image_path)

print("Loaded:", image_path)
print("Shape:", img.shape)



# 1.1 Explore directory structure
import os
from pprint import pprint

base = "/kaggle/input/UBC-OCEAN"

print("Base exists:", os.path.exists(base))
print("\nTop-level files/folders under /kaggle/input:")
pprint(sorted(os.listdir("/kaggle/input")))

print("\nContents of UBC-OCEAN:")
for root, dirs, files in os.walk(base):
    # print first few directories and files only for readability
    rel = os.path.relpath(root, base)
    if rel == ".":
        rel = "/"
    print(f"\nFolder: {rel}")
    if dirs:
        print("  subdirs:", dirs[:10])
    if files:
        print("  files (first 20):", files[:20])
    # stop descending after 3 levels for brevity
    if root.count(os.sep) - base.count(os.sep) >= 2:
        continue



# 1.2 Load train/test CSVs and show quick EDA
import pandas as pd

base = "/kaggle/input/UBC-OCEAN"
train_csv = os.path.join(base, "train.csv")
test_csv  = os.path.join(base, "test.csv")
sample_csv = os.path.join(base, "sample_submission.csv")

print("train.csv exists:", os.path.exists(train_csv))
print("test.csv exists: ", os.path.exists(test_csv))
print("sample_submission exists:", os.path.exists(sample_csv))

train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)
sample_df = pd.read_csv(sample_csv)

print("\ntrain_df.shape:", train_df.shape)
print("test_df.shape:", test_df.shape)
print("\ntrain_df.columns:", train_df.columns.tolist())
print("\nFirst 5 rows of train_df:")
display(train_df.head())

# If there is a class/label column, show distribution. Try common names.
possible_label_cols = [c for c in train_df.columns if c.lower() in ("label","target","class","subtype")]
print("\nDetected possible label columns:", possible_label_cols)

if possible_label_cols:
    lab = possible_label_cols[0]
    print(f"\nLabel distribution for '{lab}':")
    display(train_df[lab].value_counts().sort_index())
else:
    # fallback: try to infer multiclass columns
    print("\nNo obvious label column detected. Show first few columns and sample values to inspect.")
    display(train_df.iloc[:, :6].head())



try:
    import timm
    print("timm version:", timm.__version__)
except ImportError as e:
    print("timm is NOT available:", e)



import os
import numpy as np
import pandas as pd

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


import torchvision.models as models
from torchvision.models import inception_v3, swin_t



BASE_PATH = "/kaggle/input/UBC-OCEAN"
IMAGE_DIR_TRAIN = f"{BASE_PATH}/train_images"
IMAGE_DIR_TEST  = f"{BASE_PATH}/test_images"

train_df = pd.read_csv(f"{BASE_PATH}/train.csv")
test_df  = pd.read_csv(f"{BASE_PATH}/test.csv")

print("train_df shape:", train_df.shape)
print("test_df shape:", test_df.shape)
print("Columns in train_df:", train_df.columns.tolist())

train_df.head()



label_col = "label"   # e.g. "label", "subtype", etc.

assert label_col in train_df.columns, f"{label_col} not found in train_df columns: {train_df.columns.tolist()}"

le = LabelEncoder()
train_df["label_enc"] = le.fit_transform(train_df[label_col])

num_classes = train_df["label_enc"].nunique()
print("Number of classes:", num_classes)
train_df[[label_col, "label_enc"]].head()


train_df_split, val_df_split = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["label_enc"],
    random_state=42
)

print("Train size:", len(train_df_split))
print("Val size:", len(val_df_split))



IMG_SIZE = 224

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])




class OCEANDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]   # make sure this column exists in train.csv

        img_path = os.path.join(self.image_dir, f"{image_id}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label_enc"])
        return image, label


class OCEANTestDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        img_path = os.path.join(self.image_dir, f"{image_id}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Test image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id



BATCH_SIZE = 4  # smaller batch for debugging

train_dataset = OCEANDataset(train_df_split, IMAGE_DIR_TRAIN, transform=train_transform)
val_dataset   = OCEANDataset(val_df_split,   IMAGE_DIR_TRAIN, transform=val_transform)
test_dataset  = OCEANTestDataset(test_df,    IMAGE_DIR_TEST,  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("DataLoaders created.")



import torch
import torch.nn as nn

class InceptionSwinHybrid(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ---- InceptionV3 branch (base CNN) ----
        # No pretrained weights (no internet needed)
        self.inception = inception_v3(weights=None, aux_logits=False)
        incep_in_features = self.inception.fc.in_features  # last FC input dim

        # Remove original classification head
        self.inception.fc = nn.Identity()
        # Project Inception features to 512-dim embedding
        self.incep_proj = nn.Linear(incep_in_features, 512)

        # ---- Swin Transformer branch ----
        self.swin = swin_t(weights=None)
        swin_in_features = self.swin.head.in_features

        # Remove Swin classifier head
        self.swin.head = nn.Identity()
        # Project Swin features to 512-dim embedding
        self.swin_proj = nn.Linear(swin_in_features, 512)

        # ---- Fusion + final classifier ----
        # Concatenate [Inception(512), Swin(512)] -> 1024
        self.classifier = nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        # Inception branch
        incep_feat = self.inception(x)            # [B, incep_in_features]
        incep_feat = self.incep_proj(incep_feat)  # [B, 512]

        # Swin branch
        swin_feat = self.swin(x)                  # [B, swin_in_features]
        swin_feat = self.swin_proj(swin_feat)     # [B, 512]

        # Fuse
        fused = torch.cat([incep_feat, swin_feat], dim=1)  # [B, 1024]

        # Class logits
        out = self.classifier(fused)              # [B, num_classes]
        return out



images, labels = next(iter(train_loader))
print("Batch images shape:", images.shape)
print("Batch labels shape:", labels.shape)



from sklearn.preprocessing import LabelEncoder

label_col = "label"   # change if your column has a different name

le = LabelEncoder()
train_df["label_enc"] = le.fit_transform(train_df[label_col])

num_classes = train_df["label_enc"].nunique()
print("Number of classes:", num_classes)



# Create the InceptionV3 + Swin hybrid model instead of plain Swin
model = InceptionSwinHybrid(num_classes=num_classes)
model.to(device)

# Quick sanity check
x = torch.randn(1, 3, 224, 224).to(device)  # IMG_SIZE must be 224
with torch.no_grad():
    out = model(x)

print("Hybrid model output shape:", out.shape)  # should be [1, num_classes]





criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

EPOCHS = 5  
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)



def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total



best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc     = eval_one_epoch(model, val_loader, criterion, device)

    scheduler.step()

    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_inception_swin_hybrid.pth")
        print("Saved new best model (val_acc = {:.4f})".format(val_acc))


