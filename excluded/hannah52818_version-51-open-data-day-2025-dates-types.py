# =========================================================
# EfficientNet-B4 Improved Baseline (Inference-Boost Version)
# =========================================================
import os, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ---------------------
# CONFIG
# ---------------------
BASE_PATH   = "/kaggle/input/open-data-day-2025-dates-types-classification"
IMG_SIZE    = 380
BATCH_SIZE  = 16
EPOCHS      = 22
ACCUM_STEPS = 2
TTA         = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)

# ---------------------
# Seed
# ---------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all()

# ---------------------
# Load labels
# ---------------------
df = pd.read_csv(os.path.join(BASE_PATH, "train_labels.csv"))

label_list   = sorted(df["label"].unique())
label_to_idx = {l:i for i,l in enumerate(label_list)}
df["label_idx"] = df["label"].map(label_to_idx)
num_classes = len(label_list)

# ---------------------
# Dataset
# ---------------------
class DatesDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_label=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        if self.has_label:
            return img, row["label_idx"]
        else:
            return img, -1

# ------------------------------------------------------------
# NS-normalization (as you found this works best)
# ------------------------------------------------------------
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

# ---------------------
# Augmentations
# ---------------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85,1.0), ratio=(0.9,1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# *** strong random TTA ***
tta_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0), ratio=(0.95,1.05)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = DatesDataset(df, os.path.join(BASE_PATH,"train"), train_tfms)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=2, pin_memory=True)

# ---------------------
# Model
# ---------------------
model = timm.create_model(
    "tf_efficientnet_b4_ns",
    pretrained=True,
    num_classes=num_classes
).to(DEVICE)

# ---------------------
# Loss + Optimizer + Warmup
# ---------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, weight_decay=1e-4
)

warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=3)

cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS-3)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, [warmup, cosine], milestones=[3])

# ---------------------
# Training
# ---------------------
for epoch in range(EPOCHS):
    model.train()
    correct = 0; total = 0; running_loss = 0.0

    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(train_dl):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        out = model(imgs)
        loss = criterion(out, labels) / ACCUM_STEPS
        loss.backward()
        
        if (step+1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item()*imgs.size(0)*ACCUM_STEPS
        correct += (out.argmax(1)==labels).sum().item()
        total += labels.size(0)
    
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {running_loss/total:.4f} "
          f"Acc: {correct/total:.4f}")

# =========================================================
#       STEP-1  —  PURE RANDOM TTA SEARCH (INFERENCE)
# =========================================================
model.eval()
test_files = sorted(os.listdir(os.path.join(BASE_PATH,"test")))
preds = []

with torch.no_grad():
    for fname in test_files:
        img = Image.open(os.path.join(BASE_PATH,"test",fname)).convert("RGB")
        
        logits_sum = torch.zeros(1, num_classes).to(DEVICE)

        # *** ONLY TTA (remove base inference completely!) ***
        for _ in range(12):
            logits_sum += model(tta_tfms(img).unsqueeze(0).to(DEVICE))
        
        logits = logits_sum / 12
        pred_idx = logits.argmax(1).item()
        preds.append((fname, label_list[pred_idx]))

submission = pd.DataFrame(preds, columns=["filename","label"])
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


# ======================================
# Visualize training augmentations
# ======================================

import os, random
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms


# ---- define BASE_PATH ----
BASE_PATH = "/kaggle/input/open-data-day-2025-dates-types-classification"


# ---- normalization used in your model ----
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]


# ---- same training transforms as your model ----
IMG_SIZE = 380

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85,1.0), ratio=(0.9,1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ======================================
# pick a random image from dataset
# ======================================
sample_img = random.choice(os.listdir(f"{BASE_PATH}/train"))
img_path   = os.path.join(f"{BASE_PATH}/train", sample_img)
orig       = Image.open(img_path).convert("RGB")


# ======================================
# visualize 8 augmentations
# ======================================
plt.figure(figsize=(14,4))

for i in range(8):
    aug = train_tfms(orig)

    # convert tensor -> numpy + unnormalize for display
    aug = aug.permute(1,2,0).numpy()
    aug = (aug * 0.5) + 0.5   # because mean=std=0.5

    plt.subplot(2,4,i+1)
    plt.imshow(aug)
    plt.axis("off")

plt.suptitle(f"Augmentations of: {sample_img}")
plt.show()



# =========================================================
# Vision Transformer (ViT-B16 384) Improved Baseline
# =========================================================
import os, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ---------------------
# CONFIG
# ---------------------
BASE_PATH   = "/kaggle/input/open-data-day-2025-dates-types-classification"
IMG_SIZE    = 384
BATCH_SIZE  = 8
EPOCHS      = 30
ACCUM_STEPS = 2
TTA         = 12
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)


# ---------------------
# Seed
# ---------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all()


# ---------------------
# Load labels
# ---------------------
df = pd.read_csv(os.path.join(BASE_PATH, "train_labels.csv"))
label_list   = sorted(df["label"].unique())
label_to_idx = {l:i for i,l in enumerate(label_list)}
df["label_idx"] = df["label"].map(label_to_idx)
num_classes = len(label_list)


# ---------------------
# Dataset
# ---------------------
class DatesDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row["filename"])).convert("RGB")
        img = self.transform(img)
        return img, row["label_idx"]


# ---------------------
# Normalization
# ---------------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ---------------------
# Transforms
# ---------------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85,1.0), ratio=(0.9,1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

tta_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0), ratio=(0.95,1.05)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------------
# DataLoader
# ---------------------
train_ds = DatesDataset(df, os.path.join(BASE_PATH,"train"), train_tfms)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=2, pin_memory=True)


# ---------------------
# Vision Transformer
# ---------------------
model = timm.create_model(
    "vit_base_patch16_384",
    pretrained=True,
    num_classes=num_classes
).to(DEVICE)

model = model.to(memory_format=torch.channels_last)


# ---------------------
# Loss + Optimizer + Warmup
# ---------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4, weight_decay=1e-4
)

warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=3)

cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS-3)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, [warmup, cosine], milestones=[3])


# ---------------------
# Training
# ---------------------
for epoch in range(EPOCHS):
    model.train()
    correct = 0; total = 0; running_loss = 0.0

    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(train_dl):
        imgs = imgs.to(DEVICE, memory_format=torch.channels_last)
        labels = labels.to(DEVICE)

        out = model(imgs)
        loss = criterion(out, labels) / ACCUM_STEPS
        loss.backward()

        if (step+1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()*imgs.size(0)*ACCUM_STEPS
        correct += (out.argmax(1)==labels).sum().item()
        total += labels.size(0)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {running_loss/total:.4f} "
          f"Acc: {correct/total:.4f}")


# =========================================================
#       PURE RANDOM TTA INFERENCE
# =========================================================
model.eval()
test_files = sorted(os.listdir(os.path.join(BASE_PATH,"test")))
preds = []

with torch.no_grad():
    for fname in test_files:
        img = Image.open(os.path.join(BASE_PATH,"test",fname)).convert("RGB")
        
        logits_sum = torch.zeros(1, num_classes).to(DEVICE)

        for _ in range(TTA):
            logits_sum += model(tta_tfms(img).unsqueeze(0).to(DEVICE))
        
        logits = logits_sum / TTA
        pred_idx = logits.argmax(1).item()
        preds.append((fname, label_list[pred_idx]))

submission = pd.DataFrame(preds, columns=["filename","label"])
submission.to_csv("submission.csv", index=False)

print("Saved submission.csv (ViT)")


# =========================================================
# EfficientNet-B4 Improved Baseline (Inference-Boost Version)
# =========================================================
import os, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ---------------------
# CONFIG
# ---------------------
BASE_PATH   = "/kaggle/input/open-data-day-2025-dates-types-classification"
# IMG_SIZE    = 380 
# BATCH_SIZE  = 16
EPOCHS      = 22
ACCUM_STEPS = 2
TTA         = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 384
BATCH_SIZE = 8

print("DEVICE:", DEVICE)

# ---------------------
# Seed
# ---------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all()

# ---------------------
# Load labels
# ---------------------
df = pd.read_csv(os.path.join(BASE_PATH, "train_labels.csv"))

label_list   = sorted(df["label"].unique())
label_to_idx = {l:i for i,l in enumerate(label_list)}
df["label_idx"] = df["label"].map(label_to_idx)
num_classes = len(label_list)

# ---------------------
# Dataset
# ---------------------
class DatesDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_label=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        if self.has_label:
            return img, row["label_idx"]
        else:
            return img, -1

# ------------------------------------------------------------
# NS-normalization (as you found this works best)
# ------------------------------------------------------------
# MEAN = [0.5, 0.5, 0.5]
# STD  = [0.5, 0.5, 0.5]
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ---------------------
# Augmentations
# ---------------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85,1.0), ratio=(0.9,1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# *** strong random TTA ***
tta_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0), ratio=(0.95,1.05)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = DatesDataset(df, os.path.join(BASE_PATH,"train"), train_tfms)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=2, pin_memory=True)

# ---------------------
# Model
# ---------------------
# model = timm.create_model(
#     "tf_efficientnet_b4_ns",
#     pretrained=True,
#     num_classes=num_classes
# ).to(DEVICE)

model = timm.create_model(
    "convnext_base.fb_in22k_ft_in1k",
    pretrained=True,
    num_classes=num_classes
).to(DEVICE)



# ---------------------
# Loss + Optimizer + Warmup
# ---------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4, weight_decay=1e-4
)

warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=3)

cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS-3)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, [warmup, cosine], milestones=[3])

# ---------------------
# Training
# ---------------------
for epoch in range(EPOCHS):
    model.train()
    correct = 0; total = 0; running_loss = 0.0

    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(train_dl):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        out = model(imgs)
        loss = criterion(out, labels) / ACCUM_STEPS
        loss.backward()
        
        if (step+1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item()*imgs.size(0)*ACCUM_STEPS
        correct += (out.argmax(1)==labels).sum().item()
        total += labels.size(0)
    
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {running_loss/total:.4f} "
          f"Acc: {correct/total:.4f}")

# =========================================================
#       STEP-1  —  PURE RANDOM TTA SEARCH (INFERENCE)
# =========================================================
model.eval()
test_files = sorted(os.listdir(os.path.join(BASE_PATH,"test")))
preds = []

with torch.no_grad():
    for fname in test_files:
        img = Image.open(os.path.join(BASE_PATH,"test",fname)).convert("RGB")
        
        logits_sum = torch.zeros(1, num_classes).to(DEVICE)

        # *** ONLY TTA (remove base inference completely!) ***
        for _ in range(12):
            logits_sum += model(tta_tfms(img).unsqueeze(0).to(DEVICE))
        
        logits = logits_sum / 12
        pred_idx = logits.argmax(1).item()
        preds.append((fname, label_list[pred_idx]))

submission = pd.DataFrame(preds, columns=["filename","label"])
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


import os
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================
# SETTINGS
# ==========================================
BASE_PATH = "/kaggle/input/open-data-day-2025-dates-types-classification"
OUT_PATH  = "/kaggle/working/processed_hsv_rgb"

os.makedirs(f"{OUT_PATH}/train", exist_ok=True)
os.makedirs(f"{OUT_PATH}/test",  exist_ok=True)

# ==========================================
# FUNCTION
# ==========================================
def preprocess_and_save(in_folder, out_folder):
    files = os.listdir(in_folder)

    for fname in tqdm(files):
        img = cv2.imread(os.path.join(in_folder, fname))

        # --- convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # --- segmentation: keep only high saturation (texture)
        _, mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)

        # --- apply mask on EACH channel
        h_masked = cv2.bitwise_and(h, h, mask=mask)
        s_masked = cv2.bitwise_and(s, s, mask=mask)
        v_masked = cv2.bitwise_and(v, v, mask=mask)

        # --- merge back to HSV
        hsv_seg = cv2.merge([h_masked, s_masked, v_masked])

        # --- convert back to RGB (for EfficientNet)
        rgb_seg = cv2.cvtColor(hsv_seg, cv2.COLOR_HSV2BGR)

        # --- save segmented color version
        cv2.imwrite(os.path.join(out_folder, fname), rgb_seg)

# ==========================================
# RUN
# ==========================================
preprocess_and_save(f"{BASE_PATH}/train", f"{OUT_PATH}/train")
preprocess_and_save(f"{BASE_PATH}/test",  f"{OUT_PATH}/test")



# =========================================================
# EfficientNet-B4 Improved Baseline (Inference-Boost Version)
# =========================================================
import os, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ---------------------
# CONFIG
# ---------------------
BASE_PATH   = "/kaggle/input/open-data-day-2025-dates-types-classification"
IMG_SIZE    = 380
BATCH_SIZE  = 16
EPOCHS      = 22
ACCUM_STEPS = 2
TTA         = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)

# ---------------------
# Seed
# ---------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all()

# ---------------------
# Load labels
# ---------------------
df = pd.read_csv(os.path.join(BASE_PATH, "train_labels.csv"))

label_list   = sorted(df["label"].unique())
label_to_idx = {l:i for i,l in enumerate(label_list)}
df["label_idx"] = df["label"].map(label_to_idx)
num_classes = len(label_list)

# ---------------------
# Dataset
# ---------------------
class DatesDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_label=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        if self.has_label:
            return img, row["label_idx"]
        else:
            return img, -1

# ------------------------------------------------------------
# NS-normalization (as you found this works best)
# ------------------------------------------------------------
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

# ---------------------
# Augmentations
# ---------------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85,1.0), ratio=(0.9,1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# *** strong random TTA ***
tta_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0), ratio=(0.95,1.05)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = DatesDataset(df, "/kaggle/working/processed_hsv_rgb/train", train_tfms)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=2, pin_memory=True)

# ---------------------
# Model
# ---------------------
model = timm.create_model(
    "tf_efficientnet_b4_ns",
    pretrained=True,
    num_classes=num_classes
).to(DEVICE)

# ---------------------
# Loss + Optimizer + Warmup
# ---------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, weight_decay=1e-4
)

warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=3)

cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS-3)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, [warmup, cosine], milestones=[3])

# ---------------------
# Training
# ---------------------
for epoch in range(EPOCHS):
    model.train()
    correct = 0; total = 0; running_loss = 0.0

    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(train_dl):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        out = model(imgs)
        loss = criterion(out, labels) / ACCUM_STEPS
        loss.backward()
        
        if (step+1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item()*imgs.size(0)*ACCUM_STEPS
        correct += (out.argmax(1)==labels).sum().item()
        total += labels.size(0)
    
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {running_loss/total:.4f} "
          f"Acc: {correct/total:.4f}")

# =========================================================
#       STEP-1  —  PURE RANDOM TTA SEARCH (INFERENCE)
# =========================================================
model.eval()
test_files = sorted(os.listdir("/kaggle/working/processed_hsv_rgb/test"))

preds = []

with torch.no_grad():
    for fname in test_files:
        img = Image.open(os.path.join(BASE_PATH,"test",fname)).convert("RGB")
        
        logits_sum = torch.zeros(1, num_classes).to(DEVICE)

        # *** ONLY TTA (remove base inference completely!) ***
        for _ in range(12):
            logits_sum += model(tta_tfms(img).unsqueeze(0).to(DEVICE))
        
        logits = logits_sum / 12
        pred_idx = logits.argmax(1).item()
        preds.append((fname, label_list[pred_idx]))

submission = pd.DataFrame(preds, columns=["filename","label"])
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


# << run this BEFORE training >>

import os
import cv2
from tqdm import tqdm

BASE_PATH = "/kaggle/input/open-data-day-2025-dates-types-classification"
OUT_PATH  = "/kaggle/working/processed_gray_segmented"

os.makedirs(f"{OUT_PATH}/train", exist_ok=True)
os.makedirs(f"{OUT_PATH}/test",  exist_ok=True)

def preprocess_and_save(in_folder, out_folder):
    files = os.listdir(in_folder)

    for fname in tqdm(files):
        img = cv2.imread(os.path.join(in_folder, fname))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        gray_seg = cv2.bitwise_and(gray, gray, mask=mask)

        cv2.imwrite(os.path.join(out_folder, fname), gray_seg)

preprocess_and_save(f"{BASE_PATH}/train", f"{OUT_PATH}/train")
preprocess_and_save(f"{BASE_PATH}/test",  f"{OUT_PATH}/test")



# =========================================================
# EfficientNet-B4 Improved Baseline (Inference-Boost Version)
# =========================================================
import os, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ---------------------
# CONFIG
# ---------------------
BASE_PATH   = "/kaggle/input/open-data-day-2025-dates-types-classification"
IMG_SIZE    = 380
BATCH_SIZE  = 16
EPOCHS      = 22
ACCUM_STEPS = 2
TTA         = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)

# ---------------------
# Seed
# ---------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all()

# ---------------------
# Load labels
# ---------------------
df = pd.read_csv(os.path.join(BASE_PATH, "train_labels.csv"))

label_list   = sorted(df["label"].unique())
label_to_idx = {l:i for i,l in enumerate(label_list)}
df["label_idx"] = df["label"].map(label_to_idx)
num_classes = len(label_list)

# ---------------------
# Dataset
# ---------------------
class DatesDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_label=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("L")   # single channel
        img = self.transform(img)
        
        if self.has_label:
            return img, row["label_idx"]
        else:
            return img, -1

# ------------------------------------------------------------
# NS-normalization (as you found this works best)
# ------------------------------------------------------------
MEAN=[0.5]
STD=[0.5]


# ---------------------
# Augmentations
# ---------------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85,1.0), ratio=(0.9,1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# *** strong random TTA ***
tta_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0), ratio=(0.95,1.05)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = DatesDataset(df, "/kaggle/working/processed_gray_segmented/train", train_tfms)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=2, pin_memory=True)

# ---------------------
# Model
# ---------------------
model = timm.create_model(
    "tf_efficientnet_b4_ns",
    pretrained=True,
    num_classes=num_classes,
    in_chans=1     # <--- important
).to(DEVICE)


# ---------------------
# Loss + Optimizer + Warmup
# ---------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, weight_decay=1e-4
)

warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=3)

cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS-3)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, [warmup, cosine], milestones=[3])

# ---------------------
# Training
# ---------------------
for epoch in range(EPOCHS):
    model.train()
    correct = 0; total = 0; running_loss = 0.0

    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(train_dl):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        out = model(imgs)
        loss = criterion(out, labels) / ACCUM_STEPS
        loss.backward()
        
        if (step+1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item()*imgs.size(0)*ACCUM_STEPS
        correct += (out.argmax(1)==labels).sum().item()
        total += labels.size(0)
    
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {running_loss/total:.4f} "
          f"Acc: {correct/total:.4f}")

# =========================================================
#       STEP-1  —  PURE RANDOM TTA SEARCH (INFERENCE)
# =========================================================
model.eval()
test_files = sorted(os.listdir("/kaggle/working/processed_gray_segmented/test"))

preds = []

with torch.no_grad():
    for fname in test_files:
        img = Image.open(os.path.join("/kaggle/working/processed_gray_segmented/test", fname)
).convert("L")
        
        logits_sum = torch.zeros(1, num_classes).to(DEVICE)

        # *** ONLY TTA (remove base inference completely!) ***
        for _ in range(12):
            logits_sum += model(tta_tfms(img).unsqueeze(0).to(DEVICE))
        
        logits = logits_sum / 12
        pred_idx = logits.argmax(1).item()
        preds.append((fname, label_list[pred_idx]))

submission = pd.DataFrame(preds, columns=["filename","label"])
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# where original images are
ORIG = "/kaggle/input/open-data-day-2025-dates-types-classification/test"

# where segmented grayscale images are saved
SEG  = "/kaggle/working/processed_gray_segmented/test"

# pick a few random images
samples = random.sample(os.listdir(ORIG), 5)

plt.figure(figsize=(12, 6))

for i, fname in enumerate(samples):
    
    orig = Image.open(os.path.join(ORIG, fname)).convert("RGB")
    grayseg = Image.open(os.path.join(SEG, fname))  # already grayscale

    # before
    plt.subplot(2, len(samples), i+1)
    plt.imshow(orig)
    plt.axis("off")
    plt.title("Original")

    # after segmentation
    plt.subplot(2, len(samples), i+1+len(samples))
    plt.imshow(grayseg, cmap="gray")
    plt.axis("off")
    plt.title("Grey")

plt.show()



import os
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================
# SETTINGS
# ==========================================
BASE_PATH = "/kaggle/input/open-data-day-2025-dates-types-classification"
OUT_PATH  = "/kaggle/working/processed_hsv_rgb"

os.makedirs(f"{OUT_PATH}/train", exist_ok=True)
os.makedirs(f"{OUT_PATH}/test",  exist_ok=True)

# ==========================================
# FUNCTION
# ==========================================
def preprocess_and_save(in_folder, out_folder):
    files = os.listdir(in_folder)

    for fname in tqdm(files):
        img = cv2.imread(os.path.join(in_folder, fname))

        # --- convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # --- segmentation: keep only high saturation (texture)
        _, mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)

        # --- apply mask on EACH channel
        h_masked = cv2.bitwise_and(h, h, mask=mask)
        s_masked = cv2.bitwise_and(s, s, mask=mask)
        v_masked = cv2.bitwise_and(v, v, mask=mask)

        # --- merge back to HSV
        hsv_seg = cv2.merge([h_masked, s_masked, v_masked])

        # --- convert back to RGB (for EfficientNet)
        rgb_seg = cv2.cvtColor(hsv_seg, cv2.COLOR_HSV2BGR)

        # --- save segmented color version
        cv2.imwrite(os.path.join(out_folder, fname), rgb_seg)

# ==========================================
# RUN
# ==========================================
preprocess_and_save(f"{BASE_PATH}/train", f"{OUT_PATH}/train")
preprocess_and_save(f"{BASE_PATH}/test",  f"{OUT_PATH}/test")



import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# where original images are
ORIG = "/kaggle/input/open-data-day-2025-dates-types-classification/test"

# where segmented RGB images are saved
SEG  = "/kaggle/working/processed_hsv_rgb/test"

# pick a few random images
samples = random.sample(os.listdir(ORIG), 5)

plt.figure(figsize=(12, 6))

for i, fname in enumerate(samples):
    
    orig = Image.open(os.path.join(ORIG, fname)).convert("RGB")
    hsvseg = Image.open(os.path.join(SEG, fname))   # this is already RGB

    # before
    plt.subplot(2, len(samples), i+1)
    plt.imshow(orig)
    plt.axis("off")
    plt.title("Original")

    # after hsv segmentation
    plt.subplot(2, len(samples), i+1+len(samples))
    plt.imshow(hsvseg)    # no cmap
    plt.axis("off")
    plt.title("HSV Segmented")

plt.show()


