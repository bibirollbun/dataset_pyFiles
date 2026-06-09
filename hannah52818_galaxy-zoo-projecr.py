import os, zipfile

base = "/kaggle/input/galaxy-zoo-the-galaxy-challenge"
work = "/kaggle/working"

# Unzip training images
with zipfile.ZipFile(os.path.join(base, "images_training_rev1.zip"), "r") as z:
    z.extractall(work)

# Unzip test images
with zipfile.ZipFile(os.path.join(base, "images_test_rev1.zip"), "r") as z:
    z.extractall(work)

# Unzip labels
with zipfile.ZipFile(os.path.join(base, "training_solutions_rev1.zip"), "r") as z:
    z.extractall(work)

# Unzip zero benchmark (submission template)
with zipfile.ZipFile(os.path.join(base, "all_zeros_benchmark.zip"), "r") as z:
    z.extractall(work)

print(os.listdir(work))


# ================= TOP 10%-STYLE RESNET50 =================
# Trains a strong ResNet50 model with:
# - ImageNet normalization
# - Heavy augmentation + Random Erasing
# - 60 epochs, cosine LR
# - Mixed precision (AMP)
# - Best checkpoint on val loss
# - 9-view TTA
# Outputs: submission_resnet50_T10.csv

import os, cv2, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

# ---------------- CONFIG ----------------
IMG_SIZE   = 256
BATCH_SIZE = 24
EPOCHS     = 60
LR         = 2e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "/kaggle/working/images_training_rev1"
TEST_DIR  = "/kaggle/working/images_test_rev1"
LABELS    = "/kaggle/working/training_solutions_rev1.csv"
BENCH     = "/kaggle/working/all_zeros_benchmark.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------- SEEDING ----------------
def seed_everything(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(2025)

# ---------------- DATASET & TRANSFORMS ----------------
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.5)
])

val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class GalaxyDataset(Dataset):
    def __init__(self, galaxy_ids, img_dir, labels=None, transform=None):
        self.galaxy_ids = galaxy_ids
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.galaxy_ids)

    def __getitem__(self, idx):
        gid = self.galaxy_ids[idx]
        img_path = os.path.join(self.img_dir, f"{gid}.jpg")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label

        return img

# ---------------- LOAD DATA ----------------
df = pd.read_csv(LABELS)
all_ids = df["GalaxyID"].values
targets = df.drop("GalaxyID", axis=1).values

X_train_ids, X_val_ids, y_train, y_val = train_test_split(
    all_ids, targets, test_size=0.1, random_state=7
)

train_ds = GalaxyDataset(X_train_ids, TRAIN_DIR, y_train, transform=train_tf)
val_ds   = GalaxyDataset(X_val_ids,   TRAIN_DIR, y_val,   transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# ---------------- MODEL ----------------
class StrongResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 37)
        )

    def forward(self, x):
        return self.backbone(x)

model = StrongResNet50().to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()
scaler = GradScaler()

best_val_loss = float("inf")
BEST_CKPT = "/kaggle/working/resnet50_T10_best.pth"

# ---------------- TRAINING LOOP ----------------
for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"[ResNet50-T10] Epoch {epoch+1}/{EPOCHS} - train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item()

    scheduler.step()
    train_loss = train_loss_sum / len(train_loader)

    # validation
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"[ResNet50-T10] Epoch {epoch+1}/{EPOCHS} - val"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            val_loss_sum += loss.item()

    val_loss = val_loss_sum / len(val_loader)
    print(f"[ResNet50-T10] Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_CKPT)
        print(f"  ğŸ’¾ New best model saved with val_loss={best_val_loss:.5f}")

print(f"\nLoading best checkpoint from {BEST_CKPT}")
model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE))

# ---------------- 9-VIEW TTA PREDICTION ----------------
def tta_9view(model, imgs):
    # imgs: (B, C, H, W)
    aug_imgs = [
        imgs,
        torch.flip(imgs, [3]),                        # horiz flip
        torch.flip(imgs, [2]),                        # vert flip
        torch.rot90(imgs, 1, [2, 3]),
        torch.rot90(imgs, 2, [2, 3]),
        torch.rot90(imgs, 3, [2, 3]),
        torch.flip(torch.rot90(imgs, 1, [2, 3]), [3]),
        torch.flip(torch.rot90(imgs, 2, [2, 3]), [3]),
        torch.flip(torch.rot90(imgs, 3, [2, 3]), [3]),
    ]

    outs = []
    for a in aug_imgs:
        with autocast():
            outs.append(model(a))
    return torch.stack(outs).mean(0)  # (B, 37)

sub_template = pd.read_csv(BENCH)
test_ids = sub_template["GalaxyID"].values

test_ds = GalaxyDataset(test_ids, TEST_DIR, labels=None, transform=val_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

model.eval()
all_preds = []

with torch.no_grad():
    for imgs in tqdm(test_loader, desc="[ResNet50-T10] TTA predicting"):
        imgs = imgs.to(DEVICE)
        preds = tta_9view(model, imgs)
        all_preds.append(preds.cpu().numpy())

preds = np.vstack(all_preds)

# ---------------- GROUP NORMALIZATION ----------------
preds = np.clip(preds, 0.0, 1.0)
groups = [
    (0,3),(3,5),(5,8),(8,11),(11,15),
    (15,18),(18,25),(25,28),(28,31),(31,37)
]

for s, e in groups:
    g = preds[:, s:e].sum(axis=1, keepdims=True)
    g[g == 0] = 1.0
    preds[:, s:e] /= g

# -------------------- SAVE OUTPUT CSV --------------------

sub = sub_template.copy()
sub.iloc[:, 1:] = preds

# 1) Save to /kaggle/working (will go into _output.zip)
sub_path_output = "/kaggle/working/submission_resnet50_strong.csv"
sub.to_csv(sub_path_output, index=False)

# 2) Save to notebook root (download separately)
sub_path_root = "submission_resnet50_strong.csv"
sub.to_csv(sub_path_root, index=False)

print("ğŸ“� CSV saved to:")
print(f" â†’ {sub_path_output} (included in output.zip)")
print(f" â†’ {sub_path_root} (manual download, not zipped)")


# # ============================================================
# # TEAMMATE 1: ResNet50 + TTA â†’ preds_resnet50.csv
# # ============================================================

# import os, cv2, random, gc
# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# from sklearn.model_selection import train_test_split
# from torch.cuda.amp import autocast, GradScaler

# IMG_SIZE = 256
# BATCH_SIZE = 24
# EPOCHS = 10
# LR = 3e-4
# SEED = 2025
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TRAIN_DIR = "/kaggle/working/images_training_rev1"
# TEST_DIR  = "/kaggle/working/images_test_rev1"
# LABELS    = "/kaggle/working/training_solutions_rev1.csv"

# CHECKPOINT_PATH = "/kaggle/working/resnet50_seed2025.pth"
# PREDS_PATH      = "/kaggle/working/preds_resnet50.csv"

# def seed_everything(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# seed_everything(SEED)

# class GalaxyDataset(Dataset):
#     def __init__(self, galaxy_ids, img_dir, labels=None, augment=False):
#         self.galaxy_ids = galaxy_ids
#         self.img_dir = img_dir
#         self.labels = labels
#         self.augment = augment

#         self.aug = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.RandomRotation(360),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.ColorJitter(0.1,0.1,0.1,0.1),
#             transforms.Resize((IMG_SIZE, IMG_SIZE)),
#             transforms.ToTensor()
#         ])

#         self.basic = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((IMG_SIZE, IMG_SIZE)),
#             transforms.ToTensor()
#         ])

#     def __len__(self):
#         return len(self.galaxy_ids)

#     def __getitem__(self, idx):
#         gid = self.galaxy_ids[idx]
#         img_path = os.path.join(self.img_dir, f"{gid}.jpg")
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         if self.augment:
#             img = self.aug(img)
#         else:
#             img = self.basic(img)

#         if self.labels is not None:
#             label = torch.tensor(self.labels[idx], dtype=torch.float32)
#             return img, label
#         return img

# df = pd.read_csv(LABELS)
# all_ids = df["GalaxyID"].values
# targets = df.drop("GalaxyID", axis=1).values

# X_train_ids, X_val_ids, y_train, y_val = train_test_split(
#     all_ids, targets, test_size=0.1, random_state=7
# )

# train_ds = GalaxyDataset(X_train_ids, TRAIN_DIR, y_train, augment=True)
# val_ds   = GalaxyDataset(X_val_ids,   TRAIN_DIR, y_val,   augment=False)

# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
# val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# class GalaxyResNet50(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = models.resnet50(pretrained=True)
#         in_features = self.backbone.fc.in_features
#         self.backbone.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(in_features, 37)
#         )

#     def forward(self, x):
#         return self.backbone(x)

# model = GalaxyResNet50().to(DEVICE)
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# criterion = nn.MSELoss()
# scaler = GradScaler()

# for epoch in range(EPOCHS):
#     model.train()
#     train_loss = 0.0

#     for imgs, labels in tqdm(train_loader, desc=f"[ResNet50] Epoch {epoch+1}/{EPOCHS} - train"):
#         imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

#         optimizer.zero_grad()
#         with autocast():
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         train_loss += loss.item()

#     scheduler.step()

#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for imgs, labels in tqdm(val_loader, desc=f"[ResNet50] Epoch {epoch+1}/{EPOCHS} - val"):
#             imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
#             with autocast():
#                 outputs = model(imgs)
#                 loss = criterion(outputs, labels)
#             val_loss += loss.item()

#     print(f"[ResNet50] Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# torch.save(model.state_dict(), CHECKPOINT_PATH)
# print(f"[ResNet50] Saved checkpoint to {CHECKPOINT_PATH}")

# def tta_predict(model, loader):
#     model.eval()
#     preds = []
#     with torch.no_grad():
#         for imgs in tqdm(loader, desc="[ResNet50] TTA predicting"):
#             imgs = imgs.to(DEVICE)
#             batch_preds = []

#             with autocast():
#                 out = model(imgs)
#             batch_preds.append(out)

#             imgs_h = torch.flip(imgs, dims=[3])
#             with autocast():
#                 out_h = model(imgs_h)
#             batch_preds.append(out_h)

#             imgs_v = torch.flip(imgs, dims=[2])
#             with autocast():
#                 out_v = model(imgs_v)
#             batch_preds.append(out_v)

#             batch_mean = torch.stack(batch_preds, dim=0).mean(dim=0)
#             preds.append(batch_mean.cpu().numpy())

#     return np.vstack(preds)

# sub_template = pd.read_csv("/kaggle/working/all_zeros_benchmark.csv")
# test_ids = sub_template["GalaxyID"].values

# test_ds = GalaxyDataset(test_ids, TEST_DIR, labels=None, augment=False)
# test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# preds = tta_predict(model, test_loader)

# preds = np.clip(preds, 0, 1)
# groups = [
#     (0,3),(3,5),(5,8),(8,11),(11,15),
#     (15,18),(18,25),(25,28),(28,31),(31,37)
# ]
# for start, end in groups:
#     s = preds[:, start:end].sum(axis=1, keepdims=True)
#     s[s == 0] = 1.0
#     preds[:, start:end] /= s

# out = sub_template.copy()
# out.iloc[:,1:] = preds
# out.to_csv(PREDS_PATH, index=False)
# out.to_csv("submission_single_resnet50.csv", index=False)

# print(f"[ResNet50] Saved preds to {PREDS_PATH}")
# print("[ResNet50] Also wrote submission_single_resnet50.csv")

