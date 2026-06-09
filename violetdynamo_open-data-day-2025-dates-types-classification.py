# =========================================================
# EfficientNet-B4 5-Fold Baseline (memory-safe)
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
from sklearn.model_selection import StratifiedKFold

# ---------------------
# CONFIG
# ---------------------
BASE_PATH   = "/kaggle/input/open-data-day-2025-dates-types-classification"
IMG_SIZE    = 380
BATCH_SIZE  = 16
EPOCHS      = 22
ACCUM_STEPS = 2
TTA         = 5
N_FOLDS     = 5
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
        return img, row["label_idx"]


# ---------------------
# Augmentations
# ---------------------
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

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

tta_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0), ratio=(0.95,1.05)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# =========================================================
# Train each fold (save then delete)
# =========================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, _) in enumerate(skf.split(df, df["label_idx"])):

    print(f"\n===== FOLD {fold+1}/{N_FOLDS} =====")

    train_df = df.iloc[train_idx]
    train_ds = DatesDataset(train_df, os.path.join(BASE_PATH,"train"), train_tfms)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)

    model = timm.create_model(
        "tf_efficientnet_b4_ns",
        pretrained=True,
        num_classes=num_classes
    ).to(DEVICE)

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
    # Training loop
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

        print(f"Fold{fold+1} Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {running_loss/total:.4f} "
              f"Acc: {correct/total:.4f}")

    # SAVE, DELETE, CLEAR VRAM
    torch.save(model.state_dict(), f"model_fold{fold}.pth")
    del model
    torch.cuda.empty_cache()


# =========================================================
# inference (load 1 fold at a time)
# =========================================================
test_files = sorted(os.listdir(os.path.join(BASE_PATH,"test")))
pred_probs = []

for fold in range(N_FOLDS):

    print(f"\n=== Loading model_fold{fold}.pth ===")

    model = timm.create_model(
        "tf_efficientnet_b4_ns",
        pretrained=False,
        num_classes=num_classes
    ).to(DEVICE)

    model.load_state_dict(torch.load(f"model_fold{fold}.pth"))
    model.eval()

    fold_preds = []

    with torch.no_grad():
        for fname in test_files:
            img = Image.open(os.path.join(BASE_PATH,"test",fname)).convert("RGB")

            logits_sum = torch.zeros(1, num_classes).to(DEVICE)

            for _ in range(TTA):
                logits_sum += model(tta_tfms(img).unsqueeze(0).to(DEVICE))

            fold_preds.append(logits_sum.cpu().numpy())

    pred_probs.append(np.vstack(fold_preds))

    del model
    torch.cuda.empty_cache()


# average across folds
avg = np.mean(pred_probs, axis=0)
pred_idx = np.argmax(avg, axis=1)

preds = [(fname, label_list[i]) for fname, i in zip(test_files, pred_idx)]

submission = pd.DataFrame(preds, columns=["filename","label"])
submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv ðŸš€")



# from PIL import Image
# import matplotlib.pyplot as plt

# # check one random sample visually
# sample = df.sample(1).iloc[0]
# print("Random sample:", sample)

# img_path = os.path.join(BASE_PATH, "train", sample["filename"])
# print("Loading:", img_path)

# img = Image.open(img_path).convert("RGB")

# plt.imshow(img)
# plt.title(sample["label"])
# plt.axis("off")
# plt.show()



# import random

# plt.figure(figsize=(12,10))

# for i in range(16):
#     row = df.sample(1).iloc[0]
#     img_path = os.path.join(BASE_PATH, "train", row["filename"])
#     img = Image.open(img_path).convert("RGB")
    
#     plt.subplot(4,4,i+1)
#     plt.imshow(img)
#     plt.title(row["label"])
#     plt.axis("off")

# plt.tight_layout()
# plt.show()



# import matplotlib.pyplot as plt

# classes = df['label'].unique()

# for c in classes:
#     sample = df[df['label']==c].sample(1).iloc[0]
#     img = Image.open(os.path.join(BASE_PATH,"train", sample['filename'])).convert("RGB")
#     plt.figure(figsize=(3,3))
#     plt.imshow(img)
#     plt.title(c)
#     plt.axis("off")
#     plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from copy import deepcopy
# import torch
# from torchvision import transforms
# from PIL import Image
# import os

# # ---- SAME AUGS YOU USE IN TRAINING ----
# IMG_SIZE = 380

# train_tfms = transforms.Compose([
#     transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.15),
#     transforms.RandomRotation(15),
#     transforms.RandomApply(
#         [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04)],
#         p=0.7
#     ),
#     transforms.RandomAdjustSharpness(1.6, p=0.4),
#     transforms.RandomAutocontrast(p=0.4),
#     transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225]),
# ])

# # ---- unnormalize function ----
# def unnormalize(t):
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
#     std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
#     return t * std + mean


# # ---- visualize ----
# NUM_SHOW = 4

# sample_df = df.sample(NUM_SHOW).reset_index(drop=True)

# plt.figure(figsize=(10, NUM_SHOW * 4))

# for i in range(NUM_SHOW):
#     row = sample_df.iloc[i]
#     path = os.path.join(BASE_PATH, "train", row["filename"])
    
#     img_orig = Image.open(path).convert("RGB")
    
#     # apply augmentation
#     img_aug = train_tfms(deepcopy(img_orig))
#     img_aug = unnormalize(img_aug).clamp(0,1)
#     img_aug = img_aug.permute(1,2,0).cpu().numpy()
    
#     # show
#     plt.subplot(NUM_SHOW, 2, i*2 + 1)
#     plt.imshow(img_orig)
#     plt.title(f"Original â€“ {row['label']}")
#     plt.axis("off")
    
#     plt.subplot(NUM_SHOW, 2, i*2 + 2)
#     plt.imshow(img_aug)
#     plt.title("Augmented")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()



import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

train_path = "/kaggle/input/open-data-day-2025-dates-types-classification/train"
test_path  = "/kaggle/input/open-data-day-2025-dates-types-classification/test"
labels_csv = "/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv"



df = pd.read_csv(labels_csv)
df.head()
print(df.columns)



for label in df['label'].unique():
    samples = df[df.label==label].sample(5)
    fig, ax = plt.subplots(1,5, figsize=(12,3))
    for i,row in enumerate(samples.itertuples()):
        img = Image.open(os.path.join(train_path, row.filename))
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.suptitle(f"TRAIN - {label}")
    plt.show()



test_imgs = os.listdir(test_path)

for i in range(5):
    imgs = np.random.choice(test_imgs, 5, replace=False)
    fig, ax = plt.subplots(1,5, figsize=(12,3))
    for j,img_name in enumerate(imgs):
        img = Image.open(os.path.join(test_path, img_name))
        ax[j].imshow(img)
        ax[j].axis('off')
    plt.suptitle(f"TEST Random Group {i+1}")
    plt.show()



samples = df.sample(30)

fig, ax = plt.subplots(6,5, figsize=(15,15))

for i,row in enumerate(samples.itertuples()):
    img = Image.open(os.path.join(train_path, row.filename))
    ax[i//5, i%5].imshow(img)
    ax[i//5, i%5].axis('off')

plt.suptitle("TRAIN - Random 30")
plt.show()



from matplotlib.colors import rgb_to_hsv

samples = df.sample(8)  # choose 8 random train examples

fig, ax = plt.subplots(2,4, figsize=(14,6))

for i,row in enumerate(samples.itertuples()):
    img = Image.open(os.path.join(train_path, row.filename)).convert("RGB")
    hsv = rgb_to_hsv(np.array(img)/255.0)

    ax[i//4, i%4].imshow(hsv)
    ax[i//4, i%4].axis('off')

plt.suptitle("Random HSV Samples")
plt.show()



import numpy as np

h_values = []
s_values = []
v_values = []

for row in df.sample(300).itertuples():     # sample 300 for speed
    img = Image.open(os.path.join(train_path, row.filename)).convert("RGB")
    hsv = rgb_to_hsv(np.array(img)/255.0)

    h_values.extend(hsv[:,:,0].flatten())
    s_values.extend(hsv[:,:,1].flatten())
    v_values.extend(hsv[:,:,2].flatten())

plt.figure(figsize=(14,5))
plt.subplot(131); plt.hist(h_values, bins=50); plt.title("Hue")
plt.subplot(132); plt.hist(s_values, bins=50); plt.title("Saturation")
plt.subplot(133); plt.hist(v_values, bins=50); plt.title("Value")
plt.show()



import matplotlib.pyplot as plt

row = df.sample(1).iloc[0]
img = Image.open(os.path.join(train_path, row.filename)).convert("L")
img_np = np.array(img)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.imshow(img_np, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.hist(img_np.ravel(), bins=50)
plt.title("Intensity Histogram")
plt.show()


