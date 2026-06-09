# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from pathlib import Path

data_dir = os.path.join('.', '/kaggle/input/where-are-the-seagulls', 'data')
if os.path.exists(data_dir):
    print(f"Ğ¡Ñ‚Ñ€ÑƒĞºÑ‚ÑƒÑ€Ğ° Ğ´Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ğ¸:\n{os.listdir(data_dir)}")
else:
    print(f"Ğ”Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ñ� Ğ½Ğµ Ğ½Ğ°Ğ¹Ğ´ĞµĞ½Ğ°: {data_dir}")

# ĞŸÑƒÑ‚Ğ¸ Ğº Ğ´Ğ°Ğ½Ğ½Ñ‹Ğ¼
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

image_dir = Path(train_dir) / "images"
label_dir = Path(train_dir) / "labels"
test_image_dir = Path(test_dir) / "images"

# Ğ¡Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ²Ñ�ĞµÑ… Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹
image_paths = sorted(image_dir.glob("*.jpg"))
test_image_paths = sorted(test_image_dir.glob("*.jpg"))
print("Ğ’Ñ�ĞµĞ³Ğ¾ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹ Ğ² Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ĞµĞ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞµ:", len(image_paths))
print("Ğ’Ñ�ĞµĞ³Ğ¾ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹ Ğ² Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¾Ğ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞµ:", len(test_image_paths))

# ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¿ÑƒÑ�Ñ‚Ñ‹Ñ… Ğ¼ĞµÑ‚Ğ¾Ğº
empty_labels = 0
for img_path in image_paths:
    label_path = label_dir / (img_path.stem + ".txt")
    if not label_path.exists() or label_path.stat().st_size == 0:
        empty_labels += 1

print("Ğ˜Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹ Ğ±ĞµĞ· Ğ¾Ğ±ÑŠĞµĞºÑ‚Ğ¾Ğ² (Ğ¿ÑƒÑ�Ñ‚Ñ‹Ğµ Ğ¼ĞµÑ‚ĞºĞ¸):", empty_labels)


import matplotlib.pyplot as plt

# ĞŸĞ¾Ğ´Ñ�Ñ‡Ñ‘Ñ‚ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° bbox Ğ´Ğ»Ñ� ĞºĞ°Ğ¶Ğ´Ğ¾Ğ³Ğ¾ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ�
bbox_counts = []
for img_path in image_paths:
    label_path = label_dir / (img_path.stem + ".txt")
    if label_path.exists():
        with open(label_path, "r") as f:
            lines = f.readlines()
            bbox_counts.append(len(lines))
    else:
        bbox_counts.append(0)

# ĞŸĞ¾Ñ�Ñ‚Ñ€Ğ¾Ğ¸Ğ¼ Ğ³Ğ¸Ñ�Ñ‚Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ñƒ
plt.figure(figsize=(10, 5))
plt.hist(bbox_counts, bins=range(0, max(bbox_counts)+2), edgecolor='black')
plt.title("Ğ Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ ĞºĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ° Ğ¾Ğ±ÑŠĞµĞºÑ‚Ğ¾Ğ² Ğ½Ğ° Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğµ")
plt.xlabel("Ğ§Ğ¸Ñ�Ğ»Ğ¾ Ğ¾Ğ±ÑŠĞµĞºÑ‚Ğ¾Ğ² (bbox)")
plt.ylabel("ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹")
plt.xticks(range(0, max(bbox_counts)+1))
plt.grid(True)
plt.show()


from PIL import Image, ImageDraw
import random

# Ğ’Ñ‹Ğ±ĞµÑ€ĞµĞ¼ Ñ�Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ğ¾Ğµ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğµ Ñ� Ñ…Ğ¾Ñ‚Ñ� Ğ±Ñ‹ Ğ¾Ğ´Ğ½Ğ¸Ğ¼ bbox
valid_images = [p for p in image_paths if (label_dir / (p.stem + ".txt")).stat().st_size > 0]
sample_path = random.choice(valid_images)
sample_label_path = label_dir / (sample_path.stem + ".txt")

# Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ�
image = Image.open(sample_path).convert("RGB")
draw = ImageDraw.Draw(image)
w, h = image.size

# Ğ§Ñ‚ĞµĞ½Ğ¸Ğµ bbox Ğ² Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğµ YOLO
with open(sample_label_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 5:
            _, x_center, y_center, width, height = map(float, parts)
            # ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ² ĞºĞ¾Ğ¾Ñ€Ğ´Ğ¸Ğ½Ğ°Ñ‚Ñ‹ xmin, ymin, xmax, ymax
            xmin = (x_center - width / 2) * w
            xmax = (x_center + width / 2) * w
            ymin = (y_center - height / 2) * h
            ymax = (y_center + height / 2) * h
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

# ĞŸĞ¾ĞºĞ°Ğ¶ĞµĞ¼ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğµ
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.title(f"BBox visualization: {sample_path.name}")
plt.show()


# Ğ˜Ğ¼Ğ¿Ğ¾Ñ€Ñ‚ PyTorch
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# ĞŸĞ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹
IMG_SIZE = 640
BATCH_SIZE = 4
VAL_SPLIT = 0.2

# Ğ¢Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ğ¸Ğ¸
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Dataset ĞºĞ»Ğ°Ñ�Ñ�
class MyDataset(Dataset):
    def __init__(self, image_paths, label_dir, transform=None):
        self.image_paths = image_paths
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        if self.transform:
            image = self.transform(image)

        boxes = []
        if label_path.exists() and label_path.stat().st_size > 0:
            with open(label_path, "r") as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    _, xc, yc, bw, bh = parts

                    # ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ² xyxy Ğ¸ Ğ¼Ğ°Ñ�ÑˆÑ‚Ğ°Ğ±Ğ¸Ñ€ÑƒĞµĞ¼ Ğ½Ğ° Ñ€ĞµĞ°Ğ»ÑŒĞ½Ñ‹Ğµ Ñ€Ğ°Ğ·Ğ¼ĞµÑ€Ñ‹
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h

                    # Ğ¾Ñ‚Ñ„Ğ¸Ğ»ÑŒÑ‚Ñ€ÑƒĞµĞ¼ Ñ�Ğ»Ğ¸ÑˆĞºĞ¾Ğ¼ Ğ¼Ğ°Ğ»ĞµĞ½ÑŒĞºĞ¸Ğµ
                    if x2 - x1 > 1 and y2 - y1 > 1:
                        boxes.append([x1, y1, x2, y2])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        return image, boxes


class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, str(img_path)


# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğ¾Ğ²
full_dataset = MyDataset(image_paths, label_dir, transform=transform)
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

train_data, valid_data = random_split(full_dataset, [train_size, val_size])

# DataLoader'Ñ‹
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ğ¸Ğ¼ Ñ€Ğ°Ğ·Ğ¼ĞµÑ€Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ¿ĞµÑ€Ğ²Ğ¾Ğ¹ Ğ±Ğ°Ñ‚Ñ‡Ğ¸
batch = next(iter(train_loader))
imgs_batch, boxes_batch = batch[0], batch[1]
len(train_loader), len(valid_loader), imgs_batch[0].shape, boxes_batch[0].shape


test_data = TestDataset(test_image_paths, transform=transform)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ĞŸĞ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ ÑƒĞºĞ°Ğ·Ğ°Ğ½Ñ‹ Ğ´Ğ»Ñ� Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€Ğ°, Ğ¼Ğ¾Ğ¶ĞµÑ‚Ğµ Ğ¿ĞµÑ€ĞµĞ¾Ğ±Ñ€ĞµĞ´ĞµĞ»Ğ¸Ñ‚ÑŒ ĞºĞ°Ğº Ğ²Ğ°Ğ¼ ÑƒĞ´Ğ¾Ğ±Ğ½ĞµĞµ
def train_one_epoch(model, train_dataloader, optimizer, loss_fn=None, epoch=0, device='cuda', log_wandb=False, verbose=False):
    model.train()
    total_loss = 0

    for images, targets in train_dataloader:
        images = [img.to(device) for img in images]
        target_dicts = []
        for i, boxes in enumerate(targets):
            d = {"boxes": boxes.to(device), "labels": torch.ones((len(boxes),), dtype=torch.int64).to(device)}
            target_dicts.append(d)

        loss_dict = model(images, target_dicts)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    if verbose:
        print(f"[Ğ­Ğ¿Ğ¾Ñ…Ğ° {epoch}] ĞŸĞ¾Ñ‚ĞµÑ€Ğ¸ (Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ): {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def valid_one_epoch(model, valid_dataloader, loss_fn=None, epoch=0, device='cuda', log_wandb=False, verbose=False):
    model.eval()

    # Ğ¡Ğ¿Ğ¸Ñ�ĞºĞ¸ Ğ´Ğ»Ñ� Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸Ğº
    all_preds = []
    all_targets = []  # Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»ĞµĞ½Ğ¾ Ğ´Ğ»Ñ� Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ� Ğ¸Ñ�Ñ‚Ğ¸Ğ½Ğ½Ñ‹Ñ… Ğ¼ĞµÑ‚Ğ¾Ğº
    all_scores = []

    total_predictions = 0
    total_images = 0

    for images, targets in valid_dataloader:
        images = [img.to(device) for img in images]

        # Ğ¤Ğ¾Ñ€Ğ¼Ğ¸Ñ€ÑƒĞµĞ¼ Ñ†ĞµĞ»ĞµĞ²Ñ‹Ğµ Ğ¼ĞµÑ‚ĞºĞ¸ (1 ĞµÑ�Ğ»Ğ¸ ĞµÑ�Ñ‚ÑŒ Ñ‡Ğ°Ğ¹ĞºĞ¸, 0 ĞµÑ�Ğ»Ğ¸ Ğ½ĞµÑ‚)
        batch_targets = [1 if len(boxes) > 0 else 0 for boxes in targets]  # Ğ”Ğ¾Ğ±Ğ°Ğ²Ğ»ĞµĞ½Ğ¾ Ñ„Ğ¾Ñ€Ğ¼Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ¼ĞµÑ‚Ğ¾Ğº
        all_targets.extend(batch_targets)  # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ¼ĞµÑ‚ĞºĞ¸

        # ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ±ĞµĞ· target â†’ Ğ²Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµÑ‚ list of dicts
        outputs = model(images)

        # Ğ¤Ğ¾Ñ€Ğ¼Ğ¸Ñ€ÑƒĞµĞ¼ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� (1 ĞµÑ�Ğ»Ğ¸ ĞµÑ�Ñ‚ÑŒ Ñ…Ğ¾Ñ‚Ñ� Ğ±Ñ‹ Ğ¾Ğ´Ğ¸Ğ½Ğ° Ñ‡Ğ°Ğ¹ĞºĞ° Ñ� confidence > 0.5)
        batch_preds = []
        batch_scores = []
        for output in outputs:
            if len(output["scores"]) > 0:
                max_score = output["scores"].max().item()
                batch_scores.append(max_score)
                batch_preds.append(1 if max_score > 0.5 else 0)
            else:
                batch_scores.append(0.0)
                batch_preds.append(0)

        all_preds.extend(batch_preds)
        all_scores.extend(batch_scores)

        total_predictions += sum(len(output["boxes"]) for output in outputs)
        total_images += len(images)

    # Ğ’Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ñ�ĞµĞ¼ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_targets, all_scores)

    if verbose:
        print(f"\n[Ğ­Ğ¿Ğ¾Ñ…Ğ° {epoch}] ĞœĞµÑ‚Ñ€Ğ¸ĞºĞ¸ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

    avg_boxes = total_predictions / total_images
    if verbose:
        print(f"[Ğ­Ğ¿Ğ¾Ñ…Ğ° {epoch}] Ğ¡Ñ€ĞµĞ´Ğ½ĞµĞµ Ñ‡Ğ¸Ñ�Ğ»Ğ¾ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ½Ñ‹Ñ… bbox: {avg_boxes:.2f}")

    # Ğ’Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµĞ¼ Ğ²Ñ�Ğµ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸
    return avg_boxes, accuracy, precision, recall, f1, roc_auc


import pickle
from pathlib import Path
import torch

# Ğ£Ñ�Ñ‚Ñ€Ğ¾Ğ¹Ñ�Ñ‚Ğ²Ğ¾
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ğ¡ĞºĞ¾Ñ€Ğ¾Ñ�Ñ‚ÑŒ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�
LR = 1e-4

# ĞšĞ¾Ğ»Ğ¸Ñ‡ĞµÑ�Ñ‚Ğ²Ğ¾ Ñ�Ğ¿Ğ¾Ñ…
epochs = 50

model_dir = Path("./saved_models")  # Ğ”Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ñ� Ğ´Ğ»Ñ� Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»ĞµĞ¹

# Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ¿Ğ°Ğ¿ĞºÑƒ, ĞµÑ�Ğ»Ğ¸ ĞµÑ‘ Ğ½ĞµÑ‚
model_dir.mkdir(parents=True, exist_ok=True)

# Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ�Ğ»Ğ¾Ğ²Ğ°Ñ€ÑŒ Ğ´Ğ»Ñ� Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ�
config = {
    "comment": "ĞšĞ¾Ğ½Ñ„Ğ¸Ğ³ÑƒÑ€Ğ°Ñ†Ğ¸Ñ� Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�",
    "params": {
        "LR": LR,
        "epochs": epochs
    }
}

# Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ² Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ñ‹Ğ¹ Ñ„Ğ°Ğ¹Ğ»
param_path = model_dir / "config.pickle"
with open(param_path, 'wb') as f:  # 'wb' = write binary
    pickle.dump(config, f)

print(f"ĞšĞ¾Ğ½Ñ„Ğ¸Ğ³ ÑƒÑ�Ğ¿ĞµÑˆĞ½Ğ¾ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ‘Ğ½ Ğ² {param_path}")



import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights,
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights,
)

# Ğ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ğµ Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ğ¹ Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
def create_fasterrcnn(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def create_retinanet(num_classes=2):
    model = retinanet_resnet50_fpn(weights="DEFAULT")

    # ĞŸĞ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ğ´Ğ»Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ğ¾Ğ¹ Ğ³Ğ¾Ğ»Ğ¾Ğ²Ñ‹
    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors

    # Ğ—Ğ°Ğ¼ĞµĞ½Ñ�ĞµĞ¼ ĞºĞ»Ğ°Ñ�Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¾Ğ½Ğ½ÑƒÑ� Ğ³Ğ¾Ğ»Ğ¾Ğ²Ñƒ
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    return model

def create_ssdlite(num_classes=2):
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")

    # Ğ˜Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµĞ¼ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ, ÑƒĞºĞ°Ğ·Ğ°Ğ½Ğ½Ğ¾Ğµ Ğ² Ğ´Ğ¾ĞºÑƒĞ¼ĞµĞ½Ñ‚Ğ°Ñ†Ğ¸Ğ¸ TorchVision
    in_channels_list = model.head.classification_head.in_channels
    num_anchors_per_level = model.head.classification_head.num_anchors

    # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ½Ğ¾Ğ²ÑƒÑ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¾Ğ½Ğ½ÑƒÑ� Ğ³Ğ¾Ğ»Ğ¾Ğ²Ñƒ
    new_classification_head = SSDLiteClassificationHead(
        in_channels=in_channels_list,
        num_anchors=num_anchors_per_level,
        num_classes=num_classes
    )

    # ĞœĞµĞ½Ñ�ĞµĞ¼ Ğ³Ğ¾Ğ»Ğ¾Ğ²Ğ½ÑƒÑ� Ñ�ĞµÑ‚ÑŒ
    model.head.classification_head = new_classification_head

    return model

def create_fasterrcnn_mobilenet(num_classes=2):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


results = []
model_dir = Path("./saved_models")  # Ğ”Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ñ� Ğ´Ğ»Ñ� Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»ĞµĞ¹
model_dir.mkdir(exist_ok=True)  # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ğ´Ğ¸Ñ€ĞµĞºÑ‚Ğ¾Ñ€Ğ¸Ñ�, ĞµÑ�Ğ»Ğ¸ ĞµĞµ Ğ½ĞµÑ‚

# Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ğ¸Ğ· Ñ„Ğ°Ğ¹Ğ»Ğ° Ğ³Ğ¸Ğ¿ĞµÑ€-Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ²
with open(param_path, 'rb') as f:
    loaded_config = pickle.load(f)

for name, model_fn in [
    ("FasterRCNN-MobileNet", create_fasterrcnn_mobilenet),
    ("FasterRCNN", create_fasterrcnn),
    ("RetinaNet", create_retinanet)
]:
    model_path = model_dir / f"{name}.pt"
    metrics_history = []
    
    # ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ Ñ�ÑƒÑ‰ĞµÑ�Ñ‚Ğ²Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ñ„Ğ°Ğ¹Ğ»Ğ° Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
    if model_path.exists() and param_path.exists() and loaded_config["params"]["LR"] == LR and loaded_config["params"]["epochs"] == epochs:
        print(f"ĞœĞ¾Ğ´ĞµĞ»ÑŒ {name} ÑƒĞ¶Ğµ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ°. Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°ĞµĞ¼...")
        model = model_fn().to(device)
        model.load_state_dict(torch.load(model_path))
        
        # ĞŸÑ€Ğ¾Ğ²Ğ¾Ğ´Ğ¸Ğ¼ Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ�
        print(f"Ğ’Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸: {name}")
        avg_boxes, accuracy, precision, recall, f1, roc_auc = valid_one_epoch(
            model, valid_loader, epoch=0, device=device, verbose=False
        )
        
        metrics_history.append({
            'epoch': 0,
            'train_loss': None,  # ĞŸĞ¾Ñ‚ĞµÑ€Ñ� Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ� Ğ½Ğµ Ğ²Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ñ�Ğ»Ğ¸
            'avg_boxes': avg_boxes,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        })
    else:
        print(f"Ğ�Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸: {name}")
        model = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, epoch=epoch, device=device
            )
            
            avg_boxes, accuracy, precision, recall, f1, roc_auc = valid_one_epoch(
                model, valid_loader, epoch=epoch, device=device, verbose=True
            )

            metrics_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'avg_boxes': avg_boxes,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            })

        # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ¿Ğ¾Ñ�Ğ»Ğµ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ�
        torch.save(model.state_dict(), model_path)
        print(f"ĞœĞ¾Ğ´ĞµĞ»ÑŒ {name} Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ° Ğ² {model_path}")
    
    results.append((name, metrics_history))

# Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸ Ğ´Ğ»Ñ� Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´ÑƒÑ�Ñ‰ĞµĞ³Ğ¾ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ°
metrics_path = model_dir / "metrics_history.pkl"
with open(metrics_path, 'wb') as f:
    pickle.dump(results, f)
print(f"ĞœĞµÑ‚Ñ€Ğ¸ĞºĞ¸ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ñ‹ Ğ² {metrics_path}")


import pandas as pd
import matplotlib.pyplot as plt

for name, history in results:
    df = pd.DataFrame(history)

    plt.figure(figsize=(16, 10))
    plt.suptitle(f"ĞœĞµÑ‚Ñ€Ğ¸ĞºĞ¸ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸: {name}", fontsize=16)

    plt.subplot(2, 3, 1)
    plt.plot(df['epoch'], df['train_loss'], marker='o')
    plt.title("ĞŸĞ¾Ñ‚ĞµÑ€Ğ¸ (Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ)")
    plt.xlabel("Ğ­Ğ¿Ğ¾Ñ…Ğ°")
    plt.ylabel("ĞŸĞ¾Ñ‚ĞµÑ€Ğ¸")

    plt.subplot(2, 3, 2)
    plt.plot(df['epoch'], df['accuracy'], marker='o')
    plt.title("Accuracy")
    plt.xlabel("Ğ­Ğ¿Ğ¾Ñ…Ğ°")
    plt.ylabel("Accuracy")

    plt.subplot(2, 3, 3)
    plt.plot(df['epoch'], df['precision'], marker='o')
    plt.title("Precision")
    plt.xlabel("Ğ­Ğ¿Ğ¾Ñ…Ğ°")
    plt.ylabel("Precision")

    plt.subplot(2, 3, 4)
    plt.plot(df['epoch'], df['recall'], marker='o')
    plt.title("Recall")
    plt.xlabel("Ğ­Ğ¿Ğ¾Ñ…Ğ°")
    plt.ylabel("Recall")

    plt.subplot(2, 3, 5)
    plt.plot(df['epoch'], df['f1'], marker='o')
    plt.title("F1 Ğ¼ĞµÑ€Ğ°")
    plt.xlabel("Ğ­Ğ¿Ğ¾Ñ…Ğ°")
    plt.ylabel("F1")

    plt.subplot(2, 3, 6)
    plt.plot(df['epoch'], df['roc_auc'], marker='o')
    plt.title("ROC AUC")
    plt.xlabel("Ğ­Ğ¿Ğ¾Ñ…Ğ°")
    plt.ylabel("ROC AUC")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# ĞœĞµÑ‚Ñ€Ğ¸ĞºĞ¸, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğµ Ğ±ÑƒĞ´ĞµĞ¼ Ğ²Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ¸Ñ€Ğ¾Ğ²Ğ°Ñ‚ÑŒ
metric_names = ["train_loss", "accuracy", "precision", "recall", "f1", "roc_auc"]

# ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ¸Ğ· results Ğ² DataFrame
model_dfs = {name: pd.DataFrame(history) for name, history in results}

# Ğ Ğ¸Ñ�ÑƒĞµĞ¼ Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ¸ Ğ¿Ğ¾ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ°Ğ¼
for metric in metric_names:
    plt.figure(figsize=(8, 5))
    plt.title(f"Ğ¡Ñ€Ğ°Ğ²Ğ½ĞµĞ½Ğ¸Ğµ Ğ¼Ğ¾Ğ´ĞµĞ»ĞµĞ¹ Ğ¿Ğ¾ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞµ: {metric}")
    plt.xlabel("Ğ­Ğ¿Ğ¾Ñ…Ğ°")
    plt.ylabel(metric)

    for name, df in model_dfs.items():
        plt.plot(df["epoch"], df[metric], marker='o', label=name)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


!pip install opencv-python


import cv2
import numpy as np

'''
Ğ¤ÑƒĞ½ĞºÑ†Ğ¸Ñ� to_yolo Ğ¿Ñ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµÑ‚ ĞºĞ¾Ğ¾Ñ€Ğ´Ğ¸Ğ½Ğ°Ñ‚Ñ‹ Ğ¾Ğ³Ñ€Ğ°Ğ½Ğ¸Ñ‡Ğ¸Ğ²Ğ°Ñ�Ñ‰Ğ¸Ñ… Ğ¿Ñ€Ñ�Ğ¼Ğ¾ÑƒĞ³Ğ¾Ğ»ÑŒĞ½Ğ¸ĞºĞ¾Ğ² (bounding boxes)
Ğ¸Ğ· Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğ° PASCAL VOC (xyxy) Ğ² Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚ YOLO (cx, cy, w, h),
Ñ� Ğ½Ğ¾Ñ€Ğ¼Ğ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸ĞµĞ¹ Ğ¿Ğ¾ ÑˆĞ¸Ñ€Ğ¸Ğ½Ğµ Ğ¸ Ğ²Ñ‹Ñ�Ğ¾Ñ‚Ğµ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ�.
'''
def to_yolo(boxes, img_height, img_width):
    boxes = np.atleast_2d(boxes).astype(np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = ((x1 + x2) / 2) / img_width
    cy = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    return np.stack([cx, cy, w, h], axis=1)

# Ğ�Ñ†ĞµĞ½ĞºĞ° Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¾Ğ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞµ
test_results = []
model_dict = {}

for name, history in results:
    print(f"\nĞ¢ĞµÑ�Ñ‚Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸: {name}")

    model = None
    model_path = model_dir / f"{name}.pt"
    if name == "FasterRCNN":
      model = create_fasterrcnn().to(device)
      model.load_state_dict(torch.load(model_path, map_location=device))
    elif name == "FasterRCNN-MobileNet":
      model = create_fasterrcnn_mobilenet().to(device)
      model.load_state_dict(torch.load(model_path, map_location=device))
    elif name == "RetinaNet":
      model = create_retinanet().to(device)
      model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model_dict[name] = model

    all_predictions = []
    submission = []
    CONF_THRESHOLD = 0.5

    for imgs_batch, paths_batch in test_loader:
        imgs_batch = list(img.to(device) for img in imgs_batch)
        with torch.no_grad():
            # ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ´Ğ»Ñ� ĞºĞ°Ğ¶Ğ´Ğ¾Ğ³Ğ¾ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ñ� Ğ² Ğ±Ğ°Ñ‚Ñ‡Ğµ
            preds = model(imgs_batch)
            all_predictions.extend(preds)

        for i, pred in enumerate(preds):
            img_path = paths_batch[i]
            filename = os.path.basename(img_path)
            orig_img = cv2.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2]

            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            valid_boxes = boxes[scores >= CONF_THRESHOLD]

            if len(valid_boxes) == 0:
                final_bbox_str = '-1'
            else:
                valid_boxes[:, [0, 2]] *= orig_w / IMG_SIZE
                valid_boxes[:, [1, 3]] *= orig_h / IMG_SIZE
                yolo_boxes = to_yolo(valid_boxes, orig_h, orig_w)
                box_strings = [
                  f"0 {y[0]:.6f} {y[1]:.6f} {y[2]:.6f} {y[3]:.6f}"
                  for y in yolo_boxes
                ]
                final_bbox_str = " ".join(box_strings)

            submission.append({
              "filename": os.path.basename(img_path),
              "bbox": final_bbox_str,
              "model": name
            })

    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(f"submission_{name}.csv", index=False)

    num_images = len(all_predictions)
    num_with_objects = sum(len(p['boxes']) > 0 for p in all_predictions)
    percent_with_objects = num_with_objects / num_images * 100 if num_images > 0 else 0

    print(f"Ğ’Ñ�ĞµĞ³Ğ¾ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹: {num_images}")
    print(f"Ğ˜Ğ· Ğ½Ğ¸Ñ… Ñ� Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ½Ñ‹Ğ¼Ğ¸ Ğ¾Ğ±ÑŠĞµĞºÑ‚Ğ°Ğ¼Ğ¸: {num_with_objects} ({percent_with_objects:.2f}%)")

    test_results.append({
        'model': name,
        'num_images': num_images,
        'num_with_objects': num_with_objects,
        'percent_with_objects': percent_with_objects,
        'predictions': all_predictions,
    })


from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

# Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğ¹
def visualize_top_predictions(test_loader, model_dict, device, conf_threshold=0.5, top_k=2):
    for model_name, model in model_dict.items():
        print(f"\nğŸ”� {model_name}")

        scored_images = []
        with torch.no_grad():
            for imgs_batch, paths_batch in test_loader:
                imgs_batch = [img.to(device) for img in imgs_batch]
                preds = model(imgs_batch)

                for img, pred, path in zip(imgs_batch, preds, paths_batch):
                    boxes = pred['boxes'][pred['scores'] >= conf_threshold].cpu()
                    scored_images.append((img.cpu(), boxes, path, len(boxes)))

        # Ğ¡Ğ¾Ñ€Ñ‚Ğ¸Ñ€Ğ¾Ğ²ĞºĞ° Ğ¿Ğ¾ Ñ‡Ğ¸Ñ�Ğ»Ñƒ bbox
        top_images = sorted(scored_images, key=lambda x: x[3], reverse=True)[:top_k]

        fig, axes = plt.subplots(1, top_k, figsize=(top_k * 5, 5))
        if top_k == 1:
            axes = [axes]

        for (img, boxes, path, count), ax in zip(top_images, axes):
            image_with_boxes = draw_bounding_boxes((img * 255).byte(), boxes, colors='red', width=2)
            ax.imshow(to_pil_image(image_with_boxes))
            ax.set_title(f"{os.path.basename(path)}\nĞ§Ğ°ĞµĞº: {count}")
            ax.axis('off')

        plt.suptitle(f"Top-{top_k} detections â€” {model_name}")
        plt.tight_layout()
        plt.show()


visualize_top_predictions(test_loader, model_dict, device)


import ast
import pandas as pd
import matplotlib.pyplot as plt

# ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·ÑƒĞµĞ¼ Ğ² DataFrame
df = pd.DataFrame(test_results)

# Ğ£Ğ´Ğ°Ğ»Ğ¸Ğ¼ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºÑƒ predictions, ĞµÑ�Ğ»Ğ¸ Ğ¾Ğ½Ğ° ĞµÑ�Ñ‚ÑŒ
if 'predictions' in df.columns:
    df = df.drop(columns=['predictions'])

column_labels = {
    'model': 'ĞœĞ¾Ğ´ĞµĞ»ÑŒ',
    'num_images': 'Ğ’Ñ�ĞµĞ³Ğ¾ Ğ¸Ğ·Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğ¹',
    'num_with_objects': 'Ğ¡ Ğ¾Ğ±ÑŠĞµĞºÑ‚Ğ°Ğ¼Ğ¸',
    'percent_with_objects': '% Ñ� Ğ¾Ğ±ÑŠĞµĞºÑ‚Ğ°Ğ¼Ğ¸'
}

# ĞŸĞµÑ€ĞµĞ¸Ğ¼ĞµĞ½ÑƒĞµĞ¼ Ğ¸ Ğ²Ñ‹Ğ±ĞµÑ€ĞµĞ¼ Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ğ½ÑƒĞ¶Ğ½Ñ‹Ğµ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ñ‹
df = df[[col for col in column_labels if col in df.columns]]
df.rename(columns=column_labels, inplace=True)

# Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�
fig, ax = plt.subplots(figsize=(9, 2))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title("Ğ ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ñ‹ Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¾Ğ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞµ", fontweight='bold')
plt.show()

