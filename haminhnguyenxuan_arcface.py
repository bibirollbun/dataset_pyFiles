import os
import json
import math
import random
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import roc_curve, auc

# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 160
EMBEDDING_DIM = 512
BATCH_IDENTITIES = 16
IMAGES_PER_ID = 4
BATCH_SIZE = BATCH_IDENTITIES * IMAGES_PER_ID

EPOCHS_STAGE1 = 40
EPOCHS_STAGE2 = 20
WARMUP_EPOCHS = 2

BASE_LR = 3e-4
WEIGHT_DECAY = 1e-4

ARCFACE_MARGIN = 0.5
ARCFACE_SCALE = 64.0
TRIPLET_WEIGHT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PATHS
# ============================================================

BASE_DIR = '/kaggle/input/11-785-fall-20-homework-2-part-2'
TRAIN_DATASET_PATH = os.path.join(BASE_DIR, "classification_data/train_data")
VAL_DATASET_PATH = os.path.join(BASE_DIR, "classification_data/test_data")
VAL_PAIRS_FILE = os.path.join(BASE_DIR, "verification_pairs_val.txt")
OUTPUT_DIR = "/kaggle/working/"
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
OUTPUT_METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.json")

# ============================================================
# DATASETS
# ============================================================

class FaceFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.labels = {}
        self.transform = transform

        for idx, person in enumerate(sorted(os.listdir(root))):
            person_dir = os.path.join(root, person)
            if not os.path.isdir(person_dir):
                continue
            self.labels[person] = idx
            for img in os.listdir(person_dir):
                self.samples.append(
                    (os.path.join(person_dir, img), idx)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class BalancedIdentitySampler(Sampler):
    def __init__(self, labels, n_ids, n_imgs):
        self.labels = labels
        self.n_ids = n_ids
        self.n_imgs = n_imgs

        self.index_dict = defaultdict(list)
        for idx, label in enumerate(labels):
            self.index_dict[label].append(idx)

        self.labels_set = list(self.index_dict.keys())

    def __iter__(self):
        random.shuffle(self.labels_set)
        batch = []

        for label in self.labels_set:
            imgs = random.sample(
                self.index_dict[label],
                min(len(self.index_dict[label]), self.n_imgs)
            )
            batch.extend(imgs)
            if len(batch) == self.n_ids * self.n_imgs:
                yield from batch
                batch = []

    def __len__(self):
        return len(self.labels)


# ============================================================
# MODEL
# ============================================================

class FaceNet(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        backbone = models.mobilenet_v3_large(pretrained=True)
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.fc = nn.Linear(960, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return F.normalize(x)


class ArcFace(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        cosine = F.linear(embeddings, F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.margin)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = cosine * (1 - one_hot) + target_logits * one_hot
        return output * self.scale


# ============================================================
# LOSSES
# ============================================================

def triplet_loss(embeddings, labels, margin=0.3):
    dist = torch.cdist(embeddings, embeddings)
    loss = 0.0
    count = 0

    for i in range(len(embeddings)):
        pos = dist[i][labels == labels[i]]
        neg = dist[i][labels != labels[i]]
        if len(pos) > 1 and len(neg) > 0:
            loss += F.relu(pos.max() - neg.min() + margin)
            count += 1

    return loss / max(count, 1)


# ============================================================
# VALIDATION (ROC / EER)
# ============================================================

def evaluate(model, pairs_file, root, transform):
    model.eval()
    sims, labels = [], []

    with torch.no_grad():
        for line in open(pairs_file):
            p1, p2, lab = line.strip().split()
            img1 = transform(Image.open(os.path.join(root, p1)).convert("RGB")).unsqueeze(0).to(DEVICE)
            img2 = transform(Image.open(os.path.join(root, p2)).convert("RGB")).unsqueeze(0).to(DEVICE)

            e1 = model(img1)
            e2 = model(img2)
            sims.append(F.cosine_similarity(e1, e2).item())
            labels.append(int(lab))

    fpr, tpr, thresholds = roc_curve(labels, sims)
    roc_auc = auc(fpr, tpr)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    best_thr = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    return roc_auc, eer, best_thr

# def evaluate(model, pairs_file, root, transform):
#     model.eval()
#     sims, labels = [], []

#     with torch.no_grad():
#         for line in open(pairs_file):
#             parts = line.strip().split()
#             p1, p2, lab = parts[0], parts[1], parts[2]
            
#             # Remove 'verification_data/' prefix if it exists
#             p1 = p1.replace('verification_data/', '')
#             p2 = p2.replace('verification_data/', '')
            
#             img1_path = os.path.join(root, p1)
#             img2_path = os.path.join(root, p2)
            
#             # Check if files exist
#             if not os.path.exists(img1_path) or not os.path.exists(img2_path):
#                 print(f"Warning: Skipping missing pair: {img1_path} or {img2_path}")
#                 continue
            
#             img1 = transform(Image.open(img1_path).convert("RGB")).unsqueeze(0).to(DEVICE)
#             img2 = transform(Image.open(img2_path).convert("RGB")).unsqueeze(0).to(DEVICE)

#             e1 = model(img1)
#             e2 = model(img2)
#             sims.append(F.cosine_similarity(e1, e2).item())
#             labels.append(int(lab))

#     if len(sims) == 0:
#         print("ERROR: No valid pairs found!")
#         return 0.0, 1.0, 0.5

#     fpr, tpr, thresholds = roc_curve(labels, sims)
#     roc_auc = auc(fpr, tpr)
#     eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
#     best_thr = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]

#     return roc_auc, eer, best_thr
# ============================================================
# TRAINING LOOP
# ============================================================

def train():
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])

    train_ds = FaceFolderDataset(TRAIN_DATASET_PATH, transform)
    val_ds = FaceFolderDataset(VAL_DATASET_PATH, transform)

    sampler = BalancedIdentitySampler(
        [l for _, l in train_ds.samples],
        BATCH_IDENTITIES,
        IMAGES_PER_ID
    )

    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    model = FaceNet(EMBEDDING_DIM).to(DEVICE)
    arcface = ArcFace(EMBEDDING_DIM, len(train_ds.labels), ARCFACE_MARGIN, ARCFACE_SCALE).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS_STAGE1 + EPOCHS_STAGE2
    )

    scaler = GradScaler()

    best_auc = 0.0
    best_state = None
    best_threshold = 0.0

    # ================= STAGE 1 =================
    for epoch in range(EPOCHS_STAGE1):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                emb = model(imgs)
                logits = arcface(emb, labels)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        auc_val, eer, thr = evaluate(model, VAL_PAIRS_FILE, VAL_DATASET_PATH, transform)
        if auc_val > best_auc:
            best_auc = auc_val
            best_state = model.state_dict()
            best_threshold = thr

    # ================= STAGE 2 =================
    for epoch in range(EPOCHS_STAGE2):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                emb = model(imgs)
                arc_loss = F.cross_entropy(arcface(emb, labels), labels)
                tri_loss = triplet_loss(emb, labels)
                loss = arc_loss + TRIPLET_WEIGHT * tri_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        auc_val, eer, thr = evaluate(model, VAL_PAIRS_FILE, VAL_DATASET_PATH, transform)
        if auc_val > best_auc:
            best_auc = auc_val
            best_state = model.state_dict()
            best_threshold = thr

    # ========================================================
    # SAVE BEST MODEL (ONCE)
    # ========================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(best_state, OUTPUT_MODEL_PATH)

    metadata = {
        "embedding_dim": EMBEDDING_DIM,
        "img_size": IMG_SIZE,
        "threshold": float(best_threshold),
        "best_auc": float(best_auc),
        "model": "MobileNetV3 + ArcFace + Triplet"
    }

    with open(OUTPUT_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete. Best AUC:", best_auc)


if __name__ == "__main__":
    train()



!zip -r  arcfacetrain.zip /kaggle/working/

