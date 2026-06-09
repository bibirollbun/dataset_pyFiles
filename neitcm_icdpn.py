# ==============================================================================
# Baseline_Sanity.ipynb
# Chuáº©n hÃ³a pipeline & baseline cho ALASKA2 steganalysis
# ==============================================================================

# ==============================================================================
# BÆ¯á»šC 1: CÃ€I Ä�áº¶T VÃ€ CHUáº¨N Bá»Š MÃ”I TRÆ¯á»œNG
# ==============================================================================

print("--- CÃ i Ä‘áº·t cÃ¡c thÆ° viá»‡n cáº§n thiáº¿t ---")
!pip install -q clip faiss-cpu scikit-learn torch torchvision torchaudio

# ==============================================================================
# BÆ¯á»šC 2: IMPORT THÆ¯ VIá»†N CÆ  Báº¢N
# ==============================================================================

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from torch.nn import Identity
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler


# ==============================================================================
# BÆ¯á»šC 3: CLONE REPO
# ==============================================================================

repo_path = "/kaggle/working/alaska2-steganalysis"

if os.path.exists(repo_path):
    print("\nThÆ° má»¥c 'alaska2-steganalysis' Ä‘Ã£ tá»“n táº¡i. Ä�ang xÃ³a...")
    !rm -r /kaggle/working/alaska2-steganalysis

print("\n--- Clone repository 'alaska2-steganalysis' ---")
!git clone https://github.com/Rinovative/alaska2-steganalysis.git {repo_path}

print("\n--- CÃ i Ä‘áº·t thÆ° viá»‡n phá»¥ thuá»™c 'conseal' ---")
!pip install -q git+https://github.com/Rinovative/conseal.git

sys.path.append(repo_path)

# ==============================================================================
# BÆ¯á»šC 4: GHIM SEED Cá»� Ä�á»ŠNH
# ==============================================================================

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("âœ“ Seed cá»‘ Ä‘á»‹nh thÃ nh cÃ´ng (42)")

# ==============================================================================
# BÆ¯á»šC 5: Ä�á»ŠNH NGHÄ¨A Láº I HÃ€M create_model
# ==============================================================================

print("\n--- Ä�á»‹nh nghÄ©a láº¡i hÃ m create_model ---")
def create_model(name: str):
    """Factory for model creation"""
    if name == 'conseal_mobilenet':
        model = mobilenet_v2(weights="IMAGENET1K_V1")
        model.classifier = Identity()
        model.out_features = 1280
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model

# ==============================================================================
# BÆ¯á»šC 6: IMPORT MODULE TRONG Dá»° Ã�N
# ==============================================================================

print("\n--- Import cÃ¡c modules cÃ²n láº¡i ---")
try:
    from src.util import util_data, util_nb as util_pipeline
    print("âœ“ CÃ¡c modules Ä‘Ã£ Ä‘Æ°á»£c import thÃ nh cÃ´ng.")
except Exception as e:
    print(f"âœ— Lá»—i khi import modules: {e}")
    sys.exit()

# ==============================================================================
# BÆ¯á»šC 7: KIá»‚M TRA Dá»® LIá»†U
# ==============================================================================

DATA_DIR = "/kaggle/input/alaska2-image-steganalysis"
util_data.DATA_DIR = DATA_DIR
print(f"Dataset path Ä‘Ã£ Ä‘Æ°á»£c thiáº¿t láº­p: {util_data.DATA_DIR}")

try:
    img_path = f"{DATA_DIR}/Cover/00001.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Sample Cover Image")
    plt.axis("off")
    plt.show()
    print("âœ“ Hiá»ƒn thá»‹ áº£nh máº«u thÃ nh cÃ´ng.")
except Exception as e:
    print(f"âœ— Lá»—i khi hiá»ƒn thá»‹ áº£nh: {e}")
    sys.exit()

# ==============================================================================
# BÆ¯á»šC 8: Táº O INDEX, CHIA DATASET (SANITY-CHECK)
# ==============================================================================

print("\n--- Táº¡o index vÃ  chia dataset ---")
try:
    dataset_root = DATA_DIR
    class_labels = {"Cover": 0, "JMiPOD": 1, "JUNIWARD": 2, "UERD": 3}
    
    print("Ä�ang táº¡o file index vá»›i 1% dá»¯ liá»‡u (sanity-check)...")
    df = util_data.build_file_index(dataset_root, class_labels, subsample_percent=0.01)
    df_meta = util_data.add_jpeg_metadata(df, quiet=True)
    train_df, val_df, test_df = util_data.split_dataset_by_filename(df_meta)

    print("âœ“ Index vÃ  chia dataset thÃ nh cÃ´ng.")
    print(f"KÃ­ch thÆ°á»›c train set: {len(train_df)}")
    print(f"KÃ­ch thÆ°á»›c validation set: {len(val_df)}")
    print(f"KÃ­ch thÆ°á»›c test set: {len(test_df)}")
    
    print("\nDataFrame Ä‘áº§u tiÃªn:")
    print(train_df.head())

except Exception as e:
    print(f"âœ— Ä�Ã£ xáº£y ra lá»—i khi táº¡o index: {e}")
    sys.exit()


# ==============================================================================
# STEP 5: FEATURE-BASED BASELINE (SRM + GLCM + SVM/RF)
# ==============================================================================

from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Map label_name sang binary: Cover=0, Stego=1
label_map = {"Cover":0, "JMiPOD":1, "JUNIWARD":1, "UERD":1}
for df_ in [train_df, val_df, test_df]:
    df_['label'] = df_['label_name'].map(label_map)

def srm_features(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    kernels = [np.array([[0,1,0],[1,-4,1],[0,1,0]]),
               np.array([[1,-2,1],[-2,4,-2],[1,-2,1]])]
    feats = []
    for k in kernels:
        conv = cv2.filter2D(gray, -1, k)
        feats.extend([conv.mean(), conv.std(), np.median(conv)])
    return np.array(feats)

def glcm_features(img, distances=[1], angles=[0,np.pi/4,np.pi/2,3*np.pi/4]):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
    feats = []
    for prop in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']:
        feats.extend(graycoprops(glcm, prop).flatten())
    return np.array(feats)

def extract_features(img):
    return np.concatenate([srm_features(img), glcm_features(img)])

def build_feature_matrix(df, img_col='path', label_col='label'):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row[img_col]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(extract_features(img))
        y.append(row[label_col])
    return np.array(X), np.array(y)

X_train, y_train = build_feature_matrix(train_df)
X_val, y_val     = build_feature_matrix(val_df)
X_test, y_test   = build_feature_matrix(test_df)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# --- SVM ---
svm_clf = SVC(kernel='linear', probability=True)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM F1:", f1_score(y_test, y_pred_svm))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# --- Random Forest ---
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))
print("RF F1:", f1_score(y_test, y_pred_rf))
print("RF Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# ==============================================================================
# STEP 6: TINY CNN BASELINE
# ==============================================================================

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image

class StegoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform: img = self.transform(img)
        label = row['label']
        return img, int(label)

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

train_ds = StegoDataset(train_df, transform)
val_ds   = StegoDataset(val_df, transform)
test_ds  = StegoDataset(test_df, transform)

from torch.utils.data import WeightedRandomSampler
import numpy as np

# --- TÃ­nh trá»�ng sá»‘ cho tá»«ng class ---
class_counts = np.bincount(train_df['label'])   # Ä‘áº¿m sá»‘ lÆ°á»£ng máº«u theo class
class_weights = 1. / class_counts               # trá»�ng sá»‘ ngÆ°á»£c
weights = [class_weights[label] for label in train_df['label']]

# --- Khá»Ÿi táº¡o sampler ---
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# --- Táº¡o DataLoader ---
train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

class TinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64*16*16, num_classes)
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Training loop ---
for epoch in range(2):
    model.train()
    total, correct = 0,0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs,1)
        total += labels.size(0)
        correct += (preds==labels).sum().item()
    print(f"Epoch {epoch+1} - Train Acc: {correct/total:.4f}")


# FINAL (complete) - SRNet + Balanced sampler + Binary Focal Loss + TTA + Youden+F1 threshold tuning
import os, math, copy, random, io
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc, classification_report,
                             balanced_accuracy_score)

# NOTE: this script assumes util_data, train_df, val_df, test_df already defined (as in your notebook)

# ----------------------------
# 0) SEED + DEVICE
# ----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

SEED = 42
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_PIN_MEMORY = True if DEVICE.type == "cuda" else False
print("Device:", DEVICE)

# ----------------------------
# 1) SRM kernel + transforms (PIL-level -> tensor-level)
# ----------------------------
class RandomJPEG:
    def __init__(self, p=0.35, qualities=(95,85,75)):
        self.p = p
        self.qualities = qualities
    def __call__(self, pil_img):
        if random.random() < self.p:
            q = random.choice(self.qualities)
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG', quality=int(q))
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return pil_img

class AddGaussianNoise:
    def __init__(self, p=0.25, std=(0.002, 0.01)):
        self.p = p
        self.std_low, self.std_high = std
    def __call__(self, pil_img):
        if random.random() >= self.p:
            return pil_img
        arr = np.array(pil_img).astype(np.float32) / 255.0
        std = random.uniform(self.std_low, self.std_high)
        noise = np.random.normal(0.0, std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

class ToTensorGray:
    def __call__(self, pil_img):
        return transforms.ToTensor()(pil_img.convert('L'))

pre_tensor_transforms = [
    transforms.RandomResizedCrop(size=256, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    RandomJPEG(p=0.35, qualities=(95,85,75)),
    AddGaussianNoise(p=0.25, std=(0.002, 0.01)),
    transforms.ColorJitter(brightness=0.10, contrast=0.10),
]

train_tf = transforms.Compose([
    *pre_tensor_transforms,
    ToTensorGray(),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.06), ratio=(0.3, 3.3), value=0)
])

val_tf = transforms.Compose([
    transforms.Resize((256,256)),
    ToTensorGray()
])

# ----------------------------
# 2) DATASET (grayscale)
# ----------------------------
class StegoDataset(Dataset):
    def __init__(self, df, transform=None, path_col='path', label_col='label'):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path_col = path_col
        self.label_col = label_col
        self.labels = self.df[self.label_col].astype(int).values
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        r = self.df.loc[idx]
        p = r[self.path_col]
        if not os.path.isabs(p):
            p = os.path.join(util_data.DATA_DIR, p)
        pil = Image.open(p).convert("RGB")
        x = self.transform(pil) if self.transform else ToTensorGray()(pil)
        y = int(r[self.label_col])
        return x, y

train_ds = StegoDataset(train_df, transform=train_tf)
val_ds   = StegoDataset(val_df,   transform=val_tf)
test_ds  = StegoDataset(test_df,  transform=val_tf)

# ----------------------------
# 3) Balanced sampler (from train_df)
# ----------------------------
labels_np = train_df['label'].values.astype(int)
classes, counts = np.unique(labels_np, return_counts=True)
print("Train class counts:", dict(zip(classes, counts)))

inv_freq = {int(c): (1.0 / float(cnt)) for c, cnt in zip(classes, counts)}
sample_weights = np.array([inv_freq[int(l)] for l in labels_np], dtype=np.float32)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

BATCH_SIZE = 32
NUM_WORKERS = 4
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=USE_PIN_MEMORY)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=USE_PIN_MEMORY)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=USE_PIN_MEMORY)

# ----------------------------
# 4) MODEL: SRNet (Steganalysis Residual Network) - Full Version
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to handle different dimensions
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: SRM convolution
        srm_kernel = self._get_srm_kernel()
        self.srm_conv = nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.srm_conv.weight = nn.Parameter(srm_kernel, requires_grad=False)
        self.srm_bn = nn.BatchNorm2d(3)
        self.srm_relu = nn.ReLU(inplace=True)

        # Layers with pooling
        self.conv1 = nn.Conv2d(3, 30, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(30)
        self.conv2 = nn.Conv2d(30, 30, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(30)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.res1 = ResidualBlock(30, 32, stride=1)
        self.res2 = ResidualBlock(32, 32, stride=2)
        self.res3 = ResidualBlock(32, 64, stride=1)
        self.res4 = ResidualBlock(64, 64, stride=2)
        self.res5 = ResidualBlock(64, 128, stride=1)
        self.res6 = ResidualBlock(128, 128, stride=2)

        # Final layers
        self.conv_final = nn.Conv2d(128, 512, kernel_size=3, padding='same')
        self.bn_final = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 1)

    def _get_srm_kernel(self):
        srm_kernel = np.array([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)
        srm_kernel = srm_kernel / 4.0
        srm_kernel = np.stack([srm_kernel, srm_kernel, srm_kernel]) # 3x5x5
        srm_kernel = np.expand_dims(srm_kernel, 1) # 3x1x5x5
        return torch.from_numpy(srm_kernel)

    def forward(self, x):
        x = self.srm_conv(x)
        x = self.srm_bn(x)
        x = self.srm_relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.srm_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.srm_relu(x)
        x = self.pool1(x)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        
        x = self.conv_final(x)
        x = self.bn_final(x)
        x = self.srm_relu(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x).squeeze(1)
        return x

model = SRNet().to(DEVICE)
print("Using SRNet (Full) architecture.")
# ----------------------------
# 5) Binary focal loss (class-aware alpha computed from inv_freq)
# ----------------------------
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=(0.5,0.5), gamma=2.0, reduction='mean'):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = alpha_t * ((1 - pt) ** self.gamma) * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

neg_if = inv_freq.get(0, 1.0)
pos_if = inv_freq.get(1, 1.0)
alpha_neg = neg_if / (neg_if + pos_if)
alpha_pos = pos_if / (neg_if + pos_if)
print(f"Binary focal alpha (neg,pos) = ({alpha_neg:.3f}, {alpha_pos:.3f})")
criterion = BinaryFocalLoss(alpha=(alpha_neg, alpha_pos), gamma=2.0).to(DEVICE)

# ----------------------------
# 6) OPTIM + LR schedule (fixed epochs)
# ----------------------------
# For SRNet, we unfreeze all layers from the beginning
EPOCHS_FULL = 25
GRAD_CLIP = 1.0
LR_FULL = 1e-5 # Tá»‘c Ä‘á»™ há»�c ban Ä‘áº§u Ä‘Ã£ Ä‘Æ°á»£c Ä‘iá»�u chá»‰nh

optim_full = torch.optim.Adam(model.parameters(), lr=LR_FULL, weight_decay=1e-5)

# Sá»­a láº¡i hÃ m cosine_lr Ä‘á»ƒ thÃªm warmup
WARMUP_EPOCHS = 2 # Ä�áº·t 2 epoch Ä‘áº§u tiÃªn Ä‘á»ƒ warmup
def cosine_lr_with_warmup(base_lr, step, total_steps, warmup_steps, min_lr=1e-7): # min_lr giáº£m xuá»‘ng
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    cos = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    return min_lr + (base_lr - min_lr) * cos

# ----------------------------
# 7) helpers: predict probs (sigmoid), threshold tuning (Youden+F1), eval, TTA
# ----------------------------
def predict_probs(model, loader):
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy()
            ys.extend(y.numpy().tolist())
            probs.extend(p.tolist())
    return np.array(ys), np.array(probs)

def find_best_threshold(model, loader):
    y_true, y_prob = predict_probs(model, loader)
    if len(y_true) == 0:
        return 0.5
    if len(np.unique(y_true)) < 2:
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.01,0.99,99):
            y_pred = (y_prob >= t).astype(int)
            f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
            if f1m > best_f1:
                best_f1, best_t = f1m, float(t)
        return float(best_t)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_thr_youden = thr[np.argmax(youden)]
    best_thr_f1, best_f1 = 0.5, -1
    for t in np.linspace(0.1,0.9,81):
        preds = (y_prob>=t).astype(int)
        f1m = f1_score(y_true, preds, average='macro', zero_division=0)
        if f1m > best_f1:
            best_f1, best_thr_f1 = f1m, float(t)
    if best_f1 > youden.max():
        return float(best_thr_f1)
    return float(best_thr_youden)

def evaluate_with_thresh(model, loader, threshold):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy()
            preds = (p >= threshold).astype(int)
            y_true.extend(y.numpy().tolist()); y_pred.extend(preds.tolist()); y_prob.extend(p.tolist())
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    metrics = {}
    if len(y_true) == 0:
        return metrics
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['bal_acc'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['prec_w'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_w'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['cm'] = confusion_matrix(y_true, y_pred)
    metrics['y_true'] = y_true; metrics['y_pred'] = y_pred; metrics['y_prob'] = y_prob
    try:
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            metrics['auc'] = auc(fpr, tpr)
        else:
            metrics['auc'] = np.nan
    except Exception:
        metrics['auc'] = np.nan
    return metrics

def tta_predict_probs(model, x_tensor):
    model.eval()
    with torch.no_grad():
        logits_list = []
        logits_list.append(model(x_tensor))
        logits_list.append(model(torch.flip(x_tensor, dims=[3]))) # hflip
        logits_list.append(model(torch.flip(x_tensor, dims=[2]))) # vflip
        try:
            x_rot = x_tensor.transpose(2,3).contiguous()
            logits_list.append(model(x_rot))
        except Exception:
            pass
        probs = torch.stack([torch.sigmoid(l) for l in logits_list], dim=0).mean(0)
    return probs.cpu().numpy()

# ----------------------------
# ğŸ˜� TRAIN: full finetune for ~25 epochs
# ----------------------------
SAVE_DIR = "./checkpoints"; os.makedirs(SAVE_DIR, exist_ok=True)
RUN_NAME = "srnet_binary_focal"
best_state, best_epoch, best_val_f1 = None, -1, -1.0
best_threshold = 0.5

print("\n[Phase 1] full training with SRNet")
total_steps = EPOCHS_FULL * len(train_loader)
step = 0
for ep in range(1, EPOCHS_FULL + 1):
    model.train()
    total_loss = 0.0; total_samples = 0; corrects = 0
    for x, y in train_loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        
        # Sá»­a láº¡i hÃ m gá»�i cosine_lr
        for gi, pg in enumerate(optim_full.param_groups):
            pg['lr'] = cosine_lr_with_warmup(LR_FULL, step, total_steps, WARMUP_EPOCHS * len(train_loader))
        
        step += 1
        optim_full.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim_full.step()
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).long()
            corrects += (preds == y).sum().item()
    train_acc = corrects / float(total_samples) if total_samples > 0 else 0.0
    t_best = find_best_threshold(model, val_loader)
    val_m = evaluate_with_thresh(model, val_loader, t_best)
    val_f1m = val_m.get('f1_macro', 0.0)
    print(f"Ep{ep} | train_acc={train_acc:.4f} | train_loss={total_loss/total_samples if total_samples>0 else 0:.4f} | val_f1m={val_f1m:.4f} | bal_acc={val_m.get('bal_acc',np.nan):.4f} | AUC={val_m.get('auc',np.nan):.3f} | t={t_best:.3f}")
    if val_f1m > best_val_f1:
        best_val_f1 = val_f1m; best_epoch = ep
        best_state = copy.deepcopy(model.state_dict()); best_threshold = float(t_best)
        torch.save({'state':best_state,'threshold':best_threshold}, os.path.join(SAVE_DIR, f"{RUN_NAME}_best.pth"))

print(f"\nDone. Best val macro-F1: {best_val_f1:.4f} @ epoch {best_epoch} (threshold={best_threshold:.3f})")

# ----------------------------
# 9) Test (TTA) + metrics + ROC/AUC using best checkpoint + threshold
# ----------------------------
ck = torch.load(os.path.join(SAVE_DIR, f"{RUN_NAME}_best.pth"))
model.load_state_dict(ck['state'])
best_threshold = float(ck['threshold'])
model.to(DEVICE)
model.eval()

y_true_all, y_prob_all = [], []
with torch.no_grad():
    for x,y in test_loader:
        x = x.to(DEVICE)
        probs = tta_predict_probs(model, x)
        y_prob_all.extend(probs.tolist())
        y_true_all.extend(y.numpy().tolist())

y_true_all = np.array(y_true_all); y_prob_all = np.array(y_prob_all)
y_pred_all = (y_prob_all >= best_threshold).astype(int)

acc    = accuracy_score(y_true_all, y_pred_all)
bal    = balanced_accuracy_score(y_true_all, y_pred_all)
f1m    = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
f1w    = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
precw = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
recw  = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
cm    = confusion_matrix(y_true_all, y_pred_all)

print("\n--- TEST METRICS (best checkpoint + tuned threshold) ---")
print(f"Threshold used: {best_threshold:.3f}")
print(f"Accuracy: {acc:.4f} | Balanced Acc: {bal:.4f}")
print(f"Macro F1: {f1m:.4f} | Weighted F1: {f1w:.4f}")
print(f"Precision (weighted): {precw:.4f} | Recall (weighted): {recw:.4f}")
print("Confusion matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true_all, y_pred_all, target_names=['Cover','Stego'], zero_division=0))

try:
    if len(np.unique(y_true_all)) == 2:
        fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.4f}")
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (test)")
        plt.legend()
        plt.show()
    else:
        print("ROC/AUC: only one class present in test labels, skipping ROC plot.")
except Exception as e:
    print("ROC/AUC failed:", e)

os.makedirs("./checkpoints", exist_ok=True)
torch.save({'state_dict':model.state_dict(), 'threshold':best_threshold}, os.path.join("./checkpoints", "srnet_final.pth"))
print("Saved final model to:", os.path.join("./checkpoints", "srnet_final.pth"))

