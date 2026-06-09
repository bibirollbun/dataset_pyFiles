# ============================================================
# MobileFaceNet — Face Verification (from scratch, PyTorch)
# ============================================================

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# -----------------------------
# 1. CONFIG
# -----------------------------
DATA_ROOT = "/kaggle/input/11-785-fall-20-homework-2-part-2"
CLS_ROOT  = os.path.join(DATA_ROOT, "classification_data")
TRAIN_DIR = os.path.join(CLS_ROOT, "train_data")
VAL_DIR   = os.path.join(CLS_ROOT, "val_data")
TEST_DIR  = os.path.join(CLS_ROOT, "test_data")

IMG_SIZE   = 80   
BATCH_SIZE = 64
EMBED_DIM  = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -----------------------------
# 2. DATASET & DATALOADER
# -----------------------------
torch.backends.cudnn.benchmark = True  # ✅ optimize conv performance
from torchvision.transforms import InterpolationMode

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=val_transform)

# ✅ Optimized DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True, num_workers=2,
                          pin_memory=True, prefetch_factor=4)
val_loader   = DataLoader(val_dataset, batch_size=64,
                          shuffle=False, num_workers=2,
                          pin_memory=True, prefetch_factor=4)
test_loader  = DataLoader(test_dataset, batch_size=64,
                          shuffle=False, num_workers=2,
                          pin_memory=True, prefetch_factor=4)


num_classes = len(train_dataset.classes)
print("Num classes:", num_classes)


# -----------------------------
# 3. MODEL — MobileFaceNet architecture
# -----------------------------

# --- Helper blocks ---
def conv_bn(inp, oup, stride):
    """Standard 3x3 Conv + BN + PReLU"""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(oup)
    )

def conv_dw(inp, oup, stride):
    """Depthwise + Pointwise conv (main efficiency trick)"""
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.PReLU(inp),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(oup)
    )

class Bottleneck(nn.Module):
    """Inverted residual block (from MobileNetV2)"""
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise expand
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.PReLU(hidden_dim)
            ])
        # 3x3 depthwise conv + 1x1 projection
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)
        self.prelu = nn.PReLU(oup)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.prelu(self.conv(x))
        else:
            return self.prelu(self.conv(x))

class MobileFaceNet(nn.Module):
    """Main CNN model — extracts 128-D normalized embeddings"""
    def __init__(self, embedding_size=128, num_classes=None):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(3, 64, 2),
            conv_dw(64, 64, 1),
            Bottleneck(64, 64, 2, 2),
            Bottleneck(64, 64, 1, 2),
            Bottleneck(64, 128, 2, 4),
            Bottleneck(128, 128, 1, 2),
            Bottleneck(128, 128, 1, 4),
            Bottleneck(128, 128, 1, 2),
            conv_bn(128, 512, 1)
        )
        # Global depthwise conv
        # Use adaptive pooling instead of fixed 7x7 kernel for flexible input (e.g., 64x64)
        self.depthwise = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

        # Optional classification head
        self.classifier = nn.Linear(embedding_size, num_classes) if num_classes else None

    def forward(self, x, return_embedding=False):
        """Forward pass:
        - Return logits if training (classification)
        - Return L2-normalized embedding if inference
        """
        x = self.features(x)
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)          # flatten
        emb = self.linear(x)               # project to 128D
        emb = self.bn(emb)
        emb = F.normalize(emb, p=2, dim=1) # L2 normalization

        if return_embedding or self.classifier is None:
            return emb                     # inference mode
        else:
            logits = self.classifier(emb)  # training mode
            return logits


# -----------------------------
# 4. TRAINING LOOP
# -----------------------------
from tqdm.notebook import tqdm

model = MobileFaceNet(embedding_size=EMBED_DIM, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)


# -----------------------------
# 5. TRAINING STAGE (with early stopping)

EPOCHS = 30
best_val_acc = 0.0
best_val_loss = float('inf')
patience = 5           # stop if no improvement after 5 epochs
wait = 0
save_path = "mobilefacenet_best.pth"

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # scales the loss for stability

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        # ----------------------------
        # ✅ Mixed-Precision Training
        # ----------------------------
        with autocast():  # automatically choose float16/float32 per op
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        # backward with scaled loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # metrics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        pbar.set_postfix(loss=loss.item())

    train_loss = running_loss / total
    train_acc = correct / total

    # ---- VALIDATION ----
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * imgs.size(0)
            val_correct += (preds == labels).sum().item()
            val_total += imgs.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"[Epoch {epoch:02d}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} "
          f"|| Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    # ---- EARLY STOPPING LOGIC ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"✅ New best model saved at epoch {epoch} (Val Acc: {val_acc:.4f})")
        wait = 0
    else:
        wait += 1
        print(f"No improvement. Patience: {wait}/{patience}")

        if wait >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch}")
            break

print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")



# ===========================
# SIAMESE NETWORK - FACE VERIFICATION
# ===========================
import os, random, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.cuda.amp import GradScaler, autocast

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# ===========================
# CONFIG
# ===========================
CFG = {
    'data_root': '/kaggle/input/11-785-fall-20-homework-2-part-2/classification_data/train_data',
    'ver_pairs': '/kaggle/input/11-785-fall-20-homework-2-part-2/verification_pairs_val.txt',
    'ver_root': '/kaggle/input/11-785-fall-20-homework-2-part-2/',
    'img_size': 80,
    'emb_dim': 128,
    'lr': 3e-4,
    'batch_size': 128,
    'epochs': 10,
    'num_workers': 2,
    'save_path': '/kaggle/working/siamese_model.pth'
}


# ===========================
# MODEL - MOBILEFACENET
# ===========================
def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU()
    )

def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.PReLU(),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU()
    )

class MobileFaceNet(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.conv1 = conv_bn(3, 64, 2)
        self.conv2 = conv_dw(64, 64, 1)
        self.conv3 = conv_dw(64, 128, 2)
        self.conv4 = conv_dw(128, 128, 1)
        self.conv5 = conv_dw(128, 256, 2)
        self.conv6 = conv_dw(256, 256, 1)
        self.conv7 = conv_dw(256, 512, 2)
        self.conv8 = nn.Sequential(
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.conv9 = conv_dw(512, 512, 1)
        self.conv10 = conv_dw(512, 1024, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.avg(x).view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

# ===========================
# DATASET - VALIDATION
# ===========================
class VerificationDataset(Dataset):
    def __init__(self, img_paths, root):
        self.img_paths = img_paths
        self.root = root
        self.tfms = transforms.Compose([
            transforms.Resize((CFG['img_size'], CFG['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.img_paths[idx])
        try:
            img = Image.open(path).convert('RGB')
            return self.tfms(img)
        except:
            return torch.zeros(3, CFG['img_size'], CFG['img_size'])



# ===========================
# SIAMESE MODEL
# ===========================
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.backbone = MobileFaceNet(embedding_dim)
    
    def forward(self, img1, img2):
        emb1 = self.backbone(img1)
        emb2 = self.backbone(img2)
        return emb1, emb2
    
    def get_embedding(self, img):
        return self.backbone(img)

# ===========================
# CONTRASTIVE LOSS
# ===========================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, emb1, emb2, target):
        # Euclidean distance
        dist = F.pairwise_distance(emb1, emb2)
        
        # Contrastive loss
        loss_pos = target * dist.pow(2)
        loss_neg = (1 - target) * F.relu(self.margin - dist).pow(2)
        loss = (loss_pos + loss_neg).mean()
        
        return loss


# ===========================
# TRAINING
# ===========================
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    running_loss = 0.
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for img1, img2, target in pbar:
        img1 = img1.to(DEVICE)
        img2 = img2.to(DEVICE)
        target = target.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast():
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return running_loss / len(loader)

# ===========================
# VALIDATION
# ===========================
@torch.no_grad()
def validate(model, pairs_file, img_root):
    model.eval()
    
    with open(pairs_file) as f:
        pairs = [line.strip().split() for line in f if line.strip()]
    
    unique_paths = sorted(set(img1 for img1, _, _ in pairs) | 
                          set(img2 for _, img2, _ in pairs))
    path_to_idx = {path: i for i, path in enumerate(unique_paths)}
    
    val_ds = VerificationDataset(unique_paths, img_root)
    val_loader = DataLoader(val_ds, batch_size=128, num_workers=CFG['num_workers'], shuffle=False)
    
    all_embeddings = []
    for imgs in val_loader:
        imgs = imgs.to(DEVICE)
        with autocast():
            embs = model.get_embedding(imgs)
        all_embeddings.append(embs.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    similarities, labels = [], []
    for img1, img2, label in pairs:
        idx1, idx2 = path_to_idx[img1], path_to_idx[img2]
        sim = F.cosine_similarity(all_embeddings[idx1].unsqueeze(0), 
                                  all_embeddings[idx2].unsqueeze(0)).item()
        similarities.append(sim)
        labels.append(int(label))
    
    auc = roc_auc_score(labels, similarities)
    fpr, tpr, _ = roc_curve(labels, similarities)
    eer = fpr[np.nanargmin(np.abs(tpr - (1 - fpr)))]
    
    return auc, eer


# ===========================
# MAIN
# ===========================
if __name__ == '__main__':
    train_ds = SiamesePairDataset(CFG['data_root'])
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], 
                              num_workers=CFG['num_workers'], pin_memory=True)
    
    model = SiameseNetwork(CFG['emb_dim']).to(DEVICE)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
    scaler = GradScaler()
    
    best_auc = 0
    for epoch in range(1, CFG['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
        auc, eer = validate(model, CFG['ver_pairs'], CFG['ver_root'])
        
        print(f'Loss: {train_loss:.4f} | AUC: {auc:.4f} | EER: {eer:.2%}')
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), CFG['save_path'])
            print(f'Best model saved (AUC: {auc:.4f})')
    
    print(f'\nTraining Complete - Best AUC: {best_auc:.4f}')




