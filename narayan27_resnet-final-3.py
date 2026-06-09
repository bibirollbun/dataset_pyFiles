# ==================== CELL 1: IMPORTS & SETUP ====================
import os, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# misc
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Define circular_crop early so worker processes always have it
def circular_crop(img):
    """Crop to the largest bright contour (retina). Input: BGR image from cv2."""
    if img is None:
        return None
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = img[y:y+h, x:x+w]
    return cropped

# small helper for display/testing
def preprocess_for_display(path, img_size=512):
    img = cv2.imread(path)
    if img is None:
        return np.zeros((img_size, img_size, 3), dtype=np.float32)
    c = circular_crop(img)
    if c is None:
        return np.zeros((img_size, img_size, 3), dtype=np.float32)
    img = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    return img


# ==================== CELL 2: PATHS & DATAFRAME ====================
DATA_ROOT = Path('/kaggle/input/aptos2019-blindness-detection')  # change if needed
assert DATA_ROOT.exists(), f"Dataset not found at {DATA_ROOT}"

TRAIN_CSV = DATA_ROOT / 'train.csv'
TRAIN_DIR = DATA_ROOT / 'train_images'

train_df = pd.read_csv(TRAIN_CSV)
train_df['path'] = train_df['id_code'].map(lambda x: str(TRAIN_DIR / f"{x}.png"))
train_df['diagnosis'] = train_df['diagnosis'].astype(int)
train_df.head()


# ==================== CELL 3: QUICK EDA ====================
print("Class counts:\n", train_df['diagnosis'].value_counts())
plt.figure(figsize=(6,4))
train_df['diagnosis'].value_counts().sort_index().plot(kind='bar')
plt.title('Class distribution')
plt.show()


# ==================== CELL 4: TRANSFORMS (PRE-RESIZE) ====================
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 512  # change to 384 if OOM

# Because we pre-resize in Dataset, use RandomCrop not RandomResizedCrop
train_transforms = A.Compose([
    A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.06, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.OneOf([A.GaussNoise(), A.MultiplicativeNoise()], p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

valid_transforms = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

print("Transforms ready. IMG_SIZE =", IMG_SIZE)


# ==================== CELL 5: ROBUST DATASET (PRE-RESIZE) ====================
class APTOSDataset(Dataset):
    def __init__(self, df, transforms=None, img_size=IMG_SIZE, debug_failures=0):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.img_size = int(img_size)
        self._fail_count = 0
        self._debug_limit = debug_failures

    def __len__(self):
        return len(self.df)

    def _safe_resize_and_to_tensor(self, img):
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img_resized = img_resized.astype(np.float32) / 255.0
        mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        std  = np.array((0.229, 0.224, 0.225), dtype=np.float32)
        img_resized = (img_resized - mean) / std
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        return img_tensor

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        path = row['path']
        label = int(row['diagnosis'])

        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = circular_crop(img)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pre-resize to fixed size BEFORE passing to Albumentations
        try:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        except Exception:
            return self._safe_resize_and_to_tensor(img), label

        if self.transforms:
            try:
                sample = self.transforms(image=img)
                img_out = sample['image']
                if isinstance(img_out, np.ndarray):
                    img_tensor = torch.from_numpy(img_out.astype(np.float32)).permute(2,0,1)
                elif torch.is_tensor(img_out):
                    img_tensor = img_out
                else:
                    img_tensor = self._safe_resize_and_to_tensor(img)
                if img_tensor.dim() != 3 or img_tensor.shape[1] != self.img_size or img_tensor.shape[2] != self.img_size:
                    img_tensor = self._safe_resize_and_to_tensor(img)
            except Exception:
                img_tensor = self._safe_resize_and_to_tensor(img)
        else:
            img_tensor = self._safe_resize_and_to_tensor(img)

        img_tensor = img_tensor.float()
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0).repeat(3,1,1)
        if img_tensor.shape[0] != 3:
            if img_tensor.shape[-1] == 3 and img_tensor.dim() == 3:
                img_tensor = img_tensor.permute(2,0,1)
            else:
                img_tensor = self._safe_resize_and_to_tensor(img)

        return img_tensor, label

print("Dataset class ready.")


# ==================== CELL 6: SPLIT & DATALOADERS ====================
SEED = 42
BATCH_SIZE = 8  # reduce to 4 if OOM

train_df_, test_df = train_test_split(train_df, test_size=0.10, stratify=train_df['diagnosis'], random_state=SEED)
train_df, val_df = train_test_split(train_df_, test_size=0.10, stratify=train_df_['diagnosis'], random_state=SEED)

print(f"Sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

# debug_failures=0 to silence fallback prints
train_ds = APTOSDataset(train_df, transforms=train_transforms, img_size=IMG_SIZE, debug_failures=0)
val_ds   = APTOSDataset(val_df,   transforms=valid_transforms, img_size=IMG_SIZE, debug_failures=0)
test_ds  = APTOSDataset(test_df,  transforms=valid_transforms, img_size=IMG_SIZE, debug_failures=0)

# num_workers=0 avoids worker-process import issues during debugging
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# quick sanity
imgs, labels = next(iter(train_loader))
print("Sample batch shapes:", imgs.shape, labels.shape)


# ==================== CELL 7: FOCAL LOSS ====================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', logits=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = ((1 - pt) ** self.gamma) * ce_loss
            if self.alpha is not None:
                at = self.alpha.gather(0, targets)
                loss = at * loss
        else:
            pt = inputs.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = -((1 - pt) ** self.gamma) * torch.log(pt + 1e-12)
            if self.alpha is not None:
                at = self.alpha.gather(0, targets)
                loss = at * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ==================== CELL 8: MODEL (timm ResNet50) ====================
# If timm not installed on Kaggle/Colab, run: !pip install timm
import timm

def get_resnet50(num_classes=5, pretrained=True):
    # timm's resnet50 model name is 'resnet50'
    model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
    return model

model = get_resnet50(num_classes=5, pretrained=True)
model = model.to(DEVICE)
print("ResNet50 model loaded.")



# ==================== CELL 9: METRICS & EVAL HELPERS ====================
scaler = torch.amp.GradScaler(device="cuda") if torch.cuda.is_available() else torch.amp.GradScaler()

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    all_preds = []
    all_targets = []
    running_loss = 0.0
    if criterion is None:
        criterion = FocalLoss(gamma=2.0, logits=True)
    device_type = "cuda" if DEVICE.type == "cuda" else "cpu"
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type=device_type):
            logits = model(imgs)
            loss = criterion(logits, targets)
        preds = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
        running_loss += loss.item() * imgs.size(0)
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    q = qwk(all_targets, all_preds)
    return avg_loss, acc, q, np.array(all_targets), np.array(all_preds)


# ==================== CELL 10: TRAIN LOOP (MIXED PRECISION + BEST CHECKPOINT) - filenames adjusted for ResNet50
EPOCHS = 12
criterion = FocalLoss(gamma=2.0, logits=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'val_qwk':[]}
best_qwk = -1.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    device_type = "cuda" if DEVICE.type == "cuda" else "cpu"
    for imgs, targets in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_type):
            logits = model(imgs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.softmax(dim=1).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{correct/total:.4f}'})

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    val_loss, val_acc, val_qwk, _, _ = evaluate(model, val_loader, criterion=criterion)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_qwk'].append(val_qwk)

    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_qwk={val_qwk:.4f}")

    if val_qwk > best_qwk:
        best_qwk = val_qwk
        torch.save(model.state_dict(), "resnet50_best.pth")
        print(f"Saved best model with val_qwk={best_qwk:.4f}")

    scheduler.step()

torch.save(model.state_dict(), "resnet50_last.pth")
print("Training finished.")



# ==================== CELL 11: PLOT TRAINING METRICS ====================
import matplotlib.pyplot as plt

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')

plt.subplot(1,2,2)
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.plot(history['val_qwk'], label='val_qwk')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.title('Training Accuracy / Val Accuracy / Val QWK')
plt.show()


# ==================== CELL 12: TEST EVALUATION ====================
from sklearn.metrics import classification_report, confusion_matrix

ckpt_path = "resnet50_best.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Evaluate on test set
test_loss, test_acc, test_qwk, y_true, y_pred = evaluate(
    model, test_loader, criterion=FocalLoss(gamma=2.0, logits=True)
)

print(f"✅ Loaded checkpoint: {ckpt_path}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test QWK: {test_qwk:.4f}\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i,j]), ha='center', va='center')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()




