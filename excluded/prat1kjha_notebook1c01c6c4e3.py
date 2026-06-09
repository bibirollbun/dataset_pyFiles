# ============================ CELL 1: IMPORTS + CONFIG ============================
import os, random, warnings
import numpy as np, pandas as pd
import cv2, torch, timm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt, seaborn as sns

warnings.filterwarnings("ignore")

# ---- SEED + DEVICE ----
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("âœ… Device:", DEVICE)

# ---- GLOBAL PARAMS ----
IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 12
NUM_CLASSES = 5
DATA_DIR = "/kaggle/input/aptos2019-blindness-detection"

# ============================ CELL 2: LOAD CSV + SPLIT ============================
df = pd.read_csv(f"{DATA_DIR}/train.csv")
df["id_code"] = df["id_code"].astype(str)
df["image_path"] = df["id_code"].apply(lambda x: f"{DATA_DIR}/train_images/{x}.png")

train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["diagnosis"], random_state=SEED)
val_df, test_df = train_test_split(val_df, test_size=0.5, stratify=val_df["diagnosis"], random_state=SEED)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ============================ CELL 3: IMAGE PREPROCESSING ============================
def circular_crop(img):
    h, w, _ = img.shape
    x, y, r = w//2, h//2, min(w//2, h//2)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (x, y), r, 1, -1)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = img[y-r:y+r, x-r:x+r]
    return img

# ============================ CELL 4: TRANSFORMS ============================
train_transforms = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.6),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.06, rotate_limit=15, p=0.4),
    A.OneOf([A.GaussNoise(), A.ISONoise()], p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

valid_transforms = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

# ============================ CELL 5: DATASET CLASS ============================
class APTOSDataset(Dataset):
    def __init__(self, df, transforms=None, img_size=IMG_SIZE):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = circular_crop(img)
        img = cv2.resize(img, (self.img_size, self.img_size))
        if self.transforms: img = self.transforms(image=img)["image"]
        label = int(row.diagnosis)
        return img, label

# ============================ CELL 6: DATALOADERS ============================
train_ds = APTOSDataset(train_df, transforms=train_transforms)
val_ds   = APTOSDataset(val_df,   transforms=valid_transforms)
test_ds  = APTOSDataset(test_df,  transforms=valid_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ============================ CELL 7: MODEL (DENSENET121) ============================
model = timm.create_model("densenet121", pretrained=True)
in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)
model = model.to(DEVICE)
print("âœ… DenseNet121 loaded.")

# ============================ CELL 8: OPTIMIZER + LOSS ============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler()

# ============================ CELL 9: EVALUATION FUNCTION ============================
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_true, all_pred = [], []
    device_type = "cuda" if DEVICE.type == "cuda" else "cpu"
    with torch.amp.autocast(device_type=device_type):
        for imgs, targets in loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.softmax(1).argmax(1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)
            all_true.extend(targets.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
    acc = correct / total
    qwk = cohen_kappa_score(all_true, all_pred, weights='quadratic')
    return total_loss/len(loader.dataset), acc, qwk, all_true, all_pred

# ============================ CELL 10: TRAIN LOOP ============================
history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'val_qwk':[]}
best_qwk = -999.0

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    device_type = "cuda" if DEVICE.type == "cuda" else "cpu"

    for imgs, targets in pbar:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_type):
            logits = model(imgs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.softmax(1).argmax(1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{correct/total:.4f}'})

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total
    val_loss, val_acc, val_qwk, _, _ = evaluate(model, val_loader, criterion)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_qwk'].append(val_qwk)

    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_qwk={val_qwk:.4f}")

    if val_qwk > best_qwk:
        best_qwk = val_qwk
        torch.save(model.state_dict(), "densenet121_best.pth")
        print(f"âœ… Saved best model (val_qwk={best_qwk:.4f})")
    scheduler.step()

torch.save(model.state_dict(), "densenet121_last.pth")
print("Training finished.")

# ============================ CELL 11: PLOT METRICS ============================
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='train_loss'); plt.plot(history['val_loss'], label='val_loss')
plt.legend(); plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.plot(history['val_qwk'], label='val_qwk')
plt.legend(); plt.title('Accuracy / QWK')
plt.show()

# ============================ CELL 12: TEST EVALUATION ============================
ckpt_path = "densenet121_best.pth"
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.to(DEVICE); model.eval()
print(f"âœ… Loaded: {ckpt_path}")

test_loss, test_acc, test_qwk, y_true, y_pred = evaluate(model, test_loader, criterion)
print(f"\nðŸ“Š Test Loss: {test_loss:.4f}\nTest Acc: {test_acc:.4f}\nTest QWK: {test_qwk:.4f}")

print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("DenseNet121 â€“ Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.show()


