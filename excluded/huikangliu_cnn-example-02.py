import os, zipfile, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


COMP = "cifar-10-cn-n-02"  # ä½ çš„æ•°æ�®é›†è·¯å¾„
INPUT_DIR = Path(f"/kaggle/input/{COMP}")
TRAIN_DIR = INPUT_DIR / "train" / "train"
TEST_DIR  = INPUT_DIR / "test" / "test"

print("âœ… æ£€æŸ¥è·¯å¾„ï¼š")
print("TRAIN_DIR:", TRAIN_DIR.exists(), TRAIN_DIR)
print("TEST_DIR:", TEST_DIR.exists(), TEST_DIR)


# ==== è¯»å�–æ ‡ç­¾æ–‡ä»¶ ====
train_csv = pd.read_csv(INPUT_DIR/"train_labels.csv")  # Id,Label
train_csv["path"] = train_csv["Id"].apply(lambda x: TRAIN_DIR / f"{int(x):05d}.png")

# ==== ç”Ÿæˆ�æµ‹è¯•é›† DataFrame ====
test_ids = sorted([p.stem for p in TEST_DIR.glob("*.png")])
test_df = pd.DataFrame({"Id": test_ids})
test_df["path"] = test_df["Id"].apply(lambda x: TEST_DIR / f"{int(x):05d}.png")


# ==== æ•°æ�®å¢�å¼º ====
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])


# ==== æ•°æ�®é›†ç±» ====
class ImageDataset(Dataset):
    def __init__(self, df, transform=None, has_label=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.has_label = has_label
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.has_label:
            return img, int(row["Label"])  # 0~9
        else:
            return img, row["Id"]


# ===================== é…�ç½®ï¼ˆå­¦ç”Ÿä¸»è¦�æ”¹è¿™é‡Œï¼‰ =====================
SEED = 2025             # éš�æœºç§�å­�ï¼Œä¿�è¯�ç»“æ�œå�¯å¤�ç�°ï¼ˆå½±å“�æ•°æ�®åˆ’åˆ†ã€�åˆ�å§‹åŒ–ã€�shuffle ç­‰ï¼‰
EPOCHS = 10             # è®­ç»ƒè½®æ•°ï¼ˆæ¨¡å�‹å®Œæ•´çœ‹ä¸€é��æ•°æ�®çš„æ¬¡æ•°ï¼Œè¶Šå¤šå�¯èƒ½ç²¾åº¦è¶Šé«˜ä½†è®­ç»ƒæ›´ä¹…ï¼‰
BATCH_SIZE = 128        # æ¯�æ¬¡è®­ç»ƒè¯»å�–çš„æ ·æœ¬æ•°é‡�ï¼ˆè¶Šå¤§è®­ç»ƒè¶Šå¿«ä½†æ˜¾å­˜å� ç”¨æ›´é«˜ï¼‰
LR = 1e-3               # å­¦ä¹ ç�‡ï¼ˆæ�§åˆ¶æ¯�æ¬¡å�‚æ•°æ›´æ–°çš„å¹…åº¦ï¼Œå¤ªå¤§ä¼šéœ‡è�¡ï¼Œå¤ªå°�æ”¶æ•›æ…¢ï¼‰
WEIGHT_DECAY = 1e-4     # æ�ƒé‡�è¡°å‡�ï¼ˆL2æ­£åˆ™åŒ–ç³»æ•°ï¼Œç”¨äº�é˜²æ­¢è¿‡æ‹Ÿå�ˆï¼Œä½¿æ�ƒé‡�æ›´å¹³æ»‘ï¼‰
# ================================================================


# ==== åˆ’åˆ†è®­ç»ƒ/éªŒè¯�é›† ====
tr_df, val_df = train_test_split(train_csv, test_size=0.1, random_state=SEED, stratify=train_csv["Label"])
train_ds = ImageDataset(tr_df, transform=transform_train, has_label=True)
val_ds   = ImageDataset(val_df, transform=transform_test, has_label=True)
test_ds  = ImageDataset(test_df, transform=transform_test, has_label=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ======= CNN æ¨¡å�‹ =======
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            # nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),  # ç¬¬ä¸‰å±‚å�·ç§¯å±‚
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128), nn.ReLU(), nn.Dropout(0.2),   # å¦‚æ�œæ‰“å¼€ç¬¬ä¸‰å±‚å�·ç§¯å±‚ï¼Œå°±æ³¨é‡Šè¿™ä¸€è¡Œï¼Œæ‰“å¼€ä¸‹ä¸€è¡Œ
            # nn.Linear(128*4*4, 128), nn.ReLU(), nn.Dropout(0.2),  
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# ======= è®­ç»ƒä¸�éªŒè¯� =======
def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


best_acc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    acc = evaluate()
    if acc > best_acc: best_acc = acc
    print(f"Epoch {epoch}/{EPOCHS} - val_acc={acc:.4f} (best={best_acc:.4f})")


# ======= é¢„æµ‹ä¸�ä¿�å­˜æ��äº¤æ–‡ä»¶ =======
model.eval()
pred_ids, pred_labels = [], []
with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(1).cpu().numpy()
        pred_labels.extend(preds.tolist())
        pred_ids.extend(ids)

sub = pd.DataFrame({"Id": pred_ids, "Label": pred_labels}).sort_values("Id").reset_index(drop=True)
sub.to_csv("submission.csv", index=False)
print("âœ… å·²ä¿�å­˜: submission.csv")





