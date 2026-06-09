# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ====================================================
# 0. ä¾�å­˜ãƒ©ã‚¤ãƒ–ãƒ©ãƒª
# ====================================================
!pip install -q timm torchmetrics --upgrade

import os, zipfile, random, pandas as pd, numpy as np, time, torch, timm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler

# ====================================================
# 1. ãƒ‡ãƒ�ã‚¤ã‚¹è¨­å®š & åŸºæœ¬ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿
# ====================================================
assert torch.cuda.is_available(), "Settings â†’ Accelerator ã‚’ GPU ã�«ã�—ã�¦å†�èµ·å‹•ã�—ã�¦ã��ã� ã�•ã�„"
n_gpus   = torch.cuda.device_count()
device   = torch.device("cuda")
print(f"ğŸ’»  Using {n_gpus}Ã—{torch.cuda.get_device_name(0)}")

IMG_SIZE = 160
BATCH    = 32 * n_gpus          # GPU æ�šæ•°ã�«å¿œã�˜ã�¦è‡ªå‹•ã�§æ‹¡å¤§
EPOCHS   = 5
LR       = 3e-4

# é«˜é€ŸåŒ–ãƒ•ãƒ©ã‚°
torch.backends.cudnn.benchmark = True

# ====================================================
# 2. ãƒ‡ãƒ¼ã‚¿è§£å‡�
# ====================================================
DATA_DIR = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
for zipf in ["train.zip", "test.zip"]:
    with zipfile.ZipFile(f"{DATA_DIR}/{zipf}") as z:
        z.extractall("/kaggle/working")
TRAIN_DIR, TEST_DIR = "/kaggle/working/train", "/kaggle/working/test"

# ====================================================
# 3. DataFrame & Dataset
# ====================================================
train_files = os.listdir(TRAIN_DIR)
labels = [1 if f.startswith("dog") else 0 for f in train_files]
df = pd.DataFrame({"filepath": [f"{TRAIN_DIR}/{f}" for f in train_files],
                   "label": labels})
train_df, val_df = train_test_split(df, test_size=0.1,
                                    stratify=df.label, random_state=42)

mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
val_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE+32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

class CatDogDS(Dataset):
    def __init__(self, df, tfms, infer=False):
        self.df, self.tfms, self.infer = df.reset_index(drop=True), tfms, infer
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        p = self.df.filepath[idx]
        img = Image.open(p).convert("RGB"); img = self.tfms(img)
        if self.infer: return img, os.path.basename(p).split(".")[0]
        return img, torch.tensor(self.df.label[idx], dtype=torch.float32)

train_dl = DataLoader(CatDogDS(train_df, train_tfms), batch_size=BATCH,
                      shuffle=True, num_workers=2, pin_memory=True)
val_dl   = DataLoader(CatDogDS(val_df,   val_tfms),   batch_size=BATCH*2,
                      shuffle=False, num_workers=2, pin_memory=True)

# ====================================================
# 4. ãƒ¢ãƒ‡ãƒ«ãƒ»æ��å¤±ãƒ»OPT
# ====================================================
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
if n_gpus > 1:
    model = torch.nn.DataParallel(model)   # è‡ªå‹•ã�§è¤‡æ•° GPU ã�¸
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = GradScaler(device='cuda')

# ====================================================
# 5. å­¦ç¿’ãƒ«ãƒ¼ãƒ—
# ====================================================
best_loss = 1e9
for epoch in range(1, EPOCHS+1):
    # ----- train -----
    model.train(); t0 = time.time()
    for xb, yb in train_dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            loss = criterion(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
    scheduler.step()

    # ----- val -----
    model.eval(); val_loss, n = 0, 0
    with torch.no_grad(), autocast(device_type='cuda'):
        for xb, yb in val_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).unsqueeze(1)
            loss = criterion(model(xb), yb)
            val_loss += loss.item() * xb.size(0); n += xb.size(0)
    val_loss /= n
    print(f"[{epoch}/{EPOCHS}]  val_loss={val_loss:.4f}  ({time.time()-t0:.1f}s)")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best.pt")

# ====================================================
# 6. æ�¨è«– & æ��å‡º
# ====================================================
test_files = sorted(os.listdir(TEST_DIR), key=lambda x:int(x.split('.')[0]))
test_df = pd.DataFrame({"filepath":[f"{TEST_DIR}/{f}" for f in test_files]})
test_dl = DataLoader(CatDogDS(test_df, val_tfms, infer=True),
                     batch_size=BATCH*2, shuffle=False,
                     num_workers=4, pin_memory=True)

model.load_state_dict(torch.load("best.pt")); model.eval()
ids, probs = [], []
with torch.no_grad(), autocast(device_type='cuda'):
    for xb, img_ids in test_dl:
        xb = xb.to(device, non_blocking=True)
        preds = torch.sigmoid(model(xb)).squeeze().cpu().numpy()
        ids.extend(img_ids); probs.extend(preds)

sub = pd.DataFrame({"id": ids, "label": probs})
sub.to_csv("submission.csv", index=False)
print("âœ…  submission.csv saved â€” ready to upload!")

