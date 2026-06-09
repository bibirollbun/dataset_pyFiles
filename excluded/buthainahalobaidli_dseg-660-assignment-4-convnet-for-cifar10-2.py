!ls -lah /kaggle/input/cifar-10


# A) imports, seed, device
import os, math, random, glob
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18
from torch.cuda.amp import autocast, GradScaler

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)



# B) extract Kaggle train.7z and build train/val loaders
!apt -yq update >/dev/null
!apt -yq install libarchive-dev >/dev/null
!pip -q install libarchive >/dev/null

import libarchive.public

# constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
class_to_idx = {c:i for i,c in enumerate(CLASSES)}

# extract train set to /kaggle/working/train (only once)
train_out = "/kaggle/working/train"
os.makedirs(train_out, exist_ok=True)
if len(glob.glob(train_out + "/*")) < 1000:
    for _ in libarchive.public.file_pour('/kaggle/input/cifar-10/train.7z'):
        pass

# labels
labels_df = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')  # id,label
labels_df['path'] = labels_df['id'].astype(str).apply(lambda i: f"{train_out}/{i}.png")
print("Rows in labels:", len(labels_df))

# transforms
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02,0.15), ratio=(0.3,3.3), value='random')
])
test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# dataset
class KaggleTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert("RGB")
        y = class_to_idx[row['label']]
        return self.tfm(img), y

full_ds = KaggleTrainDataset(labels_df, train_tfms)

# split train/val
val_size = 5000
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                generator=torch.Generator().manual_seed(SEED))

BS = 128   # if OOM later, set 64 (also change in H)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)

print("train/val sizes:", len(train_ds), len(val_ds))



# C) ResNet-18 adapted for CIFAR-10 (random init)
def resnet18_cifar10(num_classes=10):
    m = resnet18(weights=None)   # IMPORTANT: no pre-trained weights
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

model = resnet18_cifar10().to(DEVICE)
print("Model ready (weights=None).")



# D) training setup
EPOCHS = 40          # quick first run; raise to 80+ later for higher score
BASE_LR = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9,
                            weight_decay=5e-4, nesterov=True)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler()

warmup_epochs = 5
total_steps = EPOCHS * len(train_loader)

def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))



# E) train loop + validation
def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum/total, correct/total

best_val = 0.0
global_step = 0

for epoch in range(1, EPOCHS+1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        lr = cosine_lr(global_step, total_steps, warmup_epochs*len(train_loader), BASE_LR)
        for g in optimizer.param_groups: g['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        global_step += 1

    val_loss, val_acc = evaluate(val_loader)
    print(f"Epoch {epoch:03d} | val_acc={val_acc:.4f} | val_loss={val_loss:.4f} | lr={lr:.5f}")
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_model.pth")

print("Best val acc:", best_val)



# G) extract competition test set to /kaggle/working/test
import libarchive.public
out_dir = "/kaggle/working/test"
os.makedirs(out_dir, exist_ok=True)
if len(glob.glob(out_dir + "/*")) < 1000:
    for _ in libarchive.public.file_pour('/kaggle/input/cifar-10/test.7z'):
        pass
print("Files in /kaggle/working/test:", len(glob.glob(out_dir + "/*")))



# H (fixed TTA): create a clean submission.csv
test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

class ImageFolderFromDir(torch.utils.data.Dataset):
    def __init__(self, folder, tfm):
        self.paths = sorted(glob.glob(os.path.join(folder, "*.png")), key=lambda p: int(Path(p).stem))
        self.tfm = tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), int(Path(p).stem)

kaggle_test_ds = ImageFolderFromDir("/kaggle/working/test", test_tfms)
kaggle_test_loader = DataLoader(kaggle_test_ds, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)

idx_to_label = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

ids, labels = [], []
with torch.no_grad():
    for x, img_ids in kaggle_test_loader:
        x = x.to(DEVICE, non_blocking=True)
        with autocast():
            logits1 = model(x)
            logits2 = model(torch.flip(x, dims=[3]))  # horizontal flip
            logits  = (logits1 + logits2) / 2
        pred = logits.argmax(1).cpu().numpy()

        # IMPORTANT: convert batch tensor -> list of Python ints
        ids.extend([int(i) for i in img_ids])   # <-- this fixes the "tensor(1)" issue
        labels.extend([idx_to_label[i] for i in pred])

import pandas as pd
sub = pd.DataFrame({"id": ids, "label": labels}).sort_values("id")
sub.to_csv("submission.csv", index=False)
print(sub.shape)  # should be (300000, 2)
sub.head()



import pandas as pd
s = pd.read_csv("submission.csv")
print(s.shape, s.head(), s.dtypes)



# F-exact: evaluate on the official CIFAR-10 test set (requires Internet = On)
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

test_official = CIFAR10(root="/kaggle/working/data", train=False, download=True, transform=test_tfms)
test_loader_official = DataLoader(test_official, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)

model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
tl, ta = evaluate(test_loader_official)
print("Official CIFAR-10 test accuracy (≈ Kaggle score):", ta)


