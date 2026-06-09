# Basics
import os, glob, random, csv, shutil
from pathlib import Path
import numpy as np
from PIL import Image

# Torch
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# Reproducibility
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# CIFAR-10 normalization
C10_MEAN = (0.4914, 0.4822, 0.4465)
C10_STD  = (0.2470, 0.2435, 0.2616)

# Class order used for submission
IDX_TO_CLASS = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print("Class order:", IDX_TO_CLASS)



# Transforms
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(C10_MEAN, C10_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])
eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(C10_MEAN, C10_STD),
])

# Load official CIFAR-10 train (50k)
root = "./data"
full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)

# Split 45k/5k
val_size = 5000
train_size = len(full_train) - val_size
g = torch.Generator().manual_seed(seed)
train_set, val_set = random_split(full_train, [train_size, val_size], generator=g)
val_set.dataset.transform = eval_tf  # no augs on val

# Dataloaders
BATCH_SIZE = 256
NUM_WORKERS = 2
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

xb, yb = next(iter(train_loader))
print("Train batch:", xb.shape, yb.shape, "| sizes:", len(train_set), len(val_set))



# scratch residual network for CIFAR-10 

class BasicBlock(nn.Module):
    """Two 3x3 convs + residual skip. Downsample via stride=2 and 1x1 conv if needed."""
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.down = nn.Identity()

    def forward(self, x):
        identity = self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)   # residual connection
        return out

class CifarResNet(nn.Module):
    """
    Stem: 3x3 conv (s=1), BN, ReLU (no maxpool).
    Stages: [L1, L2, L3] BasicBlocks with channel widths [64,128,256].
    Head: GAP -> Linear(10).
    """
    def __init__(self, layers=(3,3,3), widths=(64,128,256), num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_stage(widths[0], widths[0], layers[0], stride=1)  # 32x32
        self.layer2 = self._make_stage(widths[0], widths[1], layers[1], stride=2)  # 16x16
        self.layer3 = self._make_stage(widths[1], widths[2], layers[2], stride=2)  # 8x8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(widths[2], num_classes)

        self.apply(self._init_kaiming)

    def _make_stage(self, in_ch, out_ch, n_blocks, stride):
        blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    @staticmethod
    def _init_kaiming(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x

model = CifarResNet(layers=(3,3,3), widths=(64,128,256), num_classes=10)
model = model.to(device)
print("Custom residual CIFAR model ready. Params:",
      sum(p.numel() for p in model.parameters())/1e6, "M")



EPOCHS = 10
base_lr = 0.2 * (BATCH_SIZE / 256)
optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                            weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Safe EMA (only float tensors)
use_ema = True
ema_decay = 0.999
if use_ema:
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

@torch.no_grad()
def ema_update():
    if not use_ema: return
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            ema[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
        else:
            ema[k] = v.detach().clone()

best_acc, best_path = 0.0, "best_custom_cifar_resnet.pth"



def run_epoch(loader, train=True):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
                ema_update()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

warmup = 5
for epoch in range(EPOCHS):
    if epoch < warmup:
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * (epoch + 1) / warmup

    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader,   train=False)
    scheduler.step()

    if va_acc > best_acc:
        best_acc = va_acc
        torch.save({'model': model.state_dict()}, best_path)

    print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
          f"train {tr_acc:.4f}/{tr_loss:.4f} | "
          f"val {va_acc:.4f}/{va_loss:.4f} | best {best_acc:.4f}")

print("Best val acc:", best_acc)



test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(C10_MEAN, C10_STD)])
test_official = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)
test_loader = DataLoader(test_official, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

# Prefer EMA weights; else best checkpoint
if 'ema' in globals():
    model.load_state_dict(ema, strict=False)
else:
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)

model.eval()
total = correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)

offline_acc = correct / total
print("Official CIFAR-10 test accuracy (offline estimate):", offline_acc)



!pip -q install py7zr
import py7zr

# Find archive
cands = glob.glob("/kaggle/input/**/test.7z", recursive=True)
print("Candidates:", cands)
assert cands, "Attach the 'CIFAR-10 - Object Recognition in Images' competition as input."
test7z = cands[0]
print("Using:", test7z, "| size:", os.path.getsize(test7z))

# Clean and extract
outdir = "test"
if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.makedirs(outdir, exist_ok=True)

with py7zr.SevenZipFile(test7z, mode="r") as z:
    z.extractall(path=outdir)

# Flatten if nested and count
pngs_recursive = glob.glob(os.path.join(outdir, "**", "*.png"), recursive=True)
for p in pngs_recursive:
    dest = os.path.join(outdir, os.path.basename(p))
    if os.path.abspath(p) != os.path.abspath(dest) and not os.path.exists(dest):
        shutil.move(p, dest)
# remove empty dirs
for root, dirs, files in os.walk(outdir, topdown=False):
    if root != outdir and not os.listdir(root):
        os.rmdir(root)

final_count = len(glob.glob(os.path.join(outdir, "*.png")))
print("Final PNG count in ./test:", final_count)  # expect ~300000



# Load EMA (if present) or best checkpoint; eval mode
if 'ema' in globals():
    model.load_state_dict(ema, strict=False)
else:
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
model.eval()

test_tf_final = transforms.Compose([transforms.ToTensor(), transforms.Normalize(C10_MEAN, C10_STD)])

class TestFolder(Dataset):
    def __init__(self, files, tfm):
        self.files = files; self.tfm = tfm
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        fp = self.files[i]
        img = Image.open(fp).convert('RGB')
        return self.tfm(img), int(Path(fp).stem)

test_files = sorted(glob.glob('test/*.png'), key=lambda p: int(Path(p).stem))
test_loader_big = DataLoader(TestFolder(test_files, test_tf_final),
                             batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

rows = []
with torch.no_grad():
    for x, ids in test_loader_big:
        x = x.to(device)
        pred = model(x).argmax(1).cpu().tolist()
        for i, p in zip(ids.tolist(), pred):
            rows.append((i, IDX_TO_CLASS[p]))

rows.sort(key=lambda t: t[0])
with open('submission.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id','label']); w.writerows(rows)

print("Wrote submission.csv with", len(rows), "rows.")



import pandas as pd
df = pd.read_csv("submission.csv")
print("shape:", df.shape)
print("columns:", list(df.columns))
print("id range:", df['id'].min(), "…", df['id'].max())
print("unique labels:", sorted(df['label'].unique()))
assert set(df['label'].unique()) <= set(IDX_TO_CLASS)
assert df['id'].is_monotonic_increasing
assert not df.isna().any().any()
print("✅ CSV looks good.")



# ===== Cell 9: Quick preview =====
import pandas as pd
pd.read_csv("submission.csv").head()


