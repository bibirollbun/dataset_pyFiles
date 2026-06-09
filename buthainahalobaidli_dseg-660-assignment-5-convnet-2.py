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


# C') scratch-built residual CNN for CIFAR-10 (no torchvision models)

import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.relu(out + identity)
        return out

def make_stage(in_ch, out_ch, num_blocks, first_stride):
    layers = [BasicBlock(in_ch, out_ch, stride=first_stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)

class SmallResNetCIFAR(nn.Module):
    # 3 stages: [64, 128, 256]; total ~11 layers with residuals
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),  # CIFAR stem
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = make_stage(64,  64,  num_blocks=2, first_stride=1)  # 32x32
        self.stage2 = make_stage(64,  128, num_blocks=2, first_stride=2)  # 16x16
        self.stage3 = make_stage(128, 256, num_blocks=2, first_stride=2)  # 8x8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(256, num_classes)

        # kaiming initialization (still random, no pretrain)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x

model = SmallResNetCIFAR().to(DEVICE)
print("Custom residual model ready (random init; no torchvision models used).")


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


# Graphviz diagram for the scratch residual CIFAR-10 CNN
# Outputs: /kaggle/working/architecture_graphviz.png and .svg

# Install graphviz engine + Python bindings (safe to re-run)
!apt -yq install graphviz >/dev/null
!pip -q install graphviz >/dev/null

from graphviz import Digraph
from IPython.display import Image, display

# --- Theme (tweak if you like) ---
PINK       = "#ff8fab"   # lines / text
LIGHTPINK  = "#ffe5ec"   # node fill
BORDER     = "#ffc2d1"   # cluster border
FONT       = "DejaVu Sans"

g = Digraph("CIFAR10_ResCNN", format="png")
g.attr(rankdir="LR", bgcolor="white", pad="0.25", splines="spline",
       ranksep="0.65", nodesep="0.45", dpi="300")
g.attr("node", shape="box", style="rounded,filled",
       color=PINK, fillcolor=LIGHTPINK, penwidth="2.2",
       fontname=FONT, fontsize="11")
g.attr("edge", color=PINK, penwidth="1.8", arrowsize="0.85")

# ---- Nodes & clusters ----
g.node("inp", "Input\n32×32×3")

with g.subgraph(name="cluster_stem") as c:
    c.attr(label="Stem", color=BORDER, penwidth="2", style="rounded", fontname=FONT)
    c.node("stem", "3×3 conv\nBN, ReLU\n→ 32×32×64")

with g.subgraph(name="cluster_s1") as c:
    c.attr(label="Stage 1 (64)", color=BORDER, penwidth="2", style="rounded", fontname=FONT)
    c.node("s1b1", "Block 1\n3×3 → 3×3")
    c.node("plus1", "+", shape="circle", width="0.35", height="0.35",
           style="filled", fillcolor=LIGHTPINK, color=PINK, fontsize="10")
    c.node("s1b2", "Block 2\n3×3 → 3×3")

with g.subgraph(name="cluster_s2") as c:
    c.attr(label="Stage 2 (128, stride 2)", color=BORDER, penwidth="2", style="rounded", fontname=FONT)
    c.node("s2b1", "Block 1\n3×3 → 3×3")
    c.node("plus2", "+", shape="circle", width="0.35", height="0.35",
           style="filled", fillcolor=LIGHTPINK, color=PINK, fontsize="10")
    c.node("s2b2", "Block 2\n3×3 → 3×3")

with g.subgraph(name="cluster_s3") as c:
    c.attr(label="Stage 3 (256, stride 2)", color=BORDER, penwidth="2", style="rounded", fontname=FONT)
    c.node("s3b1", "Block 1\n3×3 → 3×3")
    c.node("plus3", "+", shape="circle", width="0.35", height="0.35",
           style="filled", fillcolor=LIGHTPINK, color=PINK, fontsize="10")
    c.node("s3b2", "Block 2\n3×3 → 3×3")

g.node("head", "GAP → FC(256→10)")

# ---- Main forward path ----
g.edges([
    ("inp","stem"),
    ("stem","s1b1"),
    ("s1b1","s1b2"),
    ("s1b2","s2b1"),
    ("s2b1","s2b2"),
    ("s2b2","s3b1"),
    ("s3b1","s3b2"),
    ("s3b2","head"),
])

# ---- Residual (skip) visuals (dashed, non-constraining) ----
# Stage 1: input to block1 added before block2
g.edge("stem","plus1", style="dashed", xlabel="skip", constraint="false")
g.edge("s1b1","plus1", constraint="false")
g.edge("plus1","s1b2", arrowhead="none", constraint="false")

# Stage 2: downsample skip (typically 1×1 conv); add before block2
g.edge("s1b2","plus2", style="dashed", xlabel="skip/1×1", constraint="false")
g.edge("s2b1","plus2", constraint="false")
g.edge("plus2","s2b2", arrowhead="none", constraint="false")

# Stage 3: downsample skip; add before block2
g.edge("s2b2","plus3", style="dashed", xlabel="skip/1×1", constraint="false")
g.edge("s3b1","plus3", constraint="false")
g.edge("plus3","s3b2", arrowhead="none", constraint="false")

# ---- Render & show ----
png_path = g.render(filename="architecture_graphviz")
g.format = "svg"; svg_path = g.render(filename="architecture_graphviz")
print("Saved:", png_path, "and", svg_path)
display(Image(filename=png_path))



# Cute pink Graphviz diagram (explicit Conv2d labels)
# Outputs: /kaggle/working/architecture_graphviz.png and .svg

# Install Graphviz engine + Python bindings (safe to re-run)
!apt -yq install graphviz >/dev/null
!pip -q install graphviz >/dev/null

from graphviz import Digraph
from IPython.display import Image, display

# --- Theme ---
PINK, LIGHTPINK, BORDER = "#ff8fab", "#ffe5ec", "#ffc2d1"
FONT = "DejaVu Sans"

g = Digraph("CIFAR10_ResCNN", format="png")
g.attr(rankdir="LR", bgcolor="white", pad="0.25", splines="spline",
       ranksep="0.65", nodesep="0.45", dpi="300")
g.attr("node", shape="box", style="rounded,filled",
       color=PINK, fillcolor=LIGHTPINK, penwidth="2.2",
       fontname=FONT, fontsize="11")
g.attr("edge", color=PINK, penwidth="1.8", arrowsize="0.85")

# ----- Nodes & clusters -----
g.node("inp", "Input\n32×32×3")

with g.subgraph(name="cluster_stem") as c:
    c.attr(label="Stem", color=BORDER, penwidth="2", style="rounded", fontname=FONT)
    c.node("stem", "Conv2d 3×3\nBatchNorm2d, ReLU\n→ 32×32×64")

with g.subgraph(name="cluster_s1") as c:
    c.attr(label="Stage 1 (64)", color=BORDER, penwidth="2", style="rounded", fontname=FONT)
    c.node("s1b1", "Block 1\nConv2d 3×3 → 3×3")
    c.node("plus1", "+", shape="circle", width="0.35", height="0.35",
           style="filled", fillcolor=LIGHTPINK, color=PINK, fontsize="10")
    c.node("s1b2", "Block 2\nConv2d 3×3 → 3×3")

with g.subgraph(name="cluster_s2") as c:
    c.attr(label="Stage 2 (128, stride 2)", color=BORDER, penwidth="2", style="rounded", fontname=FONT)
    c.node("s2b1", "Block 1\nConv2d 3×3 → 3×3")
    c.node("plus2", "+", shape="circle", width="0.35", height="0.35",
           style="filled", fillcolor=LIGHTPINK, color=PINK, fontsize="10")
    c.node("s2b2", "Block 2\nConv2d 3×3 → 3×3")

with g.subgraph(name="cluster_s3") as c:
    c.attr(label="Stage 3 (256, stride 2)", color=BORDER, penwidth="2", style="rounded", fontname=FONT)
    c.node("s3b1", "Block 1\nConv2d 3×3 → 3×3")
    c.node("plus3", "+", shape="circle", width="0.35", height="0.35",
           style="filled", fillcolor=LIGHTPINK, color=PINK, fontsize="10")
    c.node("s3b2", "Block 2\nConv2d 3×3 → 3×3")

g.node("head", "GAP → Linear(256→10)")

# ----- Main forward path -----
g.edges([
    ("inp","stem"),
    ("stem","s1b1"),
    ("s1b1","s1b2"),
    ("s1b2","s2b1"),
    ("s2b1","s2b2"),
    ("s2b2","s3b1"),
    ("s3b1","s3b2"),
    ("s3b2","head"),
])

# ----- Residual/skip visuals (dashed, non-constraining arcs) -----
g.edge("stem","plus1", style="dashed", xlabel="skip", constraint="false")
g.edge("s1b1","plus1", constraint="false")
g.edge("plus1","s1b2", arrowhead="none", constraint="false")

g.edge("s1b2","plus2", style="dashed", xlabel="skip / 1×1 Conv2d", constraint="false")
g.edge("s2b1","plus2", constraint="false")
g.edge("plus2","s2b2", arrowhead="none", constraint="false")

g.edge("s2b2","plus3", style="dashed", xlabel="skip / 1×1 Conv2d", constraint="false")
g.edge("s3b1","plus3", constraint="false")
g.edge("plus3","s3b2", arrowhead="none", constraint="false")

# ----- Render & preview -----
png_path = g.render(filename="architecture_graphviz")
g.format = "svg"; svg_path = g.render(filename="architecture_graphviz")
print("Saved:", png_path, "and", svg_path)
display(Image(filename=png_path))



# Verify residual connections exist in each block
from inspect import signature

total = 0
for name, m in model.named_modules():
    if m.__class__.__name__ == "BasicBlock":
        total += 1
        has_proj = getattr(m, "down", None) is not None
        print(f"{name}: residual add ✓  |  skip = {'1x1 Conv2d' if has_proj else 'identity'}")
print("Total residual blocks:", total)


