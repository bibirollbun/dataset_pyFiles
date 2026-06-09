# ======================================================
# Cell 1: Train a plain custom ConvNet on CIFAR-10
# - Random init (no pretrained)
# - No rotation (only crop + horizontal flip)
# - Saves best weights to: best_convnet.pt
# - Trains on official CIFAR-10 via torchvision (fast; no need to extract train.7z)
# ======================================================
import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler

# Repro + device
SEED=1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Device:", device)

# CIFAR-10 stats and transforms (NO rotation)
CIFAR_MEAN=(0.4914,0.4822,0.4465)
CIFAR_STD =(0.2470,0.2435,0.2616)
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),  # âœ… no rotation
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    transforms.RandomErasing(p=0.25),
])
test_tfms  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# Load official CIFAR-10 (download once to /kaggle/working)
DATA_ROOT="/kaggle/working"
_base = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=None)
idx = np.arange(len(_base)); np.random.default_rng(SEED).shuffle(idx)
VAL_SZ=5000
train_idx, val_idx = idx[VAL_SZ:], idx[:VAL_SZ]

train_set = Subset(datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=train_tfms), train_idx.tolist())
val_set   = Subset(datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=test_tfms),  val_idx.tolist())

BATCH_TRAIN, BATCH_VAL = 256, 512
train_loader = DataLoader(train_set, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_VAL,   shuffle=False, num_workers=2, pin_memory=True)

print(f"Train rows: {len(train_set)} | Val rows: {len(val_set)}")

# Model: plain custom ConvNet (random init)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(3,64,3,padding=1);  self.b1=nn.BatchNorm2d(64)
        self.c2=nn.Conv2d(64,64,3,padding=1); self.b2=nn.BatchNorm2d(64)
        self.p1=nn.MaxPool2d(2,2)
        self.c3=nn.Conv2d(64,128,3,padding=1); self.b3=nn.BatchNorm2d(128)
        self.c4=nn.Conv2d(128,128,3,padding=1);self.b4=nn.BatchNorm2d(128)
        self.p2=nn.MaxPool2d(2,2)
        self.c5=nn.Conv2d(128,256,3,padding=1);self.b5=nn.BatchNorm2d(256)
        self.c6=nn.Conv2d(256,256,3,padding=1);self.b6=nn.BatchNorm2d(256)
        self.p3=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(256*4*4,512); self.drop=nn.Dropout(0.5)
        self.fc2=nn.Linear(512,10)
    def forward(self,x):
        x=F.relu(self.b1(self.c1(x))); x=F.relu(self.b2(self.c2(x))); x=self.p1(x)
        x=F.relu(self.b3(self.c3(x))); x=F.relu(self.b4(self.c4(x))); x=self.p2(x)
        x=F.relu(self.b5(self.c5(x))); x=F.relu(self.b6(self.c6(x))); x=self.p3(x)
        x=torch.flatten(x,1); x=F.relu(self.fc1(x)); x=self.drop(x)
        return self.fc2(x)

model = Net().to(device).to(memory_format=torch.channels_last)

# Optimizer / Scheduler / Loss / AMP
EPOCHS=60
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.CrossEntropyLoss(label_smoothing=0.05)
scaler = GradScaler()

def run_epoch(dl, train=True):
    model.train(train)
    tot=correct=loss_sum=0.0
    for x,y in dl:
        x=x.to(device,non_blocking=True).contiguous(memory_format=torch.channels_last)
        y=y.to(device,non_blocking=True)
        if train: opt.zero_grad(set_to_none=True)
        with autocast():
            logits=model(x); loss=crit(logits,y)
        if train:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        loss_sum+=loss.item()*x.size(0); correct+=(logits.argmax(1)==y).sum().item(); tot+=x.size(0)
    return loss_sum/tot, correct/tot

best=(0.0,None)
for e in range(1, EPOCHS+1):
    tr_loss,tr_acc = run_epoch(train_loader, True)
    va_loss,va_acc = run_epoch(val_loader,   False)
    sch.step()
    if va_acc>best[0]: best=(va_acc, {k:v.detach().cpu() for k,v in model.state_dict().items()})
    print(f"Ep{e:02d}/{EPOCHS}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")

# Save best weights
torch.save(best[1], "best_convnet.pt")
print("âœ… Saved best_convnet.pt")



# ======================================================
# Cell 2: Extraction with OVERALL COUNT (e.g., 12345/300000)
# - Extracts /kaggle/input/cifar-10/test.7z â†’ ./test/
# - Prints one updating line: "<done>/<total> (<pct>%)"
# - Skips if already extracted (â‰¥300k)
# ======================================================
import os, glob, shutil, subprocess, sys, time
import os
import libarchive.public
import glob

# Define dataset base directory
COMP_DIR = "/kaggle/input/cifar-10"
TEST_7Z = os.path.join(COMP_DIR, "test.7z")
EXTRACT_DIR = "test"

# Extract using libarchive (âš ï¸� very slow for large .7z files)
cnt = 0
print("â�³ Reading entries from test.7z...")

for entry in libarchive.public.file_pour(TEST_7Z):
    cnt += 1
    if cnt % 1000 == 0:
        print(cnt)

print(f"âœ… Total entries read: {cnt}")

# Optional: Count number of extracted PNG files
png_count = len(glob.glob(os.path.join(EXTRACT_DIR, "*.png")))
print(f"ğŸ§¾ Extracted image count in '{EXTRACT_DIR}/': {png_count}")

TEST_7Z  = os.path.join(COMP_DIR, "test.7z")
TEST_DIR = "./test"
TOTAL    = 300_000  # CIFAR-10 Kaggle test count

assert os.path.exists(TEST_7Z), "Missing /kaggle/input/cifar-10/test.7z"
os.makedirs(TEST_DIR, exist_ok=True)

def ensure_7z():
    if shutil.which("7z"):
        return "7z"
    subprocess.run(["apt-get","update","-y"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["apt-get","install","-y","p7zip-full"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert shutil.which("7z"), "7z not found after install"
    return "7z"

# If already extracted, skip
have = len(glob.glob(os.path.join(TEST_DIR, "*.png")))
if have >= TOTAL:
    print(f"âœ… Already extracted: {have} / {TOTAL} in {TEST_DIR}")
else:
    sevenz = ensure_7z()
    print(f"Extracting to {os.path.abspath(TEST_DIR)}")
    # Start extraction in the background
    # Archive has 'test/...' inside; extract to current dir so files land in ./test/
    proc = subprocess.Popen(
        [sevenz, "x", TEST_7Z, "-y", "-o."],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Poll progress by counting files in ./test/*.png
    last_print = -1
    try:
        while True:
            # count extracted PNGs so far
            done = len(glob.glob(os.path.join(TEST_DIR, "*.png")))
            if done != last_print:
                pct = 100.0 * min(done, TOTAL) / TOTAL
                print(f"\rprogress: {done}/{TOTAL} ({pct:5.1f}%)", end="", flush=True)
                last_print = done
            # break when process finishes and count stops increasing
            if proc.poll() is not None:
                # one last update after process ends
                done = len(glob.glob(os.path.join(TEST_DIR, "*.png")))
                pct = 100.0 * min(done, TOTAL) / TOTAL
                print(f"\rprogress: {done}/{TOTAL} ({pct:5.1f}%)")
                break
            time.sleep(1)
    finally:
        # Ensure process is not left hanging
        try:
            proc.terminate()
        except Exception:
            pass

    final = len(glob.glob(os.path.join(TEST_DIR, "*.png")))
    print(f"âœ… Extracted: {final} / {TOTAL} in {TEST_DIR}")



# ======================================================
# Cell 3: Inference on Kaggle test set
# - Loads best_convnet.pt
# - If ./test/ exists: read PNGs directly
# - Else: batched tiny-extract from test.7z with py7zr (no full unzip)
# - Saves ids.npy and pred_idx.npy for the submission cell
# ======================================================
import os, glob, tempfile, subprocess, numpy as np, pandas as pd
from PIL import Image
import torch
from torchvision import transforms

# Load model
CLASS_NAMES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(3,64,3,padding=1);  self.b1=nn.BatchNorm2d(64)
        self.c2=nn.Conv2d(64,64,3,padding=1); self.b2=nn.BatchNorm2d(64)
        self.p1=nn.MaxPool2d(2,2)
        self.c3=nn.Conv2d(64,128,3,padding=1); self.b3=nn.BatchNorm2d(128)
        self.c4=nn.Conv2d(128,128,3,padding=1);self.b4=nn.BatchNorm2d(128)
        self.p2=nn.MaxPool2d(2,2)
        self.c5=nn.Conv2d(128,256,3,padding=1);self.b5=nn.BatchNorm2d(256)
        self.c6=nn.Conv2d(256,256,3,padding=1);self.b6=nn.BatchNorm2d(256)
        self.p3=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(256*4*4,512); self.drop=nn.Dropout(0.5)
        self.fc2=nn.Linear(512,10)
    def forward(self,x):
        import torch.nn.functional as F
        x=F.relu(self.b1(self.c1(x))); x=F.relu(self.b2(self.c2(x))); x=self.p1(x)
        x=F.relu(self.b3(self.c3(x))); x=F.relu(self.b4(self.c4(x))); x=self.p2(x)
        x=F.relu(self.b5(self.c5(x))); x=F.relu(self.b6(self.c6(x))); x=self.p3(x)
        x=torch.flatten(x,1); x=F.relu(self.fc1(x)); x=self.drop(x)
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device).to(memory_format=torch.channels_last)
state = torch.load("best_convnet.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
print("âœ… Loaded best_convnet.pt")

# Paths and transforms
COMP_DIR = "/kaggle/input/cifar-10"
SAMPLE_SUB = os.path.join(COMP_DIR, "sampleSubmission.csv")
TEST_7Z    = os.path.join(COMP_DIR, "test.7z")
TEST_DIR   = "./test"
assert os.path.exists(SAMPLE_SUB), "Missing sampleSubmission.csv"

CIFAR_MEAN=(0.4914,0.4822,0.4465); CIFAR_STD=(0.2470,0.2435,0.2616)
eval_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

# Read ID order
sample = pd.read_csv(SAMPLE_SUB)
ids = sample["id"].astype(int).tolist()
N = len(ids)
print("IDs to predict:", N)

from torch.cuda.amp import autocast
import torch

BATCH = 1024  # lower if OOM

pred_idx = np.empty(N, dtype=np.int64)

if os.path.isdir(TEST_DIR) and glob.glob(os.path.join(TEST_DIR, "1.png")):
    # -------- Path A: fully extracted test/ exists --------
    print("Using extracted files in ./test/")
    def batched(iterable, n):
        for i in range(0, len(iterable), n):
            yield i, iterable[i:i+n]
    with torch.no_grad():
        done=0
        for start, id_batch in batched(ids, BATCH):
            imgs=[]
            for i in id_batch:
                p = os.path.join(TEST_DIR, f"{i}.png")
                img = Image.open(p).convert("RGB")
                imgs.append(eval_tfm(img))
            xb = torch.stack(imgs).to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with autocast():
                logits = model(xb)
            preds = logits.argmax(1).cpu().numpy()
            pred_idx[start:start+len(id_batch)] = preds
            done += len(id_batch)
            if done % 10000 == 0: print(f"progress: {done}/{N}")
else:
    # -------- Path B: no extraction â†’ tiny batched extracts from test.7z --------
    print("No ./test/ found; using batched tiny-extract from test.7z")
    try:
        import py7zr  # noqa
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py7zr"], check=True)
        import py7zr

    def batched(iterable, n):
        for i in range(0, len(iterable), n):
            yield i, iterable[i:i+n]

    with torch.no_grad(), py7zr.SevenZipFile(TEST_7Z, mode="r") as z:
        done=0
        for start, id_batch in batched(ids, BATCH):
            with tempfile.TemporaryDirectory(dir="/kaggle/temp") as tmpdir:
                targets = [f"{i}.png" for i in id_batch]
                # extract ONLY this batch to tmpdir
                try:
                    z.extract(path=tmpdir, targets=targets)
                except TypeError:
                    z.extract(targets=targets, path=tmpdir)
                imgs=[]
                for i in id_batch:
                    p = os.path.join(tmpdir, f"{i}.png")
                    img = Image.open(p).convert("RGB")
                    imgs.append(eval_tfm(img))
                xb = torch.stack(imgs).to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                with autocast():
                    logits = model(xb)
                preds = logits.argmax(1).cpu().numpy()
                pred_idx[start:start+len(id_batch)] = preds
            done += len(id_batch)
            if done % 10000 == 0: print(f"progress: {done}/{N}")

# Save inference artifacts for the submission cell
np.save("ids.npy", np.array(ids, dtype=np.int32))
np.save("pred_idx.npy", pred_idx)
print("âœ… Saved ids.npy and pred_idx.npy")



# ======================================================
# Cell 4: Build submission.csv
# - Loads ids.npy + pred_idx.npy from Cell 3
# - Maps class indices â†’ labels
# - Writes submission.csv
# ======================================================
import numpy as np, pandas as pd

CLASS_NAMES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

ids = np.load("ids.npy")
pred_idx = np.load("pred_idx.npy")
labels = [CLASS_NAMES[i] for i in pred_idx.tolist()]

sub = pd.DataFrame({"id": ids.astype(int), "label": labels})
sub.to_csv("submission.csv", index=False)
print("âœ… Wrote submission.csv")
print(sub.head())
print("shape:", sub.shape, "| columns:", list(sub.columns), "| ids unique:", sub['id'].is_unique)


