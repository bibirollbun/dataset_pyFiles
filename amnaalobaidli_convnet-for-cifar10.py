# CIFAR-10 -> ResNet18 (random init) -> early-stop at 0.88 -> submission.csv
# Fast: AMP + auto batch-size tuning + efficient DataLoader

!apt -yq install -qq libarchive-dev
!pip -q install -q libarchive

import os, math, glob, csv, random, time, warnings
warnings.filterwarnings("ignore")

import torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import RandAugment
from PIL import Image

# ---------------- Config ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_ACC = 0.88
MAX_EPOCHS = 80          # will stop earlier if TARGET_ACC is hit
WARMUP_EPOCHS = 2
BASE_LR = 0.35           # slightly hot for fast convergence
WORKERS = 2
seed = 42
random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# --------------- Data -------------------
MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=6),           # good accuracy, still fast
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

trainset = datasets.CIFAR10("/kaggle/working", train=True,  download=True, transform=train_tfms)
valset   = datasets.CIFAR10("/kaggle/working", train=False, download=True, transform=test_tfms)
classes = trainset.classes

def make_loader(bs):
    return (DataLoader(trainset, batch_size=bs, shuffle=True,
                       num_workers=WORKERS, pin_memory=True,
                       persistent_workers=True, prefetch_factor=4, drop_last=True),
            DataLoader(valset, batch_size=bs, shuffle=False,
                       num_workers=WORKERS, pin_memory=True,
                       persistent_workers=True, prefetch_factor=4))

# --------- Auto-tune the biggest safe batch ---------
def find_max_batch(start=256):
    bs = start
    while bs >= 32:
        try:
            tl, vl = make_loader(bs)
            x, y = next(iter(tl))
            x = x.to(DEVICE); y = y.to(DEVICE)
            m = models.resnet18(weights=None, num_classes=10).to(DEVICE)
            with autocast(enabled=(DEVICE=="cuda")):
                _ = m(x); del m; torch.cuda.empty_cache()
            return bs
        except Exception as e:
            bs //= 2
            torch.cuda.empty_cache()
    return 32

BATCH = find_max_batch(256)
print("Using batch size:", BATCH)
train_loader, val_loader = make_loader(BATCH)

# --------------- Model (NO pretrained) ---------------
model = models.resnet18(weights=None, num_classes=10)
model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, 10))
model = model.to(DEVICE)

# --------------- Loss/Opt/Schedule -------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
scaler = GradScaler(enabled=(DEVICE=="cuda"))

steps_per_epoch = len(train_loader)
def lr_mult(step):
    if step < WARMUP_EPOCHS * steps_per_epoch:
        return step / float(WARMUP_EPOCHS * steps_per_epoch)
    p = (step - WARMUP_EPOCHS * steps_per_epoch) / float((MAX_EPOCHS - WARMUP_EPOCHS) * steps_per_epoch)
    return 0.5 * (1 + math.cos(math.pi * p))

@torch.no_grad()
def evaluate():
    model.eval(); correct=total=0; loss_sum=0.0
    with autocast(enabled=(DEVICE=="cuda")):
        for x,y in val_loader:
            x=x.to(DEVICE, non_blocking=True); y=y.to(DEVICE, non_blocking=True)
            out=model(x); loss=criterion(out,y)
            loss_sum += loss.item()*x.size(0)
            pred = out.argmax(1); correct += (pred==y).sum().item(); total += x.size(0)
    return loss_sum/total, correct/total

# --------------- Train (early-stop @ 0.88) ---------------
best = 0.0; step=0; t0=time.time()
for ep in range(1, MAX_EPOCHS+1):
    model.train()
    for x,y in train_loader:
        x=x.to(DEVICE, non_blocking=True); y=y.to(DEVICE, non_blocking=True)
        lr = BASE_LR * lr_mult(step)
        for pg in optimizer.param_groups: pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(DEVICE=="cuda")):
            out=model(x); loss=criterion(out,y)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        step += 1

    vl, va = evaluate()
    best = max(best, va)
    print(f"Epoch {ep:02d} | val_acc={va:.4f} (best {best:.4f}) | elapsed {int(time.time()-t0)}s")
    torch.cuda.empty_cache()
    if va >= TARGET_ACC:
        print(f"Early stop: val_acc {va:.4f} >= {TARGET_ACC}")
        break

# --------------- Extract Kaggle test.7z ----------------
import libarchive.public
if not os.path.exists("test"):
    assert os.path.exists("/kaggle/input/cifar-10/test.7z"), "Attach competition data (Add Data → CIFAR-10)."
    os.makedirs("test", exist_ok=True)
    for i,_ in enumerate(libarchive.public.file_pour('/kaggle/input/cifar-10/test.7z'),1):
        if i%10000==0: print("extracted:", i)

# --------------- Inference + light TTA -----------------
test_paths = sorted(glob.glob("test/*.png"))
class TestSet(torch.utils.data.Dataset):
    def __init__(self, paths, tfm): self.paths=paths; self.tfm=tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p=self.paths[idx]; img=Image.open(p).convert("RGB")
        return self.tfm(img), os.path.basename(p).split(".")[0]

test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
test_ds = TestSet(test_paths, test_tfms)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                         num_workers=WORKERS, pin_memory=True,
                         persistent_workers=True, prefetch_factor=4)

model.eval()
probs = torch.zeros((len(test_ds),10), device=DEVICE, dtype=torch.float32); ids=[]
with torch.no_grad(), autocast(enabled=(DEVICE=="cuda")):
    i0=0
    for x,ids_batch in test_loader:
        x=x.to(DEVICE, non_blocking=True)
        p1=model(x).softmax(1)
        p2=model(torch.flip(x,dims=[3])).softmax(1)
        pm=0.5*(p1+p2); b=x.size(0)
        probs[i0:i0+b]=pm; ids.extend(ids_batch); i0+=b

pred = probs.argmax(1).tolist()
labels=[classes[i] for i in pred]
with open("submission.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["id","label"])
    for i,lab in zip(ids,labels): w.writerow([i,lab])
print("submission.csv written with", len(ids), "rows")


