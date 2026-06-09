!mkdir -p /kaggle/working/stage1_train
!unzip -o -q /kaggle/input/data-science-bowl-2018/stage1_train.zip -d /kaggle/working/stage1_train



#Setup: imports, device, small helpers
#(What this cell does: imports libraries, sets device, seeds, and helper metrics + small image display util.)

import time, random, math, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

# reproducible-ish
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Metrics
def pixel_accuracy(pred, target):
    pred = pred.argmax(1)
    return (pred == target).float().mean().item()

def dice_score(pred, target, eps=1e-6):
    # pred: logits or probabilities [B,C,H,W] ; target: [B,H,W]
    if pred.dim() == 4:
        pred = pred.argmax(1)
    pred_flat = pred.view(-1).float()
    tgt_flat = target.view(-1).float()
    inter = (pred_flat * tgt_flat).sum()
    return (2. * inter + eps) / (pred_flat.sum() + tgt_flat.sum() + eps)

# A lighter Dice loss for multi-class (averaged)
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, target):
        # logits: [B,C,H,W], target: [B,H,W]
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=probs.shape[1]).permute(0,3,1,2).float()
        num = 2*(probs*target_onehot).sum(dim=(2,3))
        den = (probs + target_onehot).sum(dim=(2,3))
        dice = (num + self.eps) / (den + self.eps)
        return 1 - dice.mean()

# Quick batch visualizer
def show_images(inputs, targets, preds=None, n=3, cmap='viridis'):
    inputs = inputs.cpu().numpy().transpose(0,2,3,1)
    targets = targets.cpu().numpy()
    if preds is not None:
        preds = preds.cpu().numpy()
    fig,axs = plt.subplots(n,3,figsize=(9,3*n))
    for i in range(n):
        axs[i,0].imshow(inputs[i])
        axs[i,0].set_title("input"); axs[i,0].axis('off')
        axs[i,1].imshow(targets[i])
        axs[i,1].set_title("gt"); axs[i,1].axis('off')
        if preds is not None:
            axs[i,2].imshow(preds[i])
            axs[i,2].set_title("pred"); axs[i,2].axis('off')
    plt.tight_layout()



# Purpose: read Data Science Bowl 2018 stage1_train structure (folders -> images/*.png and masks/*.png),
# merge masks into one binary mask per image, resize to SIZE, sample SAMPLE_SIZE images,
# create train_loader and test_loader (70/30 split). Falls back to Oxford-IIIT Pet if DSB not found.

import os, glob, random
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

#(change SAMPLE_SIZE for smaller/larger)
SIZE = 128        # image resize (keep same as models)
BATCH = 8         # batch size used earlier
SAMPLE_SIZE = 100 # choose 20..100 for quick demos

ds_root = "/kaggle/working/stage1_train"



# image transform (to tensor, normalized [0,1])
img_transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
])

# resize mask with nearest neighbor to preserve label values
def load_and_resize_mask(mask_array, size=(SIZE, SIZE)):
    # mask_array: numpy 2D uint8 (0/1 or 0..255)
    pil = Image.fromarray((mask_array*255).astype(np.uint8))
    pil = pil.resize(size, resample=Image.NEAREST)
    arr = np.array(pil)
    arr = (arr > 0).astype(np.uint8)
    return torch.from_numpy(arr).long()  # shape [H,W], dtype long

data_items = []  # list of (tensor_image, tensor_mask)

if os.path.exists(ds_root):
    # find sample folders
    sample_dirs = sorted(glob.glob(os.path.join(ds_root, "*")))
    for s in sample_dirs:
        # find image file inside images/ subfolder
        img_files = glob.glob(os.path.join(s, "images", "*.png"))
        if len(img_files) == 0:
            continue
        img_path = img_files[0]
        # collect mask files (there may be many instance masks)
        mask_files = glob.glob(os.path.join(s, "masks", "*.png"))
        # load image
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            # skip if image corrupt
            print("skip image:", img_path, "error:", e); continue
        img_t = img_transform(img_pil)  # [3,H,W] float

        # build combined binary mask
        if len(mask_files) == 0:
            mask_t = torch.zeros((SIZE, SIZE), dtype=torch.long)
        else:
            # build combined mask at original resolution, then resize
            w,h = img_pil.size
            combined = np.zeros((h, w), dtype=np.uint8)
            for mf in mask_files:
                try:
                    m_pil = Image.open(mf).convert("L")
                    m_arr = np.array(m_pil)
                    combined = np.logical_or(combined, m_arr > 0)
                except:
                    continue
            mask_t = load_and_resize_mask(combined, size=(SIZE, SIZE))
        data_items.append((img_t, mask_t))
    print("Found DSB samples (after reading):", len(data_items))
else:
    print("DSB path not found at", ds_root)

# FALLBACK: if DSB not available or empty, use a small Oxford-IIIT Pet subset
if len(data_items) == 0:
    print("Falling back to small Oxford-IIIT Pet subset (so pipeline stays runnable).")
    from torchvision.datasets import OxfordIIITPet
    pet_root = "/kaggle/working/data"
    os.makedirs(pet_root, exist_ok=True)
    pet_ds = OxfordIIITPet(pet_root, download=True, target_types='segmentation',
                           transform=transforms.Compose([transforms.Resize((SIZE,SIZE)), transforms.ToTensor()]),
                           target_transform=transforms.Compose([transforms.Resize((SIZE,SIZE)), transforms.PILToTensor()]))
    # sample up to SAMPLE_SIZE
    idxs = list(range(len(pet_ds)))
    random.shuffle(idxs)
    idxs = idxs[:min(SAMPLE_SIZE, len(idxs))]
    for i in idxs:
        img, mask = pet_ds[i]
        if isinstance(mask, torch.Tensor) and mask.ndim == 3:
            mask = mask.squeeze(0)
        mask = (mask > 0).long()
        data_items.append((img, mask))

# Create a small sampled dataset for demo speed
random.shuffle(data_items)
SAMPLE_SIZE = min(SAMPLE_SIZE, len(data_items))
data_items = data_items[:SAMPLE_SIZE]
print("Using SAMPLE_SIZE =", SAMPLE_SIZE)

class SmallInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

dataset_small = SmallInMemoryDataset(data_items)

# 70/30 split (keeps variable names train_loader/test_loader)
n_total = len(dataset_small)
train_n = int(0.7 * n_total)
test_n = n_total - train_n
train_ds, test_ds = random_split(dataset_small, [train_n, test_n])

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

print("Train / Test sizes:", len(train_ds), len(test_ds))
# -----------------------------------------------------------------------------------------





#2) Model definitions (UNet, Attention-UNet, UNet++, Mobile-UNet)
#(What this cell does: defines 4 compact models. Each model is intentionally small and clear so you can understand. They all output logits with C=2 classes.)

# --- What: define small clean model classes for UNet, AttentionUNet, UNet++(simple) and MobileUNet.
# Double conv block
def conv3(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(out_ch,out_ch,3,padding=1), nn.ReLU(inplace=True))

# Depthwise separable conv for MobileUNet
class SeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.op(x)

# UNet baseline
class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, n_classes=2, base=32):
        super().__init__()
        self.d1 = conv3(in_ch, base)
        self.d2 = conv3(base, base*2)
        self.d3 = conv3(base*2, base*4)
        self.b  = conv3(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2,2)
        self.c3 = conv3(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2,2)
        self.c2 = conv3(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2,2)
        self.c1 = conv3(base*2, base)
        self.out = nn.Conv2d(base, n_classes, 1)
    def forward(self,x):
        s1 = self.d1(x); p1 = F.max_pool2d(s1,2)
        s2 = self.d2(p1); p2 = F.max_pool2d(s2,2)
        s3 = self.d3(p2); p3 = F.max_pool2d(s3,2)
        bn = self.b(p3)
        x = self.up3(bn); x = torch.cat([x,s3],1); x = self.c3(x)
        x = self.up2(x);  x = torch.cat([x,s2],1); x = self.c2(x)
        x = self.up1(x);  x = torch.cat([x,s1],1); x = self.c1(x)
        return self.out(x)

# Attention Gate small
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """F_g = channels of gating tensor you will pass in.
           F_l = channels of the skip (x).
           F_int = intermediate reduced channels (a smaller number).
        """
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # x: skip feature (B, F_l, Hx, Wx)
        # g: gating feature (B, F_g, Hg, Wg)  - may be different spatial size
        # If spatial sizes differ, upsample gating to match skip's spatial size
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Now channel dims must match the convolutions' expected channels
        g1 = self.W_g(g)      # -> B x F_int x Hx x Wx
        x1 = self.W_x(x)      # -> B x F_int x Hx x Wx

        psi = self.relu(g1 + x1)   # elementwise add in common spatial resolution
        psi = self.psi(psi)        # -> B x 1 x Hx x Wx
        alpha = self.sigmoid(psi)  # attention coefficients (0..1)
        return x * alpha           # gated/filtered skip feature map


# Attention UNet (small) uses same structure as UNet but with gates on skips
class AttUNetSmall(UNetSmall):
    def __init__(self, in_ch=3, n_classes=2, base=32):
        super().__init__(in_ch, n_classes, base)
        # IMPORTANT:
        # We will PASS the *upsampled decoder feature* (x after up conv) as gating.
        # Therefore F_g must match the channel count of that upsampled feature.
        # up3 produces channels base*4, up2 -> base*2, up1 -> base
        self.ag3 = AttentionGate(F_g=base*4, F_l=base*4, F_int=base*2)  # gate between s3 (base*4) and up3(x) (base*4)
        self.ag2 = AttentionGate(F_g=base*2, F_l=base*2, F_int=base)    # gate between s2 (base*2) and up2(x) (base*2)
        # ensure F_int>=1
        f_int1 = max(base//2, 1)
        self.ag1 = AttentionGate(F_g=base, F_l=base, F_int=f_int1)      # gate between s1 (base) and up1(x) (base)

    def forward(self, x):
        s1 = self.d1(x); p1 = F.max_pool2d(s1,2)
        s2 = self.d2(p1); p2 = F.max_pool2d(s2,2)
        s3 = self.d3(p2); p3 = F.max_pool2d(s3,2)
        bn = self.b(p3)

        # Decoder: after each upsample we pass the upsampled feature as gating to the corresponding AG
        x = self.up3(bn)                  # x channels = base*4 (matches ag3 F_g)
        s3f = self.ag3(s3, x)             # gate: skip s3 filtered by gating x
        x = torch.cat([x, s3f], 1); x = self.c3(x)

        x = self.up2(x)                   # x channels = base*2 (matches ag2 F_g)
        s2f = self.ag2(s2, x)
        x = torch.cat([x, s2f], 1); x = self.c2(x)

        x = self.up1(x)                   # x channels = base (matches ag1 F_g)
        s1f = self.ag1(s1, x)
        x = torch.cat([x, s1f], 1); x = self.c1(x)

        return self.out(x)


# UNet++ simplified: add intermediate conv to refine skip before concat
class UNetPP(nn.Module):
    def __init__(self, in_ch=3, n_classes=2, base=32):
        super().__init__()
        self.d1 = conv3(in_ch, base)
        self.d2 = conv3(base, base*2)
        self.d3 = conv3(base*2, base*4)
        self.b  = conv3(base*4, base*8)
        # intermediate refinement convs for skip
        self.r32 = nn.Conv2d(base*4, base*4, 3, padding=1)
        self.r21 = nn.Conv2d(base*2, base*2, 3, padding=1)
        # decoder
        self.up3 = nn.ConvTranspose2d(base*8, base*4,2,2); self.c3=conv3(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2,2,2); self.c2=conv3(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2,2);  self.c1=conv3(base*2, base)
        self.out = nn.Conv2d(base, n_classes,1)
    def forward(self,x):
        s1 = self.d1(x); p1 = F.max_pool2d(s1,2)
        s2 = self.d2(p1); p2 = F.max_pool2d(s2,2)
        s3 = self.d3(p2); p3 = F.max_pool2d(s3,2)
        bn = self.b(p3)
        # refine skips slightly
        s3r = F.relu(self.r32(s3))
        s2r = F.relu(self.r21(s2))
        x = self.up3(bn); x = torch.cat([x,s3r],1); x = self.c3(x)
        x = self.up2(x);  x = torch.cat([x,s2r],1); x = self.c2(x)
        x = self.up1(x);  x = torch.cat([x,s1],1);  x = self.c1(x)
        return self.out(x)

# Mobile UNet using separable convs
class MobileUNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=2, base=32):
        super().__init__()
        self.d1 = nn.Sequential(SeparableConv(in_ch, base), SeparableConv(base, base))
        self.d2 = nn.Sequential(SeparableConv(base, base*2), SeparableConv(base*2, base*2))
        self.d3 = nn.Sequential(SeparableConv(base*2, base*4), SeparableConv(base*4, base*4))
        self.b  = nn.Sequential(SeparableConv(base*4, base*8), SeparableConv(base*8, base*8))
        self.up3 = nn.ConvTranspose2d(base*8, base*4,2,2); self.c3 = nn.Sequential(SeparableConv(base*8, base*4))
        self.up2 = nn.ConvTranspose2d(base*4, base*2,2,2); self.c2 = nn.Sequential(SeparableConv(base*4, base*2))
        self.up1 = nn.ConvTranspose2d(base*2, base,2,2);   self.c1 = nn.Sequential(SeparableConv(base*2, base))
        self.out = nn.Conv2d(base, n_classes, 1)
    def forward(self,x):
        s1 = self.d1(x); p1 = F.max_pool2d(s1,2)
        s2 = self.d2(p1); p2 = F.max_pool2d(s2,2)
        s3 = self.d3(p2); p3 = F.max_pool2d(s3,2)
        bn = self.b(p3)
        x = self.up3(bn); x = torch.cat([x,s3],1); x = self.c3(x)
        x = self.up2(x);  x = torch.cat([x,s2],1); x = self.c2(x)
        x = self.up1(x);  x = torch.cat([x,s1],1); x = self.c1(x)
        return self.out(x)





#3) Training utilities: one train/validate epoch function, timing
#(What this cell does: defines train_epoch and eval_model functions which perform forward/backward, compute combined CE+Dice loss, track metrics, and measure inference time.)



# --- What: train/val helpers using CE + Dice, measuring loss, accuracy, dice, and inference time
def train_epoch(model, loader, optim, ce_loss, dice_loss):
    model.train()
    running = {"loss":0.0, "acc":0.0, "dice":0.0}
    n = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = ce_loss(logits, y) + dice_loss(logits, y)
        loss.backward()
        optim.step()
        running["loss"] += loss.item()
        running["acc"] += pixel_accuracy(logits.detach().cpu(), y.detach().cpu())
        running["dice"] += dice_score(logits.detach().cpu(), y.detach().cpu())
        n += 1
    return {k: v / n for k,v in running.items()}


#eval_model (safe on CPU or GPU)
@torch.no_grad()
def eval_model(model, loader, ce_loss, dice_loss, measure_speed=False):
    model.eval()
    running = {"loss":0.0, "acc":0.0, "dice":0.0}
    n = 0
    total_time = 0.0
    total_samples = 0

    use_cuda = torch.cuda.is_available()
    for x,y in loader:
        x,y = x.to(device), y.to(device)

        if measure_speed and use_cuda:
            # accurate GPU timing
            torch.cuda.synchronize()
            t0 = time.time()
            logits = model(x)
            torch.cuda.synchronize()
            t1 = time.time()
            total_time += (t1 - t0)
            total_samples += x.size(0)
        elif measure_speed and not use_cuda:
            # CPU timing (no cuda synchronization possible)
            t0 = time.time()
            logits = model(x)
            t1 = time.time()
            total_time += (t1 - t0)
            total_samples += x.size(0)
        else:
            logits = model(x)

        loss = ce_loss(logits, y) + dice_loss(logits, y)
        running["loss"] += loss.item()
        running["acc"] += pixel_accuracy(logits.detach().cpu(), y.detach().cpu())
        running["dice"] += dice_score(logits.detach().cpu(), y.detach().cpu())
        n += 1

    stats = {k: v / n for k,v in running.items()} if n>0 else {"loss":0,"acc":0,"dice":0}
    if measure_speed:
        if total_samples > 0:
            stats["time_per_image"] = total_time / total_samples
        else:
            stats["time_per_image"] = None
    return stats







#4) Train all models sequentially and collect stats (small epochs for demo)
#(What this cell does: instantiates each model, trains for a few epochs, records per-epoch stats for plotting. For demo it uses EPOCHS=6. Increase for serious training.)


# --- What: run short training for each model and collect history for plots. Increase EPOCHS in real runs.
EPOCHS = 25
models = {
    "UNet": UNetSmall().to(device),
    "AttUNet": AttUNetSmall().to(device),
    "UNet++": UNetPP().to(device),
    "MobileUNet": MobileUNet().to(device)
}

hist = {}
for name, model in models.items():
    print(f"\n=== TRAINING {name} ===")
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    dice = DiceLoss()
    hist[name] = {"train_loss":[],"val_loss":[],"train_acc":[], "val_acc":[], "train_dice":[], "val_dice":[]}
    for ep in range(EPOCHS):
        tr = train_epoch(model, train_loader, optim, ce, dice)
        val = eval_model(model, test_loader, ce, dice)
        print(f"{name} E{ep}: tr_loss {tr['loss']:.4f} val_loss {val['loss']:.4f} tr_dice {tr['dice']:.3f} val_dice {val['dice']:.3f}")
        hist[name]["train_loss"].append(tr["loss"]); hist[name]["val_loss"].append(val["loss"])
        hist[name]["train_acc"].append(tr["acc"]); hist[name]["val_acc"].append(val["acc"])
        hist[name]["train_dice"].append(tr["dice"]); hist[name]["val_dice"].append(val["dice"])
    # Save model
    torch.save(model.state_dict(), f"/kaggle/working/{name}.pth")





#5) Plots: training loss / dice curves and bar chart comparison (What this does: creates comparison plots across models)

# --- What: plot loss and dice curves for each model and a summary bar chart of final metrics.
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
for name in hist:
    plt.plot(hist[name]["val_loss"], label=name)
plt.title("Val Loss"); plt.legend()

plt.subplot(1,2,2)
for name in hist:
    plt.plot(hist[name]["val_dice"], label=name)
plt.title("Val Dice"); plt.legend()
plt.show()

# final summary bar chart (last epoch)
names = list(hist.keys())
final_dice = [hist[n]["val_dice"][-1] for n in names]
final_acc  = [hist[n]["val_acc"][-1] for n in names]
x = np.arange(len(names))
plt.figure(figsize=(8,4))
plt.bar(x-0.15, final_dice, width=0.3, label="Dice")
plt.bar(x+0.15, final_acc,  width=0.3, label="PixelAcc")
plt.xticks(x, names)
plt.ylim(0,1)
plt.legend(); plt.title("Final validation metrics (last epoch)")
plt.show()






#6) Per-model inference speed and sample outputs (What this does: measures inference time per image and shows several example input/gt/pred images for each model.)

# --- What: measure inference speed (time per image) and display sample predictions from each model
ce = nn.CrossEntropyLoss(); dice = DiceLoss()
summary = {}
for name, model in models.items():
    s = eval_model(model, test_loader, ce, dice, measure_speed=True)
    summary[name] = s
    print(f"{name}: val_loss {s['loss']:.4f} val_dice {s['dice']:.3f} time_per_image {s.get('time_per_image',0):.4f}s")

# show sample predictions (first batch from test_loader)
x,y = next(iter(test_loader))
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        preds = model(x.to(device)).argmax(1).cpu()
    show_images(x, y, preds, n=min(3, x.size(0)))
    plt.suptitle(name); plt.show()





#7) Save a final comparison table and conclude
#(What this cell does: prints a concise table of per-model metrics for reporting / slides.)


# --- What: summarize metrics in a compact table (ready to copy into slides)
print("Model\tValLoss\tValDice\tValAcc\tTimePerImage(s)")
for name in names:
    s = summary[name]
    print(f"{name}\t{s['loss']:.4f}\t{s['dice']:.4f}\t{s['acc']:.4f}\t{s.get('time_per_image',0):.4f}")




names = list(summary.keys())
accs  = [summary[n]["acc"]*100 for n in names]   # convert to %

plt.figure(figsize=(6,4))
plt.bar(names, accs)
plt.ylabel("Pixel Accuracy (%)")
plt.title("Pixel Level Accuracy Comparison Across Models")
for i,v in enumerate(accs):
    plt.text(i, v+0.5, f"{v:.2f}%", ha='center')  # show % on top of bar
plt.show()


