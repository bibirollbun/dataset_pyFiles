import os, gc, sys, math, random, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed()



CFG = {
    "HEIGHT": 256,
    "WIDTH": 1600,
    "NUM_CLASSES": 4,
    "VAL_SPLIT": 0.2,
    "BATCH_SIZE": 4,
    "NUM_WORKERS": 2,
    "EPOCHS": 10,
    "LR": 1e-4,
    "WEIGHT_DECAY": 1e-5,
    "DECODER_MODE": "add",   # "add" | "concat"
    "PRETRAINED": True,
    "THRESHOLD": 0.5,
}

DATA_DIR = Path("../input/severstal-steel-defect-detection")
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_DIR = DATA_DIR / "train_images"
TEST_DIR  = DATA_DIR / "test_images"       # submission iÃ§in lazÄ±msa
WORK_DIR  = Path("/kaggle/working")



import pandas as pd

# CSV oku
df = pd.read_csv(TRAIN_CSV)
print("CSV boyutu:", df.shape)
print("Ä°lk 5 satÄ±r:")
df.head()



import os
import matplotlib.pyplot as plt

# TÃ¼m train klasÃ¶rÃ¼ndeki gÃ¶rseller
all_images = os.listdir("../input/severstal-steel-defect-detection/train_images")
all_images = [img for img in all_images if img.endswith(".jpg")]

# train.csv'deki defektli gÃ¶rseller
defect_images = df["ImageId"].unique().tolist()

# Defektsizleri bul
no_defect_images = set(all_images) - set(defect_images)

print(f"Toplam gÃ¶rÃ¼ntÃ¼: {len(all_images)}")
print(f"Defektli gÃ¶rÃ¼ntÃ¼: {len(defect_images)}")
print(f"Defektsiz gÃ¶rÃ¼ntÃ¼: {len(no_defect_images)}")

# Bar plot
plt.figure(figsize=(4,4))
plt.bar(["No Defect", "Has Defect"], [len(no_defect_images), len(defect_images)], color=["red", "green"])
plt.title("Defektli vs Defektsiz GÃ¶rÃ¼ntÃ¼ DaÄŸÄ±lÄ±mÄ±")
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# train.csv yÃ¼kle
df = pd.read_csv("../input/severstal-steel-defect-detection/train.csv")

# SÄ±nÄ±f bazÄ±nda en az bir kusuru olan gÃ¶rsellerin sayÄ±sÄ±
class_counts = df.loc[df["EncodedPixels"].notnull()] \
                 .groupby("ClassId")["ImageId"].nunique()

# Toplam gÃ¶rsel sayÄ±sÄ±
total_images = df["ImageId"].nunique()

# Grafik 1: Her sÄ±nÄ±fta kusurlu gÃ¶rsel sayÄ±sÄ±
plt.figure(figsize=(6,4))
class_counts.plot(kind="bar", color=["blue","orange","green","red"])
plt.title("Her SÄ±nÄ±fta Kusurlu GÃ¶rsel SayÄ±sÄ±")
plt.ylabel("GÃ¶rsel SayÄ±sÄ±")
plt.xlabel("Defekt SÄ±nÄ±fÄ± (1-4)")
plt.show()
# Oranlar (tÃ¼m dataset Ã¼zerinden)
class_ratios = class_counts / total_images * 100

# Grafik 2: Pie chart
plt.figure(figsize=(6,6))
plt.pie(class_ratios, labels=[f"Class {c}" for c in class_counts.index],
        autopct="%.2f%%", startangle=90, colors=["blue","orange","green","red"])
plt.title("TÃ¼m GÃ¶rseller Ä°Ã§inde SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ±")
plt.show()



print("ğŸ“Š SÄ±nÄ±f BazÄ±nda Kusurlu GÃ¶rsel SayÄ±larÄ±:")
print(class_counts)
print("\nğŸ“Š SÄ±nÄ±f BazÄ±nda Oranlar (%):")
print(class_ratios.round(2))

# GÃ¶rsel baÅŸÄ±na farklÄ± sÄ±nÄ±f sayÄ±sÄ±
defects_per_image = df.loc[df["EncodedPixels"].notnull()] \
                      .groupby("ImageId")["ClassId"].nunique()

# Histogram
plt.figure(figsize=(6,4))
defects_per_image.value_counts().sort_index().plot(kind="bar", color="purple")
plt.xlabel("Bir GÃ¶rselde KaÃ§ FarklÄ± Kusur Var")
plt.ylabel("GÃ¶rsel SayÄ±sÄ±")
plt.title("Tek vs Ã‡oklu Defekt DaÄŸÄ±lÄ±mÄ±")
plt.show()

print("ğŸ“Š Tek defektli gÃ¶rseller:", (defects_per_image==1).sum())
print("ğŸ“Š Ã‡oklu defektli gÃ¶rseller:", (defects_per_image>1).sum())

summary = pd.DataFrame({
    "Kusurlu GÃ¶rsel SayÄ±sÄ±": class_counts,
    "TÃ¼m GÃ¶rsellere OranÄ± (%)": class_ratios.round(2)
})
print(summary)



def rle_decode(mask_rle: str, shape=(CFG["HEIGHT"], CFG["WIDTH"])) -> np.ndarray:
    if not isinstance(mask_rle, str) or mask_rle.strip()=="":
        return np.zeros(shape, dtype=np.uint8)
    s = list(map(int, mask_rle.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.asarray(starts) - 1
    ends   = starts + np.asarray(lengths)
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")

def rle_encode(mask: np.ndarray) -> str:
    pixels = mask.flatten(order="F")
    pads = np.pad(pixels, (1,1), constant_values=0)
    changes = np.where(pads[1:] != pads[:-1])[0] + 1
    starts, ends = changes[::2], changes[1::2]
    lengths = ends - starts
    if len(starts)==0:
        return ""
    return " ".join(f"{s} {l}" for s,l in zip(starts, lengths))

def build_masks(image_id: str, df: pd.DataFrame,
                shape=(CFG["HEIGHT"], CFG["WIDTH"]),
                num_classes=CFG["NUM_CLASSES"]) -> np.ndarray:
    m = np.zeros((shape[0], shape[1], num_classes), dtype=np.uint8)
    for c in range(1, num_classes+1):
        rles = df.loc[(df["ImageId"]==image_id) & (df["ClassId"]==c), "EncodedPixels"]
        if rles.notnull().any():
            for rle in rles:
                m[..., c-1] |= rle_decode(rle, shape)
    return m



import cv2
import matplotlib.pyplot as plt

sample_images = df["ImageId"].drop_duplicates().sample(5, random_state=42).tolist()

for image_id in sample_images:
    img_path = TRAIN_DIR / image_id
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    masks = build_masks(image_id, df)

    plt.figure(figsize=(18,4))
    plt.subplot(1,6,1)
    plt.imshow(img)
    plt.title(f"{image_id}\nOriginal")
    plt.axis("off")

    # SÄ±nÄ±f maskeleri
    for i in range(4):
        plt.subplot(1,6,i+2)
        plt.imshow(masks[..., i], cmap="gray")
        plt.title(f"Class {i+1}")
        plt.axis("off")

    # Overlay
    overlay = img.copy()
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]  # R, G, B, SarÄ±
    for i in range(4):
        mask = masks[..., i].astype(bool)
        overlay[mask] = (0.7*overlay[mask] + 0.3*np.array(colors[i])).astype(np.uint8)

    plt.subplot(1,6,6)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.show()



def get_train_transforms(H=CFG["HEIGHT"], W=CFG["WIDTH"]):
    return A.Compose([
        A.CropNonEmptyMaskIfExists(height=H, width=H, p=0.5),  # isteÄŸe gÃ¶re
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9,1.1), translate_percent=(0.02,0.02), rotate=(-5,5), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.RandomBrightnessContrast(p=0.4),
        A.Resize(H, W, interpolation=1),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)

def get_valid_transforms(H=CFG["HEIGHT"], W=CFG["WIDTH"]):
    return A.Compose([
        A.Resize(H, W, interpolation=1),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])



import random

def visualize_random_aug(df, train_dir=TRAIN_DIR, transforms=get_train_transforms()):
    # Random bir gÃ¶rsel seÃ§
    image_id = random.choice(df["ImageId"].unique().tolist())
    img_path = train_dir / image_id
    
    # GÃ¶rseli oku
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Maske oluÅŸtur
    masks = build_masks(image_id, df)   # (H,W,4)
    
    # Albumentations ile augment et
    aug = transforms(image=img, masks=[masks[...,i] for i in range(4)])
    aug_img = aug["image"].permute(1,2,0).cpu().numpy()   # CHW -> HWC
    aug_masks = np.stack(aug["masks"], axis=-1)           # geri birleÅŸtir
    
    # Overlay fonksiyonu
    def overlay_image(base, masks, alpha=0.3):
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        overlay = base.copy()
        for i in range(masks.shape[-1]):
            mask = masks[..., i].astype(bool)
            overlay[mask] = (alpha*overlay[mask] + (1-alpha)*np.array(colors[i])).astype(np.uint8)
        return overlay
    
    orig_overlay = overlay_image(img, masks)
    aug_overlay  = overlay_image(aug_img, aug_masks)
    
    # Plot
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(orig_overlay)
    plt.title(f"Original Overlay\n{image_id}")
    plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.imshow(aug_overlay)
    plt.title("Augmented Overlay")
    plt.axis("off")
    
    plt.show()

# KullanÄ±m: her Ã§alÄ±ÅŸtÄ±rmada random bir Ã¶rnek gÃ¶sterir
visualize_random_aug(df)



class SteelDefectDataset(Dataset):
    def __init__(self, image_ids, image_dir: Path, df: pd.DataFrame, transforms=None, load_rgb=True):
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
        self.df = df
        self.transforms = transforms
        self.load_rgb = load_rgb

    def __len__(self):
        return len(self.image_ids)

    def _read_image(self, image_id):
        img = cv2.imread(str(self.image_dir / image_id), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image_id)
        if self.load_rgb:
            img = np.repeat(img[..., None], 3, axis=2)  # (H,W,3)
        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = self._read_image(image_id)
        mask = build_masks(image_id, self.df)  # (H,W,C)

        if self.transforms:
            out = self.transforms(image=img, mask=mask)
            img, mask = out["image"], out["mask"].permute(2,0,1)  # -> (C,H,W)
        else:
            img = torch.from_numpy(img.transpose(2,0,1)).float()
            mask = torch.from_numpy(mask.transpose(2,0,1)).float()

        meta = {"image_id": image_id}
        return img, mask, meta


# --- Collate function ---
def collate_fn(batch):
    images, masks, metas = zip(*batch)
    images = torch.stack(images)
    masks = torch.stack(masks)
    return images, masks, metas


# --- Dataset & DataLoader test ---
train_ids, valid_ids = train_test_split(
    df["ImageId"].unique(),
    test_size=CFG["VAL_SPLIT"],
    random_state=SEED
)

train_ds = SteelDefectDataset(train_ids, TRAIN_DIR, df, transforms=get_train_transforms())
valid_ds = SteelDefectDataset(valid_ids, TRAIN_DIR, df, transforms=get_valid_transforms())

train_loader = DataLoader(train_ds, batch_size=CFG["BATCH_SIZE"],
                          shuffle=True, num_workers=CFG["NUM_WORKERS"],
                          collate_fn=collate_fn)

valid_loader = DataLoader(valid_ds, batch_size=CFG["BATCH_SIZE"],
                          shuffle=False, num_workers=CFG["NUM_WORKERS"],
                          collate_fn=collate_fn)


# --- Test ---
batch = next(iter(train_loader))
images, masks, metas = batch

print("Images shape:", images.shape)   # (B, 3, H, W)
print("Masks shape:", masks.shape)     # (B, 4, H, W)
print("Meta Ã¶rnek:", metas[0])



all_ids = df["ImageId"].unique()
any_defect = df.groupby("ImageId")["EncodedPixels"].apply(lambda s: s.notnull().any()).astype(int)
train_ids, val_ids = train_test_split(all_ids, test_size=CFG["VAL_SPLIT"],
                                      random_state=SEED, stratify=any_defect.loc[all_ids])

train_ds = SteelDefectDataset(train_ids, TRAIN_DIR, df, transforms=get_train_transforms())
val_ds   = SteelDefectDataset(val_ids,   TRAIN_DIR, df, transforms=get_valid_transforms())

train_loader = DataLoader(train_ds, batch_size=CFG["BATCH_SIZE"], shuffle=True,
                          num_workers=CFG["NUM_WORKERS"], pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=CFG["BATCH_SIZE"], shuffle=False,
                          num_workers=CFG["NUM_WORKERS"], pin_memory=True)



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class UNetResNet18(nn.Module):
    def __init__(self, num_classes=CFG["NUM_CLASSES"], pretrained=True, decoder_mode="add", dropout=0.0):
        super().__init__()
        # Encoder backbone
        base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        del base.fc, base.avgpool  # UNet'te kullanÄ±lmÄ±yor

        self.enc1 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.enc2 = nn.Sequential(base.maxpool, base.layer1)
        self.enc3 = base.layer2
        self.enc4 = base.layer3
        self.enc5 = base.layer4

        self.mode = decoder_mode

        def up_block(in_ch, out_ch, use_concat=False):
            layers = [nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
                      nn.BatchNorm2d(out_ch),
                      nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            if use_concat:
                layers += [nn.Conv2d(out_ch*2, out_ch, 3, padding=1),
                           nn.BatchNorm2d(out_ch),
                           nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        if self.mode == "add":
            self.up4 = up_block(512,256)
            self.up3 = up_block(256,128)
            self.up2 = up_block(128,64)
            self.up1 = up_block(64,64)
        else:  # concat
            self.up4 = up_block(512,256, use_concat=True)
            self.up3 = up_block(256,128, use_concat=True)
            self.up2 = up_block(128,64, use_concat=True)
            self.up1 = up_block(64,64, use_concat=True)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder
        if self.mode == "add":
            d4 = self.up4(e5) + e4
            d3 = self.up3(d4) + e3
            d2 = self.up2(d3) + e2
            d1 = self.up1(d2) + e1
        else:  # concat
            d4 = self.up4(torch.cat([F.interpolate(e5, size=e4.shape[2:], mode="bilinear", align_corners=False), e4],1))
            d3 = self.up3(torch.cat([F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False), e3],1))
            d2 = self.up2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False), e2],1))
            d1 = self.up1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False), e1],1))

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out



class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.smooth=smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p = probs.view(probs.size(0), probs.size(1), -1)
        t = targets.view(targets.size(0), targets.size(1), -1)
        inter = (p*t).sum(2); den = p.sum(2)+t.sum(2)
        dice = (2*inter + self.smooth) / (den + self.smooth)
        return 1.0 - dice.mean()

class WeightedFocalDiceLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, lam_f=0.7, lam_d=0.3):
        super().__init__()
        self.w = class_weights
        self.g = gamma
        self.lf = lam_f
        self.ld = lam_d
        self.dice = DiceLoss()

    def forward(self, logits, targets, return_details=False):
        probs = torch.sigmoid(logits)
        eps = 1e-8

        # BCE
        bce = -(targets * torch.log(probs + eps) +
                (1 - targets) * torch.log(1 - probs + eps))

        # Focal modÃ¼lasyonu
        pt = torch.where(targets == 1, probs, 1 - probs)
        fw = (1 - pt) ** self.g

        if self.w is not None:
            fw = fw * logits.new_tensor(self.w).view(1, -1, 1, 1)

        focal = (fw * bce).mean()
        dice = self.dice(logits, targets)

        loss = self.lf * focal + self.ld * dice

        if return_details:
            return loss, {"focal": focal.item(), "dice": dice.item()}
        return loss

@torch.no_grad()
def dice_coefficient(logits, targets, thr=CFG["THRESHOLD"], eps=1e-6):
    probs = torch.sigmoid(logits); preds = (probs>thr).float()
    inter = (preds*targets).sum((2,3)); den = preds.sum((2,3))+targets.sum((2,3))
    dice = (2*inter+eps)/(den+eps)
    return dice.mean()

@torch.no_grad()
def dice_per_class(logits, targets, thr=CFG["THRESHOLD"], eps=1e-6):
    probs = torch.sigmoid(logits); preds = (probs>thr).float()
    inter = (preds*targets).sum((0,2,3)); den = preds.sum((0,2,3))+targets.sum((0,2,3))
    return ((2*inter+eps)/(den+eps)).cpu().tolist()



from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    run_loss, run_dice, n = 0.0, 0.0, 0
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        bs = imgs.size(0); n += bs
        run_loss += loss.item() * bs
        run_dice += dice_coefficient(out, masks).item() * bs
    return {"loss": run_loss/n, "dice": run_dice/n}


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    run_loss, run_dice, n = 0.0, 0.0, 0
    for imgs, masks, _ in tqdm(loader, desc="Valid", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(imgs)
        loss = criterion(out, masks)
        bs = imgs.size(0); n += bs
        run_loss += loss.item() * bs
        run_dice += dice_coefficient(out, masks).item() * bs
    return {"loss": run_loss/n, "dice": run_dice/n}


class EarlyStopping:
    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.count = 0
        self.stop = False

    def __call__(self, score):
        if self.best is None:
            self.best = score; self.count = 0
        else:
            improve = (score > self.best) if self.mode=="max" else (score < self.best)
            if improve:
                self.best = score; self.count = 0
            else:
                self.count += 1
                if self.count >= self.patience:
                    self.stop = True


def fit(model, train_loader, val_loader, optimizer, scheduler, criterion, device,
        num_epochs=CFG["EPOCHS"], early_stopping_patience=5, save_path="best_model.pth"):
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")
    best_dice = -1.0

    history = {"train": [], "valid": []}

    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = validate_one_epoch(model, val_loader, criterion, device)

        # per-class dice ekle
        all_val_dice = []
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            all_val_dice.append(dice_per_class(out, masks))
        val_metrics["per_class_dice"] = np.mean(all_val_dice, axis=0).tolist()

        scheduler.step(val_metrics["dice"])

        print(f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"Valid Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")

        history["train"].append({**train_metrics, "lr": optimizer.param_groups[0]["lr"]})
        history["valid"].append(val_metrics)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), save_path)
            print(f"âœ… Best model saved at epoch {epoch} (dice={best_dice:.4f})")

        early_stopping(val_metrics["dice"])
        if early_stopping.stop:
            print("â�¹ï¸� Early stopping triggered.")
            break

    print(f"Training finished. Best Dice = {best_dice:.4f}")
    return history



# Model, loss, optimizer, scheduler tanÄ±mla
model = UNetResNet18(num_classes=CFG["NUM_CLASSES"],
                     pretrained=CFG["PRETRAINED"], 
                     decoder_mode=CFG["DECODER_MODE"]).to(device)

criterion = WeightedFocalDiceLoss(class_weights=[0.12,0.03,0.72,0.11])
optimizer = Adam(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("GPU bulunamadÄ±, CPUâ€™da Ã§alÄ±ÅŸÄ±yor.")


# EÄŸitimi baÅŸlat ve history yakala
history = fit(model, train_loader, val_loader, optimizer, scheduler, criterion, device,
              num_epochs=CFG["EPOCHS"], early_stopping_patience=5, save_path="best_model.pth")

# EÄŸitim sonunda son epochâ€™u da kaydet
torch.save(model.state_dict(), "last_model.pth")

# EÄŸitim grafikleri
plot_history(history)

# En iyi modeli tekrar yÃ¼klemek iÃ§in yol
best_path = "best_model.pth"



class SteelDefectTestDataset(Dataset):
    def __init__(self, image_ids, image_dir: Path, transforms=None, load_rgb=True):
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
        self.transforms = transforms
        self.load_rgb = load_rgb

    def __len__(self): return len(self.image_ids)

    def _read_image(self, image_id):
        img = cv2.imread(str(self.image_dir / image_id), cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(image_id)
        if self.load_rgb: img = np.repeat(img[...,None], 3, axis=2)
        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = self._read_image(image_id)
        if self.transforms:
            out = self.transforms(image=img)
            img = out["image"]
        else:
            img = torch.from_numpy(img.transpose(2,0,1)).float()
        return img, image_id


# Test transform (sadece normalize & resize)
test_transforms = A.Compose([
    A.Resize(CFG["HEIGHT"], CFG["WIDTH"], interpolation=1),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

# Test dataset & loader
test_ids = os.listdir(TEST_DIR)
test_ids = [img for img in test_ids if img.endswith(".jpg")]

test_ds = SteelDefectTestDataset(test_ids, TEST_DIR, transforms=test_transforms)
test_loader = DataLoader(test_ds, batch_size=CFG["BATCH_SIZE"],
                         shuffle=False, num_workers=CFG["NUM_WORKERS"], pin_memory=True)


# Model yÃ¼kle
model = UNetResNet18(num_classes=CFG["NUM_CLASSES"],
                     pretrained=False, decoder_mode=CFG["DECODER_MODE"])
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device).eval()



def plot_history(history, out_dir=WORK_DIR):
    tr_d = [e["dice"] for e in history["train"]]
    va_d = [e["dice"] for e in history["valid"]]
    tr_l = [e["loss"] for e in history["train"]]
    va_l = [e["loss"] for e in history["valid"]]
    lr   = [e["lr"] for e in history["train"]]

    plt.figure(); plt.plot(tr_d,label="Train"); plt.plot(va_d,label="Valid")
    plt.title("Dice"); plt.legend(); plt.savefig(out_dir/"dice_curve.png"); plt.close()

    plt.figure(); plt.plot(tr_l,label="Train"); plt.plot(va_l,label="Valid")
    plt.title("Loss"); plt.legend(); plt.savefig(out_dir/"loss_curve.png"); plt.close()

    plt.figure(); plt.plot(lr,label="LR"); plt.title("LR"); plt.legend()
    plt.savefig(out_dir/"lr_curve.png"); plt.close()



sub = []

with torch.no_grad():
    for imgs, image_ids in tqdm(test_loader, desc="Inference"):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > CFG["THRESHOLD"]).float().cpu().numpy()

        # (B, C, H, W) -> her image, her class iÃ§in RLE encode
        for i in range(preds.shape[0]):
            image_id = image_ids[i]
            for c in range(CFG["NUM_CLASSES"]):
                mask = preds[i, c]
                # geri H,W formatÄ±na Ã§evrilmiÅŸ mask
                mask = (mask > 0.5).astype(np.uint8)
                rle = rle_encode(mask)
                sub.append([image_id, c+1, rle])

# DataFrame oluÅŸtur
sub_df = pd.DataFrame(sub, columns=["ImageId","ClassId","EncodedPixels"])

# Kaggle formatÄ±: boÅŸ maskeler iÃ§in NaN olmalÄ±
sub_df.loc[sub_df["EncodedPixels"]=="", "EncodedPixels"] = np.nan

# Kaydet
sub_df.to_csv("submission.csv", index=False)
print("âœ… submission.csv hazÄ±r:", sub_df.shape)



@torch.no_grad()
def visualize_samples(model, dataset, k=3, thr=CFG["THRESHOLD"]):
    model.eval()
    idxs = np.random.choice(len(dataset), size=min(k, len(dataset)), replace=False)
    for i in idxs:
        img, mask, meta = dataset[i]   # Dataset __getitem__ -> (image, mask, meta)
        x = img.unsqueeze(0).to(device)
        pred = torch.sigmoid(model(x))[0].cpu().numpy().transpose(1,2,0)
        pred_bin = (pred > thr).astype(np.uint8)

        img_np  = img.permute(1,2,0).numpy()
        mask_np = mask.permute(1,2,0).numpy()

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8))
        axs[0].set_title(meta["image_id"]); axs[0].axis("off")
        axs[1].imshow(mask_np.max(-1), cmap="gray"); axs[1].set_title("GT any"); axs[1].axis("off")
        axs[2].imshow(pred_bin.max(-1), cmap="gray"); axs[2].set_title("Pred any"); axs[2].axis("off")
        plt.show()

# En iyi kaydedilmiÅŸ modeli yÃ¼kle
model.load_state_dict(torch.load(best_path, map_location=device))
visualize_samples(model, val_ds, k=3)





