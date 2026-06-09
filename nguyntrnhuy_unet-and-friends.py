import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import cv2
import os
import glob
import pandas as pd
import imagehash
from PIL import Image
from sklearn.model_selection import train_test_split



class SegmentationDataset(Dataset):
    def __init__(self, imgs, masks, transform=None):
        self.imgs = imgs
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = cv2.imread(self.imgs[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[idx], 0)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"].unsqueeze(0)
        else:
            image = torch.from_numpy(image).permute(2,0,1).float() / 255.
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask



train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=40, p=0.5),

    A.OneOf([
        A.RandomBrightnessContrast(0.2, 0.2),
        A.GaussNoise(std_range=(0.01, 0.03)),
        A.CLAHE(clip_limit=4.0),
    ], p=0.5),

    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2(),
])


def analyze_and_clean_dataset(root_dir, threshold=5):
    mask_paths = glob.glob(os.path.join(root_dir, "*_mask.tif"))
    img_paths = [p.replace("_mask.tif", ".tif") for p in mask_paths]

    hashes = []
    for img, msk in zip(img_paths, mask_paths):
        try:
            h = imagehash.phash(Image.open(img))
            hashes.append((h, img, msk))
        except:
            pass

    used = set()
    final_imgs, final_masks = [], []

    for i in range(len(hashes)):
        if i in used:
            continue

        h1, img1, msk1 = hashes[i]
        group = [(img1, msk1)]
        used.add(i)

        for j in range(i+1, len(hashes)):
            if j in used:
                continue
            h2, img2, msk2 = hashes[j]
            if h1 - h2 <= threshold:
                group.append((img2, msk2))
                used.add(j)

        if len(group) == 1:
            final_imgs.append(img1)
            final_masks.append(msk1)
        else:
            masks = [cv2.imread(m, 0) for _, m in group]
            avg = np.mean(masks, axis=0)
            merged = (avg > 0.5).astype(np.uint8)

            save_mask = group[0][1]
            cv2.imwrite(save_mask, merged * 255)

            final_imgs.append(group[0][0])
            final_masks.append(save_mask)

    return final_imgs, final_masks



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetIN(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        base = 32

        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            ConvBlock(base*8, base*16),
            nn.Dropout2d(p=0.3)
        )
        self.dec4 = DecoderBlock(base*16, base*8, base*8)
        self.dec3 = DecoderBlock(base*8, base*4, base*4)
        self.dec2 = DecoderBlock(base*4, base*2, base*2)
        self.dec1 = DecoderBlock(base*2, base, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b  = self.bottleneck(self.pool(s4))

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out(d1)


def up_align(src, target):
    src = F.interpolate(
        src,
        size=target.shape[2:],
        mode="bilinear",
        align_corners=False
    )
    return src


class UNetPP(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.conv00 = ConvBlock(in_ch, base)
        self.conv10 = ConvBlock(base, base*2)
        self.conv20 = ConvBlock(base*2, base*4)
        self.conv30 = ConvBlock(base*4, base*8)
        self.conv40 = nn.Sequential(
            ConvBlock(base*8, base*16),
            nn.Dropout2d(0.3)
        )

        # Decoder
        self.conv01 = ConvBlock(base*3, base)
        self.conv11 = ConvBlock(base*6, base*2)
        self.conv21 = ConvBlock(base*12, base*4)
        self.conv31 = ConvBlock(base*24, base*8)

        self.conv02 = ConvBlock(base*4, base)
        self.conv12 = ConvBlock(base*8, base*2)
        self.conv22 = ConvBlock(base*16, base*4)

        self.conv03 = ConvBlock(base*5, base)
        self.conv13 = ConvBlock(base*10, base*2)

        self.conv04 = ConvBlock(base*6, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))

        x01 = self.conv01(torch.cat([x00, up_align(x10, x00)], 1))
        x11 = self.conv11(torch.cat([x10, up_align(x20, x10)], 1))
        x21 = self.conv21(torch.cat([x20, up_align(x30, x20)], 1))
        x31 = self.conv31(torch.cat([x30, up_align(x40, x30)], 1))

        x02 = self.conv02(torch.cat([x00, x01, up_align(x11, x00)], 1))
        x12 = self.conv12(torch.cat([x10, x11, up_align(x21, x10)], 1))
        x22 = self.conv22(torch.cat([x20, x21, up_align(x31, x20)], 1))

        x03 = self.conv03(torch.cat([x00, x01, x02, up_align(x12, x00)], 1))
        x13 = self.conv13(torch.cat([x10, x11, x12, up_align(x22, x10)], 1))

        x04 = self.conv04(torch.cat([x00, x01, x02, x03, up_align(x13, x00)], 1))

        return self.out(x04)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, p, t):
        p = torch.sigmoid(p)
        tp = (p * t).sum(dim=(1,2,3))
        fn = ((1-p) * t).sum(dim=(1,2,3))
        fp = (p * (1-t)).sum(dim=(1,2,3))
        tversky = (tp + self.smooth) / (tp + self.alpha*fn + self.beta*fp + self.smooth)
        return 1 - tversky.mean()


loss_fn = lambda p,t: F.binary_cross_entropy_with_logits(p,t) + TverskyLoss()(p,t)


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total = 0

    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()

        with autocast("cuda"):
            pred = model(img)
            loss = loss_fn(pred, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()

    return total / len(loader)


def validate(model, loader, device):
    model.eval()
    dice_sum = 0

    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            pred = torch.sigmoid(model(img))

            inter = (pred * mask).sum(dim=(1,2,3))
            union = pred.sum(dim=(1,2,3)) + mask.sum(dim=(1,2,3))
            dice = (2*inter + 1e-6) / (union + 1e-6)

            dice_sum += dice.mean().item()

    return dice_sum / len(loader)



device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetIN().to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-4)
scaler = GradScaler()
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)


root_dir = "/kaggle/input/ultrasound-nerve-segmentation/train"
imgs, masks = analyze_and_clean_dataset(root_dir)

train_i, val_i, train_m, val_m = train_test_split(
    imgs, masks, test_size=0.2, random_state=42
)

train_ds = SegmentationDataset(train_i, train_m, train_transform)
val_ds   = SegmentationDataset(val_i, val_m, val_transform)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

print(f"ğŸ“¦ Tá»•ng sá»‘ áº£nh sau clean: {len(imgs)}")
print(f"ğŸŸ¢ Train images: {len(train_ds)}")
print(f"ğŸ”µ Val images:   {len(val_ds)}")


import matplotlib.pyplot as plt

best = 0
wait = 0
patience = 15

loss_hist = []
dice_hist = []

for epoch in range(120):
    loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
    dice = validate(model, val_loader, device)

    scheduler.step(1 - dice)

    loss_hist.append(loss)
    dice_hist.append(dice)

    if dice > best:
        best = dice
        wait = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Epoch {epoch+1:03d} | Loss {loss:.4f} | Dice {dice:.4f}")

# ============================
# Plot loss & dice
# ============================
epochs = range(1, len(loss_hist) + 1)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(epochs, loss_hist)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(epochs, dice_hist)
plt.xlabel("Epoch")
plt.ylabel("Dice score")
plt.title("Validation Dice")
plt.grid(True)

plt.tight_layout()
plt.show()



from scipy.ndimage import binary_fill_holes
class TestDataset(Dataset):
    def __init__(self, root, transform):
        self.paths = sorted(glob.glob(os.path.join(root, "*.tif")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img_id = int(os.path.splitext(os.path.basename(path))[0])

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]

        return img, img_id


def rle_encode(mask):
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


def remove_small_objects(mask, min_size=100):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 1
    return cleaned

def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = np.zeros_like(mask)
    cleaned[labels == largest_label] = 1
    return cleaned

def fill_holes(mask):
    return binary_fill_holes(mask).astype(np.uint8)


# ==== Load model ====
model.eval()

# ==== Test loader ====
test_ds = TestDataset(
    "/kaggle/input/ultrasound-nerve-segmentation/test",
    val_transform
)

test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False
)

best_thres = 0.9
results = []

with torch.no_grad():
    for img, img_id in test_loader:
        img = img.to(device)

        # --- Predict ---
        prob = torch.sigmoid(model(img))[0, 0].cpu().numpy()

        # --- Resize vá»� size gá»‘c ---
        prob = cv2.resize(prob, (580, 420), interpolation=cv2.INTER_LINEAR)

        # --- Threshold ---
        mask = (prob > best_thres).astype(np.uint8)

        # --- Post-process ---
        mask = remove_small_objects(mask, min_size=100)
        mask = keep_largest_component(mask)
        mask = fill_holes(mask)

        # --- Encode ---
        if mask.sum() == 0:
            rle = ""
        else:
            rle = rle_encode(mask)

        results.append({
            "img": int(img_id.item()),
            "pixels": rle
        })

# ==== Save submission ====
submission = pd.DataFrame(results).sort_values("img")
submission.to_csv("submission.csv", index=False)

print("âœ… submission.csv created (Kaggle-ready)")


