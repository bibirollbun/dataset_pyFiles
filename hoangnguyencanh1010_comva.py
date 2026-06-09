import os, zipfile

zip_path = "/kaggle/input/data-science-bowl-2018/stage1_train.zip"
extract_path = "/kaggle/working/stage1_train"

# Náº¿u Ä‘Ã£ giáº£i nÃ©n rá»“i thÃ¬ khÃ´ng giáº£i nÃ©n láº¡i ná»¯a
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("âœ… Ä�Ã£ giáº£i nÃ©n xong!")
else:
    print("ğŸ“‚ ThÆ° má»¥c Ä‘Ã£ tá»“n táº¡i, bá»� qua bÆ°á»›c giáº£i nÃ©n.")

print("Sá»‘ folder train:", len(os.listdir(extract_path)))
print("VÃ­ dá»¥ vÃ i ID:", os.listdir(extract_path)[:5])



import os
os.listdir("/kaggle/working")





import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"
device



IMG_SIZE = 256

class NucleiDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.ids = os.listdir(root)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        base = os.path.join(self.root, img_id)

        # 1. Ä�á»�c áº£nh RGB
        img_dir = os.path.join(base, "images")
        img_name = os.listdir(img_dir)[0]
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Gá»™p táº¥t cáº£ mask thÃ nh 1 mask nhá»‹ phÃ¢n
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask_dir = os.path.join(base, "masks")
        if os.path.exists(mask_dir):
            for m_name in os.listdir(mask_dir):
                m_path = os.path.join(mask_dir, m_name)
                m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                mask = np.maximum(mask, m)

        mask = (mask > 0).astype(np.uint8)  # 0/1

        # 3. Resize
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask  = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # 4. Ã�p dá»¥ng transform (augment + normalize)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        else:
            image = torch.tensor(image.transpose(2,0,1), dtype=torch.float32) / 255.
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask



import os
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2()
])

val_tf = A.Compose([
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2()
])

DATA_ROOT = "/kaggle/working/stage1_train"

full_ds = NucleiDataset(DATA_ROOT)  # táº¡m chÆ°a gÃ¡n transform
val_size = int(0.2 * len(full_ds))
train_size = len(full_ds) - val_size

train_ds, val_ds = random_split(full_ds, [train_size, val_size])

# gÃ¡n transform cho 2 táº­p
train_ds.dataset.transform = train_tf
val_ds.dataset.transform   = val_tf

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)

print("Sá»‘ áº£nh train:", len(train_ds))
print("Sá»‘ áº£nh val:", len(val_ds))



import matplotlib.pyplot as plt

img, mask = train_ds[0]   # img: tensor (3,H,W), mask: (H,W) náº¿u dÃ¹ng ToTensorV2

# chuyá»ƒn áº£nh vá»� dáº¡ng hiá»ƒn thá»‹ Ä‘Æ°á»£c
if isinstance(img, torch.Tensor):
    img_show = img.permute(1,2,0).cpu().numpy()
    img_show = (img_show * 0.5 + 0.5).clip(0,1)  # náº¿u dÃ¹ng normalize 0.5
else:
    img_show = img

if isinstance(mask, torch.Tensor):
    mask_show = mask.cpu().numpy()
    if mask_show.ndim == 3:
        mask_show = mask_show[0]

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(img_show); plt.title("áº¢nh gá»‘c"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(mask_show, cmap="gray"); plt.title("Mask gá»™p"); plt.axis("off")
plt.show()



import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv3(u3)

        return self.outc(u3)



#loss + Dice
bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum()
    return 1 - (2*inter + smooth)/(pred.sum()+target.sum()+smooth)

def dice_coef(pred, target, smooth=1e-7):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    return ((2*inter + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)).mean()

def iou_score(pred, target, smooth=1e-7):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    return ((inter + smooth) / (union + smooth)).mean()




def train_one_model(model, train_loader, val_loader, epochs=5, lr=1e-4):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            masks = masks.float()

            opt.zero_grad()
            logits = model(imgs)
            loss   = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        # ğŸ”¹ tÃ­nh Dice & IoU trÃªn val
        model.eval()
        val_dice, val_iou, n = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device)
                masks = masks.to(device)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                masks = masks.float()
                logits = model(imgs)
                val_dice += dice_coef(logits, masks).item() * imgs.size(0)
                val_iou  += iou_score(logits, masks).item() * imgs.size(0)
                n += imgs.size(0)

        val_dice /= n
        val_iou  /= n

        print(f"Epoch {ep+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} "
              f"| Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

    return model



unet = UNet()
unet = train_one_model(unet, train_loader, val_loader, epochs=10)

def show_pred(model, ds, idx=0):
    model.eval()
    img, m = ds[idx]
    img_t = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_t))[0,0].cpu().numpy()

    pred_bin = (pred > 0.5).astype("uint8")

    img_np = img.permute(1,2,0).cpu().numpy()
    img_np = (img_np * 0.5 + 0.5).clip(0,1)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img_np); plt.title("Original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(m.squeeze().cpu(), cmap="gray"); plt.title("GT Mask"); plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(img_np)
    plt.imshow(pred_bin, cmap="jet", alpha=0.4)
    plt.title("Prediction"); plt.axis("off")
    plt.show()

show_pred(unet, val_ds, 0)



