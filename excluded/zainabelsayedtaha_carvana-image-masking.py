# Import packadges
import os
import numpy as np
import pandas as pd
import zipfile
from glob import glob
from os.path import basename, splitext
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from torchvision import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Read metadata
zip_meta_path = '/kaggle/input/carvana-image-masking-challenge/metadata.csv.zip'

with zipfile.ZipFile(zip_meta_path, 'r') as zip_ref:
    csv_name = zip_ref.namelist()[0]
    with zip_ref.open(csv_name) as csv_file:
        metadata_df = pd.read_csv(csv_file)


print(metadata_df.head())


print(metadata_df.info())


duplicates = metadata_df[metadata_df.duplicated()]

print("Duplicate rows:")
print(duplicates)


null_rows = metadata_df[metadata_df.isnull().any(axis=1)]

print("Rows with null values:")
print(null_rows)


all_null_rows = metadata_df[metadata_df.isnull().all(axis=1)]
print(all_null_rows)


# Dealing with nulls: fill numeric nulls with median and text nulls with 'Unknown'
metadata_df['year'].fillna(metadata_df['year'].median(), inplace=True)
metadata_df[['make', 'model', 'trim1', 'trim2']] = \
    metadata_df[['make', 'model', 'trim1', 'trim2']].fillna('Unknown')


for col in metadata_df.columns:
    if col != 'id':
        print(f"{col}: {metadata_df[col].unique()}")
        print("-" * 40)


plt.figure(figsize=(10, 5))
sns.countplot(data=metadata_df, x='year')

plt.title('Count of Records by Year')
plt.xticks(rotation=45)
plt.show()


# Top 3 company cars samples from 2012 to 2016

metadata_df['year'] = metadata_df['year'].astype(int)

filtered_df = metadata_df[(metadata_df['year'] >= 2012) & (metadata_df['year'] <= 2016)]

top_3_makes = filtered_df['make'].value_counts().head(3).index

filtered_df = filtered_df[filtered_df['make'].isin(top_3_makes)]

counts = filtered_df.groupby(['year', 'make']).size().reset_index(name='count')

plt.figure(figsize=(10, 6))
sns.lineplot(data=counts, x='year', y='count', hue='make', marker='o')

plt.title('Top 3 Cars Companies (2012–2016)')
plt.xticks(range(2012, 2017))
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# Get top 5 models for each make
top_models_per_make = (
    filtered_df.groupby('make')['model']
    .value_counts()
    .groupby(level=0)
    .head(5)
    .reset_index(name='count')
)


# Plot top 5 models for each make

plt.figure(figsize=(12, 6))
sns.barplot(data=top_models_per_make, x='model', y='count', hue='make')

plt.title('Top 5 Models for Top 3 Makes')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.tight_layout()
plt.show()


train_zip = '/kaggle/input/carvana-image-masking-challenge/train.zip'
test_zip = '/kaggle/input/carvana-image-masking-challenge/test.zip'
mask_zip = '/kaggle/input/carvana-image-masking-challenge/train_masks.zip'


# Unzip to clean folders 
def unzip_to_folder(zip_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

train_zip = '/kaggle/input/carvana-image-masking-challenge/train.zip'
test_zip = '/kaggle/input/carvana-image-masking-challenge/test.zip'
mask_zip = '/kaggle/input/carvana-image-masking-challenge/train_masks.zip'

unzip_to_folder(train_zip, "train")
unzip_to_folder(mask_zip, "train_masks")
unzip_to_folder(test_zip, "test")

# Load all file paths 
train_images = sorted(glob("train/**/*.*", recursive=True))
mask_images = sorted(glob("train_masks/**/*.*", recursive=True))

# Create a matching function 
def normalize_name(path):
    """
    Extracts the base name without extension and removes any '_mask' suffix.
    """
    name = splitext(basename(path))[0]
    name = name.replace("_mask", "")  # remove '_mask' if exists
    return name

# Build DataFrame for train images 
df = pd.DataFrame({
    "image_path": train_images
})
df["key_id"] = df["image_path"].apply(normalize_name)

# Build a lookup dict for masks 
mask_lookup = {normalize_name(p): p for p in mask_images}

# Match images to masks 
df["mask_path"] = df["key_id"].map(mask_lookup)

# Check results 
missing_masks = df[df["mask_path"].isnull()]
if not missing_masks.empty:
    print(f"⚠ Warning: {len(missing_masks)} images do not have a matching mask!")
    print(missing_masks.head())

print(f"✅ Total images: {len(df)}, Matched masks: {df['mask_path'].notnull().sum()}")
print(df.sample(5))


# Display 3 images and their masks
def visualize_samples(df, n=3):
    samples = df.sample(n)
    plt.figure(figsize=(10, n * 3))

    for i, (_, row) in enumerate(samples.iterrows()):
        img = cv2.imread(row['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = Image.open(row['mask_path'])
        mask = np.array(mask)

        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(n, 2, 2*i + 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

visualize_samples(df, n=3)


# Count images in each file
train_dir = "train/train"
mask_dir = "train_masks/train_masks"
test_dir = "test/test"

# List files
train_images = sorted(os.listdir(train_dir))
train_masks = sorted(os.listdir(mask_dir))
test_images = sorted(os.listdir(test_dir))

# Counts
print(f"Train images: {len(train_images)}")
print(f"Train masks:  {len(train_masks)}")
print(f"Test images:  {len(test_images)}")

# Check matching
train_keys = [os.path.splitext(img)[0] for img in train_images]
mask_keys = [os.path.splitext(mask)[0].replace("_mask", "") for mask in train_masks]

if set(train_keys) == set(mask_keys):
    print("✅ All train images have matching masks.")
else:
    missing_masks = set(train_keys) - set(mask_keys)
    missing_images = set(mask_keys) - set(train_keys)
    if missing_masks:
        print(f"Missing masks for: {missing_masks}")
    if missing_images:
        print(f"Missing train images for: {missing_images}")


print(df.columns)


DATA_DIR = '/kaggle/working/'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train/train')
TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'train_masks/train_masks')

train_images = sorted(glob(os.path.join(TRAIN_IMG_DIR, '*.jpg')) + 
                      glob(os.path.join(TRAIN_IMG_DIR, '*.png')))

train_masks = sorted(glob(os.path.join(TRAIN_MASK_DIR, '*.gif')))

print(f"num of images: {len(train_images)}")
print(f"num of masks: {len(train_masks)}")


img = cv2.imread(train_images[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask_gif = Image.open(train_masks[0])
mask_gif = mask_gif.convert("L")  
mask = cv2.resize(np.array(mask_gif), (img.shape[1], img.shape[0]))

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Image")
plt.axis(False)

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title("Mask")
plt.axis(False)
plt.show()


DATA_DIR = '/kaggle/working/'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train/train')
TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'train_masks/train_masks')

img_paths = sorted(glob(os.path.join(TRAIN_IMG_DIR, '*.jpg')) +
                   glob(os.path.join(TRAIN_IMG_DIR, '*.png')))
mask_paths = sorted(glob(os.path.join(TRAIN_MASK_DIR, '*.gif')))

print(f"num images: {len(img_paths)}, num masks: {len(mask_paths)}")


# hyperparams
IMG_SIZE = (128, 128)     
BATCH_SIZE = 2             
NUM_EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=(128,128)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)

        mask_gif = Image.open(self.mask_paths[idx]).convert("L")
        mask = np.array(mask_gif)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        img_t = self.to_tensor(img)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t


# U-Net model 

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32,64,128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder
        rev_f = features[::-1]
        for f in rev_f:
            self.ups.append(nn.ConvTranspose2d(f*2 if f!=rev_f[0] else features[-1]*2, f, 2, stride=2))
            self.ups.append(DoubleConv(f*2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i+1](x)

        return self.final_conv(x)


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(1)
    return 1 - ((2.*intersection + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)).mean()

def iou_score(pred, target, thr=0.5, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > thr).float()
    intersect = (pred * target).sum((1,2,3))
    union = ((pred + target) > 0).float().sum((1,2,3))
    return ((intersect + eps) / (union + eps)).mean().item()

bce = nn.BCEWithLogitsLoss()
def combined_loss(pred, target, alpha=0.5):
    return alpha*bce(pred, target) + (1-alpha)*dice_loss(pred, target)


# Train test split
IMG_SIZE = (128,128)
BATCH_SIZE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_imgs, val_imgs, train_masks_p, val_masks_p = train_test_split(
    img_paths, mask_paths, test_size=0.15, random_state=42)

train_loader = DataLoader(SegmentationDataset(train_imgs, train_masks_p, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(SegmentationDataset(val_imgs, val_masks_p, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)


model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)


# Training
train_losses, val_losses, val_ious = [], [], []
NUM_EPOCHS = 15  

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss = combined_loss(preds, masks, alpha=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss, ious = 0.0, []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss = combined_loss(preds, masks, alpha=0.5)
            val_loss += loss.item() * imgs.size(0)
            ious.append(iou_score(preds, masks))
    val_loss /= len(val_loader.dataset)
    mean_iou = np.mean(ious)

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    val_ious.append(mean_iou)

    scheduler.step(val_loss)

    print(f"Epoch [{epoch}/{NUM_EPOCHS}] Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {mean_iou:.4f}")


# Loss & Val IoU
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1,2,2)
plt.plot(val_ious, label='Val IoU', color='green')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.title('IoU over Epochs')
plt.show()


# Classification Report & Confusion Matrix
all_preds, all_targets = [], []
model.eval()
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_targets.append(masks.cpu().numpy())

all_preds = np.concatenate(all_preds).reshape(-1)
all_targets = np.concatenate(all_targets).reshape(-1)

cm = confusion_matrix(all_targets, all_preds)
print("\nClassification Report:\n", classification_report(all_targets, all_preds, digits=4))

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Background (0)", "Object (1)"],
            yticklabels=["Background (0)", "Object (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# Prediction & Visualization
def predict_and_visualize(model, dataset, num_samples=3):
    model.eval()
    plt.figure(figsize=(12, num_samples*4))
    for i in range(num_samples):
        img, mask = dataset[i]
        img_input = img.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(img_input)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()
        pred_np = pred.squeeze().cpu().numpy()

        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img_np)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(mask_np, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(pred_np, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

predict_and_visualize(model, SegmentationDataset(val_imgs, val_masks_p, IMG_SIZE), num_samples=3)


def apply_colormap(mask, color=(255, 0, 0)):
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[mask == 1] = color
    return mask_rgb

def overlay_mask(image, mask_rgb, alpha=0.5):
    overlay = image.copy()
    overlay = (overlay * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return overlay

def predict_and_visualize(model, dataset, num_samples=3):
    model.eval()
    plt.figure(figsize=(12, num_samples * 4))

    for i in range(num_samples):
        img, mask = dataset[i]
        img_input = img.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(img_input)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_np = mask.squeeze().numpy().astype(np.uint8)
        pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)

        mask_color = apply_colormap(mask_np, color=(0, 255, 0))
        pred_color = apply_colormap(pred_np, color=(255, 0, 0))  

        img_mask_gt = overlay_mask(img_np, mask_color, alpha=0.4)
        img_mask_pred = overlay_mask(img_np, pred_color, alpha=0.4)

        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img_np)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(img_mask_gt)
        plt.title("Ground Truth Mask (Green)")
        plt.axis("off")

        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(img_mask_pred)
        plt.title("Predicted Mask (Red)")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

predict_and_visualize(model, SegmentationDataset(val_imgs, val_masks_p, IMG_SIZE), num_samples=3)


DATA_DIR = '/kaggle/working/'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train/train')
TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'train_masks/train_masks')

img_paths = sorted(glob(os.path.join(TRAIN_IMG_DIR, '*.jpg')) +
                   glob(os.path.join(TRAIN_IMG_DIR, '*.png')))
mask_paths = sorted(glob(os.path.join(TRAIN_MASK_DIR, '*.gif')))

print(f"num images: {len(img_paths)}, num masks: {len(mask_paths)}")


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=(128,128)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)

        mask_gif = Image.open(self.mask_paths[idx]).convert("L")
        mask = np.array(mask_gif)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        img_t = self.to_tensor(img)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t


class UNetResNet50(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained)
        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64
        self.enc2 = nn.Sequential(backbone.maxpool, backbone.layer1)  # 256
        self.enc3 = backbone.layer2  # 512
        self.enc4 = backbone.layer3  # 1024
        self.enc5 = backbone.layer4  # 2048

        self.up4 = self._up_block(2048, 1024)
        self.dec4 = self._conv_block(2048, 1024)

        self.up3 = self._up_block(1024, 512)
        self.dec3 = self._conv_block(1024, 512)

        self.up2 = self._up_block(512, 256)
        self.dec2 = self._conv_block(512, 256)

        self.up1 = self._up_block(256, 64)
        self.dec1 = self._conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _up_block(self, in_ch, out_ch):
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d4 = self.up4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        out = torch.nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(1)
    return 1 - ((2.*intersection + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)).mean()

bce = nn.BCEWithLogitsLoss()
def combined_loss(pred, target, alpha=0.5):
    return alpha*bce(pred, target) + (1-alpha)*dice_loss(pred, target)

def iou_score(pred, target, thr=0.5, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > thr).float()
    intersect = (pred * target).sum((1,2,3))
    union = ((pred + target) > 0).float().sum((1,2,3))
    return ((intersect + eps) / (union + eps)).mean().item()


IMG_SIZE = (128,128)
BATCH_SIZE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_imgs, val_imgs, train_masks_p, val_masks_p = train_test_split(
    img_paths, mask_paths, test_size=0.15, random_state=42)

train_loader = DataLoader(SegmentationDataset(train_imgs, train_masks_p, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SegmentationDataset(val_imgs, val_masks_p, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False)


model1 = UNetResNet50(num_classes=1, pretrained=True).to(DEVICE)
optimizer = torch.optim.Adam(model1.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)


train_losses, val_losses, val_ious = [], [], []
NUM_EPOCHS = 5

for epoch in range(1, NUM_EPOCHS+1):
    model1.train()
    running_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model1(imgs)
        loss = combined_loss(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    model1.eval()
    val_loss, ious = 0.0, []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model1(imgs)
            loss = combined_loss(preds, masks)
            val_loss += loss.item() * imgs.size(0)
            ious.append(iou_score(preds, masks))
    val_loss /= len(val_loader.dataset)
    mean_iou = np.mean(ious)

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    val_ious.append(mean_iou)

    scheduler.step(val_loss)

    print(f"Epoch [{epoch}/{NUM_EPOCHS}] Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {mean_iou:.4f}")


# Confusion Matrix & Classification Report
all_preds, all_targets = [], []
model1.eval()
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model1(imgs)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_targets.append(masks.cpu().numpy())

all_preds = np.concatenate(all_preds).reshape(-1)
all_targets = np.concatenate(all_targets).reshape(-1)

cm = confusion_matrix(all_targets, all_preds)
print("\nClassification Report:\n", classification_report(all_targets, all_preds, digits=4))

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Background (0)", "Object (1)"],
            yticklabels=["Background (0)", "Object (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# Prediction & Visualization
def predict_and_visualize(model, dataset, num_samples=3):
    model.eval()
    plt.figure(figsize=(12, num_samples*4))
    for i in range(num_samples):
        img, mask = dataset[i]
        img_input = img.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(img_input)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()
        pred_np = pred.squeeze().cpu().numpy()

        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img_np)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(mask_np, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(pred_np, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

predict_and_visualize(model1, SegmentationDataset(val_imgs, val_masks_p, IMG_SIZE), num_samples=3)


torch.save(model.state_dict(), "model.pth")

