# Unzip data
import zipfile
with zipfile.ZipFile('/kaggle/input/data-science-bowl-2018/'+ 'stage1_train.zip', 'r') as zip_ref:
    zip_ref.extractall('./train')
    
with zipfile.ZipFile('/kaggle/input/data-science-bowl-2018/' + 'stage1_test.zip', 'r') as zip_ref:
    zip_ref.extractall('./test')


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

import os
import numpy as np
import random
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

# --- âš™ï¸� CONFIGURATION ---
TRAIN_DIR = "/kaggle/working/train"
TEST_DIR = "/kaggle/working/test"

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 100
SEED = 42

VALIDATION_SPLIT = 0.1 # 10%
TEST_SPLIT = 0.2       # 20%
EARLY_STOP_PATIENCE = 15

# --- THAY Ä�á»”I: Ä�Æ°á»�ng dáº«n lÆ°u model riÃªng biá»‡t ---
UNET_SAVE_PATH = "/kaggle/working/best_unet_model.pth"
UNET_PLUS_PLUS_SAVE_PATH = "/kaggle/working/best_unet_pp_model.pth"
# -----------------------------------------------

# CÃ i Ä‘áº·t seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Using device: {DEVICE}")


print("--- 2. Starting EDA & Data Pre-loading ---")

train_ids = next(os.walk(TRAIN_DIR))[1]

# Lists to store all data
X_data = [] # Original images (resized)
Y_data = [] # Combined masks (resized)

# Stats for EDA
original_sizes = []
mask_counts = []

print('Loading, resizing, and collecting stats...')
for id_ in tqdm(train_ids):
    path = os.path.join(TRAIN_DIR, id_)
    
    # --- Load Image & Stats ---
    img_path = os.path.join(path, 'images', id_ + '.png')
    img_original = imread(img_path)
    original_sizes.append(img_original.shape[:2]) # (H, W)
    
    # Resize image
    img = resize(img_original[:,:,:IMG_CHANNELS], (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_data.append(img.astype(np.uint8))

    # --- Load Masks & Stats ---
    mask_path = os.path.join(path, 'masks')
    mask_files = next(os.walk(mask_path))[2]
    mask_counts.append(len(mask_files))
    
    # Combine masks
    mask_combined = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
    for mask_file in mask_files:
        mask_ = imread(os.path.join(mask_path, mask_file))
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask_combined = np.maximum(mask_combined, mask_)
    Y_data.append(mask_combined)

print(f"Data pre-loading complete. Loaded {len(X_data)} images.")

# --- Visualize Stats ---
print("\n--- EDA Visualizations ---")

# 1. Histogram of mask counts (Sá»‘ lÆ°á»£ng háº¡t nhÃ¢n trÃªn má»—i áº£nh)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.hist(mask_counts, bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Nuclei Count per Image')
plt.xlabel('Number of Nuclei (Masks)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)

# 2. Scatter plot of original image sizes (KÃ­ch thÆ°á»›c áº£nh gá»‘c)
plt.subplot(1, 2, 2)
sizes_np = np.array(original_sizes)
plt.scatter(sizes_np[:, 1], sizes_np[:, 0], alpha=0.3) # x=Width, y=Height
plt.title('Original Image Sizes (Width vs Height)')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 3. Visualize a random sample (Xem áº£nh vÃ  mask máº«u)
print("\n--- Sample Image and Combined Mask ---")
ix = random.randint(0, len(X_data) - 1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(X_data[ix])
plt.title(f'Sample Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.squeeze(Y_data[ix]), cmap='gray')
plt.title(f'Combined Mask ({mask_counts[ix]} nuclei)')
plt.axis('off')
plt.show()


# 3. Visualize 5 random samples
print("\n--- 5 Sample Images and Combined Masks ---")

# Láº¥y 5 index ngáº«u nhiÃªn (khÃ´ng trÃ¹ng láº·p) tá»« bá»™ dá»¯ liá»‡u
indices = random.sample(range(len(X_data)), 5)

for ix in indices:
    # Láº¥y dá»¯ liá»‡u
    image = X_data[ix]
    mask = Y_data[ix]
    mask_count = mask_counts[ix]
    image_id = train_ids[ix]

    # Váº½
    plt.figure(figsize=(10, 5))
    
    # áº¢nh gá»‘c
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Sample Image ({ix})')
    plt.axis('off')

    # Mask tá»•ng há»£p
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(mask), cmap='gray')
    plt.title(f'Combined Mask ({mask_count} nuclei)')
    plt.axis('off')
    
    plt.show() # Hiá»ƒn thá»‹ tá»«ng cáº·p áº£nh


class ConvBlock(nn.Module):
    """(Conv2d -> BN -> ReLU) x 2"""
    # LÆ¯U Ã�: Ä�Ã¢y lÃ  block giá»‘ng há»‡t nhÆ° trong UNet++
    # Náº¿u báº¡n Ä‘Ã£ Ä‘á»‹nh nghÄ©a nÃ³ á»Ÿ Cell 3 (UNet++) rá»“i thÃ¬ khÃ´ng cáº§n Ä‘á»‹nh nghÄ©a láº¡i
    # NhÆ°ng Ä‘á»ƒ cell nÃ y Ä‘á»™c láº­p, tÃ´i sáº½ thÃªm nÃ³ vÃ o Ä‘Ã¢y.
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    Implementation of U-Net (Viáº¿t theo style cá»§a UNet++)
    Sá»­ dá»¥ng filter count: [32, 64, 128, 256, 512]
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256, 512]):
        super(UNet, self).__init__()
        nb_filter = features

        self.pool = nn.MaxPool2d(2, 2)
        # Sá»­ dá»¥ng ConvTranspose2d cho nháº¥t quÃ¡n vá»›i U-Net gá»‘c
        # (UNet++ dÃ¹ng Upsample, nhÆ°ng ConvTranspose2d cÅ©ng ráº¥t phá»• biáº¿n)
        # Báº¡n cÃ³ thá»ƒ Ä‘á»•i self.up4_t thÃ nh nn.Upsample(...) náº¿u muá»‘n
        
        # --- Encoder ---
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        
        # --- Bottleneck ---
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        # --- Decoder ---
        # Up-sampling + ConvBlock
        self.up4_t = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], kernel_size=2, stride=2)
        self.conv3_1 = ConvBlock(nb_filter[4], nb_filter[3]) # (skip + up) -> conv
        
        self.up3_t = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)
        self.conv2_1 = ConvBlock(nb_filter[3], nb_filter[2])
        
        self.up2_t = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.conv1_1 = ConvBlock(nb_filter[2], nb_filter[1])
        
        self.up1_t = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.conv0_1 = ConvBlock(nb_filter[1], nb_filter[0])

        # --- Final Output ---
        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder path ---
        x0_0 = self.conv0_0(x)       # Skip 1 (32 filters)
        x1_0 = self.conv1_0(self.pool(x0_0)) # Skip 2 (64 filters)
        x2_0 = self.conv2_0(self.pool(x1_0)) # Skip 3 (128 filters)
        x3_0 = self.conv3_0(self.pool(x2_0)) # Skip 4 (256 filters)

        # --- Bottleneck ---
        x4_0 = self.conv4_0(self.pool(x3_0)) # (512 filters)

        # --- Decoder path ---
        # Táº§ng 4
        x3_1_up = self.up4_t(x4_0)
        # Cáº¯t (crop) náº¿u kÃ­ch thÆ°á»›c khÃ´ng khá»›p (Ã­t xáº£y ra vá»›i padding='same')
        if x3_1_up.shape != x3_0.shape:
             x3_1_up = TF.resize(x3_1_up, size=x3_0.shape[2:])
        x3_1_cat = torch.cat([x3_0, x3_1_up], 1) # Concat (256 + 256)
        x3_1 = self.conv3_1(x3_1_cat)
        
        # Táº§ng 3
        x2_1_up = self.up3_t(x3_1)
        if x2_1_up.shape != x2_0.shape:
             x2_1_up = TF.resize(x2_1_up, size=x2_0.shape[2:])
        x2_1_cat = torch.cat([x2_0, x2_1_up], 1) # Concat (128 + 128)
        x2_1 = self.conv2_1(x2_1_cat)

        # Táº§ng 2
        x1_1_up = self.up2_t(x2_1)
        if x1_1_up.shape != x1_0.shape:
             x1_1_up = TF.resize(x1_1_up, size=x1_0.shape[2:])
        x1_1_cat = torch.cat([x1_0, x1_1_up], 1) # Concat (64 + 64)
        x1_1 = self.conv1_1(x1_1_cat)
        
        # Táº§ng 1
        x0_1_up = self.up1_t(x1_1)
        if x0_1_up.shape != x0_0.shape:
             x0_1_up = TF.resize(x0_1_up, size=x0_0.shape[2:])
        x0_1_cat = torch.cat([x0_0, x0_1_up], 1) # Concat (32 + 32)
        x0_1 = self.conv0_1(x0_1_cat)
        
        # --- Output ---
        output = self.final(x0_1)
        return torch.sigmoid(output) # Sigmoid vÃ¬ dÃ¹ng BCELoss


class ConvBlock(nn.Module):
    """(Conv2d -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetPlusPlus(nn.Module):
    """
    Implementation of UNet++ (Nested U-Net)
    Sá»­ dá»¥ng filter count tá»« Table 2: [32, 64, 128, 256, 512]
    """
    def __init__(self, in_channels=3, out_channels=1, 
                 filters=[32, 64, 128, 256, 512], deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        
        self.deep_supervision = deep_supervision
        nb_filter = filters

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder path (X^i,0)
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        # Nested skip paths (X^i,j)
        # j=1
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        
        # j=2
        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2])
        
        # j=3
        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1])

        # j=4
        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        # Final output convs (cho deep supervision)
        # Ã�p dá»¥ng sigmoid á»Ÿ cuá»‘i cÃ¹ng
        self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder (X^i,0)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Nested path j=1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        # Nested path j=2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        # Nested path j=3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        
        # Nested path j=4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # Ã�p dá»¥ng sigmoid cho cÃ¡c output
        output1 = torch.sigmoid(self.final1(x0_1))
        output2 = torch.sigmoid(self.final2(x0_2))
        output3 = torch.sigmoid(self.final3(x0_3))
        output4 = torch.sigmoid(self.final4(x0_4))

        if self.deep_supervision:
            return [output1, output2, output3, output4]
        else:
            return output4


class DSB2018Dataset(Dataset):
    def __init__(self, X_data_list, Y_data_list, transform=None):
        self.transform = transform
        # Nháº­n dá»¯ liá»‡u Ä‘Ã£ Ä‘Æ°á»£c pre-load
        self.X_train_list = X_data_list
        self.Y_train_list = Y_data_list

    def __len__(self):
        return len(self.X_train_list)

    def __getitem__(self, index):
        image = self.X_train_list[index]
        mask = self.Y_train_list[index]
        
        # Chuáº©n hÃ³a áº£nh (vá»� [0, 1])
        image = image.astype(np.float32) / 255.0
        # Chuyá»ƒn mask (vá»� [0, 1])
        mask = (mask > 0).astype(np.float32)
        
        # Chuyá»ƒn sang tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)

        if self.transform:
            # Báº¡n cÃ³ thá»ƒ thÃªm augmentation á»Ÿ Ä‘Ã¢y
            pass
            
        return image, mask

# --- LOSS FUNCTIONS --- (KhÃ´ng thay Ä‘á»•i)
class DiceLoss(nn.Module):
    # ... (giá»¯ nguyÃªn code)
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    # ... (giá»¯ nguyÃªn code)
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice

class DeepSupervisionLoss(nn.Module):
    # ... (giá»¯ nguyÃªn code)
    def __init__(self, bce_weight=0.5, dice_weight=0.5, weights=[0.25, 0.25, 0.25, 0.25]):
        super(DeepSupervisionLoss, self).__init__()
        self.base_loss = BCEDiceLoss(bce_weight, dice_weight)
        self.weights = weights
    def forward(self, inputs: list, target):
        total_loss = 0
        for i, pred in enumerate(inputs):
            if pred.shape != target.shape:
                 target_resized = F.interpolate(target, size=pred.shape[2:], mode='bilinear', align_corners=False)
            else:
                 target_resized = target
            total_loss += self.weights[i] * self.base_loss(pred, target_resized)
        return total_loss

# --- METRIC --- (KhÃ´ng thay Ä‘á»•i)
def iou_metric(pred, target, threshold=0.5, smooth=1e-6):
    # ... (giá»¯ nguyÃªn code)
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def train_fn(loader, model, optimizer, loss_fn, device, deep_supervision):
    model.train()
    loop = tqdm(loader, desc="Training")
    
    total_loss = 0
    total_iou = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        predictions = model(data)
        
        if deep_supervision:
            loss = loss_fn(predictions, targets)
            iou_preds = predictions[-1] 
        else:
            loss = loss_fn(predictions, targets)
            iou_preds = predictions

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iou = iou_metric(iou_preds, targets)
        total_loss += loss.item()
        total_iou += iou.item()
        loop.set_postfix(loss=loss.item(), iou=iou.item())
        
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    print(f"Epoch Train Loss: {avg_loss:.4f}, Epoch Train IoU: {avg_iou:.4f}")
    # --- THÃŠM DÃ’NG NÃ€Y ---
    return {"loss": avg_loss, "iou": avg_iou}

def val_fn(loader, model, loss_fn, device, deep_supervision):
    model.eval()
    loop = tqdm(loader, desc="Validation")
    val_loss = 0
    val_iou = 0
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.to(device=device)
            predictions = model(data)
            
            if deep_supervision:
                loss = loss_fn(predictions, targets)
                iou_preds = predictions[-1]
            else:
                loss = loss_fn(predictions, targets)
                iou_preds = predictions

            val_loss += loss.item()
            val_iou += iou_metric(iou_preds, targets).item()

    avg_loss = val_loss / len(loader)
    avg_iou = val_iou / len(loader)
    print(f"==> Avg Val Loss: {avg_loss:.4f}, Avg Val IoU: {avg_iou:.4f}")
    # --- THÃŠM DÃ’NG NÃ€Y ---
    return {"loss": avg_loss, "iou": avg_iou}

# [HÃ€M Má»šI] DÃ¹ng Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ trÃªn táº­p test cuá»‘i cÃ¹ng
def evaluate_fn(loader, model, loss_fn, device, deep_supervision):
    model.eval()
    loop = tqdm(loader, desc="Evaluating Test Set")
    eval_loss = 0
    eval_iou = 0
    
    with torch.no_grad():
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)
            predictions = model(data)
            
            if deep_supervision:
                loss = loss_fn(predictions, targets)
                iou_preds = predictions[-1]
            else:
                loss = loss_fn(predictions, targets)
                iou_preds = predictions
                
            eval_loss += loss.item()
            eval_iou += iou_metric(iou_preds, targets).item()
            
    avg_loss = eval_loss / len(loader)
    avg_iou = eval_iou / len(loader)
    return avg_loss, avg_iou


print("--- 3. Initializing Dataset & Splitting ---")

# (Code táº£i vÃ  chia data... giá»¯ nguyÃªn)
# ...
full_dataset = DSB2018Dataset(X_data_list=X_data, Y_data_list=Y_data)
total_size = len(full_dataset)
val_size = int(total_size * VALIDATION_SPLIT)
test_size = int(total_size * TEST_SPLIT)
train_size = total_size - val_size - test_size
print(f"Splitting data: {train_size} Train, {val_size} Validation, {test_size} Test")
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("DataLoaders created.")

# --- NÆ¡i lÆ°u káº¿t quáº£ test cuá»‘i cÃ¹ng ---
final_test_results = {}
# --- Má»šI: NÆ¡i lÆ°u lá»‹ch sá»­ training ---
training_histories = {
    "unet": {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []},
    "unet_pp": {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}
}

# ======================================================================
# --- BÆ¯á»šC 4A: HUáº¤N LUYá»†N VÃ€ Ä�Ã�NH GIÃ� UNET ---
# ======================================================================
print(f"\n" + "="*50)
print(f"--- 4. STARTING UNET TRAINING ---")
print("="*50)

model_unet = UNet(in_channels=IMG_CHANNELS, out_channels=1).to(DEVICE)
loss_fn_unet = BCEDiceLoss()
optimizer_unet = optim.Adam(model_unet.parameters(), lr=LEARNING_RATE)
best_val_iou = -1.0
early_stop_counter = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    train_metrics = train_fn(train_loader, model_unet, optimizer_unet, loss_fn_unet, DEVICE, deep_supervision=False)
    val_metrics = val_fn(val_loader, model_unet, loss_fn_unet, DEVICE, deep_supervision=False)
    
    # --- Má»šI: LÆ°u lá»‹ch sá»­ ---
    training_histories["unet"]["train_loss"].append(train_metrics["loss"])
    training_histories["unet"]["train_iou"].append(train_metrics["iou"])
    training_histories["unet"]["val_loss"].append(val_metrics["loss"])
    training_histories["unet"]["val_iou"].append(val_metrics["iou"])
    # -----------------------

    current_val_iou = val_metrics["iou"]
    if current_val_iou > best_val_iou:
        best_val_iou = current_val_iou
        print(f"ğŸ�‰ New best UNet model! Val IoU: {best_val_iou:.4f}. Saving model to {UNET_SAVE_PATH}")
        torch.save(model_unet.state_dict(), UNET_SAVE_PATH)
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"Val IoU did not improve. Counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}")

    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print(f"--- UNet Early stopping triggered after {epoch+1} epochs ---")
        break

print("\n--- UNet Training Finished ---")
print(f"\n--- Evaluating UNet on Test Set ---")
model_unet.load_state_dict(torch.load(UNET_SAVE_PATH))
test_loss, test_iou = evaluate_fn(test_loader, model_unet, loss_fn_unet, DEVICE, deep_supervision=False)
final_test_results["UNet"] = {"loss": test_loss, "iou": test_iou}




# ======================================================================
# --- BÆ¯á»šC 4B: HUáº¤N LUYá»†N VÃ€ Ä�Ã�NH GIÃ� UNET++ ---
# ======================================================================
print(f"\n" + "="*50)
print(f"--- 5. STARTING UNET++ TRAINING ---")
print("="*50)

USE_DEEP_SUPERVISION = True
model_unet_pp = UNetPlusPlus(in_channels=IMG_CHANNELS, out_channels=1, 
                             deep_supervision=USE_DEEP_SUPERVISION).to(DEVICE)
if USE_DEEP_SUPERVISION:
    print("Using Deep Supervision Loss for UNet++.")
    loss_fn_unet_pp = DeepSupervisionLoss()
else:
    print("Using standard BCEDice Loss for UNet++.")
    loss_fn_unet_pp = BCEDiceLoss()
optimizer_unet_pp = optim.Adam(model_unet_pp.parameters(), lr=LEARNING_RATE)
best_val_iou = -1.0
early_stop_counter = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    train_metrics = train_fn(train_loader, model_unet_pp, optimizer_unet_pp, loss_fn_unet_pp, DEVICE, USE_DEEP_SUPERVISION)
    val_metrics = val_fn(val_loader, model_unet_pp, loss_fn_unet_pp, DEVICE, USE_DEEP_SUPERVISION)
    
    # --- Má»šI: LÆ°u lá»‹ch sá»­ ---
    training_histories["unet_pp"]["train_loss"].append(train_metrics["loss"])
    training_histories["unet_pp"]["train_iou"].append(train_metrics["iou"])
    training_histories["unet_pp"]["val_loss"].append(val_metrics["loss"])
    training_histories["unet_pp"]["val_iou"].append(val_metrics["iou"])
    # -----------------------

    current_val_iou = val_metrics["iou"]
    if current_val_iou > best_val_iou:
        best_val_iou = current_val_iou
        print(f"ğŸ�‰ New best UNet++ model! Val IoU: {best_val_iou:.4f}. Saving model to {UNET_PLUS_PLUS_SAVE_PATH}")
        torch.save(model_unet_pp.state_dict(), UNET_PLUS_PLUS_SAVE_PATH)
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"Val IoU did not improve. Counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}")

    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print(f"--- UNet++ Early stopping triggered after {epoch+1} epochs ---")
        break

print("\n--- UNet++ Training Finished ---")
print(f"\n--- Evaluating UNet++ on Test Set ---")
model_unet_pp.load_state_dict(torch.load(UNET_PLUS_PLUS_SAVE_PATH))
test_loss, test_iou = evaluate_fn(test_loader, model_unet_pp, loss_fn_unet_pp, DEVICE, USE_DEEP_SUPERVISION)
final_test_results["UNet++"] = {"loss": test_loss, "iou": test_iou}


# ======================================================================
# --- BÆ¯á»šC 6: Tá»”NG Káº¾T ---
# ======================================================================
print(f"\n" + "="*50)
print(f"== FINAL TEST SET RESULTS ==")
print(f"UNet:     Test Loss: {final_test_results['UNet']['loss']:.4f}, Test IoU: {final_test_results['UNet']['iou']:.4f}")
print(f"UNet++:   Test Loss: {final_test_results['UNet++']['loss']:.4f}, Test IoU: {final_test_results['UNet++']['iou']:.4f}")
print(f"================================")


print("--- 8. Visualizing Training History ---")

# Láº¥y dá»¯ liá»‡u tá»« dictionary
unet_hist = training_histories["unet"]
unet_pp_hist = training_histories["unet_pp"]

# --- 1. Biá»ƒu Ä‘á»“ Loss ---
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
# UNet
plt.plot(unet_hist["train_loss"], label='UNet - Train Loss', color='blue', linestyle='--')
plt.plot(unet_hist["val_loss"], label='UNet - Val Loss', color='blue')
# UNet++
plt.plot(unet_pp_hist["train_loss"], label='UNet++ - Train Loss', color='green', linestyle='--')
plt.plot(unet_pp_hist["val_loss"], label='UNet++ - Val Loss', color='green')

plt.title('Model Loss History')
plt.xlabel('Epochs')
plt.ylabel('Loss (BCE + Dice)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# --- 2. Biá»ƒu Ä‘á»“ IoU ---
plt.subplot(1, 2, 2)
# UNet
plt.plot(unet_hist["train_iou"], label='UNet - Train IoU', color='blue', linestyle='--')
plt.plot(unet_hist["val_iou"], label='UNet - Val IoU', color='blue')
# UNet++
plt.plot(unet_pp_hist["train_iou"], label='UNet++ - Train IoU', color='green', linestyle='--')
plt.plot(unet_pp_hist["val_iou"], label='UNet++ - Val IoU', color='green')

plt.title('Model IoU History')
plt.xlabel('Epochs')
plt.ylabel('IoU (Metric)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


print("--- 7. Visualizing Results on TEST Set (Comparing UNet vs UNet++) ---")

# --- 1. Táº£i cáº£ hai model tá»‘t nháº¥t ---
print("Loading best saved models...")
# Táº£i UNet
model_unet = UNet(in_channels=IMG_CHANNELS, out_channels=1).to(DEVICE)
model_unet.load_state_dict(torch.load(UNET_SAVE_PATH))
model_unet.eval()

# Táº£i UNet++
# (Giáº£ sá»­ model UNet++ tá»‘t nháº¥t Ä‘Ã£ Ä‘Æ°á»£c lÆ°u vá»›i deep_supervision=True)
DEEP_SUPERVISION_FOR_LOADING = True 
model_unet_pp = UNetPlusPlus(in_channels=IMG_CHANNELS, out_channels=1, 
                             deep_supervision=DEEP_SUPERVISION_FOR_LOADING).to(DEVICE)
model_unet_pp.load_state_dict(torch.load(UNET_PLUS_PLUS_SAVE_PATH))
model_unet_pp.eval()

print("Models loaded. Visualizing samples...")

# --- 2. Láº¥y máº«u vÃ  dá»± Ä‘oÃ¡n ---
num_samples = 5
indices = random.sample(range(len(test_dataset)), num_samples)

with torch.no_grad():
    for i, idx in enumerate(indices):
        # Láº¥y áº£nh vÃ  mask tá»« test_dataset
        img, mask = test_dataset[idx]
        
        # ThÃªm chiá»�u batch (1, C, H, W) vÃ  Ä‘Æ°a lÃªn GPU
        img_tensor = img.unsqueeze(0).to(DEVICE)
        
        # --- Dá»± Ä‘oÃ¡n ---
        # UNet
        pred_unet = model_unet(img_tensor)
        pred_unet_np = (pred_unet.squeeze().cpu().numpy() > 0.5).astype(float)
        
        # UNet++
        pred_unet_pp = model_unet_pp(img_tensor)
        if DEEP_SUPERVISION_FOR_LOADING:
            pred_unet_pp = pred_unet_pp[-1] # Láº¥y output cuá»‘i cÃ¹ng
        pred_unet_pp_np = (pred_unet_pp.squeeze().cpu().numpy() > 0.5).astype(float)
        
        # --- Chuyá»ƒn Ä‘á»•i Ä‘á»ƒ váº½ ---
        img_np = img.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        
        # --- Váº½ 4 áº£nh ---
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 4, 1)
        plt.title("Image (Test)")
        plt.imshow(img_np)
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(mask_np, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title("UNet Prediction")
        plt.imshow(pred_unet_np, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.title("UNet++ Prediction")
        plt.imshow(pred_unet_pp_np, cmap='gray')
        plt.axis('off')
        
        plt.show()

