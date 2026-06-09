!pip install segmentation-models-pytorch
import os
import glob
import numpy as np
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.notebook import tqdm
import segmentation_models_pytorch as smp

# --- 1. CONFIGURATION ---
CONFIG = {
    "INPUT_DIR": "/kaggle/input/rsna-2023-abdominal-trauma-detection/train_images",
    "IMG_SIZE": 256,
    "BATCH_SIZE": 16,      # Good for RTX 3060 / P100
    "ACCUM_STEPS": 2,      # Effective Batch = 32
    "LR": 1e-4,
    "EPOCHS": 30,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_WORKERS": 2       # Uses CPU to load DICOMs while GPU trains
}

# --- 2. PREPROCESSING UTILS (ON-THE-FLY) ---
def get_windowing(image, slope, intercept, window_center, window_width):
    """Apply CT windowing to raw pixels."""
    img_hu = image * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img_window = np.clip(img_hu, img_min, img_max)
    return img_window

def load_dicom_volume(series_path):
    """Reads DICOMs, selects a random chunk, and preprocesses."""
    dicom_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
    
    if not dicom_files:
        return None

    # --- EFFICIENCY HACK ---
    # Instead of loading 500 slices, pick 3 random consecutive slices (2.5D)
    # This makes it FAST enough for on-the-fly training.
    if len(dicom_files) < 3: 
        return None
    
    start_idx = np.random.randint(0, len(dicom_files) - 3)
    selected_files = dicom_files[start_idx : start_idx+3]
    
    processed_chunk = []
    
    for f in selected_files:
        try:
            ds = pydicom.dcmread(f)
            slope = float(getattr(ds, 'RescaleSlope', 1))
            intercept = float(getattr(ds, 'RescaleIntercept', 0))
            pixel_data = ds.pixel_array.astype(np.float32)
            
            # Soft Tissue Window (Abdomen)
            img = get_windowing(pixel_data, slope, intercept, 40, 400)
            
            # Normalize to [0, 1]
            img = (img - (40 - 200)) / 400 
            
            # Resize
            img = cv2.resize(img, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
            processed_chunk.append(img)
        except:
            return None

    if len(processed_chunk) != 3: 
        return None

    # Stack -> (256, 256, 3)
    return np.dstack(processed_chunk)

# --- 3. DIRECT DATASET (NO SAVING TO DISK) ---
class DirectTraumaDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Get list of all patient folders
        self.patient_ids = sorted(os.listdir(root_dir))
        # Filter out non-folders if any
        self.patient_ids = [p for p in self.patient_ids if os.path.isdir(os.path.join(root_dir, p))]
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_path = os.path.join(self.root_dir, patient_id)
        
        # Get first available series (simplified for training loop)
        series_list = os.listdir(patient_path)
        if not series_list:
            return torch.zeros((3, CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])), torch.zeros((1, CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
            
        series_id = series_list[0]
        series_path = os.path.join(patient_path, series_id)
        
        # Load Data (CPU Intense)
        image = load_dicom_volume(series_path)
        
        if image is None:
            # Fallback if load fails
            image = np.zeros((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3), dtype=np.float32)
            
        # To Tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Dummy Mask (Replace with real labels later)
        mask = torch.zeros((1, CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])).float()
        
        return image, mask

# --- 4. TRAINING LOOP ---
def train_on_the_fly():
    print(f"ğŸš€ Initializing On-the-Fly Pipeline on {CONFIG['DEVICE']}...")
    
    # Dataset
    dataset = DirectTraumaDataset(CONFIG['INPUT_DIR'])
    print(f"Found {len(dataset)} patients.")
    
    loader = DataLoader(
        dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=CONFIG['NUM_WORKERS'], 
        pin_memory=True
    )
    
    # --- FIX: CHANGED ENCODER TO 'mobilenet_v2' ---
    model = smp.Unet(
        encoder_name="mobilenet_v2",      # Supported & Fast
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    ).to(CONFIG['DEVICE'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # Loop
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for i, (img, mask) in enumerate(pbar):
            img, mask = img.to(CONFIG['DEVICE']), mask.to(CONFIG['DEVICE'])
            
            # Mixed Precision Step
            with autocast():
                pred = model(img)
                loss = criterion(pred, mask) / CONFIG['ACCUM_STEPS']
            
            scaler.scale(loss).backward()
            
            if (i + 1) % CONFIG['ACCUM_STEPS'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            epoch_loss += loss.item() * CONFIG['ACCUM_STEPS']
            pbar.set_postfix(loss=loss.item() * CONFIG['ACCUM_STEPS'])
            
        print(f"Epoch {epoch+1} Complete. Avg Loss: {epoch_loss/len(loader):.4f}")

# Run
if __name__ == "__main__":
    train_on_the_fly()


!pip install nibabel pydicom segmentation-models-pytorch


import os
import glob
import numpy as np
import pandas as pd
import cv2
import pydicom
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "ROOT_DIR": "/kaggle/input/rsna-2023-abdominal-trauma-detection",
    "IMG_SIZE": 128,
    "NUM_SLICES": 32,      # Depth of 3D chunk
    "BATCH_SIZE": 8,       # Optimized for 2x T4 GPUs
    "LR": 1e-4,
    "EPOCHS": 30,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_WORKERS": 4
}

# ==========================================
# 2. CUSTOM LOSS (The Fix for Blank Masks)
# ==========================================
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # Dice Loss
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()                            
        dice = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        # Combined: 30% BCE (Overall accuracy) + 70% Dice (Overlap focus)
        return 0.3 * bce + 0.7 * dice

# ==========================================
# 3. DATASET LOGIC
# ==========================================
def find_valid_pairs(root_dir):
    print("ğŸ”� Scanning dataset for matching pairs...")
    img_root = os.path.join(root_dir, "train_images")
    seg_root = os.path.join(root_dir, "segmentations")
    
    # Get available NIfTI masks
    available_masks = {f.split('.')[0]: os.path.join(seg_root, f) for f in os.listdir(seg_root) if f.endswith('.nii')}
    
    valid_pairs = []
    patients = os.listdir(img_root)
    
    for pid in tqdm(patients, desc="Matching Series"):
        patient_dir = os.path.join(img_root, pid)
        if not os.path.isdir(patient_dir): continue
        
        for series_id in os.listdir(patient_dir):
            if series_id in available_masks:
                valid_pairs.append({
                    'img_path': os.path.join(patient_dir, series_id),
                    'mask_path': available_masks[series_id]
                })
                
    print(f"âœ… Found {len(valid_pairs)} matched series.")
    return valid_pairs

class TraumaVolumeDataset(Dataset):
    def __init__(self, pairs_list, img_size=128, num_slices=32):
        self.pairs = pairs_list
        self.img_size = img_size
        self.num_slices = num_slices

    def __len__(self):
        return len(self.pairs)

    def load_volume_and_mask(self, img_path, mask_path):
        # 1. Load DICOMs (Sorted)
        dicom_files = glob.glob(os.path.join(img_path, "*.dcm"))
        dicom_files.sort(key=lambda x: int(pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber))
        
        # 2. Load NIfTI Mask
        nii = nib.load(mask_path)
        mask_data = nii.get_fdata() 
        # Fix Orientation: NIfTI (H, W, D) -> (D, H, W)
        mask_data = np.transpose(mask_data, (2, 1, 0)) 
        mask_data = np.rot90(mask_data, k=1, axes=(1,2))
        mask_data = np.flip(mask_data, axis=1)

        # 3. SMART CHUNKING (Crucial Fix)
        # Don't just pick a random chunk (likely empty). Pick a chunk with organs.
        total_slices = len(dicom_files)
        best_start = 0
        max_pixels = -1
        
        if total_slices > self.num_slices:
            # Try 10 random spots, keep the one with most mask content
            for _ in range(10):
                start = np.random.randint(0, total_slices - self.num_slices)
                chunk_sum = np.sum(mask_data[start : start + self.num_slices])
                if chunk_sum > max_pixels:
                    max_pixels = chunk_sum
                    best_start = start
                    if max_pixels > 500: break # Found a good chunk, stop looking
            
            files_chunk = dicom_files[best_start : best_start + self.num_slices]
            mask_chunk = mask_data[best_start : best_start + self.num_slices]
        else:
            files_chunk = dicom_files
            mask_chunk = mask_data

        # 4. Process Images
        img_vol = []
        for f in files_chunk:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, 'RescaleSlope', 1))
            intercept = float(getattr(ds, 'RescaleIntercept', 0))
            img = img * slope + intercept
            img = np.clip(img, -40, 240) # Abdominal Window
            img = (img + 40) / 280
            img = cv2.resize(img, (self.img_size, self.img_size))
            img_vol.append(img)

        # 5. Process Mask
        mask_vol = []
        for i in range(len(mask_chunk)):
            m = cv2.resize(mask_chunk[i], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask_vol.append(m)
            
        img_vol = np.array(img_vol)
        mask_vol = np.array(mask_vol)
        mask_vol = (mask_vol > 0).astype(np.float32) # Binary

        # Padding
        if img_vol.shape[0] < self.num_slices:
            pad = self.num_slices - img_vol.shape[0]
            img_vol = np.pad(img_vol, ((0, pad), (0,0), (0,0)), 'constant')
            mask_vol = np.pad(mask_vol, ((0, pad), (0,0), (0,0)), 'constant')
            
        return img_vol, mask_vol

    def __getitem__(self, idx):
        try:
            pair = self.pairs[idx]
            img, mask = self.load_volume_and_mask(pair['img_path'], pair['mask_path'])
        except:
            img = np.zeros((self.num_slices, self.img_size, self.img_size), dtype=np.float32)
            mask = np.zeros_like(img)
        return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(mask).unsqueeze(0).float()

# ==========================================
# 4. MODEL (MobileNetV3-UNet)
# ==========================================
class SEBlock3D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Hardsigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class MobileNetBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio, use_se=True):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        hidden_dim = int(round(in_ch * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.extend([nn.Conv3d(in_ch, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm3d(hidden_dim), nn.Hardswish()])
        pad = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size, stride, pad, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.Hardswish()
        ])
        if use_se: layers.append(SEBlock3D(hidden_dim))
        layers.extend([nn.Conv3d(hidden_dim, out_ch, 1, 1, 0, bias=False), nn.BatchNorm3d(out_ch)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class MobileNetV3UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_channels, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(16), nn.Hardswish())
        self.layer1 = MobileNetBlock3D(16, 24, 3, 1, 2) 
        self.layer2 = MobileNetBlock3D(24, 40, 5, 2, 4)
        self.layer3 = MobileNetBlock3D(40, 80, 5, 2, 4)
        self.layer4 = MobileNetBlock3D(80, 112, 5, 2, 4)
        
        self.up1 = nn.ConvTranspose3d(112, 80, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv3d(160, 80, 3, 1, 1), nn.BatchNorm3d(80), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose3d(80, 40, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv3d(80, 40, 3, 1, 1), nn.BatchNorm3d(40), nn.ReLU(inplace=True))
        self.up3 = nn.ConvTranspose3d(40, 24, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv3d(48, 24, 3, 1, 1), nn.BatchNorm3d(24), nn.ReLU(inplace=True))
        self.up4 = nn.ConvTranspose3d(24, 16, kernel_size=2, stride=2)
        self.final = nn.Conv3d(16, num_classes, 1)

    def forward(self, x):
        x0 = self.stem(x); x1 = self.layer1(x0); x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        d1 = self.dec1(torch.cat([self.up1(x4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), x2], dim=1))
        d3_up = self.up3(d2)
        if d3_up.shape != x1.shape: d3_up = F.interpolate(d3_up, size=x1.shape[2:], mode='nearest')
        d3 = self.dec3(torch.cat([d3_up, x1], dim=1))
        out = self.final(self.up4(d3))
        if out.shape[2:] != x.shape[2:]: out = F.interpolate(out, size=x.shape[2:], mode='nearest')
        return out

# ==========================================
# 5. PIPELINE EXECUTION
# ==========================================
def extract_roi(image, mask, padding=5):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols): return image, None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    h, w = image.shape[:2]
    y_min = max(0, y_min - padding); y_max = min(h, y_max + padding)
    x_min = max(0, x_min - padding); x_max = min(w, x_max + padding)
    return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

def run_pipeline():
    print(f"ğŸš€ STARTING PIPELINE (Smart Chunking + Dice Loss) | GPUs: {torch.cuda.device_count()}")
    
    # 1. Data Setup
    valid_pairs = find_valid_pairs(CONFIG['ROOT_DIR'])
    if not valid_pairs: print("â�Œ No Pairs Found!"); return
    
    train_pairs, val_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)
    train_loader = DataLoader(TraumaVolumeDataset(train_pairs), batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'])
    val_loader = DataLoader(TraumaVolumeDataset(val_pairs), batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'])
    
    # 2. Model Setup
    model = MobileNetV3UNet3D(in_channels=1, num_classes=1)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    model = model.to(CONFIG['DEVICE'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    criterion = DiceBCELoss() # <--- THE CRITICAL FIX
    scaler = torch.amp.GradScaler('cuda')
    history = []
    
    # 3. Training Loop
    for epoch in range(CONFIG['EPOCHS']):
        model.train(); t_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for img, mask in pbar:
            img, mask = img.to(CONFIG['DEVICE']), mask.to(CONFIG['DEVICE'])
            with torch.amp.autocast('cuda'):
                pred = model(img)
                loss = criterion(pred, mask)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            t_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = t_loss/len(train_loader)
        history.append(avg_loss)
        print(f"Ep {epoch+1} | Avg Loss: {avg_loss:.4f}")

    # 4. Final Visualization
    print("\nğŸ–¼ï¸� VISUALIZING PREDICTIONS...")
    img, _ = val_loader.dataset[0]
    model.eval()
    with torch.no_grad():
        # Predict on GPU
        logit = model(img.unsqueeze(0).to(CONFIG['DEVICE']))
        pred_mask = torch.sigmoid(logit).cpu().numpy()[0, 0]

    # Show Slice 16
    mid = 16
    sl_img = img[0, mid].numpy()
    sl_mask = pred_mask[mid]
    
    # Binarize mask for cropping (>0.5)
    binary_mask = (sl_mask > 0.5).astype(int)
    crop, bbox = extract_roi(sl_img, binary_mask)
    
    plt.figure(figsize=(16, 5))
    ax1 = plt.subplot(1,3,1); ax1.imshow(sl_img, cmap='gray'); ax1.set_title("Input CT")
    if bbox: ax1.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], ec='r', fc='none', lw=2))
    
    plt.subplot(1,3,2); plt.imshow(sl_mask, cmap='jet'); plt.title("AI Probability Map")
    
    plt.subplot(1,3,3)
    if crop is not None: plt.imshow(crop, cmap='gray'); plt.title("Final Extracted Organ")
    else: plt.text(0.5,0.5,"No Organ Found",ha='center'); plt.title("Crop Failed")
    
    plt.show()

if __name__ == "__main__":
    run_pipeline()


import os
import glob
import numpy as np
import pandas as pd
import cv2
import pydicom
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "ROOT_DIR": "/kaggle/input/rsna-2023-abdominal-trauma-detection",
    "SAVE_DIR": "/kaggle/working/processed_crops",
    "IMG_SIZE": 128,
    "NUM_SLICES": 32,      # Depth of 3D chunk
    "BATCH_SIZE": 8,       # 8 Total (Split 4 per GPU on 2x T4)
    "LR": 1e-4,
    "EPOCHS": 100,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_WORKERS": 4
}

os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)

# ==========================================
# 2. LOSS & UTILS
# ==========================================
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()                            
        dice = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 0.3 * bce + 0.7 * dice

def find_valid_pairs(root_dir):
    print("ğŸ”� Scanning dataset...")
    img_root = os.path.join(root_dir, "train_images")
    seg_root = os.path.join(root_dir, "segmentations")
    available_masks = {f.split('.')[0]: os.path.join(seg_root, f) for f in os.listdir(seg_root) if f.endswith('.nii')}
    valid_pairs = []
    if not os.path.exists(img_root): return []
    
    for pid in tqdm(os.listdir(img_root), desc="Matching"):
        p_dir = os.path.join(img_root, pid)
        if not os.path.isdir(p_dir): continue
        for sid in os.listdir(p_dir):
            if sid in available_masks:
                valid_pairs.append({'img_path': os.path.join(p_dir, sid), 'mask_path': available_masks[sid]})
    print(f"âœ… Found {len(valid_pairs)} pairs.")
    return valid_pairs

# ==========================================
# 3. DATASET (Smart Chunking)
# ==========================================
class TraumaVolumeDataset(Dataset):
    def __init__(self, pairs_list, img_size=128, num_slices=32):
        self.pairs = pairs_list
        self.img_size = img_size
        self.num_slices = num_slices

    def __len__(self): return len(self.pairs)

    def load_volume_and_mask(self, img_path, mask_path):
        dicom_files = glob.glob(os.path.join(img_path, "*.dcm"))
        # Sorting is critical for 3D consistency
        dicom_files.sort(key=lambda x: int(pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber))
        
        nii = nib.load(mask_path)
        mask_data = nii.get_fdata() 
        mask_data = np.transpose(mask_data, (2, 1, 0)) 
        mask_data = np.rot90(mask_data, k=1, axes=(1,2))
        mask_data = np.flip(mask_data, axis=1)

        # Smart Chunking: Find a chunk with content
        total_slices = len(dicom_files)
        best_start = 0
        max_pixels = -1
        
        if total_slices > self.num_slices:
            for _ in range(10): # Search 10 times for a good spot
                start = np.random.randint(0, total_slices - self.num_slices)
                chunk_sum = np.sum(mask_data[start : start + self.num_slices])
                if chunk_sum > max_pixels:
                    max_pixels = chunk_sum
                    best_start = start
                    if max_pixels > 500: break
            files_chunk = dicom_files[best_start : best_start + self.num_slices]
            mask_chunk = mask_data[best_start : best_start + self.num_slices]
        else:
            files_chunk = dicom_files
            mask_chunk = mask_data

        img_vol = []
        for f in files_chunk:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, 'RescaleSlope', 1))
            intercept = float(getattr(ds, 'RescaleIntercept', 0))
            img = img * slope + intercept
            img = np.clip(img, -40, 240)
            img = (img + 40) / 280
            img = cv2.resize(img, (self.img_size, self.img_size))
            img_vol.append(img)

        mask_vol = []
        for i in range(len(mask_chunk)):
            m = cv2.resize(mask_chunk[i], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask_vol.append(m)
            
        img_vol = np.array(img_vol)
        mask_vol = np.array(mask_vol)
        mask_vol = (mask_vol > 0).astype(np.float32)

        if img_vol.shape[0] < self.num_slices:
            pad = self.num_slices - img_vol.shape[0]
            img_vol = np.pad(img_vol, ((0, pad), (0,0), (0,0)), 'constant')
            mask_vol = np.pad(mask_vol, ((0, pad), (0,0), (0,0)), 'constant')
            
        return img_vol, mask_vol

    def __getitem__(self, idx):
        try:
            pair = self.pairs[idx]
            img, mask = self.load_volume_and_mask(pair['img_path'], pair['mask_path'])
        except:
            img = np.zeros((self.num_slices, self.img_size, self.img_size), dtype=np.float32)
            mask = np.zeros_like(img)
        return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(mask).unsqueeze(0).float()

# ==========================================
# 4. MODEL (MobileNetV3-UNet 3D)
# ==========================================

class SEBlock3D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Hardsigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class MobileNetBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio, use_se=True):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        hidden_dim = int(round(in_ch * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.extend([nn.Conv3d(in_ch, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm3d(hidden_dim), nn.Hardswish()])
        pad = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size, stride, pad, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.Hardswish()
        ])
        if use_se: layers.append(SEBlock3D(hidden_dim))
        layers.extend([nn.Conv3d(hidden_dim, out_ch, 1, 1, 0, bias=False), nn.BatchNorm3d(out_ch)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class MobileNetV3UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_channels, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(16), nn.Hardswish())
        self.layer1 = MobileNetBlock3D(16, 24, 3, 1, 2) 
        self.layer2 = MobileNetBlock3D(24, 40, 5, 2, 4)
        self.layer3 = MobileNetBlock3D(40, 80, 5, 2, 4)
        self.layer4 = MobileNetBlock3D(80, 112, 5, 2, 4)
        self.up1 = nn.ConvTranspose3d(112, 80, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv3d(160, 80, 3, 1, 1), nn.BatchNorm3d(80), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose3d(80, 40, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv3d(80, 40, 3, 1, 1), nn.BatchNorm3d(40), nn.ReLU(inplace=True))
        self.up3 = nn.ConvTranspose3d(40, 24, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv3d(48, 24, 3, 1, 1), nn.BatchNorm3d(24), nn.ReLU(inplace=True))
        self.up4 = nn.ConvTranspose3d(24, 16, kernel_size=2, stride=2)
        self.final = nn.Conv3d(16, num_classes, 1)

    def forward(self, x):
        x0 = self.stem(x); x1 = self.layer1(x0); x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        d1 = self.dec1(torch.cat([self.up1(x4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), x2], dim=1))
        d3_up = self.up3(d2)
        if d3_up.shape != x1.shape: d3_up = F.interpolate(d3_up, size=x1.shape[2:], mode='nearest')
        d3 = self.dec3(torch.cat([d3_up, x1], dim=1))
        out = self.final(self.up4(d3))
        if out.shape[2:] != x.shape[2:]: out = F.interpolate(out, size=x.shape[2:], mode='nearest')
        return out

# ==========================================
# 5. CROP UTILS
# ==========================================
def extract_roi_2d(image, mask, padding=5):
    rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols): return image, None
    y_min, y_max = np.where(rows)[0][[0, -1]]; x_min, x_max = np.where(cols)[0][[0, -1]]
    h, w = image.shape[:2]
    y_min = max(0, y_min - padding); y_max = min(h, y_max + padding)
    x_min = max(0, x_min - padding); x_max = min(w, x_max + padding)
    return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

def extract_roi_3d(volume, mask, padding=5):
    rows = np.any(mask, axis=(0, 2)); cols = np.any(mask, axis=(0, 1)); depth = np.any(mask, axis=(1, 2))
    if not np.any(rows) or not np.any(cols): return volume
    y_min, y_max = np.where(rows)[0][[0, -1]]; x_min, x_max = np.where(cols)[0][[0, -1]]; z_min, z_max = np.where(depth)[0][[0, -1]]
    h, w = volume.shape[1:]; d = volume.shape[0]
    y_min = max(0, y_min - padding); y_max = min(h, y_max + padding)
    x_min = max(0, x_min - padding); x_max = min(w, x_max + padding)
    z_min = max(0, z_min); z_max = min(d, z_max)
    return volume[z_min:z_max, y_min:y_max, x_min:x_max]

# ==========================================
# 6. TRAINING PIPELINE
# ==========================================
def run_pipeline():
    print(f"ğŸš€ STARTING TRAINING | GPUs: {torch.cuda.device_count()}")
    valid_pairs = find_valid_pairs(CONFIG['ROOT_DIR'])
    if not valid_pairs: print("â�Œ No Pairs Found!"); return None, None

    train_pairs, val_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)
    train_loader = DataLoader(TraumaVolumeDataset(train_pairs), batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'])
    val_loader = DataLoader(TraumaVolumeDataset(val_pairs), batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'])
    
    # --- MODEL SETUP WITH MULTI-GPU ---
    model = MobileNetV3UNet3D(in_channels=1, num_classes=1)
    if torch.cuda.device_count() > 1: 
        print(f"ğŸ”¥ DataParallel Enabled on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(CONFIG['DEVICE'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    criterion = DiceBCELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # --- TRAINING LOOP ---
    for epoch in range(CONFIG['EPOCHS']):
        model.train(); t_loss = 0
        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            img, mask = img.to(CONFIG['DEVICE']), mask.to(CONFIG['DEVICE'])
            with torch.amp.autocast('cuda'):
                pred = model(img)
                loss = criterion(pred, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            t_loss += loss.item()
        print(f"Ep {epoch+1} | Loss: {t_loss/len(train_loader):.4f}")

    # --- VISUALIZATION ---
    print("\nğŸ–¼ï¸� VISUALIZING VALIDATION...")
    img, _ = val_loader.dataset[0]
    model.eval()
    with torch.no_grad():
        pred_mask = torch.sigmoid(model(img.unsqueeze(0).to(CONFIG['DEVICE']))).cpu().numpy()[0, 0]
    
    mid = 16
    crop, bbox = extract_roi_2d(img[0, mid].numpy(), (pred_mask[mid]>0.5).astype(int))
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1); plt.imshow(img[0, mid], cmap='gray'); plt.title("Input")
    plt.subplot(1,3,2); plt.imshow(pred_mask[mid], cmap='jet'); plt.title("Prediction")
    plt.subplot(1,3,3); 
    if crop is not None: plt.imshow(crop, cmap='gray'); plt.title("Crop")
    plt.show()
    
    return model, val_loader

# ==========================================
# 7. CROP GENERATION
# ==========================================
def generate_crops(model, val_loader):
    print(f"ğŸ’¾ SAVING CROPS TO: {CONFIG['SAVE_DIR']}")
    model.eval()
    count = 0
    with torch.no_grad():
        for i, (img_tensor, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
            img_tensor = img_tensor.to(CONFIG['DEVICE'])
            # Prediction
            logits = model(img_tensor)
            pred_masks = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            imgs_np = img_tensor.cpu().numpy()
            
            for b in range(imgs_np.shape[0]):
                # Extract 3D ROI for each patient in batch
                cropped = extract_roi_3d(imgs_np[b, 0], pred_masks[b, 0])
                np.save(os.path.join(CONFIG['SAVE_DIR'], f"crop_{count}.npy"), cropped)
                count += 1
    print(f"âœ… Done! Saved {count} files.")

if __name__ == "__main__":
    # 1. Train & Get Model
    trained_model, validation_loader = run_pipeline()
    
    # 2. Generate Crops (Only if training succeeded)
    if trained_model is not None:
        generate_crops(trained_model, validation_loader)


import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PRODUCTION CONFIGURATION
# ==========================================
CONFIG = {
    # Paths
    "ROOT_DIR": "/kaggle/input/rsna-2023-abdominal-trauma-detection",
    "CROP_DIR": "/kaggle/working/processed_crops_labeled", # Where your .npy files are
    "CSV_PATH": "/kaggle/input/rsna-2023-abdominal-trauma-detection/train_2024.csv",
    
    # Model Params
    "IMG_SIZE": 256,
    "SEQ_LEN": 24,         # Depth (Slices per patient)
    
    # Hardware & Training
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE": 8,       # Fits on 2x T4
    "ACCUM_STEPS": 4,      # Gradient Accumulation (Virtual batch = 32)
    "LR": 3e-4,
    "EPOCHS": 100,
    "PATIENCE": 3,         # Early stopping
    "NUM_WORKERS": 4
}

print(f"âœ… Configuration Loaded. Device: {CONFIG['DEVICE']}")


import os
import glob
import numpy as np
import pandas as pd
import cv2
import pydicom
import torch
from tqdm.notebook import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
PRE_CONFIG = {
    "ROOT_DIR": "/kaggle/input/rsna-2023-abdominal-trauma-detection",
    "SAVE_DIR": "/kaggle/working/processed_crops_labeled", # Must match training CROP_DIR
    "CSV_PATH": "/kaggle/input/rsna-2023-abdominal-trauma-detection/train_2024.csv",
    "IMG_SIZE": 256,
    "SEQ_LEN": 24
}

os.makedirs(PRE_CONFIG['SAVE_DIR'], exist_ok=True)

def preprocess_patient(patient_dir):
    # 1. Load DICOMs and Sort
    files = sorted(glob.glob(os.path.join(patient_dir, "*.dcm")), 
                   key=lambda x: int(pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber))
    
    if len(files) < 2: return None

    # 2. Smart Sampling (Standardize Depth to 24)
    indices = np.linspace(0, len(files)-1, PRE_CONFIG['SEQ_LEN']).astype(int)
    selected_files = [files[i] for i in indices]
    
    vol_stack = []
    for f in selected_files:
        try:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            
            # Windowing (Abdominal)
            slope = float(getattr(ds, 'RescaleSlope', 1))
            intercept = float(getattr(ds, 'RescaleIntercept', 0))
            img = img * slope + intercept
            img = np.clip(img, -40, 240)
            img = (img + 40) / 280 # Normalize
            
            # Resize
            img = cv2.resize(img, (PRE_CONFIG['IMG_SIZE'], PRE_CONFIG['IMG_SIZE']))
            vol_stack.append(img)
        except:
            vol_stack.append(np.zeros((PRE_CONFIG['IMG_SIZE'], PRE_CONFIG['IMG_SIZE']), dtype=np.float32))
            
    return np.array(vol_stack, dtype=np.float16)

def run_preprocessing():
    print(f"ğŸ’¾ GENERATING DATA -> {PRE_CONFIG['SAVE_DIR']}")
    
    df = pd.read_csv(PRE_CONFIG['CSV_PATH'])
    patient_ids = df['patient_id'].unique()
    
    # Limit to 100 patients for testing (Remove [:100] for full run)
    count = 0
    for pid in tqdm(patient_ids[:100], desc="Processing"):
        save_path = os.path.join(PRE_CONFIG['SAVE_DIR'], f"{pid}.npy")
        
        # Skip if already exists
        if os.path.exists(save_path): 
            count += 1
            continue
            
        p_path = os.path.join(PRE_CONFIG['ROOT_DIR'], "train_images", str(pid))
        if not os.path.exists(p_path): continue
        
        series = os.listdir(p_path)
        if not series: continue
        
        # Process first series
        vol = preprocess_patient(os.path.join(p_path, series[0]))
        if vol is not None:
            np.save(save_path, vol)
            count += 1

    print(f"âœ… Generated {count} files. You can now run training.")

if __name__ == "__main__":
    run_preprocessing()


# ==========================================
# 2. DATASET (Robust & Filtered)
# ==========================================
class CachedTraumaDataset(Dataset):
    def __init__(self, crop_dir, csv_path, split='train'):
        self.crop_dir = crop_dir
        self.df = pd.read_csv(csv_path)
        
        # Create a set of valid IDs from the CSV for filtering
        valid_ids = set(self.df['patient_id'].unique())
        
        # Scan folder for .npy files
        all_files = glob.glob(os.path.join(crop_dir, "*.npy"))
        
        # Filter: Only keep files that exist in the CSV
        self.files = []
        for f in all_files:
            try:
                pid = int(os.path.basename(f).replace('.npy', ''))
                if pid in valid_ids:
                    self.files.append(f)
            except: continue
                
        print(f"   ğŸ“‚ {split.upper()}: Found {len(self.files)} valid patient volumes.")

        # Set index for fast label lookup
        self.df = self.df.set_index('patient_id')
        
        # Target: Any Injury (Binary)
        cols = ['bowel_injury', 'extravasation_injury', 'kidney_low', 'kidney_high', 
                'liver_low', 'liver_high', 'spleen_low', 'spleen_high']
        self.df['target'] = self.df[cols].max(axis=1)
        
        # Train/Val Split (80/20)
        split_idx = int(len(self.files) * 0.8)
        if split == 'train': 
            self.files = self.files[:split_idx]
        else: 
            self.files = self.files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        pid = int(os.path.basename(path).replace('.npy',''))
        
        # Load 3D Volume
        vol = np.load(path).astype(np.float32)
        
        # Input Shape: (1, Seq, H, W)
        vol = torch.tensor(vol).unsqueeze(0) 
        
        # Get Label
        label = self.df.loc[pid, 'target']
        return vol, torch.tensor(label, dtype=torch.float32)


# ==========================================
# 3. CLASSIFICATION MODEL (2.5D) - MobileNetV2
# ==========================================
class TraumaClassifierMobileNetV2(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # A. Backbone (MobileNetV2)
        # Using weights='DEFAULT' loads ImageNet weights
        self.backbone = models.mobilenet_v2(weights='DEFAULT')
        self.features = self.backbone.features
        
        # MobileNetV2 output channels = 1280 (vs 576 in V3-Small)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # B. Aggregator (Bi-LSTM)
        # Input size must match backbone output channels (1280)
        self.lstm = nn.LSTM(
            input_size=1280, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # C. Head
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Input: (Batch, 1, Seq, H, W)
        b, c, s, h, w = x.shape
        
        # 1. CNN Phase (Fold time into batch)
        x = x.view(b * s, 1, h, w)
        x = x.repeat(1, 3, 1, 1) # 1 channel -> 3 channels
        
        x = self.features(x)       # (B*S, 1280, 7, 7)
        x = self.pool(x)           # (B*S, 1280, 1, 1)
        x = x.view(b, s, -1)       # Unfold: (B, S, 1280)
        
        # 2. RNN Phase
        x, _ = self.lstm(x)        # (B, S, Hidden*2)
        
        # 3. Aggregation (Max Pool over time)
        x, _ = torch.max(x, dim=1) # (B, Hidden*2)
        
        return self.head(x)


# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_production_model():
    print(f"ğŸš€ STARTING TRAINING (MobileNetV2) on {CONFIG['DEVICE']}")
    
    # 1. Load Data
    train_ds = CachedTraumaDataset(CONFIG['CROP_DIR'], CONFIG['CSV_PATH'], 'train')
    val_ds = CachedTraumaDataset(CONFIG['CROP_DIR'], CONFIG['CSV_PATH'], 'val')
    
    if len(train_ds) == 0:
        print("â�Œ Error: No training data found. Did you run the preprocessing step?")
        return None, None, None
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'])
    
    # 2. Model & Optimizer
    # --- UPDATED TO USE MOBILENET V2 CLASS ---
    model = TraumaClassifierMobileNetV2().to(CONFIG['DEVICE'])
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda') # AMP
    
    # 3. Loop
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_loss = float('inf')
    patience_c = 0
    
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        t_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for i, (X, y) in enumerate(pbar):
            X, y = X.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
            
            with torch.amp.autocast('cuda'):
                pred = model(X).squeeze()
                loss = criterion(pred, y) / CONFIG['ACCUM_STEPS']
            
            scaler.scale(loss).backward()
            
            if (i + 1) % CONFIG['ACCUM_STEPS'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            t_loss += loss.item() * CONFIG['ACCUM_STEPS']
            pbar.set_postfix(loss=loss.item() * CONFIG['ACCUM_STEPS'])
            
        # Validation
        model.eval()
        v_loss = 0
        preds, targets = [], []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
                with torch.amp.autocast('cuda'):
                    out = model(X).squeeze()
                    v_loss += criterion(out, y).item()
                    preds.extend(torch.sigmoid(out).cpu().numpy())
                    targets.extend(y.cpu().numpy())
        
        avg_t = t_loss / len(train_loader)
        avg_v = v_loss / len(val_loader)
        try: auc_score = roc_auc_score(targets, preds)
        except: auc_score = 0.5
        
        history['train_loss'].append(avg_t)
        history['val_loss'].append(avg_v)
        history['val_auc'].append(auc_score)
        
        print(f"Ep {epoch+1} | Train: {avg_t:.4f} | Val: {avg_v:.4f} | AUC: {auc_score:.4f}")
        
        # Patience Check
        if avg_v < best_loss:
            best_loss = avg_v
            patience_c = 0
            torch.save(model.state_dict(), "best_model_mobilenetv2.pth")
        else:
            patience_c# filepath: c:\Users\sarma\Downloads\ct-dicom-preprocessing-e45610 (1).ipynb
# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_production_model():
    print(f"ğŸš€ STARTING TRAINING (MobileNetV2) on {CONFIG['DEVICE']}")
    
    # 1. Load Data
    train_ds = CachedTraumaDataset(CONFIG['CROP_DIR'], CONFIG['CSV_PATH'], 'train')
    val_ds = CachedTraumaDataset(CONFIG['CROP_DIR'], CONFIG['CSV_PATH'], 'val')
    
    if len(train_ds) == 0:
        print("â�Œ Error: No training data found. Did you run the preprocessing step?")
        return None, None, None
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'])
    
    # 2. Model & Optimizer
    # --- UPDATED TO USE MOBILENET V2 CLASS ---
    model = TraumaClassifierMobileNetV2().to(CONFIG['DEVICE'])
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda') # AMP
    
    # 3. Loop
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_loss = float('inf')
    patience_c = 0
    
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        t_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for i, (X, y) in enumerate(pbar):
            X, y = X.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
            
            with torch.amp.autocast('cuda'):
                pred = model(X).squeeze()
                loss = criterion(pred, y) / CONFIG['ACCUM_STEPS']
            
            scaler.scale(loss).backward()
            
            if (i + 1) % CONFIG['ACCUM_STEPS'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            t_loss += loss.item() * CONFIG['ACCUM_STEPS']
            pbar.set_postfix(loss=loss.item() * CONFIG['ACCUM_STEPS'])
            
        # Validation
        model.eval()
        v_loss = 0
        preds, targets = [], []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
                with torch.amp.autocast('cuda'):
                    out = model(X).squeeze()
                    v_loss += criterion(out, y).item()
                    preds.extend(torch.sigmoid(out).cpu().numpy())
                    targets.extend(y.cpu().numpy())
        
        avg_t = t_loss / len(train_loader)
        avg_v = v_loss / len(val_loader)
        try: auc_score = roc_auc_score(targets, preds)
        except: auc_score = 0.5
        
        history['train_loss'].append(avg_t)
        history['val_loss'].append(avg_v)
        history['val_auc'].append(auc_score)
        
        print(f"Ep {epoch+1} | Train: {avg_t:.4f} | Val: {avg_v:.4f} | AUC: {auc_score:.4f}")
        
        # Patience Check
        if avg_v < best_loss:
            best_loss = avg_v
            patience_c = 0
            torch.save(model.state_dict(), "best_model_mobilenetv2.pth")
        else:
            patience_c += 1
            if patience_c >= CONFIG['PATIENCE']:
                print("ğŸ›‘ Early Stopping Triggered")
                break
                
    return history, targets, preds


# ==========================================
# 5. EXECUTION & REPORTING
# ==========================================
def generate_report(history, y_true, y_pred):
    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 5))
    
    # A. Learning Curve
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='o')
    plt.title("Loss History")
    plt.xlabel("Epochs"); plt.ylabel("BCE Loss"); plt.legend()
    
    # B. ROC Curve
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title("ROC Curve")
    plt.legend()
    
    # C. Confusion Matrix
    plt.subplot(1, 3, 3)
    y_bin = (np.array(y_pred) > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_bin)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    
    plt.tight_layout()
    plt.show()
    
    print("\nğŸ“Š CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_bin))

if __name__ == "__main__":
    # 1. Train
    hist, y_true, y_prob = train_production_model()
    
    # 2. Report (Only if training happened)
    if hist is not None:
        generate_report(hist, y_true, y_prob)


#pip install timm


# import os
# import glob
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import timm
# from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm.notebook import tqdm
# import warnings

# warnings.filterwarnings('ignore')

# # ==========================================
# # 1. CONFIGURATION
# # ==========================================
# CONFIG = {
#     "CROP_DIR": "/kaggle/working/processed_crops_labeled", 
#     "CSV_PATH": "/kaggle/input/rsna-2023-abdominal-trauma-detection/train_2024.csv", 
#     "IMG_SIZE": 224,       
#     "SEQ_LEN": 24,         
#     "BATCH_SIZE": 8,       
#     "ACCUM_STEPS": 4,      
#     "LR": 1e-4,
#     "EPOCHS": 30,
#     "PATIENCE": 4,         
#     "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
#     "NUM_WORKERS": 4
# }

# # ==========================================
# # 2. DATASET (ROBUST FIX)
# # ==========================================
# class CachedTraumaDataset(Dataset):
#     def __init__(self, crop_dir, csv_path, split='train'):
#         self.files = glob.glob(os.path.join(crop_dir, "*.npy"))
#         self.df = pd.read_csv(csv_path)
        
#         # Create Target Column
#         injury_cols = [
#             'bowel_injury', 'extravasation_injury', 
#             'kidney_low', 'kidney_high', 
#             'liver_low', 'liver_high', 
#             'spleen_low', 'spleen_high'
#         ]
#         self.df['injury_label'] = self.df[injury_cols].max(axis=1)
#         self.labels = self.df.set_index('patient_id')['injury_label'].to_dict()
        
#         # --- ROBUST FILE FILTERING ---
#         self.valid_files = []
#         for f in self.files:
#             try:
#                 filename = os.path.basename(f) # e.g. "10004_21057.npy" or "11832.npy"
#                 clean_name = filename.replace('.npy', '')
                
#                 # Handle both "PID_SID" and "PID" formats
#                 if '_' in clean_name:
#                     patient_id = int(clean_name.split('_')[0])
#                 else:
#                     patient_id = int(clean_name)
                    
#                 if patient_id in self.labels:
#                     self.valid_files.append(f)
#             except ValueError:
#                 continue # Skip weird files
        
#         # Split
#         split_idx = int(len(self.valid_files) * 0.8)
#         if split == 'train': self.files = self.valid_files[:split_idx]
#         else: self.files = self.valid_files[split_idx:]

#     def __len__(self): return len(self.files)

#     def __getitem__(self, idx):
#         path = self.files[idx]
#         filename = os.path.basename(path)
#         clean_name = filename.replace('.npy', '')
        
#         # Robust ID Extraction
#         if '_' in clean_name:
#             patient_id = int(clean_name.split('_')[0])
#         else:
#             patient_id = int(clean_name)
        
#         # Load Volume
#         vol = np.load(path).astype(np.float32)
        
#         # Normalize & Resize (2.5D Stack)
#         vol = torch.tensor(vol).unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
#         vol = torch.nn.functional.interpolate(vol, size=(CONFIG['SEQ_LEN'], CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']), mode='trilinear', align_corners=False)
#         vol = vol.squeeze(0) # (1, 24, 224, 224)
        
#         label = self.labels.get(patient_id, 0.0)
#         return vol, torch.tensor(label, dtype=torch.float32)

# # ==========================================
# # 3. MODEL: EfficientNet-Lite0 + LSTM
# # ==========================================
# class EfficientNetLiteLSTM(nn.Module):
#     def __init__(self, hidden_dim=256):
#         super().__init__()
#         # Backbone
#         self.backbone = timm.create_model('tf_efficientnet_lite0', pretrained=True, features_only=True)
#         self.feature_dim = 320 
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # Aggregator
#         self.lstm = nn.LSTM(
#             input_size=self.feature_dim, 
#             hidden_size=hidden_dim, 
#             num_layers=1, 
#             batch_first=True, 
#             bidirectional=True
#         )
        
#         # Head
#         self.head = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim * 2, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         b, c, s, h, w = x.shape
#         x = x.view(b * s, 1, h, w)
#         x = x.repeat(1, 3, 1, 1) # RGB
        
#         features = self.backbone(x)[-1] 
#         x = self.pool(features)     
#         x = x.view(b, s, -1)        
        
#         x, _ = self.lstm(x)         
#         x, _ = torch.max(x, dim=1)  
#         return self.head(x)

# # ==========================================
# # 4. TRAINING ENGINE
# # ==========================================
# def train_efficientnet():
#     print(f"ğŸš€ STARTING EFFICIENTNET-LITE TRAINING | Device: {CONFIG['DEVICE']}")
    
#     if not os.path.exists(CONFIG['CROP_DIR']):
#         print("â�Œ No crops found."); return None, None

#     train_ds = CachedTraumaDataset(CONFIG['CROP_DIR'], CONFIG['CSV_PATH'], 'train')
#     val_ds = CachedTraumaDataset(CONFIG['CROP_DIR'], CONFIG['CSV_PATH'], 'val')
    
#     if len(train_ds) == 0:
#         print("â�Œ Dataset is empty after filtering. Check CSV IDs vs Filenames.")
#         return None, None

#     train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'])
#     val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'])
    
#     model = EfficientNetLiteLSTM().to(CONFIG['DEVICE'])
#     if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    
#     optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    
#     # Class Balancing
#     pos_weight = torch.tensor([3.0]).to(CONFIG['DEVICE'])
#     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
#     scaler = torch.amp.GradScaler('cuda')
    
#     best_auc = 0
#     patience_c = 0
    
#     for epoch in range(CONFIG['EPOCHS']):
#         model.train()
#         t_loss = 0
        
#         pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
#         for i, (img, label) in enumerate(pbar):
#             img, label = img.to(CONFIG['DEVICE']), label.to(CONFIG['DEVICE'])
            
#             with torch.amp.autocast('cuda'):
#                 pred = model(img).squeeze()
#                 loss = criterion(pred, label) / CONFIG['ACCUM_STEPS']
            
#             scaler.scale(loss).backward()
            
#             if (i+1) % CONFIG['ACCUM_STEPS'] == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()
            
#             t_loss += loss.item() * CONFIG['ACCUM_STEPS']
#             pbar.set_postfix(loss=loss.item() * CONFIG['ACCUM_STEPS'])
            
#         # Validation
#         model.eval()
#         preds, targets = [], []
#         with torch.no_grad():
#             for img, label in val_loader:
#                 img, label = img.to(CONFIG['DEVICE']), label.to(CONFIG['DEVICE'])
#                 with torch.amp.autocast('cuda'):
#                     out = model(img).squeeze()
#                     preds.extend(torch.sigmoid(out).cpu().numpy())
#                     targets.extend(label.cpu().numpy())
        
#         try: auc_val = roc_auc_score(targets, preds)
#         except: auc_val = 0.5
        
#         print(f"Ep {epoch+1} | Loss: {t_loss/len(train_loader):.4f} | Val AUC: {auc_val:.4f}")
        
#         if auc_val > best_auc:
#             best_auc = auc_val
#             patience_c = 0
#             torch.save(model.state_dict(), "efficientnet_best.pth")
#         else:
#             patience_c += 1
#             if patience_c >= CONFIG['PATIENCE']:
#                 print("ğŸ›‘ Early Stopping!"); break
                
#     return targets, preds

# if __name__ == "__main__":
#     y_true, y_pred = train_efficientnet()
    
#     if y_true is not None:
#         y_bin = (np.array(y_pred) > 0.5).astype(int)
#         print("\nğŸ“Š Final Report:")
#         print(classification_report(y_true, y_bin))
        
#         plt.figure(figsize=(6,5))
#         sns.heatmap(confusion_matrix(y_true, y_bin), annot=True, fmt='d', cmap='Blues')
#         plt.title("Confusion Matrix")
#         plt.xlabel("Predicted"); plt.ylabel("Actual")
#         plt.show()







