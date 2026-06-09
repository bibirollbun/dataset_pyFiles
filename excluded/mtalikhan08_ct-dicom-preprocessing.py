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
    "EPOCHS": 5,
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
    print(f"ðŸš€ Initializing On-the-Fly Pipeline on {CONFIG['DEVICE']}...")
    
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

