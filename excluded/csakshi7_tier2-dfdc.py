import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import time

# --- CONFIGURATION ---
BATCH_SIZE = 32      # Good balance for T4 GPU
IMG_SIZE = 224       # Standard for Xception
EPOCHS = 20          # Enough to learn, short enough to finish
LEARNING_RATE = 0.0001 # Low rate for fine-tuning

# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"âœ… Device Selected: {device}")
if device.type == 'cuda':
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")


# We use the sample folder for the project demo (approx 400 videos)
DATA_ROOT = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/"
METADATA_PATH = os.path.join(DATA_ROOT, "metadata.json")

print(f"ğŸ“‚ Loading metadata from: {METADATA_PATH}")

try:
    # Read JSON and transpose to DataFrame
    df = pd.read_json(METADATA_PATH).T
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'filename', 'label': 'label_str'}, inplace=True)
    
    # Create numeric label (FAKE=1, REAL=0)
    df['label'] = df['label_str'].apply(lambda x: 1 if x == 'FAKE' else 0)
    
    # Create full path column
    df['full_path'] = df['filename'].apply(lambda x: os.path.join(DATA_ROOT, x))
    
    # Verification
    print(f"âœ… Metadata Loaded. Total Videos: {len(df)}")
    print(f"   Fake Videos: {len(df[df['label'] == 1])}")
    print(f"   Real Videos: {len(df[df['label'] == 0])}")
    
except Exception as e:
    print(f"â�Œ Error loading metadata: {e}")


class DeepfakeDataset(Dataset):
    def __init__(self, df, frames_per_video, transform=None):
        self.df = df
        self.frames_per_video = frames_per_video
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_path = self.df.loc[idx, 'full_path']
        label = torch.tensor(self.df.loc[idx, 'label'], dtype=torch.float)
        
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0: return None # Skip bad videos
            
            # Smart Sampling: Get frames evenly distributed
            indices = np.linspace(0, total_frames-1, self.frames_per_video, dtype=int)
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret: continue
                
                # --- CENTER CROP (Focus on the face area) ---
                h, w, _ = frame.shape
                short_edge = min(h, w)
                crop_size = int(short_edge * 0.75) # Take 75% of center
                
                center_x, center_y = w // 2, h // 2
                x1 = center_x - crop_size // 2
                y1 = center_y - crop_size // 2
                
                frame = frame[y1:y1+crop_size, x1:x1+crop_size]
                
                # Convert BGR (OpenCV) to RGB (PIL)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Apply Transforms
                if self.transform:
                    frame = self.transform(frame)
                
                frames.append(frame)
                
            cap.release()
        except Exception as e:
            return None

        if len(frames) == 0: return None

        # Stack frames into a tensor [Sequence_Length, Channels, H, W]
        return torch.stack(frames), label

# --- COLLATE FUNCTION (Prevents crashes on bad videos) ---
def collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch: return torch.tensor([]), torch.tensor([])
    
    # Flatten the batch: We treat every frame as an independent image for training
    all_faces = []
    all_labels = []
    
    for faces, label in batch:
        all_faces.extend(faces)
        all_labels.extend([label] * len(faces))
        
    return torch.stack(all_faces), torch.stack(all_labels)

print("âœ… DeepfakeDataset and Collate Function Defined.")


# --- BLOCK 4: TRANSFORMS & LOADERS ---
# 1. AUGMENTATION (Training)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. CLEAN (Validation)
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. SPLIT DATA
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# 4. CREATE LOADERS
# FRAMES_PER_VIDEO = 10 (Good balance)
train_dataset = DeepfakeDataset(train_df.reset_index(drop=True), 10, train_transform)
val_dataset = DeepfakeDataset(val_df.reset_index(drop=True), 10, val_transform)

# SAFE MODE: num_workers=0 prevents the "300% CPU" deadlock
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

print(f"âœ… Loaders Ready. Train Videos: {len(train_df)}, Val Videos: {len(val_df)}")


# --- BLOCK 5: MODEL ARCHITECTURE (MobileNetV3) ---
from torchvision import models
import torch.nn as nn
import torch.optim as optim

# 1. Load Pretrained MobileNetV3 (Lightweight & Fast)
# We use "Large" variant for better accuracy than "Small"
model = models.mobilenet_v3_large(weights='DEFAULT')

# 2. Replace the Classifier Head
# MobileNet's head is called 'classifier'. We replace it for Binary (Real vs Fake)
in_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 1024),
    nn.Hardswish(),
    nn.Dropout(p=0.5),            # Strong Dropout to prevent overfitting
    nn.Linear(1024, 1),           # Output: 1 number
    nn.Sigmoid()                  # Squash to 0.0-1.0 probability
)

# 3. Move to GPU
model = model.to(device)

# 4. Define Loss & Optimizer
# Learning Rate 0.0001 is safe for fine-tuning
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("âœ… Model (MobileNetV3-Large) initialized and moved to GPU.")


# --- BLOCK 6: TRAINING LOOP (With Checkpointing) ---
import time

EPOCHS = 3
best_val_acc = 0.0
MODEL_SAVE_PATH = "best_model.pth"

print(f"--- Starting Training for {EPOCHS} Epochs ---")
print("â�³ This will take time. Monitor the progress bar below.")

history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

for epoch in range(EPOCHS):
    start_time = time.time()
    
    # --- TRAINING ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
        inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        # Threshold 0.5 for accuracy
        predictions = (outputs > 0.5).float()
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)
        
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = 100 * train_correct / train_total

    # --- VALIDATION ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            predictions = (outputs > 0.5).float()
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
            
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = 100 * val_correct / val_total
    
    # --- SAVE BEST MODEL ---
    saved_msg = ""
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        saved_msg = "ğŸ’¾ SAVED BEST!"
        
    # --- LOGGING ---
    elapsed = (time.time() - start_time) / 60
    print(f"Epoch {epoch+1}/{EPOCHS} | Time: {elapsed:.1f}m | "
          f"Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}% {saved_msg}")
    
    # Store for graphs
    history['train_acc'].append(epoch_train_acc)
    history['val_acc'].append(epoch_val_acc)
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)

print(f"\nâœ… Training Finished. Best Accuracy: {best_val_acc:.2f}%")
print(f"âœ… Best model saved to: {MODEL_SAVE_PATH}")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Validation')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.legend()
plt.show()


# FULL 3-TIER SYSTEM (PYTORCH)
import torch
import cv2
import numpy as np
import imagehash
from PIL import Image
import os
from torchvision import transforms, models
import torch.nn as nn

# 1. SETUP DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"âœ… Device: {device}")

# 2. LOAD THE BEST MODEL
# We rebuild the architecture and load the weights you just trained
def load_trained_model():
    print("ğŸ”„ Loading Best Model...")
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 1),
        nn.Sigmoid()
    )
    
    weights_path = "best_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"âœ… Loaded weights from {weights_path}")
    else:
        print("âš ï¸� Warning: 'best_model.pth' not found. Using untrained model for demo structure.")
        
    model.to(device)
    model.eval()
    return model

# Initialize Model
model = load_trained_model()

# 3. TIER 1: HASH CHECK (Fast Filter)
KNOWN_FAKE_HASHES = {'a1b2c3d4e5f6', '123456789abc'}

def tier_1_hash_check(frames):
    if len(frames) == 0: return False, "No frames"
    try:
        # Convert tensor to PIL
        if isinstance(frames[0], torch.Tensor):
            img = transforms.ToPILImage()(frames[0])
        else:
            img = Image.fromarray(frames[0])
            
        p_hash = str(imagehash.phash(img))
        if p_hash in KNOWN_FAKE_HASHES:
            return True, f"ğŸš« BLOCKED: Known Fake (Hash: {p_hash})"
    except: pass
    return False, "âœ… Tier 1 Passed (Unique Content)"

# 4. TIER 3: TEMPORAL SCAN (Deep Ensemble)
def tier_3_temporal_scan(model, frame_tensors):
    print("   ... Running Deep Temporal Analysis (Tier 3) ...")
    predictions = []
    with torch.no_grad():
        for i in range(len(frame_tensors)):
            input_frame = frame_tensors[i].unsqueeze(0).to(device)
            output = model(input_frame)
            predictions.append(output.item())
            
    variance = np.var(predictions)
    avg_score = np.mean(predictions)
    
    print(f"   [Tier 3 Stats] Variance: {variance:.5f} | Avg: {avg_score:.4f}")
    
    # Logic: High jitter/variance = Fake
    if variance > 0.01: 
        return "FAKE (Temporal Artifacts Found)", 0.95
    
    final_verdict = "FAKE" if avg_score > 0.5 else "REAL"
    return final_verdict, avg_score

# 5. MASTER PIPELINE
def run_full_system(video_path):
    print(f"\n========================================")
    print(f"ğŸ�¬ PROCESSING VIDEO: {os.path.basename(video_path)}")
    print(f"========================================")
    
    # A. Extract Frames
    try:
        # Use the dataset logic manually
        cap = cv2.VideoCapture(video_path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total-1, 10, dtype=int)
        
        # Basic transform for inference
        inf_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(inf_transform(frame))
        cap.release()
        
        if len(frames) == 0: return "â�Œ No frames extracted."
        frames = torch.stack(frames)
        
    except Exception as e:
        return f"â�Œ Error loading video: {e}"

    # B. Tier 1
    is_blocked, msg = tier_1_hash_check(frames)
    print(f"1ï¸�âƒ£ TIER 1: {msg}")
    if is_blocked: return "ğŸ›‘ FINAL VERDICT: FAKE"

    # C. Tier 2
    with torch.no_grad():
        mid_frame = frames[len(frames)//2].unsqueeze(0).to(device)
        t2_score = model(mid_frame).item()
    
    print(f"2ï¸�âƒ£ TIER 2: Confidence Score: {t2_score:.4f}")
    
    if t2_score > 0.85:
        return f"ğŸ›‘ FINAL VERDICT: FAKE ({t2_score:.2%} confidence)"
    elif t2_score < 0.15:
        return f"âœ… FINAL VERDICT: REAL ({1-t2_score:.2%} confidence)"

    # D. Tier 3 (Escalation)
    print(f"3ï¸�âƒ£ TIER 3: Escalating to Deep Ensemble...")
    verdict, score = tier_3_temporal_scan(model, frames)
    
    emoji = "ğŸ›‘" if "FAKE" in verdict else "âœ…"
    return f"{emoji} FINAL VERDICT: {verdict} (Ensemble Confidence: {score:.2%})"

# 6. RUN TEST
import glob
test_files = glob.glob('/kaggle/input/deepfake-detection-challenge/train_sample_videos/*.mp4')
if len(test_files) > 0:
    # Test on the first video found
    print(run_full_system(test_files[0]))
else:
    print("âš ï¸� No video files found. Check dataset path.")

