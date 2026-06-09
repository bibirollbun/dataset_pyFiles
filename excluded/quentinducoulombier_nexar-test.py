import os
import cv2
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


NUM_FRAMES    = 16              # For MVIT_V2_S, use exactly 16 frames
FRAME_SIZE    = (224, 224)      # Expected resolution: 224x224
BATCH_SIZE    = 12               # Adjust according to available resources
NUM_EPOCHS    = 4               # Number of training epochs
LEARNING_RATE = 1e-4            # Learning rate
NUM_WORKERS   = 2               # Number of workers for DataLoader

TRAIN_CSV       = "/kaggle/input/nexar-collision-prediction/train.csv"
TEST_CSV        = "/kaggle/input/nexar-collision-prediction/test.csv"
TRAIN_VIDEO_DIR = "/kaggle/input/nexar-collision-prediction/train/"
TEST_VIDEO_DIR  = "/kaggle/input/nexar-collision-prediction/test/"


class NexarDataset(Dataset):
    """
    Custom dataset for loading videos from a CSV and folder.
    - Extracts exactly NUM_FRAMES frames uniformly.
    - Converts frames from BGR to RGB and resizes to FRAME_SIZE.
    - Applies a transform (if provided) on each frame.
    - Returns a tensor of shape (C, T, H, W) and the target (for "train" mode).
    """
    def __init__(self, csv_path, video_dir, num_frames=NUM_FRAMES, transform=None, mode="train"):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def _load_video(self, video_id):
        video_path = os.path.join(self.video_dir, video_id + ".mp4")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Compute uniform indices for NUM_FRAMES frames
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        collected_frames = {}
        frame_id = 0
        ret = True
        while ret:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id in indices:
                # Convert from BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, FRAME_SIZE)
                collected_frames[frame_id] = frame
            frame_id += 1
        cap.release()
        # For any missing frames, repeat the last available frame
        video_frames = [collected_frames[idx] if idx in collected_frames 
                        else (video_frames[-1] if len(video_frames) > 0 
                              else np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8))
                        for idx in indices]
        video_array = np.stack(video_frames)  # Shape: (T, H, W, C)
        return video_array

    def __getitem__(self, idx):
        video_id = str(self.df.loc[idx, "id"]).zfill(5)
        video_array = self._load_video(video_id)
        # Apply transformation on each frame
        if self.transform:
            # Each frame becomes a tensor of shape (C, H, W)
            video_tensor = torch.stack([self.transform(frame) for frame in video_array])
        else:
            video_tensor = torch.from_numpy(video_array.astype(np.float32) / 255.0)
            video_tensor = video_tensor.permute(0, 3, 1, 2)
        # Rearrange dimensions to (C, T, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        if self.mode == "train":
            target = torch.tensor(float(self.df.loc[idx, "target"]), dtype=torch.float32)
            return video_tensor, target
        else:
            return video_tensor, video_id



# Basic transformation: resize and normalize
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),  # Convert to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


df_train = pd.read_csv(TRAIN_CSV)
#df_train = df_train.head(150)
df_train['time_of_event'] = pd.to_numeric(df_train['time_of_event'], errors='coerce')
df_train['time_of_alert']  = pd.to_numeric(df_train['time_of_alert'], errors='coerce')

train_df, val_df = train_test_split(
    df_train, test_size=0.05, stratify=df_train['target']
)
print(f"Number of training samples: {len(train_df)}")
print(f"Number of validation samples: {len(val_df)}")

# Save splits as temporary CSV files for dataset loading
train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)


train_dataset = NexarDataset(csv_path="train_split.csv", video_dir=TRAIN_VIDEO_DIR, num_frames=NUM_FRAMES, transform=transform, mode="train")
val_dataset   = NexarDataset(csv_path="val_split.csv", video_dir=TRAIN_VIDEO_DIR, num_frames=NUM_FRAMES, transform=transform, mode="train")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)



weights = MViT_V2_S_Weights.KINETICS400_V1
model = mvit_v2_s(weights=weights).to(device)
# Replace the last layer in the head to output a single value (binary classification)
in_features = model.head[-1].in_features
model.head[-1] = nn.Linear(in_features, 1)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model = model.to(device)
print(model)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)  # Expected shape: (B, C, T, H, W)
        targets = targets.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_time

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.cpu().numpy().flatten().tolist())
    val_loss = running_loss / len(dataloader.dataset)
    return val_loss, np.array(all_preds), np.array(all_targets)

def compute_map(df, predictions, thresholds=[0.5, 1.0, 1.5]):
    """
    Computes the mean Average Precision (mAP) over multiple thresholds.
    df must contain columns 'target', 'time_of_event', and 'time_of_alert'.
    """
    from sklearn.metrics import average_precision_score
    APs = []
    for thr in thresholds:
        valid_idx = df.index[(df['target'] == 0) | ((df['target'] == 1) & ((df['time_of_event'] - df['time_of_alert']) >= thr))]
        if len(valid_idx) == 0:
            APs.append(0)
            continue
        y_true = df.loc[valid_idx, 'target'].values
        y_pred = predictions[valid_idx]
        ap = average_precision_score(y_true, y_pred)
        APs.append(ap)
    mean_AP = np.mean(APs)
    return mean_AP, APs



train_losses = []
val_losses = []
val_mAPs = []

for epoch in range(NUM_EPOCHS):
    train_loss, train_time = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Time: {train_time:.2f}s")
    
    val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
    
    # Load the validation CSV for mAP computation
    val_df_eval = pd.read_csv("val_split.csv")
    val_df_eval['time_of_event'] = pd.to_numeric(val_df_eval['time_of_event'], errors='coerce')
    val_df_eval['time_of_alert']  = pd.to_numeric(val_df_eval['time_of_alert'], errors='coerce')
    
    mean_AP, APs = compute_map(val_df_eval, val_preds, thresholds=[0.5, 1.0, 1.5])
    print(f"  Validation Loss: {val_loss:.4f} - mAP: {mean_AP:.4f} | AP per threshold: {APs}")
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_mAPs.append(mean_AP)



epochs_range = range(1, NUM_EPOCHS+1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_mAPs, label="Validation mAP", color="green")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("Validation mAP")
plt.legend()

plt.tight_layout()
plt.show()


test_dataset = NexarDataset(csv_path=TEST_CSV, video_dir=TEST_VIDEO_DIR, num_frames=NUM_FRAMES, transform=transform, mode="test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model.eval()
test_predictions = []
test_video_ids = []

with torch.no_grad():
    for inputs, vids in tqdm(test_loader, desc="Test Inference", leave=False):
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
        test_predictions.extend(probs.tolist())
        test_video_ids.extend(vids)


submission_df = pd.DataFrame({"id": test_video_ids, "score": test_predictions}).sort_values("id")
submission_df.to_csv("submission.csv", index=False)
print("Submission file generated: submission.csv")

