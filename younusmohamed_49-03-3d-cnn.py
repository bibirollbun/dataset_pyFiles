# Import libraries
import cv2
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the deep custom 3D CNN model
class DeepCrashNN(nn.Module):
    def __init__(self):
        super(DeepCrashNN, self).__init__()
        # Block 1: 3D Conv + BatchNorm + ReLU, spatial pooling only (temporal resolution remains)
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))  # Only spatial pooling: reduces H & W by 2
        )
        # Block 2: Increase channels to 32
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        # Block 3: Increase channels to 64 with global pooling at the end
        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)  # Global average pooling over (T, H, W)
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.block1(x)  # (B, 16, T, H/2, W/2)
        x = self.block2(x)  # (B, 32, T, H/4, W/4)
        x = self.block3(x)  # (B, 64, 1, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (B, 64)
        x = self.dropout(x)
        x = self.fc(x)             # (B, 1)
        return torch.sigmoid(x)    # Probability in [0,1]

# Instantiate the model and set it to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCrashNN().to(device)
model.eval()
print("Deep custom 3D CNN model initialized and set to eval mode.")


# Load training and test CSV files
train_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')
print("Training data loaded. Number of training videos:", len(train_df))
print("Test data loaded. Number of test videos:", len(test_df))


# Define function to extract randomly sampled frames from a video
def extract_random_frames(video_path, num_frames=40, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    # Randomly sample 'num_frames' unique indices and sort them to preserve temporal order
    frame_indices = sorted(np.random.choice(total_frames, num_frames, replace=False))
    
    frames = []
    current_frame = 0
    next_idx = 0
    ret = True
    while ret and next_idx < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame == frame_indices[next_idx]:
            frame = cv2.resize(frame, resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0  # Normalize to [0,1]
            frames.append(frame)
            next_idx += 1
        current_frame += 1
    cap.release()
    
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])
    
    # Convert frames to tensor with shape (B, C, T, H, W)
    frames = np.stack(frames, axis=0)           # (T, H, W, C)
    frames = np.transpose(frames, (3, 0, 1, 2))   # (C, T, H, W)
    frames_tensor = torch.from_numpy(frames).unsqueeze(0)  # Add batch dimension
    return frames_tensor


# Define prediction function
def predict_video(video_path):
    frames_tensor = extract_random_frames(video_path, num_frames=16, resize=(224,224))
    if frames_tensor is None:
        return 0.0  # Default probability if video cannot be processed
    frames_tensor = frames_tensor.to(device)
    with torch.no_grad():
        output = model(frames_tensor)  # Output shape: (B, 1)
        prob = output.item()
    return prob


# Generate predictions for training videos
train_predictions = []

for idx, row in train_df.iterrows():
    # Convert video ID to integer and format with leading zeros (5 digits)
    video_id = int(float(row['id']))
    video_filename = f"{video_id:05d}.mp4"  # e.g., 01924.mp4
    video_path = os.path.join("/kaggle/input/nexar-collision-prediction/train", video_filename)
    prob = predict_video(video_path)
    train_predictions.append(prob)
    if idx % 50 == 0:
        print(f"Processed {idx} training videos...")

train_df['predicted_score'] = train_predictions
print("Training predictions generated.")


# Generate predictions for test videos
test_predictions = []

for idx, row in test_df.iterrows():
    video_id = int(float(row['id']))
    video_filename = f"{video_id:05d}.mp4"
    video_path = os.path.join("/kaggle/input/nexar-collision-prediction/test", video_filename)
    prob = predict_video(video_path)
    test_predictions.append(prob)
    if idx % 50 == 0:
        print(f"Processed {idx} test videos...")

test_df['score'] = test_predictions
print("Test predictions generated.")


# Save submission file
submission = test_df[['id', 'score']]
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")




