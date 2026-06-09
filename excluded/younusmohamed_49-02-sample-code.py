import cv2
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# class CustomCrashModel(nn.Module):
    
#     def __init__(self):
#         super(CustomCrashModel, self).__init__()

#         # Input shape - (B, 3, T, H, W)
#         self.conv1 = nn.Conv3d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)
#         self.bn1 = nn.BatchNorm3d(8)
#         self.pool1 = nn.MaxPool3d(kernel_size = (1, 2, 2))  # Pool spatial dimensions

#         self.conv2 = nn.Conv3d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
#         self.bn2 = nn.BatchNorm3d(16)
#         self.pool2 = nn.AdaptiveAvgPool3d(1)  # Global Pooling

#         self.fc = nn.Linear(16, 1)

#     def forward(self, x):
#         # x - (B, 3, T, H, W)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool1(x)  # shape: (B, 8, T, H/2, W/2)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool2(x)  # shape: (B, 16, 1, 1, 1)
#         x = x.view(x.size(0), -1)  # flatten to (B, 16)
#         x = self.fc(x)

#         return torch.sigmoid(x)


class SimpleCrashNN(nn.Module):
    def __init__(self):
        super(SimpleCrashNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(16, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # shape: (B, 8, T, H, W)
        x = F.relu(self.bn2(self.conv2(x)))  # shape: (B, 16, T, H, W)
        x = self.pool(x)                     # shape: (B, 16, 1, 1, 1)
        x = x.view(x.size(0), -1)            # flatten to (B, 16)
        x = self.fc(x)                       # shape: (B, 1)
        return torch.sigmoid(x)              # probability in [0, 1]

# Instantiate the model and move it to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCrashNN().to(device)
model.eval()  # Set model to evaluation mode
print("Simple custom NN initialized and set to eval mode.")


# Cell 3: Load training and test CSV files
train_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')
print("Training data loaded. Number of training videos:", len(train_df))
print("Test data loaded. Number of test videos:", len(test_df))


def extract_frames(video_path, num_frames=16, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_indices:
            frame = cv2.resize(frame, resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0  # Normalize pixels to [0, 1]
            frames.append(frame)
    cap.release()
    
    if len(frames) < num_frames:
        # Duplicate last frame if video is too short
        while len(frames) < num_frames:
            frames.append(frames[-1])
    
    # Convert frames to tensor and rearrange: (T, H, W, C) -> (B, C, T, H, W)
    frames = np.stack(frames, axis=0)           # (T, H, W, C)
    frames = np.transpose(frames, (3, 0, 1, 2))   # (C, T, H, W)
    frames_tensor = torch.from_numpy(frames).unsqueeze(0)
    return frames_tensor


def predict_video(video_path):
    frames_tensor = extract_frames(video_path, num_frames=16, resize=(224,224))
    if frames_tensor is None:
        return 0.0  # Default probability if video processing fails
    frames_tensor = frames_tensor.to(device)
    with torch.no_grad():
        output = model(frames_tensor)  # Model output shape: (B, 1)
        prob = output.item()
    return prob


# Cell 6: Generate predictions for training videos
train_predictions = []

# Training videos are stored in "train/" folder with filenames "<id>.mp4"
for idx, row in train_df.iterrows():
    # Convert video ID to an integer and format with leading zeros (5 digits)
    video_id = int(float(row['id']))
    video_filename = f"{video_id:05d}.mp4"  # e.g., 01924.mp4
    video_path = os.path.join("/kaggle/input/nexar-collision-prediction/train", video_filename)
    prob = predict_video(video_path)
    train_predictions.append(prob)
    if idx % 50 == 0:
        print(f"Processed {idx} training videos...")

train_df['predicted_score'] = train_predictions
print("Training predictions generated.")


# Cell 7: Generate predictions for test videos
test_predictions = []

# Test videos are stored in "test/" folder with filenames "<id>.mp4"
for idx, row in test_df.iterrows():
    video_id = int(float(row['id']))
    video_filename = f"{video_id:05d}.mp4"  # Format with 5 digits
    video_path = os.path.join("/kaggle/input/nexar-collision-prediction/test", video_filename)
    prob = predict_video(video_path)
    test_predictions.append(prob)
    if idx % 50 == 0:
        print(f"Processed {idx} test videos...")

test_df['score'] = test_predictions
print("Test predictions generated.")


submission = test_df[['id', 'score']]
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")

