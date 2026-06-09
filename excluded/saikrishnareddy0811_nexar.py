# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         (os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd

# -------------------------------
# Device Setup for CUDA
# -------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Load and Modify ResNet-18
# -------------------------------
resnet18 = models.resnet18(pretrained=True)
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])  # Remove final FC layer
resnet18 = resnet18.to(device).eval()  # Set to evaluation mode

resnet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

FPS_TARGET = 5        # Set your target FPS for sampling (adjust if needed)
SEQUENCE_LENGTH = 5     # Number of frames per sequence

# -------------------------------
# Per-Frame Label Computation Function
# -------------------------------
def compute_frame_label(t, event_time, sigma_before=1.0, sigma_after=1.0):
    """
    Compute a soft target label for a single frame timestamp 't' given the event_time.
    Returns 1.0 if 't' is (approximately) equal to event_time, and otherwise a value in (0,1)
    based on an asymmetric Gaussian decay.
    """
    # Use a small tolerance (e.g., 0.033 sec ~ one frame at 30 FPS) to consider t as event frame.
    if np.isclose(t, event_time, atol=0.18):
        return 1.0
    if t < event_time:
        # Before event: decay from 1 as the time difference increases.
        return np.exp(-((event_time - t)**2) / (2 * sigma_before**2))
    else:
        # After event: decay from 1 as well.
        return np.exp(-((t - event_time)**2) / (2 * sigma_after**2))

# -------------------------------
# Feature Extraction & Sequence Generation
# -------------------------------
def extract_features_and_labels(video_dir, df):
    """
    For each video in the provided dataframe, extract one sequence of feature vectors using the pretrained ResNet-18.
    For positive cases (where time_of_event is provided), select a sequence containing that event 
    and compute a per-frame soft target vector based on time_of_event severity.
    For negative cases, select a sequence from the middle and assign a target vector of all zeros.
    
    Returns:
      video_sequences: numpy array of shape (num_videos, SEQUENCE_LENGTH, feature_dim)
      video_labels: numpy array of shape (num_videos, SEQUENCE_LENGTH)
    """
    video_sequences = []
    video_labels = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row["id"]
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            continue
        
        # For positive cases, time_of_event is provided; for negatives it will be NaN.
        event_time = row["time_of_event"]
        if pd.isna(event_time):
            event_time = -1  # Mark negative cases
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0 or np.isnan(fps):
            fps = 30
        
        frame_step = int(fps / FPS_TARGET)
        sampled_features = []
        sampled_timestamps = []
        
        current_frame = 0
        while current_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            # Extract feature using ResNet-18.
            frame_tensor = resnet_transform(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = resnet18(frame_tensor)
            feature = feature.squeeze().cpu().numpy()  # e.g., shape (512,)
            timestamp = current_frame / fps
            sampled_features.append(feature)
            sampled_timestamps.append(timestamp)
            current_frame += frame_step
        cap.release()
        
        sampled_features = np.array(sampled_features)
        sampled_timestamps = np.array(sampled_timestamps)
        
        if len(sampled_features) < SEQUENCE_LENGTH:
            continue
        
        # Positive case: event_time provided (target == 1 in CSV)
        if event_time != -1:
            event_idx = np.argmin(np.abs(sampled_timestamps - event_time))
            start_idx = max(0, event_idx - 2)
            end_idx = start_idx + SEQUENCE_LENGTH
            if end_idx > len(sampled_features):
                end_idx = len(sampled_features)
                start_idx = end_idx - SEQUENCE_LENGTH
            sequence = sampled_features[start_idx:end_idx]
            sequence_times = sampled_timestamps[start_idx:end_idx]
            # Compute a per-frame target label
            labels_seq = [compute_frame_label(t, event_time, sigma_before=1.0, sigma_after=1.0)
                          for t in sequence_times]
        else:
            # Negative case: assign a sequence from the middle with target 0 for every frame.
            mid_idx = len(sampled_features) // 2
            start_idx = max(0, mid_idx - SEQUENCE_LENGTH//2)
            end_idx = start_idx + SEQUENCE_LENGTH
            if end_idx > len(sampled_features):
                end_idx = len(sampled_features)
                start_idx = end_idx - SEQUENCE_LENGTH
            sequence = sampled_features[start_idx:end_idx]
            labels_seq = [0.0] * SEQUENCE_LENGTH
        
        video_sequences.append(sequence)
        video_labels.append(labels_seq)
    
    return np.array(video_sequences), np.array(video_labels)

# -------------------------------
# Create Dataset and DataLoader
# -------------------------------
class AccidentFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features  # shape: (num_samples, SEQUENCE_LENGTH, feature_dim)
        self.labels = labels      # shape: (num_samples, SEQUENCE_LENGTH)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.features[idx], dtype=torch.float32)
        label_seq = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sequence, label_seq

# -------------------------------
# Main Processing Pipeline
# -------------------------------
# Load metadata CSV.
df = pd.read_csv("/kaggle/input/nexar-collision-prediction/train.csv")
# Ensure video ids are zero-padded (e.g., "00001").
df["id"] = df["id"].apply(lambda x: str(x).zfill(5))

# Limit processing to the first 500 videos.
df_subset = df.head(500)

# Set the video directory.
TRAIN_VIDEO_DIR = "/kaggle/input/nexar-collision-prediction/train"

# Extract features and per-frame labels for the subset.
sequences, labels = extract_features_and_labels(TRAIN_VIDEO_DIR, df_subset)

print("Extracted sequences shape:", sequences.shape)
print("Extracted labels shape:", labels.shape)
np.save("train_features.npy", sequences)
np.save("train_labels.npy", labels)
print("Features and labels saved successfully!")


