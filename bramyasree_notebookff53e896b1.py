# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install torch torchvision opencv-python-headless numpy



import os
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2



import os

dataset_path = "/kaggle/input/state-farm-distracted-driver-detection/imgs"
print("Folders in imgs directory:", os.listdir(dataset_path))



from torch.utils.data import random_split

# Load train dataset only
train_dir = os.path.join(dataset_path, 'train')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
full_dataset = ImageFolder(train_dir, transform=transform)

# Split dataset into train and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")



# Dummy model simulating DWPose keypoint extractor output
class DummyPoseModel(nn.Module):
    def __init__(self):
        super(DummyPoseModel, self).__init__()
    
    def forward(self, x):
        batch_size = x.size(0)
        num_keypoints = 17  # hypothetical number of keypoints
        keypoints = torch.rand(batch_size, num_keypoints, 2) * 224  # scale to input image size
        return keypoints
7
pose_model = DummyPoseModel()
pose_model.eval()

# Function to extract keypoints from a batch of images
def extract_keypoints(images, model):
    with torch.no_grad():
        keypoints = model(images)
    return keypoints.cpu().numpy()

# Demo extraction on one batch from training loader
for imgs, labels in train_loader:
    keypoints = extract_keypoints(imgs, pose_model)
    print("Keypoints output shape:", keypoints.shape)
    break



class SimpleKalmanFilter:
    def __init__(self):
        self.x = None
        self.P = np.eye(2) * 1000  # Covariance matrix
        self.A = np.eye(2)          # State transition matrix
        self.Q = np.eye(2)          # Process noise covariance
        self.R = np.eye(2) * 0.1    # Measurement noise covariance
    
    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x
        
        # Predict
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q
        
        # Kalman Gain
        K = P_pred @ np.linalg.inv(P_pred + self.R)
        
        # Update
        self.x = x_pred + K @ (measurement - x_pred)
        self.P = (np.eye(2) - K) @ P_pred
        
        return self.x

# Example usage on keypoints of one frame
kf_list = [SimpleKalmanFilter() for _ in range(17)]  # One Kalman Filter per keypoint

def kalman_smooth(keypoints):
    smoothed = []
    for i, point in enumerate(keypoints):
        smoothed_point = kf_list[i].update(point)
        smoothed.append(smoothed_point)
    return np.array(smoothed)

# Demo smoothing on first image keypoints
smoothed_points = kalman_smooth(keypoints[0])
print("Smoothed keypoints shape:", smoothed_points.shape)





