import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class DeepFakeDetector3D(nn.Module):
    def __init__(self, input_channels=4, max_frames=32):
        super(DeepFakeDetector3D, self).__init__()
        
        # Input channels = 3 (RGB) + 1 (flow) = 4
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Only pool spatially
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Only pool spatially
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # Starting with (B, 4, T, 128, 128)
        # After pool1: (B, 16, T, 64, 64)
        # After pool2: (B, 32, T, 32, 32)
        frames_used = max_frames - 1  # Since flow has T-1 elements
        fc_input_size = 32 * frames_used * 32 * 32
        
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, 1)
        
        # Loss components
        self.flow_loss_weight = 0.1
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Print shape before flattening for debugging
        # print(f"Shape before flattening: {x.shape}")
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def compute_loss(self, outputs, targets, flow_tensor):
        # Classification loss
        cls_loss = self.bce_loss(outputs.squeeze(), targets)
        
        # Temporal consistency loss
        if flow_tensor.size(2) > 1:  # Only if we have multiple frames
            temp_loss = torch.mean(torch.abs(flow_tensor[:, :, 1:] - flow_tensor[:, :, :-1]))
            return cls_loss + self.flow_loss_weight * temp_loss
        return cls_loss

def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude

def process_video(video_path, resize=(128, 128), max_frames=32):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video file {video_path}")
        return np.zeros((1, *resize, 3)), np.zeros((1, *resize))

    frames = []
    flow_magnitudes = []

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        print(f"Warning: No frames were read from {video_path}")
        return np.zeros((1, *resize, 3)), np.zeros((1, *resize))

    if resize:
        prev_frame = cv2.resize(prev_frame, resize)

    frames.append(prev_frame)
    frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break

        if resize:
            frame = cv2.resize(frame, resize)

        magnitude = compute_optical_flow(prev_frame, frame)
        flow_magnitudes.append(magnitude)
        frames.append(frame)

        prev_frame = frame
        frame_count += 1

    cap.release()
    
    # Ensure we have at least one frame
    if len(frames) == 0:
        return np.zeros((1, *resize, 3)), np.zeros((1, *resize))
    
    # Ensure we have at least max_frames (pad if necessary)
    if max_frames and len(frames) < max_frames:
        pad_frames = max_frames - len(frames)
        # Duplicate the last frame
        for _ in range(pad_frames):
            frames.append(frames[-1].copy())
        # Duplicate the last flow magnitude
        if len(flow_magnitudes) > 0:
            for _ in range(pad_frames-1):  # One less for flow
                flow_magnitudes.append(flow_magnitudes[-1].copy())
        else:
            # If no flow magnitudes, create zero arrays
            flow_magnitudes = [np.zeros(resize) for _ in range(max_frames-1)]
    
    # Trim to max_frames if needed
    if max_frames:
        frames = frames[:max_frames]
        flow_magnitudes = flow_magnitudes[:max_frames-1]  # One less for flow
    
    return np.array(frames), np.array(flow_magnitudes)

class DeepFakeDataset(Dataset):
    def __init__(self, video_dir, video_files, labels, max_frames=32):
        self.video_dir = video_dir
        self.video_files = video_files
        self.labels = labels
        self.max_frames = max_frames
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        try:
            video_filename = self.video_files[idx]
            video_path = os.path.join(self.video_dir, video_filename)
            label = self.labels[idx]
        
            # Load video and flow data
            frames, flow_magnitudes = process_video(video_path, max_frames=self.max_frames)
            
            # Handle the case when frames or flow_magnitudes are empty
            if len(frames) == 0:
                # Return dummy tensors with proper dimensions
                rgb_tensor = torch.zeros((3, self.max_frames, 128, 128))
                flow_tensor = torch.zeros((1, self.max_frames-1, 128, 128))
                return rgb_tensor, flow_tensor, torch.tensor(label, dtype=torch.float32)
            
            # Convert to tensors
            # RGB channels: (T, H, W, C) → (C, T, H, W)
            rgb_tensor = torch.tensor(frames).permute(3, 0, 1, 2).float() / 255.0
            
            # Flow magnitude: (T-1, H, W) → (1, T-1, H, W)
            flow_tensor = torch.tensor(flow_magnitudes).unsqueeze(0).float()
            
            # Ensure flow_tensor matches expected size
            if flow_tensor.shape[1] < self.max_frames - 1:
                # Pad with zeros if needed
                padding = self.max_frames - 1 - flow_tensor.shape[1]
                if padding > 0:
                    padding_tensor = torch.zeros((1, padding, 128, 128))
                    flow_tensor = torch.cat([flow_tensor, padding_tensor], dim=1)
            
            # Ensure RGB tensor has correct temporal dimension
            if rgb_tensor.shape[1] < self.max_frames:
                padding = self.max_frames - rgb_tensor.shape[1]
                if padding > 0:
                    # Duplicate the last frame
                    last_frame = rgb_tensor[:, -1:, :, :]
                    padding_tensor = last_frame.repeat(1, padding, 1, 1)
                    rgb_tensor = torch.cat([rgb_tensor, padding_tensor], dim=1)
            
            # Trim tensors if they exceed max_frames
            rgb_tensor = rgb_tensor[:, :self.max_frames]
            flow_tensor = flow_tensor[:, :self.max_frames-1]
            
            return rgb_tensor, flow_tensor, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error processing video {self.video_files[idx]}: {e}")
            # Return dummy tensors
            rgb_tensor = torch.zeros((3, self.max_frames, 128, 128))
            flow_tensor = torch.zeros((1, self.max_frames-1, 128, 128))
            return rgb_tensor, flow_tensor, torch.tensor(label, dtype=torch.float32)

def train_epoch(model, dataloader, optimizer, device="cpu"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (frames, flow, labels) in enumerate(tqdm(dataloader)):
        try:
            # Move data to device
            frames = frames.to(device)
            flow = flow.to(device)
            labels = labels.to(device)
            
            # Get the actual number of frames (could be less than max_frames)
            # Use the first frame's RGB + flow magnitude for each time step
            # Frames: (B, 3, T, H, W), Flow: (B, 1, T-1, H, W)
            
            # Prepare combined input
            rgb_part = frames[:, :, :-1]  # Use all but the last frame
            combined_input = torch.cat([rgb_part, flow], dim=1)  # (B, 4, T-1, H, W)
            
            # Debug dimensions
            # print(f"Combined input shape: {combined_input.shape}")
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(combined_input)
            
            # Compute loss
            loss = model.compute_loss(outputs, labels, flow)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            predicted = torch.sigmoid(outputs.squeeze()) > 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        except Exception as e:
            print(f"Error in training step: {e}")
            continue
    
    if total == 0:  # Avoid division by zero
        return running_loss, 0
        
    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy

# Main code to run
def main():
    # Path to the dataset
    video_dir = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/"
    metadata_path = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json"

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create a list of video files with their labels
    video_files = []
    for key in metadata.keys():
        video_files.append(key)
        
    label_map = {'REAL': 0, 'FAKE': 1}
    labels = [label_map[metadata[f]['label']] for f in video_files]
    
    # Print some information about the dataset
    print(f"Total videos: {len(video_files)}")
    print(f"First 5 videos: {video_files[:5]}")
    print(f"First 5 labels: {labels[:5]}")
    
    # Use a smaller subset for testing if needed
    max_videos = 400  # Limit number of videos for testing
    video_files = video_files[:max_videos]
    labels = labels[:max_videos]
    
    # Define max frames for consistent dimensionality
    max_frames = 32
    
    # Create dataset and dataloader
    dataset = DeepFakeDataset(video_dir, video_files, labels, max_frames=max_frames)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and optimizer
    model = DeepFakeDetector3D(input_channels=4, max_frames=max_frames).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for a few epochs
    num_epochs = 200
    for epoch in range(num_epochs):
        loss, accuracy = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), "deepfake_detector.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()




