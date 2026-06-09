import os
import gc
import time
import warnings
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CyclicLR
torch.nn.utils.clip_grad_norm_

warnings.filterwarnings("ignore")

# Check GPU availability and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# Suppress unnecessary formatting warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths to the CSV files
train_csv_path = '/kaggle/input/nexar-collision-prediction/train.csv'
test_csv_path = '/kaggle/input/nexar-collision-prediction/test.csv'
submission_csv_path = '/kaggle/input/nexar-collision-prediction/sample_submission.csv'

# Load the CSV files
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
submission_df = pd.read_csv(submission_csv_path)

# Display the first few rows of the DataFrames
print("Train.csv:")
print(train_df.head())

print("\nTest.csv:")
print(test_df.head())

print("\nSample Submission:")
print(submission_df.head())

# Optional: handle NaN values if needed, filling with zero or another value
train_df['time_of_event'] = train_df['time_of_event'].fillna(0)
train_df['time_of_alert'] = train_df['time_of_alert'].fillna(0)


def extract_keyframes(video_path, num_frames=12, target_size=(160, 160)):
    """
    Extracts key frames from the video, focusing on the final part where collisions typically occur.
    Uses exponential distribution to give more weight to frames closer to the end.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open the video: {video_path}")
        return np.zeros((num_frames, target_size[0], target_size[1], 3), dtype=np.uint8)
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        print(f"Video without frames: {video_path}")
        cap.release()
        return np.zeros((num_frames, target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Calculate video duration in seconds
    duration = total_frames / fps if fps > 0 else 0
    
    # If the video is short (less than 10 seconds), distribute frames uniformly
    if duration < 10:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Concentrate 80% of frames in the last 3 seconds (critical area)
        end_frames = int(num_frames * 0.8)
        start_frames = num_frames - end_frames
        
        # Calculate the starting index for the last 3 seconds
        last_seconds = 3
        last_frame_count = min(int(fps * last_seconds), total_frames - 1)
        start_idx = max(0, total_frames - last_frame_count)
        
        # Exponential distribution to give more weight to the last frames
        # This creates indices that are more densely packed toward the end
        end_indices = np.array([
            start_idx + int((total_frames - start_idx - 1) * (i/end_frames)**2) 
            for i in range(1, end_frames + 1)
        ])
        
        # Initial frames distributed uniformly for context
        begin_indices = np.linspace(0, start_idx - 1, start_frames, dtype=int) if start_idx > 0 else np.zeros(start_frames, dtype=int)
        
        # Combine indices
        frame_indices = np.concatenate([begin_indices, end_indices])
    
    # Extract selected frames
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Use higher resolution and better interpolation
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
    
    cap.release()
    return np.array(frames, dtype=np.uint8)

# First, define transformation classes in the global scope
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, frames):
        if np.random.random() < self.p:
            return frames[:, :, ::-1, :].copy()  # horizontally flip each frame
        return frames

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0):
        self.brightness = brightness
        self.contrast = contrast
        
    def __call__(self, frames):
        # Apply brightness jitter
        if self.brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1-self.brightness), 1+self.brightness)
            frames = frames * brightness_factor
            frames = np.clip(frames, 0, 255)
        
        # Apply contrast jitter
        if self.contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1-self.contrast), 1+self.contrast)
            frames = (frames - 128) * contrast_factor + 128
            frames = np.clip(frames, 0, 255)
            
        return frames

class AddFog(object):
    def __call__(self, frames):
        fog = np.random.uniform(0.7, 0.9, frames.shape).astype(np.float32)
        return frames * 0.8 + fog * 50  # Adjusted for 0-255 scale

class AddRain(object):
    def __call__(self, frames):
        h, w = frames.shape[1:3]
        rain = np.random.uniform(0, 1, (len(frames), h, w, 1)).astype(np.float32)
        rain = (rain > 0.97).astype(np.float32) * 200  # White rain drops
        return np.clip(frames * 0.9 + rain, 0, 255)  # Darken a bit and add drops

class RandomApply(object):
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p
        
    def __call__(self, frames):
        if np.random.random() < self.p:
            return self.transform(frames)
        return frames

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, frames):
        for t in self.transforms:
            frames = t(frames)
        return frames

class ToTensor(object):
    def __call__(self, frames):
        # Convert from (T, H, W, C) to (T, C, H, W)
        frames = frames.transpose(0, 3, 1, 2)
        # Convert to tensor and normalize to [0, 1]
        return torch.from_numpy(frames).float() / 255.0

def get_video_transforms():
    """
    Returns transformations for data augmentation in videos.
    """
    return {
        'train': Compose([
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.3, contrast=0.3),
            RandomApply(AddFog(), p=0.15),
            RandomApply(AddRain(), p=0.15),
            RandomApply(RandomNoise(0.05), p=0.2), 
            RandomApply(RandomOcclusion(), p=0.1),
            # Adicione estas novas transformaÃ§Ãµes
            RandomApply(AddSunGlare(), p=0.1),
            RandomApply(SimulateNight(), p=0.1),
            RandomApply(Motion(), p=0.2),
            ToTensor()
        ]),
        'val': Compose([
            ToTensor()  # Only tensor conversion for validation
        ])
    }

class AddSunGlare(object):
    """Simulates the effect of sun glare on the windshield."""
    def __call__(self, frames):
        h, w = frames.shape[1:3]
        
        # Random position for sun glare
        x = np.random.randint(0, w)
        y = np.random.randint(0, h // 3)  # More likely at the top
        radius = np.random.randint(30, 80)
        intensity = np.random.uniform(0.6, 0.9)
        
        result = frames.copy().astype(np.float32)
        
        for i in range(len(frames)):
            # Create a circular gradient to simulate glare
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
            mask = dist <= radius
            
            # Apply glare with gradient
            glow = np.maximum(0, (1 - dist / radius)) * intensity * 255
            result[i, :, :, :] += np.repeat(glow[:, :, np.newaxis], 3, axis=2) * mask[:, :, np.newaxis]
            
        return np.clip(result, 0, 255).astype(np.uint8)

class SimulateNight(object):
    """Simulates low-light/night conditions."""
    def __call__(self, frames):
        # Reduce overall brightness
        darkness = np.random.uniform(0.3, 0.6)
        result = frames.astype(np.float32) * darkness
        
        # Add noise to simulate low-light sensor conditions
        noise = np.random.normal(0, 5, frames.shape).astype(np.float32)
        result += noise
        
        return np.clip(result, 0, 255).astype(np.uint8)

class Motion(object):
    """Simulates motion blur."""
    def __call__(self, frames):
        # Choose direction and magnitude of motion blur
        kernel_size = np.random.choice([3, 5, 7])
        angle = np.random.uniform(0, 360)
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Draw a line at the specified angle
        x1 = center
        y1 = center
        x2 = int(center + (kernel_size - 1) / 2 * np.cos(np.radians(angle)))
        y2 = int(center + (kernel_size - 1) / 2 * np.sin(np.radians(angle)))
        
        cv2.line(kernel, (x1, y1), (x2, y2), 1.0)
        kernel = kernel / np.sum(kernel)
        
        result = np.zeros_like(frames)
        
        # Apply the kernel to each frame and channel
        for i in range(len(frames)):
            for c in range(3):
                result[i, :, :, c] = cv2.filter2D(frames[i, :, :, c], -1, kernel)
                
        return result

class RandomNoise(object):
    """
    Applies random Gaussian noise to video frames for data augmentation.
    
    This transformation helps the model become more robust to noise
    that may be present in real-world video data.
    
    Args:
        std (float): Standard deviation of the Gaussian noise as a fraction
                     of the pixel value range (default: 0.05)
    """
    def __init__(self, std=0.05):
        self.std = std
        
    def __call__(self, frames):
        """
        Apply random noise to the input frames.
        
        Args:
            frames (numpy.ndarray): Input video frames of shape (T, H, W, C)
                                   where T is number of frames
        
        Returns:
            numpy.ndarray: Noise-augmented frames, clipped to valid pixel range [0, 255]
        """
        # Generate Gaussian noise with specified standard deviation
        noise = np.random.normal(0, self.std * 255, frames.shape).astype(np.float32)
        
        # Add noise and clip to valid pixel range
        return np.clip(frames + noise, 0, 255).astype(np.uint8)


class RandomOcclusion(object):
    """
    Simulates occlusion in video frames by adding black rectangles.
    
    This transformation helps the model learn to handle partial occlusions
    that may occur in real-world scenarios when objects block the camera view.
    """
    def __call__(self, frames):
        """
        Apply random occlusion to the input frames.
        
        Args:
            frames (numpy.ndarray): Input video frames of shape (T, H, W, C)
                                   where T is number of frames
        
        Returns:
            numpy.ndarray: Frames with random occlusion applied
        """
        # Get frame dimensions
        h, w = frames.shape[1:3]
        
        # Define occlusion area: 10-25% of the image
        occl_h = np.random.randint(int(h * 0.1), int(h * 0.25))
        occl_w = np.random.randint(int(w * 0.1), int(w * 0.25))
        
        # Randomly position the occlusion
        occl_x = np.random.randint(0, w - occl_w)
        occl_y = np.random.randint(0, h - occl_h)
        
        # Create a copy to avoid modifying the original frames
        frames_copy = frames.copy()
        
        # Apply occlusion to all frames by setting pixels to zero (black)
        for i in range(len(frames)):
            frames_copy[i, occl_y:occl_y+occl_h, occl_x:occl_x+occl_w, :] = 0
            
        return frames_copy
        
def compute_optical_flow(frames, skip_frames=1):
    """Calculates optical flow skipping some frames to reduce processing."""
    if len(frames) < 2:
        return np.zeros((1, frames.shape[1], frames.shape[2], 2), dtype=np.float32)
    
    flows = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    
    for i in range(1, len(frames), skip_frames):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        try:
            # Reduce parameters for faster calculation
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                               None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
        except Exception as e:
            print(f"Error calculating optical flow: {str(e)}")
            flows.append(np.zeros((frames.shape[1], frames.shape[2], 2), dtype=np.float32))
            
        prev_gray = curr_gray
    
    if not flows:
        return np.zeros((1, frames.shape[1], frames.shape[2], 2), dtype=np.float32)
        
    return np.array(flows, dtype=np.float32)

def process_video(args):
    """
    Function to process an individual video.
    """
    video_path, video_id, num_frames = args
    try:
        # Extract frames with higher resolution
        frames = extract_keyframes(video_path, num_frames=num_frames, target_size=(160, 160))
        
        # Calculate optical flow
        optical_flow = compute_optical_flow(frames, skip_frames=1)
        
        # We return NumPy arrays instead of applying transformations now
        return video_id, {
            'frames': frames,
            'optical_flow': optical_flow,
        }
    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
        return video_id, None

def parallel_preprocess_dataset(video_dir, video_ids, num_frames=8, num_workers=4):
    """
    Pre-processes multiple videos in parallel.
    """
    args_list = []
    for video_id in video_ids:
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            args_list.append((video_path, video_id, num_frames))
    
    start_time = time.time()
    print(f"Starting parallel pre-processing of {len(args_list)} videos with {num_workers} workers...")
    
    processed_data = {}
    with Pool(num_workers) as p:
        results = p.map(process_video, args_list)
        for video_id, data in results:
            if data is not None:
                processed_data[video_id] = data
    
    print(f"Pre-processing completed in {time.time() - start_time:.2f} seconds.")
    print(f"Processed {len(processed_data)} out of {len(args_list)} videos.")
    
    return processed_data


class DashcamDataset(Dataset):
   def __init__(self, video_dir, annotations, transform=None, num_frames=8):
       self.video_dir = video_dir
       self.annotations = annotations
       self.transform = transform
       self.num_frames = num_frames
       self.video_ids = list(annotations.keys())

   def __len__(self):
       return len(self.video_ids)

   def __getitem__(self, idx):
       video_id = self.video_ids[idx]
       video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
       
       try:
           # Check if the file exists
           if not os.path.exists(video_path):
               print(f"Video not found: {video_path}")
               raise FileNotFoundError(f"File not found: {video_path}")
               
           # Extract frames with reduced resolution
           frames = extract_keyframes(video_path, self.num_frames, target_size=(112, 112))
           
           # Calculate optical flow with skip_frames
           optical_flow = compute_optical_flow(frames, skip_frames=1)
           
           # Apply transformations
           if self.transform:
               frames = self.transform(frames)
           else:
               # Convert to tensor manually
               frames = torch.from_numpy(frames.transpose(0, 3, 1, 2)).float() / 255.0
           
           # Convert optical flow to tensor
           optical_flow = torch.from_numpy(optical_flow.transpose(0, 3, 1, 2)).float()
           
           # Load label and alert time
           label = self.annotations[video_id]['label']
           alert_time = self.annotations[video_id].get('alert_time', 0)
           
           return {
               'frames': frames,
               'optical_flow': optical_flow,
               'label': torch.tensor(label).float(),
               'alert_time': torch.tensor(alert_time).float(),
               'video_id': video_id
           }
               
       except Exception as e:
           print(f"Error processing video {video_id}: {str(e)}")
           # Create a placeholder for this video
           dummy_frames = torch.zeros((self.num_frames, 3, 112, 112))
           dummy_flow = torch.zeros((max(1, self.num_frames-1), 2, 112, 112))
           return {
               'frames': dummy_frames,
               'optical_flow': dummy_flow,
               'label': torch.tensor(self.annotations[video_id]['label']).float(),
               'alert_time': torch.tensor(self.annotations[video_id].get('alert_time', 0)).float(),
               'video_id': video_id
           }

class PreprocessedDashcamDataset(Dataset):
   def __init__(self, processed_data, annotations, transform=None):
       self.processed_data = processed_data
       self.annotations = annotations
       self.transform = transform
       self.video_ids = list(processed_data.keys())
       
   def __len__(self):
       return len(self.video_ids)
   
   def __getitem__(self, idx):
       video_id = self.video_ids[idx]
       data = self.processed_data[video_id]
       
       # Get frames and optical flow
       frames = data['frames']
       optical_flow = data['optical_flow']
       
       # Apply transformations to frames
       if self.transform:
           frames_tensor = self.transform(frames)
       else:
           # Convert to tensor manually
           frames_tensor = torch.from_numpy(frames.transpose(0, 3, 1, 2)).float() / 255.0
       
       # Convert optical flow to tensor
       optical_flow_tensor = torch.from_numpy(optical_flow.transpose(0, 3, 1, 2)).float()
       
       # Load label and alert time
       label = self.annotations[video_id]['label']
       alert_time = self.annotations[video_id].get('alert_time', 0)
       
       return {
           'frames': frames_tensor,
           'optical_flow': optical_flow_tensor,
           'label': torch.tensor(label).float(),
           'alert_time': torch.tensor(alert_time).float(),
           'video_id': video_id
       }


# Spatial Attention mechanism
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch, channels, height, width]
        attention_map = self.conv(x)
        return x * attention_map

class MobileNetVisualStream(nn.Module):
    def __init__(self, output_features=64):
        super(MobileNetVisualStream, self).__init__()
        
        print("Initializing MobileNetV2 model...")
        
        # Create the model WITHOUT pre-trained weights first
        mobilenet = torchvision.models.mobilenet_v2(pretrained=False)
        
        # Check if we have local Hugging Face weights
        hf_weights_path = "/kaggle/input/googlemobilenet-v2-1-0-224/pytorch_model.bin"
        
        if os.path.exists(hf_weights_path):
            print(f"Found Hugging Face pre-trained weights: {hf_weights_path}")
            try:
                # Load the weights from Hugging Face
                state_dict = torch.load(hf_weights_path, map_location='cpu')
                
                # Map the weights from the Hugging Face format to the torchvision format
                mobilenet_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('mobilenet_v2.'):
                        # Remove the prefix 'mobilenet_v2.'
                        mobilenet_state_dict[k[13:]] = v
                
                # Load the mapped weights
                mobilenet.load_state_dict(mobilenet_state_dict, strict=False)
                print("Successfully loaded weights from Hugging Face model")
            except Exception as e:
                print(f"Error loading Hugging Face weights: {e}")
                print("Using random initialization")
        else:
            print("Hugging Face weights not found. Using random initialization.")
        
        # Divide into blocks for adding attention between them
        self.block1 = nn.Sequential(*list(mobilenet.features)[:7])  # First layers
        self.spatial_attention1 = SpatialAttention(32)  # Adjust for number of channels
        
        self.block2 = nn.Sequential(*list(mobilenet.features)[7:14])  # Middle layers
        self.spatial_attention2 = SpatialAttention(96)  # Adjust for number of channels
        
        self.block3 = nn.Sequential(*list(mobilenet.features)[14:])  # Last layers
        
        # Projection layer to reduce dimensionality
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, output_features),
            nn.ReLU()
        )
        
        self.output_features = output_features
        
    def forward(self, x):
        # x: [batch, T, C, H, W]
        batch_size, seq_len, c, h, w = x.size()
        
        # Process each frame individually
        features = []
        for t in range(seq_len):
            # Extract features with spatial attention
            feat = self.block1(x[:, t])
            feat = self.spatial_attention1(feat)
            
            feat = self.block2(feat)
            feat = self.spatial_attention2(feat)
            
            feat = self.block3(feat)
            feat = self.projection(feat)
            
            features.append(feat)
        
        # Stack along the temporal dimension
        output = torch.stack(features, dim=1)  # [batch, T, output_features]
        
        return output

class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(dim, dim)  # Query projection
        self.key = nn.Linear(dim, dim)    # Key projection
        self.value = nn.Linear(dim, dim)  # Value projection
        self.scale = dim ** -0.5          # Scaling factor for dot products
        
    def forward(self, x):
        # Project inputs to queries, keys and values
        q = self.query(x)  # [batch, seq_len, dim]
        k = self.key(x)    # [batch, seq_len, dim]
        v = self.value(x)  # [batch, seq_len, dim]
        
        # Compute scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch, seq_len, seq_len]
        attn = F.softmax(attn, dim=-1)                 # Apply softmax to get attention weights
        
        # Apply attention weights to values
        out = attn @ v  # [batch, seq_len, dim]
        return out

# Optimized version of OpticalFlowStream class
class OpticalFlowStream(nn.Module):
    def __init__(self, output_features=48):
        super(OpticalFlowStream, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 24, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(24, 48, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(48, output_features, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(output_features),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        
        # Add temporal attention specific to optical flow
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(output_features, output_features, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch, T, C, H, W] -> [batch, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        features = x.squeeze(-1).squeeze(-1)  # [batch, features, T]
        
        # Apply temporal attention
        attention = self.temporal_attention(features)
        weighted_features = features * attention
        
        return weighted_features

# Optimized version of FusionTransformer class
class FusionTransformer(nn.Module):
   def __init__(self, visual_dim=64, flow_dim=32, output_dim=64, nhead=4, num_layers=2):
       super(FusionTransformer, self).__init__()
       # Use simpler architecture to save memory
       self.visual_proj = nn.Linear(visual_dim, output_dim)
       self.flow_proj = nn.Linear(flow_dim, output_dim)
       
       # Use an LSTM layer instead of Transformer (less memory intensive)
       self.lstm = nn.LSTM(
           input_size=output_dim,
           hidden_size=output_dim,
           num_layers=1,
           batch_first=True
       )
       
       # Output layer
       self.output_layer = nn.Linear(output_dim, output_dim)
   
   def forward(self, visual_feat, flow_feat):
       # Temporal pooling
       visual_pool = visual_feat.mean(dim=2)  # [batch, visual_dim]
       flow_pool = flow_feat.mean(dim=2)      # [batch, flow_dim]
       
       # Projection to common dimension
       visual_proj = self.visual_proj(visual_pool)  # [batch, output_dim]
       flow_proj = self.flow_proj(flow_pool)        # [batch, output_dim]
       
       # Concatenate as sequence
       concat = torch.stack([visual_proj, flow_proj], dim=1)  # [batch, 2, output_dim]
       
       # Pass through LSTM
       lstm_out, _ = self.lstm(concat)
       
       # Final pooling
       output = lstm_out.mean(dim=1)        # [batch, output_dim]
       output = self.output_layer(output)   # [batch, output_dim]
       
       return output

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism that learns to focus on important time steps in a sequence.
    
    This module implements a self-attention mechanism where each time step attends
    to all other time steps, allowing the model to capture temporal dependencies.
    
    Args:
        dim (int): Dimension of the input feature space
    """
    def __init__(self, dim):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(dim, dim)  # Query projection
        self.key = nn.Linear(dim, dim)    # Key projection
        self.value = nn.Linear(dim, dim)  # Value projection
        self.scale = dim ** -0.5          # Scaling factor for dot products
        
    def forward(self, x):
        """
        Forward pass of the temporal attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, dim]
            
        Returns:
            torch.Tensor: Attention-weighted output of same shape as input
        """
        # Project inputs to queries, keys and values
        q = self.query(x)  # [batch, seq_len, dim]
        k = self.key(x)    # [batch, seq_len, dim]
        v = self.value(x)  # [batch, seq_len, dim]
        
        # Compute scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch, seq_len, seq_len]
        attn = F.softmax(attn, dim=-1)                 # Apply softmax to get attention weights
        
        # Apply attention weights to values
        out = attn @ v  # [batch, seq_len, dim]
        return out


class LightweightMultiStreamModel(nn.Module):
    def __init__(self, output_dim=64):
        super(LightweightMultiStreamModel, self).__init__()
        # Optimized parameters
        visual_features = 64
        flow_features = 32
        
        # Visual and flow processing streams
        self.visual_stream = MobileNetVisualStream(output_features=visual_features)
        self.flow_stream = OpticalFlowStream(output_features=flow_features)
        
        # Projection layers to align feature dimensions
        self.visual_proj = nn.Linear(visual_features, output_dim)
        self.flow_proj = nn.Linear(flow_features, output_dim)
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(output_dim)
        
        # Fusion layer after attention
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        
        # Stronger dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Classification branch with additional regularization
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Regression branch for alert time prediction
        self.regressor = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
        # Flag to track if we've created dynamic projections already
        self.dynamic_projections_created = False
    
    def forward(self, frames, optical_flow):
        # Process visual and optical flow streams
        visual_feat = self.visual_stream(frames)  # [batch, T, features]
        flow_feat = self.flow_stream(optical_flow)  # [batch, features, T-1]
        
        # Check tensor shapes for debugging
        batch_size = frames.size(0)
    
        # Reshape visual features based on their dimensions
        if len(visual_feat.shape) == 5:  # If [batch, features, 1, 1, 1]
            visual_feat = visual_feat.squeeze(-1).squeeze(-1).squeeze(-1)  # [batch, features]
            visual_feat = visual_feat.unsqueeze(1)  # [batch, 1, features]
        elif len(visual_feat.shape) == 3:  # If [batch, features, T]
            visual_feat = visual_feat.permute(0, 2, 1)  # [batch, T, features]
        elif len(visual_feat.shape) == 2:  # If [batch, T*features]
            # Assuming num_frames is a known constant (e.g., 8)
            num_frames = frames.size(1)
            visual_feat = visual_feat.view(batch_size, num_frames, -1)  # [batch, T, features_per_frame]
    
        # Reshape optical flow features based on their dimensions
        if len(flow_feat.shape) == 5:  # If [batch, features, 1, 1, 1]
            flow_feat = flow_feat.squeeze(-1).squeeze(-1).squeeze(-1)  # [batch, features]
            flow_feat = flow_feat.unsqueeze(1)  # [batch, 1, features]
        elif len(flow_feat.shape) == 3:  # If [batch, features, T]
            flow_feat = flow_feat.permute(0, 2, 1)  # [batch, T, features]
        elif len(flow_feat.shape) == 2:  # If [batch, T*features]
            # Assuming num_frames-1 for optical flow
            num_flow_frames = optical_flow.size(1)
            flow_feat = flow_feat.view(batch_size, num_flow_frames, -1)  # [batch, T-1, features_per_frame]
    
        # Ensure at least one temporal dimension
        if len(visual_feat.shape) == 2:  # [batch, features]
            visual_feat = visual_feat.unsqueeze(1)  # [batch, 1, features]
    
        if len(flow_feat.shape) == 2:  # [batch, features]
            flow_feat = flow_feat.unsqueeze(1)  # [batch, 1, features]
        
        # Create dynamic projection layers if shapes don't match expected dimensions
        visual_dim = visual_feat.shape[2]
        flow_dim = flow_feat.shape[2]
        
        # Create dynamic projection layers only once and without log messages
        if not self.dynamic_projections_created:
            if visual_dim != 64:
                # Print only once during first forward pass
                print(f"Creating dynamic visual projection from {visual_dim} to {self.visual_proj.out_features}")
                self.visual_proj = nn.Linear(visual_dim, self.visual_proj.out_features).to(visual_feat.device)
                
            if flow_dim != 32:
                # Print only once during first forward pass
                print(f"Creating dynamic flow projection from {flow_dim} to {self.flow_proj.out_features}")
                self.flow_proj = nn.Linear(flow_dim, self.flow_proj.out_features).to(flow_feat.device)
            
            # Set flag to avoid repeated messages
            self.dynamic_projections_created = True
        else:
            # On subsequent calls, just make sure the dimensions match without printing
            if visual_dim != 64 and visual_dim != self.visual_proj.in_features:
                self.visual_proj = nn.Linear(visual_dim, self.visual_proj.out_features).to(visual_feat.device)
            
            if flow_dim != 32 and flow_dim != self.flow_proj.in_features:
                self.flow_proj = nn.Linear(flow_dim, self.flow_proj.out_features).to(flow_feat.device)
    
        # Project features to the same dimensionality
        visual_proj = self.visual_proj(visual_feat)  # [batch, T, output_dim]
        flow_proj = self.flow_proj(flow_feat)  # [batch, T, output_dim]
    
        # Simple combination if temporal dimensions are different
        if visual_proj.size(1) != flow_proj.size(1):
            # Compute temporal average for both features
            visual_proj_avg = visual_proj.mean(dim=1, keepdim=True)  # [batch, 1, output_dim]  
            flow_proj_avg = flow_proj.mean(dim=1, keepdim=True)  # [batch, 1, output_dim]
        
            # Concatenate averaged features
            combined_feat = torch.cat([visual_proj_avg, flow_proj_avg], dim=1)  # [batch, 2, output_dim]
        else:
            # Combine features as a sequence
            combined_feat = (visual_proj + flow_proj) / 2  # [batch, T, output_dim]  
    
        # Apply temporal attention 
        attended_feat = self.temporal_attention(combined_feat)  # [batch, T, output_dim]
    
        # Temporal pooling to obtain a single representation
        fusion_feat = attended_feat.mean(dim=1)  # [batch, output_dim]
    
        # Fusion layer
        fusion_feat = self.fusion_layer(fusion_feat)
    
        # Apply dropout regularization  
        fusion_feat = self.dropout(fusion_feat)
    
        # Classification and regression
        score = self.classifier(fusion_feat)
        alert_pred = self.regressor(fusion_feat)
    
        return score, alert_pred


# Custom loss function for time prediction
def custom_time_loss(pred_time, true_time):
    """
    Custom loss function for time prediction that penalizes late predictions more heavily.
    
    This loss function combines standard MSE with an additional penalty
    when the predicted time is later than the ground truth time.
    
    Args:
        pred_time (torch.Tensor): Predicted time values
        true_time (torch.Tensor): Ground truth time values
        
    Returns:
        torch.Tensor: Combined loss value
    """
    # Calculate standard MSE loss
    base_loss = F.mse_loss(pred_time, true_time)
    
    # Identify and penalize late predictions (when pred_time > true_time)
    late_mask = (pred_time > true_time).float()
    late_penalty = F.mse_loss(pred_time * late_mask, true_time * late_mask) * 1.5
    
    # Combine base loss with late prediction penalty
    return base_loss + late_penalty

class FocalLoss(nn.Module):
    def __init__(self, alpha_init=0.25, gamma_init=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha_init
        self.gamma = gamma_init
        self.momentum = 0.9
        self.batch_history = []  # Track the recent difficulty history of batches
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        
        # More detailed analysis of difficulty
        batch_difficulty = 1 - pt.mean().item()
        self.batch_history.append(batch_difficulty)
        
        # Dynamically adjust gamma based on recent history
        if len(self.batch_history) > 5:
            recent_difficulty = sum(self.batch_history[-5:]) / 5
            trend = (recent_difficulty - sum(self.batch_history[-10:-5]) / 5) if len(self.batch_history) > 10 else 0
            
            # Increase gamma for hard examples with a worsening trend
            if recent_difficulty > 0.4 and trend > 0:
                target_gamma = 3.0
            # Decrease gamma for easy examples with an improving trend
            elif recent_difficulty < 0.3 and trend < 0:
                target_gamma = 1.5
            else:
                target_gamma = 2.0 + recent_difficulty
                
            self.gamma = self.momentum * self.gamma + (1 - self.momentum) * target_gamma
        else:
            self.gamma = self.momentum * self.gamma + (1 - self.momentum) * (2.0 + batch_difficulty)
        
        # Adjust alpha to better balance between classes
        pos_ratio = targets.mean().item()
        self.alpha = max(0.2, min(0.8, 1 - pos_ratio))  # Give more weight to the minority class
        
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()

# Optimized training function
def train_model(model, dataloader, val_dataloader=None, num_epochs=30, lr=5e-5, device='cuda'):
    """
    Train a deep learning model with combined classification and regression tasks,
    including sample difficulty analysis and early stopping.
    
    Args:
        model: Neural network model to train
        dataloader: DataLoader containing training data
        val_dataloader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate for optimizer
        device: Device to run training on ('cuda' or 'cpu')
    
    Returns:
        The trained model
    """
    # Move model to specified device (GPU/CPU)
    model = model.to(device)
    
    # Initialize Adam optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,         # Restart period
        T_mult=1,       # Multiplication factor
        eta_min=1e-6    # Minimum learning rate
    )
    
    # Define loss functions
    criterion_cls = FocalLoss(alpha_init=0.25, gamma_init=2.0)
    
    # Tracking variables for early stopping
    best_loss = float('inf')
    best_auc = 0.0
    patience = 5
    no_improve = 0
    
    # For tracking training progress
    epoch_losses = []
    val_metrics = []
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        batch_count = 0
        all_losses = []  # For storing per-sample losses
        epoch_samples = {'correct': [], 'incorrect': []}
        
        for i, batch in enumerate(dataloader):
            # Transfer batch data to device (GPU/CPU)
            frames = batch['frames'].to(device)            # Video frames
            optical_flow = batch['optical_flow'].to(device)  # Optical flow data
            labels = batch['label'].to(device).unsqueeze(1)  # Classification labels
            alert_time = batch['alert_time'].to(device).unsqueeze(1)  # Regression targets
            video_ids = batch['video_id']
            
            # Zero gradients before forward pass
            optimizer.zero_grad()
            
            try:
                # Forward pass through the model
                pred_score, pred_alert = model(frames, optical_flow)
                
                # Calculate classification and regression losses
                loss_cls = criterion_cls(pred_score, labels)
                loss_reg = custom_time_loss(pred_alert, alert_time)
                
                # Store individual sample losses for mining
                sample_losses = F.binary_cross_entropy(pred_score, labels, reduction='none')
                for j, sl in enumerate(sample_losses):
                    all_losses.append((sl.item(), video_ids[j]))
                
                # Combined loss
                loss = 0.7 * loss_cls + 0.3 * loss_reg
                
                # Backward pass to compute gradients
                loss.backward()
                
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update model parameters
                optimizer.step()
                
                # Record loss value
                current_loss = loss.item()
                running_loss += current_loss
                batch_count += 1
                
                # Track correctly/incorrectly classified samples
                binary_preds = (pred_score > 0.5).float()
                correct_mask = (binary_preds == labels).cpu().squeeze()
                
                for j, (is_correct, vid) in enumerate(zip(correct_mask, video_ids)):
                    if is_correct:
                        epoch_samples['correct'].append(vid)
                    else:
                        epoch_samples['incorrect'].append(vid)
                
                # Free memory to prevent CUDA OOM errors
                del frames, optical_flow, labels, alert_time, pred_score, pred_alert, loss
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                # Print progress information (reduced frequency)
                if i % 20 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - Batch {i}/{len(dataloader)} - Loss: {current_loss:.4f}")
                
            except RuntimeError as e:
                # Handle out of memory errors by skipping the batch
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                else:
                    raise e
            except Exception as e:
                # Handle other exceptions during training
                print(f"Error during training at batch {i}: {str(e)}")
                continue
        
        # At the end of each epoch, identify difficult samples
        if epoch > 0 and epoch % 5 == 0:  # Every 5 epochs
            difficult_samples = sorted(all_losses, key=lambda x: x[0], reverse=True)[:100]
            print(f"Top 10 most difficult samples: {difficult_samples[:10]}")
            
            # Analyze the distribution of difficult samples
            print(f"Difficult samples count: {len(difficult_samples)}")
            difficult_ids = [ds[1] for ds in difficult_samples]
            print(f"Sample difficulty distribution analysis complete")
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / max(1, batch_count)
        epoch_losses.append(epoch_loss)
        
        # Validation phase
        if val_dataloader is not None:
            metrics = validate_model(model, val_dataloader, device)
            val_metrics.append(metrics)
            current_auc = metrics['auc']
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Val AUC: {current_auc:.4f}")
            print(f"Val Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
            
            # Early stopping based on AUC
            if current_auc > best_auc:
                best_auc = current_auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'auc': current_auc,
                }, "best_model_auc.pth")
                print(f"Model saved with AUC {current_auc:.4f}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
            
            # Early stopping based on loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                }, "best_model.pth")
                print(f"Model saved with loss {epoch_loss:.4f}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Update learning rate based on scheduler
        scheduler.step()
        
        # Print training statistics
        print(f"Epoch {epoch+1} - Correctly classified: {len(epoch_samples['correct'])}")
        print(f"Epoch {epoch+1} - Incorrectly classified: {len(epoch_samples['incorrect'])}")
        
        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, f"checkpoint_epoch_{epoch+1}.pth")
        
        # Generate progress plot every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_training_progress(epoch_losses, val_metrics, f"training_progress_epoch_{epoch+1}.png")
    
    # Generate final training progress plot
    plot_training_progress(epoch_losses, val_metrics, "final_training_progress.png")
    
    return model

def validate_model(model, val_dataloader, device='cuda'):
    """
    Validates the model and returns performance metrics.
    
    Args:
        model: Model to validate
        val_dataloader: Validation data loader
        device: Device to run validation on
        
    Returns:
        dict: Dictionary containing validation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            frames = batch['frames'].to(device)
            optical_flow = batch['optical_flow'].to(device)
            labels = batch['label']
            
            scores, _ = model(frames, optical_flow)
            
            all_preds.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            del frames, optical_flow, scores
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    y_pred = np.array([p[0] for p in all_preds])
    y_true = np.array(all_labels)
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    # Compute precision, recall and F1
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
    
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def analyze_predictions(model, val_dataloader, device='cuda'):
    """
    Analyzes model performance on the validation set.

    Parameters:
        model (nn.Module): The trained model.
        val_dataloader (DataLoader): DataLoader for the validation set.
        device (str): The device to run inference on ('cuda' or 'cpu').

    Returns:
        auc (float): ROC AUC score on the validation set.
        false_positives (list): List of (video_id, prediction) for top false positives.
        false_negatives (list): List of (video_id, prediction) for top false negatives.
    """
    model.eval()
    predictions = []
    ground_truths = []
    video_ids = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            frames = batch['frames'].to(device)
            optical_flow = batch['optical_flow'].to(device)
            labels = batch['label']
            batch_video_ids = batch['video_id']
            
            scores, _ = model(frames, optical_flow)
            
            predictions.extend(scores.cpu().numpy())
            ground_truths.extend(labels.numpy())
            video_ids.extend(batch_video_ids)
            
            del frames, optical_flow, scores
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    predictions = np.array([p[0] for p in predictions])
    ground_truths = np.array(ground_truths)
    
    # Compute ROC AUC
    auc = roc_auc_score(ground_truths, predictions)
    print(f"Validation AUC: {auc:.4f}")
    
    # Analyze performance at various thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for threshold in thresholds:
        binary_preds = (predictions >= threshold).astype(int)
        
        # Compute metrics
        tp = np.sum((binary_preds == 1) & (ground_truths == 1))
        fp = np.sum((binary_preds == 1) & (ground_truths == 0))
        tn = np.sum((binary_preds == 0) & (ground_truths == 0))
        fn = np.sum((binary_preds == 0) & (ground_truths == 1))
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
        
        print(f"Threshold {threshold:.1f} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Analyze errors
    binary_preds = (predictions >= 0.5).astype(int)
    
    # False positives (predicted collision, but no collision occurred)
    false_positives = [(vid, pred) for vid, pred, gt in 
                       zip(video_ids, predictions, ground_truths) 
                       if (pred >= 0.5 and gt == 0)]
    
    # False negatives (no collision predicted, but a collision occurred)
    false_negatives = [(vid, pred) for vid, pred, gt in 
                       zip(video_ids, predictions, ground_truths) 
                       if (pred < 0.5 and gt == 1)]
    
    print(f"\nTop 10 most confident false positives: {sorted(false_positives, key=lambda x: x[1], reverse=True)[:10]}")
    print(f"Top 10 least confident false negatives: {sorted(false_negatives, key=lambda x: x[1])[:10]}")
    
    return auc, false_positives, false_negatives

def analyze_video_features(video_dir, annotations, num_videos=20):
    """
    Analyzes video characteristics to better understand challenges in the dataset.

    Parameters:
        video_dir (str): Path to the directory containing video files.
        annotations (dict): Dictionary mapping video IDs to metadata including labels.
        num_videos (int): Number of videos to analyze (default is 20).

    Returns:
        results (dict): Dictionary containing lists of statistics for positive and negative samples.
    """
    results = {'positive': [], 'negative': []}
    
    for video_id, data in list(annotations.items())[:num_videos]:
        try:
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                continue
                
            # Extract frames for analysis
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Metrics to track
            brightness_values = []
            motion_values = []
            
            # Analyze key frames
            prev_frame = None
            for frame_idx in range(0, total_frames, max(1, total_frames // 10)):  # Sample 10 frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Compute brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Compute motion (if previous frame is available)
                if prev_frame is not None:
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Basic optical flow
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                        0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion = np.mean(mag)
                    motion_values.append(motion)
                
                prev_frame = frame.copy()
            
            cap.release()
            
            # Compute statistics
            avg_brightness = np.mean(brightness_values) if brightness_values else 0
            std_brightness = np.std(brightness_values) if brightness_values else 0
            avg_motion = np.mean(motion_values) if motion_values else 0
            std_motion = np.std(motion_values) if motion_values else 0
            
            result = {
                'video_id': video_id,
                'avg_brightness': avg_brightness,
                'std_brightness': std_brightness,
                'avg_motion': avg_motion,
                'std_motion': std_motion,
                'total_frames': total_frames
            }
            
            # Append to positive or negative results
            if data['label'] == 1:
                results['positive'].append(result)
            else:
                results['negative'].append(result)
                
        except Exception as e:
            print(f"Error analyzing video {video_id}: {str(e)}")
    
    # Show comparative statistics
    print("=== Positive vs. Negative Video Comparison ===")
    
    pos_brightness = np.mean([r['avg_brightness'] for r in results['positive']])
    neg_brightness = np.mean([r['avg_brightness'] for r in results['negative']])
    print(f"Average Brightness - Positive: {pos_brightness:.2f}, Negative: {neg_brightness:.2f}")
    
    pos_motion = np.mean([r['avg_motion'] for r in results['positive']])
    neg_motion = np.mean([r['avg_motion'] for r in results['negative']])
    print(f"Average Motion - Positive: {pos_motion:.2f}, Negative: {neg_motion:.2f}")
    
    return results

def plot_training_progress(loss_history, val_metrics, output_path='training_progress.png'):
    """
    Creates a visualization of training progress over epochs.

    Parameters:
        loss_history (list of float): List of training loss values for each epoch.
        val_metrics (list of dict): List of validation metrics per epoch. Each dict should contain
                                    'auc', 'precision', 'recall', and 'f1' keys.
        output_path (str): Path where the plot image will be saved.

    Saves:
        A PNG file showing training loss and validation metrics over time.
    """
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    plt.plot(loss_history, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot validation metrics
    if val_metrics:
        epochs = list(range(len(val_metrics)))
        auc_values = [metrics['auc'] for metrics in val_metrics]
        prec_values = [metrics['precision'] for metrics in val_metrics]
        rec_values = [metrics['recall'] for metrics in val_metrics]
        f1_values = [metrics['f1'] for metrics in val_metrics]
        
        plt.subplot(2, 1, 2)
        plt.plot(epochs, auc_values, 'g-', label='AUC')
        plt.plot(epochs, prec_values, 'r-', label='Precision')
        plt.plot(epochs, rec_values, 'c-', label='Recall')
        plt.plot(epochs, f1_values, 'm-', label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Progress plot saved to {output_path}")


def evaluate_model(model, dataloader, device='cuda'):
    model = model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            frames = batch['frames'].to(device)
            optical_flow = batch['optical_flow'].to(device)
            video_ids = batch.get('video_id', None)
            pred_score, _ = model(frames, optical_flow)
            predictions.extend(pred_score.cpu().numpy())
    return predictions

def predict_with_dynamic_threshold(model, frames, optical_flow, base_threshold=0.5, device='cuda'):
    """
    Makes predictions using a dynamic threshold adjusted to video conditions.
    
    Args:
        model: Trained model
        frames: Sequence of video frames [batch, frames, C, H, W]
        optical_flow: Computed optical flow [batch, frames-1, 2, H, W]
        base_threshold: Base threshold for classification
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        prediction: Binary prediction (0 or 1)
        score: Confidence score (0 to 1)
        alert_time: Predicted alert time
    """
    model.eval()
    frames = frames.to(device)
    optical_flow = optical_flow.to(device)
    
    # Calculate frame statistics for threshold adjustment
    brightness = frames.float().mean().item()
    contrast = frames.float().std().item()
    
    # Adjust threshold based on conditions
    threshold = base_threshold
    if brightness < 0.3:  # Dark video
        threshold += 0.05  # Be more conservative in low-light conditions
    elif contrast < 0.1:  # Low contrast video
        threshold += 0.03  # Be more conservative in low-contrast conditions
    
    # Make the prediction
    with torch.no_grad():
        score, alert_time = model(frames, optical_flow)
    
    # Apply the adjusted threshold
    prediction = (score > threshold).float()
    
    return prediction, score, alert_time

def predict_with_uncertainty(model, frames, optical_flow, n_samples=10, device='cuda'):
    """
    Makes predictions with uncertainty quantification using Monte Carlo Dropout.
    
    Args:
        model: Trained model with dropout
        frames: Sequence of video frames
        optical_flow: Computed optical flow
        n_samples: Number of samples for Monte Carlo Dropout
        device: Device for execution ('cuda' or 'cpu')
    
    Returns:
        mean_score: Mean prediction score
        mean_time: Mean predicted alert time
        std_score: Standard deviation of scores (measure of uncertainty)
        std_time: Standard deviation of times (measure of uncertainty)
    """
    model.train()  # Activate dropout during inference
    predictions_score = []
    predictions_time = []
    
    frames = frames.to(device)
    optical_flow = optical_flow.to(device)
    
    for _ in range(n_samples):
        with torch.no_grad():
            score, alert_time = model(frames, optical_flow)
            predictions_score.append(score)
            predictions_time.append(alert_time)
    
    # Convert lists to tensors
    predictions_score = torch.stack(predictions_score)
    predictions_time = torch.stack(predictions_time)
    
    # Calculate mean and standard deviation
    mean_score = predictions_score.mean(dim=0)
    std_score = predictions_score.std(dim=0)
    mean_time = predictions_time.mean(dim=0)
    std_time = predictions_time.std(dim=0)
    
    # A high std_score indicates high uncertainty in the prediction
    return mean_score, mean_time, std_score, std_time

def predict_batch_efficient(model, test_dataloader, device='cuda', batch_size=4):
    """
    Efficiently performs batch predictions, with memory cleanup.
    """
    model = model.to(device)
    model.eval()
    
    predictions = []
    video_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx % 10 == 0:  # Reduzido para imprimir a cada 10 lotes
                print(f"Processing batch {batch_idx+1}/{len(test_dataloader)}")
            
            # Transfer data to the device
            frames = batch['frames'].to(device)
            optical_flow = batch['optical_flow'].to(device)
            batch_video_ids = batch['video_id']
            
            # Make predictions
            scores, _ = model(frames, optical_flow)
            
            # Collect results
            predictions.extend(scores.cpu().numpy())
            video_ids.extend(batch_video_ids)
            
            # Explicitly free memory
            del frames, optical_flow, scores
            if device == 'cuda':
                torch.cuda.empty_cache()
            
    return predictions, video_ids

def generate_submission(predictions, video_ids, output_path='submission.csv'):
    """
    Generates the submission file in the required format.
    """
    with open(output_path, 'w') as f:
        f.write("id,score\n")
        for vid, score in zip(video_ids, predictions):
            f.write(f"{vid},{score[0]:.4f}\n")
    print("Submission generated:", output_path)


def train_model_with_cross_validation(base_dir, num_folds=2, num_epochs=50, device='cuda'):
    """
    Implements cross-validation for more robust training.
    
    Args:
        base_dir: Base directory with the data
        num_folds: Number of folds for cross-validation
        num_epochs: Number of epochs per fold
        device: Device for execution
    
    Returns:
        models: List of trained models for each fold
    """
    # Load annotations
    train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))
    
    # Prepare folds
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    trained_models = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n{'='*50}\nFold {fold+1}/{num_folds}\n{'='*50}")
        
        # Split data into train and validation
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        # Create annotations per fold
        train_annotations = {}
        for _, row in train_fold.iterrows():
            video_id = f"{int(row['id']):05d}"
            train_annotations[video_id] = {
                'label': row['target'],
                'alert_time': row['time_of_alert'] if not pd.isna(row['time_of_alert']) else 0,
                'event_time': row['time_of_event'] if not pd.isna(row['time_of_event']) else 0
            }
        
        val_annotations = {}
        for _, row in val_fold.iterrows():
            video_id = f"{int(row['id']):05d}"
            val_annotations[video_id] = {
                'label': row['target'],
                'alert_time': row['time_of_alert'] if not pd.isna(row['time_of_alert']) else 0,
                'event_time': row['time_of_event'] if not pd.isna(row['time_of_event']) else 0
            }
        
        # Create datasets and dataloaders
        train_video_dir = os.path.join(base_dir, "train")
        
        # Preprocess training data
        train_video_ids = list(train_annotations.keys())
        train_processed_data = parallel_preprocess_dataset(
            train_video_dir, 
            train_video_ids,
            num_frames=8,
            num_workers=4
        )
        
        # Preprocess validation data 
        val_video_ids = list(val_annotations.keys())
        val_processed_data = parallel_preprocess_dataset(
            train_video_dir,
            val_video_ids, 
            num_frames=8,
            num_workers=4
        )
        
        # Get transformations
        transforms = get_video_transforms()
        
        # Create datasets
        train_dataset = PreprocessedDashcamDataset(
            train_processed_data,
            train_annotations, 
            transform=transforms['train']
        )
        
        val_dataset = PreprocessedDashcamDataset(
            val_processed_data,
            val_annotations,
            transform=transforms['val'] 
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True, 
            num_workers=0
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0  
        )
        
        # Create and train model
        model = LightweightMultiStreamModel(output_dim=64)
        model.to(device)
        
        # Training
        train_model(
            model,
            train_dataloader,
            num_epochs=num_epochs,
            lr=5e-5,
            device=device
        )
        
        # Evaluate on validation set
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                frames = batch['frames'].to(device)
                optical_flow = batch['optical_flow'].to(device)
                labels = batch['label']
                
                scores, _ = model(frames, optical_flow)
                
                val_preds.extend(scores.cpu().numpy())
                val_labels.extend(labels.numpy())
                
                del frames, optical_flow, scores
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        # Calculate AUC  
        auc = roc_auc_score(val_labels, [p[0] for p in val_preds])
        print(f"Fold {fold+1} - Validation AUC: {auc:.4f}")
        
        # Save model for this fold 
        torch.save({
            'fold': fold,
            'model_state_dict': model.state_dict(),
            'auc': auc,
        }, f"model_fold_{fold+1}.pth")
        
        # Perform additional fine-tuning
        fine_tuned_model = fine_tune_model(
            model=model,  # Use the already trained model
            train_dataloader=train_dataloader,
            num_epochs=5,
            device=device  
        )
        
        # Save model after fine-tuning
        torch.save({
            'fold': fold,
            'model_state_dict': fine_tuned_model.state_dict(),
            'auc': auc,  # Use the previous AUC as reference
        }, f"model_fold_{fold+1}_finetuned.pth")
        
        trained_models.append(fine_tuned_model)
        fold_scores.append(auc)
    
    print(f"\n{'='*50}")
    print(f"Cross-Validation Complete") 
    print(f"Average AUC across folds: {np.mean(fold_scores):.4f}")
    print(f"Fold AUCs: {fold_scores}")
    
    return trained_models

def fine_tune_model(model, train_dataloader, num_epochs=5, device='cuda'):
    # Model should already be on the correct device
    
    # Optimizer with a very low learning rate
    optimizer = optim.Adam(model.parameters(), lr=3e-6, weight_decay=1e-4)
    
    # Cyclical learning rate for fine-tuning
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-6,
        max_lr=1e-5,
        step_size_up=len(train_dataloader) // 2,
        cycle_momentum=False
    )
    
    # Adaptive loss function
    criterion_cls = FocalLoss(alpha_init=0.25, gamma_init=2.5)
    
    # Fine-tuning
    model.train()
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_count = 0
        
        for i, batch in enumerate(train_dataloader):
            frames = batch['frames'].to(device)
            optical_flow = batch['optical_flow'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            alert_time = batch['alert_time'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                pred_score, pred_alert = model(frames, optical_flow)
                
                # Calculate loss
                loss_cls = criterion_cls(pred_score, labels) 
                loss_reg = custom_time_loss(pred_alert, alert_time)
                loss = 0.8 * loss_cls + 0.2 * loss_reg  # Greater weight on classification
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                scheduler.step()  # Update learning rate every batch
                
                # Record loss value
                current_loss = loss.item()
                running_loss += current_loss
                batch_count += 1
                
                # Free memory
                del frames, optical_flow, labels, alert_time, pred_score, pred_alert, loss
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                # Print progress 
                if i % 20 == 0:
                    print(f"Fine-tuning - Epoch {epoch+1}/{num_epochs} - Batch {i}/{len(train_dataloader)} - Loss: {current_loss:.4f}")
                
            except Exception as e:
                print(f"Error during fine-tuning on batch {i}: {str(e)}")
                continue
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / max(1, batch_count)
        print(f"Fine-tuning - Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}") 
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = model.state_dict().copy()
    
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def ensemble_predictions(test_dataloader, model_paths, weights=None, device='cuda'):
    """
    Weighted combination of multiple models.
    
    Args:
        test_dataloader: DataLoader for the test set
        model_paths: List of paths to the trained models
        weights: Weights for each model (if None, equal weights are used)
        device: Device for execution (cuda/cpu)
        
    Returns:
        predictions: Weighted average of predictions from all models
        video_ids: Video IDs
    """
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    
    all_predictions = []
    video_ids = None
    
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f"Model {path} not found, skipping...")
            continue
            
        print(f"Generating predictions with model from {path}")
        
        # Create new model instance
        model = LightweightMultiStreamModel(output_dim=64)
        model.to(device)
        
        # Load weights from checkpoint
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Generate predictions
        predictions = []
        batch_ids = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                frames = batch['frames'].to(device)
                optical_flow = batch['optical_flow'].to(device)
                batch_video_ids = batch['video_id']
                
                # Make prediction
                scores, _ = model(frames, optical_flow)
                
                predictions.extend(scores.cpu().numpy())
                batch_ids.extend(batch_video_ids)
                
                # Free memory
                del frames, optical_flow, scores
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        # Store predictions from this model, weighted by its importance
        all_predictions.append([p * weights[i] for p in predictions])
        
        # Keep IDs (assuming they are the same for all models)
        if video_ids is None:
            video_ids = batch_ids
    
    # Calculate weighted average of predictions
    if len(all_predictions) > 0:
        ensemble_preds = np.zeros_like(all_predictions[0])
        for preds in all_predictions:
            ensemble_preds += preds
    else:
        print("No models found for ensemble!")
        return None, None
    
    return ensemble_preds, video_ids

def diverse_ensemble_predictions(test_dataloader, device='cuda'):
    """
    Ensemble using diverse models with confidence-weighted voting.
    
    Parameters:
        test_dataloader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on, e.g., 'cuda' or 'cpu'.
    
    Returns:
        final_predictions (list): Final ensemble predictions.
        video_ids (list): Corresponding video IDs.
    """
    # Different architectures and configurations
    models_configs = [
        {"model_path": "base_model.pth", "weight": 1.0},
        {"model_path": "finetuned_model.pth", "weight": 1.5},  # Higher weight for the fine-tuned model
        {"model_path": "model_fold_1.pth", "weight": 0.8},
        {"model_path": "model_fold_2.pth", "weight": 0.8},
    ]
    
    all_predictions = []
    all_weights = []
    video_ids = None
    
    for config in models_configs:
        if not os.path.exists(config["model_path"]):
            print(f"Model {config['model_path']} not found, skipping...")
            continue
            
        print(f"Generating predictions with {config['model_path']}")
        
        # Create and load the model
        model = LightweightMultiStreamModel(output_dim=64)
        model.to(device)
        
        checkpoint = torch.load(config["model_path"], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Generate predictions with confidence estimation
        predictions = []
        confidences = []
        batch_ids = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                frames = batch['frames'].to(device)
                optical_flow = batch['optical_flow'].to(device)
                batch_video_ids = batch['video_id']
                
                # Use Monte Carlo Dropout to estimate uncertainty
                model.train()  # Enable dropout for Monte Carlo sampling
                n_samples = 5
                pred_samples = []
                
                for _ in range(n_samples):
                    scores, _ = model(frames, optical_flow)
                    pred_samples.append(scores)
                
                # Compute mean and standard deviation
                pred_samples = torch.stack(pred_samples)
                mean_preds = pred_samples.mean(dim=0)
                std_preds = pred_samples.std(dim=0)
                
                # Confidence is inversely proportional to uncertainty
                pred_confidence = 1.0 / (1.0 + std_preds)
                
                # Normalize confidence to [0, 1]
                pred_confidence = pred_confidence / pred_confidence.max()
                
                predictions.extend(mean_preds.cpu().numpy())
                confidences.extend(pred_confidence.cpu().numpy())
                batch_ids.extend(batch_video_ids)
                
                # Free memory
                del frames, optical_flow, pred_samples
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        # Store predictions and sample weights
        all_predictions.append(predictions)
        # Combine model weight with per-sample confidence
        sample_weights = [config["weight"] * conf for conf in confidences]
        all_weights.append(sample_weights)
        
        if video_ids is None:
            video_ids = batch_ids
    
    # Compute the weighted average of predictions
    final_predictions = np.zeros(len(video_ids))
    sum_weights = np.zeros(len(video_ids))
    
    for preds, weights in zip(all_predictions, all_weights):
        for j, (pred, weight) in enumerate(zip(preds, weights)):
            final_predictions[j] += pred[0] * weight[0]
            sum_weights[j] += weight[0]
    
    # Normalize by total weights
    final_predictions = final_predictions / np.maximum(sum_weights, 1e-10)
    
    return final_predictions, video_ids


if __name__ == "__main__":
    # Define the directory paths
    base_dir = "/kaggle/input/nexar-collision-prediction"
    train_video_dir = os.path.join(base_dir, "train")
    test_video_dir = os.path.join(base_dir, "test")
    
    print(f"Using training directory: {train_video_dir}")
    print(f"Using testing directory: {test_video_dir}")
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Define the number of workers for parallel processing
    num_workers = 4 if device == 'cuda' else 2
    
    # Load and prepare annotations
    train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(base_dir, "test.csv"))
    
    # Create annotation dictionaries
    train_annotations = {}
    for _, row in train_df.iterrows():
        video_id = f"{int(row['id']):05d}"
        train_annotations[video_id] = {
            'label': row['target'],
            'alert_time': row['time_of_alert'] if not pd.isna(row['time_of_alert']) else 0,
            'event_time': row['time_of_event'] if not pd.isna(row['time_of_event']) else 0
        }
    
    test_annotations = {}
    for _, row in test_df.iterrows():
        video_id = f"{int(row['id']):05d}"
        test_annotations[video_id] = {
            'label': 0,  # Placeholder
            'alert_time': 0  # Placeholder
        }
    
    # Get transformations for data augmentation
    transforms = get_video_transforms()
    
    # STEP 1: Parallel Preprocessing of Training Videos
    print("Starting preprocessing of training videos...")
    train_video_ids = list(train_annotations.keys())
    
    # Preprocess training videos in parallel
    train_processed_data = parallel_preprocess_dataset(
        train_video_dir, 
        train_video_ids,
        num_frames=8,
        num_workers=num_workers
    )

    # Create dataset with preprocessed data
    train_dataset = PreprocessedDashcamDataset(
        train_processed_data, 
        train_annotations,
        transform=transforms['train']
    )
    
    # Create dataloader with larger batch size
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=16 if device == 'cuda' else 4,  # Increased batch size from 8 to 16
        shuffle=True,
        num_workers=0
    )
    
    # STEP 2: Instantiate and Train the Model
    print("Initializing model...")
    model = LightweightMultiStreamModel(output_dim=64)
    
    # Display model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Initial training
    print("Starting initial training...")
    model.to(device)
    train_model(
        model, 
        train_dataloader, 
        num_epochs=50,  # Full training
        lr=1e-4, 
        device=device
    )
    
    # Save the base model after initial training
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 50,
    }, "base_model.pth")
    
    # STEP 3: Fine-tuning
    print("Starting fine-tuning...")
    fine_tuned_model = fine_tune_model(
        model=model,  # Use the already trained model
        train_dataloader=train_dataloader,
        num_epochs=5,  # Short fine-tuning phase
        device=device
    )
    
    # Save the fine-tuned model
    torch.save({
        'model_state_dict': fine_tuned_model.state_dict(),
        'epoch': 55,  # 50 initial + 5 fine-tuning
    }, "finetuned_model.pth")
    
    # STEP 4: Preprocessing of Test Videos
    print("Starting preprocessing of test videos...")
    test_video_ids = list(test_annotations.keys())
    
    # Preprocess test videos in parallel
    test_processed_data = parallel_preprocess_dataset(
        test_video_dir, 
        test_video_ids,
        num_frames=8,
        num_workers=num_workers
    )

    # Create test dataset
    test_dataset = PreprocessedDashcamDataset(
        test_processed_data, 
        test_annotations,
        transform=transforms['val']
    )
    
    # Create test dataloader
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=16 if device == 'cuda' else 4,
        shuffle=False,
        num_workers=0
    )
    
    # STEP 5: Generate predictions with the fine-tuned model
    print("Generating predictions for the test set...")
    model = fine_tuned_model  # Use the fine-tuned model for predictions
    
    predictions, video_ids = predict_batch_efficient(
        model, 
        test_dataloader, 
        device=device, 
        batch_size=16 if device == 'cuda' else 4
    )
    
    # Convert formatted IDs to integers for submission
    original_ids = [int(vid) for vid in video_ids]
    
    # Generate submission file
    submission_path = "submission.csv"
    submission_df = pd.DataFrame({
        'id': original_ids,
        'target': [float(p[0]) for p in predictions]
    })
    
    # Ensure the IDs are in the correct order
    submission_df = submission_df.sort_values('id')
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file generated: {submission_path}")


submission_df = pd.read_csv('/kaggle/working/submission.csv')
 
print(submission_df.shape)
print(submission_df.head())




