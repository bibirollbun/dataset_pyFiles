import numpy as np
import matplotlib.pyplot as plt
import os

# List of all datasets to analyze
datasets = [
    'FlatVel_A',
    'FlatVel_B',
    'CurveFault_A',
    'CurveFault_B',
    'CurveVel_A',
    'FlatFault_A',
    'FlatFault_B',
    'Style_A'
]

# Function to load one example from a dataset
def load_example(dataset_name):
    base_path = f'/kaggle/input/waveform-inversion/train_samples/{dataset_name}'
    
    # Check if the dataset has 'data' and 'model' subdirectories (Vel and Style families)
    if os.path.exists(os.path.join(base_path, 'data')) and os.path.exists(os.path.join(base_path, 'model')):
        # Vel/Style family: FlatVel_A, FlatVel_B, CurveVel_A, Style_A
        data_dir = os.path.join(base_path, 'data')
        model_dir = os.path.join(base_path, 'model')
        
        # Get list of data and model files
        data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.npy')])
        
        if not data_files or not model_files:
            print(f"No .npy files found in {dataset_name} (data: {len(data_files)}, model: {len(model_files)}).")
            return None, None
        
        # Load the first file
        example_data_path = os.path.join(data_dir, data_files[0])
        example_model_path = os.path.join(model_dir, model_files[0])
    else:
        # Fault family: CurveFault_A, CurveFault_B, FlatFault_A, FlatFault_B
        all_files = sorted([f for f in os.listdir(base_path) if f.endswith('.npy')])
        
        # Separate files into seismic data (seis*.npy) and velocity maps (vel*.npy)
        data_files = [f for f in all_files if f.startswith('seis')]  # Changed from 'seis_' to 'seis'
        model_files = [f for f in all_files if f.startswith('vel')]  # Changed from 'vel_' to 'vel'
        
        if not data_files or not model_files:
            print(f"Could not separate files into seismic data and velocity maps in {dataset_name}.")
            print(f"Seismic files: {data_files}")
            print(f"Velocity files: {model_files}")
            return None, None
        
        # Load the first pair
        example_data_path = os.path.join(base_path, data_files[0])
        example_model_path = os.path.join(base_path, model_files[0])
    
    # Load the data
    seismic_data = np.load(example_data_path)
    velocity_map = np.load(example_model_path)
    
    return seismic_data, velocity_map

# Analyze each dataset
for dataset_name in datasets:
    print(f"\n=== Analyzing dataset: {dataset_name} ===")
    
    # Load example data
    seismic_data, velocity_map = load_example(dataset_name)
    
    if seismic_data is None or velocity_map is None:
        print(f"Skipping {dataset_name} due to loading error.")
        continue
    
    # Verify shapes to ensure correct separation
    expected_seismic_shape = (500, 5, 1000, 70)
    expected_velocity_shape = (500, 1, 70, 70)
    if seismic_data.shape != expected_seismic_shape or velocity_map.shape != expected_velocity_shape:
        print(f"Unexpected shapes in {dataset_name}:")
        print("Seismic data shape:", seismic_data.shape)
        print("Velocity map shape:", velocity_map.shape)
        print("Skipping due to shape mismatch.")
        continue
    
    # Print shapes and min/max values
    print("Seismic data shape:", seismic_data.shape)
    print("Velocity map shape:", velocity_map.shape)
    print("Seismic data min/max:", seismic_data.min(), seismic_data.max())
    print("Velocity map min/max:", velocity_map.min(), velocity_map.max())
    
    # Visualize seismic data (one source, one receiver over time)
    plt.figure(figsize=(12, 4))
    plt.plot(seismic_data[0, 0, :, 0], label="Source 0, Receiver 0")
    plt.title(f"Example Seismic Waveform ({dataset_name})")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    
    # Prepare velocity map for visualization by removing batch dimension
    velocity_map_2d = np.squeeze(velocity_map[0])  # Shape should be (70, 70)
    print("Velocity map 2D shape after squeeze:", velocity_map_2d.shape)
    
    # Visualize velocity map
    plt.figure(figsize=(8, 6))
    plt.imshow(velocity_map_2d, cmap='viridis', aspect='equal')
    plt.colorbar(label="Velocity (m/s)")
    plt.title(f"Example Velocity Map ({dataset_name})")
    plt.xlabel("Width (x)")
    plt.ylabel("Height (y)")
    plt.show()


# Define paths to training data (using FlatVel_A as an example)
data_dir = '/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data'
model_dir = '/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model'

# Get list of files
data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.npy')])

# Load one example seismic data and velocity map
example_data_path = os.path.join(data_dir, data_files[0])  # First seismic data file
example_model_path = os.path.join(model_dir, model_files[0])  # Corresponding velocity map
seismic_data = np.load(example_data_path)
velocity_map = np.load(example_model_path)

# Print shapes and basic info
print("Seismic data shape:", seismic_data.shape)  # Expected: (500, 5, 1000, 70)
print("Velocity map shape:", velocity_map.shape)  # Expected: (500, 1, 70, 70)
print("Seismic data min/max:", seismic_data.min(), seismic_data.max())
print("Velocity map min/max:", velocity_map.min(), velocity_map.max())

# Define global min/max for normalization (based on analysis of all datasets)
global_seismic_min = -27.16  # Minimum across all datasets
global_seismic_max = 55.31   # Maximum across all datasets
global_velocity_min = 1500.0  # Minimum across all datasets
global_velocity_max = 4500.0  # Maximum across all datasets

# Normalize seismic data to [-1, 1] using global min/max
seismic_data_normalized = 2 * (seismic_data - global_seismic_min) / (global_seismic_max - global_seismic_min) - 1

# Normalize velocity map to [0, 1] using global min/max
velocity_map_normalized = (velocity_map - global_velocity_min) / (global_velocity_max - global_velocity_min)

# Print new min/max to verify normalization
print("Normalized seismic data min/max:", seismic_data_normalized.min(), seismic_data_normalized.max())
print("Normalized velocity map min/max:", velocity_map_normalized.min(), velocity_map_normalized.max())

# Visualize seismic data (one source, one receiver over time) - using all 1000 steps
plt.figure(figsize=(12, 4))
plt.plot(seismic_data_normalized[0, 0, :, 0], label="Source 0, Receiver 0")
plt.title("Seismic Waveform (All 1000 Steps)")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.show()

# Prepare velocity map for visualization by removing batch dimension
velocity_map_2d = np.squeeze(velocity_map_normalized[0])  # Remove singleton dimensions
print("Velocity map 2D shape after squeeze:", velocity_map_2d.shape)  # Should be (70, 70)

# Visualize velocity map
plt.figure(figsize=(8, 6))
plt.imshow(velocity_map_2d, cmap='viridis', aspect='equal')
plt.colorbar(label="Normalized Velocity")
plt.title("Example Velocity Map")
plt.xlabel("Width (x)")
plt.ylabel("Height (y)")
plt.show()


import torch
import os
from torch.utils.data import Dataset, DataLoader, Subset

# Define global min/max values for normalization (same as in analysis step)
global_seismic_min = -27.16  # Minimum across all datasets
global_seismic_max = 55.31   # Maximum across all datasets
global_velocity_min = 1500.0  # Minimum across all datasets
global_velocity_max = 4500.0  # Maximum across all datasets

# Custom Dataset with augmentation
class SeismicDataset(Dataset):
    def __init__(self, data_dirs, model_dirs, time_steps_to_keep=1000, augment=True):
        self.data_files = []
        self.model_files = []
        
        # Process each pair of directories
        for data_dir, model_dir in zip(data_dirs, model_dirs):
            # Check if the dataset has 'data' and 'model' subdirectories (Vel/Style family)
            if os.path.exists(os.path.join(data_dir, 'data')) and os.path.exists(os.path.join(model_dir, 'model')):
                # Vel/Style family: FlatVel_A, FlatVel_B, CurveVel_A, Style_A
                data_files = sorted([f for f in os.listdir(os.path.join(data_dir, 'data')) if f.endswith('.npy')])
                model_files = sorted([f for f in os.listdir(os.path.join(model_dir, 'model')) if f.endswith('.npy')])
                self.data_files.extend([os.path.join(data_dir, 'data', f) for f in data_files])
                self.model_files.extend([os.path.join(model_dir, 'model', f) for f in model_files])
            else:
                # Fault family: CurveFault_A, CurveFault_B, FlatFault_A, FlatFault_B
                all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
                # Separate files into seismic data (seis*.npy) and velocity maps (vel*.npy)
                data_files = [f for f in all_files if f.startswith('seis')]
                model_files = [f for f in all_files if f.startswith('vel')]
                
                if len(data_files) != len(model_files):
                    print(f"Mismatch in number of data and model files in {data_dir}: {len(data_files)} data, {len(model_files)} model")
                    continue
                
                self.data_files.extend([os.path.join(data_dir, f) for f in data_files])
                self.model_files.extend([os.path.join(data_dir, f) for f in model_files])
        
        # Verify that the number of data and model files matches
        assert len(self.data_files) == len(self.model_files), f"Mismatch between data and model files: {len(self.data_files)} data files, {len(self.model_files)} model files"
        self.time_steps_to_keep = time_steps_to_keep
        self.augment = augment
    
    def __len__(self):
        return len(self.data_files) * 500
    
    def __getitem__(self, idx):
        file_idx = idx // 500
        sample_idx = idx % 500
        
        seismic_data = np.load(self.data_files[file_idx])[sample_idx]
        velocity_map = np.load(self.model_files[file_idx])[sample_idx]
        
        # Normalize seismic data to [-1, 1] using global min/max
        seismic_data = 2 * (seismic_data - global_seismic_min) / (global_seismic_max - global_seismic_min) - 1
        
        # Normalize velocity map to [0, 1] using global min/max
        velocity_map = (velocity_map - global_velocity_min) / (global_velocity_max - global_velocity_min)
        
        # Crop time steps for seismic data (using all 1000 steps)
        seismic_data = seismic_data[:, :self.time_steps_to_keep, :]
        
        # Apply augmentation if enabled
        if self.augment and np.random.rand() > 0.5:
            # Add random noise
            noise = np.random.normal(0, 0.01, seismic_data.shape)
            seismic_data = seismic_data + noise
            # Random amplitude scaling
            scale = np.random.uniform(0.9, 1.1)
            seismic_data = seismic_data * scale
            # Clip to ensure values stay in [-1, 1]
            seismic_data = np.clip(seismic_data, -1, 1)
        
        velocity_map = np.squeeze(velocity_map)
        
        seismic_data = torch.FloatTensor(seismic_data)
        velocity_map = torch.FloatTensor(velocity_map)
        
        return seismic_data, velocity_map

# Define directories for training data (include all sets)
train_data_dirs = [
    '/kaggle/input/waveform-inversion/train_samples/FlatVel_A',
    '/kaggle/input/waveform-inversion/train_samples/FlatVel_B',
    '/kaggle/input/waveform-inversion/train_samples/CurveFault_A',
    '/kaggle/input/waveform-inversion/train_samples/CurveFault_B',
    '/kaggle/input/waveform-inversion/train_samples/CurveVel_A',
    '/kaggle/input/waveform-inversion/train_samples/FlatFault_A',
    '/kaggle/input/waveform-inversion/train_samples/FlatFault_B',
    '/kaggle/input/waveform-inversion/train_samples/Style_A'
]
train_model_dirs = [
    '/kaggle/input/waveform-inversion/train_samples/FlatVel_A',
    '/kaggle/input/waveform-inversion/train_samples/FlatVel_B',
    '/kaggle/input/waveform-inversion/train_samples/CurveFault_A',
    '/kaggle/input/waveform-inversion/train_samples/CurveFault_B',
    '/kaggle/input/waveform-inversion/train_samples/CurveVel_A',
    '/kaggle/input/waveform-inversion/train_samples/FlatFault_A',
    '/kaggle/input/waveform-inversion/train_samples/FlatFault_B',
    '/kaggle/input/waveform-inversion/train_samples/Style_A'
]

# Create dataset with updated time_steps_to_keep
time_steps_to_keep = 1000  # Use all 1000 time steps
train_dataset = SeismicDataset(train_data_dirs, train_model_dirs, time_steps_to_keep=time_steps_to_keep, augment=True)

# Split dataset into train and validation
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)
train_split = int(0.8 * dataset_size)
train_indices = indices[:train_split]
val_indices = indices[train_split:]

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)

train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)


import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if norm == 'bn':
            layers.append(nn.BatchNorm2d(out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [
            nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if norm == 'bn':
            layers.append(nn.BatchNorm2d(out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, target_size=None, norm='bn'):
        super(DeconvBlock, self).__init__()
        self.target_size = target_size  # (height, width) to upsample to
        self.conv = nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(out_fea) if norm == 'bn' else None
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.target_size is not None:
            x = nn.functional.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.relu(x)
        return x

class InversionNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0):
        super(InversionNet, self).__init__()
        # Энкодер
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))  # (5, 1000, 70) -> (32, 500, 70)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # (32, 500, 70) -> (64, 250, 70)
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))  # (64, 250, 70) -> (64, 250, 70)
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # (64, 250, 70) -> (64, 125, 70)
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))  # (64, 125, 70) -> (64, 125, 70)
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # (64, 125, 70) -> (128, 62, 70)
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))  # (128, 62, 70) -> (128, 62, 70)
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)  # (128, 62, 70) -> (128, 31, 35)
        self.convblock5_2 = ConvBlock(dim3, dim3, dropout=0.3)  # (128, 31, 35) -> (128, 31, 35)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)  # (128, 31, 35) -> (256, 15, 17)
        self.convblock6_2 = ConvBlock(dim4, dim4, dropout=0.3)  # (256, 15, 17) -> (256, 15, 17)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)  # (256, 15, 17) -> (256, 7, 8)
        self.convblock7_2 = ConvBlock(dim4, dim4, dropout=0.3)  # (256, 7, 8) -> (256, 7, 8)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(3, ceil(70 * sample_spatial / 8)), padding=0)  # (256, 7, 8) -> (512, 1, 1)

        # Дополнительные слои для подгонки размеров skip-соединений
        self.skip1_adjust = ConvBlock(dim1, dim1, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # (32, 500, 70) -> (32, 250, 70)
        self.skip1_adjust2 = ConvBlock(dim1, dim1, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # (32, 250, 70) -> (32, 125, 70)
        self.skip1_adjust3 = ConvBlock(dim1, dim1, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # (32, 125, 70) -> (32, 62, 70)
        self.skip1_adjust4 = nn.Conv2d(dim1, dim1, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # (32, 62, 70) -> (32, 31, 70)
        self.skip1_adjust5 = DeconvBlock(dim1, dim1, target_size=(40, 40))  # (32, 31, 70) -> (32, 40, 40)

        self.skip4_adjust = ConvBlock(dim3, dim2, kernel_size=(3, 1), stride=(1, 2), padding=(1, 0))  # (128, 62, 70) -> (64, 62, 35)
        self.skip4_adjust2 = DeconvBlock(dim2, dim2, target_size=(20, 20))  # (64, 62, 35) -> (64, 20, 20)

        self.skip5_adjust = ConvBlock(dim3, dim3, kernel_size=(3, 1), stride=(1, 2), padding=(1, 0))  # (128, 31, 35) -> (128, 31, 17)
        self.skip5_adjust2 = DeconvBlock(dim3, dim3, target_size=(10, 10))  # (128, 31, 17) -> (128, 10, 10)

        self.skip6_adjust = ConvBlock(dim4, dim4, kernel_size=(3, 1), stride=(1, 2), padding=(1, 0))  # (256, 15, 17) -> (256, 15, 8)
        self.skip6_adjust2 = DeconvBlock(dim4, dim4, target_size=(5, 5))  # (256, 15, 8) -> (256, 5, 5)

        # Декодер
        self.deconv1_1 = DeconvBlock(dim5, dim5, target_size=(5, 5))  # (512, 1, 1) -> (512, 5, 5)
        self.deconv1_2 = ConvBlock(dim5, dim5)  # (512, 5, 5) -> (512, 5, 5)
        self.deconv2_1 = DeconvBlock(dim5 + dim4, dim4, target_size=(10, 10))  # (512+256, 5, 5) -> (256, 10, 10)
        self.deconv2_2 = ConvBlock(dim4, dim4)  # (256, 10, 10) -> (256, 10, 10)
        self.deconv3_1 = DeconvBlock(dim4 + dim3, dim3, target_size=(20, 20))  # (256+128, 10, 10) -> (128, 20, 20)
        self.deconv3_2 = ConvBlock(dim3, dim3)  # (128, 20, 20) -> (128, 20, 20)
        self.deconv4_1 = DeconvBlock(dim3 + dim2, dim2, target_size=(40, 40))  # (128+64, 20, 20) -> (64, 40, 40)
        self.deconv4_2 = ConvBlock(dim2, dim2)  # (64, 40, 40) -> (64, 40, 40)
        self.deconv5_1 = DeconvBlock(dim2 + dim1, dim1, target_size=(70, 70))  # (64+32, 40, 40) -> (32, 70, 70)
        self.deconv5_2 = ConvBlock(dim1, dim1)  # (32, 70, 70) -> (32, 70, 70)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)  # (32, 70, 70) -> (1, 70, 70)

    def forward(self, x):
        # Энкодер
        x1 = self.convblock1(x)  # (32, 500, 70)
        x2 = self.convblock2_1(x1)  # (64, 250, 70)
        x2 = self.convblock2_2(x2)  # (64, 250, 70)
        x3 = self.convblock3_1(x2)  # (64, 125, 70)
        x3 = self.convblock3_2(x3)  # (64, 125, 70)
        x4 = self.convblock4_1(x3)  # (128, 62, 70)
        x4 = self.convblock4_2(x4)  # (128, 62, 70)
        x5 = self.convblock5_1(x4)  # (128, 31, 35)
        x5 = self.convblock5_2(x5)  # (128, 31, 35)
        x6 = self.convblock6_1(x5)  # (256, 15, 17)
        x6 = self.convblock6_2(x6)  # (256, 15, 17)
        x7 = self.convblock7_1(x6)  # (256, 7, 8)
        x7 = self.convblock7_2(x7)  # (256, 7, 8)
        x = self.convblock8(x7)  # (512, 1, 1)

        # Подгонка размеров для skip-соединений
        x1_skip = self.skip1_adjust(x1)  # (32, 500, 70) -> (32, 250, 70)
        x1_skip = self.skip1_adjust2(x1_skip)  # (32, 250, 70) -> (32, 125, 70)
        x1_skip = self.skip1_adjust3(x1_skip)  # (32, 125, 70) -> (32, 62, 70)
        x1_skip = self.skip1_adjust4(x1_skip)  # (32, 62, 70) -> (32, 31, 70)
        x1_skip = self.skip1_adjust5(x1_skip)  # (32, 31, 70) -> (32, 40, 40)

        x4_skip = self.skip4_adjust(x4)  # (128, 62, 70) -> (64, 62, 35)
        x4_skip = self.skip4_adjust2(x4_skip)  # (64, 62, 35) -> (64, 20, 20)

        x5_skip = self.skip5_adjust(x5)  # (128, 31, 35) -> (128, 31, 17)
        x5_skip = self.skip5_adjust2(x5_skip)  # (128, 31, 17) -> (128, 10, 10)

        x6_skip = self.skip6_adjust(x6)  # (256, 15, 17) -> (256, 15, 8)
        x6_skip = self.skip6_adjust2(x6_skip)  # (256, 15, 8) -> (256, 5, 5)

        # Декодер с skip-соединениями
        x = self.deconv1_1(x)  # (512, 1, 1) -> (512, 5, 5)
        x = self.deconv1_2(x)  # (512, 5, 5)
        x = torch.cat([x, x6_skip], dim=1)  # (512+256, 5, 5)
        x = self.deconv2_1(x)  # (256, 10, 10)
        x = self.deconv2_2(x)  # (256, 10, 10)
        x = torch.cat([x, x5_skip], dim=1)  # (256+128, 10, 10)
        x = self.deconv3_1(x)  # (128, 20, 20)
        x = self.deconv3_2(x)  # (128, 20, 20)
        x = torch.cat([x, x4_skip], dim=1)  # (128+64, 20, 20)
        x = self.deconv4_1(x)  # (64, 40, 40)
        x = self.deconv4_2(x)  # (64, 40, 40)
        x = torch.cat([x, x1_skip], dim=1)  # (64+32, 40, 40)
        x = self.deconv5_1(x)  # (32, 70, 70)
        x = self.deconv5_2(x)  # (32, 70, 70)
        x = self.deconv6(x)  # (1, 70, 70)
        return x


import torch.optim as optim
from torch.utils.data import Subset
import matplotlib.pyplot as plt

# Split train_dataset into train and validation sets (80% train, 20% validation)
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)
train_split = int(0.8 * dataset_size)
train_indices = indices[:train_split]
val_indices = indices[train_split:]

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)

train_dataloader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=2)

# Gradient loss function
def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_dx = torch.mean(torch.abs(pred_dx - target_dx))
    loss_dy = torch.mean(torch.abs(pred_dy - target_dy))
    return (loss_dx + loss_dy) / 2

model = InversionNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)  # Уменьшенный lr
num_epochs = 25
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)  # T_max = num_epochs

train_losses = []
val_losses = []
train_grad_losses = []
val_grad_losses = []
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_grad_loss = 0.0
    for seismic_batch, velocity_batch in train_dataloader:
        seismic_batch = seismic_batch.to(device)
        velocity_batch = velocity_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(seismic_batch)
        outputs = (outputs + 1) / 2
        target = velocity_batch.unsqueeze(1)
        
        l1_loss = criterion(outputs, target)
        grad_loss = gradient_loss(outputs, target)
        loss = l1_loss + 0.1 * grad_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Градиентный клиппинг
        optimizer.step()
        
        running_loss += l1_loss.item() * seismic_batch.size(0)
        running_grad_loss += grad_loss.item() * seismic_batch.size(0)
    
    epoch_train_loss = running_loss / len(train_dataloader.dataset)
    epoch_grad_loss = running_grad_loss / len(train_dataloader.dataset)
    train_losses.append(epoch_train_loss)
    train_grad_losses.append(epoch_grad_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train L1 Loss: {epoch_train_loss:.4f}, Train Grad Loss: {epoch_grad_loss:.4f}")
    
    model.eval()
    val_loss = 0.0
    val_grad_loss = 0.0
    with torch.no_grad():
        for seismic_batch, velocity_batch in val_dataloader:
            seismic_batch = seismic_batch.to(device)
            velocity_batch = velocity_batch.to(device)
            outputs = model(seismic_batch)
            outputs = (outputs + 1) / 2
            target = velocity_batch.unsqueeze(1)
            
            l1_loss = criterion(outputs, target)
            grad_loss = gradient_loss(outputs, target)
            loss = l1_loss + 0.1 * grad_loss
            
            val_loss += l1_loss.item() * seismic_batch.size(0)
            val_grad_loss += grad_loss.item() * seismic_batch.size(0)
    
    epoch_val_loss = val_loss / len(val_dataloader.dataset)
    epoch_val_grad_loss = val_grad_loss / len(val_dataloader.dataset)
    val_losses.append(epoch_val_loss)
    val_grad_losses.append(epoch_val_grad_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Val L1 Loss: {epoch_val_loss:.4f}, Val Grad Loss: {epoch_val_grad_loss:.4f}")
    
    scheduler.step()
    
    # Save the best model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_loss': epoch_val_loss,
        }, "/kaggle/working/model_best.pth")
        print(f"Best model saved at epoch {epoch+1} with Val Loss: {epoch_val_loss:.4f}")

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train L1 Loss")
plt.plot(val_losses, label="Val L1 Loss")
plt.plot(train_grad_losses, label="Train Grad Loss")
plt.plot(val_grad_losses, label="Val Grad Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.show()

# Visualize a sample prediction
model.eval()
with torch.no_grad():
    sample_seismic, sample_velocity = next(iter(val_dataloader))
    sample_seismic = sample_seismic.to(device)
    sample_velocity = sample_velocity.to(device)
    output = model(sample_seismic)
    output = (output + 1) / 2
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(output[0, 0].cpu().numpy(), cmap='viridis', aspect='equal')
    plt.colorbar(label="Predicted Velocity (normalized)")
    plt.title("Predicted Velocity Map")
    plt.xlabel("Width (x)")
    plt.ylabel("Height (y)")
    
    plt.subplot(1, 2, 2)
    plt.imshow(sample_velocity[0].cpu().numpy(), cmap='viridis', aspect='equal')
    plt.colorbar(label="Ground Truth Velocity (normalized)")
    plt.title("Ground Truth Velocity Map")
    plt.xlabel("Width (x)")
    plt.ylabel("Height (y)")
    
    plt.tight_layout()
    plt.show()


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
from glob import glob

# Custom Dataset for test data

class TestDataset(Dataset):
    def __init__(self, test_dir):
        self.test_files = sorted(glob(os.path.join(test_dir, "*.npy")))
    
    def __len__(self):
        return len(self.test_files)
    
    def __getitem__(self, idx):
        file_path = self.test_files[idx]
        seismic_data = np.load(file_path)  # Shape: (5, 1000, 70)
        # Crop to (5, 1000, 70) by taking the first 1000 time steps
        seismic_data = seismic_data[:, :1000, :]
        seismic_data = torch.tensor(seismic_data, dtype=torch.float32)
        # Normalize the seismic data (same as training)
        seismic_data = 2 * (seismic_data - seismic_data.min()) / (seismic_data.max() - seismic_data.min()) - 1
        filename = os.path.basename(file_path).replace(".npy", "")
        return seismic_data, filename

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InversionNet()
model = model.to(device)

# Load the best weights
checkpoint_path = "/kaggle/working/model_best.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Switch to evaluation mode
print(f"Model loaded from best checkpoint with Val Loss = {checkpoint['val_loss']:.4f}")

# Create test dataset and DataLoader
test_dir = "/kaggle/input/waveform-inversion/test"
test_dataset = TestDataset(test_dir)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
print(f"Number of test samples: {len(test_dataset)}")

# Make predictions
all_predictions = []
all_filenames = []

with torch.no_grad():
    for seismic_batch, filenames in test_dataloader:
        seismic_batch = seismic_batch.to(device)
        # Forward pass
        outputs = model(seismic_batch)  # Shape: (batch_size, 1, 70, 70)
        outputs = (outputs + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        all_predictions.append(outputs.cpu().numpy())
        all_filenames.extend(filenames)

# Concatenate all predictions
all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: (num_samples, 1, 70, 70)
all_predictions = all_predictions.squeeze(1)  # Shape: (num_samples, 70, 70)
print(f"Predictions shape: {all_predictions.shape}")

# Extract odd-numbered columns
odd_columns = np.arange(1, 70, 2)  # [1, 3, 5, ..., 69]
predictions_odd = all_predictions[:, :, odd_columns]  # Shape: (num_samples, 70, 35)
print(f"Predictions (odd columns) shape: {predictions_odd.shape}")

# Denormalize predictions to m/s
predictions_denorm = predictions_odd * 2999.0 + 1501.0  # Shape: (num_samples, 70, 35)
print(f"Denormalized predictions shape: {predictions_denorm.shape}")

# Prepare submission file
submission_rows = []
num_samples = len(test_dataset)

for sample_idx in range(num_samples):
    filename = all_filenames[sample_idx]
    for row in range(70):
        # Form the id in the format filename_y_row
        sample_id = f"{filename}_y_{row}"
        # Get values for odd-numbered columns
        values = predictions_denorm[sample_idx, row, :]  # Shape: (35,)
        row_data = [sample_id] + values.tolist()
        submission_rows.append(row_data)

# Create DataFrame
columns = ['oid_ypos'] + [f"x_{col}" for col in odd_columns]
submission_df = pd.DataFrame(submission_rows, columns=columns)

# Verify submission format
print("Submission DataFrame shape:", submission_df.shape)
print("Submission DataFrame columns:", submission_df.columns.tolist())
print(submission_df.head())

# Compare with sample submission
sample_submission = pd.read_csv("/kaggle/input/waveform-inversion/sample_submission.csv")
print("\nSample submission shape:", sample_submission.shape)
print("Sample submission columns:", sample_submission.columns.tolist())
print(sample_submission.head())

# Save to CSV with the correct path
submission_df.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission file saved as /kaggle/working/submission.csv")

