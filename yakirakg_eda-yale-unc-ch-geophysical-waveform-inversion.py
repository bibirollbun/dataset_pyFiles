import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import torch
import pandas as pd
from torchvision.transforms import Compose
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List
import logging
import csv
import torch.nn.functional as F
import csv

# Configure logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)


import statistics
import math


# Root Paths
BASE_DIR = '/kaggle/input/waveform-inversion'
TRAIN_DIR = os.path.join(BASE_DIR, 'train_samples')
TEST_DIR = os.path.join(BASE_DIR, 'test')

print("Train Folders:", os.listdir(TRAIN_DIR))


def load_dataset_config(config_path, dataset_name):
    """Loads normalization parameters from dataset_config.json."""
    try:
        with open(config_path) as f:
            ctx = json.load(f)[dataset_name]
        print(f"Loaded config for dataset: {dataset_name}")
        return ctx
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    except KeyError:
        print(f"Error: Dataset '{dataset_name}' not found in {config_path}.")
        sys.exit(1)


def get_transforms(ctx, k):
    """Gets the transformations for data and label based on test.py."""
    log_data_min = T.log_transform(ctx['data_min'], k=k)
    log_data_max = T.log_transform(ctx['data_max'], k=k)
    transform_data = Compose([
        T.LogTransform(k=k),
        T.MinMaxNormalize(log_data_min, log_data_max),
    ])

    return transform_data
    
# ================================================================
# Flexible Exploration for Any Family
# Auto-Skip empty folders
# ================================================================


def explore_family(folder_name):
    folder_path = os.path.join(TRAIN_DIR, folder_name)
    print(f"\nExploring {folder_name} Dataset")
    print("Available Files:", os.listdir(folder_path))

    seis_files = sorted([f for f in os.listdir(folder_path) if f.startswith('seis')])
    vel_files = sorted([f for f in os.listdir(folder_path) if f.startswith('vel')])

    print(f"Found {len(seis_files)} Seismic files")
    print(f"Found {len(vel_files)} Velocity files")

    # Check before loading
    if seis_files and vel_files:
        example_seis = load_npy(os.path.join(folder_path, seis_files[0]))
        example_vel = load_npy(os.path.join(folder_path, vel_files[0]))
        example_vel = np.squeeze(example_vel)

        print("Seismic Shape:", example_seis.shape)
        print("Velocity Shape:", example_vel.shape)
    else:
        print("Skipping... No seismic or velocity files found.")


def load_npy(file_path):
    return np.load(file_path)
    
# =============================================================================
# 1. Data Preparation
# =============================================================================
def collect_input_files(data_dir: str) -> list:
    """
    Recursively search for .npy files in data_dir that contain 'seis' or 'data' in their filename.
    """
    return [f for f in Path(data_dir).rglob("*.npy") if ("seis" in f.stem) or ("data" in f.stem)]

def map_input_to_output(input_files: list) -> list:
    """
    Map each input file to its corresponding output file by replacing keywords.
    """
    return [Path(str(f).replace("seis", "vel").replace("data", "model")) for f in input_files]

# Define training sample directory
TRAIN_DIR = "/kaggle/input/waveform-inversion/train_samples"
inputs_all = collect_input_files(TRAIN_DIR)
outputs_all = map_input_to_output(inputs_all)

# Check all output files exist
assert all(f.exists() for f in outputs_all)

# Split dataset into training and validation based on sampling frequency
train_inputs = [inputs_all[i] for i in range(0, len(inputs_all), 2)]
valid_inputs = [f for f in inputs_all if f not in train_inputs]
train_outputs = map_input_to_output(train_inputs)
valid_outputs = map_input_to_output(valid_inputs)


# =============================================================================
# 2. Dataset Definition
# =============================================================================
class SeismicDataset(Dataset):
    """
    Dataset handling seismic files with multiple examples per file.
    """
    def __init__(self, in_files: list, out_files: list, examples_per_file: int = 500):
        assert len(in_files) == len(out_files)
        self.in_files = in_files
        self.out_files = out_files
        self.examples_per_file = examples_per_file

    def __len__(self):
        return len(self.in_files) * self.examples_per_file

    def __getitem__(self, idx: int):
        file_index = idx // self.examples_per_file
        sample_index = idx % self.examples_per_file

        # Memory map the file to reduce memory usage
        x_data = np.load(self.in_files[file_index], mmap_mode="r")
        y_data = np.load(self.out_files[file_index], mmap_mode="r")
        try:
            return x_data[sample_index].copy(), y_data[sample_index].copy()
        finally:
            del x_data, y_data

# Create DataLoaders for training and validation
train_dataset = SeismicDataset(train_inputs, train_outputs, examples_per_file=500)
valid_dataset = SeismicDataset(valid_inputs, valid_outputs, examples_per_file=500)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, pin_memory=True,
    drop_last=True, num_workers=4, persistent_workers=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=64, shuffle=False, pin_memory=True,
    drop_last=False, num_workers=4, persistent_workers=True
)



# =============================================================================
# 3. Model Architecture: SmartConvNet
# =============================================================================
class SmartConvNet(nn.Module):
    """A convolutional network with adaptive pooling and dense layers."""
    def __init__(self, input_channels: int = 5, output_size: int = 70 * 70):
        super().__init__()
        # Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # spatial reduction
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # further reduction
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Pool output to a fixed size
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        feat = self.feature_extractor(x)
        pooled = self.pool(feat)
        flat = pooled.view(batch_size, -1)
        out = self.fc(flat)
        # Reshape output to (batch_size, 1, 70, 70) and apply scaling and bias
        return out.view(batch_size, 1, 70, 70) * 1000 + 1500


curve_fault_a_path = os.path.join(TRAIN_DIR, 'CurveFault_A')
print("Files in CurveFault_A:", os.listdir(curve_fault_a_path))
seis_file = os.path.join(curve_fault_a_path, 'seis2_1_0.npy')
vel_file = os.path.join(curve_fault_a_path, 'vel2_1_0.npy')

seis = load_npy(seis_file)
vel = load_npy(vel_file)

print("Seismic Data shape:", seis.shape)  
print("Velocity Data shape:", vel.shape)

## np.squeezeはサイズが1の次元をすべて削除
vel = np.squeeze(vel)  

print("Velocity Shape after squeeze:", vel.shape)

sample_id = 0


# seisは震源ごとにデータを分ける
receiver_box_data = []  # 5個分のデータを格納

for i in range(5):
    # 各Receiverのデータ: shape = (500, 1000, 70)
    receiver_i = seis[:, i, :, :]
    # flatten: shape = (500*1000*70,)
    receiver_i_flat = receiver_i.reshape(-1)
    receiver_box_data.append(receiver_i_flat)



# 震源別のデータ(振幅)
for i, data in enumerate(receiver_box_data):
    print(f"震源{i}")
    print("最大値：", max(data))
    print("最小値：", min(data))
    print("中央値：", statistics.median(data))
    print("最頻値：", statistics.mode(data))
    print("分散：", statistics.pvariance(data))
    print("\n")


# 箱ひげ図の描画
plt.figure(figsize=(10, 6))
plt.boxplot(receiver_box_data, showfliers=False)
plt.title("Seismic Data Distribution by Source")
plt.xlabel("Source Index (0 to 4)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


# 箱ひげ図の描画 外れ値描写あり
plt.figure(figsize=(10, 6))
plt.boxplot(receiver_box_data, showfliers=True)
plt.title("Seismic Data Distribution by Source")
plt.xlabel("Source Index (0 to 4)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


# データを(震源, 受信機)の組み合わせで分ける
seismic_by_source_receiver = {}

for s in range(5):          # 震源
    for r in range(70):     # レシーバー
        # shape: (500, 1000) を取得
        data_sr = seis[:, s, :, r]
        # flatten: (500 * 1000,)
        data_sr_flat = data_sr.reshape(-1)
        seismic_by_source_receiver[(s, r)] = data_sr_flat


# 箱ひげ図にするために、受信機を10個ずつまとめる
grouped_data = []
for i in range(0, 70, 10):
    group = seis[:, 0, :, i:i+10].reshape(-1)
    grouped_data.append(group)

plt.boxplot(grouped_data, showfliers=True)
plt.title("Receiver Groups (10 each) for Source 0")
plt.xlabel("Group Index")
plt.ylabel("Amplitude")
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# ヒストグラムを描画（重ねる or 並べる）
plt.figure(figsize=(12, 6))

for idx, group in enumerate(grouped_data):
    plt.hist(group, bins=100, alpha=0.5, label=f"Group {idx}")

plt.title("Amplitude Histogram, Grouped by Receiver 10s")
plt.xlabel("Amplitude")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# grouped_data を再作成（±1を除く）
grouped_data = []
for i in range(0, 70, 10):
    group = seis[:, 0, :, i:i+10].reshape(-1)
    # ±1の範囲を除外
    group_filtered = group[(group < -1) | (group > 1)]
    grouped_data.append(group_filtered)

# ヒストグラムを描画（重ねる or 並べる）
plt.figure(figsize=(12, 6))

for idx, group in enumerate(grouped_data):
    plt.hist(group, bins=100, alpha=0.5, label=f"Group {idx}")

plt.title("Amplitude Histogram (|x| > 1), Grouped by Receiver 10s")
plt.xlabel("Amplitude")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# seismic_data.shape = (500, 5, 1000, 70)

# 各レシーバーの平均波形（震源0）
waveforms = np.mean(seis[:, 0, :, :], axis=0)  # shape: (1000, 70)

plt.figure(figsize=(12, 6))
plt.imshow(waveforms.T, aspect='auto', cmap='seismic', extent=[0, 1000, 69, 0])
plt.colorbar(label="Amplitude")
plt.xlabel("Time Step")
plt.ylabel("Receiver Index")
plt.title("Wave Propagation (Source 0)")
plt.show()



arrival_times = []

for r in range(70):
    signal = np.mean(seis[:, 0, :, r], axis=0)  # shape: (1000,)
    arrival_time = np.argmax(np.abs(signal))  # 最も強い振幅の時間を到達時間と仮定
    arrival_times.append(arrival_time)

plt.plot(range(70), arrival_times, marker='o')
plt.xlabel("Receiver Index")
plt.ylabel("Estimated Arrival Time (Time Step)")
plt.title("Estimated Wave Arrival Times (Source 0)")
plt.grid(True)
plt.show()



mean_arrival_times = []
std_arrival_times = []

for r in range(70):
    arrival_times = []
    for s in range(500):
        signal = seis[s, 0, :, r]  # shape: (1000,)
        t = np.argmax(np.abs(signal))     # 各サンプルの到達時刻
        arrival_times.append(t)
    mean_arrival_times.append(np.mean(arrival_times))
    std_arrival_times.append(np.std(arrival_times))


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.errorbar(
    range(70),
    mean_arrival_times,
    yerr=std_arrival_times,
    fmt='o',
    capsize=3,
    label='Arrival Time (mean ± std)'
)
plt.xlabel("Receiver Index")
plt.ylabel("Estimated Arrival Time (Time Step)")
plt.title("Wave Arrival Time per Receiver (Source 0)")
plt.grid(True)
plt.legend()
plt.show()


# velは1次元にする
# フラット化後のshape (500*70*70,)
flat_vel = vel.reshape(-1)


print("最大値：", max(flat_vel))
print("最小値：", min(flat_vel))
print("中央値：", statistics.median(flat_vel))
print("最頻値：", statistics.mode(flat_vel))
print("分散：", statistics.pvariance(flat_vel))


# 箱ひげ図の描写 外れ値あり
plt.boxplot(flat_vel, showfliers=True)


plt.figure(figsize=(10, 6))
plt.title(f"Seismic Data - Batch 0, Source 0")
plt.imshow(seis[0, 0], aspect='auto', cmap='seismic')
plt.colorbar(label="Amplitude")
plt.xlabel("Receivers")
plt.ylabel("Timesteps")
plt.show()


# ヒストグラムを描画（重ねる or 並べる）
plt.figure(figsize=(12, 6))


plt.hist(flat_vel, bins=100, alpha=0.5)

plt.title("Velocity Histgram")
plt.xlabel("Velocity")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()


sample_id = 1
plt.figure(figsize=(8, 6))
plt.title(f"Velocity Map (Ground Truth) - Sample {sample_id}")
sns.heatmap(vel[sample_id], cmap='viridis')
plt.show()


sample_id = 2
plt.figure(figsize=(8, 6))
plt.title(f"Velocity Map (Ground Truth) - Sample {sample_id}")
sns.heatmap(vel[sample_id], cmap='viridis')
plt.show()




