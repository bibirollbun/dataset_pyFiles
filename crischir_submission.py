# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%%writefile data_helper.py
# Copying from https://www.kaggle.com/code/tpmeli/exploratory-deep-dive-geo-wfi-data-insights
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from torch.utils.data import Dataset

###############################################################################
# 1) LIGHT SEISMIC HELPER
###############################################################################
class SeismicDataHelperLight:
    """
    Minimal version of the Seismic Data Helper:
      - Scans a root folder for subdirectories (datasets).
      - Each dataset has matching .npy files: dataXXXX.npy <-> modelXXXX.npy.
      - Access them through .datasets (list of dataset names) or pairs(dataset).
    """

    def __init__(self, root_dir: str):
        """
        Args:
            root_dir (str): Directory path containing subfolders with
                            seismic/velocity .npy files.
        """
        self.root_dir = root_dir
        self._pairs: Dict[str, List[Tuple[str, str]]] = self._scan_pairs()

    def _scan_pairs(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Look for subdirectories under root_dir. 
        For each subdirectory, try:
          - If /data and /model exist, match dataXXXX.npy with modelXXXX.npy
          - Otherwise, match seisXXXX.npy with velXXXX.npy.
        Return a dict:  {"folder_name": [ (seis_path, vel_path), ... ], ... }
        """
        pairs: Dict[str, List[Tuple[str, str]]] = {}
        for folder_name in os.listdir(self.root_dir):
            ds_dir = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(ds_dir):
                continue

            data_dir = os.path.join(ds_dir, 'data')
            model_dir = os.path.join(ds_dir, 'model')

            # If /data and /model exist:
            if os.path.isdir(data_dir) and os.path.isdir(model_dir):
                matched_list = []
                for fname in os.listdir(data_dir):
                    if fname.startswith('data') and fname.endswith('.npy'):
                        suf = fname[len('data'):-4]  # part after 'data' before '.npy'
                        seis_file = os.path.join(data_dir, fname)
                        vel_file  = os.path.join(model_dir, f'model{suf}.npy')
                        if os.path.exists(vel_file):
                            matched_list.append((seis_file, vel_file))
                if matched_list:
                    pairs[folder_name] = sorted(matched_list)
                continue

            # Otherwise, try to match seisXXXX.npy with velXXXX.npy
            all_files = os.listdir(ds_dir)
            seis_fs = [f for f in all_files if f.startswith('seis')  or f.startswith('data') and f.endswith('.npy')]
            vel_fs  = [f for f in all_files if f.startswith('vel') or f.startswith('model') and f.endswith('.npy')]
            if seis_fs and vel_fs:
                vel_set = set(vel_fs)
                matched_list = []
                for sf in seis_fs:
                    suf = sf[len('seis'):-4]
                    vf  = f'vel{suf}.npy'
                    if vf in vel_set:
                        matched_list.append((os.path.join(ds_dir, sf),
                                             os.path.join(ds_dir, vf)))
                if matched_list:
                    pairs[folder_name] = sorted(matched_list)

        return pairs

    @property
    def datasets(self) -> List[str]:
        """List all available subfolders that contain matched (seis, vel) pairs."""
        return list(self._pairs.keys())

    def is_there_folder(self, folder_name: str) -> bool:
        return folder_name in self._pairs

    def pairs(self, folder_name: str) -> List[Tuple[str, str]]:
        """All (seis_path, vel_path) pairs belonging to a named dataset folder."""
        return self._pairs[folder_name]

class SeismicDataset(Dataset):
    def __init__(self, inputs_files, output_files, n_examples_per_file=500):
        assert len(inputs_files) == len(output_files)
        self.inputs_files = inputs_files
        self.output_files = output_files
        self.n_examples_per_file = n_examples_per_file

    def __len__(self):
        return len(self.inputs_files) * self.n_examples_per_file

    def __getitem__(self, idx):
        # Calculate file offset and sample offset within file
        file_idx = idx // self.n_examples_per_file
        sample_idx = idx % self.n_examples_per_file

        X = np.load(self.inputs_files[file_idx], mmap_mode='r')
        y = np.load(self.output_files[file_idx], mmap_mode='r')

        try:
            return X[sample_idx].copy(), y[sample_idx].copy()
        finally:
            del X, y


class TestDataset(Dataset):
    def __init__(self, test_files):
        self.test_files = test_files


    def __len__(self):
        return len(self.test_files)


    def __getitem__(self, i):
        test_file = self.test_files[i]

        return np.load(test_file), test_file.stem


# end of data_helper.py



%%writefile unet_def.py
# Just copying from https://www.kaggle.com/code/egortrushin/gwi-unet-with-float16-dataset/notebook
# Model
import torch.nn as nn
import torch.nn.functional as F
import torch


class ResidualDoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2 + Residual Connection"""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_bn=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # First convolution layer
        # 3*3畳み込み. padding=1にすることで、出力のサイズは入力と変わらない。チャネル数はin_channelsからmid_channelsに変わる。
        # テンキーで説明してみる。5,6,2,3の部分が左上の端だとする。でもpadding=1だから1,4,7,8,9の部分が埋められている。ので、5の部分も3*3の畳み込みができる.
        # (batch_size, in_channels, height, width) -> (batch_size, mid_channels, height, width)
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=not use_bn
        )
        if use_bn:
            self.bn1 = nn.BatchNorm2d(mid_channels)
        else:
            self.bn1 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        # Second convolution layer
        # (batch_size, mid_channels, height, width) -> (batch_size, out_channels, height, width)
        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn
        )
        if use_bn:
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn2 = nn.Identity()

        # Shortcut connection to handle potential channel mismatch
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # Projection shortcut: 1x1 conv + BN to match output channels
            # 1*1畳み込み. padding=0にすることで、出力のサイズは入力と変わらない。チャネル数はin_channelsからout_channelsに変わる。
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))

            self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        identity = x  # Store the input for the residual connection

        """
        x-------------------------\
        |                         |
        v                         |
        conv1                     |
        |                         |
        v                         |
        batch normalization 1     |
        |                         |
        v                         |
        ReLU                      |
        |                         |
        v                         |
        Conv2                     |
        |                         |
        v                         |
        batch normalization 2     |
        |                         |
        V                         |
        Add <---------------------/
        |
        v
        ReLu
        |
        v
        out
        """

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block (without final ReLU yet)
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply shortcut to the identity path
        identity_mapped = self.shortcut(identity)

        # Add the residual connection
        out += identity_mapped

        # Apply final ReLU
        out = self.relu(out)
        return out


class Up(nn.Module):
    """Upscaling then ResidualDoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            # Input to ResidualDoubleConv = channels from upsampled layer below + channels from skip connection
            # Output of ResidualDoubleConv = desired output channels for this decoder stage
            self.conv = ResidualDoubleConv(
                in_channels + out_channels, out_channels
            )  # Use ResidualDoubleConv

        else:  # Using ConvTranspose2d
            # ConvTranspose halves the channels: in_channels -> in_channels // 2
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            # Input channels to ResidualDoubleConv
            conv_in_channels = in_channels // 2  # Channels after ConvTranspose
            skip_channels = out_channels  # Channels from skip connection
            total_in_channels = conv_in_channels + skip_channels
            self.conv = ResidualDoubleConv(
                total_in_channels, out_channels
            )  # Use ResidualDoubleConv

    def forward(self, x1, x2):
        # x1 is the feature map from the layer below (needs upsampling)
        # x2 is the skip connection from the corresponding encoder layer
        x1 = self.up(x1)

        # Pad x1 if its dimensions don't match x2 after upsampling
        # Input is CHW
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        # Pad format: (padding_left, padding_right, padding_top, padding_bottom)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 Convolution for the output layer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture implementation with Residual Blocks"""

    def __init__(
        self,
        n_channels=5,
        n_classes=1,
        init_features=32,
        depth=5,  # number of pooling layers
        bilinear=True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth

        # initial poolingをやめてみる→あんま効果なし.
        self.initial_pool = nn.AvgPool2d(kernel_size=(14, 1), stride=(14, 1))

        # --- Encoder ---
        # convは最初のconv以外、チャネルを倍々にしていく
        self.encoder_convs = nn.ModuleList()  # Store conv blocks (ResidualDoubleConv)
        self.encoder_pools = nn.ModuleList()  # Store pool layers (MaxPool2d)

        # Initial conv block (no pooling before it)
        # Use ResidualDoubleConv for the initial convolution block
        self.inc = ResidualDoubleConv(n_channels, init_features)
        self.encoder_convs.append(self.inc)

        current_features = init_features
        for _ in range(depth):
            # Define convolution block for this stage
            conv = ResidualDoubleConv(current_features, current_features * 2, use_bn=True)
            # Define pooling layer for this stage
            # MaxPool2d をカーネルサイズ2, ストライド2で行う。
            pool = nn.MaxPool2d(2)
            self.encoder_convs.append(conv)
            self.encoder_pools.append(pool)
            current_features *= 2

        # --- Bottleneck ---
        # Use ResidualDoubleConv for the bottleneck
        self.bottleneck = ResidualDoubleConv(current_features, current_features, use_bn=True)

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        # Input features start from bottleneck output features
        # Output features at each stage are halved
        for _ in range(depth):
            # Up block uses ResidualDoubleConv internally and handles channels
            up_block = Up(current_features, current_features // 2, bilinear)
            self.decoder_blocks.append(up_block)
            current_features //= 2  # Halve features for next Up block input

        # --- Output Layer ---
        # Input features are the output features of the last Up block
        self.outc = OutConv(current_features, n_classes)

    def _pad_or_crop(self, x, target_h=70, target_w=70):
        """Pads or crops input tensor x to target height and width."""
        _, _, h, w = x.shape
        # Pad Height if needed
        if h < target_h:
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            x = F.pad(x, (0, 0, pad_top, pad_bottom))  # Pad height only
            h = target_h
        # Pad Width if needed
        if w < target_w:
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            x = F.pad(x, (pad_left, pad_right, 0, 0))  # Pad width only
            w = target_w
        # Crop Height if needed
        if h > target_h:
            crop_top = (h - target_h) // 2
            # Use slicing to crop
            x = x[:, :, crop_top : crop_top + target_h, :]
            h = target_h
        # Crop Width if needed
        if w > target_w:
            crop_left = (w - target_w) // 2
            x = x[:, :, :, crop_left : crop_left + target_w]
            w = target_w
        return x

    def forward(self, x):
        # Initial pooling and resizing
        x_pooled = self.initial_pool(x) # (14,1)のkernelを(14,1)のstrideで avg-poolingする。 サイズが(1/14, 1)倍になる
        # 画像のサイズが(1000, 70)だったなら、(1000/14, 70) = (71, 70)
        x_resized = self._pad_or_crop(x_pooled, target_h=70, target_w=70) # cropするので、(70,70)になる

        # --- Encoder Path ---
        skip_connections = []
        xi = x_resized

        # Apply initial conv (inc)
        xi = self.encoder_convs[0](xi)
        skip_connections.append(xi)  # Store output of inc

        # Apply subsequent encoder convs and pools
        # self.depth is the number of pooling layers
        for i in range(self.depth):
            # Apply conv block for this stage

            # (batch_size, current_features, height, width) -> (batch_size, current_features*2, height, width)になる
            xi = self.encoder_convs[i + 1](xi)
            # Store skip connection *before* pooling
            skip_connections.append(xi)
            # Apply pooling layer for this stage
            # MaxPooling2dを行う
            # (batch_size, current_features*2, height, width) -> (batch_size, current_features*2, height/2, width/2)になる
            xi = self.encoder_pools[i](xi)

        # Apply bottleneck conv
        xi = self.bottleneck(xi)

        # --- Decoder Path ---
        xu = xi  # Start with bottleneck output
        # Iterate through decoder blocks and corresponding skip connections in reverse
        for i, block in enumerate(self.decoder_blocks):
            # Determine the correct skip connection index from the end
            # Example: depth=5. Skips stored: [inc, enc1, enc2, enc3, enc4] (indices 0-4)
            # Decoder 0 (Up(1024, 512)) needs skip 4 (enc4)
            # Decoder 1 (Up(512, 256)) needs skip 3 (enc3) ...
            # Decoder 4 (Up(64, 32)) needs skip 0 (inc)
            skip_index = self.depth - 1 - i
            skip = skip_connections[skip_index]
            xu = block(xu, skip)  # Up block combines xu (from below) and skip

        # --- Final Output ---
        logits = self.outc(xu)
        # Apply scaling and offset specific to the problem's target range
        # だいたい中央値は3000くらいなので、3000にする。
        # 標準偏差が800くらいなんだけど速度の分布は正規分布よりももっと裾が広いきがするから、スケールは1000くらい
        output = logits * 1000.0 + 3000.0
        return output


# end of unet_def.py


# Add the directory where data_helper.py lives to the system path
import sys
sys.path.append('/kaggle/working')  # or wherever the .py files were saved

from data_helper import TestDataset
from unet_def import UNet



# predict_for_submission.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

from data_helper import TestDataset
from unet_def import UNet

import torch.serialization
torch.serialization.add_safe_globals({'_reconstruct': np.core.multiarray._reconstruct})



def load_model(model_path, init_features=32, depth=5, device='cpu'):
    model = UNet(n_channels=5, n_classes=1, init_features=init_features, depth=depth)
    # checkpoint = torch.load(model_path, map_location=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    new_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(new_state)
    model.to(device)
    model.eval()
    return model
def make_submission(model, test_dir, output_csv, batch_size=32, device='cpu'):
    from torch.utils.data import DataLoader
    from data_helper import TestDataset
    from pathlib import Path
    import torch
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    test_files = list(Path(test_dir).glob("*.npy"))
    dataset = TestDataset(test_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_rows = []
    for inputs, oids in tqdm(dataloader, desc="Predicting"):
        # Ensure all inputs are float32 tensors
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

        with torch.inference_mode():
            outputs = model(inputs).cpu().numpy()  # Shape: (batch_size, 1, 70, 70)

        for pred, oid in zip(outputs, oids):
            pred = pred[0]  # Remove channel dimension → shape (70, 70)
            for y in range(pred.shape[0]):
                row = [f"{oid}_y_{y}"] + pred[y][1::2].tolist()  # odd x_ values
                all_rows.append(row)

    # Build DataFrame and save
    columns = ["oid_ypos"] + [f"x_{i}" for i in range(1, 70, 2)]
    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"✅ Submission saved to {output_csv}")
        # Ensure all inputs are float32 tensors



# def make_submission(model, test_dir, output_csv, batch_size=32, device='cpu'):
#     test_files = list(Path(test_dir).glob("*.npy"))
#     dataset = TestDataset(test_files)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     all_rows = []
#     with torch.inference_mode(), torch.autocast(device_type=device):
#         # for inputs, oids in tqdm(dataloader, desc="Predicting"):
#         #     inputs = inputs.to(device)
#         #     outputs = model(inputs).float().cpu().numpy()  # (batch_size, 1, 70, 70)
#             for inputs, oids in tqdm(dataloader, desc="Predicting"):
#                 inputs = torch.tensor(inputs).to(dtype=torch.float32, device=device)
#                 with torch.inference_mode(), torch.autocast(device_type=device):
#                     outputs = model(inputs).float().cpu().numpy()


#             for pred, oid in zip(outputs, oids):
#                 pred = pred[0]  # shape (70, 70)
#                 for y_pos in range(70):
#                     row_id = f"{oid}_y_{y_pos}"
#                     odd_vals = pred[y_pos, 1::2]  # x_1, x_3, ..., x_69
#                     all_rows.append([row_id] + odd_vals.tolist())

#     columns = ["oid_ypos"] + [f"x_{i}" for i in range(1, 70, 2)]
#     df = pd.DataFrame(all_rows, columns=columns)
#     df.to_csv(output_csv, index=False)
#     print(f"Submission saved to {output_csv}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, required=True, help="Path to model weights")
#     parser.add_argument("--test-dir", type=str, required=True, help="Directory with .npy input files")
#     parser.add_argument("--output", type=str, required=True, help="Submission CSV file path")
#     parser.add_argument("--batch-size", type=int, default=32)
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--init-features", type=int, default=32)
#     parser.add_argument("--depth", type=int, default=5)
#     args = parser.parse_args()

#     model = load_model(args.model, args.init_features, args.depth, args.device)
#     make_submission(model, args.test_dir, args.output, args.batch_size, args.device)


# Manually defined arguments
model_path = "/kaggle/input/seismic-wave-inversion/pytorch/default/6/UNet_models/UNetFloat16TPUWithCheckpointMAX/checkpoint_all_UNetFloat16TPUWithCheckpointMAX_bs256_epoch08_valloss82.47.pth.pth"
test_dir = "/kaggle/input/open-wfi-test/test"
output_csv = "/kaggle/working/submission.csv"
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
init_features = 32
depth = 5
# Run prediction
model = load_model(model_path, init_features, depth, device)
make_submission(model, test_dir, output_csv, batch_size, device)



import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from data_helper import TestDataset
from torch.utils.data import DataLoader
import torch
from unet_def import UNet
from collections import OrderedDict

# Correct the line that caused the error by passing the actual function
import numpy as np
torch.serialization.add_safe_globals({'_reconstruct': np.core.multiarray._reconstruct})

# # --- Config ---
# model_path = "/kaggle/input/seismic-wave-inversion/pytorch/default/4/UNet_models/UNetFloat16TPUWithCheckpointMAX/checkpoint_all_UNetFloat16TPUWithCheckpointMAX_bs256_epoch11_valloss88.56.pth.pth"
# test_dir = "/kaggle/input/open-wfi-test/test"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# init_features = 32
# depth = 5
# batch_size = 8

# # --- Load model ---
# model = UNet(n_channels=5, n_classes=1, init_features=init_features, depth=depth).to(device)
# state_dict = torch.load(model_path, map_location=device, weights_only=False) # Changed weights_only to False based on previous context 
# state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
# model.load_state_dict(state_dict)
# model.eval()

# --- Load dataset ---
test_files = sorted(list(Path(test_dir).glob("*.npy")))
dataset = TestDataset(test_files)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# --- Choose 5 random indices ---
random.seed(42)
indices = random.sample(range(len(dataset)), 5)

# --- Predict and plot ---
fig, axes = plt.subplots(5, 2, figsize=(10, 15))
for i, idx in enumerate(indices):
    input_data, oid = dataset[idx]
    # Ensure input_data is a float32 tensor as expected by the model
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.inference_mode(): # Removed torch.autocast since it might not be needed or could cause issues without proper setup for mixed precision
        pred = model(input_tensor).squeeze().cpu().numpy()

    axes[i, 0].imshow(input_data[0], cmap="gray")
    axes[i, 0].set_title(f"Input (channel 0) - {oid}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(pred, cmap="viridis")
    axes[i, 1].set_title(f"Predicted Velocity - {oid}")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()




