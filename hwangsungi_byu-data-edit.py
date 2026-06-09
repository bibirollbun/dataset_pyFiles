# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import numpy as np
from torch.utils.data import Dataset

class LazyLoadingDataset(Dataset):
    def __init__(self, patch_dirs):
        self.file_paths = []
        for d in patch_dirs:
            self.file_paths += [
                os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")
            ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        patch = np.load(self.file_paths[idx])  # ë˜�ëŠ” torch.load
        return patch



!pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1



try:
    import monai
    print("âœ… MONAI is already installed.")
except ImportError:
    print("ğŸ“¦ Installing MONAI from local .whl file...")
    !pip install /kaggle/input/monai-1-3/monai-1.3.0-202310121228-py3-none-any.whl





import numpy as np
import torch
from torch.utils.data import Dataset

PATCH_D, PATCH_H, PATCH_W = 64, 256, 256

class OnTheFlyPatchDataset(Dataset):
    def __init__(self, npz_files):
        self.vol_paths = npz_files
        self.records = []             # (file_idx, z, y, x)
        self._vol_cache = (None, None)  # (file_idx, (volume, mask))

        for fi, path in enumerate(npz_files):
            with np.load(path) as d:
                infos = d["patch_infos"].astype(int)
            self.records.extend([(fi, z, y, x) for z, y, x in infos])

    def _get_volume(self, fi):
        if self._vol_cache[0] != fi:
            with np.load(self.vol_paths[fi]) as d:
                vol = d["volume"].astype(np.float32)
                msk = d["mask"].astype(np.float32)
            self._vol_cache = (fi, (vol, msk))
        return self._vol_cache[1]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fi, z, y, x = self.records[idx]
        vol, msk = self._get_volume(fi)

        patch  = vol[z:z+PATCH_D, y:y+PATCH_H, x:x+PATCH_W]
        target = msk[z:z+PATCH_D, y:y+PATCH_H, x:x+PATCH_W]

        patch_tensor  = torch.from_numpy(patch).unsqueeze(0).contiguous()   # (1,D,H,W)
        target_tensor = torch.from_numpy(target).unsqueeze(0).contiguous()
        return patch_tensor, target_tensor



class LazyMultiNPZDataset(Dataset):
    def __init__(self, npz_paths):
        self.npz_paths = npz_paths
        self.index_map = []  # (file_idx, local_patch_idx)

        self.vol_infos = []
        for fi, path in enumerate(npz_paths):
            with np.load(path) as d:
                infos = d['patch_infos'].astype(int)
            self.vol_infos.append((path, len(infos)))
            self.index_map.extend([(fi, i) for i in range(len(infos))])

        self._vol_cache = (None, None)  # (fi, (volume, mask, infos))

    def _load_volume(self, fi):
        if self._vol_cache[0] != fi:
            path, _ = self.vol_infos[fi]
            d = np.load(path)
            vol = d['volume'].astype(np.float32)
            msk = d['mask'].astype(np.float32)
            infos = d['patch_infos'].astype(int)
            self._vol_cache = (fi, (vol, msk, infos))
        return self._vol_cache[1]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fi, patch_idx = self.index_map[idx]
        vol, msk, infos = self._load_volume(fi)
        z, y, x = infos[patch_idx]
        patch = vol[z:z+64, y:y+256, x:x+256]
        target = msk[z:z+64, y:y+256, x:x+256]
        return torch.from_numpy(patch).unsqueeze(0), torch.from_numpy(target).unsqueeze(0)



# 1. Dataset ì •ì�˜
from tqdm import tqdm
from torch.utils.data import DataLoader
# 1. Dataset ì •ì�˜
patch_dirs = [
    "/kaggle/input/byu-flagellar-part-01",
    *[f"/kaggle/input/byu-preprocessed-part-{i:02d}" for i in range(2, 34)]
]

npz_paths = [
    os.path.join(d, f)
    for d in patch_dirs
    for f in os.listdir(d)
    if f.endswith(".npz")
]

dataset = LazyMultiNPZDataset(npz_paths)




import torch
from torch.utils.data import DataLoader, Subset
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# í•™ìŠµ íŒŒë�¼ë¯¸í„°
PATCH_SIZE = (64, 256, 256)
STRIDE = (32, 128, 128)
BATCH_SIZE = 1            # âœ… ì•ˆì • ìš°ì„ 
NUM_WORKERS = 1           # âœ… worker ì—†ì�Œ (RAM ì•ˆì „)
CHUNK_SIZE = 10000
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ëª¨ë�¸ ì •ì�˜
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm='instance'
).to(DEVICE)

loss_fn = DiceLoss(sigmoid=True)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()

# ì „ì²´ ë�°ì�´í„° í�¬ê¸°
TOTAL_SIZE = len(dataset)

# Chunk ë‹¨ìœ„ë¡œ í•™ìŠµ
for chunk_start in range(0, TOTAL_SIZE, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, TOTAL_SIZE)
    subset = Subset(dataset, list(range(chunk_start, chunk_end)))

    dataloader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor = 2
    )

    print(f"\nğŸ”¥ Training patch chunk [{chunk_start} ~ {chunk_end}]")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        print(f"\nğŸŒ€ Epoch {epoch+1}/{EPOCHS}")

        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"[Chunk {chunk_start}-{chunk_end}] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    # ì²­í�¬ë³„ ì €ì�¥
    filename = f"unet3d_chunk_{chunk_start}_{chunk_end}.pth"
    torch.save(model.state_dict(), filename)
    print(f"âœ… ëª¨ë�¸ ì €ì�¥ ì™„ë£Œ: {filename}")



patch_dirs = [
    "/kaggle/input/byu-flagellar-part-01",
    *[f"/kaggle/input/byu-preprocessed-part-{i:02d}" for i in range(2, 34)]
]

npz_paths = [
    os.path.join(d, f)
    for d in patch_dirs
    for f in os.listdir(d)
    if f.endswith(".npz")
]





