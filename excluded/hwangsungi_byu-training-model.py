import os, cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from tqdm import tqdm




!pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1



try:
    import monai
    print("âœ… MONAI is already installed.")
except ImportError:
    print("ğŸ“¦ Installing MONAI from local .whl file...")
    !pip install /kaggle/input/monai-1-3/monai-1.3.0-202310121228-py3-none-any.whl

DATA_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
test_DIR = os.path.join(DATA_DIR,"test")





from collections import OrderedDict
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PatchBasedTrainDataset(Dataset):
    def __init__(self, tomo_root_dir, label_df, patch_size=(64, 256, 256), stride=(32, 128, 128), max_cache=4):
        self.tomo_root_dir = tomo_root_dir
        self.label_df = label_df
        self.patch_size = patch_size
        self.stride = stride
        self.patch_infos = []  # (tomo_id, z_start, y_start, x_start)
        self.volume_cache = OrderedDict()  # soft cache
        self.max_cache = max_cache

        self._build_index()

    def _build_index(self):
        unique_ids = self.label_df['tomo_id'].unique()
        print(f"Indexing patches from {len(unique_ids)} volumes...")

        for tomo_id in tqdm(unique_ids, desc="Indexing"):
            tomo_path = os.path.join(self.tomo_root_dir, tomo_id)
            slices = sorted(os.listdir(tomo_path))
            if not slices:
                continue

            sample = cv2.imread(os.path.join(tomo_path, slices[0]), cv2.IMREAD_GRAYSCALE)
            z = len(slices)
            y, x = sample.shape

            if z < self.patch_size[0] or y < self.patch_size[1] or x < self.patch_size[2]:
                continue

            for z0 in range(0, z - self.patch_size[0] + 1, self.stride[0]):
                for y0 in range(0, y - self.patch_size[1] + 1, self.stride[1]):
                    for x0 in range(0, x - self.patch_size[2] + 1, self.stride[2]):
                        self.patch_infos.append((tomo_id, z0, y0, x0))

        print(f"[DEBUG] Total patches created: {len(self.patch_infos)}")

    def __len__(self):
        return len(self.patch_infos)

    def _load_volume(self, tomo_id):
        tomo_path = os.path.join(self.tomo_root_dir, tomo_id)
        slices = sorted(os.listdir(tomo_path))
        volume = np.stack([
            cv2.imread(os.path.join(tomo_path, sl), cv2.IMREAD_GRAYSCALE)
            for sl in slices
        ]).astype(np.float32) / 255.0
        return volume

    def __getitem__(self, idx):
        tomo_id, z0, y0, x0 = self.patch_infos[idx]

        # soft cache with LRU strategy
        if tomo_id not in self.volume_cache:
            volume = self._load_volume(tomo_id)
            self.volume_cache[tomo_id] = volume
            if len(self.volume_cache) > self.max_cache:
                self.volume_cache.popitem(last=False)
        else:
            # move to end to mark as recently used
            self.volume_cache.move_to_end(tomo_id)

        volume = self.volume_cache[tomo_id]

        patch = volume[z0:z0+self.patch_size[0], y0:y0+self.patch_size[1], x0:x0+self.patch_size[2]]
        patch = np.expand_dims(patch, axis=0)  # (1, Z, Y, X)

        coords = self.label_df[self.label_df['tomo_id'] == tomo_id][['Motor axis 2', 'Motor axis 1', 'Motor axis 0']].values
        mask = np.zeros_like(patch[0], dtype=np.uint8)
        for x, y, z in coords:
            if z0 <= z < z0+self.patch_size[0] and y0 <= y < y0+self.patch_size[1] and x0 <= x < x0+self.patch_size[2]:
                mask[int(z - z0), int(y - y0), int(x - x0)] = 1
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(patch, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)






import torch
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.optim import Adam
from torch.amp import autocast, GradScaler

# í•™ìŠµì—� í•„ìš”í•œ íŒŒë�¼ë¯¸í„° ì„¤ì •
PATCH_SIZE = (64, 256, 256)
STRIDE = (32, 128, 128)
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
max_cache_num = 2

# ë�°ì�´í„°ì…‹ ìƒ�ì„±
dataset = PatchBasedTrainDataset(
    tomo_root_dir="/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train",
    label_df=pd.read_csv("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv"),
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    max_cache = max_cache_num
)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,     # ğŸ’¡ ì‹œìŠ¤í…œ ìƒ�í™© ë”°ë�¼ 2~8 ì‚¬ì�´ ì‹¤í—˜ ê°€ëŠ¥
    pin_memory=True,    # ğŸ’¡ GPUë¡œ tensor ì „ì†¡ì‹œ ì†�ë�„ í–¥ìƒ�
    persistent_workers=False,
    prefetch_factor = 2
)
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

# ì†�ì‹¤ í•¨ìˆ˜ ë°� ì˜µí‹°ë§ˆì�´ì €
loss_fn = DiceLoss(sigmoid=True)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()


# í•™ìŠµ ë£¨í”„
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):  # deprecation warning ëŒ€ì�‘
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")


# ëª¨ë�¸ ì €ì�¥
torch.save(model.state_dict(), "unet3d_patch_based.pth")



print(next(model.parameters()).device)



print(inputs.device, targets.device)
# ë‘˜ ë‹¤ cuda:0 ì�´ì–´ì•¼ GPUê°€ ì�‘ë�™í•´

















