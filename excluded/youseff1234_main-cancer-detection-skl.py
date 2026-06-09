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


filing = '/kaggle/input/rsna-breast-cancer-detection-poi-images/bc_1280_train_lut'


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# train_3_convnext_on_roi_with_epoch_saving.py

import os
import glob
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

from tqdm import tqdm
import numpy as np

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0) FORCE GPU USAGE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
assert torch.cuda.is_available(), "CUDA is not available â€“ check your Kaggle Accelerator setting!"
DEVICE = torch.device('cuda')
print(f"Using device: {DEVICE} â†’ {torch.cuda.get_device_name(0)}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONFIGURATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
IMAGE_DIR   = "/kaggle/input/rsna-breast-cancer-detection-poi-images/bc_1280_train_lut"
CSV_FILE    = "/kaggle/input/rsna-breast-cancer-detection/train.csv"
SAVE_DIR    = "/kaggle/working/model_weights"
BATCH_SIZE  = 32
LR          = 1e-4
EPOCHS      = 8       # decreased to 10
NUM_MODELS  = 3        # decreased to 3
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# 1) Dataset for ROI images named "<patient_id>_<image_id>.png"
class RoiDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.transform = transform
        df = pd.read_csv(csv_path, dtype=str)
        df['key'] = df['patient_id'].str.strip() + "_" + df['image_id'].str.strip()
        df['cancer'] = df['cancer'].astype(int)
        mapping = df.set_index('key')['cancer'].to_dict()

        all_paths = glob.glob(os.path.join(image_dir, "*.png"))
        self.samples = [
            (fp, mapping[os.path.splitext(os.path.basename(fp))[0]])
            for fp in all_paths
            if os.path.splitext(os.path.basename(fp))[0] in mapping
        ]
        if not self.samples:
            raise RuntimeError(f"No matching ROI PNGs in {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, label = self.samples[idx]
        img = Image.open(fp).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# 2) ConvNeXt factory
class Creating_Convnet:
    def creating_single_model(self):
        return models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        )

# 3) Full training harness with per-epoch best-model saving
class FullTrainingOfModel(Creating_Convnet):
    def __init__(self, train_ds, val_ds, test_ds,
                 batch_size, lr, num_epochs,
                 device, class_names):
        self.device       = device
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                       pin_memory=True, num_workers=4)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                       pin_memory=True, num_workers=2)
        self.test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                       pin_memory=True, num_workers=2)
        self.lr           = lr
        self.num_epochs   = num_epochs
        self.class_names  = class_names

        self.models       = {}
        self.histories    = {}
        self.best_val_loss= {}

    def initialize_models(self, num_models):
        for i in range(num_models):
            m = self.creating_single_model().to(self.device)
            opt = optim.Adam(m.parameters(), lr=self.lr)
            crit= nn.CrossEntropyLoss()
            name= f"model_{i}"
            self.models[name]        = m
            self.histories[name]     = {'opt':opt, 'crit':crit, 'train_loss':[], 'val_loss':[]}
            self.best_val_loss[name]= float('inf')

    def train(self):
        print("â–¶ï¸� Starting trainingâ€¦", flush=True)
        for name, m in self.models.items():
            h = self.histories[name]
            for epoch in range(1, self.num_epochs+1):
                print(f"\n--- [{name}] Epoch {epoch}/{self.num_epochs} ---", flush=True)

                # TRAIN LOOP
                m.train()
                running = 0.0
                for x, y in tqdm(self.train_loader,
                                 desc=f"{name}|train", unit="img", leave=False):
                    x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                    h['opt'].zero_grad()
                    out  = m(x)
                    loss = h['crit'](out, y)
                    loss.backward()
                    h['opt'].step()
                    running += loss.item() * x.size(0)
                tr_loss = running / len(self.train_loader.dataset)
                h['train_loss'].append(tr_loss)

                # VALIDATION LOOP
                m.eval()
                val_running = 0.0
                with torch.no_grad():
                    for x, y in tqdm(self.val_loader,
                                     desc=f"{name}|val", unit="img", leave=False):
                        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                        val_running += h['crit'](m(x), y).item() * x.size(0)
                val_loss = val_running / len(self.val_loader.dataset)
                h['val_loss'].append(val_loss)

                print(f"[{name}] Epoch {epoch} â†’ train={tr_loss:.4f}, val={val_loss:.4f}", flush=True)

                # SAVE BEST MODEL FOR THIS EPOCH
                if val_loss < self.best_val_loss[name]:
                    self.best_val_loss[name] = val_loss
                    best_path = os.path.join(self.save_dir, f"{name}_best.pth")
                    torch.save(m.state_dict(), best_path)
                    print(f"ğŸ’¾ New best for {name} (val_loss={val_loss:.4f}), saved to {best_path}", flush=True)

    def save_all_models(self, out_dir):
        # final save of last epoch
        for name, m in self.models.items():
            final_path = os.path.join(out_dir, f"{name}_final.pth")
            torch.save(m.state_dict(), final_path)
            print(f"âœ… Saved final {name} â†’ {final_path}", flush=True)

    def train_and_save(self, out_dir):
        # ensure save directory exists and store for use in train()
        os.makedirs(out_dir, exist_ok=True)
        self.save_dir = out_dir
        self.train()
        self.save_all_models(out_dir)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    # transforms
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # load & split dataset
    full_ds = RoiDataset(IMAGE_DIR, CSV_FILE, transform=tfm)
    print(f"Total ROI samples: {len(full_ds)}", flush=True)

    n = len(full_ds)
    n_tr  = int(0.8 * n)
    n_val = int(0.1 * n)
    n_te  = n - n_tr - n_val
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_tr, n_val, n_te],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Splits â†’ train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}", flush=True)

    # build & run trainer
    trainer = FullTrainingOfModel(
        train_ds, val_ds, test_ds,
        batch_size=BATCH_SIZE,
        lr=LR,
        num_epochs=EPOCHS,
        device=DEVICE,
        class_names=['no_tumor','tumor']
    )
    trainer.initialize_models(NUM_MODELS)
    trainer.train_and_save(SAVE_DIR)


