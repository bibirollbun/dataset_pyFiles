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


import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T



# 1. Load JSON File
with open('/kaggle/input/deforestation/train.json') as f:
    train_label = json.load(f)

# Ambil entry pertama
sample = train_label['0']
base_path = '/kaggle/input/deforestation/train/public/'


# 2. Load file .npy (img_t0, img_t1, mask)
img_t0 = np.load(os.path.join(base_path, sample['files']['satellite_img_first']))  # (512, 512, 13)
img_t1 = np.load(os.path.join(base_path, sample['files']['satellite_img_second'])) # (512, 512, 13)
mask = np.load(os.path.join(base_path, sample['files']['mask']))                   # (512, 512)



# 3. Normalisasi dan Ambil RGB (B4, B3, B2) â†’ Index 3, 2, 1
def normalize(x):
    x = x.astype(np.float32)
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-6)

# Ambil band RGB
img_t0_rgb = img_t0[:, :, [3, 2, 1]]
img_t1_rgb = img_t1[:, :, [3, 2, 1]]

# Normalisasi ke 0â€“1
img_t0_rgb = normalize(img_t0_rgb)
img_t1_rgb = normalize(img_t1_rgb)



# 4. Tampilkan Gambar
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_t0_rgb)
plt.title("ðŸŸ© Sebelum Deforestasi (T0)")

plt.subplot(1, 3, 2)
plt.imshow(img_t1_rgb)
plt.title("ðŸŸ¥ Setelah Deforestasi (T1)")

plt.subplot(1, 3, 3)
plt.imshow(mask, cmap='gray')
plt.title("â¬› Mask Deforestasi")

plt.tight_layout()
plt.show()


# 5. Hitung Luasan Deforestasi

# Jumlah piksel bernilai 1
def_pixels = np.sum(mask == 1)

# Luas 1 piksel Sentinel-2 (10m Ã— 10m) = 100 mÂ²
area_per_pixel_m2 = 100

# Total luas deforestasi
area_def_m2 = def_pixels * area_per_pixel_m2
area_def_ha = area_def_m2 / 10_000  # 1 ha = 10,000 mÂ²

print(f"Jumlah piksel deforestasi: {def_pixels}")
print(f"Luas deforestasi: {area_def_m2:,.0f} mÂ² ({area_def_ha:.2f} hektar)")


