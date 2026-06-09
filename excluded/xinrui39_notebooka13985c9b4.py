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


import pandas as pd
from pathlib import Path
import shutil, os

# === Step 1. 载入 CSV 文件 ===
data_root = Path("/kaggle/input/rsna-intracranial-aneurysm-detection")
df = pd.read_csv(data_root / "train.csv")

# === Step 2. 筛选 MRI 模态 ===
df_mri = df[df["Modality"].str.contains("MR", case=False, na=False)]

# === Step 3. 检查哪些 MRI 有 segmentation 文件 ===
seg_dir = data_root / "segmentations"
seg_files = {f.stem for f in seg_dir.glob("*.nii*")}
df_mri_seg = df_mri[df_mri["SeriesInstanceUID"].isin(seg_files)]

print(f"Found {len(df_mri_seg)} MRI cases with segmentation available.")
df_mri_seg.head()



import os
import numpy as np
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
DATA_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection"
SERIES_DIR = f"{DATA_DIR}/series"
SEG_DIR = f"{DATA_DIR}/segmentations"



!ls /kaggle/input/rsna-intracranial-aneurysm-detection


!ls /kaggle/input/rsna-intracranial-aneurysm-detection/segmentations | head



nii_files = sorted(glob(os.path.join(SEG_DIR, "*.nii")))
# 匹配 MRI 和 segmentation 对
pairs = []
for f in nii_files:
    if f.endswith("_cowseg.nii"):
        base = f.replace("_cowseg.nii", ".nii")
        if os.path.exists(base):
            pairs.append((base, f))

print(f"Found {len(pairs)} paired MRI & segmentation files.")
if pairs:
    print("Example pair:\n", pairs[0])


# 选一个 pair
mri_path, seg_path = pairs[0]

# 读取 MRI 和 segmentation
mri_img = nib.load(mri_path)
seg_img = nib.load(seg_path)

mri_data = mri_img.get_fdata()
seg_data = seg_img.get_fdata()

print("MRI shape:", mri_data.shape)
print("Segmentation shape:", seg_data.shape)
print("Segmentation unique labels:", np.unique(seg_data))



def show_overlay(slice_idx):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(mri_data[:,:,slice_idx], cmap='gray')
    plt.title(f"MRI Slice {slice_idx}")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(mri_data[:,:,slice_idx], cmap='gray')
    plt.imshow(seg_data[:,:,slice_idx], cmap='autumn', alpha=0.4)
    plt.title(f"Overlay Slice {slice_idx}")
    plt.axis('off')
    plt.show()

# 展示几层
for i in [1, 50, 80]:
    show_overlay(i)



nonzero_ratio = np.count_nonzero(seg_data) / seg_data.size
print(f"Non-zero voxel ratio: {nonzero_ratio:.6f}")

unique_vals, counts = np.unique(seg_data, return_counts=True)
for u, c in zip(unique_vals, counts):
    if u != 0:
        print(f"Label {int(u):2d}: {c} voxels")



summary = []
for mri_path, seg_path in pairs:
    seg_data = nib.load(seg_path).get_fdata()
    nonzero = np.count_nonzero(seg_data) / seg_data.size
    labels = np.unique(seg_data)
    summary.append((os.path.basename(seg_path), seg_data.shape, nonzero, labels))

print("Example results:")
for name, shape, nonzero, labels in summary[:5]:
    print(f"{name:80s}  shape={shape}, nonzero_ratio={nonzero:.6f}, labels={labels}")





