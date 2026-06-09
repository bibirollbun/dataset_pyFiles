#Libraries from Demo
import os
import shutil
from collections import defaultdict

import pandas as pd
import polars as pl
import pydicom as dicom


#Libraries from attempt
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, numpy as np, torch

from collections import Counter
from scipy import ndimage

from scipy.ndimage import zoom as ndi_zoom
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from typing import Tuple, List

from sklearn.preprocessing import StandardScaler


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility in deep learning projects.
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_CSV = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
test_frac = 0.2
val_frac = 0.1
val_frac_within_trainval = val_frac / (1 - test_frac)
generated_mask_dir = '/kaggle/working/generated_masks'
seed = 42


train = pd.read_csv(TRAIN_CSV)
print(f"On the original Dataset, the percentage of aneurysms is: {100 * sum(train['Aneurysm Present'])/len(train)}%")
print(f"The original dataset has {len(train)} samples.")
print(f"The original dataset has {sum(train['Aneurysm Present']==1)} positive samples.")


train


train['Aneurysm Present']==1


PROCESSED_DATA_DIRS_MASKS = [
    "/kaggle/input/binary-masks-dataset/masks_quart1_2_5",
    "/kaggle/input/binary-masks-dataset/masks_quart2_2_5",
    "/kaggle/input/binary-masks-dataset/masks_quart3_2_5",
    "/kaggle/input/binary-masks-dataset/masks_quart4_2_5",
]
mask_sizes = []
all_files = []
sids_empty_not_lost = []
for mdir in PROCESSED_DATA_DIRS_MASKS:
    for mfile in os.listdir(mdir):
        all_files.append(os.path.join(mdir, mfile))
        sids_empty_not_lost.append(mfile[:-4])

print(f'After removing masks fully lost in the 160^3 volume we got from {len(train)} to {len(sids_empty_not_lost)}')


for fpath in tqdm(all_files, desc="Processing masks"):
    mask = np.load(fpath)["vol"].astype(np.float32)
    mask_sizes.append(float(mask.sum()))


max_size = max(mask_sizes)
print("Max mask size using a 1.5mm isotropic resampling with a 50 mm cube size:", max_size)


from collections import Counter

size_counts = Counter(mask_sizes)
size_counts_sorted = dict(sorted(size_counts.items()))
print(size_counts_sorted)


original_positive_samples = sum(train['Aneurysm Present']==1)
print(f'Originally we had a total number of postivie samples: {original_positive_samples} out of {len(train)}')


total = sum(size_counts_sorted.values())

print(f'After normalizing input size to 160 cubic sizes {original_positive_samples- total+2485} were lost, having now {total-2485}')
print(f'That leaves us with {100*(total-2485)/original_positive_samples}%')


samples_above_80 = 0
samples_above_75 = 0
samples_above_60 = 0
for size, count in size_counts_sorted.items():
    if size > 110592.0 * 0.8:
        samples_above_80 += count
    if size > 110592.0 * 0.75:
        samples_above_75 += count
    if size > 110592.0 * 0.60:
        samples_above_60 += count

print(f'If we want to keep masks that kept 80% of its original size after standardizing we get {samples_above_80} samples.')
print(f'This leaves us with the {100*samples_above_80/original_positive_samples}% of data')
print(f'If we want to keep masks that kept 75% of its original size after standardizing we get {samples_above_75} samples.')
print(f'This leaves us with the {100*samples_above_75/original_positive_samples}% of data')
print(f'If we want to keep masks that kept 60% of its original size after standardizing we get {samples_above_60} samples.')
print(f'This leaves us with the {100*samples_above_60/original_positive_samples}% of data')


sids_usefull = []
for fpath in tqdm(all_files, desc="Processing masks"):
    mask = np.load(fpath)["vol"].astype(np.float32)
    size = float(mask.sum())
    if size > 0 and size >= 110592.0 * 0.8:
        sids_usefull.append(fpath[len(mdir)+1:-4])
    if size == 0: 
        sids_usefull.append(fpath[len(mdir)+1:-4])


# print(len(sids_not_usefull)), 
# np.savez_compressed('/kaggle/working/not_usefull.npz', lst=sids_empty_not_lost)
# #242


print(len(sids_usefull)), 
np.savez_compressed('/kaggle/working/usefull.npz', lst=sids_usefull)

