import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob
from itertools import product

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import torch


COMP_DIR = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
LABEL = f"{COMP_DIR}/train_labels.csv"
TRAIN_DIR = f"{COMP_DIR}/train"
TAU = 1000



def stack_slices(slice_paths: list[str]) -> np.ndarray:
    tomo = []
    for slice_ in slice_paths:
        img = cv2.imread(slice_, -1)
        img = img / 255
        tomo.append(img)
    return np.array(tomo)


def set_label(array: np.ndarray, radius: int, z: int, y: int, x: int) -> np.ndarray:
    if z <= 0 or y <= 0 or x <= 0:
        return array

    shape = array.shape

    z, y, x = int(z), int(y), int(x)

    z_min = max(0, z - radius)
    z_max = min(shape[0], z + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(shape[1], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(shape[2], x + radius + 1)

    for zi, yi, xi in product(
        range(z_min, z_max), range(y_min, y_max), range(x_min, x_max)
    ):
        if (zi - z) ** 2 + (yi - y) ** 2 + (xi - x) ** 2 <= radius**2:
            array[zi, yi, xi] += 1
    return array


def create_label_array(
    tomo_shape: tuple[int, int, int], coords: np.ndarray, radius: int
) -> np.ndarray:
    label = np.zeros(tomo_shape)
    for coord in coords:
        z = coord[0]
        y = coord[1]
        x = coord[2]
        label = set_label(label, radius, z, y, x)

    label = np.clip(label, 0, 1)

    return label


def scale_array(array: np.ndarray, scale: float) -> np.ndarray:
    array = torch.Tensor(array).unsqueeze(0).unsqueeze(0)
    array = F.interpolate(array, scale_factor=scale, mode="trilinear")
    array = array.squeeze().numpy()
    return array


def process(
    tomo_dir: str,
    dataset_dir: str,
    scale: float,
    radius_multi: float,
    df_label: pd.DataFrame,
):
    tomo_id = tomo_dir.split("/")[-1]
    df_label_ = df_label[df_label["tomo_id"] == tomo_id]

    tomo_save_path = f"{dataset_dir}/{tomo_id}/tomo.npy"
    label_save_path = f"{dataset_dir}/{tomo_id}/label.npy"

    os.makedirs(f"{dataset_dir}/{tomo_id}", exist_ok=True)

    slice_paths = sorted(glob(f"{tomo_dir}/*.jpg"))
    tomo = stack_slices(slice_paths)
    tomo_shape = tomo.shape
    tomo = scale_array(tomo, scale)
    np.save(tomo_save_path, tomo)

    coords = df_label_[["Motor axis 0", "Motor axis 1", "Motor axis 2"]].values
    spacing = df_label_["Voxel spacing"].values[0]
    radius = int(TAU / spacing * radius_multi)
    label = create_label_array(tomo_shape, coords, radius)
    label = scale_array(label, scale).astype(np.bool_)
    np.save(label_save_path, label)


dataset_dir = "/kaggle/working/dataset/train"
train_tomo_dirs = sorted(glob(f"{TRAIN_DIR}/*"))
df_label = pd.read_csv(LABEL)
shutil.rmtree(dataset_dir, ignore_errors=True)

train_tomo_dirs = train_tomo_dirs[:5]
scale = 0.5
radius_multi = 0.2

# n_workers = 2
# with ProcessPoolExecutor(max_workers=n_workers) as executor:
#     process_func = partial(
#         process_tomo,
#         dataset_dir=dataset_dir,
#         df_label=df_label,
#         scale=scale
#         radius_multi=radius_multi,
#     )
#     results = list(
#         tqdm(executor.map(process_func, train_tomo_dirs), total=len(train_tomo_dirs))
#     )

for tomo_dir in tqdm(train_tomo_dirs):
    process(tomo_dir, dataset_dir, scale, radius_multi, df_label)



import matplotlib.pyplot as plt

tomo_dirs = sorted(glob(f"{dataset_dir}/*"))
for tomo_dir in tomo_dirs:
    tomo_id = tomo_dir.split("/")[-1]

    tomo_path = f"{tomo_dir}/tomo.npy"
    label_path = f"{tomo_dir}/label.npy"

    tomo = np.load(tomo_path)
    label = np.load(label_path)

    if label.sum() == 0:
        continue

    tomo_shape = tomo.shape
    for i_z in range(50, tomo_shape[0] - 50, 2):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(tomo[i_z])
        ax[1].imshow(label[i_z])
        fig.suptitle(f"{tomo_id} z={i_z}")
        plt.show()

    break


