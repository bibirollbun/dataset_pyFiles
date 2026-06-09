import os
import cv2
import random
from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import albumentations as A

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# paths and variables
train_root_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train"
train_labels = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv"

test_root_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test"

val_ratio = 0.2
batch_size = 4
num_workers = 4


class TomogramDataset(Dataset):
    def __init__(
        self, 
        root_path: str, 
        metadata: pd.DataFrame,
        num_negative_samples: int = 0,
        only_with_motor: bool = True,
        transforms: callable = None,
        augmentations: callable = None
    ):
        """
        Args:
            root_path (str): Path to the directory containing tomogram folder.
            metadata (pd.DataFrame): Label file.
            num_negative_samples (int): Number of negative samples to generate for the dataset from each tomogram. 
                                        If pssitive, random images are selected.
            only_with_motor (bool): Whether to filter out samples without motor data. If True, random image from tomogram is selected.
            transforms (callable): Torch transformation functions to apply to each image.
            augmentations (callable): Albumentation augmentation to apply random to the data.
        """
        self.data = []
        self.labels = []
        self.transforms = transforms
        self.augmentations = augmentations

        if only_with_motor:
            metadata = metadata[metadata['Number of motors'] != 0]

        tomo_ids = list(set(metadata["tomo_id"]))
        for tomo_id in tqdm(tomo_ids):
            tomo_data = metadata[metadata['tomo_id'] == tomo_id]
            tomo_folder = os.listdir(os.path.join(root_path, tomo_id))
            tomo_folder.sort()

            z_pos = [int(z) for z in list(tomo_data['Motor axis 0']) if z >= 0]
            z_avail = [i for i in range(len(tomo_folder)) if i not in z_pos]

            # add positive samples (and without motor)
            for ri, row in tomo_data.iterrows():
                z = int(row['Motor axis 0'])
                y = int(row['Motor axis 1'])
                x = int(row['Motor axis 2'])
                vs = float(row['Voxel spacing'])

                if z < 0:
                    z = np.random.choice(z_avail)
                    z_avail.remove(z)
                    
                _data = os.path.join(root_path, tomo_id, tomo_folder[z])
                _label = (x, y, vs)
                self.data.append(_data)
                self.labels.append(_label)

            # add negative samples
            if num_negative_samples > 0:
                vs = list(tomo_data['Voxel spacing'])[0]
                z = np.random.choice(z_avail, num_negative_samples)
                for _z in z:
                    _data = os.path.join(root_path, tomo_id, tomo_folder[_z])
                    _label = (-1, -1, vs)
                    self.data.append(_data)
                    self.labels.append(_label)

    def __len__(self):
        return len(self.data)

    def load_data(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
            
        return image
        
    def __getitem__(self, idx: int):
        """
        Returns:
            tuple: A tuple (image, xy, motor_label) where:
                - image (torch.Tensor): The transformed image tensor. Grayscale image is used.
                - xy (torch.Tensor): The coordinates (x, y) of the motor, if motor is not present (-1,-1) is returned.
                - motor_label (torch.Tensor): A label indicating whether the sample has a motor - one or zero.
        """
        data = self.data[idx]
        label = self.labels[idx]
        motor_label = torch.tensor(1 if label[0] >= 0 else 0)

        # load data
        xy = label[:2]
        image = self.load_data(data)
        # image = cv2.resize(image, (256,256))

        # apply augmentations
        if self.augmentations is not None:
            _xy = [xy] if xy[0] >= 0 else []
            augmented = self.augmentations(image=image, keypoints=_xy)
            image = augmented['image']
            xy = augmented['keypoints'][0] if _xy else xy
        
        # apply transforms
        if self.transforms is not None:
            image = self.transforms(image)
            h, w = image.shape[1:]
            xy = torch.tensor(xy).float() 
            if xy[0] >= 0:
                xy /= torch.tensor([w, h])
        
        return image, xy, motor_label


dataset = TomogramDataset(
    train_root_path, 
    pd.read_csv(train_labels),
    only_with_motor = True,
    num_negative_samples = 0
)

fig, ax = plt.subplots(4,4, figsize=(4*5,4*5))
ax = ax.flatten()
for ai, _ax in enumerate(ax):
    image, xy, motor_label = dataset[ai]
    x, y = xy
    if x >= 0:
        circle = Circle((x, y), radius=40, fill=False, edgecolor="tab:red", lw=3)
        ax[ai].add_patch(circle)
    ax[ai].imshow(image)
plt.show()

fig, ax = plt.subplots(4,4, figsize=(4*5,4*5))
ax = ax.flatten()
for ai, _ax in enumerate(ax):
    image, xy, motor_label = dataset[ai]
    x, y = xy
    if x >= 0:
        h, w = image[0].shape
        circle = Circle((x, y), radius=40, fill=False, edgecolor="tab:red", lw=4)
        ax[ai].add_patch(circle)
        ax[ai].set_ylim(y-100, y+100)
        ax[ai].set_xlim(x-100, x+100)
    ax[ai].imshow(image)
plt.show()


# split data
all_data = pd.read_csv(train_labels)
tomo_ids = list(set(all_data["tomo_id"]))
random.shuffle(tomo_ids)

idx = int(np.ceil(len(tomo_ids) * val_ratio))
val_tomo_ids = tomo_ids[:idx]
train_tomo_ids = tomo_ids[idx:]

val_data = all_data[all_data["tomo_id"].isin(val_tomo_ids)]
train_data = all_data[all_data["tomo_id"].isin(train_tomo_ids)]


# prepare transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

resize = [
    A.LongestMaxSize(512, p=1),
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=0, p=1)
    # A.Resize(height=256, width=256, p=1)
]

augmentation_train = A.Compose([
    A.RandomRotate90(p=1),
    A.HorizontalFlip(p=0.5),
    *resize
], keypoint_params=A.KeypointParams(format='xy'))

augmentation_val = A.Compose([
    *resize
], keypoint_params=A.KeypointParams(format='xy'))


# create datasets and dataloaders
dataset_train = TomogramDataset(
    train_root_path, 
    train_data,
    only_with_motor = False,
    num_negative_samples = 2,
    transforms = transform,
    augmentations = augmentation_train
)

dataset_val = TomogramDataset(
    train_root_path, 
    val_data,
    only_with_motor = True,
    num_negative_samples = 0,
    transforms = transform,
    augmentations = augmentation_val
)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)




