# Importing Libraries
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.tv_tensors import Mask

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


# Set seed for reproducibility
SEED = 42

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Load the dataset
dataset = pd.read_csv('/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv')
dataset.head(10)


dataset.info()


duplicate_percent = dataset.duplicated(subset=['ImageId']).sum() / dataset.shape[0] * 100
print("Percentages of duplicate rows: {:.5}".format(duplicate_percent))


class SegmentationDataProcessor:
    def __init__(self, dataset_root: str, csv_file: str, seed: int, image_subdir: str = 'train_v2'):
        self.dataset_root = dataset_root
        self.csv_file = csv_file
        self.seed = seed
        self.image_subdir = image_subdir
        self.csv_path = os.path.join(self.dataset_root, self.csv_file)

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self._split_data(self._preprocess_and_group())

    def _preprocess_and_group(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df.dropna(subset=['EncodedPixels'], inplace=True)
        image_dir = os.path.join(self.dataset_root, self.image_subdir)
        df['ImageId'] = df['ImageId'].apply(lambda image_id: os.path.join(image_dir, image_id))

        # Group RLEs by ImageId
        grouped_df = df.groupby('ImageId')['EncodedPixels'].apply(list).reset_index()
        return grouped_df

    def _split_data(self, df: pd.DataFrame):
        train, evaluation = train_test_split(
            df,
            test_size=0.2,
            random_state=self.seed,
            shuffle=True
        )
        val, test = train_test_split(
            evaluation,
            test_size=0.5,
            random_state=self.seed,
        )
        return train, val, test


%%time
train, test, val = SegmentationDataProcessor(
    dataset_root = '/kaggle/input/airbus-ship-detection',
    csv_file = 'train_ship_segmentations_v2.csv', 
    seed = SEED
).get_data()


class SegmentationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, original_shape: tuple[int, int] = (768, 768), transform=None):
        self.df = df
        self.original_shape = original_shape
        self.transform = transform or v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
        ]
    )
        self.image_dtype_transform = v2.ToDtype(torch.float32, scale=True)
        self.mask_dtype_transform = v2.ToDtype(torch.float32, scale=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path, rle_list = self.df.iloc[idx]
        combined_mask = torch.zeros(self.original_shape, dtype=torch.uint8)

        for rle_mask_str in rle_list:
            single_mask = self.rle_decode(rle_mask_str)
            combined_mask = torch.logical_or(combined_mask, single_mask.to(combined_mask.device))

        mask = Mask(combined_mask.to(torch.uint8))

        try:
            image = Image.open(image_path).convert('RGB')
            image, mask = self.transform(image, mask)

            image = self.image_dtype_transform(image)
            mask = self.mask_dtype_transform(mask)
        except (OSError, UnidentifiedImageError):
            next_idx = idx + 1
            return self.__getitem__(next_idx)

        return image, mask

    @staticmethod
    def rle_decode(mask_rle: str, shape: tuple[int, int] = (768, 768)) -> torch.Tensor:
        s_np = np.asarray(mask_rle.split(), dtype=int)     

        starts = torch.from_numpy(s_np[0::2] - 1) 
        lengths = torch.from_numpy(s_np[1::2])
        ends = starts + lengths

        img_size = shape[0] * shape[1]

        temp_ary = torch.zeros(img_size + 1, dtype=torch.int16)
        # index_add_(dim, index, tensor) -> adds tensor elements to self at indices in index
        temp_ary.index_add_(0, starts, torch.ones_like(starts, dtype=torch.int16))
        temp_ary.index_add_(0, ends, torch.full_like(ends, -1, dtype=torch.int16)) # Add -1 at ends

        # Compute cumulative sum and reshape
        flat_mask = torch.cumsum(temp_ary, dim=0)[:-1] # Remove the extra element

        # Reshape to (W, H) then transpose to (H, W)
        mask = flat_mask.reshape((shape[1], shape[0])).T

        # Return the mask as a tensor of type uint8
        return (mask > 0).to(torch.uint8) 


train_data = SegmentationDataset(train)
train_dataloader = DataLoader(dataset=train_data,
                                    batch_size=16,
                                    num_workers=0,
                                    shuffle=False) 


len(train_dataloader)


image, mask = next(iter(train_dataloader))

def display_image(image_batch: torch.Tensor, mask_batch: torch.Tensor, idx: int=2, alpha: float=0.3, cmap: str='gray'):
    # Select the sample image and its corresponding mask
    image = image_batch[idx].permute(1, 2, 0).cpu().numpy()
    mask = mask_batch[idx].cpu().numpy()

    # Create subplots: original image, mask, and overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    
    # Display mask only
    axes[1].imshow(mask, cmap=cmap)
    axes[1].set_title("Mask")
    
    # Display overlay of mask on image
    axes[2].imshow(image)
    axes[2].imshow(mask, cmap=cmap, alpha=alpha)
    axes[2].set_title("Overlay")
    
    # Remove axis ticks for clarity
    for ax in axes:
        ax.axis('off')
    
    # Set a suptitle for the entire figure
    fig.suptitle(f"Sample {idx} with Mask Overlay", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()

image, mask = next(iter(train_dataloader))
display_image(image, mask)




