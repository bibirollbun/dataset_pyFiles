import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision.transforms import v2
import os
from sklearn.model_selection import train_test_split
import random
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed=None, seed_torch=True):
    """
    Function that controls randomness. NumPy and random modules must be imported.

    Args:
      seed : Integer
        A non-negative integer that defines the random state. Default is `None`.
      seed_torch : Boolean
        If `True` sets the random seed for pytorch tensors, so pytorch module
        must be imported. Default is `True`.

    Returns:
      Nothing.
    """
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f'Random seed {seed} has been set.')


def plot_images_from_dataloader(dataloader):
    for images, labels in dataloader:
        # Plot batch of images
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        if len(images) == 1:
            axes = [axes]  # Ensure it's iterable for a single image

        for i, (image, label) in enumerate(zip(images, labels)):
            # Convert the image back to a NumPy array for plotting
            image = image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
            image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalize
            image = image.clip(0, 1)  # Clip values to valid range [0, 1]

            axes[i].imshow(image)
            axes[i].set_title(f'Label: {label.item()}')
            axes[i].axis('off')

        plt.show()
        break  # Display only the first batch


class DataProcessor:
    def __init__(self, dataset_root: str, csv_file: str, seed: int):
        self.dataset_root = dataset_root
        self.csv_file = csv_file
        self.seed = seed
        self.csv_path = os.path.join(self.dataset_root, self.csv_file)
        self.df = self.__preprocess_data()
        self.train_data, self.val_data, self.test_data = self.__split_data()

    def __preprocess_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df['ImagePath'] = df['ImageId'].apply(lambda id: os.path.join(self.dataset_root, 'train_v2', id))
        df['label'] = df['EncodedPixels'].notna().astype(int)
        df.drop(columns=['ImageId', 'EncodedPixels'], inplace=True)
        return df

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.train_data, self.val_data, self.test_data

    def __split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train, evaluation = train_test_split(
            self.df,
            test_size=0.2,
            random_state=self.seed,
            stratify=self.df['label']
        )
        val, test = train_test_split(
            evaluation,
            test_size=0.5,
            random_state=self.seed,
            stratify=evaluation['label']
        )
        return train, val, test


class ClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform or v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['ImagePath']
        label = self.df.iloc[idx]['label']

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


dataset_root = '/kaggle/input/airbus-ship-detection'
csv_file = '/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv'
seed = 2005
set_seed(seed)


dp = DataProcessor(dataset_root, csv_file, seed)
train, val, test = dp.get_data()


dataset = ClassificationDataset(train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


plot_images_from_dataloader(dataloader)

