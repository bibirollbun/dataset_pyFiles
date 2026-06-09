import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class CropsDataset(Dataset):

    def __init__(self, csv_file, root_dir, return_labels=True):

        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.return_labels = return_labels
        
        self.training = "label" in self.frame.columns
        self.train_x = self.frame if not self.training else self.frame.drop("label", axis=1)
    
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.train_x.iloc[idx, 0])
        image = np.load(img_name)
        # dropping some first and last spectral bands
        image = image[..., 10:110]
    
            
        if not self.training or not self.return_labels:
            return image
        
        label = self.frame.iloc[idx, 1]
        return (image, label)


ROOT = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot"
TRAIN_PATH = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/train.csv"


dataset = CropsDataset(TRAIN_PATH, ROOT, return_labels=False)


from sklearn.decomposition import IncrementalPCA

# using incremental algorithm version for memory efficiency
ipca = IncrementalPCA()

# Process images in batches
batch_size = 100  # Adjust based on memory
for i in tqdm(range(0, len(dataset), batch_size)):
    
    # the last batch is less than batch_size elements
    if i + batch_size >= len(dataset):
        batch = [dataset[j] for j in range(i, len(dataset) - 1)]
    else:
        batch = [dataset[j] for j in range(i, i+batch_size)]
    
    # reshaping hyperspectral images into 128*128 by 100 matrices
    # then stacking them on top of each other to use as input for PCA
    batch_pixels = np.vstack([img.reshape(-1, 100) for img in batch])
    ipca.partial_fit(batch_pixels)


ex_var = np.array(ipca.explained_variance_ratio_)
np.cumsum(ex_var)

