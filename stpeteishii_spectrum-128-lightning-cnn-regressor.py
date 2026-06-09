!pip install lightning


import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split, SubsetRandomSampler
from torchvision import datasets, transforms, models 
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lightning.pytorch import LightningDataModule
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
import lightning.pytorch as L
print(L.__version__)

import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image


transform = transforms.Compose([
            #transforms.Resize(224),             # resize shortest side to 224 pixels
            #transforms.CenterCrop(224),         # crop longest side to 224 pixels at center            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


df1=pd.read_csv('/kaggle/input/visible-spectrum-split-data-by-shape/data128.csv')
df2=pd.read_csv('/kaggle/input/visible-spectrum-split-data-by-shape/data057.csv')
df=pd.concat([df1,df2],axis=0)
display(df[0:3])
print(len(df))
train=df[df['traintest']=='train']
test=df[df['traintest']=='test']
print(len(train),len(test))


path_label = list(train[['path','label']].itertuples(index=False, name=None))
tpath_label = list(test[['path','label']].itertuples(index=False, name=None))
print(path_label[0:3])


class Custom3DDataset(Dataset):
    def __init__(self, path_label, transform=None):
        self.path_label = path_label
        self.transform = transform

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        file_path, label = self.path_label[idx]
        
        # Load the 3D data (assumes it's a NumPy array)
        data = np.load(file_path)  # shape: (128, 128, 125)
        
        # Add channel dimension: (C, D, H, W)
        data = torch.from_numpy(data).float().unsqueeze(0)
        
        if self.transform:
            data = self.transform(data)
            
        return data, torch.tensor(label, dtype=torch.float)


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_path_label: str,  # Required
        test_path_label: str = None,  # Optional
        root_dir: str = None,
        batch_size: int = 32,
        val_split: float = 0.2,  # 20% for validation
    ):
        super().__init__()
        self.train_path_label = train_path_label
        self.test_path_label = test_path_label
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.val_split = val_split

        # Transformations for 3D data
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Load full training dataset
        full_dataset = Custom3DDataset(
            self.train_path_label, 
            transform=self.transform
        )

        # Split into train/val
        dataset_size = len(full_dataset)
        val_size = int(self.val_split * dataset_size)
        train_size = dataset_size - val_size
        
        indices = list(range(dataset_size))
        self.train_dataset = Subset(full_dataset, indices[:train_size])
        self.val_dataset = Subset(full_dataset, indices[train_size:])

        # Load test data if provided
        if self.test_path_label:
            self.test_dataset = Custom3DDataset(
                self.test_path_label,
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset not provided. Set test_path_label.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size
        )


class ConvolutionalRegressor3D(LightningModule):
    def __init__(self):
        super(ConvolutionalRegressor3D, self).__init__()
        
        # Input shape: [batch, 1, 128, 128, 125]
        self.conv1 = nn.Conv3d(1, 6, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(6)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # -> [batch, 6, 64, 64, 62]
        
        self.conv2 = nn.Conv3d(6, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # -> [batch, 16, 32, 32, 31]
        
        # Compute the input size of the fully connected layer
        self.flattened_size = self._get_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)  # Output is 1 for regression task
        
        # Initialize weights
        self._initialize_weights()
        
    def _get_flattened_size(self):
        """Calculate the size after flattening using dummy input"""
        dummy_input = torch.zeros(1, 1, 128, 128, 125)
        x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        return x.numel()  # 16 * 32 * 32 * 31 = 507,904
        
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: [batch, 1, 128, 128, 125]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # -> [batch, 6, 64, 64, 62]
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # -> [batch, 16, 32, 32, 31]
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation since this is a regression task
        return x

    def preprocess_data(self, x):
        """
        Normalize each sample individually to zero mean and unit variance
        across spatial dimensions (D, H, W) only
        """
        # Calculate mean and std over spatial dimensions for each sample
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        std = x.std(dim=(2, 3, 4), keepdim=True)
        
        # Normalize with numerical stability
        eps = 1e-5
        normalized_x = (x - mean) / (std + eps)
        
        # Clamp values to prevent extreme outliers
        normalized_x = torch.clamp(normalized_x, -5.0, 5.0)
        
        return normalized_x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.preprocess_data(x)
        y_hat = self(x)
        
        # Ensure target has correct shape and type
        y = y.float()
        if len(y.shape) > 1:
            y = y.squeeze(-1)
        
        loss = F.mse_loss(y_hat, y)
        
        # Log training metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.preprocess_data(x)
        y_hat = self(x)
        
        y = y.float()
        if len(y.shape) > 1:
            y = y.squeeze(-1)
        
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.preprocess_data(x)
        y_hat = self(x)
        
        y = y.float()
        if len(y.shape) > 1:
            y = y.squeeze(-1)
        
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0005, weight_decay=1e-4)
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            },
            'gradient_clip_val': 0.5,
            'gradient_clip_algorithm': 'value'
        }






if __name__ == '__main__':
    # Initialize data module and model
    datamodule = DataModule(train_path_label=path_label, test_path_label=tpath_label)
    datamodule.setup()
    
    # Initialize model with additional debugging
    model = ConvolutionalRegressor3D()
    
    # Setup trainer with early stopping and checkpointing
    trainer = L.Trainer(
        max_epochs=30,
        callbacks=[
            L.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            ),
            L.callbacks.ModelCheckpoint(
                monitor='val_loss',
                save_top_k=1,
                mode='min'
            )
        ],
        log_every_n_steps=10,
        detect_anomaly=True  # Enable anomaly detection for debugging
    )
    
    # Train the model
    trainer.fit(model, datamodule)
    
    # Test the model
    test_loader = datamodule.test_dataloader()
    trainer.test(dataloaders=test_loader)





device = torch.device("cpu")   #"cuda:0"
model.eval()
y_pred=[]
with torch.no_grad():
    for test_data in datamodule.test_dataloader():
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        pred = model(test_images)  ##########
        for i in range(len(pred)):
            y_pred.append(pred[i].item())


test['label']=y_pred
display(test)


test.to_csv('submission_128.csv',index=False)




