import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

#gpus he
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## this is default loading from torchvision.datasets thing
# Data loading and preprocessing
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# train_dataset = CIFAR100(root='./data', train=True, transform=transform, download=True)
# test_dataset = CIFAR100(root='./data', train=False, transform=transform, download=True)

# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)




# import h5py
# from torch.utils.data import DataLoader, Dataset

# # class CIFAR100H5Dataset(Dataset):
# #     def __init__(self, hdf5_file, train=True):
# #         self.hf = h5py.File(hdf5_file, 'r')
# #         self.images = self.hf['images']
# #         self.labels = self.hf['labels'] if train else None

# #     def __len__(self):
# #         return len(self.images)
    
# #     def __getitem__(self, idx):
# #         if self.labels is not None:
# #             return self.images[idx], self.labels[idx]
# #         return self.images[idx]
    

# class CIFAR100H5Dataset(Dataset):
#     def __init__(self, hdf5_file, train=True):
#         self.hf = h5py.File(hdf5_file, 'r')
#         self.images = self.hf['images']
#         self.labels = self.hf['labels'] if train else None

#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         image = torch.tensor(self.images[idx])
#         if self.labels is not None:
#             label = torch.tensor(self.labels[idx])
#             return image.permute(2, 0, 1), label
#         return image.permute(2, 0, 1)
    
    
# # Load datasets from HDF5
# train_dataset = CIFAR100H5Dataset('train.h5', train=True)
# test_dataset = CIFAR100H5Dataset('test.h5', train=False)



## just a simple function to randomly display images from the dataset

import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader

def plot_random_images(dataset, num_images=5):
    """Plot random images from the dataset."""
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    indices = random.sample(range(len(dataset)), num_images)
    
    for idx, ax in zip(indices, axes):
        image, label = dataset[idx]
        image = image.permute(1, 2, 0).numpy()  #HWC format
        ax.imshow(image)
        ax.set_title(f"Label: {label.item()}")
        ax.axis('off')
    
    plt.show()

# plot_random_images(train_dataset, num_images=10)


# import pandas as pd
# import numpy as np

# def convert_csv_to_parquet(csv_file, parquet_file, is_train=True):
#     # Load the entire CSV file into a DataFrame
#     data = pd.read_csv(csv_file)
    
#     # Convert the 'image' column from strings to arrays of integers
#     data['image'] = data['image'].apply(lambda x: np.fromstring(x.strip("[]"), sep=', ', dtype=np.uint8).tolist())
    
#     # Save to Parquet format
#     if is_train:
#         data[['image', 'TARGET']].to_parquet(parquet_file, index=False, compression='snappy')
#     else:
#         data[['image']].to_parquet(parquet_file, index=False, compression='snappy')

# convert_csv_to_parquet('train.csv', 'train_data.parquet', is_train=True)
# convert_csv_to_parquet('test.csv', 'test_data.parquet', is_train=False)



# import torch
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np

# class CIFAR100ParquetDataset(Dataset):
#     def __init__(self, parquet_file, transform=None, target_column='TARGET'):
#         self.data = pd.read_parquet(parquet_file)
#         self.transform = transform
#         self.target_column = target_column
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         image = np.array(self.data.iloc[idx]['image'], dtype=np.uint8).reshape(32, 32, 3)  # Adjust if needed
#         image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # Normalize and permute

#         if self.target_column in self.data.columns:
#             target = self.data.iloc[idx][self.target_column]
#             return image_tensor, target
#         return image_tensor

# train_dataset = CIFAR100ParquetDataset('train_data.parquet', transform=None)
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

# test_dataset = CIFAR100ParquetDataset('test_data.parquet', transform=None)
# test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)



# class CIFAR100TensorDataset(Dataset):
#     def __init__(self, data_path):
#         self.images, self.targets = torch.load(data_path)
    
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         if self.targets is not None:
#             return self.images[idx], self.targets[idx]
#         return self.images[idx]

# train_dataset = CIFAR100TensorDataset('train_data.pt')
# test_dataset = CIFAR100TensorDataset('test_data.pt')

# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)



# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import ast

# class CustomCIFAR100Dataset(Dataset):
#     def __init__(self, csv_file, transform=None, is_train=True):
#         """
#         Args:
#             csv_file (str): Path to the csv file with annotations.
#             transform (callable, optional): Optional transform to be applied on a sample.
#             is_train (bool): Whether this is training data (with targets) or test data.
#         """
#         self.data_frame = pd.read_csv(csv_file)
#         self.transform = transform
#         self.is_train = is_train
        
#     def __len__(self):
#         return len(self.data_frame)
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
            
#         # Convert string representation of numpy array back to array
#         image_str = self.data_frame.iloc[idx]['image']
#         # Convert string to numpy array using ast.literal_eval
#         image_array = np.array(ast.literal_eval(image_str))
        
#         # Convert to torch tensor and ensure correct format (C,H,W)
#         image = torch.from_numpy(image_array).float() / 255.0  # Normalize to [0, 1]
#         if image.shape[-1] == 3:  # If image is (H,W,C)
#             image = image.permute(2, 0, 1)  # Convert to (C,H,W)
            
#         if self.transform:
#             image = self.transform(image)
            
#         if self.is_train:
#             target = self.data_frame.iloc[idx]['TARGET']
#             return image, target
#         else:
#             return image, self.data_frame.iloc[idx]['ID']

# from torch.utils.data import DataLoader
# from torchvision import transforms

# def get_data_loaders(train_csv, test_csv, batch_size=32):
#     # Define transforms
#     transform = transforms.Compose([
#         transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
#                            std=[0.2675, 0.2565, 0.2761])
#     ])
    
#     train_dataset = CustomCIFAR100Dataset(
#         csv_file=train_csv,
#         # transform=transform,
#         is_train=True
#     )
    
#     test_dataset = CustomCIFAR100Dataset(
#         csv_file=test_csv,
#         # transform=transform,
#         is_train=False
#     )
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     plot_random_images(train_dataset, num_images=10)
#     plot_random_images(test_dataset, num_images=7)
#     return train_loader, test_loader


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import ast
from torchvision import transforms
import random

class CustomCIFAR100Dataset(Dataset):
    def __init__(self, csv_file, transform=None, is_train=True):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_train (bool): Whether this is training data (with targets) or test data.
        """ ##chatgpt to thank for this, some bug with doccumentation one loader (couldnt figure out what)
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Convert string representation of numpy array back to array
        image_str = self.data_frame.iloc[idx]['image']
        image_array = np.array(ast.literal_eval(image_str), dtype=np.uint8)
        
        # Convert to torch tensor and ensure correct format (C, H, W)
        image = torch.from_numpy(image_array).float() / 255.0  #normalization [0, 1]
        if image.shape[-1] == 3:  #from (H, W, C) to (C, H, W)
            image = image.permute(2, 0, 1)
            
        if self.transform:
            image = self.transform(image)
            
        if self.is_train:
            target = self.data_frame.iloc[idx]['TARGET']
            return image, target
        else:
            return image, self.data_frame.iloc[idx]['ID']

def get_data_loaders(train_csv, test_csv, batch_size=32):
    #data augmentation transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),          # Flip images horizontally
        transforms.RandomRotation(15),              # Rotate by ±15 degrees
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Random crop with resizing
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color augmentation
        # transforms.GaussianBlur(kernel_size=(3, 3)),  # Gaussian blur
        transforms.RandomAffine(degrees=0, shear=10), # Shearing
        # transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    test_transform = transforms.Compose([
        # transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    train_dataset = CustomCIFAR100Dataset(
        csv_file=train_csv,
        transform=train_transform,
        is_train=True
    )
    
    test_dataset = CustomCIFAR100Dataset(
        csv_file=test_csv,
        transform=test_transform,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    plot_random_images(train_dataset, num_images=10)
    plot_random_images(test_dataset, num_images=7)

    return train_loader, test_loader


train_loader, test_loader = get_data_loaders('/kaggle/input/pixel-odyssey-saidl/cifar100/train.csv', '/kaggle/input/pixel-odyssey-saidl/cifar100/test.csv', batch_size=32)


# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)



# class MLPClassifier(nn.Module):
#     def __init__(self):
#         super(MLPClassifier, self).__init__()
#         self.model = nn.Sequential(
#             nn.Flatten(),  # Flatten the 32x32x3 input to a vector of size 3072
#             nn.Linear(32*32*3, 8000),
#             nn.ReLU(),
#             nn.Linear(8000, 6000),
#             nn.ReLU(),
#             nn.Linear(6000, 4000),
#             nn.ReLU(),
#             nn.Linear(4000, 6000),
#             nn.ReLU(),
#             nn.Linear(6000, 4000),
#             nn.ReLU(),
#             nn.Linear(4000, 100)  # CIFAR-100 has 100 output classes
#         )
    
#     def forward(self, x):
#         return self.model(x)

# model = MLPClassifier().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# class SpatialMLP(nn.Module):
# #    def __init__(self, input_size=32*32*3, hidden_sizes=[2048, 1584, 768], num_classes=100):
#     def __init__(self, input_size=32*32*3, hidden_sizes=[2560, 1980, 960, 768], num_classes=100):

#         super(SpatialMLP, self).__init__()
        
#         self.spatial_attention = nn.Sequential(
#             nn.Linear(input_size, input_size),
#             nn.LayerNorm(input_size),
#             nn.GELU(),
#             nn.Linear(input_size, input_size),
#             nn.Sigmoid()
#         )
        
#         layers = []
#         prev_size = input_size
        
#         for hidden_size in hidden_sizes:
#             layers.extend([
#                 nn.Linear(prev_size, hidden_size),
#                 nn.LayerNorm(hidden_size),
#                 nn.GELU(),
#                 nn.Dropout(0.3)
#             ])
#             prev_size = hidden_size
            
#         self.feature_extractor = nn.Sequential(*layers)
        
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
#             nn.LayerNorm(hidden_sizes[-1] // 2),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_sizes[-1] // 2, num_classes)
#         )
        
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = x.view(batch_size, -1)
        
#         attention = self.spatial_attention(x)
#         x = x * attention
        
#         features = self.feature_extractor(x)
        
#         output = self.classifier(features)
#         return output

# model = SpatialMLP().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


!pip install torchdiffeq


# from torchdiffeq import odeint

# class ODEFunc(nn.Module):
#     def __init__(self, input_dim):
#         super(ODEFunc, self).__init__()
#         self.linear = nn.Linear(input_dim, input_dim)
#         self.activation = nn.Tanh()

#     def forward(self, t, x):
#         return self.activation(self.linear(x))

# class ODEBlock(nn.Module):
#     def __init__(self, odefunc):
#         super(ODEBlock, self).__init__()
#         self.odefunc = odefunc

#     def forward(self, x):
#         t = torch.tensor([0, 1], dtype=torch.float32).to(x.device)
#         out = odeint(self.odefunc, x, t, method='rk4') 
#         return out[1]

# class NeuralODEClassifier(nn.Module):
#     def __init__(self, input_dim=3072, hidden_dim=512, output_dim=100):
#         super(NeuralODEClassifier, self).__init__()
#         self.input_layer = nn.Linear(input_dim, hidden_dim)
#         self.odeblock = ODEBlock(ODEFunc(hidden_dim))
#         self.output_layer = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  #(32x32x3 -> 3072)
#         x = torch.relu(self.input_layer(x))
#         x = self.odeblock(x)
#         x = self.output_layer(x)
#         return x

# model = NeuralODEClassifier().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# from torchdiffeq import odeint
# import torch.nn as nn
# import torch

# class LightODEFunc(nn.Module):
#     def __init__(self, hidden_dim):
#         super(LightODEFunc, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(hidden_dim, int(hidden_dim * 1.5)),
#             nn.BatchNorm1d(int(hidden_dim * 1.5)),
#             nn.ReLU(),
#             nn.Linear(int(hidden_dim * 1.5), hidden_dim)
#         )
        
#         self.time_net = nn.Linear(hidden_dim, hidden_dim)
        
#     def forward(self, t, x):
#         main_out = self.net(x)
#         time_out = torch.relu(self.time_net(x))
#         return main_out + x + time_out * t

# class ODEBlock(nn.Module):
#     def __init__(self, odefunc):
#         super(ODEBlock, self).__init__()
#         self.odefunc = odefunc
#         self.integration_times = torch.tensor([0.0, 1.0])
        
#     def forward(self, x):
#         out = odeint(self.odefunc, x, self.integration_times, method='rk4')
#         return out[-1]

# class LightNeuralODEClassifier(nn.Module):
#     def __init__(self, input_dim=3072, hidden_dim=256, output_dim=100):
#         super(LightNeuralODEClassifier, self).__init__()
        
#         self.input_layers = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU()
#         )
        
#         self.ode_block = ODEBlock(LightODEFunc(hidden_dim))
        
#         self.output_layers = nn.Sequential(
#             nn.Linear(hidden_dim, output_dim)
#         )
        
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.input_layers(x)
#         x = self.ode_block(x)
#         return self.output_layers(x)

# model = LightNeuralODEClassifier().to(device)
# criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.AdamW(model.parameters(), 
#                             lr=0.001, 
#                             weight_decay=0.001)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, 
#     mode='min',
#     patience=3,
#     factor=0.5
# )


# from torchdiffeq import odeint
# import torch.nn as nn
# import torch

# class BIGODEFunc(nn.Module):
#     def __init__(self, hidden_dim):
#         super(BIGODEFunc, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.BatchNorm1d(hidden_dim * 2),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim * 2, hidden_dim * 2),
#             nn.BatchNorm1d(hidden_dim * 2),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
        
#         #new try i did:
#         # aditional network for normalizing flows
#         self.time_net = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
        
#     def forward(self, t, x):
#         #combingin main network with time-dependent flow
#         #this helps in learning more expressive transformations - normalizing flows type concept
#         main_out = self.net(x)
#         time_out = self.time_net(x)
        
#         ##residual connection + time-weighted flow
#         return main_out + x + time_out * t

# class ODEBlock(nn.Module):
#     def __init__(self, odefunc, solver_method='dopri5'):
#         super(ODEBlock, self).__init__()
#         self.odefunc = odefunc
#         self.solver_method = solver_method
#         #multiple integration times for better trajectory learning
#         self.integration_times = torch.tensor([0.0, 0.5, 1.0])
        
#     def forward(self, x):
#         out = odeint(self.odefunc, x, self.integration_times, method=self.solver_method)
#         return out[-1]  # Return final state

# class BIGNeuralODEClassifier(nn.Module):
#     def __init__(self, input_dim=3072, hidden_dim=256, output_dim=100):
#         super(BIGNeuralODEClassifier, self).__init__()
        
#         #input preprocessing
#         self.input_layers = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim * 2),
#             nn.BatchNorm1d(hidden_dim * 2),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
        
#         #multiple ODE blocks with different solvers
#         #first block uses dopri5 (adaptive) for precise initial dynamics
#         #second block uses rk4 (fixed) for stability
#         self.ode_blocks = nn.ModuleList([
#             # ODEBlock(BIGODEFunc(hidden_dim), solver_method='dopri5'),
#             ODEBlock(BIGODEFunc(hidden_dim), solver_method='rk4')
#         ])
        
#         #output layers with skip connection from input
#         self.output_layers = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim * 2),  # *2 for concatenated skip connection
#             nn.BatchNorm1d(hidden_dim * 2),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim * 2, output_dim)
#         )
        
#         #project input for skip connection
#         self.skip_proj = nn.Linear(hidden_dim, hidden_dim)
        
#     def forward(self, x):
#         #flatten input image
#         x = x.view(x.size(0), -1)
        
#         #initial processing
#         x = self.input_layers(x)
        
#         #store for skip connection
#         skip = self.skip_proj(x)
        
#         #process through ODE blocks
#         for ode_block in self.ode_blocks:
#             x = ode_block(x)
            
#         #combine with skip connection
#         x = torch.cat([x, skip], dim=1)
        
#         return self.output_layers(x)

# model = BIGNeuralODEClassifier().to(device)
# criterion = nn.CrossEntropyLoss()

# ###AdamW with weight decay for better regularization
# optimizer = torch.optim.AdamW(model.parameters(), 
#                             lr=0.001, 
#                             weight_decay=0.01)

# ###reduce LR when loss plateaus
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, 
#     mode='min',
#     patience=5,
#     factor=0.9
# )



# ###A bigger (more param) resmlpmixer
# class ResMLPBlock(nn.Module):
#     def __init__(self, dim, mlp_hidden_dim):
#         super(ResMLPBlock, self).__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.mlp1 = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Linear(mlp_hidden_dim, dim)
#         )
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp2 = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Linear(mlp_hidden_dim, dim)
#         )

#     def forward(self, x):
#         out = self.mlp1(self.norm1(x)) + x  #1st residual connection
#         out = self.mlp2(self.norm2(out)) + out  #2nd residual connection
#         return out

# class ResMLPWithMixer(nn.Module):
#     def __init__(self, num_classes=100, patch_size=2, img_dim=32, embed_dim=512, mlp_hidden_dim=768, num_blocks=8):
#         super(ResMLPWithMixer, self).__init__()
#         num_patches = (img_dim // patch_size) ** 2
#         self.patch_embedding = nn.Linear(patch_size * patch_size * 3, embed_dim)
#         self.mixer_blocks = nn.Sequential(
#             *[ResMLPBlock(embed_dim, mlp_hidden_dim) for _ in range(num_blocks)]
#         )
#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = nn.Linear(num_patches * embed_dim, num_classes)

#     def forward(self, x):
#         batch_size = x.size(0)
#         patches = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, -1, 2 * 2 * 3)
#         x = self.patch_embedding(patches)
#         x = self.mixer_blocks(x)
#         x = self.norm(x)
#         x = x.flatten(1)
#         return self.head(x)


# model = ResMLPWithMixer().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)



class ResMLPBlock(nn.Module):
    def __init__(self, dim, mlp_hidden_dim):
        super(ResMLPBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mlp1 = nn.Sequential(nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = nn.Sequential(nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim))

    def forward(self, x):
        out = self.mlp1(self.norm1(x)) + x  
        out = self.mlp2(self.norm2(out)) + out
        return out

class ResMLPWithMixer(nn.Module):
    def __init__(self, num_classes=100, patch_size=4, img_dim=32, embed_dim=256, mlp_hidden_dim=512, num_blocks=6):
        super(ResMLPWithMixer, self).__init__()
        num_patches = (img_dim // patch_size) ** 2
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, embed_dim)
        
        self.mixer_blocks = nn.Sequential(*[ResMLPBlock(embed_dim, mlp_hidden_dim) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(num_patches * embed_dim, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        patches = x.unfold(2, 4, 4).unfold(3, 4, 4).contiguous().view(batch_size, -1, 4 * 4 * 3)
        x = self.patch_embedding(patches)
        x = self.mixer_blocks(x)
        x = self.norm(x)
        x = x.flatten(1)
        return self.head(x)

model = ResMLPWithMixer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


print(model)
print(device)


from tqdm import tqdm

epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        ##gradient clipping 
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        ## only use this in case of Light/BIG_NeuralODE 
        
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")




# # Testing loop
# model.eval()
# correct, total = 0, 0
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# print(f"Test Accuracy: {accuracy:.2f}%")



import csv

# to calculate on the training dataset
##completely useless as it takes time to evaluate on the entire train dataset, just not needed
# model.eval()
# correct, total = 0, 0
# with torch.no_grad():
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# train_accuracy = 100 * correct / total
# print(f"Train Accuracy: {train_accuracy:.2f}%")

##
#
# performing the final predictions on the given test dataset
submission = []
model.eval()
with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for id, pred in zip(ids, predicted):
            submission.append([id.item(), pred.item()])

#submission.csv part
with open('/kaggle/working/submission.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'TARGET'])
    writer.writerows(submission)

print("Submission file created: submission.csv")

