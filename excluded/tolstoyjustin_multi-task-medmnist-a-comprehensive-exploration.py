# Cell 1: Install and Import Dependencies

!pip install --quiet medmnist

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import medmnist
from medmnist import INFO
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Kaggle Notebook settings
%matplotlib inline
plt.rcParams["figure.figsize"] = (8, 5)
sns.set_style("whitegrid")

print("Dependencies installed and libraries imported successfully.")



# Cell 2: Patch the medmnist.INFO to match local .npz file sizes

# Your local organ data has the following sizes (checked by np.load):
# organamnist   -> train: 34,581, val: 6,491,  test: 17,778
# organcmnist   -> train: 13,000, val: 2,392,  test: 8,268
# organsmnist   -> train: 13,940, val: 2,452,  test: 8,829

INFO["organamnist"]["n_samples"] = {"train": 34581, "val": 6491, "test": 17778}
INFO["organcmnist"]["n_samples"] = {"train": 13000, "val": 2392, "test": 8268}
INFO["organsmnist"]["n_samples"] = {"train": 13940, "val": 2452, "test": 8829}

print("Patched medmnist.INFO for organamnist, organcmnist, and organsmnist.")



# Cell 3: Define Utility Functions

import random

def compute_dataset_stats(dataset):
    """
    Compute mean and standard deviation of pixel intensities
    in a given MedMNIST dataset.
    Returns (mean, std) as PyTorch tensors.
    """
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    channel_sum = 0.0
    channel_sq_sum = 0.0
    num_samples = 0
    
    for data, _ in loader:
        data = data.float()
        batch_size = data.size(0)
        # data shape: [batch_size, channels, height, width]
        
        channel_sum += data.sum(dim=[0, 2, 3])
        channel_sq_sum += (data ** 2).sum(dim=[0, 2, 3])
        num_samples += batch_size * data.shape[2] * data.shape[3]
    
    mean = channel_sum / num_samples
    std = torch.sqrt((channel_sq_sum / num_samples) - (mean ** 2))
    return mean, std


def plot_class_distribution(labels, dataset_name, class_labels=None):
    """
    Plot a bar chart for class distribution.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=labels)
    plt.title(f"Class Distribution - {dataset_name}")
    plt.xlabel("Class Index")
    plt.ylabel("Count")
    
    if class_labels is not None and len(class_labels) == len(set(labels)):
        plt.xticks(range(len(class_labels)), class_labels, rotation=45)
    plt.tight_layout()
    plt.show()


def show_random_samples(dataset, dataset_name, n_samples=5):
    """
    Show random samples from the dataset to visualize image quality.
    """
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    fig, axes = plt.subplots(1, n_samples, figsize=(3 * n_samples, 3))
    fig.suptitle(f"Random Samples - {dataset_name}", fontsize=14)
    
    if n_samples == 1:
        axes = [axes]
        
    for ax, idx in zip(axes, indices):
        img, label = dataset[idx]
        np_img = img.numpy().transpose(1, 2, 0)  # shape => (H, W, C)
        if np_img.shape[2] == 1:
            np_img = np_img.squeeze(-1)
        
        ax.imshow(np_img, cmap='gray' if np_img.ndim == 2 else None)
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

print("Utility functions for plotting and statistics are defined.")



# Cell 4: Define Datasets and Perform Exploration

datasets_to_explore = [
    "bloodmnist",
    "breastmnist",
    "dermamnist",
    "octmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "pathmnist",
    "pneumoniamnist",
    "retinamnist",
    "tissuemnist"
]

# Basic transform: convert to Tensor
transform = transforms.Compose([transforms.ToTensor()])

# Path to data
data_root = "/kaggle/input/tensor-reloaded-multi-task-med-mnist/data"

# Dictionary to store dataset exploration results
dataset_summaries = {}

for dset_name in datasets_to_explore:
    print(f"===================== {dset_name.upper()} =====================")
    try:
        info = INFO[dset_name]
        task_type = info['task']
        n_channels = info['n_channels']
        class_labels = info['label']
        n_classes = len(class_labels)
        
        print(f"Task Type       : {task_type}")
        print(f"Num. Channels   : {n_channels}")
        print(f"Num. Classes    : {n_classes}")
        print(f"Class Labels    : {class_labels}")

        # Dynamically get dataset class
        DatasetClass = getattr(medmnist, info['python_class'])

        # Load train, val, test splits
        train_dataset = DatasetClass(root=data_root, split='train', transform=transform, download=False)
        val_dataset   = DatasetClass(root=data_root, split='val',   transform=transform, download=False)
        test_dataset  = DatasetClass(root=data_root, split='test',  transform=transform, download=False)

        # Print dataset sizes
        print(f"Train samples   : {len(train_dataset)}")
        print(f"Val samples     : {len(val_dataset)}")
        print(f"Test samples    : {len(test_dataset)}")
        
        # --- A) Class Distribution (for classification tasks) ---
        train_labels = train_dataset.labels.squeeze()
        if train_labels.ndim > 1:  
            # Flatten if labels have shape [N,1]
            train_labels = train_labels[:, 0]
        
        # For multi-label or ordinal tasks, you might skip or revise the plot
        # Here we do it for simpler classification tasks
        plot_class_distribution(train_labels, dset_name, class_labels=class_labels)

        # --- B) Visual Exploration: Random Images ---
        show_random_samples(train_dataset, dset_name, n_samples=5)

        # --- C) Statistical Analysis: Mean and Std ---
        mean, std = compute_dataset_stats(train_dataset)
        print(f"Pixel Intensity Mean (per channel) : {mean}")
        print(f"Pixel Intensity STD  (per channel) : {std}")

        # Store details for final summary
        dataset_summaries[dset_name] = {
            "task_type": task_type,
            "num_channels": n_channels,
            "num_classes": n_classes,
            "class_labels": class_labels,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "mean": mean.tolist(),
            "std": std.tolist()
        }

    except AssertionError as e:
        print(f"[ERROR] Skipping '{dset_name}' due to AssertionError:\n{e}")
    
    print("--------------------------------------------------\n")

print("Data exploration loop completed.")



# Cell 5: Summaries and Conclusions

print("===== FINAL DATASET SUMMARIES =====\n")
for dataset_name, details in dataset_summaries.items():
    print(f"Dataset: {dataset_name}")
    for key, val in details.items():
        print(f"  {key}: {val}")
    print()
print("Exploration complete.")



# Cell 5: Create Dataloaders for All Datasets

def get_all_dataloaders(dataset_names, batch_size=64, use_weighted_sampler=False):
    """
    Returns a dict of (train_loader, val_loader, test_loader) for each dataset in dataset_names.
    """
    loaders_dict = {}
    
    for dset_name in dataset_names:
        print(f"Creating loaders for: {dset_name}...")
        DatasetClass = getattr(medmnist, medmnist.INFO[dset_name]['python_class'])
        
        # Transforms
        train_transform = get_transforms(dset_name, is_train=True)
        test_transform  = get_transforms(dset_name, is_train=False)
        
        # Datasets
        train_dataset = DatasetClass(split='train', transform=train_transform, download=False,
                                     root="/kaggle/input/tensor-reloaded-multi-task-med-mnist/data")
        val_dataset   = DatasetClass(split='val', transform=test_transform, download=False,
                                     root="/kaggle/input/tensor-reloaded-multi-task-med-mnist/data")
        test_dataset  = DatasetClass(split='test', transform=test_transform, download=False,
                                     root="/kaggle/input/tensor-reloaded-multi-task-med-mnist/data")
        
        # Dataloaders
        train_loader = create_dataloader(train_dataset, batch_size, use_weighted_sampler)
        val_loader   = create_dataloader(val_dataset,   batch_size, use_weighted_sampler=False)
        test_loader  = create_dataloader(test_dataset,  batch_size, use_weighted_sampler=False)
        
        loaders_dict[dset_name] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
    
    return loaders_dict

# Example usage
datasets_to_process = [
    "bloodmnist",
    "breastmnist",
    "dermamnist",
    "octmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "pathmnist",
    "pneumoniamnist",
    "retinamnist",
    "tissuemnist"
]

all_loaders = get_all_dataloaders(datasets_to_process, batch_size=64, use_weighted_sampler=False)
print("All dataloaders created successfully.")





