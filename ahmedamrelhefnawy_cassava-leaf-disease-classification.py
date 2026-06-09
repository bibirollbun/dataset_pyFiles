import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torchvision.io.image import decode_image, ImageReadMode
from torch.utils.data import DataLoader, Dataset, random_split

import matplotlib.pyplot as plt
import numpy as np
import copy
import torchvision
from collections import defaultdict
import random
import PIL

from tqdm import tqdm
from copy import deepcopy
import gc


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


import os


DATA_DIR = '/kaggle/input/cassava-leaf-disease-classification'
TRAIN_DIR = os.path.join(DATA_DIR, 'train_images')


import json
mapping_path = os.path.join(DATA_DIR, 'label_num_to_disease_map.json')
with open(mapping_path, 'r') as f:
    idx_to_class = json.load(f)

idx_to_class


import pandas as pd
mapping_csv = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
mapping_csv.head()


test_img = decode_image(os.path.join(TRAIN_DIR, '1000015157.jpg'), mode=ImageReadMode.RGB)
img_shape = test_img.shape
print("Single image shape:", img_shape)


class CustomDataset(Dataset):
    def __init__(self, root_dir, img_label_df, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                self.image_files.append(os.path.join(root, file))
                label = img_label_df[img_label_df['image_id'] == file]['label'].iloc[0]
                self.labels.append(label)
 
    def __len__(self):
        return len(self.image_files)
 
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = PIL.Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.Tensor(image)

        label = self.labels[idx]
        ohe_label = torch.eye(5)[label]
        return image, ohe_label


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.0625, 0.0625), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=mean, std=std),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=mean, std=std),
])


dataset = CustomDataset(TRAIN_DIR, mapping_csv)

train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(dataset, (train_size, val_size, test_size))


train_data.dataset.transform = train_transform
val_data.dataset.transform = val_transform
test_data.dataset.transform = val_transform


train_loader = DataLoader(train_data, batch_size=64, num_workers=4)
val_loader = DataLoader(val_data, batch_size=64, num_workers=4)
test_loader = DataLoader(test_data, batch_size=64, num_workers=4)


def visualise_images(dataset, grid, mapping):
    """
    Displays a grid of images from a dataset, with one random image per class.

    Args:
        dataset: The dataset object containing the images and labels.
        grid (tuple): A tuple specifying the number of rows and columns for the image grid.
    """

    # Create a shallow copy of the dataset to avoid modifying the original
    dataset_copy = copy.copy(dataset)
    # Set the transform on the copied dataset to convert images to tensors
    dataset_copy.transform = torchvision.transforms.ToTensor()

    # Create a DataLoader to handle batching and shuffling of the data
    loader = DataLoader(dataset_copy, batch_size=64, shuffle=True)

    # Unpack the grid dimensions from the input tuple
    rows, cols = grid
    # Calculate the total number of images to display in the grid
    num_images_to_show = rows * cols

    # Get the dataset object from the DataLoader
    dataset_to_show = loader.dataset

    # Create a dictionary to store lists of indices for each class
    class_indices = defaultdict(list)
    # Iterate through the dataset to populate the class_indices dictionary
    for idx, target in enumerate(dataset_to_show.labels):
        class_indices[target].append(idx)
        
    # Get the list of class names from the dataset
    class_names = list(mapping.values())
    # Create a figure and a set of subplots for the grid layout
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Iterate over each subplot in the grid
    for i, ax in enumerate(axes.flat):
        # If the current index is out of bounds, turn off the subplot axis
        if i >= num_images_to_show or i >= len(class_names):
            ax.axis('off')
            continue
            
        # Set the class label based on the current iteration index
        class_label = i
        # Get the list of image indices for the current class
        indices_for_class = class_indices[class_label]
        # If there are no images for this class, turn off the subplot axis
        if not indices_for_class:
            ax.axis('off')
            continue

        # Choose a random image index from the list for the current class
        random_image_index = random.choice(indices_for_class)
        
        # Retrieve the image tensor and its corresponding label from the dataset
        image_tensor, _ = dataset_to_show[random_image_index]
        
        # Convert the tensor to a NumPy array and transpose dimensions for display
        img_to_display = image_tensor.numpy().transpose((1, 2, 0))
        
        # Get the name of the class corresponding to the class label
        class_name = class_names[class_label]
        
        # Display the image on the current subplot
        ax.imshow(img_to_display)
        
        # Set the title of the subplot to the capitalized class name
        ax.set_title(class_name.capitalize(), fontsize=16)
        # Turn off the axis for a cleaner look
        ax.axis('off')

    # Adjust subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Display the plot
    plt.show()

    # Clean up the copied dataset to free up memory
    del dataset_copy


visualise_images(train_data.dataset, grid=(1, 5), mapping=idx_to_class)


from torchinfo import summary


next(iter(train_loader))[0].shape


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),  
        )
        
        self.flat = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(265216, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, num_classes),
        )
        
    def forward(self, x):
        x = self.cnn(x)        
        x = self.flat(x)
        x = self.classifier(x)
        
        return x

summary(CustomCNN(5), input_size= [2, 3, 224, 298])


custom_model = CustomCNN(5)
custom_model.to(device)
print()


mn_model = models.mobilenet_v2(weights='DEFAULT')
mn_model.classifier[1] = nn.Linear(in_features= 1280, out_features=5)
mn_model.to(device)
print()


import timm

hf_model = timm.create_model('tf_efficientnet_b3.ns_jft_in1k', pretrained=True)
n_features = hf_model.classifier.in_features
hf_model.classifier = nn.Linear(n_features, 5)
hf_model.to(device)
print()


pt_model = models.efficientnet_b3(weights='DEFAULT')
pt_model.classifier[1] = nn.Linear(1536, 5)
pt_model.to(device)
print()


arch_model = models.efficientnet_b3()
arch_model.classifier[1] = nn.Linear(1536, 5)
arch_model.to(device)
print()


def train_one_epoch(model, train_loader, optimizer, loss_func, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader)
    for i, (images, labels) in enumerate(pbar, 1):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_description(f"Train Loss: {running_loss / (i * train_loader.batch_size):.4f}")
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate_one_epoch(model, val_loader, loss_func, device):
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            val_loss = loss_func(outputs, labels)
            running_val_loss += val_loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, dim=1)[1]).sum().item()
    
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_accuracy = 100.0 * correct / total
    
    return epoch_val_loss, epoch_accuracy


def update_best_model(current_model, best_model, current_val_loss, least_val_loss):
    if best_model is None or current_val_loss < least_val_loss:
        if best_model is not None:
            del best_model
            gc.collect()
            torch.cuda.empty_cache()
        
        best_model = deepcopy(current_model)
        least_val_loss = current_val_loss
    
    return best_model, least_val_loss


def train_model(model, optimizer, loss_func, num_epochs, train_loader, val_loader, verbose=True):
    best_model = None
    least_val_loss = float('inf')

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        epoch_loss = train_one_epoch(model, train_loader, optimizer, loss_func, device)
        train_losses.append(epoch_loss)
        
        # Validation phase
        epoch_val_loss, epoch_accuracy = validate_one_epoch(model, val_loader, loss_func, device)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)
        
        # Update best model
        best_model, least_val_loss = update_best_model(model, best_model, epoch_val_loss, least_val_loss)
        
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%")
    
    metrics = [train_losses, val_losses, val_accuracies]
    return model, best_model, metrics


loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(custom_model.parameters(), lr= 1e-4)
custom_model, best_custom_model, custom_model_metrics = train_model(custom_model, optimizer, loss_func, 10, train_loader, val_loader)

val_loss, val_acc = validate_one_epoch(best_custom_model, val_loader, loss_func, device)
print(f"Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.4f}")

loss, acc = validate_one_epoch(best_custom_model, test_loader, loss_func, device)
print(f"Test Accuracy: {acc:.2f}%, Test Loss: {loss:.4f}")

torch.save(best_custom_model, 'custom_cnn.torch')


loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(mn_model.parameters(), lr= 1e-4)
mn_model, best_mn_model, mn_model_metrics = train_model(mn_model, optimizer, loss_func, 10, train_loader, val_loader)

val_loss, val_acc = validate_one_epoch(best_mn_model, val_loader, loss_func, device)
print(f"Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.4f}")

loss, acc = validate_one_epoch(best_mn_model, test_loader, loss_func, device)
print(f"Test Accuracy: {acc:.2f}%, Test Loss: {loss:.4f}")

torch.save(best_mn_model, 'mobilenetv2.torch')


loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(hf_model.parameters(), lr=1e-4)
hf_model, best_hf_model, hf_model_metrics = train_model(hf_model, optimizer, loss_func, 10, train_loader, val_loader)

val_loss, val_acc = validate_one_epoch(best_hf_model, val_loader, loss_func, device)
print(f"Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.4f}")

loss, acc = validate_one_epoch(best_hf_model, test_loader, loss_func, device)
print(f"Test Accuracy: {acc:.2f}%, Test Loss: {loss:.4f}")

torch.save(best_hf_model, 'hf_EfficientNet.torch')


loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(pt_model.parameters(), lr=1e-4)
pt_model, best_pt_model, pt_model_metrics = train_model(pt_model, optimizer, loss_func, 10, train_loader, val_loader)
loss, acc = validate_one_epoch(best_pt_model, test_loader, loss_func, device)

val_loss, val_acc = validate_one_epoch(best_pt_model, val_loader, loss_func, device)
print(f"Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.4f}")

loss, acc = validate_one_epoch(best_pt_model, test_loader, loss_func, device)
print(f"Test Accuracy: {acc:.2f}%, Test Loss: {loss:.4f}")

torch.save(best_pt_model, 'pt_efficientnet.torch')


loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(arch_model.parameters(), lr=1e-4)
arch_model, best_arch_model, arch_model_metrics = train_model(arch_model, optimizer, loss_func, 10, train_loader, test_loader)
loss, acc = validate_one_epoch(best_arch_model, test_loader, loss_func, device)

val_loss, val_acc = validate_one_epoch(best_arch_model, val_loader, loss_func, device)
print(f"Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.4f}")

loss, acc = validate_one_epoch(best_arch_model, test_loader, loss_func, device)
print(f"Test Accuracy: {acc:.2f}%, Test Loss: {loss:.4f}")

torch.save(best_arch_model, 'arch_efficientnet.torch')


TEST_DIR= os.path.join(DATA_DIR, 'test_images')
submission_df = pd.DataFrame(columns=['image_id', 'label'])

image_files = []
for root, dirs, files in os.walk(TEST_DIR):
    for file in files:
        image =  PIL.Image.open(os.path.join(TEST_DIR, file))
        image = val_transform(image)
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            proba = best_pt_model(image)
            pred = proba.max(1)[1]
            submission_df.loc[len(submission_df)] = [file, pred.item()]

submission_df.to_csv('submission.csv')

