# Import required libraries
import numpy as np # numpy is used for handling numerical operations on arrays
import pandas as pd # pandas is used for handling and manipulating structured data
from torchvision.io import read_image # For reading image data from a file
from torch.utils.data import Dataset, DataLoader # For creating custom datasets and dataloaders
from torchvision.transforms import ToTensor 
import torch # PyTorch library for neural network operations
import torchvision # a package consists of popular datasets, model architectures, and common image transformations for computer vision.
from torchvision import transforms # for performing transformations on our dataset
import matplotlib.pyplot as plt # for plotting graphs and visualizing data
from torch.utils.data import random_split # for splitting the dataset into train and validation
import os # useful for operating system operations
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm



# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)



# Define the custom dataset
class MNISTxCIFAR(Dataset):
    def __init__(self, img_dir, transform=None, train=True):
        self.img_dir = img_dir # The directory where our images are stored
        self.transform = transform # The transformations to be applied on the images
        self.labels = [] # List to hold the labels of the images

        if train: # If this is the training set
            self.classes = os.listdir(img_dir) # Get the list of classes (folders) in the directory
            self.classes.sort() # Sort the classes for consistency
            self.img_paths = [] # List to hold the paths of the images
            self.labels = [] # Reset the labels list
            for c in self.classes: # For each class
                c_dir = os.path.join(img_dir, c) # Get the directory of the class
                c_imgs = os.listdir(c_dir) # Get the list of images in the class directory
                for img in c_imgs: # For each image
                    self.img_paths.append(os.path.join(c_dir, img)) # Add the path of the image to the list
                    self.labels.append(int(c)) # Add the class label to the labels list
        else: # If this is the test set
            self.img_paths = [os.path.join(img_dir, img) for img in os.listdir(img_dir)] 
            self.img_paths.sort()
            # No labels are included for the test set

    def __len__(self):
        return len(self.img_paths) # The length of the dataset is the number of images

    def __getitem__(self, idx): # Method to get an item from the dataset
        img_path = self.img_paths[idx] # Get the path of the image
        image = read_image(img_path) # Read the image from the path
        if self.transform: # If there are any transformations to be applied
            image = self.transform(image) # Apply the transformations
        if self.labels: # If there are labels
            return image, self.labels[idx] # Return the image and its label
        else:
            return image  # For the test set, we return just the image


# Training and testing transformations
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=15),   # Rotate digits slightly
    transforms.RandomAffine(degrees=0, shear=5, scale=(0.95, 1.05)),  # Distortions
    transforms.RandomResizedCrop(size=28, scale=(0.9, 1.1)),  # Random zoom
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Simulate lighting changes
    transforms.ToTensor(), #TODO: Convert to tensor
    # transforms.RandomRotation(15),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), #TODO: Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = "/kaggle/input/mo-i-competition-mnis-tx-cifar/MNISTxCIFAR/train"
test_dir = "/kaggle/input/mo-i-competition-mnis-tx-cifar/MNISTxCIFAR/test"

# Load the training data
full_train_dataset = MNISTxCIFAR(train_dir, transform=train_transform) # Create the full training dataset

# Split the full training dataset into training and validation sets
# Let's say we want to use 80% of the samples for training and 20% for validation

# TODO: Split the dataset based on 80-20 split
train_size = int(0.80 * len(full_train_dataset))  # 80% of the dataset size
val_size = len(full_train_dataset) - train_size  # The remaining samples

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create data loaders 
# TODO: Fill in the loader parameters
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True) # Create the training dataloader
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False) # Create the validation dataloader

# Load the test data
# TODO: Fill in the loader parameters
test_dataset = MNISTxCIFAR(test_dir, train=False, transform=test_transform) # Create the test dataset
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False) # Create the test dataloader

# Check out the datasets
for images, labels in train_dataloader: # For each batch in the training dataloader
    print(images.shape, labels.shape) # Print the shapes of the images and labels tensors
    break # Stop after the first batch

for images, labels in val_dataloader: # For each batch in the validation dataloader
    print(images.shape, labels.shape) # Print the shapes of the images and labels tensors
    break # Stop after the first batch

for images in test_dataloader: # For each batch in the test dataloader
    print(images.shape) # Print the shape of the images tensor
    break # Stop after the first batch


def imshow(inp, mean, std, title=None):
    """Imshow for Tensor after reversing normalization."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Reverse the normalization
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


# Create a subplot grid of 10 rows and 6 columns (total 60 subplots)
fig, axs = plt.subplots(10, 6, figsize=(10, 20))  # 10 classes, 6 images each

# Flatten the array of Axes instances for easy iteration
axs = axs.ravel()

# Dictionary to hold images of each class
class_images = {}

# Iterate over batches of images and labels in the training dataloader
for images, labels in train_dataloader:
    # Iterate over individual images and labels in the batch
    for img, label in zip(images, labels):
        label = label.item()  # Convert the label tensor to a Python scalar
        if label not in class_images:  # If this class hasn't been seen before
            class_images[label] = []  # Create a new list for this class
        if len(class_images[label]) < 6:  # If fewer than 6 images have been collected for this class
            class_images[label].append(img)  # Add the current image to the class's list
        if all(len(imgs) >= 6 for imgs in class_images.values()):  # If 6 images have been collected for all classes
            break  # Break out of the loop over images and labels
    else:  # If the loop over images and labels wasn't broken
        continue  # Continue with the next batch
    break  # If the loop over images and labels was broken, break the loop over batches


# Iterate over the collected images of each class
for label, imgs in class_images.items():
    for i, img in enumerate(imgs):  # For each image
        img = img.numpy().transpose((1, 2, 0))  # Convert to numpy array and transpose
        img = std * img + mean  # Reverse normalization
        img = np.clip(img, 0, 1)  # Clip values to range [0, 1]
        axs[label*6 + i].imshow(img)  # Display the image in the appropriate subplot
        axs[label*6 + i].axis('off')  # Turn off the axis
        if i == 0:  # For the first image of each class
            axs[label*6 + i].set_title(f'Class: {label}')  # Set the subplot's title to the class label
plt.show()  # Display the plot


import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Utility: 3x3 convolution helper function ---
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding (bias is False because BatchNorm is used)"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# --- Basic Residual Block ---
class BasicBlock(nn.Module):
    expansion = 1  # For BasicBlock, expansion is 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        inplanes: number of input channels.
        planes: number of output channels.
        stride: stride for the first convolution.
        downsample: Optional downsampling module if the dimensions change.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample  # if not None, will be used to match dimensions
        self.stride = stride

    def forward(self, x):
        residual = x  # save the input for the shortcut connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)  # match dimensions

        out += residual  # add the shortcut
        out = self.relu(out)
        return out


# --- Custom ResNet for Small Images ---
class CustomResNet(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale=False):
        """
        block: block class to use (e.g., BasicBlock)
        layers: list with the number of blocks per layer (e.g., [2, 2, 2, 2] for ResNet-18)
        num_classes: number of output classes
        grayscale: If True, expects single-channel input; otherwise, 3-channel.
        """
        self.inplanes = 32
        in_dim = 1 if grayscale else 3
        super(CustomResNet, self).__init__()
        # Use a smaller initial conv layer for 28x28 images
        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # For small images, we remove the aggressive maxpool or use a mild version.
        # Uncomment the following line if you want some downsampling:
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Create the residual layers
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=2)

        
        # Use Adaptive Average Pooling to get a fixed 1x1 feature map
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Additional fully connected layers with dropout for regularization
        self.fc1 = nn.Linear(256 * block.expansion, 512)
        self.fc2 = nn.Linear(512, num_classes)


        # Weight initialization (Kaiming Normal for conv layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # If you want to use maxpooling, uncomment the next line:
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)  # (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


# --- Convenience Function to Build a Custom ResNet-18 ---
def custom_resnet18(num_classes):
    return CustomResNet(block=BasicBlock, layers=[2,2,2,2,2,2], num_classes=num_classes, grayscale=GRAYSCALE)


# Set global variables
NUM_CLASSES = 10
GRAYSCALE = False

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = custom_resnet18(NUM_CLASSES).to(device)



%%time

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)

# TODO: Implement an exponential learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
# Training loop
# TODO: Select your training epochs
num_epochs = 30





# ================================
# ================================
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Training Phase
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(train_dataloader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass; extract logits for loss calculation
        logits, _ = model(inputs)
        _, preds = torch.max(logits, 1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    train_loss = running_loss / len(train_dataloader.dataset)
    train_acc = running_corrects.double() / len(train_dataloader.dataset)
    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
    
    # Validation Phase
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits, _ = model(inputs)
            _, preds = torch.max(logits, 1)
            loss = criterion(logits, labels)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
    epoch_val_loss = val_loss / len(val_dataloader.dataset)
    epoch_val_acc = val_corrects.double() / len(val_dataloader.dataset)
    print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
    
    scheduler.step()  # Cosine annealing scheduler step
    


print("Training complete.")





import pandas as pd

# Placeholder for your predictions and image identifiers
predictions = []
image_ids = []  # If you have image IDs to track

model.eval()
# Iterate over test set
for images in test_dataloader:
    images = images.cuda()  # Move images to GPU

    with torch.no_grad():
        # Get the logits and probabilities from the model output
        logits, _ = model(images)  # Extract the logits

    # Convert logits to class predictions
    preds = logits.argmax(dim=1)

    # Append to our lists
    predictions.extend(preds.tolist())




indices = list(range(len(predictions)))
indices = ['{:07d}'.format(i) for i in indices]

# Create DataFrame
df = pd.DataFrame({
    'ID': indices,  # convert tensor to list
    'Label': predictions
})

# Save DataFrame as CSV without headers
df.to_csv('submission.csv', index=False)




