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
    transforms. ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms. ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = "/kaggle/input/mo-i-competition-mnis-tx-cifar/MNISTxCIFAR/train"
test_dir = "/kaggle/input/mo-i-competition-mnis-tx-cifar/MNISTxCIFAR/test"

# Load the training data
full_train_dataset = MNISTxCIFAR(train_dir, transform=train_transform) # Create the full training dataset

# Split the full training dataset into training and validation sets
# Let's say we want to use 80% of the samples for training and 20% for validation

train_size = int(0.8 * len(full_train_dataset))  # 80% of the dataset size
val_size = len(full_train_dataset) - train_size  # Remaining 20%

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create data loaders 
batch_size = 32  # Definning the batch size

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle during training
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation data

# Load the test data
test_dataset = MNISTxCIFAR(test_dir, train=False, transform=test_transform)  # Set train=False for test dataset
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle test data


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



# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models
# from torch.optim import lr_scheduler
# import torch.nn.functional as F
# from tqdm import tqdm

# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
        
#         self.conv1 = nn.Conv2d(3, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
#         self.pool1 = nn.MaxPool2d(2, 2) # Max pooling over a (2, 2) window
#         self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 square convolution kernel
#         self.pool2 = nn.MaxPool2d(2, 2) # Max pooling over a (2, 2) window
        
#         # TODO: Construct your linear layers
#         self.fc1 = nn.LazyLinear(120) # an affine operation: y = Wx + b, 120 output nodes
#         self.fc2 = nn.Linear(120,84) # 84 output nodes
#         self.fc3 = nn.Linear(84,10) # 10 output nodes for the 10 classes

#     def forward(self, x):
#         bs = x.shape[0]
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = x.view(bs, -1) # reshape tensor
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # model instance
# model = LeNet()

# # Move model to GPU if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Loss function and optimizer
# # TODO: Select a criterion and fill in the learning rate and momentum values
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# # TODO: Implement an exponential learning rate scheduler
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# # Training loop
# # TODO: Select your training epochs

# num_epochs = 20
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}/{num_epochs}")

#     # Each epoch has a training and validation phase
#     for phase in ['train', 'val']:
#         if phase == 'train':
#             model.train()  # Set model to training mode
#             dataloader = train_dataloader
#         else:
#             model.eval()   # Set model to evaluate mode
#             dataloader = val_dataloader

#         running_loss = 0.0
#         running_corrects = 0

#         # Iterate over data.
#         for inputs, labels in tqdm(dataloader):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Forward
#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)

#                 # Backward + optimize only if in training phase
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()

#             # Statistics
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         if phase == 'train':
#             exp_lr_scheduler.step()

#         epoch_loss = running_loss / len(dataloader.dataset)
#         epoch_acc = running_corrects.double() / len(dataloader.dataset)

#         print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# print('Training complete')


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.optim import lr_scheduler
# from tqdm import tqdm
# from torchvision import models

# # Swish activation function
# def swish(x):
#     return x * torch.sigmoid(x)

# # Squeeze-and-Excitation (SE) block
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.fc1 = nn.Linear(channels, channels // reduction)
#         self.fc2 = nn.Linear(channels // reduction, channels)
    
#     def forward(self, x):
#         batch, channels, _, _ = x.size()
#         se = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
#         se = swish(self.fc1(se))
#         se = torch.sigmoid(self.fc2(se)).view(batch, channels, 1, 1)
#         return x * se

# class CNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNN, self).__init__()
        
#         self.block1 = self.conv_block(3, 128, dropout=0.3)
#         self.block2 = self.conv_block(128, 256, dropout=0.25)
#         self.block3 = self.conv_block(256, 512, dropout=0.2)
#         self.block4 = self.conv_block(512, 1024, dropout=0.15)
        
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, num_classes)
    
#     def conv_block(self, in_channels, out_channels, dropout):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU(inplace=True),  # Swish activation
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU(inplace=True),  # Swish activation
#             SEBlock(out_channels),  # Adding Squeeze-and-Excitation block
#             nn.MaxPool2d(2),
#             nn.Dropout(dropout)
#         )
    
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
        
#         x = self.global_pool(x)
#         x = x.view(x.size(0), -1)
        
#         x = F.silu(self.fc1(x))  # Swish activation
#         x = self.fc2(x)
        
#         return x

# # Model setup
# model = CNN(num_classes=10)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Label smoothing in CrossEntropyLoss
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# # Using AdamW Optimizer instead of Lookahead + RAdam
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# # Learning rate scheduler
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# # Training loop
# num_epochs = 30
# batch_size = 32  # Reduced batch size

# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}/{num_epochs}")
    
#     for phase in ['train', 'val']:
#         if phase == 'train':
#             model.train()
#             dataloader = train_dataloader
#         else:
#             model.eval()
#             dataloader = val_dataloader

#         running_loss = 0.0
#         running_corrects = 0
        
#         for inputs, labels in tqdm(dataloader):
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
            
#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)
                
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()
            
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)
        
#         if phase == 'train':
#             scheduler.step()
        
#         epoch_loss = running_loss / len(dataloader.dataset)
#         epoch_acc = running_corrects.double() / len(dataloader.dataset)
        
#         print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# print('Training complete')




import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Define CNN blocks
        self.block1 = self.conv_block(3, 128)      # First block (input channels = 3, output channels = 128)
        self.block2 = self.conv_block(128, 256)    # Second block
        self.block3 = self.conv_block(256, 512)    # Third block
        self.block4 = self.conv_block(512, 1024)   # Fourth block for deeper features

        # Fully connected layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc1 = nn.Linear(1024, 512)                  # First fully connected layer
        self.fc2 = nn.Linear(512, num_classes)           # Final output layer

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        # Pass input through each block
        x = self.block1(x)  # [batch_size, 128, 14, 14] if input is 28x28
        x = self.block2(x)  # [batch_size, 256, 7, 7]
        x = self.block3(x)  # [batch_size, 512, 3, 3]
        x = self.block4(x)  # [batch_size, 1024, 1, 1]

        x = self.global_pool(x)  # [batch_size, 1024, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 1024]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

      

model = CNN(num_classes=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_dataloader
        else:
            model.eval()
            dataloader = val_dataloader

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print('Training complete')


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from tqdm import tqdm

# def conv_block(in_channels, out_channels, pool=False):
#     layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
#               nn.BatchNorm2d(out_channels), 
#               nn.ReLU(inplace=True)]
#     if pool:
#         layers.append(nn.MaxPool2d(2))  # Pooling after each conv block
#     return nn.Sequential(*layers)

# class ResNet9(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(ResNet9, self).__init__()
        
#         self.conv1 = conv_block(in_channels, 64)
#         self.conv2 = conv_block(64, 128, pool=True)
#         self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
#         self.conv3 = conv_block(128, 256)
#         self.conv4 = conv_block(256, 512)
#         self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
#         self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),  # Adaptive average pool to reduce spatial size
#                                         nn.Flatten(), 
#                                         nn.Dropout(0.2),
#                                         nn.Linear(512, num_classes))
        
#     def forward(self, xb):
#         out = self.conv1(xb)
#         out = self.conv2(out)
#         out = self.res1(out) + out
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.res2(out) + out
#         out = self.classifier(out)
#         return out

# # Model instance
# model = ResNet9(in_channels=3, num_classes=10)

# # Move model to GPU if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Learning rate scheduler
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# # Number of training epochs
# num_epochs = 30

# # Training loop
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}/{num_epochs}")

#     for phase in ['train', 'val']:
#         if phase == 'train':
#             model.train()
#             dataloader = train_dataloader
#         else:
#             model.eval()
#             dataloader = val_dataloader

#         running_loss = 0.0
#         running_corrects = 0

#         for inputs, labels in tqdm(dataloader):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()

#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)

#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         if phase == 'train':
#             exp_lr_scheduler.step()

#         epoch_loss = running_loss / len(dataloader.dataset)
#         epoch_acc = running_corrects.double() / len(dataloader.dataset)

#         print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# print('Training complete')


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from tqdm import tqdm

# # Define a Basic Residual Block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
        
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
        
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         self.downsample = downsample  # Used to match dimensions for skip connection

#     def forward(self, x):
#         identity = x  # Store original input
        
#         if self.downsample is not None:
#             identity = self.downsample(x)  # Adjust identity if necessary
        
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
        
#         out = self.conv2(out)
#         out = self.bn2(out)
        
#         out += identity  # Skip Connection (Residual Addition)
#         out = self.relu(out)  # Apply activation after addition
        
#         return out

# # Define ResNet18 Architecture
# class ResNet18(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(ResNet18, self).__init__()
        
#         # Initial Convolution Layer
#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         # Residual Blocks
#         self.layer1 = self.make_layer(64, 64, 2)   # 2 Blocks
#         self.layer2 = self.make_layer(64, 128, 2, stride=2)  # Downsample
#         self.layer3 = self.make_layer(128, 256, 2, stride=2)  # Downsample
#         self.layer4 = self.make_layer(256, 512, 2, stride=2)  # Downsample
        
#         # Adaptive Pooling and Fully Connected Layer
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)

#     def make_layer(self, in_channels, out_channels, blocks, stride=1):
#         """Creates a ResNet layer with residual blocks"""
#         downsample = None
#         if stride != 1 or in_channels != out_channels:
#             downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
        
#         layers = []
#         layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))  # First Block (may need downsampling)
#         for _ in range(1, blocks):
#             layers.append(ResidualBlock(out_channels, out_channels))  # Additional Blocks
        
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         # Initial Conv Block
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         # Residual Blocks
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         # Pooling & Fully Connected Layer
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x

# # Model instance
# model = ResNet18(in_channels=3, num_classes=10)

# # Move model to GPU if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Learning rate scheduler
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# # Number of training epochs
# num_epochs = 20

# # Training loop
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}/{num_epochs}")

#     for phase in ['train', 'val']:
#         if phase == 'train':
#             model.train()
#             dataloader = train_dataloader
#         else:
#             model.eval()
#             dataloader = val_dataloader

#         running_loss = 0.0
#         running_corrects = 0

#         for inputs, labels in tqdm(dataloader):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()

#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)

#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         if phase == 'train':
#             exp_lr_scheduler.step()

#         epoch_loss = running_loss / len(dataloader.dataset)
#         epoch_acc = running_corrects.double() / len(dataloader.dataset)

#         print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# print('Training complete')


# import torch
# import torch.nn as nn

# # Define a Basic Residual Block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
        
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
        
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         self.downsample = downsample  # Used to match dimensions for skip connection

#     def forward(self, x):
#         identity = x  # Store original input
        
#         if self.downsample is not None:
#             identity = self.downsample(x)  # Adjust identity if necessary
        
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
        
#         out = self.conv2(out)
#         out = self.bn2(out)
        
#         out += identity  # Skip Connection (Residual Addition)
#         out = self.relu(out)  # Apply activation after addition
        
#         return out

# # Define ResNet34 Architecture
# class ResNet34(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(ResNet34, self).__init__()
        
#         # Initial Convolution Layer
#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         # Residual Blocks
#         self.layer1 = self.make_layer(64, 64, 3)   # 3 Blocks
#         self.layer2 = self.make_layer(64, 128, 4, stride=2)  # 4 Blocks, Downsample
#         self.layer3 = self.make_layer(128, 256, 6, stride=2)  # 6 Blocks, Downsample
#         self.layer4 = self.make_layer(256, 512, 3, stride=2)  # 3 Blocks, Downsample
        
#         # Adaptive Pooling and Fully Connected Layer
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)

#     def make_layer(self, in_channels, out_channels, blocks, stride=1):
#         """Creates a ResNet layer with residual blocks"""
#         downsample = None
#         if stride != 1 or in_channels != out_channels:
#             downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
        
#         layers = []
#         layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))  # First Block (may need downsampling)
#         for _ in range(1, blocks):
#             layers.append(ResidualBlock(out_channels, out_channels))  # Additional Blocks
        
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         # Initial Conv Block
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         # Residual Blocks
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         # Pooling & Fully Connected Layer
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x

# # Model instance
# model = ResNet34(in_channels=3, num_classes=10)

# # Move model to GPU if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Learning rate scheduler
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# # Number of training epochs
# num_epochs = 20

# # Training loop
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}/{num_epochs}")

#     for phase in ['train', 'val']:
#         if phase == 'train':
#             model.train()
#             dataloader = train_dataloader
#         else:
#             model.eval()
#             dataloader = val_dataloader

#         running_loss = 0.0
#         running_corrects = 0

#         for inputs, labels in tqdm(dataloader):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()

#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)

#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         if phase == 'train':
#             exp_lr_scheduler.step()

#         epoch_loss = running_loss / len(dataloader.dataset)
#         epoch_acc = running_corrects.double() / len(dataloader.dataset)

#         print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# print('Training complete')


# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models
# from torch.optim import lr_scheduler
# import torch.nn.functional as F
# from tqdm import tqdm

# def patchify(images, n_patches):
#     n, c, h, w = images.shape

#     assert h == w, "Patchify method is implemented for square images only"

#     patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
#     patch_size = h // n_patches

#     for idx, image in enumerate(images):
#         for i in range(n_patches):
#             for j in range(n_patches):
#                 patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
#                 patches[idx, i * n_patches + j] = patch.flatten()
#     return patches

# class MyMSA(nn.Module):
#     def __init__(self, d, n_heads=2):
#         super(MyMSA, self).__init__()
#         self.d = d
#         self.n_heads = n_heads

#         assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

#         d_head = int(d / n_heads)
#         self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
#         self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
#         self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
#         self.d_head = d_head
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, sequences):
#         # Sequences has shape (N, seq_length, token_dim)
#         # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
#         # And come back to    (N, seq_length, item_dim)  (through concatenation)
#         result = []
#         for sequence in sequences:
#             seq_result = []
#             for head in range(self.n_heads):
#                 q_mapping = self.q_mappings[head]
#                 k_mapping = self.k_mappings[head]
#                 v_mapping = self.v_mappings[head]

#                 seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
#                 q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

#                 attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
#                 seq_result.append(attention @ v)
#             result.append(torch.hstack(seq_result))
#         return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

# class MyViTBlock(nn.Module):
#     def __init__(self, hidden_d, n_heads, mlp_ratio=4):
#         super(MyViTBlock, self).__init__()
#         self.hidden_d = hidden_d
#         self.n_heads = n_heads

#         self.norm1 = nn.LayerNorm(hidden_d)
#         self.mhsa = MyMSA(hidden_d, n_heads)
#         self.norm2 = nn.LayerNorm(hidden_d)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_d, mlp_ratio * hidden_d),
#             nn.GELU(),
#             nn.Linear(mlp_ratio * hidden_d, hidden_d)
#         )

#     def forward(self, x):
#         out = x + self.mhsa(self.norm1(x))
#         out = out + self.mlp(self.norm2(out))
#         return out

# class MyViT(nn.Module):
#     def __init__(self, chw, n_patches=4, n_blocks=2, hidden_d=32, n_heads=8, out_d=10):
#         # Super constructor
#         super(MyViT, self).__init__()
        
#         # Attributes
#         self.chw = chw # ( C , H , W )
#         self.n_patches = n_patches
#         self.n_blocks = n_blocks
#         self.n_heads = n_heads
#         self.hidden_d = hidden_d
        
#         # Input and patches sizes
#         assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
#         assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
#         self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

#         # 1) Linear mapper
#         self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
#         self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
#         # 2) Learnable classification token
#         self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
#         # 3) Positional embedding
#         self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
        
#         # 4) Transformer encoder blocks
#         self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
#         # 5) Classification MLPk
#         self.mlp = nn.Sequential(
#             nn.Linear(self.hidden_d, out_d),
#             nn.Softmax(dim=-1)
#         )

#     def forward(self, images):
#         # Dividing images into patches
#         n, c, h, w = images.shape
#         patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        
#         # Running linear layer tokenization
#         # Map the vector corresponding to each patch to the hidden size dimension
#         tokens = self.linear_mapper(patches)
        
#         # Adding classification token to the tokens
#         tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
#         # Adding positional embedding
#         out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
#         # Transformer Blocks
#         for block in self.blocks:
#             out = block(out)
            
#         # Getting the classification token only
#         out = out[:, 0]
        
#         return self.mlp(out) # Map to output dimension, output category distribution

# # Model instance
# model = MyViT((3, 28, 28))

# # Move model to GPU if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.001)

# # Learning rate scheduler
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# # Number of training epochs
# num_epochs = 30

# # Training loop
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}/{num_epochs}")

#     for phase in ['train', 'val']:
#         if phase == 'train':
#             model.train()
#             dataloader = train_dataloader
#         else:
#             model.eval()
#             dataloader = val_dataloader

#         running_loss = 0.0
#         running_corrects = 0

#         for inputs, labels in tqdm(dataloader):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()

#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)

#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         if phase == 'train':
#             exp_lr_scheduler.step()

#         epoch_loss = running_loss / len(dataloader.dataset)
#         epoch_acc = running_corrects.double() / len(dataloader.dataset)

#         print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# print('Training complete')


import pandas as pd

# Placeholder for your predictions and image identifiers
predictions = []
image_ids = []

model.eval()
# Iterate over test set
for image in test_dataloader:
    image = image.cuda()
    
    with torch.no_grad():
        # Predict the label using your model (modify this as needed)
        pred = model(image)
    
    # Convert scores to class predictions
    pred = pred.argmax(dim=1)

    # Append to our lists
    predictions.extend(pred.tolist())


indices = list(range(len(predictions)))
indices = ['{:07d}'.format(i) for i in indices]

# Create DataFrame
df = pd.DataFrame({
    'ID': indices,  # convert tensor to list
    'Label': predictions
})

# Save DataFrame as CSV without headers
df.to_csv('submission.csv', index=False)

