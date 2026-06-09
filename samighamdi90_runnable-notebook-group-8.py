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
import torch.nn.functional as F
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import OneCycleLR




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

"""
# Training and testing transformations
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
"""
train_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),  # تحويلها مرة أخرى إلى Tensor بعد Augmentation
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), #TODO: Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dir = "/kaggle/input/mnistxcifar-competition/MNISTxCIFAR/train"
test_dir = "/kaggle/input/mnistxcifar-competition/MNISTxCIFAR/test"


# Load the training data
full_train_dataset = MNISTxCIFAR(train_dir, transform=train_transform) # Create the full training dataset

# Split the full training dataset into training and validation sets
# Let's say we want to use 80% of the samples for training and 20% for validation

# TODO: Split the dataset based on 80-20 split
train_size = int(0.8 * len(full_train_dataset))  # حجم مجموعة التدريب
val_size = len(full_train_dataset) - train_size  # The remaining samples

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_dataset = MNISTxCIFAR(test_dir, train=False, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

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
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import OneCycleLR


# Basic block for ResNet - has two conv layers with a shortcut connection
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # dropout to help prevent overfitting
        self.dropout = nn.Dropout2d(0.05)

        # shortcut - needed when output size changes
        self.shortcut = None
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # save input for shortcut connection
        identity = x if self.shortcut is None else self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        
        # add shortcut connection
        out += identity
        return F.relu(out, inplace=True)


# Complete ResNet architecture
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # input layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # main layers - each unit doubles the channels
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)

        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # initialize weights for all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, out_channels, num_blocks, stride):
        # make list of strides - first one as given, rest are 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # apply cutout during training
        if self.training:
            if torch.rand(1).item() < 0.3:  # 30% chance
                size = 16
                h, w = x.size(2), x.size(3)
                y = torch.randint(0, h - size + 1, (1,)).item()
                x_pos = torch.randint(0, w - size + 1, (1,)).item()
                x[:, :, y:y+size, x_pos:x_pos+size] = 0

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        return self.linear(out)


# Function to create ResNet34
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


# Setting up model and training tools
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet34().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # loss function with small smoothing
optimizer = optim.AdamW(model.parameters(),
                     lr=0.001,      # initial learning rate
                     weight_decay=0.03)  # for regularization

# Setup learning rate schedule
num_epochs = 80
steps_per_epoch = len(train_dataloader)
total_steps = num_epochs * steps_per_epoch

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.003,
    total_steps=total_steps,
    pct_start=0.2,    # warmup percentage
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0
)

scaler = torch.amp.GradScaler('cuda')

# Lists to track training
train_loss_history = []
val_loss_history = []
lr_history = []
best_val_acc = 0
best_model_state = None


# Train
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_dataloader
        else:
            model.eval()
            dataloader = val_dataloader

        running_loss = 0.0
        running_corrects = 0
        num_samples = 0

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)  # تحسين كفاءة الذاكرة

            with torch.set_grad_enabled(phase == 'train'):
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    # زيادة قيمة max_norm
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                num_samples += inputs.size(0)

        epoch_loss = running_loss / num_samples
        epoch_acc = running_corrects / num_samples

        if phase == 'train':
            train_loss_history.append(epoch_loss)
        else:
            val_loss_history.append(epoch_loss)
            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_state = model.state_dict().copy()

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

model.load_state_dict(best_model_state)
print('Training complete')


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
        pred = torch.softmax(model(image), dim=1)  


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

