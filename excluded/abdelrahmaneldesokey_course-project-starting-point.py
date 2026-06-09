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
    ..., #TODO: Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    ..., #TODO: Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = "/kaggle/input/mnistxcifar-competition/MNISTxCIFAR/train"
test_dir = "/kaggle/input/mnistxcifar-competition/MNISTxCIFAR/test"

# Load the training data
full_train_dataset = MNISTxCIFAR(train_dir, transform=train_transform) # Create the full training dataset

# Split the full training dataset into training and validation sets
# Let's say we want to use 80% of the samples for training and 20% for validation

# TODO: Split the dataset based on 80-20 split
train_size = int(... * len(full_train_dataset))  # 80% of the dataset size
val_size = len(full_train_dataset) - train_size  # The remaining samples

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create data loaders 
# TODO: Fill in the loader parameters
train_dataloader = DataLoader(train_dataset, batch_size=..., shuffle=...) # Create the training dataloader
val_dataloader = DataLoader(val_dataset, batch_size=..., shuffle=...) # Create the validation dataloader

# Load the test data
# TODO: Fill in the loader parameters
test_dataset = MNISTxCIFAR(test_dir, train=..., transform=test_transform) # Create the test dataset
test_dataloader = DataLoader(test_dataset, batch_size=..., shuffle=...) # Create the test dataloader

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



import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.pool1 = nn.MaxPool2d(2, 2) # Max pooling over a (2, 2) window
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 square convolution kernel
        self.pool2 = nn.MaxPool2d(2, 2) # Max pooling over a (2, 2) window
        
        # TODO: Construct your linear layers
        self.fc1 = nn.LazyLinear(...) # an affine operation: y = Wx + b, 120 output nodes
        self.fc2 = nn.Linear(..., ...) # 84 output nodes
        self.fc3 = nn.Linear(..., ...) # 10 output nodes for the 10 classes

    def forward(self, x):
        bs = x.shape[0]
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(bs, -1) # reshape tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# model instance
model = LeNet()

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
# TODO: Select a criterion and fill in the learning rate and momentum values
criterion = ...
optimizer = optim.SGD(model.parameters(), lr=..., momentum=...)

# Decay LR by a factor of 0.1 every 7 epochs
# TODO: Implement an exponential learning rate scheduler
exp_lr_scheduler = ...

# Training loop
# TODO: Select your training epochs

num_epochs = ...
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            dataloader = train_dataloader
        else:
            model.eval()   # Set model to evaluate mode
            dataloader = val_dataloader

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            exp_lr_scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

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

