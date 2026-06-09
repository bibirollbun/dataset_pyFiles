import os
import numpy as np
import pandas as pd
from PIL import Image
from time import time as timer
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, DataLoader


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


labels_csv = pd.read_csv("/kaggle/input/cifar-10/trainLabels.csv")
labels_csv


labels_csv.info()


# Since the labels are stored as text in the DataFrame, we'll have to the map the labels to numbers
label_mapping = {label: idx for idx, label in enumerate(labels_csv['label'].unique())}

# Now let's encode them, and rename the text column
labels_csv.rename({"label": "label_txt"}, axis=1, inplace=True)
labels_csv['label'] = labels_csv['label_txt'].map(label_mapping)

label_mapping, labels_csv


sample_submissions_csv = pd.read_csv("/kaggle/input/cifar-10/sampleSubmission.csv")
sample_submissions_csv.head()


!pip install py7zr
import py7zr
import os

# Define paths
archive_path = ["/kaggle/input/cifar-10/train.7z", "/kaggle/input/cifar-10/test.7z"]  # Change this to your actual 7z file
extract_path = "/kaggle/working"    # Temporary folder

# Extract files
for i in range(2):
    with py7zr.SevenZipFile(archive_path[i], mode='r') as archive:
        archive.extractall(path=extract_path)

print("Extraction done")


train_dir = "/kaggle/working/train"
test_dir = "/kaggle/working/test"

train_imgs = os.listdir(train_dir)

print(f"Number of train images: {len(train_imgs)}")
print(f"First 5 images' paths: {train_imgs[:5]}")

img1 = Image.open(train_dir + "/" + train_imgs[0])
img1.size


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))    # To shift each pixel's value between -1 and 1, helping avoid exploding and vanishing gradients
])

img_tensor = transform(img1)
img_tensor.shape   # Shape is 32x32x3, which tracks with how it should be


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.2),   # Randomly flip about 20% images horizontally, to help model learn right-left symmetry
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),   # To shift each pixel's value between [-1, 1], helping avoid exploding and vanishing gradients
])

# RandomCrop not applied, because Cropping an image in the CIFAR-10 dataset may make the image inconclusive.

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None):
        self.transforms = transform
        self.imgs_dir = images_dir
        self.labels_csv = labels_csv

    def __len__(self):
        return len(self.labels_csv)
    
    # Function to load each image, since we don't have the default structure that PyTorch wants it's Datasets to have
    def __getitem__(self, idx):
        img_name = str(self.labels_csv.iloc[idx, 0]) + ".png"    # Get file name of image with index "idx"
        label = self.labels_csv.loc[idx, "label"]   # Get label of the image with index "idx"

        img_path = os.path.join(self.imgs_dir, img_name)

        img = Image.open(img_path).convert("RGB")   # Open the image in a PIL.Image format, and convert it to an RGB image

        # Applying transforms
        if self.transforms:
            img = self.transforms(img)
        
        return img, label


# Creating the DataLoader for training data
train_dataset = CustomDataset(
    images_dir = "/kaggle/working/train", labels_csv = labels_csv,
    transform = transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class CNN(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # Conv Layer 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv Layer 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pooling 1
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv Layer 3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling 2
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),  # Fully connected layer 1
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),  # Fully connected layer 2
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # Dropout for regularization, increased to 0.5
            nn.Linear(128, 10)  # Output layer (10 classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        
        return x

# Initialize model
model = CNN()
print(model)


# Shapes of all parameters
for param in model.parameters():
    print(param.shape)


# Setting the Loss criteria and the optimizer
loss = nn.CrossEntropyLoss()   # Not MSE as this is not a regression problem, but a multi classification problem
optimizer = optim.Adam(model.parameters(), lr=1e-3)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device


# You cannot calculate metrics on your training data, and you can't do it on the testing set as well,
# so we first need to create a training and validation set as well
train_split = int(0.8 * len(train_dataset))
val_split = int(0.2 * len(train_dataset))
train_split, val_split

train_dataset, val_dataset = random_split(train_dataset, [train_split, val_split])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class EarlyStopping:
    def __init__(self, patience = 5, min_change = 0.001, checkpoint_pth = "best_model.pth"):
        self.patience = patience
        self.min_change = min_change
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.checkpoint_pth = checkpoint_pth

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_change:
            
            self.reset()
            
            self.best_loss = val_loss

            torch.save(model.state_dict(), self.checkpoint_pth)
        else:
            self.counter += 1

            print(f"Early Stopping Counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True

    def reset(self):
        self.counter = 0


def plot_train_and_val_losses(train_losses, val_losses):
    plt.plot(train_losses, color='b', label='Training Loss')
    plt.plot(val_losses, color='g', label='Validation Loss')

    plt.title("Loss VS Epochs")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    
    plt.show()


def calculate_metrics(model, val_loader, device):
    model.eval()    # Set the model to evaluation mode

    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1


# Creating a more complex training function, with EarlyStopping and calculating all evaluation metrics
early_stopper = EarlyStopping()

def train(model, train_loader, val_loader, loss_fn, optimizer, early_stopper, device, n_epochs=5):
    print(f"Device being used: {device}")
    
    model.to(device)
    train_losses = []
    val_losses = []
    
    for epoch_num in range(1, n_epochs+1):
        start_time = timer()
        model.train()    # Set the model to training model
        train_loss = 0.0

        # Training step
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()             # Reset gradients to 0

            outputs = model(images)           # Forward pass
            loss = loss_fn(outputs, labels)
            loss.backward()                   # Start Backprop
            
            optimizer.step()                  # Update gradients
    
            train_loss += loss.item()
    
        train_loss /= len(train_loader)
        accuracy, precision, recall, f1 = calculate_metrics(model, val_loader, device)

        train_losses.append(train_loss)
    
        # Validation step
        model.eval()    # Set the model to evaluation mode
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        
        val_losses.append(val_loss)
        
        print(
            f"Epoch {epoch_num}/{n_epochs} |  Time: {timer()-start_time:.2f} | Train Loss = {train_loss:.4f} |",
            f"Validation Loss: {val_loss:.4f} |",
            f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
        )
        
        # Call Early Stopping
        early_stopper(val_loss, model)
        
        if early_stopper.early_stop:
            print(f"Early Stopping Triggered. Best model saved at {early_stopper.checkpoint_pth}")
            break

    return train_losses, val_losses


# Increasing the Learning Rate
model = CNN(dropout_prob=0.7)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

earlyStopper = EarlyStopping()

train_losses, val_losses = train(model, train_loader, val_loader, loss, optimizer, earlyStopper, device, 10)
plot_train_and_val_losses(train_losses, val_losses)


# Creating a custom Dataset class and the DataLoader

class CustomTestDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.img_files = sorted(os.listdir(images_dir))  # Ensure correct order

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")  # Open image as RGB

        if self.transform:
            image = self.transform(image)

        return image, self.img_files[idx]


test_dataset = CustomTestDataset("/kaggle/working/test", test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Loading best model
model = CNN()
model.load_state_dict(torch.load("best_model.pth"))  
model.to(device)
model.eval()     # Set to evaluation mode


# Make predictions
all_preds = []
img_idxs = []

with torch.no_grad():
    for images, idxs in test_loader:
        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        img_idxs.extend(idxs)


# Re-mapping predictions
idxs_to_classes = {v: k for k, v in label_mapping.items()}

predicted_labels = [idxs_to_classes[pred] for pred in all_preds]

submission_df = pd.DataFrame({
    "id": [int(file.split(".")[0]) for file in img_idxs],  
    "label": predicted_labels
})

# Sort by ID to ensure correct order
submission_df = submission_df.sort_values(by="id")

# Save to CSV
submission_df.to_csv("submission.csv", index=False)


submission_df




