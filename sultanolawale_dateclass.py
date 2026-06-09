# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import os
data_dir = "/kaggle/input/open-data-day-2025-dates-types-classification" 
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
train_labels_path = os.path.join(data_dir, "train_labels.csv")
sample_submission_path = os.path.join(data_dir, "sample_submission.csv")


from PIL import Image
import os

image_path = os.path.join(train_dir,"fe011678.jpg")
image = Image.open(image_path)

image.resize((224,224))


df_train_labels = pd.read_csv(train_labels_path)


from torch.utils.data import Dataset
from PIL import Image

from PIL import Image
import torch

class DateDataset(Dataset):
    def __init__(self, image_dir, labels_df, transform=None):
        self.image_dir = image_dir
        self.labels_df = labels_df
        self.transform = transform

        # Create class-to-index mapping
        self.classes = sorted(self.labels_df["label"].unique())  # Get unique class names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        label = self.labels_df.iloc[idx, 1]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Convert label to index
        label = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)
        
        return image, label


from torchvision import transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),  # Ensures 3 channels
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


transform_valid_test = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),  # Ensures 3 channels
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


import torch
from torch.utils.data import random_split, DataLoader
dataset = DateDataset(train_dir, df_train_labels, transform=transform)

train_size = int(0.8 * len(dataset)) 
val_size = len(dataset) - train_size  


train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


import matplotlib.pyplot as plt
import numpy as np


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class_names = dataset.classes

fig, axes = plt.subplots(2, 4, figsize=(18, 7))  


imgs_indices = [10, 150, 233, 43, 93, 96, 125, 32]

for i in range(8):
    img, label = train_dataset[imgs_indices[i]] 

    
    img_np = img.numpy().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)

   
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    class_name = class_names[label]
    row = i // 4  
    col = i % 4   
    axes[row, col].imshow(img_np)
    axes[row, col].set_title(f"Label: {class_name}")
    axes[row, col].axis('off')  

plt.show()


from torchvision import models
from torch import nn

model = models.efficientnet_b0(pretrained=True)

num_classes = 7
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


import torch
from tqdm import tqdm   


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        
        outputs = model(images)  

        
        loss = criterion(outputs, labels)

        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

        total_loss += loss.item()

        # Track accuracy
        _, predicted = torch.max(outputs, 1)  
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            
            outputs = model(images)  

            # Compute the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            
            _, predicted = torch.max(outputs, 1)  
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy



import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)

    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Train Loss={train_loss:.4f}, Train Accuracy={train_accuracy:.2f}%, "
          f"Val Loss={val_loss:.4f}, Val Accuracy={val_accuracy:.2f}%")


import matplotlib.pyplot as plt


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label="Train Accuracy", marker='o')
plt.plot(range(1, num_epochs+1), val_accuracies, label="Validation Accuracy", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve")
plt.legend()

plt.show()



import matplotlib.pyplot as plt
import numpy as np
import torch


def show_predictions(model, dataloader, device, num_images=10):
    model.eval()  
    images, labels = next(iter(dataloader))  
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():  
        outputs = model(images)
        predictions = outputs.argmax(dim=1)  

   
    date_classes = dataset.classes

    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])  

   
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat[:num_images]):
        img = images[i].cpu().numpy().transpose(1, 2, 0) 
        img = img * std + mean 
        img = np.clip(img, 0, 1)  

        pred_label = predictions[i].item()
        true_label = labels[i].item()

        

        ax.imshow(img)
        ax.set_title(f"Pred: { date_classes[pred_label]}\nTrue: { date_classes[true_label]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


show_predictions(model, val_loader, device)



torch.save(model, "model.pth")


import torch
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Define the dataset class (Modify as per your dataset structure)
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))  # Get all image names
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name  # Return image and filename


test_image_dir = "/kaggle/input/open-data-day-2025-dates-types-classification/test"  # Change this to your test image directory
output_csv_path = "predictions.csv"


transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),  
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


test_dataset = TestDataset(test_image_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = torch.load("model.pth") 
model.eval()  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class_to_idx =dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()} 


predictions = []


with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Store results with class names
        for img_name, pred_label in zip(img_names, predicted.cpu().numpy()):
            class_name = idx_to_class[pred_label]  
            predictions.append([img_name, class_name])


df = pd.DataFrame(predictions, columns=["filename", "label"])
df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")





