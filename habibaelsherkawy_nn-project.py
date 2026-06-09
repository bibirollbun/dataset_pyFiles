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


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Set device, random seed, and load the CSV file with image labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
print(f'Using device: {device}')

data_dir = '/kaggle/input/cassava-leaf-disease-classification/train_images'
csv_path = '/kaggle/input/cassava-leaf-disease-classification/train.csv'
df = pd.read_csv(csv_path)
print(f'Total samples: {len(df)}')
print(f'Class distribution:\n{df["label"].value_counts()}')
print(df.head())


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class CassavaDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        img_name = self.dataframe.loc[index, 'image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.loc[index, 'label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

print('Transformations and Dataset class defined successfully')


# 70% Train, 30% Temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df['label']
)

# 20% Val, 10% Test
val_df, test_df = train_test_split(
    temp_df,
    test_size=1/3,   # يعني 10% من الداتا الكلية
    random_state=42,
    stratify=temp_df['label']
)

print(f'Training samples: {len(train_df)}')
print(f'Validation samples: {len(val_df)}')
print(f'Test samples: {len(test_df)}')
train_dataset = CassavaDataset(train_df, data_dir, transform=train_transform)
val_dataset   = CassavaDataset(val_df, data_dir, transform=val_transform)
test_dataset  = CassavaDataset(test_df, data_dir, transform=val_transform)



train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

print(f'Number of training batches: {len(train_loader)}')
print(f'Number of validation batches: {len(val_loader)}')
print(f'Number of test batches: {len(test_loader)}')



# Test data loading and display sample images to verify preprocessing
images, labels = next(iter(train_loader))
print(f'Batch image shape: {images.shape}')
print(f'Batch labels shape: {labels.shape}')
print(f'Sample labels: {labels[:5]}')

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for idx, ax in enumerate(axes.flat):
    if idx < len(images):
        img = denormalize(images[idx]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f'Label: {labels[idx].item()}')
        ax.axis('off')
plt.tight_layout()
plt.show()
print('Data preprocessing completed successfully')


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
             # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4)
        )

        # Adaptive Pooling to fix input size for FC
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3),

            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x)  
        x = x.view(x.size(0), -1) 
        x = self.fc_layers(x)
        return x

print("SimpleCNN model defined successfully")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(model, dataloader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total



num_epochs = 15

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")



test_acc = evaluate(model, test_loader)
print(f"Test Accuracy (CNN from scratch): {test_acc:.4f}")



import torchvision.models as models
# 1. Load Pre-trained ResNet50
my_model = models.resnet50(pretrained=True)

# 2. Freeze all layers
for param in my_model.parameters():
    param.requires_grad = False
# Unfreeze last block
for param in my_model.layer4.parameters():
    param.requires_grad = True    

# 3. Replace the last layer (for 5 disease classes)
num_ftrs = my_model.fc.in_features
my_model.fc = nn.Linear(num_ftrs, 5)

# 4. Move model to GPU
my_model = my_model.to(device)

print("Pre-trained model is ready on GPU!")



# Define Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()), lr=0.001)

print("Step 2 complete: Loss and Optimizer defined.")


def train_model(model, criterion, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        # train_loader must be defined in the previous cells
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

print("Step 3 complete: Training function is ready.")


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    
    accuracy = float(num_correct) / num_samples
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    return accuracy

# Execute Training and Evaluation
train_model(my_model, criterion, optimizer, epochs=10)
check_accuracy(train_loader, my_model)


test_acc = check_accuracy(test_loader, my_model)


