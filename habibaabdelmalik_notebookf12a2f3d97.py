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


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from PIL import Image



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



data_path = "/kaggle/input/cassava-leaf-disease-classification/"
images_path = os.path.join(data_path, "train_images")
csv_path = os.path.join(data_path, "train.csv")



df = pd.read_csv(csv_path)

df.head()



train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=1/3, stratify=temp_df['label'], random_state=42)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))





train_transforms = transforms.Compose([
    transforms.Resize((300, 300)),           
    transforms.RandomHorizontalFlip(),      
    transforms.RandomRotation(20),          
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id']
        label = self.df.iloc[idx]['label']
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label



# Datasets
train_dataset = CassavaDataset(train_df, images_path, transform=train_transforms)
val_dataset = CassavaDataset(val_df, images_path, transform=val_transforms)
test_dataset = CassavaDataset(test_df, images_path, transform=val_transforms)

# DataLoaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)



criterion = nn.CrossEntropyLoss()


!pip install efficientnet-pytorch



from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



model = EfficientNet.from_pretrained('efficientnet-b3')

model._fc = nn.Linear(model._fc.in_features, 5)

model = model.to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0
    running_correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += torch.sum(preds == labels)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_correct.double() / len(loader.dataset)
    return epoch_loss, epoch_acc
loss, acc = train_one_epoch(model, train_loader, optimizer, criterion)

print("Training Loss:", loss)
print("Training Accuracy:", acc)




def validate(model, loader, criterion):
    model.eval()
    running_loss = 0
    running_correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == labels)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_correct.double() / len(loader.dataset)
    return epoch_loss, epoch_acc



epochs = 15
patience = 3
best_acc = 0
trigger_times = 0

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)

    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        trigger_times = 0
        print(">>> Saved Best Model")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping activated!")
            break





