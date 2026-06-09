import os
from PIL import Image
from tqdm import tqdm
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision
import numpy as np


import cv2
from glob import glob
class ImbaDataset(Dataset):
    def __init__(self, root_dir, transform=None,is_test = False):
        self.samples = []
        self.labels = []
        self.transform = transform
        if is_test:
            self.samples = sorted(glob(root_dir))
            self.labels = [0]* len(self.samples)

        else:
            for class_name in os.listdir(root_dir):
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append(os.path.join(class_dir, fname))
                        self.labels.append(class_to_idx[class_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label
        

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dir ='/kaggle/input/iaio-2026-sf-r-image-classification/train_img/train'


class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(class_names)
class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
idx_to_class = {v: k for k, v in class_to_idx.items()}

dataset = ImbaDataset(train_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Train Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Epoch {epoch+1} Val Loss: {val_loss / len(val_loader):.4f} | Val Acc: {100 * correct / total:.2f}%")


model.eval()
test_dataset = ImbaDataset('/kaggle/input/iaio-2026-sf-r-image-classification/test_img/test/*', transform=transform,is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
preds = []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc=f"test"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy().tolist())



import pandas as pd
sub = pd.DataFrame({'path':sorted(os.listdir('/kaggle/input/iaio-2026-sf-r-image-classification/test_img/test')),'label':preds})
sub['label'] = sub['label'].apply(lambda x: idx_to_class[x])
sub.to_csv('submission.csv',index=False)

