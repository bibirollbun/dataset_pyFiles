import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import urllib.request
from PIL import Image
from pathlib import Path


train_dir = '/kaggle/input/garbage-guru-challenge/dataset/train'
test_dir = '/kaggle/input/garbage-guru-challenge/dataset/test'
val_dir = '/kaggle/input/garbage-guru-challenge/dataset/val'
torch.manual_seed(42)


transform_train = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
transform_test = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
transform_val = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])



train_data = datasets.ImageFolder(train_dir, transform=transform_train)
val_data   = datasets.ImageFolder(val_dir, transform=transform_val)
test_data  = datasets.ImageFolder(test_dir, transform=transform_test)



def show_sample(img, label):
    print("Label:", train_data.classes[label], "(Class No: "+ str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))


print(len(train_data))
img, label = train_data[100]
show_sample(img, label)


train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=16, shuffle=False)



print("Classes:", train_data.classes)



def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow = 16).permute(1, 2, 0))
        break
show_batch(train_loader)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace FC head
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_data.classes))

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()



def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, running_corrects = 0.0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()


def fit(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
              f"- Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")



fit(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)



torch.save(model.state_dict(), "resnet50_model.pth")



# Example if you saved best model
model.load_state_dict(torch.load("resnet50_model.pth"))
model.eval()



from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd

# Define same transforms as validation
test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Load test data (ImageFolder works only if subfolders exist;
# for Kaggle test, usually it's unlabeled, so use ImageFolder with dummy)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Predictions
preds = []
model.eval()
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy())



idx_to_class = {v: k for k, v in train_data.class_to_idx.items()}
pred_labels = [idx_to_class[p] for p in preds]
pred_labels


# Get file names in same order as test_loader.dataset
filenames = [path[0].split("/")[-1] for path in test_loader.dataset.samples]

submission = pd.DataFrame({
    "image_id": filenames,
    "label": pred_labels
})

submission.to_csv("submission.csv", index=False)




