from __future__ import print_function

import glob
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


batch_size = 8
epochs = 10
lr = 5e-5
gamma = 0.7
seed = 42


def seed_everything(seed):
    """
    Sets the random seed for reproducibility in python, numpy, and torch.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device used :", device)


# -- Paths to the data --
base_dir = '/kaggle/input/cidaut-ai-fake-scene-classification-2024'
train_csv_path = os.path.join(base_dir, 'train.csv')

# Directory containing training images
train_images_dir = os.path.join(base_dir, 'Train')

# Directory containing test images (without labels)
test_images_dir = os.path.join(base_dir, 'Test')

# -- Reading the train CSV file --
df = pd.read_csv(train_csv_path)
# df contains two columns: "images" (file name), "label" ("editada" or "real")

# -- Converting labels: editada -> 0, real -> 1 --
cls_to_idx = {'editada': 0, 'real': 1}
df['label'] = df['label'].map(cls_to_idx)

# -- Retrieving image paths and labels --
all_image_paths = [os.path.join(train_images_dir, img_name) for img_name in df['image']]
all_labels = df['label'].values

# -- Train/validation split (85% / 15%) --
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths,
    all_labels,
    test_size=0.15,
    stratify=all_labels,
    random_state=seed
)

print(f"Train Data: {len(train_paths)}")
print(f"Validation Data: {len(val_paths)}")


# -- Transforms --
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class FakeRealDataset(Dataset):
    """
    Dataset for training (train/val) where labels are available.
    """
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


class FakeRealTestDataset(Dataset):
    """
    Dataset for testing (no labels).
    Returns the transformed image and the image name for submission.
    """
    def __init__(self, directory, transform=None):
        self.transform = transform
        # Retrieve all file names in the Test directory
        self.image_names = sorted(os.listdir(directory))
        self.image_paths = [os.path.join(directory, img) for img in self.image_names]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]  # for constructing the submission file
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name

# -- Datasets and DataLoaders --
train_data = FakeRealDataset(train_paths, train_labels, transform=train_transforms)
val_data   = FakeRealDataset(val_paths,   val_labels,   transform=val_transforms)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(dataset=val_data,   batch_size=batch_size, shuffle=False, num_workers=4)


# -- Loading the pre-trained ViT model --
model = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
model.to(device)

# -- Modifying the last layer to output 2 classes (0 or 1) --
num_ftrs = model.heads.head.in_features
model.heads.head = nn.Linear(num_ftrs, 2)

# Use multiple GPUs if available (ONLY if u use GPU T4*2)
# model = nn.DataParallel(model)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = output.argmax(dim=1)
        acc = (preds == label).float().mean()
        epoch_accuracy += acc.item()

    # Average per batch
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # -- Validation --
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for data, label in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)
            val_loss += loss.item()

            preds = output.argmax(dim=1)
            acc = (preds == label).float().mean()
            val_acc += acc.item()

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    scheduler.step()


plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(train_losses, label="Training")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title("Training and Validation Accuracy")
plt.plot(train_accuracies, label="Training")
plt.plot(val_accuracies, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Create a test dataset/loader without labels
test_dataset = FakeRealTestDataset(test_images_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model.eval()
predictions = []
image_names = []

with torch.no_grad():
    for data, names in tqdm(test_loader, desc="Predicting"):
        data = data.to(device)
        output = model(data)
        # output has size (batch_size, 2)
        probs = nn.Softmax(dim=1)(output)  # (batch_size, 2)
        # Directly retrieve the probability of class 1 (real)
        real_probs = probs[:, 1].cpu().numpy()
        predictions.extend(real_probs)
        image_names.extend(names)

# -- Construct the submission DataFrame --
submission_df = pd.DataFrame({
    'image': image_names,
    'label': predictions  # probability of class 1 (real)
})


# -- Saving to a CSV file --
submission_df.to_csv("submit_ViT.csv", index=False)
print("Submission file generated: submit_ViT.csv")


batch_size = 8
epochs = 25
lr = 5e-5
gamma = 0.7
seed = 42


def seed_everything(seed):
    """
    Sets the random seed for reproducibility in python, numpy, and torch.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device used :", device)


# Paths to the dataset
base_dir = '/kaggle/input/cidaut-ai-fake-scene-classification-2024'
train_csv_path = os.path.join(base_dir, 'train.csv')
train_images_dir = os.path.join(base_dir, 'Train')
test_images_dir = os.path.join(base_dir, 'Test')

# Load training labels
train_df = pd.read_csv(train_csv_path)
# Map labels 'editada' -> 0 and 'real' -> 1
cls_to_idx = {'editada': 0, 'real': 1}
train_df['label'] = train_df['label'].map(cls_to_idx)

# Extract image paths and labels
all_image_paths = [os.path.join(train_images_dir, img_name) for img_name in train_df['image']]
all_labels = train_df['label'].values

# Split train/validation (95% / 5%) --
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths,
    all_labels,
    test_size=0.05,        
    stratify=all_labels,
    random_state=seed
)

print(f"Train Data: {len(train_paths)}")
print(f"Validation Data: {len(val_paths)}")


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class FakeRealDataset(Dataset):
    """
    Dataset for training and validation with labels.
    """
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

class FakeRealTestDataset(Dataset):
    """
    Dataset for testing without labels.
    """
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.image_names = sorted(os.listdir(directory))
        self.image_paths = [os.path.join(directory, img) for img in self.image_names]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name

# Create DataLoaders
train_data = FakeRealDataset(train_paths, train_labels, transform=train_transforms)
val_data = FakeRealDataset(val_paths, val_labels, transform=val_transforms)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Train Dataset size: {len(train_data)}")
print(f"Validation Dataset size: {len(val_data)}")


# Load pre-trained EfficientNet
model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)

# Modify the classifier to output 2 classes (real and fake)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

# Use multiple GPUs if available (ONLY if u use GPU T4*2)
# model = nn.DataParallel(model)
model.to(device)



# Define loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = output.argmax(dim=1)
        acc = (preds == label).float().mean()
        epoch_accuracy += acc.item()

    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # -- Validation --
    model.eval()
    val_loss = 0
    val_acc = 0
    val_preds_list = []
    val_labels_list = []


    with torch.no_grad():
        for data, label in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)
            val_loss += loss.item()

            preds = output.argmax(dim=1)
            acc = (preds == label).float().mean()
            val_acc += acc.item()
            
            val_preds_list.extend(output[:, 1].cpu().numpy()) 
            val_labels_list.extend(label.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    val_auc = roc_auc_score(val_labels_list, val_preds_list)


    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
    )

    scheduler.step()


plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(train_losses, label="Training")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title("Training and Validation Accuracy")
plt.plot(train_accuracies, label="Training")
plt.plot(val_accuracies, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


test_dataset = FakeRealTestDataset(test_images_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model.eval()
predictions = []
image_names = []

with torch.no_grad():
    for data, names in tqdm(test_loader, desc="Predicting"):
        data = data.to(device)
        output = model(data)
        probs = nn.Softmax(dim=1)(output)
        real_probs = probs[:, 1].cpu().numpy()
        predictions.extend(real_probs)
        image_names.extend(names)

submission_df = pd.DataFrame({
    'image': image_names,
    'label': predictions
})


submission_df.to_csv("submit_efficientNet.csv", index=False)
print("Submission file generated: submit_efficientNet.csv")


batch_size = 8
epochs = 10 # 5 for the best result (0.9323 AUC ROC) 
lr = 5e-5
gamma = 0.7
seed = 42


def seed_everything(seed):
    """
    Sets the random seed for reproducibility in python, numpy, and torch.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device used :", device)


# Paths to the dataset
base_dir = '/kaggle/input/cidaut-ai-fake-scene-classification-2024'
train_csv_path = os.path.join(base_dir, 'train.csv')
train_images_dir = os.path.join(base_dir, 'Train')
test_images_dir = os.path.join(base_dir, 'Test')

# Load training labels
train_df = pd.read_csv(train_csv_path)
# Map labels 'editada' -> 0 and 'real' -> 1
cls_to_idx = {'editada': 0, 'real': 1}
train_df['label'] = train_df['label'].map(cls_to_idx)

# Extract image paths and labels
all_image_paths = [os.path.join(train_images_dir, img_name) for img_name in train_df['image']]
all_labels = train_df['label'].values

# Split train/validation (95% / 5%) --
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths,
    all_labels,
    test_size=0.05,        
    stratify=all_labels,
    random_state=seed
)

print(f"Train Data: {len(train_paths)}")
print(f"Validation Data: {len(val_paths)}")


import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

train_transforms = T.Compose([
    T.Resize(224, interpolation=InterpolationMode.BICUBIC),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

val_transforms = T.Compose([
    T.Resize(224, interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

test_transforms = T.Compose([
    T.Resize(224, interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class FakeRealDataset(Dataset):
    """
    Dataset for training and validation with labels.
    """
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

class FakeRealTestDataset(Dataset):
    """
    Dataset for testing without labels.
    """
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.image_names = sorted(os.listdir(directory))
        self.image_paths = [os.path.join(directory, img) for img in self.image_names]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name

# Create DataLoaders
train_data = FakeRealDataset(train_paths, train_labels, transform=train_transforms)
val_data = FakeRealDataset(val_paths, val_labels, transform=val_transforms)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Train Dataset size: {len(train_data)}")
print(f"Validation Dataset size: {len(val_data)}")



# Load pre-trained Regnet
model = models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1)

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)


#model = nn.DataParallel(model)
model.to(device)



# Define loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = output.argmax(dim=1)
        acc = (preds == label).float().mean()
        epoch_accuracy += acc.item()

    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # -- Validation --
    model.eval()
    val_loss = 0
    val_acc = 0
    val_preds_list = []
    val_labels_list = []


    with torch.no_grad():
        for data, label in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)
            val_loss += loss.item()

            preds = output.argmax(dim=1)
            acc = (preds == label).float().mean()
            val_acc += acc.item()
            
            val_preds_list.extend(output[:, 1].cpu().numpy()) 
            val_labels_list.extend(label.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    val_auc = roc_auc_score(val_labels_list, val_preds_list)


    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
    )

    scheduler.step()


plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(train_losses, label="Training")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title("Training and Validation Accuracy")
plt.plot(train_accuracies, label="Training")
plt.plot(val_accuracies, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


test_dataset = FakeRealTestDataset(test_images_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model.eval()
predictions = []
image_names = []

with torch.no_grad():
    for data, names in tqdm(test_loader, desc="Predicting"):
        data = data.to(device)
        output = model(data)
        probs = nn.Softmax(dim=1)(output)
        real_probs = probs[:, 1].cpu().numpy()
        predictions.extend(real_probs)
        image_names.extend(names)

submission_df = pd.DataFrame({
    'image': image_names,
    'label': predictions
})


submission_df.to_csv("submit_regnet.csv", index=False)
print("Submission file generated: submit_regnet.csv")

