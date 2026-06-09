from __future__ import print_function

import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


#################################
# Configuration & Reproductibilité
#################################
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



# -- Chemins vers les données --
base_dir = '/kaggle/input/cidaut-ai-fake-scene-classification-2024'
train_csv_path = os.path.join(base_dir, 'train.csv')
images_dir = os.path.join(base_dir, 'Train')

# -- Lecture du CSV --
df = pd.read_csv(train_csv_path)
# df contient deux colonnes : "images" (nom du fichier .jpg, .png, etc.) et "label" ("editada" ou "real")

# -- Conversion des labels editada -> 0, real -> 1 --
cls_to_idx = {'editada': 0, 'real': 1}
df['label'] = df['label'].map(cls_to_idx)

# -- Récupération des chemins d'images et des labels --
# on suppose que la colonne 'images' contient les noms de fichiers (ex: "image123.jpg")
all_image_paths = [os.path.join(images_dir, img_name) for img_name in df['image']]
all_labels = df['label'].values

# -- Split 70% / 15% / 15% --
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_image_paths,
    all_labels,
    test_size=0.30,        # 30% à répartir ensuite pour val et test
    stratify=all_labels,
    random_state=seed
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,        # la moitié de 30% -> 15%
    stratify=temp_labels,
    random_state=seed
)

print(f"Train Data: {len(train_paths)}")
print(f"Validation Data: {len(val_paths)}")
print(f"Test Data: {len(test_paths)}")


# -- Transforms --
"""
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
"""


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



"""
# -- Transforms --
train_transforms = transforms.Compose([
    transforms.Resize((518, 518)),  # Redimensionner à la hauteur correcte
    transforms.RandomHorizontalFlip(),  # Ajout d'augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((518, 518)),  # Redimensionner à 518x518
    transforms.CenterCrop(518),    # Crop au centre si nécessaire
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((518, 518)),  # Redimensionner à 518x518
    transforms.CenterCrop(518),    # Crop au centre
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
"""



# -- Custom Dataset --
class FakeRealDataset(Dataset):
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

# -- Datasets et DataLoaders --
train_data = FakeRealDataset(train_paths, train_labels, transform=train_transforms)
val_data   = FakeRealDataset(val_paths,   val_labels,   transform=val_transforms)
test_data  = FakeRealDataset(test_paths,  test_labels,  transform=test_transforms)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(dataset=val_data,   batch_size=batch_size, shuffle=False, num_workers=4)
test_loader  = DataLoader(dataset=test_data,  batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Train Dataset size: {len(train_data)}")
print(f"Validation Dataset size: {len(val_data)}")
print(f"Test Dataset size: {len(test_data)}")



# -- Chargement du modèle ViT pré-entraîné --
model = models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1)
model.to(device)

# Récupération du nombre de features en sortie de la penultimate layer
num_ftrs = model.fc.in_features

# On remplace la couche fully-connected par une couche linéaire 2 classes
model.fc = nn.Linear(num_ftrs, 2)


# -- Si plusieurs GPUs sont disponibles --
#model = nn.DataParallel(model)
model.to(device)



# -- Setup perte, optimizer, scheduler --
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



# -- Entraînement --
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

    # Moyenne des pertes et accuracy par batch
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

            # Pour calculer l'AUC plus tard
            val_preds_list.extend(output[:, 1].cpu().numpy())  # probabilité de la classe 1
            val_labels_list.extend(label.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    # -- AUC sur la validation --
    # On compare la "probabilité" de la classe 1 (sortie du réseau) avec le label 0 ou 1
    val_auc = roc_auc_score(val_labels_list, val_preds_list)

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
    )

    scheduler.step()


# -- Courbes de Training / Validation --
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



# -- Test --
model.eval()
test_loss = 0
test_acc = 0
test_preds_list = []
test_labels_list = []

with torch.no_grad():
    for data, label in tqdm(test_loader, desc="Testing"):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)
        test_loss += loss.item()

        preds = output.argmax(dim=1)
        acc = (preds == label).float().mean()
        test_acc += acc.item()

        test_preds_list.extend(output[:, 1].cpu().numpy())  # prob classe 1
        test_labels_list.extend(label.cpu().numpy())

test_loss /= len(test_loader)
test_acc /= len(test_loader)
test_auc = roc_auc_score(test_labels_list, test_preds_list)

print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f} - Test AUC: {test_auc:.4f}")


