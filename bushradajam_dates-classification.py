import os
import gc
import cv2
from torchvision import transforms
import torch
import random
import numpy as np
import pandas as pd
from glob import glob
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import torch.optim as optim
import torch
import cv2
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, Subset


train_df = pd.read_csv("/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv")
train_df


train_df["image_path"] = "/kaggle/input/open-data-day-2025-dates-types-classification/train/" + train_df["filename"]
train_df.drop("filename", axis=1, inplace=True)
train_df = train_df[["image_path","label"]]
train_df


import matplotlib.pyplot as plt
import cv2
from PIL import Image

image_paths = train_df["image_path"].head(5).tolist() 
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, path in enumerate(image_paths):
    try:
        # Load image using OpenCV
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Display the image
        axes[i].imshow(image)
        axes[i].axis("off")
    except Exception as e:
        print(f"Error loading {path}: {e}")

plt.show()


test_df = glob("/kaggle/input/open-data-day-2025-dates-types-classification/test/*")
test_df = pd.DataFrame({"image_path": test_df})
test_df


train_df.head()


train_df.shape


train_df.columns


test_df['image_path'][0]


test_df.shape


test_df.columns


train_df['label'].value_counts()


classes_map = {
    "Ajwa":        0,
    "Medjool":     1,
    "Meneifi":     2,
    "Nabtat Ali":  3,
    "Shaishe":     4,
    "Sokari":      5,
    "Sugaey":      6
}


train_df["label_idx"] = train_df["label"].replace(classes_map)


train_df


train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df


import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image


aug_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize first to a larger size to retain details
    #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Ensure most of the fruit is visible
    transforms.RandomHorizontalFlip(p=0.5),  # Flip left-right
    transforms.RandomRotation(10),  # Small rotation since extreme rotations are unrealistic
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # Mild changes to simulate lighting
    #transforms.RandomAffine(degrees=5, shear=5),  # Small distortions to introduce diversity
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for EfficientNet
])


image_paths = train_df["image_path"].head(5).tolist()
augmented_images = []
for path in image_paths:
    try:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = Image.fromarray(image)  # Convert to PIL format

        aug_image = aug_transforms(image)
        aug_image = aug_image.permute(1, 2, 0).numpy()  # Convert back to NumPy

        augmented_images.append(aug_image)
    except Exception as e:
        print(f"Error loading {path}: {e}")

fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, img in enumerate(augmented_images):
    axes[i].imshow(img)
    axes[i].axis("off")
plt.show()


from sklearn.model_selection import StratifiedKFold

class DatesDataset(Dataset):
    def __init__(self, df, mode="train", transforms=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        image_path = row["image_path"]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)  # Convert to PIL Image

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)

        if self.mode != "test":
            label_idx = row.get("label_idx", -1)  # Default to -1 if not found
            return image, torch.tensor(label_idx, dtype=torch.long)
        else:
            return image

    @staticmethod

    def create_folds(df, k_folds=5):
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        df["fold"] = -1
    
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label_idx"])):
            df.loc[val_idx, "fold"] = fold
    
        folds = []
        for fold in range(k_folds):
            train_df = df[df["fold"] != fold].reset_index(drop=True)
            val_df = df[df["fold"] == fold].reset_index(drop=True)
            folds.append((train_df, val_df))
    
        return folds


import torch
import torch.nn as nn
from torchvision import models  

class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)

        # Freeze all layers except the last FC layer
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify the final classification layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)



def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Evaluate
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

    return val_acc


def k_fold_training(df, k_folds=5, num_epochs=10, batch_size=32, lr=0.001, save_path="best_model.pth"): #64 model.parameters(), lr=0.0005, weight_decay=1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), # Flip left-right
    transforms.RandomRotation(10),  # Small rotations 
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  
    #transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.RandomHorizontalFlip(p=0.5),  # Flip left-right (helps with viewpoint variation)
    transforms.RandomRotation(15),  # Rotate images slightly
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),  # Simulate shifts & shearing
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Simulate lighting changes
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Random occlusion
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])




    val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    folds = DatesDataset.create_folds(df, k_folds=k_folds)

    fold_accuracies = []
    best_fold_acc = 0.0
    best_model_state = None  

    for fold, (train_df, val_df) in enumerate(folds):
        print(f"\n========== Fold {fold+1}/{k_folds} ==========\n")

        train_dataset = DatesDataset(train_df, mode="train", transforms=train_transforms)
        val_dataset = DatesDataset(val_df, mode="val", transforms=val_transforms)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)# since f-fold make shuffle
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        num_classes = len(set(train_df["label_idx"]))  # Get classes from train data
        model = ImageClassifier(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=1e-4) 

        val_acc = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
        fold_accuracies.append(val_acc)

        model_path = f"model_fold{fold+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f" Model for Fold {fold+1} saved as {model_path}")

        # Track the best overall model
        if val_acc > best_fold_acc:
            best_fold_acc = val_acc
            best_model_state = model.state_dict()

    if best_model_state:
        torch.save({'model_state_dict': best_model_state, 'best_accuracy': best_fold_acc}, save_path)
        print(f"\nBest model saved as {save_path} with accuracy {best_fold_acc:.4f}")


    # Print final results
    print("\n========== Final K-Fold Results ==========")
    print(f"Average Validation Accuracy: {np.mean(fold_accuracies):.4f}")


k_fold_training(train_df, k_folds=5, num_epochs=5)


class TestDatesDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.image_paths = df["image_path"].values  
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)

        return image, os.path.basename(image_path) 


test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


model = ImageClassifier(num_classes=7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load checkpoint
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


test_dataset = TestDatesDataset(test_df, transforms=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Reverse the mapping so model outputs map to class names
classes_map = {v: k for k, v in classes_map.items()}


submission_data = []
with torch.no_grad():
    for images, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)  
        #print
        print("Predicted indices:", [pred.item() for pred in preds])
        #print("Available keys in classes_map:", classes_map.keys())

        predicted_labels = [classes_map.get(pred.item(), "Unknown") for pred in preds]

        submission_data.extend(zip(filenames, predicted_labels))


submission_df = pd.DataFrame(submission_data, columns=["filename", "label"])
submission_df.to_csv("submission.csv", index=False)
submission_df

