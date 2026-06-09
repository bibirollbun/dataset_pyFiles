import os
import torch
import timm  # Load pre-trained model
import pandas as pd
import numpy as np
from PIL import Image
import h5py
from io import BytesIO
import random
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Current Device:", device)
print("Visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print(torch.cuda.device_count()) 


train_csv_path = "/kaggle/input/isic-2024-challenge/train-metadata.csv"

train_metadata = pd.read_csv(train_csv_path, low_memory=False)
train_metadata.head()


test_csv_path  = "/kaggle/input/isic-2024-challenge/test-metadata.csv"
test_metadata = pd.read_csv(test_csv_path, low_memory=False)
test_metadata.head()


class SkinDataset(Dataset):
    def __init__(self, csv_file, hdf5_file, transform=None):
        """
        csv_file: csv file include isic_id and target
        hdf5_file: HDF5 file saving image data
        transform: 
        """
        self.df = pd.read_csv(csv_file, low_memory=False)
        self.hdf5_fp = h5py.File(hdf5_file, mode="r")
        self.transform = transform
        # Take our isic_ids and target
        self.isic_ids = self.df['isic_id'].values
        # If 'target' column doesn't exist, fill with zeros.
        if 'target' in self.df.columns:
            self.targets = self.df['target'].values
        else:
            self.targets = np.zeros(len(self.df))

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        isic_id = self.isic_ids[idx]
        label = self.targets[idx]
        # from HDF5 read image data and transger to PIL Image
        image_bytes = self.hdf5_fp[isic_id][()]
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label



train_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # Normalizaiton, using ImageNet's mean and standard deviation
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation
val_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


train_hdf5_path = "/kaggle/input/isic-2024-challenge/train-image.hdf5"


train_dataset = SkinDataset(
    csv_file=train_csv_path,
    hdf5_file=train_hdf5_path,
    transform=train_transform
)

# Randomly seperate some data as validation data
val_split = 0.2
num_samples = len(train_dataset)
num_val = int(val_split * num_samples)
num_train = num_samples - num_val

train_subset, val_subset = torch.utils.data.random_split(train_dataset, [num_train, num_val])

batch_size = 40
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Num_train: {num_train}, Num_val: {num_val}")



def display_random_images(dataset: Dataset, n: int = 5, seed: int = None):
    if seed is not None:
        random.seed(seed)
    indices = random.sample(range(len(dataset)), n)
    
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        # image is a Tensor, convert the channel dimension back to the last dimension for matplotlib display
        img_np = image.permute(1, 2, 0).cpu().numpy()
        # denormalize the normalized image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        plt.subplot(1, n, i+1)
        plt.imshow(img_np)
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.show()


display_random_images(train_dataset, n=5, seed=42)


# Pick the first data and show
sample_id = train_metadata.iloc[0]['isic_id']
print("First sample's isic_id:", sample_id)

with h5py.File(train_hdf5_path, mode="r") as hf:
    if sample_id in hf:
        image_bytes = hf[sample_id][()]
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.title("Original Image In HDF5")
        plt.axis("off")
        plt.show()
    else:
        print(f"{sample_id} not in HDF5 file")


# Change pretrained to False for submission
model = timm.create_model('tf_efficientnet_b4_ns', pretrained=False, num_classes=1)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)   # For more than one GPU
model = model.to(device)

# print(model)


# Loss function and optimize
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

print("Train samples:", len(train_loader.dataset))
print("Validation samples:", len(val_loader.dataset))


# Train
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            if i % 500 == 0:
                print(f"Batch {i}")
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)

        # evaluate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate the accuracy rate
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        # Learning rate scheduler
        scheduler.step()

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")

    print("Fininshed Training!")


# train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5)


# Load test data
test_hdf5_path = "/kaggle/input/isic-2024-challenge/test-image.hdf5"
test_dataset = SkinDataset(test_csv_path, test_hdf5_path, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model's weights
if os.path.exists("best_model.pth"):
    weights_path = "best_model.pth"
else:
    weights_path = "/kaggle/input/modelweights/best_model.pth"

# predict
model.load_state_dict(torch.load(weights_path))
model.eval()

predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        predictions.extend(probs)

# Create submission file
submission = pd.DataFrame({"isic_id": test_dataset.isic_ids, "target": predictions})
submission.to_csv("submission.csv", index=False)
print("Saved!")


