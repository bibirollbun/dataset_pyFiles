!unzip -q /kaggle/input/dogs-vs-cats/train.zip


import os
import shutil
import random

# Define paths
BASE_DIR = "/kaggle/working"
SOURCE_DIR = os.path.join(BASE_DIR, "train")  # Folder with all images
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

# Ensure required directories exist
for category in ["cat", "dog"]:
    os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, category), exist_ok=True)

# Organize images into "cat/" and "dog/" folders
for filename in os.listdir(SOURCE_DIR):
    if filename.startswith("cat"):
        shutil.move(os.path.join(SOURCE_DIR, filename), os.path.join(TRAIN_DIR, "cat", filename))
    elif filename.startswith("dog"):
        shutil.move(os.path.join(SOURCE_DIR, filename), os.path.join(TRAIN_DIR, "dog", filename))

print("Dataset structured successfully!")

# Split function
def split_data(category, train_ratio=0.8):
    source_path = os.path.join(TRAIN_DIR, category)
    images = [f for f in os.listdir(source_path) if f.endswith(".jpg")]
    random.shuffle(images)

    train_count = int(train_ratio * len(images))
    train_images = images[:train_count]
    val_images = images[train_count:]

    # Move images to train and val folders
    for img in val_images:
        shutil.move(os.path.join(source_path, img), os.path.join(VAL_DIR, category, img))

# Apply split for both categories
split_data("cat")
split_data("dog")

print("Dataset split successfully!")


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bars

# Paths
TRAIN_PATH = "./data/train"
VAL_PATH = "./data/val"

# Training parameters
NUM_BATCH = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

# Load datasets
train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
val_dataset = datasets.ImageFolder(root=VAL_PATH, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=NUM_BATCH, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=NUM_BATCH, shuffle=False)

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
print(model.fc.in_features)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the fully connected layer
model.fc = nn.Linear(in_features=512, out_features=2)

# Ensure the new `fc` layer is trainable
for param in model.fc.parameters():
    param.requires_grad = True

# Move model to device
model = model.to(DEVICE)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# Training and validation loop
for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss, correct, total = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Training", leave=True)
    
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update tqdm progress bar
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

    # Validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Validation", leave=True)

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update tqdm progress bar
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    val_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

print("Training complete!")


torch.save(model.state_dict(), "ResNet18_CatDog_224_Norm.pth")




