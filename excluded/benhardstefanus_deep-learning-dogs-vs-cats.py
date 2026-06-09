import os
import zipfile

# Directory containing ZIP files
zip_dir = "/kaggle/input/dogs-vs-cats"
extract_path = "/kaggle/working/"

# Extract all ZIP files in the directory
for file in os.listdir(zip_dir):
    if file.endswith(".zip"):
        with zipfile.ZipFile(os.path.join(zip_dir, file), 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print(f"Extracted: {file}")



import os
import zipfile
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import Counter
from PIL import Image

# Paths to dataset
train_dir = "/kaggle/input/dogs-vs-cats/train.zip"
extract_path = "/kaggle/working"

# Extract files if not already extracted
if not os.path.exists(extract_path):
    with zipfile.ZipFile(train_dir, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Define dataset path
train_path = os.path.join(extract_path, "train")

# Get list of all images
all_images = os.listdir(train_path)

# Count number of images for each class
labels = [img.split(".")[0] for img in all_images]  # Extract "dog" or "cat"
label_counts = Counter(labels)

print("Number of Dog images:", label_counts["dog"])
print("Number of Cat images:", label_counts["cat"])

# Display a few sample images
fig, axes = plt.subplots(3, 4, figsize=(12, 9))  # 3 rows, 4 columns
fig.suptitle("Sample Images from Dataset", fontsize=14)

for i, ax in enumerate(axes.flat):
    img_path = os.path.join(train_path, all_images[i])
    img = Image.open(img_path)  # Open image using PIL
    ax.imshow(img)
    ax.set_title(f"Class: {labels[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# Check image sizes
image_sizes = []
for img_name in all_images[:200]:  # Checking first 200 images
    img_path = os.path.join(train_path, img_name)
    img = Image.open(img_path)
    image_sizes.append(img.size)

# Count unique image sizes
unique_sizes = Counter(image_sizes)
print("Unique Image Sizes:", unique_sizes)

# Display Image Size Distribution
sizes, counts = zip(*unique_sizes.items())
plt.figure(figsize=(10, 5))
plt.bar(range(len(sizes)), counts, tick_label=[f"{w}x{h}" for w, h in sizes])
plt.xlabel("Image Resolution")
plt.ylabel("Count")
plt.title("Distribution of Image Sizes in Dataset")
plt.xticks(rotation=45)
plt.show()

# Check for corrupted images
corrupted_files = []
for img_name in all_images:
    img_path = os.path.join(train_path, img_name)
    try:
        img = Image.open(img_path)
        img.verify()  # Verify if image is valid
    except Exception as e:
        corrupted_files.append(img_name)

print("Number of Corrupted Images:", len(corrupted_files))
if corrupted_files:
    print("Sample Corrupted Image:", corrupted_files[:5])  # Show first few corrupted images



import os
import glob
import time
import random  import os
import glob
import time
import random  
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models  # <-- Added import
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
extract_path = "/kaggle/working/"
train_path = os.path.join(extract_path, "train")  

# Load all images and split them into cats and dogs
all_images = glob.glob(os.path.join(train_path, "*.jpg"))
cat_images = [img for img in all_images if "cat" in os.path.basename(img)]
dog_images = [img for img in all_images if "dog" in os.path.basename(img)]

# Reduce dataset size by 30% (keep only 70%)
num_cats = int(len(cat_images) * 0.3)
num_dogs = int(len(dog_images) * 0.3)

# Randomly select 70% of images
selected_cat_images = random.sample(cat_images, num_cats)
selected_dog_images = random.sample(dog_images, num_dogs)

# Combine selected images for training
reduced_dataset = selected_cat_images + selected_dog_images
random.shuffle(reduced_dataset)  
print(f"Reduced dataset size: {len(reduced_dataset)} images (Cats: {num_cats}, Dogs: {num_dogs})")

# Define transforms (optimized with smaller image size)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ====== Custom Dataset Loader ======
class CatsDogsDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        self.labels = [1 if "dog" in os.path.basename(f) else 0 for f in self.image_list]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# Load dataset with reduced size
dataset = CatsDogsDataset(reduced_dataset, transform=transform)

# Split dataset into training (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

# ===== REPLACE CNN WITH RESNET-18 =====
# We remove the custom CNNModel and use ResNet-18 with a modified final layer.
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ====== Training Loop ======
num_epochs = 10  
train_losses = []

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)  

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

# Total training time
total_time = time.time() - start_time
print(f"Total Training Time: {total_time:.2f} seconds")

# Save model
torch.save(model.state_dict(), "cats_vs_dogs_cnn.pth")

# ====== Model Evaluation ======
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Generate Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# ====== Compare Predictions vs Actual ======
fig, axes = plt.subplots(3, 4, figsize=(12, 9))  

for i, ax in enumerate(axes.flat):
    img_path = val_dataset[i][0].permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy
    ax.imshow(img_path)
    actual_label = "Dog" if all_labels[i] == 1 else "Cat"
    predicted_label = "Dog" if all_preds[i] == 1 else "Cat"
    ax.set_title(f"Actual: {actual_label}\nPredicted: {predicted_label}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
extract_path = "/kaggle/working/"
train_path = os.path.join(extract_path, "train")  

# Load all images and split them into cats and dogs
all_images = glob.glob(os.path.join(train_path, "*.jpg"))
cat_images = [img for img in all_images if "cat" in os.path.basename(img)]
dog_images = [img for img in all_images if "dog" in os.path.basename(img)]

# Reduce dataset size by 30% (keep only 70%)
num_cats = int(len(cat_images) * 0.3)
num_dogs = int(len(dog_images) * 0.3)

# Randomly select 70% of images
selected_cat_images = random.sample(cat_images, num_cats)
selected_dog_images = random.sample(dog_images, num_dogs)

# Combine selected images for training
reduced_dataset = selected_cat_images + selected_dog_images
random.shuffle(reduced_dataset)  
print(f"Reduced dataset size: {len(reduced_dataset)} images (Cats: {num_cats}, Dogs: {num_dogs})")

# Define transforms (optimized with smaller image size)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ====== Custom Dataset Loader ======
class CatsDogsDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        self.labels = [1 if "dog" in os.path.basename(f) else 0 for f in self.image_list]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# Load dataset with reduced size
dataset = CatsDogsDataset(reduced_dataset, transform=transform)

# Split dataset into training (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

# ===== Define CNN Model =====
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 32 * 32, 256),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  
        x = self.fc_layers(x)
        return x

# Initialize model
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ====== Training Loop ======
num_epochs = 10  
train_losses = []

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)  

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

# Total training time
total_time = time.time() - start_time
print(f"Total Training Time: {total_time:.2f} seconds")

# Save model
torch.save(model.state_dict(), "cats_vs_dogs_cnn.pth")

# ====== Model Evaluation ======
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Generate Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# ====== Compare Predictions vs Actual ======
fig, axes = plt.subplots(3, 4, figsize=(12, 9))  

for i, ax in enumerate(axes.flat):
    img_path = val_dataset[i][0].permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy
    ax.imshow(img_path)
    actual_label = "Dog" if all_labels[i] == 1 else "Cat"
    predicted_label = "Dog" if all_preds[i] == 1 else "Cat"
    ax.set_title(f"Actual: {actual_label}\nPredicted: {predicted_label}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()



import os
import glob
import time
import random  
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models  # <-- Added import
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
extract_path = "/kaggle/working/"
train_path = os.path.join(extract_path, "train")  

# Load all images and split them into cats and dogs
all_images = glob.glob(os.path.join(train_path, "*.jpg"))
cat_images = [img for img in all_images if "cat" in os.path.basename(img)]
dog_images = [img for img in all_images if "dog" in os.path.basename(img)]

# Reduce dataset size by 30% (keep only 70%)
num_cats = int(len(cat_images) * 0.3)
num_dogs = int(len(dog_images) * 0.3)

# Randomly select 70% of images
selected_cat_images = random.sample(cat_images, num_cats)
selected_dog_images = random.sample(dog_images, num_dogs)

# Combine selected images for training
reduced_dataset = selected_cat_images + selected_dog_images
random.shuffle(reduced_dataset)  
print(f"Reduced dataset size: {len(reduced_dataset)} images (Cats: {num_cats}, Dogs: {num_dogs})")

# Define transforms (optimized with smaller image size)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ====== Custom Dataset Loader ======
class CatsDogsDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        self.labels = [1 if "dog" in os.path.basename(f) else 0 for f in self.image_list]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# Load dataset with reduced size
dataset = CatsDogsDataset(reduced_dataset, transform=transform)

# Split dataset into training (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

# ===== REPLACE CNN WITH RESNET-18 =====
# We remove the custom CNNModel and use ResNet-18 with a modified final layer.
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ====== Training Loop ======
num_epochs = 10  
train_losses = []

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)  

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

# Total training time
total_time = time.time() - start_time
print(f"Total Training Time: {total_time:.2f} seconds")

# Save model
torch.save(model.state_dict(), "cats_vs_dogs_cnn.pth")

# ====== Model Evaluation ======
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Generate Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# ====== Compare Predictions vs Actual ======
fig, axes = plt.subplots(3, 4, figsize=(12, 9))  

for i, ax in enumerate(axes.flat):
    img_path = val_dataset[i][0].permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy
    ax.imshow(img_path)
    actual_label = "Dog" if all_labels[i] == 1 else "Cat"
    predicted_label = "Dog" if all_preds[i] == 1 else "Cat"
    ax.set_title(f"Actual: {actual_label}\nPredicted: {predicted_label}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()


