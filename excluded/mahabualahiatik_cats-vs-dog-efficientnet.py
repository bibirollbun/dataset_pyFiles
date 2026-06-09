import zipfile
import os

# Path to the ZIP file
zip_path = '/kaggle/input/dogs-vs-cats/train.zip'
extract_dir = '/kaggle/working'

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)




import os

# Check the root extraction path
root_dir = '/kaggle/working'
train_dir = os.path.join(root_dir, 'train')
test_dir= os.path.join(root_dir, 'test1')

# Count and preview files
all_files = os.listdir(train_dir)
print(f"Total files in 'train/': {len(all_files)}")

# Separate by class for info
cat_files = [f for f in all_files if f.startswith('cat')]
dog_files = [f for f in all_files if f.startswith('dog')]

print(f"Number of cat images: {len(cat_files)}")
print(f"Number of dog images: {len(dog_files)}")
print("\nSample filenames:")
print("Cats:", cat_files[:5])
print("Dogs:", dog_files[:5])






import matplotlib.pyplot as plt
from PIL import Image
import random


# Get file lists
cat_files = [f for f in os.listdir(train_dir) if f.startswith('cat')]
dog_files = [f for f in os.listdir(train_dir) if f.startswith('dog')]

# Randomly select 5 images from each class
sample_cats = random.sample(cat_files, 5)
sample_dogs = random.sample(dog_files, 5)

# Plot cats
plt.figure(figsize=(15, 3))
for i, file in enumerate(sample_cats):
    img = Image.open(os.path.join(train_dir, file))
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title("Cat")
    plt.axis('off')
plt.suptitle("Sample Cat Images", fontsize=16)
plt.show()

# Plot dogs
plt.figure(figsize=(15, 3))
for i, file in enumerate(sample_dogs):
    img = Image.open(os.path.join(train_dir, file))
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title("Dog")
    plt.axis('off')
plt.suptitle("Sample Dog Images", fontsize=16)
plt.show()



import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# Constants
DATA_DIR = '/kaggle/working/train'
IMG_SIZE = 224  # EfficientNetB0 prefers 224x224

# Define preprocessing transforms
efficientnet_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Imagenet mean
        std=[0.229, 0.224, 0.225]    # Imagenet std
    )
])

# Custom Dataset
class CatDogDataset(Dataset):
    def __init__(self, file_list, root_dir, transform=None):
        self.file_list = file_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Label extraction from filename
        label = 1 if 'dog' in img_name else 0

        if self.transform:
            image = self.transform(image)

        return image, label

# Prepare file list and split
all_images = os.listdir(DATA_DIR)
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))  # 80% train, 20% val
train_files = all_images[:split_idx]
val_files = all_images[split_idx:]

# Dataset & DataLoader
train_dataset = CatDogDataset(train_files, DATA_DIR, transform=efficientnet_transforms)
val_dataset = CatDogDataset(val_files, DATA_DIR, transform=efficientnet_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)




print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")




from collections import Counter

# Get all labels from training dataset
train_labels = [label for _, label in train_dataset]
label_counts = Counter(train_labels)

# Print label distribution
print("Label distribution in training dataset:")
print(f"Cats (label 0): {label_counts[0]}")
print(f"Dogs (label 1): {label_counts[1]}")



import matplotlib.pyplot as plt

# Function to visualize transformed images from the dataset
def show_transformed_images(dataset, num_images=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        idx = random.randint(0, len(dataset) - 1)
        img_tensor, label = dataset[idx]
        
        # Convert tensor to numpy image for display
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Unnormalize
        img = img.clip(0, 1)
        
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title("Dog" if label == 1 else "Cat")
        plt.axis('off')
    plt.suptitle("Transformed Sample Images from Training Set", fontsize=16)
    plt.show()

# Show images
show_transformed_images(train_dataset)






import torch
import torch.nn as nn
import torchvision.models as models

# Load EfficientNet-B0 with pretrained weights
model = models.efficientnet_b0(pretrained=True)

# Freeze feature extractor if desired (optional)
for param in model.features.parameters():
    param.requires_grad = True  # Set to False if you want to fine-tune only classifier

# Modify the classifier for binary classification
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

model = model.to('cuda' if torch.cuda.is_available() else 'cpu')



device = 'cuda' if torch.cuda.is_available() else 'cpu'


from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

# Loss, optimizer, and scheduler
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

# Early stopping settings
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 5  # stop if no improvement in 5 epochs

# Track loss and accuracy
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []



EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss_total, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = val_loss_total / len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Scheduler step
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')  # Save best model
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break



import matplotlib.pyplot as plt

# Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()






# Load best model weights
model.load_state_dict(torch.load('/kaggle/input/cats-vs-dog-efficientnet/best_model.pt', map_location=device))
model.eval()



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import numpy as np

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.numpy())

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"ðŸ“Š Validation Set Evaluation:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Cat', 'Dog']))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix on Validation Set")
plt.show()














