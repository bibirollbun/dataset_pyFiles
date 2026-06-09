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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import zipfile
from tqdm import tqdm
import random



TRAIN_ZIP = '/kaggle/input/dogs-vs-cats/train.zip'
TEST_ZIP  = '/kaggle/input/dogs-vs-cats/test1.zip'
WORK_DIR  = '/kaggle/working/dogs-vs-cats'

os.makedirs(WORK_DIR, exist_ok=True)

def safe_unzip(zip_path, extract_to):
    print('Unzipping', zip_path, '->', extract_to)
    with zipfile.ZipFile(zip_path,'r') as z:
        z.extractall(extract_to)
if os.path.exists(TRAIN_ZIP):
    safe_unzip(TRAIN_ZIP, WORK_DIR)
else:
    print('Train zip not found at', TRAIN_ZIP)
if os.path.exists(TEST_ZIP):
    safe_unzip(TEST_ZIP, WORK_DIR)
else:
    print('Test zip not found at', TEST_ZIP)

possible_train_dirs = [
    os.path.join(WORK_DIR, 'train'),
    WORK_DIR
]
train_found = None
for d in possible_train_dirs:
    if os.path.isdir(d) and any(p.suffix.lower() == '.jpg' for p in pathlib.Path(d).glob('*.jpg')):
        train_found = d
        break

if train_found is None:
    for root, dirs, files in os.walk(WORK_DIR):
        if any(f.lower().endswith('.jpg') for f in files):
            train_found = root
            break

if train_found is None:
    raise FileNotFoundError('Could not find train images after unzipping. Looked in: ' + str(possible_train_dirs))
print('Train images found in:', train_found)


# Cek struktur folder
train_dir = '/kaggle/working/dogs-vs-cats/train'
test_dir = '/kaggle/working/dogs-vs-cats/test'

train_files = os.listdir(train_dir)
print(f"Total training images: {len(train_files)}")

# Hitung jumlah gambar per kelas
cats = [f for f in train_files if 'cat' in f]
dogs = [f for f in train_files if 'dog' in f]
print(f"Cats: {len(cats)}")
print(f"Dogs: {len(dogs)}")

# Tampilkan beberapa contoh nama file
print("\nContoh nama file:")
print(train_files[:5])


# Visualisasi beberapa gambar
def show_images(image_list, n=8):
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for i, img_name in enumerate(random.sample(image_list, n)):
        img_path = os.path.join(train_dir, img_name)
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(img_name.split('.')[0].upper())
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

show_images(train_files)


# Custom Dataset Class
class DogsVsCatsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Label: 0 untuk cat, 1 untuk dog
        label = 0 if 'cat' in img_name else 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data Augmentation dan Preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Split data menjadi train dan validation (80:20)
from sklearn.model_selection import train_test_split

all_files = os.listdir(train_dir)
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

# Create temporary directories for train and val
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/val', exist_ok=True)

# Copy files (atau bisa gunakan symlink untuk hemat space)
import shutil

for f in tqdm(train_files, desc='Copying train files'):
    shutil.copy(os.path.join(train_dir, f), os.path.join('data/train', f))

for f in tqdm(val_files, desc='Copying val files'):
    shutil.copy(os.path.join(train_dir, f), os.path.join('data/val', f))

print(f"Train samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")


# Create datasets and dataloaders
batch_size = 32

train_dataset = DogsVsCatsDataset('data/train', transform=train_transform)
val_dataset = DogsVsCatsDataset('data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# Menggunakan Pre-trained ResNet18 dengan Transfer Learning
class DogCatClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DogCatClassifier, self).__init__()
        self.model = models.resnet18(weights=True)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DogCatClassifier(num_classes=2).to(device)
print(model)


# Loss function dan optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Training parameters
num_epochs = 15
best_val_acc = 0.0

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# Training loop
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 50)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Learning rate scheduler
    scheduler.step(val_loss)
    
    print(f'\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, 'best_model.pth')
        print(f'✓ Saved best model with accuracy: {val_acc:.2f}%')

print('\n' + '='*50)
print(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%')
print('='*50)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss
ax1.plot(history['train_loss'], label='Train Loss', marker='o')
ax1.plot(history['val_loss'], label='Val Loss', marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Accuracy
ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
ax2.plot(history['val_acc'], label='Val Accuracy', marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()


# Load best model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Getting predictions'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

# Get predictions
preds, labels = get_predictions(model, val_loader, device)

# Confusion matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Classification report
print('\nClassification Report:')
print(classification_report(labels, preds, target_names=['Cat', 'Dog']))


# Visualize predictions
def visualize_predictions(model, dataset, device, n=8):
    model.eval()
    indices = random.sample(range(len(dataset)), n)
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    class_names = ['Cat', 'Dog']
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            
            output = model(image_tensor)
            prob = torch.softmax(output, dim=1)
            pred_class = output.argmax(1).item()
            confidence = prob[0][pred_class].item() * 100
            
            # Denormalize image for display
            img = image.cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            color = 'green' if pred_class == label else 'red'
            axes[i].set_title(f'True: {class_names[label]}\nPred: {class_names[pred_class]} ({confidence:.1f}%)', 
                            color=color, fontweight='bold')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_sample.png', dpi=300, bbox_inches='tight')
    plt.show()

visualize_predictions(model, val_dataset, device)

