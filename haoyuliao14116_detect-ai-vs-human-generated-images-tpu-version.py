!pip install pretrainedmodels
!pip install albumentations
# for TPU
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev


import torch, time, os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.vision_transformer import vit_b_16
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from albumentations import Rotate
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp



# Get TPU device
device = xm.xla_device()
print(f"Using device: {device}")

# Check all TPU devices
tpu_cores = xm.get_xla_supported_devices()
print(f"Available TPU cores: {tpu_cores}")


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path='best_model.pth', log_file='training_log.txt'):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            xm.optimizer_step(optimizer, barrier=True)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        end_time = time.time()
        epoch_time = end_time - start_time
        log_message = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.2f} sec"
        print(log_message)
        with open(log_file, 'a') as f:
            f.write(log_message + '')
            
        # Save best model
        if val_acc > best_val_acc and val_loss < best_val_loss:
            best_val_acc, best_val_loss = val_acc, val_loss
            model_filename = f"best_model_epoch{epoch+1}_acc{best_val_acc:.2f}_loss{best_val_loss:.4f}.pth"
            torch.save(model.state_dict(), model_filename)
            log_message = f"Best model saved with Val Acc: {best_val_acc:.2f}% and Val Loss:{best_val_loss}"
            print(log_message)
            with open(log_file, 'a') as f:
                f.write(log_message + '')
    
    # Plot loss and accuracy
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(range(1, num_epochs+1), train_accs, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.savefig('training_validation_loss.png')
    plt.savefig('training_validation_accuracy.png')
    plt.show()



# Define transformations (Data Augmentation & Normalization)
class DeepfakeDataset(Dataset):
    def __init__(self, excel_file, is_training=True):
        self.data = pd.read_csv(excel_file, header=0)
        self.image_paths = self.data.iloc[:, 1].values  # Column 1: Image paths
        self.labels = self.data.iloc[:, 2].values  # Column 2: Labels (0 or 1)
        self.is_training = is_training  # Flag to enable/disable augmentation

        # Define transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = "/kaggle/input/ai-vs-human-generated-dataset/"+self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.is_training:
            image = self.train_transform(image)
        else:
            image = self.test_transform(image)
        
        return image, label
        
# Load dataset from Excel file
EXCEL_FILE = "/kaggle/input/ai-vs-human-generated-dataset/train.csv"
full_dataset = DeepfakeDataset(EXCEL_FILE, is_training=True)


# Ensure the dataset has at least some samples
dataset_size = len(full_dataset)
assert dataset_size > 0, "Dataset is empty. Check the Excel file!"

# Define split sizes
train_size = int(0.7 * dataset_size)  # 60% for training
val_size = int(0.2 * dataset_size)    # 20% for validation
test_size = dataset_size - (train_size + val_size)  # Remaining for testing

# Split dataset correctly
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Apply transformations correctly
train_dataset.dataset.is_training = True  # Enable augmentations
val_dataset.dataset.is_training = False   # Disable augmentations
test_dataset.dataset.is_training = False  # Disable augmentations

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# Confirm dataset sizes
print(f"Total Dataset Size: {dataset_size}")
print(f"Training Set Size: {len(train_dataset)}")
print(f"Validation Set Size: {len(val_dataset)}")
print(f"Testing Set Size: {len(test_dataset)}")

# Load Vision Transformer model
model = vit_b_16(pretrained=True)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, 2)  # 2 classes: Real & Fake
model.to(device)
print(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='binary') * 100
    recall = recall_score(all_labels, all_preds, average='binary') * 100
    f1 = f1_score(all_labels, all_preds, average='binary') * 100
    
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Test Precision: {precision:.2f}%")
    print(f"Test Recall: {recall:.2f}%")
    print(f"Test F1 Score: {f1:.2f}%")

evaluate_model(model, test_loader)


import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

"""
# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vit_b_16(pretrained=False)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, 2)  # Adjust based on classes
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
"""
device = xm.xla_device()
model.to(device)
model.eval()


# Define transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
unknown_data = pd.read_csv("/kaggle/input/ai-vs-human-generated-dataset/test.csv", header=0)

res_pred = []
for idx in range(len(unknown_data)):
    image_path = os.path.join("/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/", 
                              os.path.basename(unknown_data['id'].iloc[idx]))

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)  # Get predicted class
        res_pred.append(predicted.item())  # Store prediction

# Save predictions
unknown_data['label'] = res_pred
unknown_data.to_csv("test.csv", index=False)
print("Predictions saved to test.csv")







