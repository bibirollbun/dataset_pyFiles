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


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import timm  # Import timm for EfficientNet models




# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class DateDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Whether this is the test dataset or not.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
        if not is_test:
            self.labels = sorted(self.data_frame['label'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
            self.idx_to_label = {idx: label for idx, label in enumerate(self.labels)}
            print(f"Classes: {self.labels}")
            print(f"Number of classes: {len(self.labels)}")
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, self.data_frame.iloc[idx, 0]
        else:
            label = self.data_frame.iloc[idx, 1]
            label_idx = self.label_to_idx[label]
            return image, label_idx

def get_transforms():
    """Define transformations for EfficientNet-B3 (recommended input size: 300x300)."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(300),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    
    val_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(num_classes, model_variant='efficientnet_b3', dropout_rate=0.3, freeze_ratio=0.3):
    """
    Create a pretrained EfficientNet model with a custom classifier using PyTorch.
    
    Args:
        num_classes (int): Number of output classes
        model_variant (str): EfficientNet variant (b0-b7, default: 'efficientnet_b3')
        dropout_rate (float): Dropout probability (default: 0.3)
        freeze_ratio (float): Proportion of layers to freeze (0 to 1, default: 0.3)
    """
    # Model variant mapping
    model_dict = {
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'efficientnet_b5': models.efficientnet_b5,
        'efficientnet_b6': models.efficientnet_b6,
        'efficientnet_b7': models.efficientnet_b7
    }
    
    if model_variant not in model_dict:
        raise ValueError(f"Unsupported variant. Choose from {list(model_dict.keys())}")
    
    # Load pretrained model
    model = model_dict[model_variant](weights='IMAGENET1K_V1')
    
    # Get the number of features from the classifier
    in_features = model.classifier[1].in_features
    
    # Replace the classifier with an enhanced version
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, in_features // 2),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(in_features // 2),
        nn.Dropout(p=dropout_rate/2),
        nn.Linear(in_features // 2, num_classes)
    )
    
    # Dynamic layer freezing
    total_params = sum(1 for _ in model.named_parameters())
    freeze_count = int(total_params * freeze_ratio)
    params_frozen = 0
    
    for name, param in model.named_parameters():
        if params_frozen < freeze_count and 'classifier' not in name:
            param.requires_grad = False
            params_frozen += 1
        else:
            param.requires_grad = True
            
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    """Train the model and validate it."""
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_all_preds = []
        val_all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = accuracy_score(val_all_labels, val_all_preds)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        
        scheduler.step(val_epoch_loss)
        
        if val_epoch_acc > best_val_acc and epoch_acc > 0.7:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), "best_date_classifier.pth")
            print("Saved new best model with accuracy:", best_val_acc)
    
    model.load_state_dict(torch.load("best_date_classifier.pth"))
    return model

def predict(model, test_loader, dataset):
    """Generate predictions for test data."""
    model.eval()
    all_filenames = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            pred_labels = [dataset.idx_to_label[idx.item()] for idx in preds]
            
            all_filenames.extend(filenames)
            all_predictions.extend(pred_labels)
    
    return pd.DataFrame({'filename': all_filenames, 'label': all_predictions})

# Define paths
train_csv = "/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv"
train_dir = "/kaggle/input/open-data-day-2025-dates-types-classification/train"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get transformations
train_transform, val_transform = get_transforms()

# Create dataset
full_dataset = DateDataset(train_csv, train_dir, transform=train_transform)
labels = full_dataset.data_frame['label'].values

# Perform stratified split
train_indices, val_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.3,
    stratify=labels,
    random_state=42
)

train_dataset = torch.utils.data.Subset(
    DateDataset(img_dir=train_dir, csv_file=train_csv, transform=train_transform),
    train_indices
)
val_dataset = torch.utils.data.Subset(
    DateDataset(img_dir=train_dir, csv_file=train_csv, transform=val_transform),
    val_indices
)

# Create data loaders
batch_size = 35
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Create model
# num_classes = len(full_dataset.labels)
# model = create_model(num_classes=num_classes , model_variant='efficientnet_b6')  # Using default efficientnet_b3



# Initial training
model = create_model(num_classes, freeze_ratio=0.3 ,  model_variant='efficientnet_b6')
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15)





# Fine-tune more layers
model = create_model(num_classes, freeze_ratio=0.3 ,  model_variant='efficientnet_b6')
model.load_state_dict(torch.load("/kaggle/working/best_date_classifier.pth"))
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)



test_dir = "/kaggle/input/open-data-day-2025-dates-types-classification/test"
testfilenames = sorted(os.listdir(test_dir))


# Creating a DataFrame
df = pd.DataFrame(testfilenames, columns=['Name'])

# Saving to CSV
df.to_csv('/kaggle/working/testlabels.csv', index=False)


test_label ='/kaggle/working/testlabels.csv'


# Test data
test_dataset = DateDataset(test_label, test_dir, transform=val_transform, is_test=True)
test_dataset.idx_to_label = full_dataset.idx_to_label  # Copy the label mapping
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
model_path = "/kaggle/working/best_date_classifier.pth"
# Recreate the model architecture
num_classes = len(full_dataset.labels)  # Ensure you have the correct number of classes
model = create_model(num_classes, model_variant='efficientnet_b6')  # Create model instance

# Load the trained weights (state_dict) properly
model.load_state_dict(torch.load(model_path, map_location=device))

# Move to device
model.to(device)
model.eval()  # Set to evaluation mode

# Generate predictions
predictions = predict(model, test_loader, full_dataset)

# Save predictions
predictions.to_csv("submission.csv", index=False)
print(f"Predictions saved to submission.csv")





