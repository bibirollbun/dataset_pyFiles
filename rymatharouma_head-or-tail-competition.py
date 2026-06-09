import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()


from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

# load the VGG16 network *pre-trained* on the ImageNet dataset
weights = VGG16_Weights.DEFAULT
vgg_model = vgg16(weights=weights)


vgg_model.to(device)
vgg_model.requires_grad_(False)
print("VGG16 Frozen")


N_CLASSES = 1

my_model = nn.Sequential(
    vgg_model,
    nn.Linear(1000, N_CLASSES)
)

my_model.to(device)


for idx, param in enumerate(my_model.parameters()):
    print(idx, param.requires_grad)


loss_function = nn.BCEWithLogitsLoss()
optimizer = Adam(my_model.parameters())
my_model = my_model.to(device)


pre_trans = weights.transforms()


import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import glob
from PIL import Image
from sklearn.model_selection import train_test_split  # Import for stratified split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations including pre_trans for normalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(45),  
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    # Assuming pre_trans includes normalization as its final step
])

# For validation, we typically use simpler transformations (no data augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Same normalization as in pre_trans would be here
])

class TransformedDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Custom dataset that takes already loaded images and labels
        
        Args:
            images: List of already processed images
            labels: List of corresponding labels
            transform: Optional transform to apply (for dynamic transformation)
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Apply transforms if needed (useful for train-time augmentation)
        if self.transform and not isinstance(img, torch.Tensor):
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return len(self.images)

# Data collection class to prepare for stratified split
class DataCollector:
    def __init__(self, data_dir, class_labels=None):
        self.images = []
        self.labels = []
        
        if class_labels is None:
            class_labels = ["heads", "tails"]  # Default labels
            
        # Collect all image paths and labels for later processing
        for l_idx, label in enumerate(class_labels):
            data_paths = glob.glob(data_dir + '/' + label + '/*.jpg', recursive=True)
            for path in data_paths:
                img = Image.open(path)
                self.images.append(img)
                self.labels.append(l_idx)
                
        print(f"Dataset collected: {len(self.images)} images")
        print(f"Labels distribution: {self._get_label_distribution()}")
                
    def _get_label_distribution(self):
        counts = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts
    
    def get_split_datasets(self, train_transform, val_transform, test_size=0.2, random_state=42):
        """
        Split dataset with stratification and apply appropriate transforms
        """
        # Stratified split to ensure class balance in both train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.images, self.labels, 
            test_size=test_size, 
            stratify=self.labels,  # This ensures both sets have similar class distributions
            random_state=random_state
        )
        
        # Create datasets with appropriate transforms
        train_processed = self._process_images(X_train, y_train, train_transform)
        val_processed = self._process_images(X_val, y_val, val_transform)
        
        train_dataset = TransformedDataset(*train_processed)
        val_dataset = TransformedDataset(*val_processed)
        
        return train_dataset, val_dataset
    
    def _process_images(self, images, labels, transform):
        """Process images with the given transform"""
        processed_images = []
        processed_labels = []
        
        for img, label in zip(images, labels):
            if transform:
                img = transform(img).to(device)
            else:
                img = pre_trans(img).to(device)  # Use pre_trans if no transform provided
                
            processed_images.append(img)
            processed_labels.append(torch.tensor(label).to(device).float())
            
        return processed_images, processed_labels

# Load and prepare datasets
train_dir = '/kaggle/input/heads-or-tails-image-classification/train'

# Create data collector and get stratified splits
data_collector = DataCollector(train_dir)
train_dataset, val_dataset = data_collector.get_split_datasets(train_transform, val_transform)

# Create data loaders
loader_train = DataLoader(train_dataset, shuffle=True, batch_size=16)
loader_val = DataLoader(val_dataset, shuffle=False, batch_size=16)

# Check shapes and distributions
try:
    train_images, train_labels = next(iter(loader_train))
    print("Training batch shape:", train_images.shape)
    
    val_images, val_labels = next(iter(loader_val))
    print("Validation batch shape:", val_images.shape)
    
    # Count labels in train and validation sets
    def count_labels(loader):
        counts = {}
        for _, labels in loader:
            for label in labels:
                l = label.item()
                counts[l] = counts.get(l, 0) + 1
        return counts
    
    train_dist = count_labels(loader_train)
    val_dist = count_labels(loader_val)
    
    print(f"Training set label distribution: {train_dist}")
    print(f"Validation set label distribution: {val_dist}")
    
    # Print dataset sizes
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    print("Data loading successful!")
except Exception as e:
    print(f"Error loading batch: {e}")


train_N = len(loader_train.dataset)
valid_N = len(loader_val.dataset)


torch.cuda.empty_cache()


def get_batch_accuracy(output, y, N):
    zero_tensor = torch.tensor([0]).to(device)
    pred = torch.gt(output, zero_tensor)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def train(model, check_grad=False):
    loss = 0
    accuracy = 0

    model.train()
    for x, y in loader_train:
        output = torch.squeeze(model(x))
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    if check_grad:
        print('Last Gradient:')
        for param in model.parameters():
            print(param.grad)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))


import torch
from torchmetrics.classification import BinaryAUROC

def validate(model):
    loss = 0
    accuracy = 0
    
    # Initialize AUC metric
    device = next(model.parameters()).device  # Get the device from the model
    auroc = BinaryAUROC().to(device)  # Move metric to same device as model
    
    model.eval()
    with torch.no_grad():
        for x, y in loader_val:
            output = torch.squeeze(model(x))
            
            # Update AUC metric with batch predictions and labels
            auroc.update(output, y.int())
            
            # Calculate batch loss and accuracy
            loss += loss_function(output, y.float()).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    
    # Compute final AUC
    auc_score = auroc.compute().item()
    
    print('Valid - Loss: {:.4f} Accuracy: {:.4f} AUC: {:.4f}'.format(loss, accuracy, auc_score))


epochs = 200
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train(my_model, check_grad=False)
    validate(my_model)


# Unfreeze the base model
vgg_model.requires_grad_(True)
optimizer = Adam(my_model.parameters(), lr=.000001)


epochs = 3

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train(my_model, check_grad=False)
    validate(my_model)
    
# Save the trained model
torch.save(my_model.state_dict(), 'my_model_2.pth')
print("Model saved to my_model.pth")


import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import re

class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(test_dir))
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

def extract_numeric_id(filename):
    # This will extract the number from unknown_001.jpg, unknown_123.jpg, etc.
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else filename

transform = transforms.Compose([
    transforms.Resize((600, 600)),  
    transforms.ToTensor(),
])

test_dataset = TestDataset(test_dir='/kaggle/input/heads-or-tails-image-classification/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure model is on the correct device
my_model = my_model.to(device)
my_model.eval()  # Set to evaluation mode

ids = []
probs_class0 = []

with torch.no_grad():
    for images, names in test_loader:
        images = images.to(device)
        outputs = my_model(images)
        
        # Apply sigmoid to convert logits to probabilities
        probs_class1 = torch.sigmoid(outputs)
        
        # Ensure outputs are properly squeezed to remove extra dimensions
        probs_class1 = torch.squeeze(probs_class1)
        
        # Calculate probability of class 0 (heads)
        probs_class0_batch = 1 - probs_class1
        
        # Convert to scalar values (not lists)
        if probs_class0_batch.dim() > 0:
            probs_class0_values = probs_class0_batch.cpu().numpy().tolist()
        else:
            probs_class0_values = [probs_class0_batch.item()]
        
        # For single-value elements
        flat_probs = [p if isinstance(p, (int, float)) else p[0] for p in probs_class0_values]
        
        probs_class0.extend(flat_probs)
        ids.extend([extract_numeric_id(name) for name in names])

# Create submission dataframe
submission = pd.DataFrame({'prediction_id': ids, 'probability_of_heads': probs_class0})
submission = submission.sort_values('prediction_id')

# Clamp values between 0 and 1 to ensure they're valid probabilities
submission['probability_of_heads'] = submission['probability_of_heads'].clip(0, 1)

# Save submission file
submission.to_csv('submission_2.csv', index=False)

print(f"Submission file created with {len(submission)} predictions.")
print(f"Sample of predictions:")
print(submission.head())

