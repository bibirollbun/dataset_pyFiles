import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm.auto import tqdm

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models

# Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CONFIG = {
    'seed': 42,
    'img_size': 224,
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 1e-4,
    'num_classes': 5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])
print(f"Using Device: {CONFIG['device']}")


# --- DATA SPLITTING (70/20/10) ---
BASE_DIR = '/kaggle/input/cassava-leaf-disease-classification'
TRAIN_IMG_PATH = os.path.join(BASE_DIR, 'train_images')
df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))

# 1. Split 70% Train, 30% Temp
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df['label'], random_state=CONFIG['seed']
)

# 2. Split Temp into 20% Val and 10% Test
# (0.3333 of 30% is approx 10%)
val_df, test_df = train_test_split(
    temp_df, test_size=0.3333, stratify=temp_df['label'], random_state=CONFIG['seed']
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# --- DATASET & AUGMENTATIONS ---
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = os.path.join(self.img_dir, row['image_id'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            image = self.transforms(image=image)['image']
            
        return image, torch.tensor(row['label'], dtype=torch.long)

train_transforms = A.Compose([
    A.Resize(CONFIG['img_size'], CONFIG['img_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

valid_transforms = A.Compose([
    A.Resize(CONFIG['img_size'], CONFIG['img_size']),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

train_ds = CassavaDataset(train_df, TRAIN_IMG_PATH, train_transforms)
val_ds = CassavaDataset(val_df, TRAIN_IMG_PATH, valid_transforms)
test_ds = CassavaDataset(test_df, TRAIN_IMG_PATH, valid_transforms)

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()
        # 4 Convolutional Blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 112x112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 56x56
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 28x28
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 14x14
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetTransfer(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNetTransfer, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


# --- UNIVERSAL TRAINING FUNCTION ---
def train_model(model_class, model_name, epochs=8):
    print(f"\nğŸš€ STARTING TRAINING: {model_name}")
    print("="*40)
    
    model = model_class(num_classes=5).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(images)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')
            
    return history, best_acc

# 1. Train Custom CNN
hist_custom, acc_custom = train_model(CustomCNN, "CustomCNN", epochs=10)

# 2. Train ResNet50
hist_resnet, acc_resnet = train_model(ResNetTransfer, "ResNet50", epochs=8)


# --- COMPARISON & RESULTS ---

print("\nğŸ�† FINAL COMPARISON")
print("="*40)
print(f"1. Custom CNN Validation Accuracy: {acc_custom:.2f}%")
print(f"2. ResNet50 Validation Accuracy:   {acc_resnet:.2f}%")

# Plotting Comparison
plt.figure(figsize=(10, 5))
plt.plot(hist_custom['val_acc'], label=f'Custom CNN (Best: {acc_custom:.1f}%)', marker='o')
plt.plot(hist_resnet['val_acc'], label=f'ResNet50 (Best: {acc_resnet:.1f}%)', marker='o')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy %')
plt.legend()
plt.grid(True)
plt.show()

# --- FINAL TEST EVALUATION (Using the better model) ---
print("\nğŸ“� Evaluating Best Model on Test Set (10% split)...")
best_model_name = "ResNet50" if acc_resnet > acc_custom else "CustomCNN"
model_class = ResNetTransfer if acc_resnet > acc_custom else CustomCNN

final_model = model_class().to(CONFIG['device'])
final_model.load_state_dict(torch.load(f'best_{best_model_name}.pth'))
final_model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(CONFIG['device'])
        outputs = final_model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Selected Model: {best_model_name}")
print(classification_report(all_labels, all_preds))


# --- KAGGLE SUBMISSION ---
import glob

def predict_image(image_path, model, transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply the same validation transforms
    if transform:
        augmented = transform(image=image)
        image = augmented['image']
    
    # Add batch dimension (3, 224, 224) -> (1, 3, 224, 224)
    image = image.unsqueeze(0).to(CONFIG['device'])
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        
    return predicted.item()

# Load Test Images (Kaggle Evaluation Images)
test_files = glob.glob(os.path.join(BASE_DIR, 'test_images', '*.jpg'))
submission = {'image_id': [], 'label': []}

print(f"ğŸ“� Generating predictions for {len(test_files)} images...")

# 3. Predict Loop
for img_path in test_files:
    img_id = os.path.basename(img_path)
    pred_label = predict_image(img_path, final_model, valid_transforms)
    
    submission['image_id'].append(img_id)
    submission['label'].append(pred_label)

# 4. Save CSV
sub_df = pd.DataFrame(submission)
sub_df.to_csv('submission.csv', index=False)

print("âœ… 'submission.csv' saved!")
print(sub_df.head())

