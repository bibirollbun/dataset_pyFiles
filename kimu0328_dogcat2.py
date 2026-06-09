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


#!/usr/bin/env python3
"""
Dogs vs Cats Redux: Kernels Edition - æ”¹è‰¯ç‰ˆ
Kaggleã‚³ãƒ³ãƒšãƒ†ã‚£ã‚·ãƒ§ãƒ³ç”¨ã�®ç”»åƒ�åˆ†é¡�ãƒ¢ãƒ‡ãƒ«

ä¸»ã�ªæ”¹è‰¯ç‚¹:
ResNet50ã‚’ä½¿ç”¨ï¼ˆå…ƒã�¯ResNet18ï¼‰
ãƒ‡ãƒ¼ã‚¿æ‹¡å¼µã�®è¿½åŠ 
3. å­¦ç¿’ç�‡ã‚¹ã‚±ã‚¸ãƒ¥ãƒ¼ãƒ©ãƒ¼
4. æ—©æœŸå�œæ­¢æ©Ÿèƒ½
5. ã‚ˆã‚Šè©³ç´°ã�ªè©•ä¾¡ã�¨ãƒ­ã‚°å‡ºåŠ›
6. ãƒ¢ãƒ‡ãƒ«ä¿�å­˜ã�¨å�¯è¦–åŒ–
"""

import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import warnings
warnings.filterwarnings('ignore')

# ===== 1. ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆè§£å‡� =====
def extract_datasets():
    """zipãƒ•ã‚¡ã‚¤ãƒ«ã‚’è§£å‡�"""
    print("ğŸ“¦ ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®è§£å‡�ã‚’é–‹å§‹...")
    
    input_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
    work_dir = "/kaggle/working/"
    
    with zipfile.ZipFile(input_dir + "train.zip", "r") as zip_ref:
        zip_ref.extractall(work_dir + "train/")
    
    with zipfile.ZipFile(input_dir + "test.zip", "r") as zip_ref:
        zip_ref.extractall(work_dir + "test/")
    
    print("âœ… ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®è§£å‡�å®Œäº†")
    return work_dir

# ===== 2. ãƒ‡ãƒ¼ã‚¿çµ±è¨ˆè¡¨ç¤º =====
def show_data_info(work_dir):
    """ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®çµ±è¨ˆæƒ…å ±ã‚’è¡¨ç¤º"""
    train_root = os.path.join(work_dir, "train")
    test_root = os.path.join(work_dir, "test")
    
    print("\nğŸ“Š ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆæƒ…å ±:")
    print(f"ğŸ“� train ãƒ•ã‚©ãƒ«ãƒ€ã�®å†…å®¹: {os.listdir(train_root)}")
    
    train_dir = os.path.join(train_root, "train")
    test_dir = os.path.join(test_root, "test")
    
    all_files = os.listdir(train_dir)
    cat_count = sum(1 for f in all_files if f.startswith('cat'))
    dog_count = sum(1 for f in all_files if f.startswith('dog'))
    
    print(f"  - çŒ«ã�®ç”»åƒ�: {cat_count:,}æ�š")
    print(f"  - çŠ¬ã�®ç”»åƒ�: {dog_count:,}æ�š")
    print(f"  - ç·�ç”»åƒ�æ•°: {len(all_files):,}æ�š")
    print(f"  - ãƒ†ã‚¹ãƒˆç”»åƒ�æ•°: {len(os.listdir(test_dir)):,}æ�š")
    
    return train_dir, test_dir

# ===== 3. è¨­å®š =====
class Config:
    """ãƒ¢ãƒ‡ãƒ«ã�¨ãƒˆãƒ¬ãƒ¼ãƒ‹ãƒ³ã‚°ã�®è¨­å®š"""
    BATCH_SIZE = 32
    EPOCHS = 3
    IMG_SIZE = 224
    LR = 0.001
    EARLY_STOPPING_PATIENCE = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def print_config(cls):
        print("\nâš™ï¸� è¨­å®š:")
        print(f"  - ãƒ‡ãƒ�ã‚¤ã‚¹: {cls.DEVICE}")
        print(f"  - ãƒ�ãƒƒãƒ�ã‚µã‚¤ã‚º: {cls.BATCH_SIZE}")
        print(f"  - ã‚¨ãƒ�ãƒƒã‚¯æ•°: {cls.EPOCHS}")
        print(f"  - ç”»åƒ�ã‚µã‚¤ã‚º: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"  - å­¦ç¿’ç�‡: {cls.LR}")

# ===== 4. Dataset ã‚¯ãƒ©ã‚¹ =====
class CatsDogsDataset(Dataset):
    """çŒ«ã�¨çŠ¬ã�®ç”»åƒ�ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆ"""
    
    def __init__(self, filepaths, labels=None, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.filepaths[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            if self.labels is not None:
                return image, self.labels[idx]
            else:
                return image
        except Exception as e:
            print(f"Error loading image {self.filepaths[idx]}: {e}")
            # ã‚¨ãƒ©ãƒ¼ã�®å ´å�ˆã�¯é»’ã�„ç”»åƒ�ã‚’è¿”ã�™
            if self.transform:
                image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.labels is not None:
                return image, self.labels[idx]
            else:
                return image

# ===== 5. ãƒ‡ãƒ¼ã‚¿å¤‰æ�› =====
def get_transforms():
    """ãƒ‡ãƒ¼ã‚¿æ‹¡å¼µã�®è¨­å®š"""
    # è¨“ç·´ç”¨ï¼šãƒ‡ãƒ¼ã‚¿æ‹¡å¼µã�‚ã‚Š
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # æ¤œè¨¼ãƒ»ãƒ†ã‚¹ãƒˆç”¨ï¼šãƒ‡ãƒ¼ã‚¿æ‹¡å¼µã�ªã�—
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nğŸ�¨ ãƒ‡ãƒ¼ã‚¿æ‹¡å¼µã�®è¨­å®š:")
    print("  - è¨“ç·´ç”¨: æ°´å¹³å��è»¢ã€�å›�è»¢ã€�è‰²èª¿å¤‰æ›´")
    print("  - æ¤œè¨¼ç”¨: ãƒªã‚µã‚¤ã‚ºã�®ã�¿")
    
    return train_transform, val_transform

# ===== 6. ãƒ‡ãƒ¼ã‚¿ãƒ­ãƒ¼ãƒ€ãƒ¼æº–å‚™ =====
def prepare_data_loaders(train_dir, train_transform, val_transform):
    """ãƒ‡ãƒ¼ã‚¿ãƒ­ãƒ¼ãƒ€ãƒ¼ã�®æº–å‚™"""
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    labels = [1 if 'dog' in f else 0 for f in train_files]  # 0: cat, 1: dog
    
    # å±¤åŒ–ã‚µãƒ³ãƒ—ãƒªãƒ³ã‚°ã�§è¨“ç·´ãƒ»æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã‚’åˆ†å‰²
    X_train, X_val, y_train, y_val = train_test_split(
        train_files, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    print(f"\nğŸ“Š ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆåˆ†å‰²:")
    print(f"  - è¨“ç·´ãƒ‡ãƒ¼ã‚¿: {len(X_train):,}æ�š")
    print(f"  - æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿: {len(X_val):,}æ�š")
    print(f"  - è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�®çŠ¬: {sum(y_train):,}æ�š, çŒ«: {len(y_train) - sum(y_train):,}æ�š")
    print(f"  - æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�®çŠ¬: {sum(y_val):,}æ�š, çŒ«: {len(y_val) - sum(y_val):,}æ�š")
    
    # Datasetä½œæˆ�
    train_dataset = CatsDogsDataset(X_train, y_train, train_transform)
    val_dataset = CatsDogsDataset(X_val, y_val, val_transform)
    
    # DataLoaderä½œæˆ�
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=2)
    
    print(f"\nâœ… DataLoaderä½œæˆ�å®Œäº†")
    print(f"  - è¨“ç·´ãƒ�ãƒƒãƒ�æ•°: {len(train_loader)}")
    print(f"  - æ¤œè¨¼ãƒ�ãƒƒãƒ�æ•°: {len(val_loader)}")
    
    return train_loader, val_loader

# ===== 7. ãƒ¢ãƒ‡ãƒ«æº–å‚™ =====
def create_model():
    """ResNet50ãƒ¢ãƒ‡ãƒ«ã�®ä½œæˆ�"""
    model = models.resnet50(weights='IMAGENET1K_V2')
    
    # æœ€å¾Œã�®åˆ†é¡�å±¤ã‚’2ã‚¯ãƒ©ã‚¹åˆ†é¡�ç”¨ã�«å¤‰æ›´
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )
    
    model = model.to(Config.DEVICE)
    
    # æ��å¤±é–¢æ•°ã�¨ã‚ªãƒ—ãƒ†ã‚£ãƒ�ã‚¤ã‚¶ãƒ¼
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    
    print(f"\nğŸ¤– ãƒ¢ãƒ‡ãƒ«æº–å‚™å®Œäº†:")
    print(f"  - ã‚¢ãƒ¼ã‚­ãƒ†ã‚¯ãƒ�ãƒ£: ResNet50")
    print(f"  - ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿æ•°: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - å­¦ç¿’å�¯èƒ½ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿æ•°: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, criterion, optimizer, scheduler

# ===== 8. æ¤œè¨¼é–¢æ•° =====
def validate_model(model, val_loader, criterion):
    """ãƒ¢ãƒ‡ãƒ«ã�®æ¤œè¨¼"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    
    return avg_loss, accuracy

# ===== 9. å­¦ç¿’ãƒ«ãƒ¼ãƒ— =====
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    """ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’"""
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    patience_counter = 0
    
    print("\nğŸš€ å­¦ç¿’é–‹å§‹")
    print("-" * 60)
    
    for epoch in range(Config.EPOCHS):
        # è¨“ç·´ãƒ•ã‚§ãƒ¼ã‚º
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # é€²æ�—è¡¨ç¤º
            if (i + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # æ¤œè¨¼ãƒ•ã‚§ãƒ¼ã‚º
        val_loss, val_acc = validate_model(model, val_loader, criterion)
        
        # ã‚¹ã‚±ã‚¸ãƒ¥ãƒ¼ãƒ©ãƒ¼ã�®æ›´æ–°
        scheduler.step()
        
        # çµ�æ�œã�®è¨˜éŒ²
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"\n[Epoch {epoch+1}/{Config.EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # æœ€è‰¯ãƒ¢ãƒ‡ãƒ«ã�®ä¿�å­˜
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
            print(f"  âœ… æ–°ã�—ã�„æœ€è‰¯ãƒ¢ãƒ‡ãƒ«ã‚’ä¿�å­˜ (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"  âš ï¸� æ—©æœŸå�œæ­¢: {Config.EARLY_STOPPING_PATIENCE}ã‚¨ãƒ�ãƒƒã‚¯æ”¹å–„ã�ªã�—")
                break
        
        print("-" * 60)
    
    print(f"\nğŸ�‰ å­¦ç¿’å®Œäº†! æœ€è‰¯æ¤œè¨¼ç²¾åº¦: {best_val_acc:.2f}%")
    return train_losses, val_losses, val_accuracies, best_val_acc

# ===== 10. å­¦ç¿’çµ�æ�œã�®å�¯è¦–åŒ– =====
def visualize_training(train_losses, val_losses, val_accuracies):
    """å­¦ç¿’çµ�æ�œã�®å�¯è¦–åŒ–"""
    plt.figure(figsize=(15, 5))
    
    # æ��å¤±ã�®å�¯è¦–åŒ–
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # ç²¾åº¦ã�®å�¯è¦–åŒ–
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Val Accuracy', marker='o', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # å­¦ç¿’ç�‡ã�®å�¯è¦–åŒ–
    plt.subplot(1, 3, 3)
    lr_history = [Config.LR * (0.5 ** (i // 2)) for i in range(len(train_losses))]
    plt.plot(lr_history, marker='o', color='red')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("âœ… å­¦ç¿’å±¥æ­´ã‚’ training_history.png ã�«ä¿�å­˜ã�—ã�¾ã�—ã�Ÿ")

# ===== 11. ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�§ã�®äºˆæ¸¬ =====
def predict_test_data(model, test_dir, val_transform):
    """ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�§ã�®äºˆæ¸¬ã�¨submissionä½œæˆ�"""
    # æœ€è‰¯ãƒ¢ãƒ‡ãƒ«ã‚’èª­ã�¿è¾¼ã�¿
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®æº–å‚™
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    test_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    test_dataset = CatsDogsDataset(test_files, None, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    # äºˆæ¸¬å®Ÿè¡Œ
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            dog_probs = probs[:, 1].cpu().numpy()
            predictions.extend(dog_probs)
    
    # Submissionä½œæˆ�
    ids = [int(os.path.basename(f).split('.')[0]) for f in test_files]
    submission = pd.DataFrame({
        'id': ids,
        'label': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    
    print(f"\nâœ… Submissionå®Œäº†!")
    print(f"  - äºˆæ¸¬æ•°: {len(predictions):,}ä»¶")
    print(f"  - çŠ¬ã�®ç¢ºç�‡ã�®çµ±è¨ˆ:")
    print(f"    - å¹³å�‡: {np.mean(predictions):.3f}")
    print(f"    - æ¨™æº–å��å·®: {np.std(predictions):.3f}")
    print(f"    - æœ€å°�å€¤: {np.min(predictions):.3f}")
    print(f"    - æœ€å¤§å€¤: {np.max(predictions):.3f}")
    
    return predictions

# ===== 12. äºˆæ¸¬åˆ†å¸ƒã�®å�¯è¦–åŒ– =====
def visualize_predictions(predictions):
    """äºˆæ¸¬åˆ†å¸ƒã�®å�¯è¦–åŒ–"""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(predictions, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Prediction Distribution')
    plt.xlabel('Dog Probability')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(sorted(predictions))
    plt.title('Sorted Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Dog Probability')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('prediction_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("âœ… äºˆæ¸¬åˆ†å¸ƒã‚’ prediction_distribution.png ã�«ä¿�å­˜ã�—ã�¾ã�—ã�Ÿ")

# ===== 13. æœ€çµ‚ã‚µãƒ�ãƒªãƒ¼ =====
def print_final_summary(best_val_acc, num_epochs):
    """æœ€çµ‚çµ�æ�œã�®ã‚µãƒ�ãƒªãƒ¼"""
    print("\n" + "="*60)
    print("ğŸ�¯ Dogs vs Cats åˆ†é¡�ã‚¿ã‚¹ã‚¯ - æœ€çµ‚çµ�æ�œ")
    print("="*60)
    print(f"ğŸ“Š ãƒ¢ãƒ‡ãƒ«æ€§èƒ½:")
    print(f"  - æœ€çµ‚æ¤œè¨¼ç²¾åº¦: {best_val_acc:.2f}%")
    print(f"  - è¨“ç·´ã‚¨ãƒ�ãƒƒã‚¯æ•°: {num_epochs}")
    print(f"  - ä½¿ç”¨ãƒ¢ãƒ‡ãƒ«: ResNet50")
    print(f"  - ç”»åƒ�ã‚µã‚¤ã‚º: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"\nğŸ”§ å®Ÿè£…ã�—ã�Ÿå·¥å¤«:")
    print(f"  - ãƒ‡ãƒ¼ã‚¿æ‹¡å¼µ (æ°´å¹³å��è»¢ã€�å›�è»¢ã€�è‰²èª¿å¤‰æ›´)")
    print(f"  - ImageNetäº‹å‰�å­¦ç¿’æ¸ˆã�¿ãƒ¢ãƒ‡ãƒ«")
    print(f"  - å­¦ç¿’ç�‡ã‚¹ã‚±ã‚¸ãƒ¥ãƒ¼ãƒ©ãƒ¼")
    print(f"  - æ—©æœŸå�œæ­¢")
    print(f"  - ãƒ‰ãƒ­ãƒƒãƒ—ã‚¢ã‚¦ãƒˆ")
    print(f"  - AdamWã‚ªãƒ—ãƒ†ã‚£ãƒ�ã‚¤ã‚¶ãƒ¼")
    print(f"  - å±¤åŒ–ã‚µãƒ³ãƒ—ãƒªãƒ³ã‚°")
    print(f"\nğŸ“� å‡ºåŠ›ãƒ•ã‚¡ã‚¤ãƒ«:")
    print(f"  - best_model.pth (æœ€è‰¯ãƒ¢ãƒ‡ãƒ«)")
    print(f"  - submission.csv (Kaggleæ��å‡ºç”¨)")
    print(f"  - training_history.png (å­¦ç¿’å±¥æ­´)")
    print(f"  - prediction_distribution.png (äºˆæ¸¬åˆ†å¸ƒ)")
    print("="*60)

# ===== ãƒ¡ã‚¤ãƒ³å®Ÿè¡Œé–¢æ•° =====
def main():
    """ãƒ¡ã‚¤ãƒ³å®Ÿè¡Œé–¢æ•°"""
    print("ğŸ�•ğŸ�± Dogs vs Cats Redux: Kernels Edition - æ”¹è‰¯ç‰ˆ")
    print("="*60)
    
    # 1. ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆæº–å‚™
    work_dir = extract_datasets()
    train_dir, test_dir = show_data_info(work_dir)
    
    # 2. è¨­å®šè¡¨ç¤º
    Config.print_config()
    
    # 3. ãƒ‡ãƒ¼ã‚¿å¤‰æ�›è¨­å®š
    train_transform, val_transform = get_transforms()
    
    # 4. ãƒ‡ãƒ¼ã‚¿ãƒ­ãƒ¼ãƒ€ãƒ¼æº–å‚™
    train_loader, val_loader = prepare_data_loaders(train_dir, train_transform, val_transform)
    
    # 5. ãƒ¢ãƒ‡ãƒ«æº–å‚™
    model, criterion, optimizer, scheduler = create_model()
    
    # 6. å­¦ç¿’å®Ÿè¡Œ
    train_losses, val_losses, val_accuracies, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler
    )
    
    # 7. å­¦ç¿’çµ�æ�œå�¯è¦–åŒ–
    visualize_training(train_losses, val_losses, val_accuracies)
    
    # 8. ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�§äºˆæ¸¬
    predictions = predict_test_data(model, test_dir, val_transform)
    
    # 9. äºˆæ¸¬åˆ†å¸ƒå�¯è¦–åŒ–
    visualize_predictions(predictions)
    
    # 10. æœ€çµ‚ã‚µãƒ�ãƒªãƒ¼
    print_final_summary(best_val_acc, len(train_losses))

if __name__ == "__main__":
    main()

