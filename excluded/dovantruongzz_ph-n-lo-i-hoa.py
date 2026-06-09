# OPTIMIZED FOR OXFORD FLOWERS 102 COMPETITION
# Designed for maximum accuracy with proper submission format

import os, json, warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# ---------- 1. Config ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ğŸš€ Using device: {device}")

# Adjust paths for your dataset
DATA_DIR = "/kaggle/input/oxford-flowers-102"  # Change this to your dataset path
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid") 
TEST_DIR = os.path.join(DATA_DIR, "test")

# Verify paths
for path, name in [(TRAIN_DIR, "Train"), (VALID_DIR, "Valid"), (TEST_DIR, "Test")]:
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if not f.startswith('.')])
        print(f"âœ… {name}: {count} items")
    else:
        print(f"â�Œ {name} not found: {path}")

# ---------- 2. Enhanced Transforms for Flowers ----------
# Flowers benefit from color augmentation and rotation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),  # Flowers can be rotated
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- 3. Datasets & Loaders ----------
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
valid_ds = datasets.ImageFolder(VALID_DIR, transform=val_transform)

batch_size = 32 if device.type == 'cuda' else 16
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2)

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.root, fname)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname

test_ds = TestDataset(TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

num_classes = len(train_ds.classes)
print(f"ğŸ“Š Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")
print(f"ğŸ�¯ Classes: {num_classes}")

# Create mapping from model index to class ID (1-102)
inv_map = {v: int(k) for k, v in train_ds.class_to_idx.items()}
print(f"ğŸ”„ Class mapping sample: {dict(list(inv_map.items())[:5])}")

# ---------- 4. Model Selection (Multiple Options) ----------
def create_model(model_name, num_classes, use_pretrained=True):
    """Create model with different architectures"""
    
    if model_name == 'resnet50':
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if use_pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.fc.in_features, num_classes)
            )
        except:
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
    elif model_name == 'efficientnet':
        try:
            from torchvision.models import EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
            model = models.efficientnet_b0(weights=weights)
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )
        except:
            # Fallback to ResNet
            model = models.resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
    else:  # default: resnet18
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if use_pretrained else None
            model = models.resnet18(weights=weights)
        except:
            model = models.resnet18(weights=None)
            use_pretrained = False
        
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    return model, use_pretrained

# Try different models based on available resources
if device.type == 'cuda':
    model_name = 'resnet50'  # Better model for GPU
else:
    model_name = 'resnet18'  # Lighter for CPU

model, has_pretrained = create_model(model_name, num_classes, use_pretrained=True)
model = model.to(device)

print(f"ğŸ”¥ Using {model_name} {'with pretrained weights' if has_pretrained else 'from scratch'}")

# ---------- 5. Training Setup ----------
criterion = nn.CrossEntropyLoss()

# Different strategies based on pretrained status
if has_pretrained:
    # Freeze early layers, fine-tune later ones
    for name, param in model.named_parameters():
        if 'fc' not in name and 'classifier' not in name:
            if 'layer4' not in name and 'features.7' not in name:  # Keep last layers trainable
                param.requires_grad = False
    
    optimizer = optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'fc' in n or 'classifier' in n], 'lr': 1e-3},
        {'params': [p for n, p in model.named_parameters() if ('layer4' in n or 'features.7' in n) and 'fc' not in n], 'lr': 1e-4}
    ], weight_decay=1e-4)
else:
    # Train all layers
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# ---------- 6. Training Loop ----------
best_val_acc = 0.0
patience_counter = 0
max_patience = 7

EPOCHS = 25 if device.type == 'cuda' else 15
print(f"ğŸš€ Training for up to {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    total, correct = 0, 0
    
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / total
    train_acc = correct / total
    
    # Validation
    model.eval()
    v_total, v_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            v_correct += (preds == labels).sum().item()
            v_total += labels.size(0)
    val_acc = v_correct / v_total
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        patience_counter = 0
        print(f"ğŸ�‰ Epoch {epoch+1}: New best! Train: {train_acc:.3f}, Val: {val_acc:.3f}")
    else:
        patience_counter += 1
        print(f"ğŸ“Š Epoch {epoch+1}: Train: {train_acc:.3f}, Val: {val_acc:.3f} (patience: {patience_counter})")
    
    scheduler.step(val_acc)
    
    # Early stopping
    if patience_counter >= max_patience:
        print(f"â�¹ï¸� Early stopping at epoch {epoch+1}")
        break

print(f"ğŸ�† Best validation accuracy: {best_val_acc:.4f}")

# ---------- 7. Load Best Model & Predict ----------
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("âœ… Loaded best model")

model.eval()
ids = []
classes_out = []

print("ğŸ”® Making predictions...")
with torch.no_grad():
    for imgs, fnames in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        for p, fname in zip(preds, fnames):
            class_id = inv_map[int(p)]  # Convert to 1-102 range
            ids.append(fname)
            classes_out.append(int(class_id))

# ---------- 8. Create Competition Submission ----------
submission_df = pd.DataFrame({
    "id": ids,
    "class": classes_out
})

# Verify submission format
print(f"ğŸ“‹ Submission shape: {submission_df.shape}")
print(f"ğŸ”¢ Class range: {submission_df['class'].min()}-{submission_df['class'].max()}")
print("ğŸ“„ Sample submission:")
print(submission_df.head())

# Save submission
submission_df.to_csv("submission.csv", index=False)
submission_df.to_csv("/kaggle/working/submission.csv", index=False)  # Ensure it's in working directory

print("ğŸ’¾ Submission saved successfully!")
print(f"âœ… Ready for competition submission with {len(submission_df)} predictions")

