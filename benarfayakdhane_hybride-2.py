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


# BLOC : MODÃˆLE HYBRIDE ResNet50 + DeiT-Basic (entraÃ®nement end-to-end)
import torch
import torch.nn as nn
from torchvision import models
import timm

class HybridResNet50_DeitBasic(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 1. ResNet50 (features locaux)
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Identity()  # Supprime la couche finale â†’ sortie [B, 2048]
        
        # 2. DeiT-Basic (contexte global)
        # DeiT-Basic = deit_base_patch16_224
        self.deit = timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=0)  # [B, 768]
        
        # 3. Fusion simple (concatÃ©nation)
        self.fusion = nn.Linear(2048 + 768, 512)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extraire features ResNet50
        res_feat = self.resnet(x)  # [B, 2048]
        
        # Extraire token [CLS] de DeiT (dÃ©jÃ  1D car num_classes=0)
        deit_feat = self.deit(x)   # [B, 768]
        
        # Fusion
        fused = torch.cat([res_feat, deit_feat], dim=1)  # [B, 2816]
        return self.head(self.fusion(fused))  # [B, 5]

# Test rapide
model = HybridResNet50_DeitBasic()
x = torch.randn(2, 3, 224, 224)
print("âœ… Shape sortie :", model(x).shape)  # [2, 5]
print("âœ… Nombre de paramÃ¨tres :", sum(p.numel() for p in model.parameters()))


# BLOC 2 : DATASET + DATALOADERS POUR RESNET50 + DEIT-BASIC

from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

# Chemins
TRAIN_DIR = Path("/kaggle/input/aptos-2019/aptos2019/processed/train")
VAL_DIR = Path("/kaggle/input/aptos-2019/aptos2019/processed/validation")
CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"

# Dataset
class APTOSDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        df = pd.read_csv(csv_path)
        self.samples = df[['id_code', 'diagnosis']].values.tolist()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = self.img_dir / f"{img_name}.png"
        if not img_path.exists():
            img_path = Path("/kaggle/input/aptos2019-blindness-detection/train_images") / f"{img_name}.png"
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Transforms (224x224)
from torchvision import transforms
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# CrÃ©er les datasets
train_ds = APTOSDataset(TRAIN_DIR, CSV_PATH, transform=train_tf)
val_ds = APTOSDataset(VAL_DIR, CSV_PATH, transform=val_tf)

# DataLoader (batch_size rÃ©duit pour modÃ¨le lourd)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

print("âœ… Datasets et DataLoaders prÃªts")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridResNet50_DeitBasic().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # lr plus faible (modÃ¨le lourd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

best_acc = 0.0
print("ğŸš€ EntraÃ®nement ResNet50 + DeiT-Basic (end-to-end)...")

for epoch in range(20):
    model.train()
    running_loss = 0.0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    
    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_acc = correct / total
    scheduler.step()
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), '/kaggle/working/resnet50_deit_basic_best.pth')
        print(f"â­� Nouveau meilleur modÃ¨le ! Val Acc: {val_acc:.4f}")
    
    print(f"Ã‰poque {epoch+1}/20 | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

print(f"\nâœ… EntraÃ®nement terminÃ©. Meilleure prÃ©cision : {best_acc:.4f}")


# Passer en mode Ã©valuation
model.eval()

# Collecter toutes les prÃ©dictions
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Afficher le rapport
from sklearn.metrics import classification_report

print("\nğŸ“Š RAPPORT DE CLASSIFICATION â€“ ModÃ¨le Hybride ResNet50 + DeiT-Basic")
print(classification_report(
    all_labels,
    all_preds,
    target_names=['0-No DR', '1-Mild', '2-Moderate', '3-Severe', '4-Proliferative'],
    digits=4
))

# PrÃ©cision globale (optionnel)
import numpy as np
final_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nğŸ�¯ PrÃ©cision globale : {final_acc:.4f} ({100 * final_acc:.2f}%)")

