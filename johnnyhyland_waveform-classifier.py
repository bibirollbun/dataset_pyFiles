from types import SimpleNamespace
import torch

cfg = SimpleNamespace()
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.local_rank = 0
cfg.seed = 123
cfg.subsample = None 

import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

class_map = {
    "CurveFault_A": 0,
    "CurveFault_B": 1,
    "CurveVel_A": 2,
    "CurveVel_B": 3,
    "FlatFault_A": 4,
    "FlatFault_B": 5,
    "FlatVel_A": 6,
    "FlatVel_B": 7,
    "Style_A": 8,
    "Style_B": 9,
}

class AlignedSeismicClassificationDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.samples = self.load_aligned_samples()

    def load_aligned_samples(self):
        df = pd.read_csv("/kaggle/input/openfwi-preprocessed-72x72/folds.csv")
        
        if self.cfg.subsample is not None:
            df = df.groupby(["dataset", "fold"]).head(self.cfg.subsample)

        if self.mode == "train":
            df = df[df["fold"] != 0]
        else:
            df = df[df["fold"] == 0]

        samples = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {self.mode} samples"):
            dataset = row["dataset"]
            label = class_map.get(dataset)
            if label is None:
                continue

            p1 = os.path.join("/kaggle/input/open-wfi-1/openfwi_float16_1/", row["data_fpath"])
            p2 = os.path.join("/kaggle/input/open-wfi-1/openfwi_float16_1/", row["data_fpath"].split("/")[0], "*", row["data_fpath"].split("/")[-1])
            p3 = os.path.join("/kaggle/input/open-wfi-2/openfwi_float16_2/", row["data_fpath"])
            p4 = os.path.join("/kaggle/input/open-wfi-2/openfwi_float16_2/", row["data_fpath"].split("/")[0], "*", row["data_fpath"].split("/")[-1])
            farr = glob.glob(p1) + glob.glob(p2) + glob.glob(p3) + glob.glob(p4)
            
            if farr:
                file_path = farr[0]
                for sample_idx in range(500):
                    samples.append((file_path, sample_idx, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, sample_idx, label = self.samples[idx]
        data = np.load(file_path, mmap_mode='r')[sample_idx]  # shape: (sources, time, receivers)
        data = torch.from_numpy(data).float()
        # Take mean across sources dimension to get single channel
        data = data.mean(dim=0, keepdim=True)  # shape: (1, time, receivers)
        return data, label

class SeismicEfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        if self.backbone.features[0][0].in_channels != 1:
            self.backbone.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def train_aligned_classifier():
    print("Creating datasets...")
    train_dataset = AlignedSeismicClassificationDataset(cfg, mode="train")
    val_dataset = AlignedSeismicClassificationDataset(cfg, mode="valid")
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = SeismicEfficientNetClassifier(num_classes=10).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc="Training")):
            data, labels = data.to(cfg.device), labels.to(cfg.device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Validation"):
                data, labels = data.to(cfg.device), labels.to(cfg.device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * data.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "aligned_classifier.pth")
            print(f"New best model saved! Val Acc: {val_acc:.4f}")
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
    return model

# Train the model
print("Starting training...")
model = train_aligned_classifier()




