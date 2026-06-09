import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import timm
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import f1_score
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

# Configurations
BATCH_SIZE = 16
SEED = 42
IMG_SIZE = 384
LR = 1e-4
NUM_EPOCHS = 3
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    # Correct CSV file paths
    train_csv = '/kaggle/input/detect-ai-vs-human-generated-images/train.csv'
    test_csv = '/kaggle/input/detect-ai-vs-human-generated-images/test.csv'
    
    base_path = '/kaggle/input/ai-vs-human-generated-dataset'
    
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    return train_df, test_df, base_path, base_path  # Using base_path for images

class CustomDataset(Dataset):
    def __init__(self, df, data_dir, transforms=None, is_train=True):
        self.df = df
        self.data_dir = data_dir
        self.transforms = transforms
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['file_name']  # Ensure correct column name
        label = self.df.iloc[idx]['label'] if self.is_train else -1
        
        img_path = os.path.join(self.data_dir, filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None  # Skip if the image cannot be loaded

        if self.transforms:
            image = self.transforms(image=image)['image']
        
        return (image, label) if self.is_train else image

# Data Augmentation
train_transform = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    albu.GaussianBlur(blur_limit=(3, 5), p=0.2),
    albu.CoarseDropout(max_holes=8, max_height=IMG_SIZE//10, max_width=IMG_SIZE//10, p=0.5),
    albu.Resize(IMG_SIZE, IMG_SIZE),
    albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = albu.Compose([
    albu.Resize(IMG_SIZE, IMG_SIZE),
    albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def get_dataloaders(train_df, train_dir):
    train_df, val_df = train_test_split(train_df, test_size=0.3, random_state=SEED, stratify=train_df['label'])
    
    train_dataset = CustomDataset(train_df, train_dir, transforms=train_transform, is_train=True)
    val_dataset = CustomDataset(val_df, train_dir, transforms=val_transform, is_train=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def create_model():
    model = timm.create_model('convnext_base.fb_in22k', pretrained=True, num_classes=2)  # Using convnext_base for efficiency
    return model.to(dev)

def train_model(model, train_loader, val_loader):
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    best_f1 = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation Phase
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(dev), labels.to(dev)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with F1: {best_f1:.4f}")
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_f1'].append(val_f1)

        scheduler.step()  # Move scheduler update after validation

    model.load_state_dict(best_weights)
    return model, history

def predict(model, test_df, test_dir):
    test_df['img_path'] = test_df['file_name'].apply(lambda x: os.path.join(test_dir, x))  # Use correct column name
    predictions = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = row['img_path']
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            input_tensor = val_transform(image=image)['image'].unsqueeze(0).to(dev)
            with torch.no_grad():
                pred_label = model(input_tensor).argmax(dim=-1).item()
            predictions.append((row['file_name'], pred_label))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    submission_df = pd.DataFrame(predictions, columns=["file_name", "label"])
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file saved!")
    return submission_df

if __name__ == '__main__':
    train_df, test_df, train_dir, test_dir = load_data()
    train_loader, val_loader = get_dataloaders(train_df, train_dir)
    model = create_model()
    model, history = train_model(model, train_loader, val_loader)
    submission_df = predict(model, test_df, test_dir)


