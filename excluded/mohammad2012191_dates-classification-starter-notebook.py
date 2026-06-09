#################################
# 1. IMPORTS & CONFIG
#################################
import os
import gc
import cv2
import torch
import random
import numpy as np
import pandas as pd

from glob import glob
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T


#################################
# 2. BASIC SETTINGS
#################################
class CFG:
    seed = 42
    n_splits = 5       # Basic K-Fold
    epochs = 5
    train_bs = 16
    valid_bs = 32
    lr = 1e-3
    num_workers = 2
    img_size = 224
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)


#################################
# 3. LOAD DATA
#################################
train_df = pd.read_csv("/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv")
train_df["path"] = "/kaggle/input/open-data-day-2025-dates-types-classification/train/" + train_df["filename"]
train_df.drop("filename", axis=1, inplace=True)
train_df = train_df[["path","label"]]

test_paths = glob("/kaggle/input/open-data-day-2025-dates-types-classification/test/*")
test_df = pd.DataFrame({"path": test_paths})

print("Train Shape:", train_df.shape)
print("Test  Shape:", test_df.shape)
print(train_df.head())

# Map string labels to integer indices
label_mapping = {
    "Ajwa":        0,
    "Medjool":     1,
    "Meneifi":     2,
    "Nabtat Ali":  3,
    "Shaishe":     4,
    "Sokari":      5,
    "Sugaey":      6
}
train_df["label_idx"] = train_df["label"].map(label_mapping)


#################################
# 4. K-FOLD Split
#################################
train_df["fold"] = -1
kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

for fold_number, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
    train_df.loc[val_idx, "fold"] = fold_number

print(train_df.groupby("fold").size())


#################################
# 5. DATASET & TRANSFORMS
#################################
train_transforms = T.Compose([
    T.ToPILImage(),                         
    T.Resize((224, 224)),                  
    T.RandomHorizontalFlip(p=0.5),         
    T.ToTensor(),                          
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


valid_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])



class DatesDataset(Dataset):
    def __init__(self, df, mode="train", transforms=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        image_path = row["path"]
        label_idx = row["label_idx"] if "label_idx" in row else None
        
        # Read image (BGR), then convert to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)

        if self.mode != "test":
            return image, label_idx
        else:
            return image


#################################
# 6. MODEL DEFINITION (EffNet-B0)
#################################
# We'll modify the final layer for the number of classes we have (7).
def get_model(num_classes=7, pretrained=True):
    model = torchvision.models.efficientnet_b0(pretrained=pretrained)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model


#################################
# 7. TRAIN & VALID FUNCTIONS
#################################
def train_one_epoch(model, optimizer, dataloader, device, criterion):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    epoch_loss = total_loss / len(dataloader.dataset)
    return epoch_loss

def valid_one_epoch(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


#################################
# 8. K-FOLD TRAINING LOOP
#################################
def run_training(fold):
    print(f"========== Fold: {fold} ==========")

    # Split
    train_data = train_df[train_df["fold"] != fold].reset_index(drop=True)
    valid_data = train_df[train_df["fold"] == fold].reset_index(drop=True)

    # Datasets
    train_dataset = DatesDataset(train_data, mode="train", transforms=train_transforms)
    valid_dataset = DatesDataset(valid_data, mode="valid", transforms=valid_transforms)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs,
                              shuffle=True, num_workers=CFG.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs,
                              shuffle=False, num_workers=CFG.num_workers)

    # Model, Optimizer, Loss
    model = get_model(num_classes=len(label_mapping), pretrained=True)
    model.to(CFG.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    best_acc = 0.0
    for epoch in range(CFG.epochs):
        print(f"Fold {fold} | Epoch {epoch+1}/{CFG.epochs}")
        
        train_loss = train_one_epoch(model, optimizer, train_loader, CFG.device, criterion)
        valid_loss, valid_acc = valid_one_epoch(model, valid_loader, CFG.device, criterion)

        print(f"  [Train Loss: {train_loss:.4f}]  [Valid Loss: {valid_loss:.4f}]  [Valid Acc: {valid_acc:.4f}]")

        # Save best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            save_path = f"effb0_fold_{fold}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  --> Model saved to {save_path}")
    
    print(f"Fold {fold} best accuracy: {best_acc:.4f}\n")


for fold in range(CFG.n_splits):
    run_training(fold)


#################################
# 10. INFERENCE & SUBMISSION
#################################
def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(dataloader, desc="Inferring", leave=False):
            imgs = imgs.to(device)
            outputs = model(imgs)
            # We'll use softmax for probabilities 
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            preds.append(probabilities)
    return np.concatenate(preds, axis=0)

def run_inference_on_test():
    # We prepare the test dataset
    test_dataset = DatesDataset(test_df, mode="test", transforms=valid_transforms)
    test_loader = DataLoader(test_dataset, batch_size=CFG.valid_bs,
                             shuffle=False, num_workers=CFG.num_workers)

    # Load each fold's best model, do predictions for each, average all models predictions
    fold_preds = []
    for f_ in range(CFG.n_splits):
        model_path = f"effb0_fold_{f_}.pth"
        if not os.path.exists(model_path):
            print(f"Warning: no model found at {model_path}, skipping this fold.")
            continue
        
        model = get_model(num_classes=len(label_mapping), pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=CFG.device))
        model.to(CFG.device)

        preds = inference_fn(model, test_loader, CFG.device)
        fold_preds.append(preds)

        # cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Average predictions across folds
    final_preds = np.mean(fold_preds, axis=0)  # shape = [len(test), #classes]
    class_indices = np.argmax(final_preds, axis=1)

    inv_map = {v: k for k, v in label_mapping.items()}
    final_labels = [inv_map[i] for i in class_indices]

    # Create submission
    submission = pd.DataFrame({
        "filename": test_df["path"].apply(os.path.basename),
        "label": final_labels
    })
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv!")
    print(submission.head())


run_inference_on_test()

