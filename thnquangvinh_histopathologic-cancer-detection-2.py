import numpy as np
import pandas as pd

import os
base_dir = '../input/histopathologic-cancer-detection/'
print(os.listdir(base_dir))

# Matplotlib and Seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")

# OpenCV Image Library
import cv2

# Import PyTorch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import torchvision
import torch.optim as optim

# Import useful sklearn functions
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
)
from PIL import Image
from tqdm import tqdm


# Data paths
train_path = '../input/histopathologic-cancer-detection/train/'
test_path = '../input/histopathologic-cancer-detection/test/'


# Read CSV file
train_df = pd.read_csv("../input/histopathologic-cancer-train-test-index/train_split.csv")
test_df = pd.read_csv("../input/histopathologic-cancer-train-test-index/test_split.csv")


# Customize class for datasets
class CreateDataset(Dataset):
    def __init__(self, df_data, data_dir='./', transform=None):
        super().__init__()
        self.df = df_data.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_name, label = self.df.iloc[index]
        img_path = os.path.join(self.data_dir, img_name + '.tif')
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image {img_path} not found.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


transforms_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_data = CreateDataset(df_data=train_df, data_dir=train_path, transform=transforms_train)


batch_size = 128    # Set Batch Size
valid_size = 0.2    # Percentage of training set to use as validation

num_train = len(train_data)  
train_size = int((1 - valid_size) * num_train)  
valid_size = num_train - train_size  

train_data, valid_data = random_split(train_data, [train_size, valid_size])

# Prepare data loaders (combine dataset and sampler)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)


transforms_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Creating test data
test_data = CreateDataset(df_data=test_df, data_dir=train_path, transform=transforms_test)

# Prepare the test loader
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # Convolutional and Pooling Layers
        self.conv1=nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        self.conv2=nn.Sequential(
                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        self.conv3=nn.Sequential(
                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        self.conv4=nn.Sequential(
                nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        self.conv5=nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2))
        
        self.dropout2d = nn.Dropout2d()
        
        self.fc=nn.Sequential(
                nn.Linear(512*3*3,1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(1024,512),
                nn.Dropout(0.4),
                nn.Linear(512, 1),
                nn.Sigmoid())
        
    def forward(self,x):
        """Method for Forward Prop"""
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        return x


# Create a complete CNN
model = CNN()
print(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")


# Trainable Parameters
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: \n{}".format(pytorch_total_params))


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    """ 
    Train & validate a PyTorch model 
        Args: 
            model : CNN model 
            train_loader: DataLoader cho táº­p train 
            val_loader : DataLoader cho táº­p validation 
            criterion : Loss function (nn.BCELoss hoáº·c nn.CrossEntropyLoss) 
            optimizer : Optimizer (Adam, SGD,...) 
            device : 'cuda' hoáº·c 'cpu' 
            num_epochs : sá»‘ epoch huáº¥n luyá»‡n 
        Returns: 
            model : mÃ´ hÃ¬nh sau khi train 
            history : dict chá»©a loss/acc theo epoch 
    """
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": []
    }
    
    best_val_loss = float("inf")
    best_model_wts = model.state_dict()
    
    model.to(device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # ------------------------
        # TRAINING PHASE
        # ------------------------
        model.train()
        running_loss, total = 0.0, 0
        all_labels, all_outputs = [], []

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(images).squeeze()   # Ä‘Ã£ sigmoid rá»“i
            loss = criterion(outputs, labels.float())

            all_outputs.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
        
        epoch_train_loss = running_loss / total
        try:
            epoch_train_auc = roc_auc_score(all_labels, all_outputs)
        except ValueError:
            epoch_train_auc = 0.0

        # ------------------------
        # VALIDATION PHASE
        # ------------------------
        model.eval()
        val_loss, val_total = 0.0, 0
        val_labels, val_outputs = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images).squeeze() 
                loss = criterion(outputs, labels.float())

                val_outputs.extend(outputs.detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())
                
                val_loss += loss.item() * images.size(0)
                val_total += labels.size(0)
        
        epoch_val_loss = val_loss / val_total
        try:
            epoch_val_auc = roc_auc_score(val_labels, val_outputs)
        except ValueError:
            epoch_val_auc = 0.0

        # ------------------------
        # LOGGING
        # ------------------------
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_auc"].append(epoch_train_auc)
        history["val_auc"].append(epoch_val_auc)

        print(f"Train Loss: {epoch_train_loss:.4f} AUC: {epoch_train_auc:.4f}")
        print(f"Valid Loss: {epoch_val_loss:.4f} AUC: {epoch_val_auc:.4f}")
        
        # ------------------------
        # SAVE BEST MODEL (theo val_loss nhá»� nháº¥t)
        # ------------------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, "best_model.pth")
            print(">>> Saved best model with Val Loss:", best_val_loss)
    
    # load best weights
    model.load_state_dict(best_model_wts)
    return model, history


criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00015, weight_decay=1e-4)


model, history = train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=30)


import pickle

with open("train_history.pkl", "wb") as f:
    pickle.dump(history, f)


def plot_history(history):
    # Plot loss
    plt.figure(figsize=(10,5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.show()


def evaluate_model(model, data_loader, device, threshold=None, min_tpr=0.95, tune_threshold=True):
    """
    Evaluate model trÃªn táº­p dá»¯ liá»‡u vÃ  (tuá»³ chá»�n) tá»± Ä‘á»™ng tÃ¬m threshold tá»‘t nháº¥t
    theo tiÃªu chÃ­ Recall (TPR) Æ°u tiÃªn.
    
    Args:
        model : mÃ´ hÃ¬nh Ä‘Ã£ huáº¥n luyá»‡n
        data_loader : DataLoader cho táº­p validation/test
        device : 'cuda' hoáº·c 'cpu'
        threshold : threshold cá»‘ Ä‘á»‹nh (náº¿u khÃ´ng muá»‘n auto-tune)
        min_tpr : ngÆ°á»¡ng tá»‘i thiá»ƒu cá»§a Recall khi tune threshold
        tune_threshold : náº¿u True -> tá»± Ä‘á»™ng tÃ¬m threshold tá»‘i Æ°u theo Recall

    Returns:
        metrics : dict chá»©a káº¿t quáº£ Ä‘Ã¡nh giÃ¡
    """
    model.eval()
    y_true, y_probs = [], []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            
            # Náº¿u model cuá»‘i cÃ³ Sigmoid rá»“i thÃ¬ bá»� sigmoid á»Ÿ Ä‘Ã¢y
            if logits.shape[-1] == 1:
                probs = logits.squeeze().detach().cpu().numpy()
            else:
                probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

            y_probs.extend(probs)
            y_true.extend(yb.cpu().numpy())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # =====================
    # 1ï¸�âƒ£ Tune threshold náº¿u báº­t
    # =====================
    if tune_threshold:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        valid_idx = np.where(tpr >= min_tpr)[0]

        if len(valid_idx) == 0:
            print(f"âš ï¸� KhÃ´ng cÃ³ threshold nÃ o Ä‘áº¡t TPR >= {min_tpr}. Sáº½ chá»�n threshold vá»›i TPR cao nháº¥t.")
            best_idx = np.argmax(tpr)
        else:
            best_idx = valid_idx[np.argmin(fpr[valid_idx])]

        threshold = thresholds[best_idx]
        print(f"\nğŸ”� Tuning threshold (TPR priority): {threshold:.4f}")
        print(f"TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")

    elif threshold is None:
        threshold = 0.5  # Máº·c Ä‘á»‹nh

    # =====================
    # 2ï¸�âƒ£ Ã�p dá»¥ng threshold Ä‘á»ƒ tÃ­nh metric
    # =====================
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = None

    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Evaluation Report ===")
    print(f"Accuracy : {acc:.4f}")
    if auc is not None:
        print(f"ROC AUC  : {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print("\nClassification Report:\n", report)

    # =====================
    # 3ï¸�âƒ£ Confusion Matrix
    # =====================
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Threshold={threshold:.3f})")
    plt.show()

    # =====================
    # 4ï¸�âƒ£ ROC Curve
    # =====================
    if auc is not None:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

    return {
        "accuracy": acc,
        "auc": auc,
        "report": report,
        "confusion_matrix": cm,
        "threshold": threshold
    }


%matplotlib inline
plot_history(history)


test_results = evaluate_model(
    model,
    test_loader,
    device,
    tune_threshold=True,
    min_tpr=0.95
)

