!pip install -q timm seaborn opendatasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from PIL import Image


# Load CSVs
train_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
test_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")

print("ğŸ“Š Label Distribution:")
sns.countplot(data=train_df, x="diagnosis")
plt.title("DR Class Distribution")
plt.show()


# Visualize one image from each class
img_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i in range(5):
    img_id = train_df[train_df.diagnosis == i].iloc[0].id_code
    path = f"/kaggle/input/aptos2019-blindness-detection/train_images/{img_id}.png"
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[i].imshow(img)
    axes[i].set_title(f"Class {i}")
    axes[i].axis('off')
plt.suptitle("Sample Images per Class")
plt.show()


def preprocess_image(path, size=512):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    final = cv2.resize(final, (size, size))
    return final

# Visualize processed image
example = train_df.iloc[0].id_code
img_path = f"/kaggle/input/aptos2019-blindness-detection/train_images/{example}.png"
processed_img = preprocess_image(img_path)

plt.imshow(processed_img)
plt.title("Preprocessed Sample")
plt.axis("off")
plt.show()


class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row.id_code
        img_path = f"/kaggle/input/aptos2019-blindness-detection/train_images/{img_id}.png"

        if not os.path.exists(img_path):
            print(f"â�Œ Image not found: {img_path}")

        image = preprocess_image(img_path)
        image = Image.fromarray(image)  # Required for torchvision transforms
        label = row.diagnosis

        if self.transform:
            image = self.transform(image)

        return image, label


# Transforms
transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class DRModel(nn.Module):
    def __init__(self, backbone='mobilenetv3_large_100', num_classes=5):
        super(DRModel, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


BATCH_SIZE = 16
EPOCHS = 5  # Start low for Colab; increase later
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    probs_list = []  # List to store probabilities

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)  # Move imgs to device
            labels = labels.to(device)  # Move labels to device

            outputs = model(imgs)

            # Apply softmax to get probabilities (logits -> probabilities)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()  # Apply softmax here

            # Get predicted class (from probabilities)
            pred = np.argmax(probs, axis=1)
            preds.extend(pred)

            targets.extend(labels.cpu().numpy())  # Ensure labels are on CPU for compatibility
            probs_list.extend(probs)  # Store the probabilities for later analysis if needed

    return preds, targets, probs_list


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.diagnosis)):
    if fold != 4:
        continue  # Skip first 4 folds
    print(f"\nğŸ“‚ Fold {fold+1}/5")

    train_data = train_df.iloc[train_idx]
    val_data = train_df.iloc[val_idx]

    train_ds = DRDataset(train_data, img_dir, transform)
    val_ds = DRDataset(val_data, img_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = DRModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = FocalLoss()

    best_score = -np.inf  # Initialize the best score variable

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        preds, targets = evaluate(model, val_loader)
        
        qwk = cohen_kappa_score(targets, preds, weights='quadratic')
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | QWK: {qwk:.4f}")
        print(classification_report(targets, preds, digits=3))
        
        # Confusion Matrix
        cm = confusion_matrix(targets, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Fold {fold+1} - Epoch {epoch+1} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        
        # If this epoch gives a better QWK, save the model
        if qwk > best_score:
            best_score = qwk
            torch.save(model.state_dict(), f"best_model_fold{fold+1}.pt")  # Save best model



def predict_image(path, model):
    model.eval()
    img = preprocess_image(path)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        pred = torch.argmax(out, dim=1).item()
    return pred

# Predict on 5 random samples
model.eval()
sample_paths = train_df.sample(5).id_code.values
for img_id in sample_paths:
    img_path = f"/kaggle/input/aptos2019-blindness-detection/train_images/{img_id}.png"
    pred = predict_image(img_path, model)
    print(f"ğŸ–¼ï¸� {img_id}.png â†’ Predicted DR Stage: {pred}")
    img = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {pred}")
    plt.axis("off")
    plt.show()


from sklearn.metrics import cohen_kappa_score
print("Quadratic Weighted Kappa:", cohen_kappa_score(targets, preds, weights='quadratic'))


import random
from torchvision import transforms
import matplotlib.pyplot as plt

# Load your best model (you can change to fold1, fold2, etc.)
model = DRModel().to(device)
model.load_state_dict(torch.load("dr_model_fold4.pt"))
model.eval()

# Define transform again if needed
test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Include normalization
])

# Pick 5 random samples from validation set (use val_data from fold 5)
sample_idxs = random.sample(range(len(val_data)), 5)
fig, axs = plt.subplots(1, 5, figsize=(20, 5))

for i, idx in enumerate(sample_idxs):
    row = val_data.iloc[idx]
    img_path = os.path.join(img_dir, row['id_code'] + ".png")
    image = Image.open(img_path).convert("RGB")
    input_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()

    true_label = row['diagnosis']

    axs[i].imshow(image)
    axs[i].axis('off')
    axs[i].set_title(f"GT: {true_label} | Pred: {pred}", fontsize=12)

plt.tight_layout()
plt.show()


