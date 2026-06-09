import os
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt



DATA_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TEST_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
label_map = json.load(open(os.path.join(DATA_DIR, "label_num_to_disease_map.json")))
label_map = {int(k): v for k, v in label_map.items()}

IMG_SIZE = 128

train_transforms = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])



class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row["image_id"])).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, int(row["label"])



def get_resnet50_model(num_classes=5, freeze_backbone=True):
    model = models.resnet50(weights=None)   # <<< NO DOWNLOAD

    # Replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if "fc" not in name:
                p.requires_grad = False

    return model



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_df["fold"] = -1

for fold, (_, val_idx) in enumerate(skf.split(train_df["image_id"], train_df["label"])):
    train_df.loc[val_idx, "fold"] = fold

FOLD = 0

df_train = train_df[train_df.fold != FOLD]
df_val   = train_df[train_df.fold == FOLD]

train_loader = DataLoader(CassavaDataset(df_train, TRAIN_DIR, train_transforms),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(CassavaDataset(df_val, TRAIN_DIR, val_transforms),
                          batch_size=32, shuffle=False)



def train_one_epoch(model, optimizer, loader, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return loss_sum / total, correct / total, y_true, y_pred



train_losses_res, val_losses_res = [], []
train_accs_res, val_accs_res = [], []

y_true_res = []
y_pred_res = []



device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

EPOCHS = 2
criterion = nn.CrossEntropyLoss()

resnet_model = get_resnet50_model(num_classes=5, freeze_backbone=True).to(device)
optimizer_resnet = torch.optim.Adam(
    filter(lambda p: p.requires_grad, resnet_model.parameters()), 
    lr=2e-4
)

for epoch in range(EPOCHS):
    tr_loss, tr_acc = train_one_epoch(resnet_model, optimizer_resnet, train_loader, criterion)
    val_loss, val_acc, y_t, y_p = validate(resnet_model, val_loader, criterion)

    train_losses_res.append(tr_loss)
    val_losses_res.append(val_loss)
    train_accs_res.append(tr_acc)
    val_accs_res.append(val_acc)

    y_true_res = y_t
    y_pred_res = y_p

    print(f"Epoch {epoch+1}/{EPOCHS}",
          f"Train Acc={tr_acc:.4f}", f"Val Acc={val_acc:.4f}")



plt.plot(train_accs_res, label="Train Accuracy")
plt.plot(val_accs_res, label="Val Accuracy")
plt.legend()
plt.title("ResNet50 Accuracy Curve")
plt.show()

plt.plot(train_losses_res, label="Train Loss")
plt.plot(val_losses_res, label="Val Loss")
plt.legend()
plt.title("ResNet50 Loss Curve")
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true_res, y_pred_res)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("ResNet50 Confusion Matrix")
plt.show()



class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE//8) * (IMG_SIZE//8), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = SimpleCNN(num_classes=5).to(device)
optimizer_cnn = torch.optim.Adam(base_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS_CNN = 3

train_losses_cnn, val_losses_cnn = [], []
train_accs_cnn, val_accs_cnn = [], []

for epoch in range(EPOCHS_CNN):
    tr_loss, tr_acc = train_one_epoch(base_model, optimizer_cnn, train_loader, criterion)
    val_loss, val_acc, _, _ = validate(base_model, val_loader, criterion)

    train_losses_cnn.append(tr_loss)
    val_losses_cnn.append(val_loss)
    train_accs_cnn.append(tr_acc)
    val_accs_cnn.append(val_acc)

    print(f"[CNN] Epoch {epoch+1}/{EPOCHS_CNN} "
          f"Train Acc={tr_acc:.4f} Val Acc={val_acc:.4f}")



# CNN Accuracy Curve
plt.figure(figsize=(6,4))
plt.plot(train_accs_cnn, label="Train Accuracy")
plt.plot(val_accs_cnn, label="Validation Accuracy")
plt.title("CNN Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



# CNN Loss Curve
plt.figure(figsize=(6,4))
plt.plot(train_losses_cnn, label="Train Loss")
plt.plot(val_losses_cnn, label="Validation Loss")
plt.title("CNN Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get predictions on validation set
_, _, y_true_cnn, y_pred_cnn = validate(base_model, val_loader, criterion)

cm = confusion_matrix(y_true_cnn, y_pred_cnn)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



@torch.no_grad()
def ensemble_predict(model1, model2, loader):
    model1.eval()
    model2.eval()
    
    final_preds = []
    final_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        p1 = F.softmax(model1(x), dim=1)   # CNN
        p2 = F.softmax(model2(x), dim=1)   # ResNet50

        avg = (p1 + p2) / 2
        preds = avg.argmax(1)

        final_preds.extend(preds.cpu().numpy())
        final_labels.extend(y.cpu().numpy())

    return np.array(final_labels), np.array(final_preds)



print("Running Ensemble...")
y_true_ens, y_pred_ens = ensemble_predict(base_model, resnet_model, val_loader)

ensemble_accuracy = (y_true_ens == y_pred_ens).mean()
print("Ensemble Accuracy:", ensemble_accuracy)



import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true_ens, y_pred_ens)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.title("Hybrid Ensemble Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



plt.figure(figsize=(7,5))
plt.bar(
    ["CNN", "ResNet50", "Ensemble"],
    [
        val_accs_cnn[-1],        # last epoch CNN val accuracy
        val_accs_res[-1],        # last epoch ResNet val accuracy
        ensemble_accuracy        # final ensemble acc
    ],
    color=["skyblue", "orange", "purple"]
)
plt.ylabel("Accuracy")
plt.title("Model Comparison: CNN vs ResNet50 vs Ensemble")
plt.ylim(0, 1)
plt.show()



plt.figure(figsize=(7,5))
plt.plot(train_accs_cnn, label="CNN Train Acc", linestyle="--")
plt.plot(val_accs_cnn, label="CNN Val Acc")

plt.plot(train_accs_res, label="ResNet50 Train Acc", linestyle="--")
plt.plot(val_accs_res, label="ResNet50 Val Acc")

plt.title("Accuracy Curve: CNN vs ResNet50")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(7,5))
plt.plot(train_losses_cnn, label="CNN Train Loss", linestyle="--")
plt.plot(val_losses_cnn, label="CNN Val Loss")

plt.plot(train_losses_res, label="ResNet50 Train Loss", linestyle="--")
plt.plot(val_losses_res, label="ResNet50 Val Loss")

plt.title("Loss Curve: CNN vs ResNet50")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()



accuracies = [
    val_accs_cnn[-1],
    val_accs_res[-1],
    ensemble_accuracy
]

plt.figure(figsize=(7,5))
bars = plt.bar(["CNN", "ResNet50", "Ensemble"], accuracies,
               color=["skyblue", "orange", "purple"], width=0.6)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005,
             f"{yval:.3f}", ha="center", fontsize=12)

plt.title("Final Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()



fig, ax = plt.subplots(1, 3, figsize=(18,5))

cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
cm_res = confusion_matrix(y_true_res, y_pred_res)
cm_ens = confusion_matrix(y_true_ens, y_pred_ens)

sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("CNN Confusion Matrix")

sns.heatmap(cm_res, annot=True, fmt="d", cmap="Greens", ax=ax[1])
ax[1].set_title("ResNet50 Confusion Matrix")

sns.heatmap(cm_ens, annot=True, fmt="d", cmap="Purples", ax=ax[2])
ax[2].set_title("Ensemble Confusion Matrix")

plt.show()


