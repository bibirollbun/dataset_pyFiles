!pip install timm transformers -q


from tqdm import tqdm
import os, time
import numpy as np
import pandas as pd
from PIL import Image
import torch, timm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


cfg = {
    "data_dir": "/kaggle/input/grand-xray-slam-division-a",
    "train_csv": "train1.csv",
    "test_csv": "sample_submission_1.csv",
    "train_folder": "train1",
    "test_folder": "test1",

    "labels": [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
        "Pneumonia", "Pneumothorax", "Support Devices"
    ],

    "img_col": "Image_name",
    "img_size": 224,
    "batch_size": 32,
    "epochs": 5,
    "lr": 2e-5,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_file": "submission.csv",
}


LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices"
]

class XRayDataset(Dataset):
    def __init__(self, dataframe, img_dir, labels, img_col, transform=None, is_test=False):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.labels = labels
        self.img_col = img_col
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row[self.img_col])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, row[self.img_col]
        else:
            labels = torch.tensor(row[self.labels].values.astype("float32"))
            return image, labels


train_df = pd.read_csv(os.path.join(cfg["data_dir"], cfg["train_csv"]))
test_df  = pd.read_csv(os.path.join(cfg["data_dir"], cfg["test_csv"]))
train_split, valid_split = train_test_split(train_df, test_size=0.2, random_state=42)

train_tfms = transforms.Compose([
    transforms.Resize((cfg["img_size"], cfg["img_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
])

valid_tfms = transforms.Compose([
    transforms.Resize((cfg["img_size"], cfg["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
])


train_dataset = XRayDataset(train_split, os.path.join(cfg["data_dir"], cfg["train_folder"]),cfg["labels"], cfg["img_col"], transform=train_tfms)
valid_dataset = XRayDataset(valid_split, os.path.join(cfg["data_dir"], cfg["train_folder"]),cfg["labels"], cfg["img_col"], transform=valid_tfms)
test_dataset  = XRayDataset(test_df, os.path.join(cfg["data_dir"], cfg["test_folder"]),cfg["labels"], cfg["img_col"], transform=valid_tfms, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,num_workers=cfg["num_workers"], pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=cfg["batch_size"], shuffle=False,num_workers=cfg["num_workers"], pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False,num_workers=cfg["num_workers"], pin_memory=True)


class ViTMultiLabel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.head.in_features
        self.backbone.reset_classifier(0)  # remove classification head
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features // 2, num_classes)
        )


    def forward(self, x):
        features = self.backbone(x)
        out = self.fc(features)
        return out

model = ViTMultiLabel(
    model_name=cfg.get("model_name", "vit_base_patch16_224"),
    num_classes=len(cfg["labels"]),pretrained=True
)

# Multi-GPU support
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)
model = model.to(cfg["device"])
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=cfg['lr'])
scaler = torch.amp.GradScaler('cuda')


def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images, labels = images.to(cfg["device"]), labels.to(cfg["device"])
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")


def validate(epoch, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Valid]"):
            images, labels = images.to(cfg["device"]), labels.to(cfg["device"])
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > threshold).astype(int)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(preds)

    avg_loss = running_loss / len(valid_loader)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)

    # Metrics
    auc = roc_auc_score(all_labels, all_probs, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    f1_micro = f1_score(all_labels, all_preds, average="micro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    print(
        f"Epoch {epoch+1} Valid Loss: {avg_loss:.4f}, "
        f"AUC: {auc:.4f}, Acc: {acc:.4f}, "
        f"F1(micro): {f1_micro:.4f}, F1(macro): {f1_macro:.4f}, "
        f"Prec: {precision:.4f}, Recall: {recall:.4f}"
    )



for epoch in range(cfg["epochs"]):
    train_one_epoch(epoch)
    validate(epoch)


model.eval()
all_preds, all_ids = [], []
with torch.no_grad():
    for images, ids in tqdm(test_loader, desc="Inference"):
        images = images.to(cfg["device"])
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)  # one-hot encoding
        all_preds.append(preds)
        all_ids.extend(ids)

all_preds = np.vstack(all_preds)


submission = pd.DataFrame(all_preds, columns=cfg["labels"])
submission.insert(0, cfg["img_col"], all_ids)
submission.to_csv(cfg["output_file"], index=False)
print(f"✅ Submission saved to {cfg['output_file']}")

