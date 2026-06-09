# copy the weights and configurations for the pre-trained models 
!mkdir ~/.keras
!mkdir ~/.keras/models7
!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/


import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm



def preprocess_retina(img_path, radius=300):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = radius / (min(h, w) / 2)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 7
    img = img * mask[..., None]
    img = cv2.resize(img, (224, 224))
    return img

def label_to_levels(label, num_classes):
    return torch.tensor([1 if i < label else 0 for i in range(num_classes - 1)], dtype=torch.float32)

class DRDataset(Dataset):
    def __init__(self, df, num_classes, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.num_classes = num_classes

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = preprocess_retina(row.image_path)
        if self.transform:
            img = self.transform(img)
        label = int(row.level)
        levels = label_to_levels(label, self.num_classes)
        return img, levels, label

    def __len__(self):
        return len(self.df)



labels_df = pd.read_csv("/kaggle/input/diabetic-retinopathy-detection/trainLabels.csv.zip")
labels_df['image_path'] = labels_df['image'].apply(lambda x: f"/kaggle/input/diabetic-retinopathy-train-unzipped/train/{x}.jpeg")
labels_df = labels_df[labels_df['image_path'].apply(os.path.exists)]

NUM_CLASSES = labels_df.level.nunique()
train_df, val_df = train_test_split(labels_df, stratify=labels_df.level, test_size=0.2, random_state=42)

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_weights = compute_class_weight('balanced', classes=np.unique(train_df.level), y=train_df.level)
sample_weights = [class_weights[l] for l in train_df.level]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))

train_ds = DRDataset(train_df, NUM_CLASSES, transform_train)
val_ds = DRDataset(val_df, NUM_CLASSES, transform_val)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)



class CoralLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, levels):
        return self.loss(logits, levels.float())

def create_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes - 1)
    return model



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(NUM_CLASSES).to(device)
loss_fn = CoralLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float('inf')
best_state = None

for epoch in range(1, 16):
    model.train()
    train_loss = 0
    for x, levels, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
        x, levels = x.to(device), levels.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, levels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, levels, _ in val_loader:
            x, levels = x.to(device), levels.to(device)
            out = model(x)
            loss = loss_fn(out, levels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()



model.load_state_dict(best_state)
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x, _, labels in val_loader:
        x = x.to(device)
        out = model(x)
        preds = (torch.sigmoid(out) > 0.5).sum(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Validation Confusion Matrix")
plt.show()



from torch.utils.data import Dataset
import glob

# Load test image paths
test_paths = glob.glob("/kaggle/input/diabetic-retinopathy-test-unzipped/test/*.jpeg")
test_df = pd.DataFrame({"image": [os.path.basename(p) for p in test_paths], "image_path": test_paths})

# Test dataset
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx].image_path
        img = preprocess_retina(img_path)
        if self.transform:
            img = self.transform(img)
        return img, self.df.iloc[idx].image

    def __len__(self):
        return len(self.df)

# Use validation transform (no augmentation)
test_ds = TestDataset(test_df, transform=transform_val)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)



model.eval()
submission_preds = []
image_ids = []

with torch.no_grad():
    for x, ids in test_loader:
        x = x.to(device)
        out = model(x)
        preds = (torch.sigmoid(out) > 0.5).sum(dim=1).cpu().numpy()
        submission_preds.extend(preds)
        image_ids.extend(ids)



submission_df = pd.DataFrame({
    "image": image_ids,
    "level": submission_preds
})

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())


