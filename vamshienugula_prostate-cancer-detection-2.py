# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O

# Input data files are available in the read-only "../input/" directory
# Running this code will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')



import os
import pandas as pd
import numpy as np
import openslide
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, classification_report, confusion_matrix



CSV_PATH = "/kaggle/input/prostate-cancer-grade-assessment/train.csv"
IMG_DIR = "/kaggle/input/prostate-cancer-grade-assessment/train_images"

df = pd.read_csv(CSV_PATH)
df = df[['image_id', 'isup_grade']]
df = df.sample(500, random_state=42).reset_index(drop=True)
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(IMG_DIR, f"{x}.tiff"))

print(df.head())



IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])



class ProstateFeatureDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        slide = openslide.OpenSlide(img_path)
        w, h = slide.dimensions
        patch = slide.read_region((w//2 - 256, h//2 - 256), 0, (512, 512)).convert("RGB")
        if self.transform:
            patch = self.transform(patch)
        label = row['isup_grade']  # multiclass 0 to 5
        return patch, label



# Load pretrained ResNet50 and MobileNetV2 models

resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet_features = nn.Sequential(*list(resnet.children())[:-1])
resnet_features.eval()

mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
mobilenet_features = nn.Sequential(*list(mobilenet.children())[:-1])
mobilenet_features.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_features = resnet_features.to(device)
mobilenet_features = mobilenet_features.to(device)

# Dataset and DataLoader
feature_dataset = ProstateFeatureDataset(df, transform=transform)
feature_loader = DataLoader(feature_dataset, batch_size=8, shuffle=False)

resnet_feats = []
mobilenet_feats = []
labels_list = []

with torch.no_grad():
    for images, labels in feature_loader:
        images = images.to(device)

        # --- ResNet50 features ---
        r_features = resnet_features(images)  # [batch, 2048, 1, 1]
        r_features = r_features.view(r_features.size(0), -1)  # flatten to [batch, 2048]

        # --- MobileNetV2 features ---
        m_features = mobilenet_features(images)  # [batch, 1280, 7, 7]
        m_features = torch.nn.functional.adaptive_avg_pool2d(m_features, (1, 1))
        m_features = m_features.view(m_features.size(0), -1)  # flatten to [batch, 1280]

        # --- Concatenate features ---
        fused = torch.cat((r_features, m_features), dim=1)  # [batch, 3328]

        resnet_feats.append(r_features.cpu().numpy())
        mobilenet_feats.append(m_features.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

# Combine all batches
resnet_array = np.vstack(resnet_feats)     # [500, 2048]
mobilenet_array = np.vstack(mobilenet_feats)  # [500, 1280]
fused_features = np.hstack((resnet_array, mobilenet_array))  # [500, 3328]
labels_array = np.array(labels_list)

print("Fused feature shape:", fused_features.shape)
print("Labels shape:", labels_array.shape)



X_train, X_temp, y_train, y_temp = train_test_split(fused_features, labels_array,
                                                    test_size=0.4, stratify=labels_array,
                                                    random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                test_size=0.5, stratify=y_temp,
                                                random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")



class HiFuseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=6):
        super(HiFuseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_dim = X_train.shape[1]  # 3328
model_hifuse = HiFuseClassifier(input_dim).to(device)



import torch
from torch.utils.data import TensorDataset, DataLoader

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Datasets and DataLoaders
train_dataset_hf = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset_hf = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset_hf = TensorDataset(X_test_tensor, y_test_tensor)

train_loader_hf = DataLoader(train_dataset_hf, batch_size=16, shuffle=True)
val_loader_hf = DataLoader(val_dataset_hf, batch_size=16)
test_loader_hf = DataLoader(test_dataset_hf, batch_size=16)

# Confirm shape
sample_features, _ = next(iter(train_loader_hf))
print("Sample shape from HiFuse loader:", sample_features.shape)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_hifuse.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model_hifuse.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in train_loader_hf:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_hifuse(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_loss = total_loss / total
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Accuracy={train_acc:.4f}")



model_hifuse.eval()
y_true_train = []
y_pred_train = []

with torch.no_grad():
    for features, labels in train_loader_hf:
        features = features.to(device)
        outputs = model_hifuse(features)
        preds = outputs.argmax(dim=1).cpu().numpy()

        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(preds)

train_accuracy = accuracy_score(y_true_train, y_pred_train)
print(f"Training Accuracy: {train_accuracy:.4f}")



y_true_val = []
y_pred_val = []

with torch.no_grad():
    for features, labels in val_loader_hf:
        features = features.to(device)
        outputs = model_hifuse(features)
        preds = outputs.argmax(dim=1).cpu().numpy()

        y_true_val.extend(labels.cpu().numpy())
        y_pred_val.extend(preds)

val_accuracy = accuracy_score(y_true_val, y_pred_val)
val_f1 = f1_score(y_true_val, y_pred_val, average='weighted')
val_kappa = cohen_kappa_score(y_true_val, y_pred_val)

y_true_val_bin = [1 if y > 0 else 0 for y in y_true_val]
y_pred_val_bin = [1 if y > 0 else 0 for y in y_pred_val]
val_roc = roc_auc_score(y_true_val_bin, y_pred_val_bin)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"F1 Score: {val_f1:.4f}")
print(f"Kappa: {val_kappa:.4f}")
print(f"ROC AUC: {val_roc:.4f}")



y_true_test = []
y_pred_test = []

with torch.no_grad():
    for features, labels in test_loader_hf:
        features = features.to(device)
        outputs = model_hifuse(features)
        preds = outputs.argmax(dim=1).cpu().numpy()

        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(preds)

test_accuracy = accuracy_score(y_true_test, y_pred_test)
test_f1 = f1_score(y_true_test, y_pred_test, average='weighted')
test_kappa = cohen_kappa_score(y_true_test, y_pred_test)

y_true_test_bin = [1 if y > 0 else 0 for y in y_true_test]
y_pred_test_bin = [1 if y > 0 else 0 for y in y_pred_test]
test_roc = roc_auc_score(y_true_test_bin, y_pred_test_bin)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"Kappa: {test_kappa:.4f}")
print(f"ROC AUC: {test_roc:.4f}")



results = pd.DataFrame({
    "Set": ["Training", "Validation", "Test"],
    "Accuracy": [train_accuracy, val_accuracy, test_accuracy],
    "F1 Measure": ["-", f"{val_f1:.4f}", f"{test_f1:.4f}"],
    "Kappa": ["-", f"{val_kappa:.4f}", f"{test_kappa:.4f}"],
    "ROC Area": ["-", f"{val_roc:.4f}", f"{test_roc:.4f}"]
})

print(results.to_string(index=False))



print("Test Set Accuracy:", test_accuracy)
print("\nðŸ“Š Classification Report:")
print(classification_report(y_true_test, y_pred_test))



cm = confusion_matrix(y_true_test, y_pred_test)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=[0, 1, 2, 3, 4, 5],
            yticklabels=[0, 1, 2, 3, 4, 5])
plt.title("ðŸ§© HiFuse Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()





