# Starter PyTorch Notebook for Applications of Deep Learning Competition
# Combines a pretrained image CNN with tabular features, handling id-based image lookup.

import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Sampling percentage for quick testing
SAMPLE_PERCENT = 0.05  # use 0.1 for 10% of the training data
NUM_EPOCHS = 5

# Paths
data_dir      = os.path.dirname('../input/applications-of-deep-learning-wustl-summer-2025/train.csv')
train_csv     = os.path.join(data_dir, 'train.csv')
test_csv      = os.path.join(data_dir, 'test.csv')

# Load CSVs
df_train_full = pd.read_csv(train_csv)
df_test        = pd.read_csv(test_csv)

# Sample down the training set for faster iteration
df_train = df_train_full.sample(frac=SAMPLE_PERCENT, random_state=42).reset_index(drop=True)
print(f"Sampling {SAMPLE_PERCENT*100}% of training data: {len(df_train)} rows selected from {len(df_train_full)} total.")

# Identify columns
id_col      = 'id'
target_col  = 'preservation_score'  # actual target column in dataset
ignore_cols = [id_col, target_col]

# Tabular features
tabular_cols = [c for c in df_train.columns if c not in ignore_cols]

# Split numeric vs categorical
numeric_cols = df_train[tabular_cols].select_dtypes(include=['number']).columns.tolist()
cat_cols     = df_train[tabular_cols].select_dtypes(include=['object', 'category']).columns.tolist()
# Keep only features also present in test set
numeric_cols = [c for c in numeric_cols if c in df_test.columns]
cat_cols     = [c for c in cat_cols     if c in df_test.columns]

# Scale numeric features
scaler = StandardScaler()
if numeric_cols:
    df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
    df_test[numeric_cols]  = scaler.transform(df_test[numeric_cols])

# Encode categorical features
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([df_train[col].astype(str), df_test[col].astype(str)])
    le.fit(combined)
    df_train[col] = le.transform(df_train[col].astype(str))
    df_test[col]  = le.transform(df_test[col].astype(str))
    encoders[col] = le

# Final tabular feature list
tabular_feats = numeric_cols + cat_cols
num_tab_feats = len(tabular_feats)
print(f"Using {num_tab_feats} tabular features: {tabular_feats}")

# Dataset class definition
torch.manual_seed(0)
class MultiModalDataset(Dataset):
    def __init__(self, df, data_dir, id_col, transform=None, is_train=True):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = data_dir
        self.id_col    = id_col
        self.transform = transform
        self.is_train  = is_train
        self.tab_data  = torch.tensor(df[tabular_feats].fillna(0).values, dtype=torch.float32)
        if is_train:
            self.targets = torch.tensor(df[target_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct image filename from id (images are flat in data_dir)
        img_id   = int(row[self.id_col])
        img_name = f"{img_id}.jpg"
        img_path = os.path.join(self.data_dir, img_name)
        img      = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        tab = self.tab_data[idx]
        if self.is_train:
            return img, tab, self.targets[idx]
        else:
            return img, tab, img_id

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# DataLoader for sampled data
batch_size = 32
dataset    = MultiModalDataset(df_train, data_dir, id_col, transform=transform, is_train=True)
loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Multimodal model definition
class MultiModalModel(nn.Module):
    def __init__(self, num_tab_feats, backbone='resnet18'):
        super().__init__()
        cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = cnn.fc.in_features
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.tab_net = nn.Sequential(
            nn.Linear(num_tab_feats, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, img, tab):
        img_feat = self.cnn(img)
        tab_feat = self.tab_net(tab)
        x = torch.cat([img_feat, tab_feat], dim=1)
        return self.head(x)

# Training loop on sampled subset
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = MultiModalModel(num_tab_feats).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    for imgs, tabs, ys in loader:
        imgs, tabs, ys = imgs.to(device), tabs.to(device), ys.unsqueeze(1).to(device)
        preds = model(imgs, tabs)
        loss  = criterion(preds, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    print(f"Epoch {epoch+1} Loss: {running_loss/len(dataset):.4f}")

# Next: validation, checkpointing, and submission




# === Submission Generation ===
# Prepare test dataset and loader
# Note: df_test must be preprocessed similarly to df_train (scaled, encoded)
test_dataset = MultiModalDataset(df_test, data_dir, id_col, transform=transform, is_train=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Inference on test set
model.eval()
all_ids   = []
all_preds = []
with torch.no_grad():
    for imgs, tabs, img_ids in test_loader:
        imgs, tabs = imgs.to(device), tabs.to(device)
        outputs = model(imgs, tabs).squeeze(1).cpu().numpy()
        all_ids.extend(img_ids.tolist())
        all_preds.extend(outputs.tolist())

# Build submission DataFrame
df_submission = pd.DataFrame({
    id_col: all_ids,
    target_col: all_preds
})
# Save to CSV in Kaggle required format
submission_file = 'submission.csv'
df_submission.to_csv(submission_file, index=False)
print(f"Saved submission file: {submission_file}")


