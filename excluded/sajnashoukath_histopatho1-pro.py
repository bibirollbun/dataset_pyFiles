# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install -q efficientnet_pytorch



import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from PIL import Image

# -----------------------------

# Configuration

# -----------------------------

DATA_DIR = '/kaggle/input/histopathologic-cancer-detection/'
BATCH_SIZE = 64
IMG_SIZE = 96
NUM_FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)



class HistoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if 'label' in row:
            img_path = os.path.join(DATA_DIR, 'train', row['id'] + '.tif')
        else:
            img_path = os.path.join(DATA_DIR, 'test', row['id'])
        img = Image.open(img_path).convert('RGB')
        label = torch.tensor(row['label'], dtype=torch.float32) if 'label' in row else torch.tensor(0.0)
        if self.transform:
            img = self.transform(img)
        return img, label



# Strong Train-Time Augmentations

train_transform = T.Compose([
T.RandomHorizontalFlip(),
T.RandomVerticalFlip(),
T.RandomRotation(20),
T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
T.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0), ratio=(0.9,1.1)),
T.ToTensor(),
T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Minimal TTA Transform

tta_transform = T.Compose([
T.ToTensor(),
T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])



def get_model(model_name='efficientnet_b3'):
    if model_name == 'efficientnet_b3':
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=1)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(DEVICE)



def train_fold(model, train_loader, val_loader, epochs=5, lr=1e-4):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

    # Validation AUC
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            preds = torch.sigmoid(model(imgs))
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    auc = roc_auc_score(np.vstack(all_labels), np.vstack(all_preds))
    return model, auc



df = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

models_dict = {'efficientnet_b3': [], 'densenet121': [], 'resnet50': []}

for model_name in models_dict.keys():
    fold = 0
    for train_idx, val_idx in skf.split(df, df['label']):
        fold += 1
        print(f"Training {model_name} Fold {fold}")
        
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        train_ds = HistoDataset(train_df, transform=train_transform)
        val_ds = HistoDataset(val_df, transform=tta_transform)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        model = get_model(model_name)
        model, auc = train_fold(model, train_loader, val_loader, epochs=5)
        print(f"Fold {fold} AUC: {auc:.4f}")
        
        models_dict[model_name].append(model)



def predict_tta(models_list, df_test, tta_times=5):
    all_preds = []
    test_ds = HistoDataset(df_test, transform=tta_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    for model in models_list:
        model.eval()
        preds = []

        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(DEVICE)

                tta_pred = torch.zeros(imgs.size(0), 1).to(DEVICE)
                for _ in range(tta_times):
                    tta_pred += torch.sigmoid(model(imgs))

                tta_pred /= tta_times
                preds.append(tta_pred.cpu().numpy())

        all_preds.append(np.vstack(preds))

    return np.mean(np.array(all_preds), axis=0)



df_test = pd.DataFrame({'id': [f for f in os.listdir(os.path.join(DATA_DIR, 'test')) if f.endswith('.tif')]})

final_preds = []
weights = {'efficientnet_b3': 0.4, 'densenet121': 0.35, 'resnet50': 0.25}

for model_name, model_list in models_dict.items():
    preds = predict_tta(model_list, df_test, tta_times=5)
    final_preds.append(preds * weights[model_name])

final_preds = np.sum(final_preds, axis=0)

df_submission = pd.DataFrame({
    'id': [f.replace('.tif', '') for f in df_test['id']],
    'label': final_preds.flatten()
})

df_submission.to_csv('submission.csv', index=False)
print("Submission saved!")





