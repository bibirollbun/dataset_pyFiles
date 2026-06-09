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


import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm


train_df = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/train.csv')
test_df = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/test.csv')


train_img_dir = '/kaggle/input/petfinder-pawpularity-score/train/'
test_img_dir = '/kaggle/input/petfinder-pawpularity-score/test/'


# Просмотр данных
print(train_df.head())
print(f'Размер обучающей выборки: {len(train_df)}')
print(f'Размер тестовой выборки: {len(test_df)}')

# Визуализация распределения Pawpularity 
plt.hist(train_df['Pawpularity'], bins=50)
plt.title('Распределение Pawpularity')
plt.show()

meta_cols = [
    'Subject Focus','Eyes','Face','Near','Action','Accessory',
    'Group','Collage','Human','Occlusion','Info','Blur'
]

# Приводим мета-колонки к числовому типу и заполняем NaN нулями
for col in meta_cols:
    if col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0).astype(np.float32)
    if col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0).astype(np.float32)

# Убедимся, что Pawpularity числовой
train_df['Pawpularity'] = pd.to_numeric(train_df['Pawpularity'], errors='coerce').fillna(0).astype(np.float32)

# Быстрая проверка типов
print(train_df[meta_cols].dtypes)
print("Pawpularity dtype:", train_df['Pawpularity'].dtype)


num_bins = 10
train_df['bin'] = pd.qcut(train_df['Pawpularity'], q=num_bins, labels=False)


train_data, val_data = train_test_split(
train_df,
test_size=0.2,
random_state=42,
stratify=train_df['bin']
)


train_data = train_data.drop(columns='bin')
val_data = val_data.drop(columns='bin')
train_df = train_df.drop(columns='bin')


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[meta_cols] = scaler.fit_transform(train_data[meta_cols])
val_data[meta_cols]   = scaler.transform(val_data[meta_cols])
test_df[meta_cols]    = scaler.transform(test_df[meta_cols])


class PetDataset(Dataset):
    def __init__(self, df, img_dir, meta_cols, transform=None, is_test=False):
        self.df = df.reset_index(drop=True).copy()
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.meta_cols = meta_cols

        # На всякий случай гарантируем numeric (еще раз) и float32
        self.df[self.meta_cols] = self.df[self.meta_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        if not is_test:
            self.df['Pawpularity'] = pd.to_numeric(self.df['Pawpularity'], errors='coerce').fillna(0).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Id'] + '.jpg')
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        meta = row[self.meta_cols].values.astype(np.float32)
        meta = torch.from_numpy(meta)

        if self.is_test:
            return img, meta, row['Id']
        else:
            label = np.array(row['Pawpularity'], dtype=np.float32)
            label = torch.from_numpy(label)
            # Если label одномерный скаляр, превращаем в тензор float
            if label.dim() == 0:
                label = label.unsqueeze(0)
            return img, meta, label.squeeze(0)


IMG_SIZE = 380

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.08, 0.08, 0.04),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])



batch_size = 16
train_ds = PetDataset(train_data, train_img_dir, meta_cols, transform=train_tfms, is_test=False)
val_ds   = PetDataset(val_data,   train_img_dir, meta_cols, transform=val_tfms,   is_test=False)
test_ds  = PetDataset(test_df,    test_img_dir,  meta_cols, transform=val_tfms,   is_test=True)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)


from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
class PawpularityModel(nn.Module):
    def __init__(self, num_meta=12):
        super().__init__()

        self.backbone = efficientnet_b4(
            weights=EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.meta_net = nn.Sequential(
            nn.Linear(num_meta, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.regressor = nn.Sequential(
            nn.Linear(in_features + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, img, meta):
        img_feat = self.backbone(img)
        meta_feat = self.meta_net(meta)
        x = torch.cat([img_feat, meta_feat], dim=1)
        return self.regressor(x).squeeze(1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PawpularityModel(num_meta=len(meta_cols)).to(device)

# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss(beta=10.0)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for param in model.backbone.parameters():
    param.requires_grad = False




best_rmse = 1e9
num_epochs = 30

for epoch in range(num_epochs):
    if epoch == 3:
        for param in model.backbone.parameters():
            param.requires_grad = True
    model.train()
    running_loss = 0.0
    n_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for img, meta, y in pbar:
        # y приходит как tensor float32 (скаляр) из Dataset
        img = img.to(device)
        meta = meta.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()
        preds = model(img, meta)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)
        n_samples += img.size(0)
        pbar.set_postfix({'loss': loss.item()})

    scheduler.step()
    epoch_loss = running_loss / n_samples

    # Валидация
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for img, meta, y in val_loader:
            img = img.to(device)
            meta = meta.to(device)
            preds = model(img, meta).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(y.numpy())

    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
    print(f"Epoch {epoch+1}: Train loss {epoch_loss:.4f} | Val RMSE {val_rmse:.4f}")

    if val_rmse < best_rmse:
        best_rmse = val_rmse
        torch.save(model.state_dict(), 'best.pth')
        print("Saved best model:", best_rmse)


model.load_state_dict(torch.load('best.pth', map_location=device))
model.eval()

ids_all = []
preds_all = []
with torch.no_grad():
    for img, meta, ids in tqdm(test_loader, desc="Inference"):
        img = img.to(device)
        meta = meta.to(device)
        preds = model(img, meta).cpu().numpy()
        preds = np.clip(preds, 0, 100)   # ограничиваем диапазон
        ids_all.extend(ids)
        preds_all.extend(preds)



submission = pd.DataFrame({'Id': ids_all, 'Pawpularity': preds_all})
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv, sample:")
print(submission.head())

