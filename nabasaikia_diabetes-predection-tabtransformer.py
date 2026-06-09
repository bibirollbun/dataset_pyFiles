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


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

from tqdm import tqdm


#Load dataset

BASE_PATH = '/kaggle/input/playground-series-s5e12'

train_df = pd.read_csv(f'{BASE_PATH}/train.csv')
test_df = pd.read_csv(f'{BASE_PATH}/test.csv')
sample = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")

TARGET = 'diagnosed_diabetes'
ID_COL = 'id'


categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = [
    col for col in train_df.columns
    if col not in categorical_cols + [TARGET, ID_COL]
]

print('Categorical:', categorical_cols)
print('Numerical:', numerical_cols)


#Encode categorical columns

label_encoder = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoder[col] = le

#Scale numerical columns

scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])


#Pytorch dataset

class DiabetesDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.cat = df[categorical_cols].values.astype(np.int64)
        self.num = df[numerical_cols].values.astype(np.float32)
        self.is_test = is_test

        if not is_test:
            self.y = df[TARGET].values.astype(np.float32)

    def __len__(self):
        return len(self.num)

    def __getitem__(self, idx):
        if self.is_test:
            return self.cat[idx], self.num[idx]
        return self.cat[idx], self.num[idx], self.y[idx]


#Transformer model for tabular data

class TabularTransformer(nn.Module):
    def __init__(
        self,
        num_numerical,
        categories,
        emb_dim = 32,
        n_heads = 4,
        n_layers = 2,
        dropout = 0.1
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_size, emb_dim) for cat_size in categories
        ])

        self.num_linear = nn.Linear(num_numerical, emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = emb_dim,
            nhead = n_heads,
            dropout = dropout,
            batch_first = True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers
        )

        self.head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_cat, x_num):
        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embeds = torch.stack(cat_embeds, dim=1)

        num_embed = self.num_linear(x_num).unsqueeze(1)

        x = torch.cat([cat_embeds, num_embed], dim=1)
        x = self.transformer(x)
        x = x.mean(dim=1)

        return torch.sigmoid(self.head(x)).squeeze()


#Train-Validation Split

train_df, val_df = train_test_split(
    train_df,
    test_size = 0.2,
    stratify = train_df[TARGET],
    random_state = 42
)

#DataLoader

train_ds = DiabetesDataset(train_df)
val_ds = DiabetesDataset(val_df)
test_ds = DiabetesDataset(test_df, is_test=True)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512)
test_loader = DataLoader(test_ds, batch_size=512)


#Model Initialization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model  = TabularTransformer(
    num_numerical = len(numerical_cols),
    categories = [train_df[col].nunique() for col in categorical_cols]
).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


#Training Loop

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x_cat, x_num, y in tqdm(train_loader):
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x_cat, x_num)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    val_preds, val_targets = [], []

    with torch.no_grad():
        for x_cat, x_num, y in val_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)

            preds = model(x_cat, x_num).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(y.numpy())

    auc = roc_auc_score(val_targets, val_preds)
    print(f'Epoch {epoch+1} | Loss {total_loss:.4f} | Val AUC {auc:.5f}')

    


#Test prediction and submission

model.eval()
test_preds = []

with torch.no_grad():
    for x_cat, x_num in test_loader:
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        preds = model(x_cat, x_num).cpu().numpy()
        test_preds.extend(preds)


submission = pd.DataFrame({
    'id' : test_df[ID_COL],
    'Diagnosed_diabetes' : test_preds
})

submission.to_csv('submission.csv', index=False)
submission.head()




