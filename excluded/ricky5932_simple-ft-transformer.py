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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_csv_path = '/kaggle/input/playground-series-s5e1/train.csv'
#load the data
train = pd.read_csv(train_csv_path)
train = train.dropna()



# preprocess the train data
train['date'] = pd.to_datetime(train['date'])
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
# time embedding
train['sin_month'] = np.sin(2*np.pi*train['month']/12)
train['cos_month'] = np.cos(2*np.pi*train['month']/12)
train['sin_day'] = np.sin(2*np.pi*train['day']/31)
train['cos_day'] = np.cos(2*np.pi*train['day']/31)
# year embedding
train['sin_year'] = np.sin(2*np.pi*train['year']/365)
train['cos_year'] = np.cos(2*np.pi*train['year']/365)
# add time embedding together
train['sin_time'] = train['sin_month'] + train['sin_day'] + train['sin_year']
train['cos_time'] = train['cos_month'] + train['cos_day'] + train['cos_year']
train = train.drop(['sin_month', 'cos_month', 'sin_day', 'cos_day', 'sin_year', 'cos_year'], axis=1)
# drop year, month, day
train = train.drop(['year', 'month', 'day'], axis=1)
train['date'] = train['date'].dt.strftime('%Y%m%d')

train = train.drop('id', axis=1)

# change the country, product, store into numerical values
train['country'] = train['country'].replace({'Canada': 0, 'Kenya': 1, 'Italy': 2, 'Norway': 3, 'Finland': 4, 'Singapore': 5})
train['product'] = train['product'].replace({'Holographic Goose': 0, 'Kaggle': 1, 'Kaggle Tiers': 2, 'Kerneler': 3, 'Kerneler Dark Mode': 4})
train['store'] = train['store'].replace({'Discount Stickers': 0, 'Stickers for Less': 1, 'Premium Sticker Mart': 2})
#move the num_sold to the last column
num_sold = train.pop('num_sold')
train['num_sold'] = num_sold
train = train.drop('date', axis=1)
print(train.head())


print(train['country'].unique())
print(train['product'].unique())
print(train['store'].unique())


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class TabDataset(Dataset):
    def __init__(self, X,test=False):
        if test:
            self.X = X[:,:]
        else:
            self.X = X[:,:-1]
            self.y = X[:,-1]
        self.test = test
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.test:
            return self.X[idx,:].reshape(5)
        else:
            return self.X[idx,:].reshape(5), self.y[idx].reshape(1)


class Ft_transformer(nn.Module):
    def __init__(
        self,
        ts_input_size=5,
        d_model=32,
        nhead=16,
        num_layers=3,
        dim_feedforward=32,
        dropout=0.1
    ):
        super(Ft_transformer, self).__init__()
        self.transformerEncodeLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformerEncoder = nn.TransformerEncoder(self.transformerEncodeLayer, num_layers=num_layers)

        self.catagorical = [6, 3, 5]

        self.embedding_catagorical = nn.ModuleList([nn.Embedding(cat, d_model) for cat in self.catagorical])

        self.embedding_pos1 = nn.Linear(1, d_model)
        self.embedding_pos2 = nn.Linear(1, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.fc_out = nn.Linear(d_model, 1)

        self.d_model = d_model

    def forward(self, x_tab):
        batch_size,  _ = x_tab.shape

        x_emb_cat = torch.cat([emb(x_tab[:, i].long()) for i, emb in enumerate(self.embedding_catagorical)], dim=-1).reshape(batch_size, -1, self.d_model)
        x_emb_pos1 = self.embedding_pos1(x_tab[:, 3].unsqueeze(1)).unsqueeze(1)
        x_emb_pos2 = self.embedding_pos2(x_tab[:, 4].unsqueeze(1)).unsqueeze(1)
        x_emb_pos = torch.cat([x_emb_pos1, x_emb_pos2], dim=1)
        x_emb = torch.cat([x_emb_cat, x_emb_pos], dim=1)

        cls_token = self.cls_token.repeat(batch_size, 1, 1)

        x_with_cls = torch.cat([cls_token, x_emb], dim=1)

        x_transformed = self.transformerEncoder(x_with_cls)

        out = self.fc_out(x_transformed[:, 0, :])

        out = torch.sigmoid(out) * 5950 + 5

        return out




# test the Ft_transformer
model = Ft_transformer(ts_input_size=5)
x_tab = torch.zeros(3, 5)
out = model(x_tab)
print(out.shape)


print(train.to_numpy().shape)
datasets = TabDataset(train.to_numpy())
# random split the data into train and test by torch
train_size = int(0.8 * len(datasets))
test_size = len(datasets) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(datasets, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


# define the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Ft_transformer(ts_input_size=5).to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def MAPELoss(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))


# define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

# define the loss function
criterion = MAPELoss

patience = 100
epoch = 10000



train_losses = []
val_losses = []
min_val_loss = np.inf
counter = 0
best_model = None
for i in range(epoch):
    train_loss = 0
    val_loss = 0
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = x.float()
        y = y.float()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.float()
            y = y.float()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        counter = 0
        best_model = model
    else:
        counter += 1

    if counter >= patience:
        print(f'Early stopping at epoch {i}')
        break

    scheduler.step(val_loss)

    print(f'Epoch {i}, Train Loss: {train_loss}, Val Loss: {val_loss}', end='\r')


plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='val loss')
plt.legend()
plt.show()


test_csv_path = '/kaggle/input/playground-series-s5e1/test.csv'
test = pd.read_csv(test_csv_path)
print(test.head())


# preprocess the test data
test['date'] = pd.to_datetime(test['date'])
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
# time embedding
test['sin_month'] = np.sin(2*np.pi*test['month']/12)
test['cos_month'] = np.cos(2*np.pi*test['month']/12)
test['sin_day'] = np.sin(2*np.pi*test['day']/31)
test['cos_day'] = np.cos(2*np.pi*test['day']/31)
# year embedding
test['sin_year'] = np.sin(2*np.pi*test['year']/365)
test['cos_year'] = np.cos(2*np.pi*test['year']/365)
# add time embedding together
test['sin_time'] = test['sin_month'] + test['sin_day'] + test['sin_year']
test['cos_time'] = test['cos_month'] + test['cos_day'] + test['cos_year']
test = test.drop(['sin_month', 'cos_month', 'sin_day', 'cos_day', 'sin_year', 'cos_year'], axis=1)
# drop year, month, day
test = test.drop(['year', 'month', 'day'], axis=1)
test['date'] = test['date'].dt.strftime('%Y%m%d')
# change the country, product, store into numerical values
test['country'] = test['country'].replace({'Canada': 0, 'Kenya': 1, 'Italy': 2, 'Norway': 3, 'Finland': 4, 'Singapore': 5})
test['product'] = test['product'].replace({'Holographic Goose': 0, 'Kaggle': 1, 'Kaggle Tiers': 2, 'Kerneler': 3, 'Kerneler Dark Mode': 4})
test['store'] = test['store'].replace({'Discount Stickers': 0, 'Stickers for Less': 1, 'Premium Sticker Mart': 2})
#move the num_sold to the last column
#drop date
test = test.drop('date', axis=1)
test = test.drop('id', axis=1)
print(test.head())


# catorgorical data unique values
print(test['country'].unique())
print(test['product'].unique())
print(test['store'].unique())


#dataloaders
test_dataset = TabDataset(test.to_numpy(),test=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


#predict the test data
from tqdm import tqdm
best_model.eval()
preds = []
with torch.no_grad():
    for x in tqdm(test_loader):
        x = x.to(device)
        x = x.float()
        y_pred = best_model(x)
        preds.append(y_pred.item())
        


# save the predictions as submission.csv
submission = pd.DataFrame()
submission['id'] = range(230130, 230130 + len(preds))
submission['num_sold'] = preds
submission.to_csv('submission.csv', index=False)

