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
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

import sklearn.preprocessing
import matplotlib


test_df = pd.read_csv("/kaggle/input/ml2021spring-hw1/covid.test.csv")
train_df = pd.read_csv("/kaggle/input/ml2021spring-hw1/covid.train.csv")

print(test_df.shape, train_df.shape)
print(test_df.head(), train_df.head())


X_train = train_df.drop(columns = ['id', 'tested_positive']).values
y_train = train_df['tested_positive'].values

X_test = test_df.drop(columns = ['id']).values

print(X_train.shape, y_train.shape, X_test.shape)


class CovidDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype = torch.float32)
        else:
            self.y = None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]
full_set = CovidDataset(X_train, y_train)
n_total = len(full_set)

n_val = int(n_total*0.2)
n_train = n_total-n_val

val_set, train_set = random_split(full_set, [n_val, n_train])
train_dataLoader = DataLoader(train_set, batch_size = 32, shuffle = True)
val_dataLoader = DataLoader(val_set, batch_size = 32, shuffle = True)



class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x)

model = DNN(input_dim = 93)
print(model)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)


n_epo = 100
train_curve, val_curve = [], []

for epo in range(n_epo):
    model.train()
    sq_loss, count = 0.0, 0
    for X_batch, y_batch in train_dataLoader:
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sq_loss += ((y_pred - y_batch) ** 2).sum().item()
        count += y_batch.size(0)

    train_rmse = np.sqrt(sq_loss / count)
    train_curve.append(train_rmse)

    model.eval()
    sq_loss, count = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_dataLoader:
            y_pred = model(X_batch).squeeze()
            sq_loss += ((y_pred - y_batch) ** 2).sum().item()
            count += y_batch.size(0)

    val_rmse = np.sqrt(sq_loss / count)
    val_curve.append(val_rmse)
    print(f"Epoch {epo+1}, Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")



import matplotlib.pyplot as plt
plt.figure(figsize = (8, 5))
plt.plot(range(1, n_epo+1), train_curve, label="Train RMSE")
plt.plot(range(1, n_epo+1), val_curve, label="Val RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Training vs Validation RMSE")
plt.legend()
plt.grid(True)
plt.show()


test_set = CovidDataset(X_test, y=None)
test_dataLoader = DataLoader(test_set, batch_size = 32, shuffle = False)

model.eval()
preds = []
with torch.no_grad():
    for X_batch in test_dataLoader:
        y_pred = model(X_batch).squeeze()
        preds.append(y_pred.tolist())

output = pd.DataFrame({
    'id': range(len(preds)),
    'tested_positive': preds
})
output.to_csv("submission.csv", index=False)
print("Finish!")




