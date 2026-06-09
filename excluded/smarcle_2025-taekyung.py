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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, output_dim):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim_1, True)
        self.hidden_layer_1 = nn.Linear(hidden_dim_1, output_dim, True)
        self.activator = nn.ReLU()
        self.model = nn.Sequential(self.input_layer, self.activator, self.hidden_layer_1)

    def forward(self, X):
        return self.model(X)

print('model set')


model = MLP(784, 1024, 10)
print(list(model.modules()))


train = pd.read_csv('/kaggle/input/2024-smarcle-ks-3-fashion-mnist2/train.csv')
test = pd.read_csv('/kaggle/input/2024-smarcle-ks-3-fashion-mnist2/test.csv')

train.head()


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train.drop('label', axis=1), train['label'], test_size=0.1, random_state = 42)
X_train.head()


y_train.head()


from torch.utils.data import TensorDataset

batchsize = 64
train_X = np.array(X_train)
train_Y = np.array(y_train)
val_X = np.array(X_val)
val_Y = np.array(y_val)
test_X = np.array(test)

train_X = torch.tensor(train_X)
train_Y = torch.tensor(train_Y)
val_X = torch.tensor(val_X)
val_Y = torch.tensor(val_Y)
test_X = torch.tensor(test_X)

train_dataset = TensorDataset(train_X,train_Y)
val_dataset = TensorDataset(val_X, val_Y)
test_dataset = TensorDataset(test_X)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

train_dataloader


import tqdm

optim = torch.optim.Adam(model.parameters(), lr = 0.001)
loss = nn.CrossEntropyLoss()

print('gg')


for iter in tqdm.tqdm(range(30)):
    for x, y in train_dataloader:
        optim.zero_grad()

loss(model(x.float()), y).backward()
optim.step()


for iter in tqdm.tqdm(range(30)):
    for x, y in train_dataloader:
        y_answ = model(x.float())
        loss_value = loss(y_answ, y)
        optim.zero_grad()
        loss_value.backward()
        optim.step()


for iter in tqdm.tqdm(range(30)):
    for x, y in train_dataloader:
        y_answ = model(x.float())
        loss_value = loss(y_answ, y)
        optim.zero_grad()
        loss_value.backward()
        optim.step()

