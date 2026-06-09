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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
train_data = train_data.dropna()
train_data
X_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
train_data


import matplotlib.pyplot as plt

plt.hist(train_data['country'], 6)



plt.hist(train_data['product'], 5)



plt.hist(train_data['store'], 3)



plt.hist(train_data['num_sold'], 6)


test_data = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

plt.hist(test_data['country'], 6)
test_data


plt.hist(test_data['product'], 5)






plt.hist(train_data['store'], 3)


y_train = train_data['num_sold']
X_train = train_data.drop(columns=['num_sold', 'id'])
X_train.shape


### No emissions!!!

concat = pd.concat([X_train, X_test], axis=1)
num_concat = concat.select_dtypes(include=['int', 'float'])
num_concat



concat = pd.get_dummies(concat)


concat.isna().sum()
X_train = concat[:221259]
X_test = concat[221259:]
X_train = X_train.drop(columns=['id'])
X_test = X_test.drop(columns=['id'])
X_test


import torch
from torch.utils.data import TensorDataset, DataLoader
X_train_t = torch.tensor(X_train.astype(float).values, dtype=torch.float)
y_train_t = torch.tensor(y_train.astype(float).values, dtype=torch.float)
X_test_t = torch.tensor(X_test.astype(float).values, dtype=torch.float)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t)

train_dataloader = DataLoader(train_dataset, batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64,shuffle=False)



from torch import nn, optim
model = nn.Sequential(
    nn.Linear(3680, 1024),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(128, 1)
)






class MAPELoss(nn.Module):
    def forward(self, y, y_pred):
        return torch.mean(torch.abs(y - y_pred) / (y + 1e-8)) * 100



loss_fn = MAPELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Commonly used Adam optimizer, if you want to read about it --> https://pytorch.org/docs/stable/generated/torch.optim.Adam.html


# Training our model using gradient descent
def run_train(model, loss_fn, dataloader, optimizer):
    total_loss = 0
    model.train()
    for X, y in dataloader:
        y_pred = model(X)
        loss = loss_fn(y_pred, y.unsqueeze(1))
        total_loss += loss.item()

        loss.backward()
        
        optimizer.step()
        
        optimizer.zero_grad()
    return total_loss
        


def run_eval(model, dataloader):
    y_test_pred = []
    model.eval()
    with torch.no_grad():
        for X in dataloader:
            X = X[0]
            pred = model(X)
            y_test_pred.append(pred)
    y_test_pred = torch.cat(y_test_pred, dim=0)
    return y_test_pred

            


def show_loss(loss_hist):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 1, 1)
    plt.title('Train Loss')
    plt.plot(np.arange(len(loss_hist)), loss_hist, label="Train Loss")
    plt.yscale('log')
    plt.grid()


NUM_EPOCHS = 50
loss_hist = []
for i in range(NUM_EPOCHS):
    loss = run_train(model, loss_fn, train_dataloader, optimizer)
    loss_hist.append(loss)
    print(f"Epoch:{i}, loss: {loss}")



show_loss(loss_hist)


output = run_eval(model, test_dataloader)
output.to_csv('submission.csv', index=False)

