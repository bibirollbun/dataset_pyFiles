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
from sklearn.preprocessing import LabelEncoder, StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


SEED = 12
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_df.head()


train_df = train_df.drop("id" , axis = 1)
print(train_df.isna().sum())
train_df.head()


train_df["Weight Capacity (kg)"].fillna(train_df["Weight Capacity (kg)"].mean(), inplace=True)
train_df.isna().sum()


le = LabelEncoder()

for col in train_df.select_dtypes(include=["object"]).columns:
    train_df[col] = le.fit_transform(train_df[col])

train_df.head()


X , y = train_df.drop("Price" , axis = 1) , train_df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.2 , random_state = SEED)

len(X_train), len(y_train), len(X_test), len(y_test)


class BackpackDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = BackpackDataset(X_train, y_train)
test_dataset = BackpackDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_dim , hidden_dim , output_dim):
        super(MLP, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim , hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim*2 , hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dim , output_dim)
        )

    def forward(self , input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1D
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation, since this is regression
        return x


def train(model, epochs, criterion, optim, train_loader, test_loader, device=DEVICE):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optim.zero_grad()
            y_preds = model(inputs)
            loss = torch.sqrt(criterion(y_preds, targets))
            loss.backward()
            optim.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0.0
        all_preds, all_targets = [] , []

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                test_preds = model(inputs)
            
                all_preds.append(test_preds)
                all_targets.append(targets)

                loss = torch.sqrt(criterion(test_preds, targets))
                test_loss += loss.item()
        
        test_loss /= len(test_loader)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Test RMSE: {test_loss:.4f}")
        print("-" * 80)

    return all_preds, all_targets 


mlp = MLP(X_train.shape[1], 32 , 1).to(DEVICE)
optimizer = torch.optim.Adam(params = mlp.parameters() , lr = LEARNING_RATE)
criterion = nn.MSELoss()

cnn = CNN(X_train.shape[1]).to(DEVICE)
optimizer_cnn = torch.optim.Adam(params = cnn.parameters() , lr = LEARNING_RATE)
criterion_cnn = nn.MSELoss()


preds , targets = train(mlp, EPOCHS , criterion , optimizer, train_loader , test_loader)


preds_cnn , targets_cnn = train(cnn , EPOCHS , criterion_cnn , optimizer_cnn , train_loader , test_loader , DEVICE)


preds = np.vstack([p.cpu().detach().numpy() for p in preds])
targets = np.vstack([t.cpu().detach().numpy() for t in targets])

plt.scatter([i for i in range(len(preds))],preds , color = "r" , label = "preds" , alpha = 1 )
plt.scatter([i for i in range(len(targets))],targets , color = "b" , label = "targets" , alpha=0.01)


preds_cnn = np.vstack([p.cpu().detach().numpy() for p in preds_cnn])
targets_cnn = np.vstack([t.cpu().detach().numpy() for t in targets_cnn])

plt.scatter([i for i in range(len(preds_cnn))],preds_cnn , color = "r" , label = "preds" , alpha = 1 )
plt.scatter([i for i in range(len(targets_cnn))],targets_cnn , color = "b" , label = "targets" , alpha=0.01)

