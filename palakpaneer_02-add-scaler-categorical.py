import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
submit_df = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


# Imputing median for missing data
# train_df.isnull().sum()
# train_df.dtypes
  # Episode_Length_minutes -> float64
  # Guest_Popularity_percentage -> float64
  # Number_of_Ads -> float64

for df in [train_df, test_df]:  
    
    median_Episode_Length_minutes = df['Episode_Length_minutes'].median()
    median_Guest_Popularity_percentage = df['Guest_Popularity_percentage'].median()
    median_Number_of_Ads = df['Number_of_Ads'].median()
    
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(median_Episode_Length_minutes)
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(median_Guest_Popularity_percentage)
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(median_Number_of_Ads)

# train_df.isnull().sum()
# test_df.isnull().sum()


# Remove id
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])


# Deal with categorical data
# Episode_Title
train_df['Episode_Number'] = train_df['Episode_Title'].str.extract(r'Episode (\d+)').astype(float)
train_df = train_df.drop(columns=['Episode_Title'])

test_df['Episode_Number'] = test_df['Episode_Title'].str.extract(r'Episode (\d+)').astype(float)
test_df = test_df.drop(columns=['Episode_Title'])

# One-hot encoding
train_categorical = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
train_df = pd.get_dummies(train_df, columns=train_categorical, dtype=float, drop_first=True)

test_categorical = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
test_df = pd.get_dummies(test_df, columns=test_categorical, dtype=float, drop_first=True)

# train_df.columns
# test_df.columns


# Data check
train_df.shape


# Data check
test_df.shape


# Standard Scaler
scaler = StandardScaler()

target_col = 'Listening_Time_minutes'

train_numeric = train_df.columns[train_df.dtypes == 'float64']
train_numeric = [col for col in train_numeric if col != target_col]
test_numeric = test_df.columns[test_df.dtypes == 'float64']

train_df[train_numeric] = scaler.fit_transform(train_df[train_numeric])
test_df[test_numeric] = scaler.transform(test_df[test_numeric])

# train_df.head()
# test_df.head()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


torch.manual_seed(86)
num_batches = 100


class TrainDataset(Dataset):
    def __init__(self, df):
        self.X = df.drop('Listening_Time_minutes', axis=1).values
        self.y = df['Listening_Time_minutes'].values
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


train_dataset = TrainDataset(train_df)
train_dataloader = DataLoader(train_dataset, batch_size=num_batches, shuffle=True)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1)
        )
    def forward(self, x):
        output = self.classifier(x)
        return output
        


input_dim = train_df.drop('Listening_Time_minutes', axis=1).shape[1]
model = MLP(input_dim = input_dim)
model.to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 100
losses = []
rmses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    running_rmse = 0.0
    
    for x_batch, y_batch in train_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        rmse = torch.sqrt(loss)
        running_rmse += rmse.item()

    running_loss /= len(train_dataloader)
    running_rmse /= len(train_dataloader)

    print('epoch: {}, loss: {}, rmse: {}'.format(epoch, running_loss, running_rmse))
    losses.append(running_loss)
    rmses.append(running_rmse)


plt.plot(losses)


plt.plot(rmses)


class TestDataset(Dataset):
    def __init__(self, df):
        self.X = df.values
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32)


test_dataset = TestDataset(test_df)
test_dataloader = DataLoader(test_dataset, batch_size=num_batches, shuffle=False)


model.eval()
prediction = []

with torch.no_grad():
    for x_batch in test_dataloader:
        x_batch = x_batch.to(device)
        y_pred = model(x_batch).squeeze().cpu().numpy()
        prediction.extend(y_pred)
    


submit_df['Listening_Time_minutes'] = prediction
submit_df.head()


submit_df.to_csv("submission.csv", index=False)

