import numpy as np 
import pandas as pd 
import torch


train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


print(f"Shape of training data : {train_df.shape}")
print(f"Shape of testing data : {test_df.shape}")


train_df.head()


# function that takes dataframe features and returns standarsized data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
def make_standard(data, is_train=False):
    if is_train:
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)
    return data_scaled


X = train_df.drop(columns=['BeatsPerMinute', 'id'])
y = np.array(train_df['BeatsPerMinute'])

# y scaler 
y_scaler = StandardScaler()
y_stan = y_scaler.fit_transform(y.reshape(-1, 1))

# standardized X
X_stan = make_standard(X, True)
print(f"Shape of standardized data : {X_stan.shape}")
print(f"Shape of y standardized : {y_stan.shape}")


# making a custom Dataset
import torch
from torch.utils.data import Dataset

class CustomData(Dataset):
    def __init__(self, features, vals):
        self.features = features 
        self.vals = vals

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        val = self.vals[idx]
        feat = torch.from_numpy(feat)
        val = torch.tensor(val)

        return feat, val


training_data = CustomData(X_stan, y_stan)


# lets make a dataloader
from torch.utils.data import DataLoader
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)

batch1 = next(iter(train_loader))
batch1[0].shape


import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                        nn.Linear(9, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                    )

    def forward(self, x):
        pred = self.net(x)
        return pred


device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNet()
model = model.double().to(device)


epoch_loss = []
epochs = 10
batch_size = 32
mse_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(epochs):
    batches = len(train_loader)
    model.train()
    batch_loss = 0
    for i, data in enumerate(train_loader):
        X, y = data
        X = X.to(device)
        y = y.to(device)
        # print(f"Shape of y : {y.shape}")
        preds = model(X)
        # print(f"Shape of preds : {preds.shape}")
        loss = mse_criterion(preds, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_loss += loss.item()

    epoch_loss.append((batch_loss/batches))
    print(f"Epoch : {epoch+1}, Loss : {epoch_loss[epoch]}")


print(preds[:5], y[:5])


import matplotlib.pyplot as plt
plt.plot(epoch_loss)


ids = test_df['id']
beats = []
features = make_standard(test_df.drop(columns=['id']))

features.shape


for i in range(features.shape[0]):
    feat = torch.tensor(features[i])
    pred = model(feat.to(device))
    beats.append(pred.item())

beats = np.array(beats)


beats.shape


y_scaler.inverse_transform(beats[:5].reshape(-1, 1))


beats = np.array(beats)
beats_ser = pd.Series(y_scaler.inverse_transform(beats.reshape(-1, 1)).reshape(-1))


df_submission = pd.DataFrame({
    "id": ids,
    "BeatsPerMinute": beats_ser
})

df_submission.to_csv("submissionS5E9.csv", index=False)


df_submission.head()




