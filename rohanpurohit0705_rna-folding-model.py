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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


DATA_DIR = "/kaggle/input/stanford-rna-3d-folding"


train_sequences = pd.read_csv(f"{DATA_DIR}/train_sequences.csv")
train_labels = pd.read_csv(f"{DATA_DIR}/train_labels.csv")
test_sequences = pd.read_csv(f"{DATA_DIR}/test_sequences.csv")


print("Train Sequences Columns:", train_sequences.columns)
print("Train Labels Columns:", train_labels.columns)


print(train_sequences["target_id"].head())
print(train_labels["ID"].head())


train_sequences["base_target_id"] = train_sequences["target_id"].apply(lambda x: x.split("_")[0])
train_labels["base_target_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0])

train_df = train_sequences.merge(train_labels, left_on="base_target_id", right_on="base_target_id")

print("Merge successful! Train dataset shape:", train_df.shape)


def encode_sequence(seq, max_length=100):
    mapping = {"A": 0, "U": 1, "G": 2, "C": 3}  
    encoded = [mapping.get(base, 4) for base in seq]  
    encoded = encoded[:max_length] + [4] * (max_length - len(encoded))  
    return np.array(encoded, dtype=np.float32)


class RNADataset(Dataset):
    def __init__(self, dataframe, is_test=False):
        self.data = dataframe
        self.is_test = is_test
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = encode_sequence(row["sequence"])
        if self.is_test:
            return torch.tensor(sequence, dtype=torch.float32)
        else:
            label = row[["x_1", "y_1", "z_1"]].astype(float).values  
            return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32).squeeze()


batch_size = 32
train_dataset = RNADataset(train_df)
test_dataset = RNADataset(test_sequences, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class RNAFoldModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=64):
        super(RNAFoldModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


print("Columns in dataset:", train_labels.columns)


print(train_df.isna().sum())  
print(train_df.describe()) 


train_df[["x_1", "y_1", "z_1"]] = train_df[["x_1", "y_1", "z_1"]].fillna(train_df[["x_1", "y_1", "z_1"]].mean())
most_common_seq = train_df["all_sequences"].mode()[0]
train_df["all_sequences"] = train_df["all_sequences"].fillna(most_common_seq)


print(train_df.isna().sum())


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_df[["x_1", "y_1", "z_1"]] = scaler.fit_transform(train_df[["x_1", "y_1", "z_1"]])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNAFoldModel().to(device)
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
print("Training model...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

print("Generating predictions...")
model.eval()
predictions = []
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())

