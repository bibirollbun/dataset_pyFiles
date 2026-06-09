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


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler


encoder = LabelEncoder()
categorical = train.select_dtypes(include='object').columns
for i in categorical:
    train[i] = encoder.fit_transform(train[i])
    test[i] = encoder.transform(test[i])


X = train.drop(['id', 'loan_paid_back'], axis=1)
y = train['loan_paid_back']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)


class LoanRepaymentNN(nn.Module):
    def __init__(self, input_dim):
        super(LoanRepaymentNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        return self.layers(x)

# Initialize model
input_dim = X_train.shape[1]
model = LoanRepaymentNN(input_dim)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 25
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")


model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_tensor)
    y_pred = (y_pred_prob >= 0.5).float()

acc = accuracy_score(y_test_tensor, y_pred)
print(f"\n✅ Test Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test_tensor, y_pred))


scaler = StandardScaler()
X = test.drop('id',axis=1)
X_train = scaler.fit_transform(X)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)


input_dim = 11  # number of features (excluding 'id')
model = LoanRepaymentNN(input_dim)
model


model.eval()
with torch.no_grad():
    y_pred = model(X_train_tensor)
    y_pred_class = y_pred.float()


y_pred_class = y_pred_class.view(-1)
y_test_tensor = y_test_tensor.view(-1)
X_train.shape,y_test_tensor.shape


y_pred_tensor = y_pred.detach().cpu()

# Example ID column (from your dataset)
ids = test['id']  # or whatever your ID column is called

# Ensure both have the same length
print("IDs:", len(ids))
print("Predictions:", len(y_test_tensor))

# Create the table
pred_df = pd.DataFrame({
    'id': ids[:len(y_test_tensor)],  # trim in case of mismatch
    'predicted_loan_pay_back': y_test_tensor.view(-1).numpy()
})
pred_df


pred_df.to_csv("submission.csv", index=False)

