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


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# 1. Load data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')

# 2. Preprocess numerical features
num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.drop(['id', 'y'])
scaler = MinMaxScaler()
X_num = scaler.fit_transform(df_train[num_cols])

# 3. Preprocess categorical features
cat_cols = df_train.select_dtypes(include=['object']).columns
df_cats = df_train[cat_cols].replace('unknown', np.nan).fillna('missing')

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_cat = ohe.fit_transform(df_cats)

# 4. Combine features
X = np.hstack([X_num, X_cat])

# 5. Prepare target
y = df_train['y'].astype(int).values

# 6. Convert to PyTorch tensors and create DataLoader
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # float for BCEWithLogitsLoss

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 7. Define model
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Single output neuron for binary classification
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)  # No sigmoid here; use BCEWithLogitsLoss
        return x

# 8. Setup device, model, optimizer, and loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model(input_dim=X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

# 9. Training loop
def train(model, optimizer, loss_fn, dataloader, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')

# 10. Run training
train(model, optimizer, loss_fn, dataloader, epochs=15)



# 1. Load and preprocess test set using the SAME scalers/encoders
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
X_test_num = scaler.transform(df_test[num_cols])
df_test_cats = df_test[cat_cols].replace('unknown', np.nan).fillna('missing')
X_test_cat = ohe.transform(df_test_cats)
X_test = np.hstack([X_test_num, X_test_cat])
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# 2. Make predictions
model.eval()
with torch.no_grad():
    test_logits = model(X_test_tensor)
    test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
    test_preds = (test_probs >= 0.5).astype(int)

# 3. Save sample_submission.csv
submission = pd.DataFrame({
    'id': df_test['id'],
    'y': test_preds
})
submission.to_csv('submission.csv', index=False)
print(submission.head())


