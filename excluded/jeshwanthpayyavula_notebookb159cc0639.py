# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/regression-models-and-polynomial-features/sample_submission.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import time


import pandas as pd 


file_path="//kaggle/input/regression-models-and-polynomial-features/sample_submission.csv"
file_path1='/kaggle/input/regression-models-and-polynomial-features/test.csv'
file_path2='/kaggle/input/regression-models-and-polynomial-features/train.csv'
df=pd.read_csv(file_path)


test_df=pd.read_csv(file_path1)
train_df=pd.read_csv(file_path2)
s_df=pd.read_csv(file_path)


test_df.head()


train_df.head()


test_df.columns


s_df


train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

features = ['milage','model_year']

train_df['milage'] = pd.to_numeric(train_df['milage'], errors='coerce').fillna(0.0)
train_df['model_year'] = pd.to_numeric(train_df['model_year'], errors='coerce').fillna(0.0)


test_df['milage'] = pd.to_numeric(test_df['milage'], errors='coerce').fillna(0.0)
test_df['model_year'] = pd.to_numeric(test_df['model_year'], errors='coerce').fillna(0.0)


scaler = RobustScaler()

X = train_df[features].values
X_test = test_df[features].values

X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

y = train_df['price'].values
y_scaled = y / 1e4  # Scaling the target by 10,000



X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(2,64) 
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        x = self.fc4(x)
        return x
model = RegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 300
losses = []
start_time = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    
    y_pred = model(X_train)
    loss = torch.sqrt(criterion(y_pred, y_train))  # RMSE loss
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 30 == 0:
        print(f"Epoch: {epoch}/{epochs}, Loss: {loss.item():.4f}")

duration = time.time() - start_time
print(f"Training completed in {duration:.2f} seconds")

model.eval()
with torch.no_grad():
    y_val_pred = model(X_val)
    val_loss = torch.sqrt(criterion(y_val_pred, y_val)).item()
    print(f"Validation RMSE: {val_loss:.4f}")


model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze().numpy() * 1e4  


for i in range(10):
    diff = np.abs(y_val_pred[i].item()-y_val[i].item())
    print(f'{i}.) Predicted: {y_val_pred[i].item():8.2f} True: {y_val[i].item():8.2f} Diff: {diff:8.2f}')


torch.save(model.state_dict(),'Jeshwanth_regr_model.pt')


new_model =RegressionModel()  
new_model.load_state_dict(torch.load('Jeshwanth_regr_model.pt'))


submission = pd.DataFrame({'id': test_df['id'], 'price': predictions})
submission.to_csv('submission.csv', index=False)
print(f"Submission file saved as 'submission.csv' with combined features")
submission







