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
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
print("Import Complete....")
print("Reading data....")
train_data = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/test.csv')
train_data = train_data.drop(columns=['ID','fullAddress','postcode'])
test_id = test_data['ID']
test_data = test_data.drop(columns=['ID','fullAddress','postcode'])
cat_cols = list(train_data.dtypes[train_data.dtypes == 'object'].index)
print(cat_cols)
train_data = pd.get_dummies(train_data,columns=cat_cols,drop_first=True)
test_data = pd.get_dummies(test_data,columns=cat_cols,drop_first=True)

X = train_data.drop(columns=['price']).fillna(train_data.mean(numeric_only=True))
y = train_data['price']
test_data = test_data.reindex(columns=X.columns, fill_value=0)
test_data = test_data.fillna(0)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# Scaling X
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
test_data = scaler_X.transform(test_data)
# Scaling y
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
print("NaNs in X_train:", np.isnan(X_train).any())
print("Infs in X_train:", np.isinf(X_train).any())
print("NaNs in y_train:", np.isnan(y_train).any())
print("Infs in y_train:", np.isinf(y_train).any())
print("data completely filtered scalled read to train....")
X_train_tensor = torch.tensor(X_train).float()
X_test_tensor = torch.tensor(X_test).float()
y_train_tensor = torch.tensor(y_train).float()
test_data_tensor = torch.tensor(test_data).float()

class Model(nn.Module):
  def __init__(self,input_features):
    super().__init__()
    self.network = nn.Sequential(
        nn.Linear(input_features, 128), 
        nn.ReLU(), 
        nn.Dropout(0.2), 
        nn.Linear(128, 64), 
        nn.ReLU(), 
        nn.Dropout(0.2), 
        nn.Linear(64, 1)
    )
  def forward(self,features):
    out = self.network(features)
    return out
print("Modle set....")
model = Model(X_train_tensor.shape[1])
model(X_train_tensor)
print("Model Tranning Started....")
# tranning 
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1300
for epoch in range(epochs):
    model.train()

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 130 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
print("Model Tranned....")
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)
y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
y_test_actual = y_test.values.reshape(-1, 1)
mse = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R^2 Score: {r2:.4f}")
print("predict test")
model.eval()
with torch.no_grad():
    test_pred_scaled = model(test_data_tensor)
test_pred = scaler_y.inverse_transform(test_pred_scaled.numpy())
submission = pd.DataFrame({
    'ID': test_id,
    'price': test_pred.flatten()
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")




