import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler 
import warnings
import os
warnings.filterwarnings("ignore")

root_dir ="/kaggle/input/playground-series-s5e5"
train_df= pd.read_csv(os.path.join(root_dir,'train.csv'))
test_df = pd.read_csv(os.path.join(root_dir,'test.csv'))

train_df= train_df.drop(['id'],axis=1)
train_df['Sex'] = train_df['Sex'].map({'male':0,"female":1}).astype('float32')
test_df['Sex'] = test_df['Sex'].map({'male':0,"female":1}).astype('float32')

features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
target ='Calories'


X = train_df[features]
y= train_df[target]
scalar= StandardScaler()
X_scaled = scalar.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y , test_size=0.1, random_state=42)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
        


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

# RMSLE function
def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    log_true = np.log1p(np.maximum(y_true, 0))
    log_pred = np.log1p(np.maximum(y_pred, 0))
    
    return np.sqrt(np.mean((log_true - log_pred) ** 2))


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, learning_rate, momentum, hidden_dim=[32, 20, 8], output_dim=1):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], output_dim)
        )
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

    def forward(self, x):
        return self.model(x)

    def train_model(self, X_train, y_train, epochs=100, batch_size=10, verbose=1):
        if isinstance(y_train, pd.Series):
            y_train = torch.tensor(y_train.values, dtype=torch.float32)
        if y_train.dim() == 1:
            y_train = y_train.view(-1, 1)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                output = self.forward(xb)
                loss = self.criterion(output, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        # Save model
        save_dir = '/kaggle/working/'
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'model.pth')
        # torch.save(self.model.state_dict(), model_path)
        # print(f"✅ Model saved to {model_path}")

    def evaluate(self, X_test, y_test):
        self.eval()
        with torch.no_grad():
            pred = self.forward(X_test).cpu().numpy()
            y_true = y_test
            score = rmsle(y_true, pred)
        print(f"✅ Test RMSLE: {score:.4f}")
        return score

mlp = MLPRegressor(input_dim=len(features), learning_rate=0.01, momentum=0.9, hidden_dim=[32, 20, 8], output_dim=1)

mlp.train_model(X_train_tensor, y_train_tensor, epochs=50, batch_size=10000)

mlp.evaluate(X_test_tensor, y_test_tensor)




