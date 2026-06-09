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


# import statements

from sklearn.model_selection import train_test_split,KFold, StratifiedKFold

from torch.utils.data import Subset, DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.init as init
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



testi = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


main_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
final = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


main_train['Fertilizer Name'].value_counts()


train,labels = main_train.iloc[:,:-1], main_train.iloc[:,-1:]


for x in final.columns:
    train_uniques = main_train[x].unique()
    test_uniques = final[x].unique()

    print(f"Feature: {x}")
    print(f"  - Train: total unique value counts= {main_train[x].value_counts()}")
    print(f"  - Test : total unique value counts= {final[x].value_counts()}")
    print("-" * 80)



def process(func='encoding', df=None):
    df.drop(['id'],axis=1, inplace=True)
    if func == 'encoding' and df is not None:
        # Identify object (categorical) columns
        cat_cols = df.select_dtypes(include='object').columns
        # Apply one-hot encoding
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    return df



train = process(df=train)
final = process(df=final)


labels, uniques = labels['Fertilizer Name'].factorize()
# xtrain, xval, ytrain, yval = train_test_split(train, test, test_size=0.2, stratify=test)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
def scaling(scaler, train=None, val=None, test=None, cols=None):
    
    scal = scaler()

    # auto checks for continous columns :)
    if cols == None:
        one_hot_cols = [col for col in train.columns if set(train[col]) <= {0, 1}]
        cols = [col for col in train.columns if col not in one_hot_cols]

    # scaling the train,test
    scaled_train,scaled_test = train.copy(), test.copy()
    scaled_train[cols] = scal.fit_transform(train[cols])
    scaled_test[cols] = scal.transform(test[cols])
    
    if val is not None:
        # scaling the train
        scaled_val = val.copy()
        scaled_val[cols] = scal.transform(val[cols])
        return scaled_train, scaled_val, scaled_test
    

    return scaled_train, scaled_test



train, final = scaling(StandardScaler, train=train, val=None, test=final, cols=None)


# # training tensors
# xtrain_tensor = torch.tensor(xtrain.values, dtype=torch.float32)
# ytrain_tensor = torch.tensor(ytrain, dtype=torch.float32)

# # validation tensors
# xval_tensor = torch.tensor(xval.values, dtype=torch.float32)
# yval_tensor = torch.tensor(yval, dtype=torch.float32)

# # final testdata tensors
# final = torch.tensor(final.values, dtype=torch.float32)

# train_set = TensorDataset(xtrain_tensor, ytrain_tensor)
# val_set = TensorDataset(xval_tensor, yval_tensor)


class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.SiLU(),
            
            nn.Linear(64, 128),
            nn.SiLU(),
            # nn.Dropout(0.3),

            # nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            # nn.Dropout(0.4),
            # nn.SiLU(),

            # nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            # nn.Dropout(0.3),
            # nn.SiLU(),

            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.4),
            # nn.SiLU(),
            
            nn.Linear(128,7),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')  # or 'leaky_relu'\
        if m.bias is not None:
                init.zeros_(m.bias)
            
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)  # gamma = 1
        init.constant_(m.bias, 0) 

model = FNN()
model.apply(init_weights)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

# Hyperparameters
num_epochs = 10
n_splits = 5
batch_size = 512
learning_rate = 1e-3
weight_decay = 1e-3

# Prepare tensors
X_tensor = torch.tensor(train.values, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.long)
full_dataset = TensorDataset(X_tensor, y_tensor)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tensor, y_tensor)):
    print(f"\nFold {fold + 1}")

    # Subsets and loaders
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)

    # Re-initialize model per fold
    model = FNN().to(device)  # reinit the model
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8, amsgrad=True)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = loss_fn(output, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        # --- Validation ---
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                output = model(xb)
                loss = loss_fn(output, yb)
                val_loss += loss.item() * xb.size(0)
                preds = output.argmax(dim=1)
                correct += (preds == yb).sum().item()

        
        print(f"Epoch {epoch+1}: "
              f"Train Loss = {running_loss / len(train_loader.dataset):.4f}, "
              f"Val Loss = {val_loss / len(val_loader.dataset):.4f}, "
              f"Accuracy = {correct / len(val_loader.dataset):.4f}")



import torch
import pandas as pd

def submission(model, final, filename="submission.csv", device='cuda'):
    model.eval()
    test_tensor = torch.tensor(final.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(test_tensor)  # (N, num_classes)
        top3 = outputs.topk(3, dim=1).indices.cpu().numpy()

    df = pd.DataFrame({
        "id": pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')['id'],
        "Fertilizer Name": [' '.join([str(uniques[i]) for i in row]) for row in top3]
    })

    df.to_csv(filename, index=False)
    print(f"Saved submission file as: {filename}")

    return df


dff = submission(model, final, filename="submission.csv", device='cuda')




