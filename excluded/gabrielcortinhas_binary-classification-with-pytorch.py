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
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')


# Read the data 
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train.head()


y= train['y']
train= train.drop('y',axis=1)



# We want to check out any problems with our data - everything looks good on that front 
print(f"train summary: \n {train.isna().sum()}")
print(f"test summary: \n {test.isna().sum()}")



categorical_cols = test.select_dtypes(include=['object']).columns 
numerical_cols = test.select_dtypes(include=['int64','float64']).columns 

print(f"Our categorical columns : {categorical_cols.values}")
print(f"Our numerical columns : {numerical_cols.values}")



# Dealing with the day problem
numerical_cols = numerical_cols.drop('day')
categorical_cols = categorical_cols.append(pd.Index(['day']))


# Dealing with the pdays problem, create a marker for being contacted
train['was contacted'] = (train['pdays']!=-1).astype(int)
test['was contacted'] = (test['pdays']!=-1).astype(int)

# Replacing the pdays with a large number to get an idea of scale
train['pdays']=train['pdays'].replace(-1,100000)
test['pdays']= test['pdays'].replace(-1,100000)




print(f"our new categorical cols: {categorical_cols} " )
train.head()


encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)


train_encoded = encoder.fit_transform(train[categorical_cols])
test_encoded = encoder.transform(test[categorical_cols])


train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=train.index)
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=test.index)


train = pd.concat([train.drop(columns=categorical_cols), train_encoded_df], axis=1)
test = pd.concat([test.drop(columns=categorical_cols), test_encoded_df], axis=1)



scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class BinaryClassifier(nn.Module):
    def __init__(self,input_dim):
        super(BinaryClassifier,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,128), # Applies a standard learnable linear weighted sum + bias to the input, alters dimension 
            nn.BatchNorm1d(128), # Performing a Batch normalisation to stabilise the training
            nn.LeakyReLU(),# LeakyReLU is like ReLU but allows for a small slope at negative values, this avoids the "dying ReLU problem"
            nn.Dropout(0.1),# Performing a dropout to prevent overfitting
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(), # Ensures non-linearity 
            nn.Dropout(0.1),
            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x): 
        return self.net(x)
    


class EarlyStopping:
    def __init__(self,patience=5,mode='max'):
        self.patience = patience # How many epochs without improvement before stopping
        self.mode = mode # "max" if higher is better (e.g. AUC), "min" if lower is better (e.g. loss)
        self.best_score = None # Stores the best metric value we've seen
        self.counter = 0 # Consecutive epochs without seeing improvment
        self.early_stop = False  # Flags to True when we stop training

    def __call__(self,score,model):
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(model)
            # We didn't improve
        elif (self.mode == "max" and score <= self.best_score) or (self.mode=="min" and score >=self.best_score):
            self.counter += 1 
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # We did improve
            self.best_score = score 
            self.save_checkpoint(model)
            self.counter = 0 
    # Saving our best model
    def save_checkpoint(self,model):
        self.best_model_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Loading the best model
    def load_checkpoint(self,model):
        model.load_state_dict(self.best_model_wts)
            
            


X = train.values 
X_test = test.values


# This is just for faster testing
X_small = X[:1000]
y_small = y[:1000]


skf = StratifiedKFold(n_splits=5,shuffle=True)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# For tracking our metrics 
history = {
    "fold":[],
    "epoch":[],
    "train_loss":[],
    "val_loss": [],
    "val_auc": []
}

for fold, (train_idx,val_idx) in enumerate(skf.split(X,y)):
    print(f"\n Fold {fold+1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train,y_val = y[train_idx], y[val_idx]

    # Convert everything to tensors to use the model, making sure it's on the GPU
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device).unsqueeze(1)
    X_val_tensor   = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor   = torch.tensor(y_val.values, dtype=torch.float32, device=device).unsqueeze(1)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32, device=device)

    train_loader = DataLoader(TensorDataset(X_train_tensor,y_train_tensor),
                             batch_size=64,shuffle = True)
   

    # Need a model per fold 
    model = BinaryClassifier(X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay = 0.00001)

    early_stopping = EarlyStopping(patience=5,mode="max")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience =3 , verbose = True
    )
    
    # Actually training 
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for xb,yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds,yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        # Validation of each epoch
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits,y_val_tensor).item()
            val_preds = torch.sigmoid(val_logits).squeeze().cpu()
        val_auc = roc_auc_score(y_val,val_preds)
        # Step scheduler 
        scheduler.step(val_auc)
        
        # Save our metrics 
        history["fold"].append(fold+1)
        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")

        # We check early stopping using our class
        early_stopping(val_auc,model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    early_stopping.load_checkpoint(model)
    model.eval()
    with torch.no_grad():
        val_preds = torch.sigmoid(model(X_val_tensor)).squeeze().cpu().numpy()
        test_fold_preds = torch.sigmoid(model(X_test_tensor)).squeeze().cpu().numpy()
    
    oof_preds[val_idx] = val_preds
    test_preds += test_fold_preds/skf.n_splits

    auc = roc_auc_score(y_val,val_preds)
    print(f"Fold{fold+1} ROC AUC: {auc:.4f}")

# Look after our metrics for plotting
metrics_df = pd.DataFrame(history)


# Final CV score
cv_auc = roc_auc_score(y,oof_preds)
print(f"`n CV ROC AUC: {cv_auc:.4f}")




"""
Plot Loss, Accuracy, and ROC AUC curves across epochs, averaged over folds.
"""
# Group by epoch, then compute mean and std across folds
grouped = metrics_df.groupby("epoch").agg({
    "train_loss": ["mean", "std"],
    "val_loss": ["mean", "std"],
    "val_auc": ["mean", "std"]
})

# Extract values
epochs = grouped.index
train_loss_mean, train_loss_std = grouped["train_loss"]["mean"], grouped["train_loss"]["std"]
val_loss_mean, val_loss_std = grouped["val_loss"]["mean"], grouped["val_loss"]["std"]
val_auc_mean, val_auc_std = grouped["val_auc"]["mean"], grouped["val_auc"]["std"]

cmap = plt.get_cmap("Paired")
plt.figure(figsize=(17, 6))

# Loss plot
plt.subplot(1, 3, 1)
plt.plot(epochs, train_loss_mean, label="Train Loss", color=cmap(0))
plt.fill_between(epochs, train_loss_mean-train_loss_std, train_loss_mean+train_loss_std, alpha=0.2, color=cmap(0))
plt.plot(epochs, val_loss_mean, label="Val Loss", color=cmap(1))
plt.fill_between(epochs, val_loss_mean-val_loss_std, val_loss_mean+val_loss_std, alpha=0.2, color=cmap(1))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()


# ROC AUC plot
plt.subplot(1, 3, 2)
plt.plot(epochs, val_auc_mean, label="Val ROC AUC", color=cmap(4))
plt.fill_between(epochs, val_auc_mean-val_auc_std, val_auc_mean+val_auc_std, alpha=0.2, color=cmap(4))
plt.xlabel("Epoch")
plt.ylabel("ROC AUC")
plt.title("ROC AUC per Epoch")
plt.legend()

plt.tight_layout()
plt.show()




sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission = pd.DataFrame({
    "id": sub['id'],
    "y": test_preds 
})

submission.to_csv("submission.csv", index=False)


