import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time


VER=1


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
orig = pd.read_csv("/kaggle/input/calories-burnt-prediction/calories.csv", index_col='User_ID')
orig = orig.rename({"Gender":"Sex"},axis=1)
orig['id'] = np.arange( len(orig) ) + 1_000_000


train_df.info()


train_df.head()


test_df.info()


test_df.head()


FEATURES = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
TARGET = 'Calories'


for df in [train_df,test_df, orig]:
    df['Sex'] = df['Sex'].map({'male':0,'female':1}).astype('float32')


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.SiLU()  
        )
    
        self.layer2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.SiLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )

        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)

def train_model(model, train_loader, valid_loader, epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    best_valid_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range (epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                valid_loss += criterion(y_pred, y_batch).item() * X_batch.size(0)
                
        valid_loss /= len(valid_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {valid_loss:.6f}")

        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping triggered")
                break
    
    if best_model:
        model.load_state_dict(best_model)
    
    return model

def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch in data_loader:
            if isinstance(X_batch, list) or isinstance(X_batch, tuple):
                X_batch = X_batch[0].to(device)
            else:
                X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            predictions.append(y_pred.cpu().numpy())
    return np.concatenate(predictions).flatten()


FOLDS = 5
EPOCHS = 25  
BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(train_df))
pred = np.zeros(len(test_df))

for i, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
    print(f"\n{'#'*28}")
    print(f"{'#'*10} Fold {i+1} {'#'*10}")
    print(f"{'#'*28}")

    X_train = train_df.loc[train_idx, FEATURES].copy()
    y_train = np.log1p(train_df.loc[train_idx, TARGET])

    for k in range(4):
        X_train = pd.concat([X_train, orig[FEATURES]], axis=0)
        y_train = pd.concat([y_train, np.log1p(orig[TARGET])], axis=0)

    X_valid = train_df.loc[valid_idx, FEATURES].copy()
    y_valid = np.log1p(train_df.loc[valid_idx, TARGET])

    X_test = test_df[FEATURES].copy()

    print("Normalizing...", end='')
    norm_cols = [c for c in FEATURES if c not in []]
    means = X_train[norm_cols].mean()
    stds = X_train[norm_cols].std()
    stds = stds.replace(0, 1)
    X_train[norm_cols] = (X_train[norm_cols] - means) / stds
    X_valid[norm_cols] = (X_valid[norm_cols] - means) / stds
    X_test[norm_cols] = (X_test[norm_cols] - means) / stds
    print("done")
    
    start = time.time()

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_valid_tensor = torch.FloatTensor(X_valid.values)
    y_valid_tensor = torch.FloatTensor(y_valid.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test.values)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=512)

    model = NeuralNetwork(X_train.shape[1]).to(device)
    model = train_model(model, train_loader, valid_loader, EPOCHS, device)

    # Make predictions
    oof[valid_idx] = predict(model, valid_loader, device)
    pred += predict(model, test_loader, device)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
    print(f"Feature engineering & training time: {time.time() - start:.1f} sec")

pred /= FOLDS
    


full_rmse = np.sqrt(mean_squared_error(np.log1p(train_df[TARGET]), oof))
print(f"Overall CV RMSE: {full_rmse:.5f}")
np.save(f"oof_v{VER}",oof)


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


mn = train_df.Calories.min()
mx = train_df.Calories.max()
sample_submission['Calories'] = np.clip( np.expm1( pred ),mn,mx )
sample_submission.to_csv(f"submission_v{VER}.csv",index=False)
sample_submission[['id','Calories']].head()

