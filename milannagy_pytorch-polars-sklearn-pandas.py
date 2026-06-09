import torch
import os
from pathlib import Path
import polars as pl

from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import gc


class CFG:
    root = Path(os.path.abspath('/kaggle/input/playground-series-s5e2'))
    work = Path(os.path.abspath('/kaggle/working/'))
    target = 'Price'
    trainyn = True
    old_sk = True
    model = work / 'model.pth'
cfg = CFG()


def getData():
    df_train = pl.read_csv(cfg.root / 'train.csv')
    df_test = pl.read_csv(cfg.root / 'test.csv')
    return df_train, df_test


df_train, df_test = getData()
df_train.to_pandas().info(), df_test.to_pandas().info()
df_train


def prepData():
    df_train, df_test = getData()
    col_ign = [cfg.target, 'id']
    cols_cat = [x for x in df_train.columns if df_train.select(pl.col(x)).dtypes[0] == pl.String and x not in col_ign]
    cols_num = [x for x in df_train.columns if df_train.select(pl.col(x)).dtypes[0] != pl.String and x not in col_ign]

    df_train = df_train.filter(pl.col('Weight Capacity (kg)').is_not_null())

    df_test_w = df_test.with_columns((pl.lit(None)).alias(cfg.target))
    
    df_merged = pl.concat((df_train, df_test_w))

    tr_imp_cat = SimpleImputer(fill_value='Unknown', strategy='constant')
    tr_imp_num = SimpleImputer(strategy='median')
    tr_ord = OrdinalEncoder()
    tr_rob = RobustScaler()
    

    if cfg.old_sk:
        tr_imp_cat.set_output(transform='pandas')
        tr_imp_num.set_output(transform='pandas')
        tr_ord.set_output(transform='pandas')
        tr_rob.set_output(transform='pandas')

        df_merged = df_merged.with_columns(pl.from_pandas(tr_imp_cat.fit_transform(df_merged.select(pl.col(cols_cat)).to_pandas())))
        df_merged = df_merged.with_columns(pl.from_pandas(tr_imp_num.fit_transform(df_merged.select(pl.col(cols_num)).to_pandas())))
        df_merged = df_merged.with_columns(pl.from_pandas(tr_imp_num.fit_transform(df_merged.select(pl.col(cols_num)).to_pandas())))
        df_merged = df_merged.with_columns(pl.from_pandas(tr_ord.fit_transform(df_merged.select(pl.col(cols_cat)).to_pandas())))
        df_merged = df_merged.with_columns(pl.from_pandas(tr_rob.fit_transform(df_merged.select(pl.col(cols_cat + cols_num)).to_pandas())))
        
    else:
        tr_imp_cat.set_output(transform='polars')
        tr_imp_num.set_output(transform='polars')
        tr_ord.set_output(transform='polars')
        tr_rob.set_output(transform='polars')

        df_merged = df_merged.with_columns(tr_imp_cat.fit_transform(df_merged.select(pl.col(cols_cat))))
        df_merged = df_merged.with_columns(tr_imp_num.fit_transform(df_merged.select(pl.col(cols_num))))
        df_merged = df_merged.with_columns(tr_ord.fit_transform(df_merged.select(pl.col(cols_cat))))
        df_merged = df_merged.with_columns(tr_rob.fit_transform(df_merged.select(pl.col(cols_cat + cols_num))))
        

    df_train = df_merged.filter(pl.col(cfg.target).is_not_null())
    df_test = df_merged.filter(pl.col(cfg.target).is_null()).select(pl.all().exclude(cfg.target))
    
    return df_train, df_test

df_train, df_test = prepData()


# df_train.to_pandas().info(), df_test.to_pandas().info()


class CustomDataset(Dataset):
    def __init__(self, in_data: np.ndarray, in_labels: np.ndarray):
        self.data = in_data
        self.labels = in_labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {'features': torch.tensor(self.data[idx], dtype= torch.float32), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

col_ign = [cfg.target, 'id']
X = df_train.select(pl.all().exclude(col_ign)).to_numpy()
y = df_train.select(pl.col(cfg.target)).to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state= 42)

ds_train = CustomDataset(X_train, y_train)
ds_test = CustomDataset(X_test, y_test)
del df_train
del X_train
del y_train
del X_test
del y_test


batch = 20000
num_workers = os.cpu_count()
dl_train = DataLoader(ds_train, num_workers= num_workers, batch_size= batch)
dl_test = DataLoader(ds_test, num_workers= num_workers, batch_size= batch)


class CustomModel(nn.Module):
    def __init__(self, in_shape: int, in_hidden_units: int, in_output: int):
        super(CustomModel, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features= in_shape, out_features= in_hidden_units),
            nn.BatchNorm1d(in_hidden_units),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(in_features= in_hidden_units, out_features= in_hidden_units),
            nn.BatchNorm1d(in_hidden_units),
            nn.Linear(in_features= in_hidden_units, out_features= in_output)
        )

    def forward(self, x):
        return self.seq(x)

col_ign = [cfg.target, 'id']

model = CustomModel(X.shape[1], 2000, 1)



X.shape, y.shape


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

if cfg.trainyn == True:
    class RMSELoss(nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, y_pred, y_true):
            if y_pred.shape != y_true.shape:
                raise ValueError("Input tensors must have the same shape")
    
            squared_error = (y_pred - y_true)**2
            rmse = torch.sqrt(torch.mean(squared_error))
            return rmse
    
    

    
    loss_fn = RMSELoss()
    optimiser = Adam(model.parameters(), lr= 0.001)
    epochs = 5000
    c = 0
    
    model.to(device)
    
    for i in range(epochs):
        loss_train = 0
        for j, b in enumerate(dl_train):
            X = b['features'].to(device)
            y = b['labels'].to(device)
            model.train()
            
            optimiser.zero_grad()
            outputs = model(X.to(device))
            loss = loss_fn(outputs, y.float())
            loss_train += loss
            loss.backward()
            optimiser.step()
        # print(f'{loss_train}')
    
        with torch.no_grad():
            model.eval()
            loss_test_outer = 0
            for j_test, b_test in enumerate(dl_test):
                X_test = b['features'].to(device)
                y_test = b['labels'].to(device)
                preds = model(X_test.to(device))
                loss_test = loss_fn(preds, y_test.to(device))
                loss_test_outer += loss_test
            
        if i == 0:
            prev_loss = loss_test_outer / (j_test+1)
            best_model = model
            best_model_id = i
        elif (loss_test_outer / (j_test+1) < prev_loss):
            prev_loss = loss_test_outer / (j_test+1)
            best_model = model
            c = 0
            best_model_id = i
        else:
            c += 1
    
        if c == 5:
            break
            
        print(f'epoch: {i}, train loss: {loss_train / (j+1)}, test loss: {loss_test_outer / (j_test+1)}, best model iteration: {best_model_id}')
    model = best_model.to(device)  
    torch.save(model.state_dict(), cfg.model)
else:
    model.load_state_dict(torch.load(cfg.model))


try:
    del X_train
    del X_test
    del y_train
    del y_test
    del b
    del b_test
    del y
    del X
    gc.collect()
    torch.cuda.empty_cache()
except:
    pass
X_sub = torch.tensor(df_test.select(pl.all().exclude(col_ign)).to_numpy(), dtype=torch.float32).to(device)

model.to(device)
list_preds = list()
step = 1000
with torch.no_grad():
    model.eval()
    for i in range(int(len(X_sub) / step)):
        X_sub_iter = X_sub[i * step: i*(step + step)]
        preds_sub = model(X_sub)
        list_preds.append(preds_sub.to('cpu').detach().numpy())
        
model = model.to('cpu')


%%time
preds_sub = preds_sub.to('cpu').detach().numpy()
pred_sub_numpy = np.asarray(preds_sub)
df_sub = df_test.select(pl.col('id')).with_columns((pl.Series(pred_sub_numpy[:,0])).alias(cfg.target))
df_sub.write_csv(cfg.work / 'submission.csv')

