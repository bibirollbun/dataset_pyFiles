import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


train_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")


print(train_df.isna().sum().sum())
print("#"*30)
print(test_df.isna().sum().sum())


train_df["label"].describe()


print(f"Any Inf: ==> {np.isinf(train_df.to_numpy()).sum()} && {np.isinf(test_df.to_numpy()).sum()}")


count_inf = pd.DataFrame({
    "train_inf": np.isinf(train_df).sum(),
    "test_inf": np.isinf(test_df).sum()
})


count_inf = count_inf[(count_inf["train_inf"] > 0) | (count_inf["test_inf"] > 0)]


print("Columns That contain Inf Values..... ")
print(f"There are ==> {len(count_inf)} Columns Contain inf value")
count_inf


X = train_df.drop(columns=["label"] + list(count_inf.index))
y= train_df["label"]


X_test = test_df.drop(columns=['label'] + list(count_inf.index))


tscv = TimeSeriesSplit(n_splits=5)
splits = list(tscv.split(X))
train_idx, val_idx = splits[-1] 
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]  # Add .iloc here
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx] 


scaler = RobustScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)


X_train_scaled.head()


# Select Best Features

selector = SelectKBest(score_func=mutual_info_regression, k=100)
X_train = selector.fit_transform(X_train_scaled, y_train)
X_val = selector.transform(X_val_scaled)




