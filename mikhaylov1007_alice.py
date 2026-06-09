pip install -U scikit-learn imbalanced-learn


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from tqdm.notebook import tqdm
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from collections import Counter


data_train = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train_sessions.csv')
data_test = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/test_sessions.csv')


data_train


data_test


data_train.info()


data_test.info()


data_train.fillna(0, inplace=True)
data_test.fillna(0, inplace=True)


def add_time_diffs_at_seconds(df):
    time_cols = sorted([col for col in df.columns if col.startswith('time')])
    data_test[time_cols] = data_test[time_cols].apply(pd.to_datetime, errors='coerce')
    data_train[time_cols] = data_train[time_cols].apply(pd.to_datetime, errors='coerce')

    for i in range(len(time_cols)-1):
        col1 = time_cols[i]
        col2 = time_cols[i+1] 
        diff_col = f'diff{col1}_{col2}'

        df[diff_col] = (df[col2]-df[col1]).dt.total_seconds()

    return df


data_test = add_time_diffs_at_seconds(data_test)
data_train = add_time_diffs_at_seconds(data_train)


def time_features_for_alice(df):
    time_cols = sorted([col for col in df.columns if col.startswith('time')])
    data_test[time_cols] = data_test[time_cols].apply(pd.to_datetime, errors='coerce')
    data_train[time_cols] = data_train[time_cols].apply(pd.to_datetime, errors='coerce')


    for i in range(len(time_cols)):
        
        col = time_cols[i]
        df[f'year_{col}'] = df[col].dt.year
        df[f'month_{col}'] = df[col].dt.month
        df[f'day_{col}'] = df[col].dt.day
        df[f'hour_{col}'] = df[col].dt.hour
        df[f'minute_{col}'] = df[col].dt.minute
        df[f'second_{col}'] = df[col].dt.second
        df[f'weekday_{col}'] = df[col].dt.dayofweek
        df = df.drop(col, axis=1)

    return df


data_test = time_features_for_alice(data_train)
data_train = time_features_for_alice(data_test)


def cap_large_values_in_time_cols(df, threshold=864000):

    time_cols = [col for col in df.columns if col.startswith('diff_')]
    
    for col in time_cols:
        mask = df[col].abs() > threshold
        df.loc[mask, col] = threshold * df.loc[mask, col].apply(lambda x: 1 if x > 0 else -1)
    
    return df


data_test = cap_large_values_in_time_cols(data_train)
data_train = cap_large_values_in_time_cols(data_test)


features = ['year_time1', 'month_time1', 'day_time1', 'weekday_time1', 'hour_time1',
       'minute_time1', 'second_time1', 'year_time2', 'month_time2',
       'day_time2', 'weekday_time2', 'hour_time2', 'minute_time2',
       'second_time2', 'year_time3', 'month_time3', 'day_time3',
       'weekday_time3', 'hour_time3', 'minute_time3', 'second_time3',
       'year_time4', 'month_time4', 'day_time4', 'weekday_time4', 'hour_time4',
       'minute_time4', 'second_time4', 'year_time5', 'month_time5',
       'day_time5', 'weekday_time5', 'hour_time5', 'minute_time5',
       'second_time5', 'year_time6', 'month_time6', 'day_time6',
       'weekday_time6', 'hour_time6', 'minute_time6', 'second_time6',
       'year_time7', 'month_time7', 'day_time7', 'weekday_time7', 'hour_time7',
       'minute_time7', 'second_time7', 'year_time8', 'month_time8',
       'day_time8', 'weekday_time8', 'hour_time8', 'minute_time8',
       'second_time8', 'year_time9', 'month_time9', 'day_time9',
       'weekday_time9', 'hour_time9', 'minute_time9', 'second_time9',
       'year_time10', 'month_time10', 'day_time10', 'weekday_time10',
       'hour_time10', 'minute_time10', 'second_time10']


data_train = pd.get_dummies(data_train, columns=features, dtype=int)


data_anomaly = data_train.loc[(data_train['target']==1)]


data_not_anomaly = data_train.loc[(data_train['target']==0)]


data_not_anomaly = data_not_anomaly.drop('session_id', axis=1)


X = data_not_anomaly.drop('target', axis=1)
y = data_not_anomaly['target']


scaler = MinMaxScaler()
X = scaler.fit_transform(X)


X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


train_dataset = TensorDataset(X,y)


train_loader = DataLoader(train_dataset, batch_size = 256, shuffle = True)


X.shape


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(1867)
        )

    def forward(self,x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


model = Encoder()


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 15
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    for data, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, data)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'epoch {epoch + 1} / {epochs}, Loss: {running_loss / len(train_loader)}')


torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'model.pth')


model.eval()


def analos(model, dataloader):
    criterion = nn.MSELoss(reduction='none')
    losses = []
    with torch.no_grad():

        for X_batch, _ in dataloader:
            output = model(X_batch)
            loss = criterion(X_batch, output)
            losses.extend(loss.mean(dim=1).tolist())

    return losses


losses = analos(model, train_loader)


max(losses)


type(losses[90])


y = data_train['target']
X = data_train.drop(['target', 'session_id'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)


def inference(X, y, model):
    X = scaler.transform(X)
    
    y = torch.tensor(y, dtype=torch.float32)
    X = torch.tensor(X, dtype=torch.float32)

    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle = True)

    criterion = nn.MSELoss(reduction='none')

    model.eval()
    pred=[]
    loss_data=[]

    with torch.no_grad():

        for data, _ in train_loader:
            pred.append(model.encoder(data))
            output = model(data)
            loss = criterion(data, output)
            loss_data.extend(loss.mean(dim=1).tolist())

    pred = torch.cat(pred, dim=0).numpy()
    loss_frame = pd.DataFrame(data={'loss': loss_data})

    data_orig = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    data_pred = pd.DataFrame(pred, columns=[f'reconstructed_{i}' for i in range(pred.shape[1])])
    data = pd.concat([data_orig, data_pred], axis=1)
    data = pd.concat([data, loss_frame], axis=1)

    return data


train_X = inference(X_train, y_train, model)


train_X


smote = SMOTE(random_state = 42, sampling_strategy={1: 5_000})
X_smote, y_smote = smote.fit_resample(train_X, y_train)


X_smote


class_counts = Counter(y_smote)
total = sum(class_counts.values())
class_weights = {cls: total / count for cls, count in class_counts.items()}
class_weights_list = [class_weights[cls] for cls in sorted(class_weights.keys())]


cat = CatBoostClassifier(
    iterations=5_000, 
    learning_rate=0.01, 
    depth=7, verbose=100, 
    random_state=42, 
    class_weights=class_weights_list)


cat.fit(X_smote, y_smote)


test_X = inference(X_test, y_test.values, model)


test_X


predictions = cat.predict(test_X)


roc_auc_score(y_test, predictions)


print(classification_report(y_test, predictions))


cat.save_model('cat_model.cbm')

