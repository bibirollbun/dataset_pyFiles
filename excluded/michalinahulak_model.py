import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import optuna
import warnings

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
embeddings_train = pd.read_csv('/kaggle/input/predict-podcast-listening-time-embeddings/podcast_embeddings_train.csv')
embeddings_test = pd.read_csv('/kaggle/input/predict-podcast-listening-time-embeddings/podcast_embeddings_test.csv')


train.head(3)


def val_loss_function(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def cross_val_predict(model, X_train, y_train, X_test, val_loss_function, n_splits=5, random_state=42):
    print(f"Model: {model.__class__.__name__}")

    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    val_score = 0
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Fold {fold + 1}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr) 
        X_val = scaler.transform(X_val)
        
        model.fit(X_tr, y_tr)
        
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        cur_val_score = val_loss_function(y_val, val_preds)
        print(f"Current validation score: {cur_val_score}")
        val_score += cur_val_score / n_splits

        
        test_preds += model.predict(scaler.transform(X_test)) / n_splits

    print(f"Average validation score: {val_score}")
    return oof_preds, test_preds, val_score


def preprocess_data(df, fill_method='mean', 
                    encode_genre='one_hot',
                    encode_day='sin_cos', 
                    encode_time='one_hot' ):

    # Nan
    if fill_method == 'mean':
        df.fillna(df.mean(numeric_only=True), inplace=True)
        df.fillna(0, inplace=True)
    elif fill_method == 'zero':
        df.fillna(0, inplace=True)
    elif fill_method == 'drop':
        df.dropna(inplace=True)

    # id
    df.drop(columns=['id'], inplace = True)

    # Podcast_Name and Episode_Title
    df.drop(columns=['Podcast_Name', 'Episode_Title'], inplace = True)
    
    # Genre
    if encode_genre == 'one_hot':
        df = pd.get_dummies(df, columns=['Genre'])
    elif encode_genre == 'label':
        encoder = LabelEncoder()
        df['Genre_Encoded'] = encoder.fit_transform(df['Genre'])
        df.drop(columns=['Genre'], inplace=True)
    elif encode_genre == 'target':
        df['Genre_Target'] = df.groupby('Genre')['Listening_Time_minutes'].transform('mean')
        df.drop(columns=['Genre'], inplace=True)
    
    # Publication_Day
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Day_Num'] = df['Publication_Day'].apply(lambda x: days.index(x) if x in days else np.nan)
    
    if encode_day == 'sin_cos':
        df['Day_Sin'] = np.sin(2 * np.pi * df['Day_Num'] / 7)
        df['Day_Cos'] = np.cos(2 * np.pi * df['Day_Num'] / 7)
        df.drop(columns=['Publication_Day', 'Day_Num'], inplace=True)
    elif encode_day == 'one_hot':
        df = pd.get_dummies(df, columns=['Publication_Day'])
    
    # Publication_Time
    time_categories = ['Night',  'Evening', 'Afternoon', 'Morning']
    df['Time_Num'] = df['Publication_Time'].apply(lambda x: time_categories.index(x) if x in time_categories else np.nan)
    
    if encode_time == 'sin_cos':
        df['Time_Sin'] = np.sin(2 * np.pi * df['Time_Num'] / len(time_categories))
        df['Time_Cos'] = np.cos(2 * np.pi * df['Time_Num'] / len(time_categories))
        df.drop(columns=['Publication_Time', 'Time_Num'], inplace=True)
    elif encode_time == 'one_hot':
        df = pd.get_dummies(df, columns=['Publication_Time'])
    df.drop(columns=['Time_Num'], inplace=True)
    
    # Episode_Sentiment
    sentiment_order = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['Sentiment_Encoded'] = df['Episode_Sentiment'].map(sentiment_order)
    df.drop(columns=['Episode_Sentiment'], inplace=True)
    

    
    return df


train = preprocess_data(train, fill_method='mean', 
                    encode_genre='label',
                    encode_day='sin_cos', 
                    encode_time='one_hot' )

test = preprocess_data(test, fill_method='mean', 
                    encode_genre='label',
                    encode_day='sin_cos', 
                    encode_time='one_hot' )


from sklearn.decomposition import PCA

def reduce_embeddings_with_pca(df, n_components=2):
    # Wybieramy tylko kolumny embeddingów (wszystkie kolumny oprócz 'id')
    embedding_columns = [col for col in df.columns if col != 'id']
    embeddings = df[embedding_columns]
    
    # Inicjalizacja PCA i dopasowanie
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Dodanie wyników do DataFrame jako kolumn 'x' i 'y'
    df_reduced = pd.DataFrame()
    df_reduced['id'] = df['id']
    df_reduced['x'] = reduced_embeddings[:, 0]
    df_reduced['y'] = reduced_embeddings[:, 1]
    
    return df_reduced

embeddings_train_reduced = reduce_embeddings_with_pca(embeddings_train)


# train = pd.concat([train.reset_index(drop=True), 
#                             embeddings_train.drop(columns=['id']).reset_index(drop=True)], axis=1)

train = pd.concat([train.reset_index(drop=True), 
                             embeddings_train_reduced.drop(columns=['id']).reset_index(drop=True)], axis=1)


embeddings_test_reduced = reduce_embeddings_with_pca(embeddings_test)


test = pd.concat([test.reset_index(drop=True), 
                            embeddings_test_reduced.drop(columns=['id']).reset_index(drop=True)], axis=1)


numerical_cols = train.columns.tolist()
target_column = 'Listening_Time_minutes'
numerical_cols.remove(target_column)


X_train = train[numerical_cols]
X_test = test[numerical_cols]
y_train = train[target_column]


X_train


models = [
    # LGBMRegressor(boosting_type='gbdt'),
    # XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6),
    CatBoostRegressor(verbose=0),
    # LinearRegression()
]

results = {}

for model in models:
    oof, test, score = cross_val_predict(model, X_train, y_train, X_test, val_loss_function)
    results[model.__class__.__name__] = {
        "oof": oof,
        "test": test,
        "score": score
    }
    print(f"Final validation score for {model.__class__.__name__}: {score}\n")


X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=43)


from sklearn.preprocessing import MinMaxScaler

columns_scaled = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                  'Guest_Popularity_percentage', 'Number_of_Ads', 'Genre_Encoded',
                 
                 ]

scaler = MinMaxScaler()


# Dopasowanie i transformacja danych
X_tr[columns_scaled] = scaler.fit_transform(X_tr[columns_scaled])
X_val[columns_scaled] = scaler.transform(X_val[columns_scaled])
X_test[columns_scaled] = scaler.transform(X_test[columns_scaled])


X_tr_tensor = torch.tensor(X_tr.astype('float32').values)
y_tr_tensor = torch.tensor(y_tr.astype('float32').values).view(-1, 1)

X_val_tensor = torch.tensor(X_val.astype('float32').values)
y_val_tensor = torch.tensor(y_val.astype('float32').values).view(-1, 1)

X_test_tensor = torch.tensor(X_test.astype('float32').values)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_tr_tensor = X_tr_tensor.to(device)
y_tr_tensor = y_tr_tensor.to(device)

X_val_tensor = X_val_tensor.to(device)
y_val_tensor = y_val_tensor.to(device)

X_test_tensor = X_test_tensor.to(device)


import torch.nn.functional as F

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(F.mse_loss(y_pred, y_true))


class Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)

        )

    def forward(self, x):
        return self.model(x)


# 4. Inicjalizacja
input_dim = X_train.shape[1]
model = Regressor(input_dim)
# criterion = nn.MSELoss()
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
model = model.to(device)


# 5. Trening
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tr_tensor)
    loss = criterion(outputs, y_tr_tensor)
    loss.backward()
    optimizer.step()

    # Ewaluacja
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        rmse = mean_squared_error(y_val_tensor.cpu().numpy(), val_preds.cpu().numpy(), squared=False)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val RMSE: {rmse:.4f}")


model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)

test_preds


min(test_preds)


max(test_preds)


min(y_train)


max(y_train)


y_pred = results['CatBoostRegressor']['test']
# y_pred = test_preds


sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sub[target_column] = y_pred
# sub[target_column] = y_pred.cpu().detach().numpy()
sub.to_csv('submission.csv', index = False)

