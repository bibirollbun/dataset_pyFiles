import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import random

# from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler, PowerTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.impute import KNNImputer
# from sklearn.metrics import accuracy_score, roc_auc_score
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential, Model
# from keras.layers import Dense,Dropout,BatchNormalization, Input, LeakyReLU
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.metrics import AUC
# from keras.regularizers import l2
# import keras_tuner as kt
# from imblearn.over_sampling import SMOTE
# from scipy.stats import rankdata
# import warnings
# warnings.simplefilter('ignore')

# from imblearn.over_sampling import RandomOverSampler

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


## KNN - Handling missing values
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
test_df["winddirection"] = imputer.fit_transform(test_df[["winddirection"]])
test_df['winddirection']=test_df['winddirection'].fillna(value=test_df['winddirection'].mean())
train_df['winddirection']=train_df['winddirection'].fillna(value=train_df['winddirection'].mean())
train_df['windspeed']=train_df['windspeed'].fillna(value=train_df['windspeed'].mean())


def feature_generation(df):
    ## lag feature
    for lag in [1, 3, 7]:
        df[f'Pressure_lag{lag}'] = df['pressure'].shift(lag)
        df[f'Humidity_lag{lag}'] = df['humidity'].shift(lag)

    ## amount of change
    df['Pressure_change_1d'] = df['pressure'] - df['pressure'].shift(1)
    df['Humidity_change_1d'] = df['humidity'] - df['humidity'].shift(1)

    ## temperature related
    df['Temp_range'] = df['maxtemp'] - df['mintemp']
    df["avg_temp"] = (df["maxtemp"] + df["mintemp"]) / 2
    df['Dewpoint_diff'] = df['temparature'] - df['dewpoint']

    ## sunshine, cloud amount
    df['Sunshine_per_hour'] = df['sunshine'] / 24
    df['Cloud_per_hour'] = df['cloud'] / 24
    df['Cloud_Humidity_ratio'] = df['cloud'] / (df['humidity'] + 1e-5)
    df['Cloud_Sunshine_ratio'] = df['cloud'] / (df['sunshine'] + 1e-5)

    ## wind related
    df['Wind_x'] = df['windspeed'] * np.cos(np.radians(df['winddirection']))
    df['Wind_y'] = df['windspeed'] * np.sin(np.radians(df['winddirection']))

    ## others
    df['humidity_cloud_interaction'] = df['humidity'] * df['cloud']
    df['humidity_sunshine_interaction'] = df['humidity'] * df['sunshine']
    df['Pressure_Humidity_Interaction'] = df['pressure'] * df['humidity']
    df["cloud_wind_interaction"] = df["cloud"] * df["windspeed"]
    df['relative_dryness'] = 100 - df['humidity']
    df['sunshine_percentage'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['cloud_percentage'] = df['cloud'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['weather_index'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])
    df['Temp_Ratio'] = df['temparature'] / df['maxtemp'].max()
    # df['Try'] = (df['humidity'] + df['cloud'] + df['dewpoint'])
    # df['Try_2'] = (df['cloud'] - df['sunshine']) + df['temparature']
    
    # Moving Standard Deviations
    for window in [7, 14]:
        df[f'temp_std_{window}d'] = df['temparature'].rolling(window, min_periods=4).std().fillna(0)
        df[f'pressure_std_{window}d'] = df['pressure'].rolling(window, min_periods=4).std().fillna(0)
        df[f'humidity_std_{window}d'] = df['humidity'].rolling(window, min_periods=4).std().fillna(0)
    
    
    # wet-bulb temperature
    def calc_wet_bulb(T, RH):
        return T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + \
               np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
               0.00391838 * RH**(3/2) * np.arctan(0.023101 * RH) - 4.686035

    df['wet_bulb_temp'] = calc_wet_bulb(df['temparature'], df['humidity'])

    # saturated vapor pressure
    def calc_saturation_vapor_pressure(temp):
        return 6.11 * np.exp((17.27 * temp) / (temp + 237.3))

    df['e_s_temp'] = calc_saturation_vapor_pressure(df['temparature'])
    df['e_s_dewpoint'] = calc_saturation_vapor_pressure(df['dewpoint'])

    # vapor pressure deficit
    df['vapor_pressure_deficit'] = df['e_s_temp'] - df['e_s_dewpoint']
    
    df.fillna(method='bfill', inplace=True)
    
    return df





train_df = feature_generation(train_df)
test_df = feature_generation(test_df)


train_df.head()


SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


X = train_df.drop(columns=["id", "rainfall"])
y = train_df["rainfall"]
test = test_df.drop(columns=["id"])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = scaler.fit_transform(X)
test = scaler.transform(test)


X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
test = torch.tensor(test, dtype=torch.float32)
y = y.reshape(-1,1)


X.shape


from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)

cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f'Fold {fold + 1}/{k_folds}')

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_dataset = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_dataset = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

    model = nn.Sequential(
        nn.Linear(47, 47),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(47, 20),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
        nn.Sigmoid()
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss().to(device)

    # Training
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dataset:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_losses = []
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_dataset:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            val_losses.append(loss.item())
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # scoring
    auc_score = roc_auc_score(y_true, y_pred)
    cv_scores.append(auc_score)
    print(f'Fold {fold + 1} AUC: {auc_score:.4f}')

    models.append(model)

print(f'Cross-validated ROC AUC score: {np.mean(cv_scores):.5f} +/- {np.std(cv_scores):.5f}')


test_id = test_df["id"]
submit_score = []

for fold_, model in enumerate(models):
    pred_ = model(test)
    submit_score.append(pred_)

# predict test data
pred = np.mean([score.detach().cpu().numpy() for score in submit_score], axis=0)


submission = pd.DataFrame({
    'id': test_id,
    'rainfall': pred.flatten()})

# Save
submission.to_csv('submission.csv', index=False)




