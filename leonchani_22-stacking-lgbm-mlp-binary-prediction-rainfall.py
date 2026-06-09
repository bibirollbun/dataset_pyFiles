# table manipulation, calculating
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100) # increase the maximum number of columns

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# learning
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold

import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold, cross_val_predict # <- ここにcross_val_predictのインポートを追加

from sklearn.neural_network import MLPClassifier # MLPClassifierをインポート

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")





df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


df_train


df_test








## KNN - Handling missing values
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_test["winddirection"] = imputer.fit_transform(df_test[["winddirection"]])





def generate_features(df):
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


df_train = generate_features(df_train)
df_test = generate_features(df_test)





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
import random


SEED = 42
NUM_SPLITS = 4 # 10


X = df_train.drop(columns=["rainfall"])
y = df_train["rainfall"]

X_test = df_test


from sklearn.preprocessing import StandardScaler # StandardScalerをインポート
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# ベースモデルの定義（分類モデルを使用）
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
model_svm = SVC(kernel='rbf', probability=True)
model_knn = KNeighborsClassifier(n_neighbors=5)
model_lgb = lgb.LGBMClassifier(n_estimators=100, random_state=42)
model_lr = LogisticRegression(random_state=42)


# クロスバリデーションの定義
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# クロスバリデーションによるベースモデルの確率予測
pred_rf = cross_val_predict(model_rf, X, y, cv=kf, method='predict_proba')[:, 1]
pred_gb = cross_val_predict(model_gb, X, y, cv=kf, method='predict_proba')[:, 1]
pred_svm = cross_val_predict(model_svm, X, y, cv=kf, method='predict_proba')[:, 1]
pred_knn = cross_val_predict(model_knn, X, y, cv=kf, method='predict_proba')[:, 1]
pred_lgb = cross_val_predict(model_lgb, X, y, cv=kf, method='predict_proba')[:, 1]
pred_lr = cross_val_predict(model_lr, X, y, cv=kf, method='predict_proba')[:, 1]


# メタモデルの学習データを作成
meta_X = np.column_stack((pred_rf, pred_gb, pred_svm, pred_knn, pred_lgb, pred_lr, X))


# PyTorchテンソルに変換
X_torch = torch.tensor(meta_X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)

cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_torch)):
    print(f'Fold {fold + 1}/{k_folds}')

    X_train, X_val = X_torch[train_idx], X_torch[val_idx]
    y_train, y_val = y_torch[train_idx], y_torch[val_idx]

    train_dataset = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_dataset = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
    
    model = nn.Sequential(
        nn.Linear(X_torch.shape[1], 32), # 層を削減
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.2), # ドロップアウト率の調整
        nn.Linear(32, 16), # 層を削減
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(0.2), # ドロップアウト率の調整
        nn.Linear(16, 1),
        nn.Sigmoid()
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)     
    loss_fn = nn.BCELoss().to(device)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dataset:
            optimizer.zero_grad()
            pred = model(x_batch.to(device))
            loss = loss_fn(pred, y_batch.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    val_losses = []
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_dataset:
            pred = model(x_batch.to(device))
            loss = loss_fn(pred, y_batch.to(device))
            val_losses.append(loss.item())
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    auc_score = roc_auc_score(y_true, y_pred)
    cv_scores.append(auc_score)
    print(f'Fold {fold + 1} AUC: {auc_score:.4f}')

    models.append(model)

print(f'Cross-validated ROC AUC score: {np.mean(cv_scores):.5f} +/- {np.std(cv_scores):.5f}')


# Plot the ROC curve for each fold
plt.figure(figsize=(8, 6))
for fold, model in enumerate(models):
    model.eval()
    y_pred_val = []
    y_true_val = []
    with torch.no_grad():
        for x_batch, y_batch in val_dataset:
            pred = model(x_batch)
            y_pred_val.extend(pred.cpu().numpy()) # 括弧を閉じる
            y_true_val.extend(y_batch.cpu().numpy())
    y_pred_val = np.array(y_pred_val)
    y_true_val = np.array(y_true_val)
    fpr, tpr, thresholds = roc_curve(y_true_val, y_pred_val)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Fold {fold+1} (AUC = {roc_auc:.2f})')

# 対角線
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# グラフの設定
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# AUCスコアの平均と標準偏差を表示
print(f'Cross-validated ROC AUC score: {np.mean(cv_scores):.5f} +/- {np.std(cv_scores):.5f}')


# 全訓練データでベースモデルを学習
model_rf.fit(X, y)
model_gb.fit(X, y)
model_svm.fit(X, y)
model_knn.fit(X, y)
model_lgb.fit(X, y)
model_lr.fit(X, y)


# テストデータに対するベースモデルの確率予測
pred_rf_test = model_rf.predict_proba(X_test)[:, 1]
pred_gb_test = model_gb.predict_proba(X_test)[:, 1]
pred_svm_test = model_svm.predict_proba(X_test)[:, 1]
pred_knn_test = model_knn.predict_proba(X_test)[:, 1]
pred_lgb_test = model_lgb.predict_proba(X_test)[:, 1]
pred_lr_test = model_lr.predict_proba(X_test)[:, 1]


# テストデータに対するメタモデルの入力データを作成
meta_X_test = np.column_stack((pred_rf_test, pred_gb_test, pred_svm_test, pred_knn_test, pred_lgb_test, pred_lr_test, X_test))


# テストデータに対するメタモデルの予測
X_test_torch = torch.tensor(meta_X_test, dtype=torch.float32).to(device)



# 全訓練データでメタモデルを学習
meta_model = nn.Sequential(
    nn.Linear(X_torch.shape[1], 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 1),
    nn.Sigmoid()
).to(device)

# 最適化アルゴリズムと損失関数
optimizer = optim.Adam(meta_model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.BCELoss().to(device)

# 全訓練データで学習
train_dataset = DataLoader(TensorDataset(X_torch, y_torch), batch_size=32, shuffle=True)
num_epochs = 50
for epoch in range(num_epochs):
    meta_model.train()
    for x_batch, y_batch in train_dataset:
        optimizer.zero_grad()
        pred = meta_model(x_batch.to(device))
        loss = loss_fn(pred, y_batch.to(device))
        loss.backward()
        optimizer.step()

# テストデータに対する予測
meta_model.eval()
with torch.no_grad():
    pred_test = meta_model(X_test_torch)

pred_test_np = pred_test.cpu().numpy()

print(pred_test_np)


test_id = df_test["id"]


submission = pd.DataFrame({
    'id': test_id,
    'rainfall': pred_test_np.flatten()
})

# Save
submission.to_csv('submission.csv', index=False)

submission

