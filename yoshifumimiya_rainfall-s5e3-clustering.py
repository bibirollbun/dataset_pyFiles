import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv').set_index('id')
print(train.shape)
print(train['rainfall'].value_counts())
train.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv').set_index('id')
print(test.shape)
test.head()


test['winddirection'].fillna(test['winddirection'].median(), inplace=True)


X, y = train.drop(['day','rainfall'],axis=1), train['rainfall']



import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    torch.manual_seed(seed)  # PyTorchのシード
    torch.cuda.manual_seed(seed)  # GPUでのシード
    torch.cuda.manual_seed_all(seed)  # 複数GPU用のシード
    np.random.seed(seed)  # NumPyのシード
    random.seed(seed)  # Python標準ライブラリの乱数シード

    # 再現性を確保するための設定
    torch.backends.cudnn.deterministic = True  # CuDNNの決定論的動作を有効化
    torch.backends.cudnn.benchmark = False  # ベンチマークモードを無効化（速度と再現性のトレードオフ）





scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N, 1) に変換

dataset = TensorDataset(X_tensor, y_tensor)

# ----- 2. ニューラルネットワークモデルの定義 -----
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# ----- 3. クロスバリデーションの設定（StratifiedKFold） -----
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

auc_scores = []
set_seed(42) 
# ----- 4. 交差検証ループ -----
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}")
    set_seed(42) 
    # データ分割
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    # モデル定義
    model = BinaryClassifier(input_size=X.shape[1]).to(device)

    # クラス不均衡に対応するための重み計算
    pos_weight = torch.tensor([y_tensor[train_idx].sum() / len(train_idx)]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # ロジット出力でBCE
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # ----- 5. EarlyStopping の設定 -----
    patience = 3  # 何エポック改善しなかったら停止するか
    best_val_loss = float("inf")
    counter = 0

    # ----- 6. モデルの学習 -----
    epochs = 30
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # バリデーション
        model.eval()
        val_loss = 0
        all_targets, all_probs = [], []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                all_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        # AUC 計算
        auc_score = roc_auc_score(all_targets, all_probs)
        auc_scores.append(auc_score)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, AUC: {auc_score:.4f}")

        # Early Stopping の判定
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # カウンターリセット
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

# ----- 7. 平均AUCの計算 -----
mean_auc = np.mean(auc_scores)
print(f"\nAverage AUC: {mean_auc:.4f}")



from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


# ----- 3. クロスバリデーションの設定（StratifiedKFold） -----
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

auc_scores = []
set_seed(42)
# ----- 4. 交差検証ループ -----
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}")
    set_seed(42)
    # ----- 4.1 各クラスタリング手法を適用 (train のみ) -----
    def apply_clustering(method, train_X, val_X, **kwargs):
        """ クラスタリング手法を適用し、クラスタラベルを取得する関数 """
        model = method(**kwargs)
        train_clusters = model.fit_predict(train_X)
        val_clusters = model.predict(val_X) if hasattr(model, "predict") else model.fit_predict(val_X)
        return train_clusters.reshape(-1, 1), val_clusters.reshape(-1, 1)

    # クラスタリング（trainデータのみで学習、valデータは推論のみ）
    train_kmeans, val_kmeans = apply_clustering(KMeans, X[train_idx], X[val_idx], n_clusters=2, random_state=42)
    train_hierarchical, val_hierarchical = apply_clustering(AgglomerativeClustering, X[train_idx], X[val_idx], n_clusters=2)
    train_dbscan, val_dbscan = apply_clustering(DBSCAN, X[train_idx], X[val_idx], eps=0.5, min_samples=100)
    train_gmm, val_gmm = apply_clustering(GaussianMixture, X[train_idx], X[val_idx], n_components=2, random_state=42)

    # DBSCAN の場合、-1 (ノイズ) を 0 に変換
    train_dbscan[train_dbscan == -1] = 0
    val_dbscan[val_dbscan == -1] = 0

    # クラスタ情報を特徴量として追加
    X_train_clustered = np.hstack([X[train_idx], train_kmeans, train_hierarchical, train_dbscan, train_gmm])
    X_val_clustered = np.hstack([X[val_idx], val_kmeans, val_hierarchical, val_dbscan, val_gmm])

    # PyTorch Tensor へ変換
    X_train_tensor = torch.tensor(X_train_clustered, dtype=torch.float32)
    y_train_tensor = torch.tensor(y[train_idx].to_numpy(), dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_clustered, dtype=torch.float32)
    y_val_tensor = torch.tensor(y[val_idx].to_numpy(), dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ----- 4.2 モデルの設定 -----
    model = BinaryClassifier(input_size=X_train_clustered.shape[1]).to(device)

    # クラス不均衡対応のための損失関数の重み
    pos_weight = torch.tensor([y_train_tensor.sum() / len(train_idx)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # ----- 4.3 EarlyStopping の設定 -----
    patience = 3
    best_val_loss = float("inf")
    counter = 0

    # ----- 5. モデルの学習 -----
    epochs = 30
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # バリデーション
        model.eval()
        val_loss = 0
        all_targets, all_probs = [], []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                all_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        # AUC 計算
        auc_score = roc_auc_score(all_targets, all_probs)
        auc_scores.append(auc_score)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, AUC: {auc_score:.4f}")

        # Early Stopping の判定
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

# ----- 6. 平均AUCの計算 -----
mean_auc = np.mean(auc_scores)
print(f"\nAverage AUC: {mean_auc:.4f}")



import os

MODEL_PATH = "/kaggle/working/binary_classifier.pth"

# ディレクトリがない場合は作成
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ モデルを {MODEL_PATH} に保存しました。")



# ----- 1. モデルの保存 & 読み込み -----
# モデルのロード（推論時）
model = BinaryClassifier(input_size=X.shape[1] + 4)  # クラスタリング特徴4つを含む
model.load_state_dict(torch.load("/kaggle/working/binary_classifier.pth"))
model.to(device)
model.eval()  # 推論モード

# ----- 2. `test` データの前処理 -----
# (1) `test` データを準備
test = test.drop(['day'],axis=1)
test = scaler.transform(test)   # 標準化（学習時のscalerを適用）


# クラスタリング（trainデータのみで学習、valデータは推論のみ）
train_kmeans, test_kmeans = apply_clustering(KMeans, X[train_idx], test, n_clusters=2, random_state=42)
train_hierarchical, test_hierarchical = apply_clustering(AgglomerativeClustering, X[train_idx], test, n_clusters=2)
train_dbscan, test_dbscan = apply_clustering(DBSCAN, X[train_idx], test, eps=0.5, min_samples=100)
train_gmm, test_gmm = apply_clustering(GaussianMixture, X[train_idx], test, n_components=2, random_state=42)

# DBSCAN の -1 (ノイズ) を 0 に変換
test_dbscan[test_dbscan == -1] = 0

# (3) `test` データにクラスタリング特徴を追加
test_clustered = np.hstack([test, test_kmeans, test_hierarchical, test_dbscan, test_gmm])

# ----- 3. `test` データを PyTorch Tensor に変換 -----
X_test_tensor = torch.tensor(test_clustered, dtype=torch.float32)
test_dataset = TensorDataset(X_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----- 4. モデルによる `test` データの予測 -----
predictions = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs[0].to(device)  # DataLoader から取り出し
        outputs = model(inputs)
        preds = torch.sigmoid(outputs).cpu().numpy().flatten()  # Sigmoid で確率化
        predictions.extend(preds)



# Save Submission
df_subm= pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
df_subm['rainfall'] = predictions
df_subm.to_csv('submission.csv', index=False)
df_subm.head()




