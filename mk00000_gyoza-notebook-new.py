# **Imports**
import random
import numpy as np
import torch
import os

# 再現性のための乱数シードの設定
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

SEED = 42
seed_everything(seed=SEED)

import pandas as pd
import polars as pl
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
from tqdm import tqdm

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

import kaggle_evaluation.cmi_inference_server
from matplotlib import pyplot as plt

# **Read data**
print("Loading datasets...")
# 学習データとデモグラフィックデータを読み込む
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

# ラベルをエンコードする
label_encoder = LabelEncoder()
train_df['gesture'] = label_encoder.fit_transform(train_df['gesture'].astype(str))
gesture_classes = label_encoder.classes_

# BFRBジェスチャーのリストとインデックス
bfrb_gestures = [
    'Above ear - pull hair',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Neck - pinch skin',
    'Neck - scratch',
    'Cheek - pinch skin'
]
bfrb_indices = label_encoder.transform(bfrb_gestures)

# 特徴量カラムを定義する
imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
tof_thm_cols = [c for c in train_df.columns if c.startswith('thm_') or c.startswith('tof_')]
feature_cols = imu_cols + tof_thm_cols # IMU特徴量を先に配置

imu_dim = len(imu_cols)
tof_thm_dim = len(tof_thm_cols)
print(f"IMU features: {imu_dim}, TOF/Thermal features: {tof_thm_dim}, Total features: {len(feature_cols)}")

# 学習データ内の欠損値を確認する
nan_counts = train_df[feature_cols].isna().sum().sum()
print("Total NaNs in train features:", nan_counts)

# IMUデータの利き手依存性を除去するための対称変換
def apply_symmetry(data):
    transformed = data.copy()
    transformed['acc_z'] = -transformed['acc_z']
    transformed['acc_y'] = -transformed['acc_y']
    
    transformed['rot_w'] = transformed['rot_w']
    transformed['rot_x'] = transformed['rot_x']
    transformed['rot_y'] = -transformed['rot_y']
    transformed['rot_z'] = -transformed['rot_z']
    return transformed

# デモグラフィックデータを学習データに結合し、利き手に基づいてIMUデータに対称変換を適用
train_df = train_df.merge(
    train_dem_df,
    on='subject',
    how='left',
    validate='many_to_one'
)
right_handed_mask = train_df['handedness'] == 1
train_df.loc[right_handed_mask, imu_cols] = apply_symmetry(train_df.loc[right_handed_mask, imu_cols])

# Create dataset
sequences = train_df.groupby('sequence_id')
X_list = []
lengths = []
y_list = []

sequence_info = []
# 各シーケンスを処理し、特徴量とラベルを抽出
for i, (seq_id, seq) in enumerate(sequences):
    seq_data = seq[feature_cols].ffill().bfill().fillna(0).values # 欠損値を埋める
    X_list.append(seq_data)
    lengths.append(seq_data.shape[0])
    sequence_info.append({
        'sequence_id': seq_id,
        'subject': seq['subject'].iloc[0],
        'gesture': seq['gesture'].iloc[0]
    })

# シーケンス長を90パーセンタイルでパディング/切り捨て
pad_len = int(np.percentile(lengths, 90))
print(f"Pad/truncate all sequences to length {pad_len} (90th percentile).")

seq_df = pd.DataFrame(sequence_info)
# シーケンスをパディングまたは切り捨て
X_array = keras_pad_sequences(
    X_list,
    maxlen=pad_len,
    dtype='float32',
    padding='post',
    truncating='post'
)  # 形状: (n_samples, pad_len, total_features)

y_array = seq_df['gesture'].values  # 形状: (n_samples,)
num_classes = len(np.unique(y_array))
y_array = np.eye(num_classes)[y_array]  # ワンホットエンコーディング 形状: (n_samples, num_classes)

# PyTorch向けに転置 (n_samples, features, seq_len)
X_array = np.transpose(X_array, (0, 2, 1))

# PyTorchのDatasetクラス
class SequenceDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

# **Model**
# Squeeze-Excitationブロック
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = x.mean(dim=2) # グローバル平均プーリング
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        se = se.unsqueeze(2)
        return x * se

# Residual SEブロック
class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2, dropout_rate=0.3):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock(out_channels, reduction=8)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        out = out + shortcut # スキップコネクション
        out = self.relu(out)

        out = self.pool(out)
        out = self.dropout(out)
        return out

# アテンションメカニズム
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.score_fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        scores = torch.tanh(self.score_fc(x))
        scores = scores.squeeze(2)
        weights = F.softmax(scores, dim=1)
        weights = weights.unsqueeze(2)
        weighted = x * weights
        context = weighted.sum(dim=1)
        return context
        
# IMUベースのHARモデル
class IMU_HARModel(nn.Module):
    def __init__(self, total_features, imu_dim, pad_len, num_classes):
        super(IMU_HARModel, self).__init__()
        # IMUブランチのResidual SEブロック
        self.resblock1 = ResidualSEBlock(imu_dim, 64, kernel_size=3, pool_size=2, dropout_rate=0.1)
        self.resblock2 = ResidualSEBlock(64, 128, kernel_size=3, pool_size=2, dropout_rate=0.1)

        merged_channels = 128 # IMUのみを使用

        # BiGRU層
        self.bigru = nn.GRU(
            input_size=merged_channels,
            hidden_size=merged_channels//2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        # アテンション層
        self.attention = Attention(input_dim=merged_channels)

        # 全結合層
        self.fc1 = nn.Linear(merged_channels, 128, bias=True)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.drop_fc1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(128, 64, bias=True)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.drop_fc2 = nn.Dropout(0.1)

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, total_features, seq_len)
        x_imu = x[:, :imu_dim, :] # IMU特徴量を抽出
        # x_ttf = x[:, imu_dim:, :] # TOF/Thermal特徴量は今回は使用しない

        # IMUブランチを通過
        b1 = self.resblock1(x_imu)
        b1 = self.resblock2(b1)

        merged = b1 # IMUブランチのみ使用

        # GRU用に形状を変更: (batch, seq_len/4, features)
        merged = merged.permute(0, 2, 1)

        # BiGRUを通過
        lstm_out, _ = self.bigru(merged)

        # アテンションを適用
        context = self.attention(lstm_out)

        # 全結合層を通過
        x = self.fc1(context)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.drop_fc1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.drop_fc2(x)

        out = self.out(x)
        return out

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# **Utils**
# ソフトクロスエントロピー損失関数
def soft_cross_entropy(pred, soft_targets):
    log_probs = F.log_softmax(pred, dim=1)
    loss = -torch.sum(soft_targets * log_probs, dim=1).mean()
    return loss

# Mixupデータ拡張
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

# # **Training loop**
# seed_everything(seed=SEED)

# criterion = soft_cross_entropy # 損失関数
# n_splits = 5 # クロスバリデーションの分割数
# batch_size = 128
# gkf = GroupKFold(n_splits=n_splits) # グループK分割クロスバリデーション

# fold_metrics = []
# best_fold_metrics = []
# best_models = []

# # グループK分割クロスバリデーションループ
# for fold, (train_idx, val_idx) in enumerate(gkf.split(X_array, y_array, groups=seq_df['subject'])):
#     print(f"\n{'='*50}")
#     print(f"Fold {fold + 1}/{n_splits}")
#     print(f"Train subjects: {len(np.unique(seq_df.iloc[train_idx]['subject']))}")
#     print(f"Val subjects: {len(np.unique(seq_df.iloc[val_idx]['subject']))}")
#     print(f"{'='*50}")
    
#     train_dataset = SequenceDataset(X_array[train_idx], y_array[train_idx])
#     val_dataset = SequenceDataset(X_array[val_idx], y_array[val_idx])
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
#     seed_everything(seed=SEED + fold) # 各foldで異なるシードを設定
#     model = IMU_HARModel(
#         total_features=len(feature_cols),
#         imu_dim=imu_dim,
#         pad_len=pad_len,
#         num_classes=num_classes,
#     ).to(device)
    
#     # オプティマイザとスケジューラ
#     optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
#     steps_per_epoch = len(train_loader)
#     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
#         optimizer,
#         T_0=5 * steps_per_epoch,
#         T_mult=2,
#         eta_min=1e-5,
#     )
    
#     # 早期停止
#     best_metric = -np.inf
#     best_binary_f1 = -np.inf
#     best_macro_f1 = -np.inf
#     patience = 15 # 改善が見られないエポック数
#     epochs_no_improve = 0
#     num_epochs = 100
    
#     for epoch in range(1, num_epochs + 1):
#         # 学習フェーズ
#         model.train()
#         train_loss = 0.0
#         total = 0
#         for batch_x, batch_y in train_loader:
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
    
#             # Mixupを適用
#             mixed_x, mixed_y = mixup_data(batch_x, batch_y, alpha=0.2)
    
#             optimizer.zero_grad()
#             outputs = model(mixed_x)
#             loss = criterion(outputs, mixed_y)
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
    
#             train_loss += loss.item() * batch_x.size(0)
#             total += batch_x.size(0)
#         train_loss /= total
    
#         # 検証フェーズ
#         model.eval()
#         val_loss = 0.0
#         total = 0
#         all_true = []
#         all_pred = []
        
#         with torch.no_grad():
#             for batch_x, batch_y in val_loader:
#                 batch_x = batch_x.to(device)
#                 batch_y = batch_y.to(device)
                
#                 outputs = model(batch_x)
#                 loss = criterion(outputs, batch_y)
#                 val_loss += loss.item() * batch_x.size(0)
#                 total += batch_x.size(0)
                
#                 preds = torch.argmax(outputs, dim=1).cpu().numpy()
#                 trues = torch.argmax(batch_y, dim=1).cpu().numpy()
                
#                 all_true.append(trues)
#                 all_pred.append(preds)
        
#         val_loss /= total
#         all_true = np.concatenate(all_true)
#         all_pred = np.concatenate(all_pred)
        
#         # 競技の評価指標を計算
#         # バイナリ分類: BFRB (1) vs non-BFRB (0)
#         binary_true = np.isin(all_true, bfrb_indices).astype(int)
#         binary_pred = np.isin(all_pred, bfrb_indices).astype(int)
#         binary_f1 = f1_score(binary_true, binary_pred)
        
#         # non-BFRBジェスチャーを単一のクラスに統合
#         collapsed_true = np.where(
#             np.isin(all_true, bfrb_indices),
#             all_true,
#             len(bfrb_gestures)  # 非BFRB用の単一クラス
#         )
#         collapsed_pred = np.where(
#             np.isin(all_pred, bfrb_indices),
#             all_pred,
#             len(bfrb_gestures)  # 非BFRB用の単一クラス
#         )
        
#         # 統合されたクラスでのマクロF1スコア
#         macro_f1 = f1_score(collapsed_true, collapsed_pred, average='macro')
#         final_metric = (binary_f1 + macro_f1) / 2 # 最終評価指標
        
#         print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
#         print(f"  Binary F1 = {binary_f1:.4f}, Macro F1 = {macro_f1:.4f}, Final Metric = {final_metric:.4f}")
        
#         # 早期停止のロジック
#         if final_metric > best_metric:
#             best_metric = final_metric
#             best_binary_f1 = binary_f1
#             best_macro_f1 = macro_f1
#             epochs_no_improve = 0
#             best_model_state = model.state_dict()
#             print(f"  New best metric! Saving model...")
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 print(f"Early stopping triggered at epoch {epoch}")
#                 model.load_state_dict(best_model_state) # 改善が止まった場合、最良のモデルをロード
#                 break
    
#     torch.save(best_model_state, f"best_model_fold{fold}.pth") # 各foldの最良モデルを保存
#     best_models.append(best_model_state)
    
#     fold_metrics.append({
#         'binary_f1': binary_f1,
#         'macro_f1': macro_f1,
#         'final_metric': final_metric
#     })
    
#     best_fold_metrics.append({
#         'binary_f1': best_binary_f1,
#         'macro_f1': best_macro_f1,
#         'final_metric': best_metric
#     })
    
#     print(f"\nFold {fold + 1} completed.")
#     print(f"Final validation metrics - Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Final: {final_metric:.4f}")
#     print(f"Best validation metrics - Binary F1: {best_binary_f1:.4f}, Macro F1: {best_macro_f1:.4f}, Final: {best_metric:.4f}")

# print("\n" + "="*50)
# print("Cross-Validation Results")
# print("="*50)

# # クロスバリデーションの結果を統計表示
# best_binary_f1 = [m['binary_f1'] for m in best_fold_metrics]
# best_macro_f1 = [m['macro_f1'] for m in best_fold_metrics]
# best_metrics = [m['final_metric'] for m in best_fold_metrics]

# print("\nBest Fold-wise Metrics:")
# for i, (bf1, mf1, fm) in enumerate(zip(best_binary_f1, best_macro_f1, best_metrics)):
#     print(f"Fold {i+1}: Binary F1 = {bf1:.4f}, Macro F1 = {mf1:.4f}, Final = {fm:.4f}")

# print("\nGlobal Statistics (Best Metrics):")
# print(f"Mean Best Final Metric: {np.mean(best_metrics):.4f} ± {np.std(best_metrics):.4f}")
# print(f"Mean Best Binary F1: {np.mean(best_binary_f1):.4f} ± {np.std(best_binary_f1):.4f}")
# print(f"Mean Best Macro F1: {np.mean(best_macro_f1):.4f} ± {np.std(best_macro_f1):.4f}")

# Reloading best model
model_ensemble = []
for fold in range(5):
    model = IMU_HARModel(
        total_features=len(feature_cols),
        imu_dim=imu_dim,
        pad_len=pad_len,
        num_classes=num_classes,
    ).to(device)
    checkpoint = torch.load(f"/kaggle/input/mitsuki/pytorch/default/1/best_model_fold{fold}.pth", map_location=device) # 保存されたモデルをロード
    model.load_state_dict(checkpoint)
    model.eval() # 評価モードに設定
    model_ensemble.append(model)

# **Submission**
# 単一シーケンスの前処理関数
def preprocess_sequence_mitsuki(df_seq: pd.DataFrame):
    data = df_seq[feature_cols].ffill().bfill().fillna(0).values # 欠損値を埋める
    padded = keras_pad_sequences(
        [data],
        maxlen=pad_len,
        dtype='float32',
        padding='post',
        truncating='post'
    )[0]
    tensor = torch.from_numpy(padded.T).unsqueeze(0).float()
    return tensor
    
# 推論関数 (Kaggle評価APIによって呼び出される)
def predict_mitsuki(sequence: pl.DataFrame, demographics: pl.DataFrame):
    df_seq = sequence.to_pandas()
    df_demo = demographics.to_pandas()
    
    # デモグラフィックデータを結合し、利き手補正を適用
    df_seq = df_seq.merge(
    df_demo,
    on='subject',
    how='left',
    validate='many_to_one',
    )
    right_handed_mask = df_seq['handedness'] == 1
    df_seq.loc[right_handed_mask, imu_cols] = apply_symmetry(df_seq.loc[right_handed_mask, imu_cols])

    x_tensor = preprocess_sequence_mitsuki(df_seq).to(device)
    
    all_outputs = []
    with torch.no_grad():
        # モデルアンサンブルで予測
        for model in model_ensemble:
            outputs = model(x_tensor).softmax(dim=-1) # ソフトマックスを適用
            all_outputs.append(outputs)

    avg_outputs = torch.mean(torch.stack(all_outputs), dim=0) # アンサンブルの平均
    pred_idx = torch.argmax(avg_outputs, dim=1).item() # 最も確率の高いクラスを取得
    
    return avg_outputs.to('cpu').detach().numpy().copy() # 予測されたジェスチャーの文字列を返す




import lightgbm as lgb
import pandas as pd
import numpy as np

# ======================
# 1. モデルの読み込み
# ======================

# Add input で指定したパス名に置き換えてください（例: /kaggle/input/gbm-trained-model）
MODEL_PATH = "/kaggle/input/cmi_lgbm/other/default/1/gbm_model.txt"

miyairi_lgbm_model = lgb.Booster(model_file=MODEL_PATH)

# ======================
full_test_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
all_test_subjects = full_test_dem_df['subject'].unique().tolist()
subject_cat_type = pd.api.types.CategoricalDtype(categories=all_test_subjects, ordered=True)
# ======================

# ======================
# 2. 推論用データの読み込み
# ======================
# 例: 検証用 CSV を使う場合
X_test = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")

# 必要に応じて前処理を行う（例：ID列や不要列の削除）
exclude_cols = ["row_id", "orientation", "sequence_type", "behavior", "phase"]
X = X_test.drop(columns=exclude_cols, errors="ignore")

# カテゴリ型の列があれば、型変換
cat_cols = [col for col in X.columns if X[col].dtype == 'object']
for col in cat_cols:
    X[col] = X[col].astype("category")

# ======================
# 3. 推論
# ======================

# 推論（multiclass想定、必要に応じてargmax）
y_pred_proba = miyairi_lgbm_model.predict(X)
y_pred = np.argmax(y_pred_proba, axis=1)

# ======================
# 4. 結果の確認
# ======================

print("予測ラベル（先頭5件）:", y_pred[:5])
"""
def predict_miyairi(sequence: pl.DataFrame, demographics: pl.DataFrame) -> list[float]:
    # Polars → pandas
    df_seq = sequence.to_pandas()
    df_demo = demographics.to_pandas()

    # マージ（subject ごとの情報を統合）
    df = df_seq.merge(df_demo, on="subject", how="left", validate="many_to_one")

    # 不要列の削除（安全に）
    for col in ["row_id", "behavior", "phase", "gesture", "orientation", "sequence_type"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # ======================
    if 'subject' in df.columns:
        df['subject'] = df['subject'].astype(subject_cat_type)
    # ======================

    # 特徴量リストに不要列が混ざっていたら除外
    safe_feature_cols = [col for col in feature_cols if col in df.columns]

    # カテゴリ型の整備（cat_colsに合わせて）
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 特徴量の取り出し
    df_input = df[safe_feature_cols].copy()

    # 予測（1件のデータなので [0] で抽出）
    probs = miyairi_lgbm_model.predict(df_input, num_iteration=miyairi_lgbm_model.best_iteration) #gbmをmiyairi_lgbm_modelに変更(radon)
    return probs[0]"""


import lightgbm as lgb
import pandas as pd
import numpy as np
import polars as pl

# ==========================================================
# 1. モデルをロードし、モデル自身から「正解」のカテゴリ情報を抽出
# ==========================================================

MODEL_PATH = "/kaggle/input/cmi_lgbm/other/default/1/gbm_model.txt"
miyairi_lgbm_model = lgb.Booster(model_file=MODEL_PATH)

# モデルの `pandas_categorical` 属性から、学習時に使用されたカテゴリの完全なリストを取得
model_categorical_info = miyairi_lgbm_model.pandas_categorical
if not model_categorical_info or len(model_categorical_info) < 2:
    raise ValueError("モデルファイルから期待される2つのカテゴリカル情報セットを抽出できませんでした。")

# 診断結果に基づき、2つのカテゴリカル変数の「正解」のカテゴリリストを抽出
SEQ_ID_CATEGORIES = list(model_categorical_info[0])   # 診断結果のセット1
SUBJECT_CATEGORIES = list(model_categorical_info[1]) # 診断結果のセット2

# 抽出した情報から、絶対に間違いないカテゴリ型（Dtype）を生成
SEQ_ID_CAT_TYPE = pd.api.types.CategoricalDtype(categories=SEQ_ID_CATEGORIES, ordered=True)
SUBJECT_CAT_TYPE = pd.api.types.CategoricalDtype(categories=SUBJECT_CATEGORIES, ordered=True)

# テストデータなどで未知のカテゴリが現れた際に使用するデフォルト値を設定
DEFAULT_SEQ_ID = SEQ_ID_CATEGORIES[0]
DEFAULT_SUBJECT = SUBJECT_CATEGORIES[0]

print("✅ モデルから2つのカテゴリカル特徴量の情報を正しく抽出し、準備が完了しました。")


# ==================================================
# 2. モデル情報に基づいた、自己完結・高信頼性の推論関数
# ==================================================

def predict_miyairi(sequence: pl.DataFrame, demographics: pl.DataFrame) -> np.ndarray:
    df_seq = sequence.to_pandas()
    df_demo = demographics.to_pandas()
    df = pd.merge(df_seq, df_demo, on="subject", how="left")

    # --- 2つのカテゴリカル変数（sequence_id と subject）を安全に処理 ---

    # A) 'sequence_id' の処理
    # モデルから抽出した「正解」のカテゴリ型に変換します。
    # これにより、テスト用の未知のIDは 'NaN' (欠損値) になります。
    df['sequence_id'] = df['sequence_id'].astype(SEQ_ID_CAT_TYPE)
    # 欠損値（未知のID）が発生した場合、モデルが知っているデフォルト値に置き換えます。
    if df['sequence_id'].isnull().any():
        df['sequence_id'].fillna(DEFAULT_SEQ_ID, inplace=True)

    # B) 'subject' の処理
    # 同様に、'subject' 列も未知のカテゴリを安全に処理します。
    df['subject'] = df['subject'].astype(SUBJECT_CAT_TYPE)
    if df['subject'].isnull().any():
        df['subject'].fillna(DEFAULT_SUBJECT, inplace=True)
        
    # --- 特徴量の最終整形 ---
    # モデルが学習した全ての特徴量のリストをモデル自身から取得
    expected_features = miyairi_lgbm_model.feature_name()

    # モデルが期待する通りの特徴量と順序でDataFrameを再構成
    df_input = df[expected_features]

    # 予測を実行
    probs = miyairi_lgbm_model.predict(df_input)
    
    return probs

# ========================================================
# 3. ローカルテスト
# ========================================================
try:
    print("\nMiyairiモデルのローカルテストを実行中...")
    local_test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
    local_demographics_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
    
    first_seq_id = local_test_df['sequence_id'].iloc[0]
    first_sequence_df = local_test_df[local_test_df['sequence_id'] == first_seq_id]
    
    pl_sequence = pl.from_pandas(first_sequence_df)
    pl_demographics = pl.from_pandas(local_demographics_df)
    
    test_probs = predict_miyairi(pl_sequence, pl_demographics)
    print(f"✅ ローカルテスト成功。予測確率の形状: {test_probs.shape}")

except Exception as e:
    print(f"❌ ローカルテスト中にエラーが発生しました: {e}")


### import libs
import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.utils import Sequence, to_categorical, pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate, GRU, GaussianNoise
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold
from scipy.spatial.transform import Rotation as R

### fix seed
import random
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed_everything(seed=42)

### configuration
# (Competition metric will only be imported when TRAINing)
TRAIN = False                  # ← set to True when you want to train
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
# PRETRAINED_DIR = Path("/kaggle/input/quit-diff2")  # used when TRAIN=False
PRETRAINED_DIR = Path("/kaggle/input/model_nomura/tensorflow2/1/3") #一つのセッションで学習と推論回せる場合は、とりあえずworking dirに保存しておく
EXPORT_DIR = Path("./")                                    # artefacts will be saved here
BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 5e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 10
PATIENCE = 40
print("▶ imports ready · tensorflow", tf.__version__)

### Utility functions
#Tensor Manipulations
def time_sum(x):
    return K.sum(x, axis=1)

def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

def se_block(x, reduction=8):
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

# Residual CNN Block with SE
def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    shortcut = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                          kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x

def attention_layer(inputs):
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context

### Data helpers
# Normalizes and cleans the time series sequence. 
def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

# MixUp the data argumentation in order to regularize the neural network. 
class MixupGenerator(Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2):
        self.X, self.y = X, y
        self.batch = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(X))
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))
    def __getitem__(self, i):
        idx = self.indices[i*self.batch:(i+1)*self.batch]
        Xb, yb = self.X[idx], self.y[idx]
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(Xb))
        X_mix = lam * Xb + (1-lam) * Xb[perm]
        y_mix = lam * yb + (1-lam) * yb[perm]
        return X_mix, y_mix
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def remove_gravity_from_acc(acc_data, rot_data):

    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :] 
            continue

        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
             linear_accel[i, :] = acc_values[i, :]
             
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200): # Assuming 200Hz sampling rate
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]

        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)

            # Calculate the relative rotation
            delta_rot = rot_t.inv() * rot_t_plus_dt
            
            # Convert delta rotation to angular velocity vector
            # The rotation vector (Euler axis * angle) scaled by 1/dt
            # is a good approximation for small delta_rot
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            # If quaternion is invalid, angular velocity remains zero
            pass
            
    return angular_vel

### Model difinition
def build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    inp = Input(shape=(pad_len, imu_dim+tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    # IMU deep branch
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)

    # TOF/Thermal lighter branch
    x2 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)
    x2 = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x2)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)

    merged = Concatenate()([x1, x2])

    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xc = GaussianNoise(0.09)(merged)
    xc = Dense(16, activation='elu')(xc)
    
    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    x = attention_layer(x)

    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Dropout(drop)(x)

    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    return Model(inp, out)

tmp_model = build_two_branch_model(127,7,325,18)

### Train or inference
if TRAIN:
    print("▶ TRAIN MODE – loading dataset …")
    df = pd.read_csv(RAW_DIR / "train.csv")

    train_dem_df = pd.read_csv(RAW_DIR / "train_demographics.csv")
    df_for_groups = pd.merge(df.copy(), train_dem_df, on='subject', how='left')

    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)
    gesture_classes = le.classes_

    print("  Calculating base engineered IMU features (magnitude, angle)...")
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    
    print("  Calculating engineered IMU derivatives (jerk, angular velocity) for original acc_mag...")
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    print("  Removing gravity and calculating linear acceleration features...")
    
    linear_accel_list = []
    for _, group in df.groupby('sequence_id'):
        acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
        linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))
    
    df_linear_accel = pd.concat(linear_accel_list)
    df = pd.concat([df, df_linear_accel], axis=1)

    df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
    df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)

    print("  Calculating angular velocity from quaternion derivatives...")
    angular_vel_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group)
        angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))
    
    df_angular_vel = pd.concat(angular_vel_list)
    df = pd.concat([df, df_angular_vel], axis=1)

    print("  Calculating angular jerk from angular velocity...")
    df['angular_jerk_x'] = df.groupby('sequence_id')['angular_vel_x'].diff().fillna(0)
    df['angular_jerk_y'] = df.groupby('sequence_id')['angular_vel_y'].diff().fillna(0)
    df['angular_jerk_z'] = df.groupby('sequence_id')['angular_vel_z'].diff().fillna(0)

    print("  Calculating angular snap from angular jerk...")
    df['angular_snap_x'] = df.groupby('sequence_id')['angular_jerk_x'].diff().fillna(0)
    df['angular_snap_y'] = df.groupby('sequence_id')['angular_jerk_y'].diff().fillna(0)
    df['angular_snap_z'] = df.groupby('sequence_id')['angular_jerk_z'].diff().fillna(0)

    meta_cols = { } # This was an empty dict in your provided code, keeping it as is.

    imu_cols_base = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z']
    imu_cols_base.extend([c for c in df.columns if c.startswith('rot_') and c not in ['rot_angle', 'rot_angle_vel']])
    
    imu_engineered_features = [
        'acc_mag', 'rot_angle',
        'acc_mag_jerk', 'rot_angle_vel',
        'linear_acc_mag', 'linear_acc_mag_jerk',
        'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
        'angular_jerk_x', 'angular_jerk_y', 'angular_jerk_z',
        'angular_snap_x', 'angular_snap_y', 'angular_snap_z' # Added new angular snap features
    ]
    imu_cols = imu_cols_base + imu_engineered_features
    imu_cols = list(dict.fromkeys(imu_cols))

    thm_cols_original = [c for c in df.columns if c.startswith('thm_')]
    
    tof_aggregated_cols_template = []
    for i in range(1, 6):
        tof_aggregated_cols_template.extend([f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max'])

    final_feature_cols = imu_cols + thm_cols_original + tof_aggregated_cols_template
    imu_dim_final = len(imu_cols)
    tof_thm_aggregated_dim_final = len(thm_cols_original) + len(tof_aggregated_cols_template)
    
    print(f"  IMU (incl. engineered & derivatives) {imu_dim_final} | THM + Aggregated TOF {tof_thm_aggregated_dim_final} | total {len(final_feature_cols)} features")
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(final_feature_cols))

    print("  Building sequences with aggregated TOF and preparing data for scaler...")
    seq_gp = df.groupby('sequence_id') 
    
    all_steps_for_scaler_list = []
    X_list_unscaled, y_list_int_for_stratify, lens = [], [], [] 

    for seq_id, seq_df_orig in seq_gp:
        seq_df = seq_df_orig.copy()

        for i in range(1, 6):
            pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
            tof_sensor_data = seq_df[pixel_cols_tof].replace(-1, np.nan)
            seq_df[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1)
            seq_df[f'tof_{i}_std']  = tof_sensor_data.std(axis=1)
            seq_df[f'tof_{i}_min']  = tof_sensor_data.min(axis=1)
            seq_df[f'tof_{i}_max']  = tof_sensor_data.max(axis=1)
        
        mat_unscaled = seq_df[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')
        
        all_steps_for_scaler_list.append(mat_unscaled)
        X_list_unscaled.append(mat_unscaled)
        y_list_int_for_stratify.append(seq_df['gesture_int'].iloc[0])
        lens.append(len(mat_unscaled))

    print("  Fitting StandardScaler...")
    all_steps_concatenated = np.concatenate(all_steps_for_scaler_list, axis=0)
    scaler = StandardScaler().fit(all_steps_concatenated)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")
    del all_steps_for_scaler_list, all_steps_concatenated

    print("  Scaling and padding sequences...")
    X_scaled_list = [scaler.transform(x_seq) for x_seq in X_list_unscaled]
    del X_list_unscaled

    pad_len = int(np.percentile(lens, PAD_PERCENTILE))
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    
    X = pad_sequences(X_scaled_list, maxlen=pad_len, padding='post', truncating='post', dtype='float32')
    del X_scaled_list
    
    y_int_for_stratify = np.array(y_list_int_for_stratify)
    y = to_categorical(y_int_for_stratify, num_classes=len(le.classes_))

    print("  Splitting data and preparing for training...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=82, stratify=y_int_for_stratify)

    cw_vals = compute_class_weight('balanced', classes=np.arange(len(le.classes_)), y=y_int_for_stratify)
    class_weight = dict(enumerate(cw_vals))

    model = build_two_branch_model(pad_len, imu_dim_final, tof_thm_aggregated_dim_final, len(le.classes_), wd=WD)
    
    steps = len(X_tr) // BATCH_SIZE
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(5e-4, first_decay_steps=15 * steps) 
    
    model.compile(optimizer=Adam(lr_sched),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    train_gen = MixupGenerator(X_tr, y_tr, batch_size=BATCH_SIZE, alpha=MIXUP_ALPHA)
    cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1, monitor='val_accuracy', mode='max')
    
    print("  Starting model training...")
    model.fit(train_gen, epochs=EPOCHS, validation_data=(X_val, y_val),
              class_weight=class_weight, callbacks=[cb], verbose=1)

    model.save(EXPORT_DIR / "gesture_two_branch_mixup.h5")
    print("✔ Training done – artefacts saved in", EXPORT_DIR)

    from cmi_2025_metric_copy_for_import import CompetitionMetric
    preds_val = model.predict(X_val).argmax(1)
    true_val_int  = y_val.argmax(1)
    
    h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': le.classes_[true_val_int]}),
        pd.DataFrame({'gesture': le.classes_[preds_val]}))
    print("Hold‑out H‑F1 =", round(h_f1, 4))
else:
    print("▶ INFERENCE MODE – loading artefacts from", PRETRAINED_DIR)
    final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    pad_len        = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
    scaler         = joblib.load(PRETRAINED_DIR / "scaler.pkl")
    gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    # Re-calculate imu_dim_final based on the actual features that will be used
    imu_features_in_final_cols = [c for c in final_feature_cols if any(c.startswith(prefix) for prefix in ['linear_acc_', 'acc_', 'rot_', 'angular_vel_', 'angular_jerk_', 'angular_snap_'])]
    imu_dim_final = len(imu_features_in_final_cols)

    tof_thm_aggregated_dim_final = len(final_feature_cols) - imu_dim_final

    custom_objs = {
        'time_sum': time_sum,
        'squeeze_last_axis': squeeze_last_axis,
        'expand_last_axis': expand_last_axis,
        'se_block': se_block,
        'residual_se_cnn_block': residual_se_cnn_block,
        'attention_layer': attention_layer,
    }
    model = load_model(PRETRAINED_DIR / "gesture_two_branch_mixup.h5",
                       compile=False, custom_objects=custom_objs)
    print("  Model, scaler, feature_cols, pad_len loaded – ready for evaluation")


def predict_nomura(sequence: pl.DataFrame, demographics: pl.DataFrame) -> np.ndarray:
    df_seq = sequence.to_pandas()

    df_seq['acc_mag'] = np.sqrt(df_seq['acc_x']**2 + df_seq['acc_y']**2 + df_seq['acc_z']**2)
    df_seq['rot_angle'] = 2 * np.arccos(df_seq['rot_w'].clip(-1, 1))
    df_seq['acc_mag_jerk'] = df_seq['acc_mag'].diff().fillna(0)
    df_seq['rot_angle_vel'] = df_seq['rot_angle'].diff().fillna(0)

    acc_cols = ['acc_x', 'acc_y', 'acc_z']
    rot_cols = ['rot_x', 'rot_y', 'rot_z', 'rot_w']

    if all(col in df_seq.columns for col in acc_cols + rot_cols):
        acc_data = df_seq[acc_cols]
        rot_data = df_seq[rot_cols]
        linear_acc = remove_gravity_from_acc(acc_data, rot_data)
        df_seq['linear_acc_x'] = linear_acc[:, 0]
        df_seq['linear_acc_y'] = linear_acc[:, 1]
        df_seq['linear_acc_z'] = linear_acc[:, 2]
    else:
        df_seq['linear_acc_x'] = df_seq.get('acc_x', 0)
        df_seq['linear_acc_y'] = df_seq.get('acc_y', 0)
        df_seq['linear_acc_z'] = df_seq.get('acc_z', 0)

    df_seq['linear_acc_mag'] = np.sqrt(
        df_seq['linear_acc_x']**2 + df_seq['linear_acc_y']**2 + df_seq['linear_acc_z']**2
    )
    df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)

    if all(col in df_seq.columns for col in rot_cols):
        angular_vel = calculate_angular_velocity_from_quat(df_seq[rot_cols])
        df_seq['angular_vel_x'] = angular_vel[:, 0]
        df_seq['angular_vel_y'] = angular_vel[:, 1]
        df_seq['angular_vel_z'] = angular_vel[:, 2]
        for kind in ['jerk', 'snap']:
            for axis in ['x', 'y', 'z']:
                prev = f'angular_{ "vel" if kind == "jerk" else "jerk" }_{axis}'
                curr = f'angular_{kind}_{axis}'
                df_seq[curr] = df_seq[prev].diff().fillna(0)
    else:
        for kind in ['vel', 'jerk', 'snap']:
            for axis in ['x', 'y', 'z']:
                df_seq[f'angular_{kind}_{axis}'] = 0

    for i in range(1, 6): 
        cols = [f"tof_{i}_v{p}" for p in range(64)]
        if all(col in df_seq.columns for col in cols):
            tof = df_seq[cols].replace(-1, np.nan)
            df_seq[f'tof_{i}_mean'] = tof.mean(axis=1)
            df_seq[f'tof_{i}_std']  = tof.std(axis=1)
            df_seq[f'tof_{i}_min']  = tof.min(axis=1)
            df_seq[f'tof_{i}_max']  = tof.max(axis=1)
        else:
            for stat in ['mean', 'std', 'min', 'max']:
                df_seq[f'tof_{i}_{stat}'] = 0

    if 'tof_range_across_sensors' in final_feature_cols:
        tof_means = [f'tof_{i}_mean' for i in range(1, 6) if f'tof_{i}_mean' in df_seq]
        thm_cols = [f'thm_{i}' for i in range(1, 6) if f'thm_{i}' in df_seq]
        if tof_means:
            t = df_seq[tof_means]
            df_seq['tof_range_across_sensors'] = t.max(axis=1) - t.min(axis=1)
            df_seq['tof_std_across_sensors'] = t.std(axis=1)
        else:
            df_seq['tof_range_across_sensors'] = 0
            df_seq['tof_std_across_sensors'] = 0
        if thm_cols:
            t = df_seq[thm_cols]
            df_seq['thm_range_across_sensors'] = t.max(axis=1) - t.min(axis=1)
            df_seq['thm_std_across_sensors'] = t.std(axis=1)
        else:
            df_seq['thm_range_across_sensors'] = 0
            df_seq['thm_std_across_sensors'] = 0

    # 特徴量選択と前処理
    df_feats = pd.DataFrame(index=df_seq.index)
    for col in final_feature_cols:
        df_feats[col] = df_seq.get(col, 0)
    mat = df_feats.ffill().bfill().fillna(0).values.astype('float32')
    mat_scaled = scaler.transform(mat)
    pad_input = pad_sequences([mat_scaled], maxlen=pad_len, padding='post', truncating='post', dtype='float32')

    # 推論結果を確率で返す（argmaxしない）
    probs = model.predict(pad_input, verbose=0)  # shape = (1, n_classes)
    return probs.astype('float32')


import os, json, joblib, numpy as np, pandas as pd
import random, math
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from timm.scheduler import CosineLRScheduler
from scipy.signal import firwin
from cmi_2025_metric_copy_for_import import CompetitionMetric

from tqdm import tqdm

import lightgbm as lgb

import polars as pl

import pickle

# ================================
# TTA Helpers
# ================================
def tta_jitter(x, sigma=0.01):
    noise = np.random.randn(*x.shape) * sigma
    return x + noise

def tta_scaling(x, scale_low=0.95, scale_high=1.05):
    factor = np.random.uniform(scale_low, scale_high, size=(1, x.shape[1]))
    return x * factor

def tta_time_shift(x, max_shift=5):
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, shift, axis=0)

def tta_channel_dropout(x, p_drop=0.1):
    mask = np.ones(x.shape[1], dtype=x.dtype)
    drop_idx = np.random.choice(x.shape[1],
                                size=int(p_drop * x.shape[1]),
                                replace=False)
    mask[drop_idx] = 0
    return x * mask[np.newaxis, :]

# Configuration
TRAIN = False                # ← set to True when you want to train
RAW_DIR = Path("../input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/radon_auto_update_model/pytorch/default/1")#Path("/kaggle/input/cmi3-models-p") # used when TRAIN=False
EXPORT_DIR = Path("/kaggle/working")                                    # artefacts will be saved here
BATCH_SIZE = 64
PAD_PERCENTILE = 100
maxlen = PAD_PERCENTILE
LR_INIT = 1e-3
WD = 3e-3
# MIXUP_ALPHA = 0.4
PATIENCE = 40
FOLDS = 5
random_state = 42
epochs_warmup = 20
warmup_lr_init = 1.822126131809773e-05
lr_min = 3.810323058740104e-09

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"▶ imports ready · pytorch {torch.__version__} · device: {device}")

# ================================
# Model Components
# ================================
mean = torch.tensor([
    0,  0, 0, 0, 0,
    0,  9.0319e-03,  1.0849e+00, -2.6186e-03,  3.7651e-03,
    -5.3660e-03, -2.8177e-03,  1.3318e-03, -1.5876e-04,  6.3495e-01,
     6.2877e-01,  6.0607e-01,  6.2142e-01,  6.3808e-01,  6.5420e-01,
     7.4102e-03, -3.4159e-03, -7.5237e-03, -2.6034e-02,  2.9704e-02,
    -3.1546e-02, -2.0610e-03, -4.6986e-03, -4.7216e-03, -2.6281e-02,
     1.5799e-02,  1.0016e-02,0,0,0,0,0,0,
], dtype=torch.float32).view(1, -1, 1).to(device)         

std = torch.tensor([
    1, 1, 1, 1, 1, 1, 0.2067, 0.8583, 0.3162,
    0.2668, 0.2917, 0.2341, 0.3023, 0.3281, 1.0264, 0.8838, 0.8686, 1.0973,
    1.0267, 0.9018, 0.4658, 0.2009, 0.2057, 1.2240, 0.9535, 0.6655, 0.2941,
    0.3421, 0.8156, 0.6565, 1.1034, 1.5577,1,1,1,1,1,1,
], dtype=torch.float32).view(1, -1, 1).to(device) + 1e-8  

class ImuFeatureExtractor(nn.Module):
    def __init__(self, fs=100., add_quaternion=False):
        super().__init__()
        self.fs = fs
        self.add_quaternion = add_quaternion

        k = 15
        self.lpf = nn.Conv1d(6, 6, kernel_size=k, padding=k//2,
                             groups=6, bias=False)
        nn.init.kaiming_uniform_(self.lpf.weight, a=math.sqrt(5))

        self.lpf_acc  = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
        self.lpf_gyro = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)

    def forward(self, imu):
        # imu: 
        B, C, T = imu.shape
        acc  = imu[:, 0:3, :]                 # acc_x, acc_y, acc_z
        gyro = imu[:, 3:6, :]                 # gyro_x, gyro_y, gyro_z
        extra = imu[:, 6:, :]                 

        # 1) magnitude
        acc_mag  = torch.norm(acc,  dim=1, keepdim=True)          # (B,1,T)
        gyro_mag = torch.norm(gyro, dim=1, keepdim=True)

        # 2) jerk 
        jerk = F.pad(acc[:, :, 1:] - acc[:, :, :-1], (1,0))       # (B,3,T)
        gyro_delta = F.pad(gyro[:, :, 1:] - gyro[:, :, :-1], (1,0))

        # 3) energy
        acc_pow  = acc ** 2
        gyro_pow = gyro ** 2

        # 4) LPF / HPF 
        acc_lpf  = self.lpf_acc(acc)
        acc_hpf  = acc - acc_lpf
        gyro_lpf = self.lpf_gyro(gyro)
        gyro_hpf = gyro - gyro_lpf

        # velocity
        velocity = torch.cumsum(acc, dim=2)

        # position
        position = torch.cumsum(velocity, dim=2)

        features = [
            acc, gyro,
            acc_mag, gyro_mag,
            jerk, gyro_delta,
            acc_pow, gyro_pow,
            acc_lpf, acc_hpf,
            gyro_lpf, gyro_hpf,
            velocity, position,
        ]
        return torch.cat(features, dim=1)  # (B, C_out, T)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        # First conv
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv
        out = self.bn2(self.conv2(out))
        
        # SE block
        out = self.se(out)
        
        # Add shortcut
        out += shortcut
        out = F.relu(out)
        
        # Pool and dropout
        out = self.pool(out)
        out = self.dropout(out)
        
        return out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context

class TwoBranchModel(nn.Module):
    def __init__(self, pad_len, imu_dim_raw, tof_dim, n_classes, dropouts=[0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.3, 0.4, 0.3], feature_engineering=True, **kwargs):
        super().__init__()
        self.feature_engineering = feature_engineering
        if feature_engineering:
            self.imu_fe = ImuFeatureExtractor(**kwargs)
            imu_dim = 38            
        else:
            self.imu_fe = nn.Identity()
            imu_dim = imu_dim_raw   
            
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim

        self.fir_nchan = 7

        weight_decay = 3e-3

        numtaps = 33  
        fir_coef = firwin(numtaps, cutoff=1.0, fs=10.0, pass_zero=False)
        fir_kernel = torch.tensor(fir_coef, dtype=torch.float32).view(1, 1, -1)
        fir_kernel = fir_kernel.repeat(7, 1, 1)  # (imu_dim, 1, numtaps)
        self.register_buffer("fir_kernel", fir_kernel)
        
        # IMU deep branch
        self.imu_block1 = ResidualSECNNBlock(imu_dim, 64, 3, dropout=dropouts[0], weight_decay=weight_decay)
        self.imu_block2 = ResidualSECNNBlock(64, 128, 5, dropout=dropouts[1], weight_decay=weight_decay)
        
        # TOF/Thermal lighter branch
        self.tof_conv1 = nn.Conv1d(tof_dim, 64, 3, padding=1, bias=False)
        self.tof_bn1 = nn.BatchNorm1d(64)
        self.tof_pool1 = nn.MaxPool1d(2)
        self.tof_drop1 = nn.Dropout(dropouts[2])
        
        self.tof_conv2 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        self.tof_bn2 = nn.BatchNorm1d(128)
        self.tof_pool2 = nn.MaxPool1d(2)
        self.tof_drop2 = nn.Dropout(dropouts[3])
        
        # BiLSTM
        self.bilstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropouts[4])
        
        # Attention
        self.attention = AttentionLayer(256)  # 128*2 for bidirectional
        
        # Dense layers
        self.dense2_gr = nn.Linear(263, 128, bias=False)
        
        self.dense1 = nn.Linear(256, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropouts[5])
        
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropouts[6])
        
        self.dense3 = nn.Linear(128, 512, bias=False)
        self.bn_dense3 = nn.BatchNorm1d(512)
        self.drop3 = nn.Dropout(dropouts[7])

        self.dense4 = nn.Linear(512, 128, bias=False)
        self.bn_dense4 = nn.BatchNorm1d(128)
        self.drop4 = nn.Dropout(dropouts[8])
        
        
        self.classifier = nn.Linear(128, n_classes)
        
    def forward(self, x, grs, use_gr=False):
        # grs (batch, 1, 7)
        # Split input
        
        imu = x[:, :, :self.fir_nchan].transpose(1, 2)  # (batch, imu_dim, seq_len)
        tof = x[:, :, self.fir_nchan:].transpose(1, 2)  # (batch, tof_dim, seq_len)

        imu = self.imu_fe(imu)   # (B, imu_dim, T)
        filtered = F.conv1d(
            imu[:, :self.fir_nchan, :],        # (B,7,T)
            self.fir_kernel,
            padding=self.fir_kernel.shape[-1] // 2,
            groups=self.fir_nchan,
        )
        
        imu = torch.cat([filtered, imu[:, self.fir_nchan:, :]], dim=1)  
        imu = (imu - mean) / std 
        # IMU branch
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)
        
        # TOF branch
        x2 = F.relu(self.tof_bn1(self.tof_conv1(tof)))
        x2 = self.tof_drop1(self.tof_pool1(x2))
        x2 = F.relu(self.tof_bn2(self.tof_conv2(x2)))
        x2 = self.tof_drop2(self.tof_pool2(x2))
        
        # Concatenate branches
        merged = torch.cat([x1, x2], dim=1).transpose(1, 2)  # (batch, seq_len, 256)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(merged)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Attention
        attended = self.attention(lstm_out)
        
        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        if use_gr:
            x = torch.cat([x, grs], dim=1)
            x = F.relu(self.bn_dense2(self.dense2_gr(x)))
        else:
            x = F.relu(self.bn_dense2(self.dense2(x)))
            x = self.drop2(x)
        x = F.relu(self.bn_dense3(self.dense3(x)))
        x = self.drop3(x)
        x = F.relu(self.bn_dense4(self.dense4(x)))
        x = self.drop4(x)
        
        # Classification
        logits = (self.classifier(x))
        return logits
    def extract_features(self, x, grs, use_gr=False):
        imu = x[:, :, :self.fir_nchan].transpose(1, 2)  # (batch, imu_dim, seq_len)
        tof = x[:, :, self.fir_nchan:].transpose(1, 2)  # (batch, tof_dim, seq_len)

        imu = self.imu_fe(imu)   # (B, imu_dim, T)
        filtered = F.conv1d(
            imu[:, :self.fir_nchan, :],        # (B,7,T)
            self.fir_kernel,
            padding=self.fir_kernel.shape[-1] // 2,
            groups=self.fir_nchan,
        )
        
        imu = torch.cat([filtered, imu[:, self.fir_nchan:, :]], dim=1)  
        imu = (imu - mean) / std 
        # IMU branch
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)
        
        # TOF branch
        x2 = F.relu(self.tof_bn1(self.tof_conv1(tof)))
        x2 = self.tof_drop1(self.tof_pool1(x2))
        x2 = F.relu(self.tof_bn2(self.tof_conv2(x2)))
        x2 = self.tof_drop2(self.tof_pool2(x2))
        
        # Concatenate branches
        merged = torch.cat([x1, x2], dim=1).transpose(1, 2)  # (batch, seq_len, 256)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(merged)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Attention
        attended = self.attention(lstm_out)
        
        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        if use_gr:
            x = torch.cat([x, grs], dim=1)
            x = F.relu(self.bn_dense2(self.dense2_gr(x)))
        else:
            x = F.relu(self.bn_dense2(self.dense2(x)))
            x = self.drop2(x)
        x = F.relu(self.bn_dense3(self.dense3(x)))
        x = self.drop3(x)
        x = F.relu(self.bn_dense4(self.dense4(x)))
        x = self.drop4(x)
        return x


# ================================
# Data Handling
# ================================

def pad_sequences_torch(sequences, maxlen, padding='post', truncating='post', value=0.0):
    result = []
    for seq in sequences:
        if len(seq) >= maxlen:
            seq = seq[:maxlen] if truncating=='post' else seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            pad_array = np.full((pad_len, seq.shape[1]), value)
            seq = np.concatenate([seq, pad_array], axis=0) if padding=='post' else np.concatenate([pad_array, seq], axis=0)
        result.append(seq)
    return np.array(result, dtype=np.float32)

def preprocess_sequence_radon(df_seq: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

class CMI3Dataset(Dataset):
    def __init__(self,
                 X_list,
                 y_list,
                 gr_list,
                 maxlen,
                 mode="train",
                 imu_dim=7,
                 augment=None):
        self.X_list = X_list
        self.mode = mode
        self.y_list = y_list
        self.gr_list = gr_list
        self.maxlen = maxlen
        self.imu_dim = imu_dim     
        self.augment = augment   

    def pad_sequences_torch(self, seq, maxlen, padding='post', truncating='post', value=0.0):

        if seq.shape[0] >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:  # 'pre'
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - seq.shape[0]
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        return seq  
        
    def __getitem__(self, index):
        X = self.X_list[index]
        y = self.y_list[index]

        # ---------- (A)  Augmentation ----------
        if self.mode == "train" and self.augment is not None:
            X = self.augment(X, self.imu_dim)     

        X = self.pad_sequences_torch(X, self.maxlen, 'pre', 'pre')
        return (X, self.gr_list[index]), y
    
    def __len__(self):
        return len(self.X_list)


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

def set_seed(seed: int = 42):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

class Augment:
    def __init__(self,
                 p_jitter=0.8, sigma=0.02, scale_range=[0.9,1.1],
                 p_dropout=0.3,
                 p_moda=0.5,          
                 drift_std=0.005,     
                 drift_max=0.25):      
        self.p_jitter  = p_jitter
        self.sigma     = sigma
        self.scale_min, self.scale_max = scale_range
        self.p_dropout = p_dropout
        self.p_moda    = p_moda
        self.drift_std = drift_std
        self.drift_max = drift_max

    # ---------- Jitter & Scaling ----------
    def jitter_scale(self, x: np.ndarray) -> np.ndarray:
        noise  = np.random.randn(*x.shape) * self.sigma
        scale  = np.random.uniform(self.scale_min,
                                   self.scale_max,
                                   size=(1, x.shape[1]))
        return (x + noise) * scale

    # ---------- Sensor Drop-out ----------
    def sensor_dropout(self,
                       x: np.ndarray,
                       imu_dim: int) -> np.ndarray:

        if random.random() < self.p_dropout:
            x[:, imu_dim:] = 0.0
        return x

    def motion_drift(self, x: np.ndarray, imu_dim: int) -> np.ndarray:

        T = x.shape[0]

        drift = np.cumsum(
            np.random.normal(scale=self.drift_std, size=(T, 1)),
            axis=0
        )
        drift = np.clip(drift, -self.drift_max, self.drift_max)   

        x[:, :6] += drift

        if imu_dim > 6:
            x[:, 6:imu_dim] += drift     
        return x
    
    # ---------- master call ----------
    def __call__(self,
                 x: np.ndarray,
                 imu_dim: int) -> np.ndarray:
        if random.random() < self.p_jitter:
            x = self.jitter_scale(x)

        if random.random() < self.p_moda:
            x = self.motion_drift(x, imu_dim)

        x = self.sensor_dropout(x, imu_dim)
        return x


# ================================
# Training Pipeline
# ================================
if TRAIN:
    print("▶ TRAIN MODE – loading dataset …")
    df = pd.read_csv(RAW_DIR / "train.csv")
    df_gr = pd.read_csv(RAW_DIR / "train_demographics.csv")

    # Label encoding
    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

    # Feature list
    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]
    print(f"  IMU {len(imu_cols)} | TOF/THM {len(tof_cols)} | total {len(feature_cols)} features")

    # Global scaler
    scaler = StandardScaler().fit(df[feature_cols].ffill().bfill().fillna(0).values)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")

    # Build sequences
    seq_gp = df.groupby('sequence_id')
    X_list, y_list, id_list, gr_list = [], [], [], []
    for seq_id, seq in seq_gp:
        mat = preprocess_sequence_radon(seq, feature_cols, scaler)
        X_list.append(mat)
        y_list.append(seq['gesture_int'].iloc[0])
        id_list.append(seq_id)
        gr_list.append(df_gr[df_gr['subject'] == seq['subject'].iloc[0]].drop(columns='subject').to_numpy().reshape(-1))
        # lens.append(len(mat))
    
    pad_len = PAD_PERCENTILE#int(np.percentile(lens, PAD_PERCENTILE))
    print(pad_len)
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(feature_cols))
    id_list = np.array(id_list)
    X_list_all = pad_sequences_torch(X_list, maxlen=pad_len, padding='pre', truncating='pre')
    y_list_all = np.eye(len(le.classes_))[y_list].astype(np.float32)  # One-hot encoding
    gr_list = np.array(gr_list).astype(np.float32)

    augmenter = Augment(
        p_jitter=0.9844818619033621, sigma=0.03291295776089293, scale_range=(0.7542342630597011,1.1625052821731077),
        p_dropout=0.41782786013520684,
        p_moda=0.3910622476959722, drift_std=0.0040285239353308015, drift_max=0.3929358950258158    
    )


EPOCHS = 125
if TRAIN:
    # Split
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=random_state)
    models = []
    lgbm_models = []
    lgbm_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(id_list, np.argmax(y_list_all, axis=1))):

        train_list= X_list_all[train_idx]
        train_y_list= y_list_all[train_idx]
        train_gr_list = gr_list[train_idx]
        val_list = X_list_all[val_idx]
        val_y_list= y_list_all[val_idx]
        val_gr_list = gr_list[train_idx]

        
        # Data loaders
        train_dataset = CMI3Dataset(train_list, train_y_list,train_gr_list, maxlen, mode="train", imu_dim=len(imu_cols),
                                augment=augmenter)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=True)
    
        val_dataset = CMI3Dataset(val_list, val_y_list,val_gr_list, maxlen, mode="val")
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,drop_last=True)

    
        # Model
        model = TwoBranchModel(maxlen, len(imu_cols), len(tof_cols), 
                      len(le.classes_)).to(device)
        ema = EMA(model, decay=0.999)
        # Optimizer and scheduler
        optimizer = Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
        
        steps_per_epoch = len(train_loader)
        nbatch = len(train_loader)
        warmup = epochs_warmup * nbatch
        nsteps = EPOCHS * nbatch
        scheduler = CosineLRScheduler(optimizer,
                          warmup_t=warmup, warmup_lr_init=warmup_lr_init, warmup_prefix=True,
                          t_initial=(nsteps - warmup), lr_min=lr_min) 
    
        early_stopping = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        val_best_acc = 0.0
        i_scheduler = 0
        
        # Training loop
        print("▶ Starting training...")
        for epoch in tqdm(range(EPOCHS)):
            model.train()
            train_preds = []
            train_targets = []
            for X_, y in (train_loader):  
                X, grs = X_
                X, y, grs = X.float().to(device), y.to(device), grs.to(device)
                optimizer.zero_grad()
                logits = model(X, grs)
    
                loss = -torch.sum(F.log_softmax(logits, dim=1) * y, dim=1).mean()
                loss.backward()
                optimizer.step()
                ema.update(model)
                train_preds.extend(logits.argmax(dim=1).cpu().numpy())
                train_targets.extend(y.argmax(dim=1).cpu().numpy())
                scheduler.step(i_scheduler)
                i_scheduler +=1
    
                train_loss += loss.item()
                
            model.eval()
            with torch.inference_mode():
                val_preds = []
                val_targets = []
                for X_, y in (val_loader):
                    X, grs = X_
                    half = BATCH_SIZE // 2         

                    x_front = X[:half]               
                    x_back  = X[half:].clone()      
                    
                    x_back[:, :, 7:] = 0.0    
                    X = torch.cat([x_front, x_back], dim=0)  # (B, C, T)
                    X, y, grs = X.float().to(device), y.to(device), grs.to(device)
                    
                    logits = model(X, grs)
                    val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    val_targets.extend(y.argmax(dim=1).cpu().numpy())
                    
                    loss = F.cross_entropy(logits, y)
                    val_loss += loss.item()
    
            train_acc = CompetitionMetric().calculate_hierarchical_f1(
                pd.DataFrame({'gesture': le.classes_[train_targets]}),
                pd.DataFrame({'gesture': le.classes_[train_preds]}))
            val_acc = CompetitionMetric().calculate_hierarchical_f1(
                pd.DataFrame({'gesture': le.classes_[val_targets]}),
                pd.DataFrame({'gesture': le.classes_[val_preds]}))
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

        def extract_features_from_loader(loader, model_to_use):
            """指定されたモデルとデータローダーから特徴量とラベルを抽出するヘルパー関数"""
            all_feats = []
            all_lbls = []
            with torch.no_grad():
                for X_, y in loader:
                    X, grs = X_
                    X, grs = X.float().to(device), grs.to(device)
                    # extract_featuresメソッドを呼び出す
                    features = model_to_use.extract_features(X, grs)
                    all_feats.append(features.cpu().numpy())
                    all_lbls.append(y.argmax(dim=1).cpu().numpy())
            return np.concatenate(all_feats), np.concatenate(all_lbls)

        # 学習データと検証データの両方から特徴量を抽出
        X_train_features, y_train_lgbm = extract_features_from_loader(train_loader, model)
        X_val_features, y_val_lgbm = extract_features_from_loader(val_loader, model)
            
        print(f"✔ Features extracted. Train shape: {X_train_features.shape}, Val shape: {X_val_features.shape}")
    
        # --- Step 3: LightGBMの学習と評価 ---
        print("\n▶ Step 3: Training and evaluating LightGBM model for this fold...")
            
        lgbm = lgb.LGBMClassifier(objective='multiclass', random_state=random_state, n_estimators=1000, num_leaves=50, verbose=-1, verbose_eval=False)
            
        lgbm.fit(X_train_features, y_train_lgbm,
            eval_set=[(X_val_features, y_val_lgbm)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(100, verbose=False)])
        lgbm_models.append(lgbm)
            
        # 検証データで予測
        preds_lgbm = lgbm.predict(X_val_features)
            
        # CompetitionMetricで評価
        true_gestures = le.classes_[y_val_lgbm]
        pred_gestures = le.classes_[preds_lgbm]
        true_df = pd.DataFrame({'gesture': true_gestures})
        pred_df = pd.DataFrame({'gesture': pred_gestures})
            
        metric = CompetitionMetric()
        score = metric.calculate_hierarchical_f1(true_df, pred_df)
        lgbm_scores.append(score)
        print(f"✔ Fold {fold} LGBM Competition Score: {score:.4f}")
        models.append(model)
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'imu_dim': len(imu_cols),
            'tof_dim': len(tof_cols),
            'n_classes': len(le.classes_),
            'pad_len': pad_len
        }, EXPORT_DIR / f"gesture_two_branch_fold{fold}.pth")
        
        print(f"fold: {fold} train_all_acc: {train_acc:.4f} val_all_acc: {val_acc:.4f}")
        print("✔ Training done – artefacts saved in", EXPORT_DIR)
        
    pickle.dump(lgbm_models, open(EXPORT_DIR / 'trained_lgbm_models.pkl', 'wb'))

else:
    print("▶ INFERENCE MODE – loading artefacts from", PRETRAINED_DIR)

    lgbm_models = pickle.load(open(PRETRAINED_DIR / 'trained_lgbm_models.pkl', 'rb'))
    
    feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
    scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
    gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]

    MODELS = [f'gesture_two_branch_fold{i}.pth' for i in range(FOLDS)]
    models = []
    for path in MODELS:
        checkpoint = torch.load(PRETRAINED_DIR / path, map_location=device)
        model = TwoBranchModel(
            checkpoint['pad_len'],
            checkpoint['imu_dim'],
            checkpoint['tof_dim'],
            checkpoint['n_classes']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)

    print("  model, scaler, pads loaded – ready for evaluation")

# Make sure gesture_classes exists in both modes
if TRAIN:
    gesture_classes = le.classes_


def predict_radon(sequence: pl.DataFrame, demographics: pl.DataFrame, n_tta=8):
    """Prediction function with TTA"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global gesture_classes
    if gesture_classes is None:
        gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    # Original pretreatment
    df_seq = sequence.to_pandas()
    df_gr = demographics.to_pandas()
    mat = preprocess_sequence_radon(df_seq, feature_cols, scaler)

    # Generate n_tta enhanced samples
    tta_funcs = [tta_jitter, tta_scaling, tta_time_shift, tta_channel_dropout]
    tta_samples = []
    tta_grs = []
    for _ in range(n_tta):
        x_aug = mat.copy()
        funcs = np.random.choice(tta_funcs, size=3, replace=False)
        for f in funcs:
            x_aug = f(x_aug)
        tta_samples.append(x_aug)
        tta_grs.append(df_gr.drop(columns='subject').to_numpy().reshape(-1))

    # Pad and stacked
    pads = pad_sequences_torch(tta_samples, maxlen=pad_len,
                               padding='pre', truncating='pre')
    X = torch.FloatTensor(pads).to(device)  # (n_tta, C, T)
    grs = torch.FloatTensor(np.array(tta_grs).astype('float32')).to(device)
    
    # Multi-model + TTA prediction
    with torch.no_grad():
        ensemble_preds = []
        ensemble_preds_lgbm = []
        for i in range(len(models)):
            m = models[i]
            lgbm_m = lgbm_models[i]
            logits = m(X, grs)     # (n_tta, n_classes)
            probs  = torch.softmax(logits, dim=1)
            lgbm_probs = torch.tensor(lgbm_m.predict_proba(m.extract_features(X, grs).cpu()), device=torch.device(device))
            ensemble_preds.append(probs)
            ensemble_preds_lgbm.append(lgbm_probs)
        # Average by model → (n_tta, n_classes)
        avg_per_tta = torch.stack(ensemble_preds + ensemble_preds_lgbm).mean(0)
        # Re-average all TTA sample probabilities → (1, n_classes)
        final_prob = avg_per_tta.mean(0, keepdim=True)
        idx = int(final_prob.argmax(dim=1)[0].cpu().numpy())
    return final_prob.to('cpu').detach().numpy().copy()



from pathlib import Path

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame, n_tta=8) -> str:
    global gesture_classes
    if 'gesture_classes' not in globals() or gesture_classes is None:
        PRETRAINED_DIR = Path("/kaggle/input/radon_auto_update_model/pytorch/default/1")
        gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    # --- 1. 各モデルから予測確率を取得 ---
    prob_radon = predict_radon(sequence, demographics, n_tta=8)         # 形状: (1, クラス数)
    prob_mitsuki = predict_mitsuki(sequence, demographics)         # 形状: (1, クラス数)
    prob_miyairi_per_step = predict_miyairi(sequence, demographics)  # 形状: (シーケンス長, クラス数)

    # --- 2. Miyairiモデルのタイムスタンプ毎の予測を集約 ---
    # 全てのタイムスタンプの予測確率を平均し、シーケンス全体の単一の予測確率ベクトルを計算します。
    prob_miyairi_agg = prob_miyairi_per_step.mean(axis=0)  # 形状: (クラス数,)

    # --- 3. 全モデルの予測をアンサンブル（平均化） ---
    # RadonとMitsukiモデルの出力から不要な次元を削除し、形状を合わせる (1, クラス数) -> (クラス数,)
    avg_prob = (np.squeeze(prob_radon) + np.squeeze(prob_mitsuki) + prob_miyairi_agg) / 3

    # --- 4. 最終的な予測クラスを決定 ---
    # 平均化された確率が最も高いクラスのインデックスを取得
    final_pred_index = np.argmax(avg_prob)
    
    # 対応する単一のジェスチャー名を文字列として返す
    return gesture_classes[final_pred_index]

# Kaggle competition interface
import kaggle_evaluation.cmi_inference_server
import os

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )


"""def predict(sequence: pl.DataFrame, demographics: pl.DataFrame, n_tta=8):
    global gesture_classes
    if gesture_classes is None:
        gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)
    prob_radon = predict_radon(sequence, demographics, n_tta=8)
    prob_mitsuki = predict_mitsuki(sequence, demographics)
    prob_miyairi = predict_miyairi(sequence, demographics)
    prob_nomura = predict_nomura(sequence, demographics)
    final_prob = np.argmax((prob_radon+prob_mitsuki+prob_miyairi+prob_nomura)/4, axis=1)
    return str(gesture_classes[final_prob])

# Kaggle competition interface
import kaggle_evaluation.cmi_inference_server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )"""

