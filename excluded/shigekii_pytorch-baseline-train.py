import pandas as pd
import numpy as np
from tqdm import tqdm

folder_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data"

train_demo = pd.read_csv(f"{folder_path}/train_demographics.csv")
train = pd.read_csv(f"{folder_path}/train.csv")
test_demo = pd.read_csv(f"{folder_path}/test_demographics.csv")
test = pd.read_csv(f"{folder_path}/test.csv")


test.head()


train.head()


train_demo.head()


print(np.shape(train_demo), np.shape(train))
print(np.shape(test_demo), np.shape(test))


print([col for col in train.columns if col not in test.columns])


import torch
from torch.utils.data import Dataset, DataLoader

class SubjectDataset(Dataset):
    def __init__(self, data_demo, data, have_labels=True):
        self.have_labels = have_labels
        self.unique_seequence_ids = sorted(data["sequence_id"].unique())
        self.sequence_id_to_subject = data.groupby("sequence_id")["subject"].first().to_dict()
        
        numerical_demo_cols = data_demo.select_dtypes(include=np.number).columns.tolist()
        if 'subject' in numerical_demo_cols:
            numerical_demo_cols.remove('subject')
        
        self.subject_to_demographics = {
            subject: torch.tensor(row[numerical_demo_cols].values.astype(np.float32))
            for subject, row in data_demo.set_index('subject').iterrows()
        }

        self.labels = None
        
        if have_labels:
            all_unique_gestures = sorted(data["gesture"].unique())
            self.gesture_to_id = {gesture: i for i, gesture in enumerate(all_unique_gestures)}
            sequence_labels_map_str = data.groupby("sequence_id")["gesture"].first().to_dict()
            self.labels = torch.tensor(
                [self.gesture_to_id[sequence_labels_map_str[sid]] for sid in self.unique_seequence_ids], 
                dtype=torch.long
            )

        cols_to_drop = ["row_id", "subject", "sequence_id", "sequence_type", "orientation", "behavior", "phase", "gesture"]
        sequence_feature_cols = [col for col in data.columns if col not in cols_to_drop]
        
        self.preprocessed_sequences = {
            seq_id: torch.tensor(
                data[data["sequence_id"] == seq_id][sequence_feature_cols].values.astype(np.float32)
            ).T
            for seq_id in tqdm(self.unique_seequence_ids)
        }

    def __len__(self):
        return len(self.unique_seequence_ids)

    def __getitem__(self, idx):
        sequence_id = self.unique_seequence_ids[idx]
        
        subject_data_tensor = self.subject_to_demographics[self.sequence_id_to_subject[sequence_id]]
        sequence_data_tensor = self.preprocessed_sequences[sequence_id]

        if self.have_labels:
            return subject_data_tensor, sequence_data_tensor, self.labels[idx]
        else:
            return sequence_id, subject_data_tensor, sequence_data_tensor


train_dataset = SubjectDataset(train_demo, train)


test_dataset = SubjectDataset(test_demo, test, have_labels=False)


seed = 42
torch.manual_seed(seed)


# trainとvalの分割
from torch.utils.data import random_split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


import torch
import torch.nn.functional as F # F.pad のために必要

# --- custom_collate_fn の定義 ---
def custom_collate_fn(batch):
    # バッチ内の最初の要素の長さで、ラベルがあるか（3つ要素）ないか（2つ要素）を判別
    has_labels = (len(batch[0]) == 3) # (demo_tensor, sequence_tensor, label_tensor) なら3

    if has_labels:
        # ラベルがある場合、(デモデータ, シーケンスデータ, ラベル) のタプルを受け取る
        demo_list, sequence_list, label_list = zip(*batch)
    else:
        # ラベルがない場合、(sequence_id, デモデータ, シーケンスデータ) のタプルを受け取る
        sequence_id_list, demo_list, sequence_list = zip(*batch)

    # 1. デモグラフィックデータは固定長と仮定し、そのままスタックしてバッチを作成
    demographics_batch = torch.stack(demo_list)

    # 2. シーケンスデータのパディング
    # sequence_listは既に(C, L)のテンソルリストになっていることを前提
    
    # バッチ内で最も長いシーケンスの長さを取得
    max_len = max(s.shape[1] for s in sequence_list) # s.shape[1] がシーケンス長 (L)

    # 各シーケンスを最長シーケンスの長さにパディング
    padded_sequences = []
    for s in sequence_list:
        # s.shape[1]は現在のシーケンス長、max_lenはバッチ内の最長シーケンス長
        padding_needed = max_len - s.shape[1] 
        if padding_needed > 0:
            # torch.nn.functional.pad を使って (C, L_new) にパディング
            # pad=(0, padding_needed) は「最後の次元（ここではL）の右側にpadding_neededだけゼロを追加」
            padded_s = F.pad(s, (0, padding_needed))
        else:
            padded_s = s # パディングが不要な場合はそのまま
        padded_sequences.append(padded_s)
    
    # パディングされたテンソルを結合し、1つのバッチテンソルにする
    sequences_batch = torch.stack(padded_sequences)


    # 3. ラベルの処理
    if has_labels:
        # ラベルは全て同じ形状なので、そのままスタック
        labels_batch = torch.stack(label_list)
        return demographics_batch, sequences_batch, labels_batch
    else:
        # ラベルがない場合、sequence_idもバッチとして返す
        # sequence_id_listは文字列なのでstackできないことに注意（リストのまま返す）
        return sequence_id_list, demographics_batch, sequences_batch


BATCH_SIZE = 32
NUM_WORKERS = 0

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=NUM_WORKERS)


import torch.nn as nn
import torch.nn.functional as F

class HybridGestureClassifier(nn.Module):
    def __init__(self, num_sequence_features: int, sequence_length: int, num_subject_features: int, num_classes: int = 18):
        super().__init__()

        # CNNブロックをさらに深く、広くする
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_sequence_features, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128), # BatchNormを追加
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256), # BatchNormを追加
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1), # 新しい層: 256 -> 512
            nn.ReLU(),
            nn.BatchNorm1d(512), # BatchNormを追加
            nn.MaxPool1d(kernel_size=2) # 最終的なシーケンス長は元の1/8になる
        )
        
        # MaxPool1dが3回になるので、シーケンス長は元の1/8になる
        self.cnn_output_length = sequence_length // 8 
        self.lstm_input_size = 512 # CNNの最終出力チャンネル数に合わせる

        # LSTMをさらに深く、広くする
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size, 
            hidden_size=512, # 隠れ層のサイズをさらに増やす: 256 -> 512
            num_layers=4,    # 層数をさらに増やす: 3 -> 4
            batch_first=True, 
            dropout=0.4      # ドロップアウト率も調整
        )
        
        # 被験者データ処理器を深くする
        self.subject_fc = nn.Sequential(
            nn.Linear(num_subject_features, 128), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256), # 新しい層: 128 -> 256
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 結合される特徴量の次元が増える
        combined_feature_dim = 512 + 256 # LSTMのhidden_size + subject_fcの出力 
        
        # 分類器を深く、広くする
        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_dim, 512), # 入力を増やす
            nn.ReLU(),
            nn.Dropout(0.5), # ドロップアウト率を調整
            nn.Linear(512, 256), # 新しい層: 512 -> 256
            nn.ReLU(),
            nn.Dropout(0.5), # ドロップアウト率を調整
            nn.Linear(256, num_classes) # 最終出力
        )

    def forward(self, subject_data, sequence_data):
        cnn_out = self.conv_block(sequence_data)
        lstm_input = cnn_out.transpose(1, 2) 
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        sequence_features = h_n[-1] 

        subject_features = self.subject_fc(subject_data)

        combined_features = torch.cat((sequence_features, subject_features), dim=1)

        logits = self.classifier(combined_features)
        return logits


import torch.optim as optim

DEMO_INPUT_DIM = train_dataset[0][0].shape[0]
SEQ_INPUT_CHANNELS = train_dataset[0][1].shape[0]
NUM_CLASSES = 18

# モデルのインスタンスを作成
model = HybridGestureClassifier(
    num_sequence_features=SEQ_INPUT_CHANNELS,
    sequence_length=300,
    num_subject_features=DEMO_INPUT_DIM,
    num_classes=NUM_CLASSES
)

# 損失関数の定義 (多クラス分類のためCrossEntropyLoss)
criterion = nn.CrossEntropyLoss()

# 最適化アルゴリズムの定義 (Adamオプティマイザ)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# モデルをGPUに移動（利用可能な場合）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)


x,y,z = next(iter(train_loader))
print(f"x: {x.shape}, y: {y.shape}, z: {z.shape if z is not None else 'No labels'}")


x[torch.isnan(x)] = 0.0  # NaNを0で埋める
y[torch.isnan(y)] = 0.0  # NaNを0で埋める


has_nan_demo = torch.isnan(x).any()
has_nan_seq = torch.isnan(y).any()
print(f"demo_data に NaN が含まれているか: {has_nan_demo}")
print(f"seq_data に NaN が含まれているか: {has_nan_seq}")


import torch
# モデル、DataLoader、criterion、optimizer、deviceは前のセルで定義済みと仮定

# --- 訓練ループ ---
NUM_EPOCHS = 200 # 訓練エポック数

print("\n--- 訓練開始 ---")
best_val_accuracy = 0.0 # 最も良い検証精度を記録
# モデルの重みを保存するパス（必要に応じて変更）
model_save_path = 'best_gesture_classifier.pth' 

for epoch in range(NUM_EPOCHS):
    # 訓練フェーズ
    model.train() # モデルを訓練モードに設定
    train_loss = 0
    train_correct = 0
    train_total = 0

    for demo_data, seq_data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Train"):
        # --- ここから追加・修正 ---
        # NaN/Infのチェックとゼロ補完
        if torch.isnan(seq_data).any() or torch.isinf(seq_data).any():
            seq_data[torch.isnan(seq_data)] = 0.0 # NaNを0で埋める
            seq_data[torch.isinf(seq_data)] = 0.0 # Infを0で埋める
        # --- ここまで追加・修正 ---
        
        demo_data, seq_data, labels = demo_data.to(device), seq_data.to(device), labels.to(device)

        optimizer.zero_grad() # 勾配をゼロクリア
        outputs = model(demo_data, seq_data) # 順伝播
        loss = criterion(outputs, labels) # 損失計算
        loss.backward() # 逆伝播
        optimizer.step() # パラメータ更新

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1) # 最も高い確率を持つクラスIDを取得
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item() # 正しく予測された数
    
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = train_correct / train_total

    # 検証フェーズ
    model.eval() # モデルを評価モードに設定 (Dropoutなどが無効になる)
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad(): # 勾配計算を無効化
        for demo_data, seq_data, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Val"):
            if torch.isnan(seq_data).any() or torch.isinf(seq_data).any():
                seq_data[torch.isnan(seq_data)] = 0.0 # NaNを0で埋める
                seq_data[torch.isinf(seq_data)] = 0.0 # Infを0で埋める
                
            demo_data, seq_data, labels = demo_data.to(device), seq_data.to(device), labels.to(device)

            outputs = model(demo_data, seq_data)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # 最も良い検証精度を達成したモデルの重みを保存
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), model_save_path) # モデルの重み（state_dict）を保存
        print(f"  --> Best model saved with Val Acc: {best_val_accuracy:.4f}")

print("--- 訓練終了 ---")




