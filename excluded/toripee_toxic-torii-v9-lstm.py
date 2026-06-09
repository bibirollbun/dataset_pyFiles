import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import gc

# ==========================================
# ★設定エリア (LSTM用)
# ==========================================
class Config:
    MAX_LEN = 200        # 文章の長さ
    MAX_FEATURES = 20000 # 頻出2万語を使用
    EMBED_DIM = 128      # 単語ベクトルのサイズ
    HIDDEN_DIM = 64      # LSTMの記憶容量
    BATCH_SIZE = 64      # バッチサイズ
    EPOCHS = 2           # 学習回数
    LR = 0.001           # 学習率

# ==========================================
# 1. データ読み込み
# ==========================================
print("【LSTM】データを読み込んでいます...")
train_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
sub = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

# 欠損値埋め
train_df['comment_text'] = train_df['comment_text'].fillna("fillna")
test_df['comment_text'] = test_df['comment_text'].fillna("fillna")

# ==========================================
# 2. 前処理（URL削除など）の追加
# ==========================================
print("前処理（URL削除など）を実行中...")

def clean_for_transformer(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # URL
    text = re.sub(r'<.*?>', '', text)                 # HTMLタグ
    text = re.sub(r'\n+', ' ', text)                  # 改行
    text = re.sub(r'\t+', ' ', text)                  # タブ
    text = re.sub(r' {2,}', ' ', text).strip()        # 連続スペース
    return text

# データフレーム全体に適用
train_df['comment_text'] = train_df['comment_text'].apply(clean_for_transformer)
test_df['comment_text'] = test_df['comment_text'].apply(clean_for_transformer)

# ==========================================
# 3. 辞書作成 & ベクトル化
# ==========================================
def simple_clean(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower().split()

print("辞書を作成中...")
all_words = []
for text in train_df['comment_text'][:50000]:
    all_words.extend(simple_clean(text))

word_counts = Counter(all_words)
vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.most_common(Config.MAX_FEATURES))}
vocab_size = len(vocab) + 1

def text_to_sequence(text, max_len):
    words = simple_clean(text)
    seq = [vocab.get(w, 0) for w in words]
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

print("テキストをベクトル化中...")
X = np.array([text_to_sequence(t, Config.MAX_LEN) for t in train_df['comment_text']])
X_test = np.array([text_to_sequence(t, Config.MAX_LEN) for t in test_df['comment_text']])
y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# ==========================================
# 4. PyTorch Dataset & Model (Bi-LSTM)
# ==========================================
class ToxicDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float) if y is not None else None
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]
    
    def __len__(self):
        return len(self.x)

train_loader = DataLoader(ToxicDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ToxicDataset(X_val, y_val), batch_size=Config.BATCH_SIZE)
test_loader = DataLoader(ToxicDataset(X_test), batch_size=Config.BATCH_SIZE)

# LSTMモデル定義
class BiLSTM_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, Config.EMBED_DIM, padding_idx=0)
        # Bidirectional LSTM
        self.lstm = nn.LSTM(Config.EMBED_DIM, Config.HIDDEN_DIM, num_layers=2, 
                            bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(Config.HIDDEN_DIM * 2, 6)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # Global Max Pooling
        out, _ = torch.max(lstm_out, dim=1)
        out = self.fc(out)
        return out

# ==========================================
# 5. 学習実行
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTM_Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
criterion = nn.BCEWithLogitsLoss()

print(f"【LSTM】学習を開始します (Device: {device})")

for epoch in range(Config.EPOCHS):
    model.train()
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1} / {Config.EPOCHS} 完了")

# ==========================================
# 6. 予測と提出
# ==========================================
print("予測を実行中...")
model.eval()
preds = []

with torch.no_grad():
    for x_batch in test_loader:
        x_batch = x_batch.to(device)
        out = model(x_batch)
        preds.append(torch.sigmoid(out).cpu().numpy())

final_preds = np.concatenate(preds)
sub[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = final_preds

# 保存
sub.to_csv('submission_lstm_cleaned.csv', index=False)
print("完了！ 'submission_lstm_cleaned.csv' を保存しました。")




