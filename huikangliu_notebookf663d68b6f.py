# -------------------- 必要的函数包 --------------------
import os, re, random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


# -------------------- 可复现性设置 --------------------
SEED          = 2025   # 随机种子（保证可复现：数据划分、初始化、shuffle）
def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -------------------- 读取数据 --------------------
INPUT_DIR = Path("/kaggle/input/lstm-transformer-01/data")
train_df = pd.read_csv(INPUT_DIR / "train.csv")
test_df  = pd.read_csv(INPUT_DIR / "test.csv")

assert {"text", "label"}.issubset(train_df.columns), "train.csv 需包含列：Text, Label"
assert {"Id", "text"}.issubset(test_df.columns),      "test.csv 需包含列：Id, Text"



# -------------------- 基础分词（无 torchtext） --------------------
def basic_english_tokenize(s: str):
    """
    简单英文分词：
      - 小写化
      - 非字母数字转空格
      - 压缩多余空白
    """
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip().split()

# -------------------- 构建词表（无 torchtext） --------------------
SPECIALS = ["<unk>", "<pad>"]

MIN_FREQ      = 1      # 词表最小词频（≥2 可降噪，但会增加 <unk>）
MAX_VOCAB     = None   # 词表最大大小（None 不限；如 50000 可控内存）
def build_vocab(texts, min_freq: int = 1, max_size: int | None = None):
    cnt = Counter()
    for t in texts:
        cnt.update(basic_english_tokenize(t))

    # 频率过滤
    items = [w for w, c in cnt.items() if c >= min_freq]
    # 先按频次降序，再按词字典序升序，保证稳定
    items.sort(key=lambda w: (-cnt[w], w))

    # 截断
    if max_size is not None:
        items = items[:max_size]

    stoi = {w: i for i, w in enumerate(SPECIALS + items)}
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos, cnt

stoi, itos, counter = build_vocab(train_df["text"], min_freq=MIN_FREQ, max_size=MAX_VOCAB)
UNK, PAD = stoi["<unk>"], stoi["<pad>"]
vocab_size = len(stoi)
print(f"Vocab size = {vocab_size} (min_freq={MIN_FREQ}, max={MAX_VOCAB})")

def encode(text: str, max_len: int = 256):
    ids = [stoi.get(tok, UNK) for tok in basic_english_tokenize(text)]
    if not ids:  # 防止空序列
        ids = [PAD]
    return torch.tensor(ids[:max_len], dtype=torch.long)


# -------------------- 数据集与打包 --------------------
class IMDBDataset(Dataset):
    """训练/验证/测试统一数据集；训练/验证返回 (x, y)，测试返回 (x, id)"""
    def __init__(self, df: pd.DataFrame, has_label: bool = True):
        self.df = df.reset_index(drop=True)
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        x = encode(row["text"], max_len=MAX_LEN)
        if self.has_label:
            y = int(row["label"])
            return x, y
        else:
            return x, int(row["Id"])

def collate_fn(batch):
    """将变长序列 padding 到批次内的最大长度；Label/Id 组为张量"""
    xs, ys = zip(*batch)
    xs = [x if len(x) > 0 else torch.tensor([PAD], dtype=torch.long) for x in xs]
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=PAD)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs_padded, ys




# ===================== 配置（学生主要改这里） =====================
EPOCHS        = 5      # 训练轮数
BATCH_SIZE    = 128    # 批大小（CPU 可用 64~256；显存小就调小）
LR            = 2e-3   # 学习率（1e-3~3e-3 常用）
WEIGHT_DECAY  = 1e-4   # L2 正则（0~1e-4 常用）
MAX_LEN       = 256    # 序列最大长度（截断；越大越慢、越小信息丢失）
EMBED_DIM     = 128    # 词向量维度（表达能力 vs 过拟合风险）
HIDDEN_SIZE   = 128    # LSTM 隐层维度（64/128/256 常用）
NUM_LAYERS    = 2      # LSTM 堆叠层数（>1 更深但更难训）
DROPOUT       = 0.20   # 全连接前 Dropout（防过拟合）
# ================================================================



# 分层划分训练/验证
tr_df, val_df = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["label"]
)

train_loader = DataLoader(IMDBDataset(tr_df, has_label=True),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
val_loader   = DataLoader(IMDBDataset(val_df, has_label=True),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
test_loader  = DataLoader(IMDBDataset(test_df, has_label=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)



# -------------------- 模型：BiLSTM + 线性分类 --------------------
class BiLSTM(nn.Module):
    """
    结构：
      - Embedding(vocab_size, EMBED_DIM, padding_idx=PAD)
      - BiLSTM(EMBED_DIM -> 2*HIDDEN_SIZE)
        * 取最后一层双向隐藏态 (h[-2], h[-1]) 拼接
      - Dropout(DROPOUT)
      - Linear(2*HIDDEN_SIZE -> 2)
    """
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD)
        self.rnn = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_SIZE,         # LSTM 隐层维度（64/128/256 常用）
            num_layers=NUM_LAYERS,           # LSTM 堆叠层数（>1 更深但更难训）
            batch_first=True,
            bidirectional=True,
        )
        self.dp = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, 2)

    def forward(self, x):
        e = self.emb(x)               # (B, T, E)
        o, (h, c) = self.rnn(e)       # h: (2*L, B, H)
        h_last = torch.cat((h[-2], h[-1]), dim=1)  # (B, 2H) 取最后一层双向隐藏态
        out = self.fc(self.dp(h_last))             # (B, 2)
        return out



model = BiLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

@torch.inference_mode()
def evaluate(loader) -> float:
    """返回准确率（Accuracy）"""
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total   += yb.size(0)
    return correct / total


# -------------------- 训练循环 --------------------
best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        # 可选：梯度裁剪，稳定训练（如需要）
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    val_acc = evaluate(val_loader)
    best_acc = max(best_acc, val_acc)
    print(f"Epoch {epoch}/{EPOCHS}  val_acc={val_acc:.4f}  (best={best_acc:.4f})")


# -------------------- 推理与导出提交 --------------------
ids, preds = [], []
model.eval()
with torch.no_grad():
    for xb, ib in test_loader:
        xb = xb.to(device)
        pr = model(xb).argmax(1).cpu().numpy().tolist()
        preds += pr
        ids   += ib.numpy().tolist()

sub = pd.DataFrame({"Id": ids, "Label": preds}).sort_values("Id").reset_index(drop=True)
sub.to_csv("submission.csv", index=False)
print("✅ saved submission.csv")




