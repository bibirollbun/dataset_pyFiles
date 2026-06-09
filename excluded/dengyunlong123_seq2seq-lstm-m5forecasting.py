# Cell 1：基础导入 & 配置
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 让结果稳一点
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# Cell 2：读取 M5 数据 & 选一条时间序列
# 只拿一个商品在一个店的销量来玩 seq2seq，不搞 4w 条那种自虐。

# 数据路径
DATA_DIR = "/kaggle/input/m5-forecasting-accuracy"

sales_path = os.path.join(DATA_DIR, "sales_train_validation.csv")
calendar_path = os.path.join(DATA_DIR, "calendar.csv")

sales_df = pd.read_csv(sales_path)
calendar_df = pd.read_csv(calendar_path)

print("sales_train_validation shape:", sales_df.shape)
print("calendar shape:", calendar_df.shape)

# 简单选第一行作为一个时间序列（某个商品在某个店的销量）
row = sales_df.iloc[0]

id_col = row["id"]
series_values = row.filter(like="d_").values.astype(np.float32)  # shape (1913,)

print("Chosen series id:", id_col)
print("Series length:", len(series_values))

# 画一下原始销量曲线
plt.figure(figsize=(12, 3))
plt.plot(series_values)
plt.title(f"Raw sales series: {id_col}")
plt.xlabel("Day index")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# 简单做一个 log1p 变换 + 标准化（避免 LSTM 喂太扁 / 太爆）
series_log = np.log1p(series_values)

mean = series_log.mean()
std = series_log.std() + 1e-6
series_scaled = (series_log - mean) / std

print("Scaled series mean/std:", series_scaled.mean(), series_scaled.std())



# Cell 3：构造 Seq2Seq 用的 Dataset
ENCODER_LEN = 90     # encoder 输入长度
DECODER_LEN = 28     # 预测 horizon（M5 也是 28 天）

class M5Seq2SeqDataset(Dataset):
    def __init__(self, series_1d, enc_len, dec_len):
        """
        series_1d: 已经缩放好的 1D numpy / list，长度 T
        每个样本：x: enc_len, y: dec_len
        """
        self.series = torch.tensor(series_1d, dtype=torch.float32)
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.max_idx = len(self.series) - enc_len - dec_len
        assert self.max_idx > 0, "时间序列太短，窗口塞不下"

    def __len__(self):
        return self.max_idx

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.enc_len]               # (enc_len,)
        y = self.series[idx + self.enc_len :
                        idx + self.enc_len + self.dec_len]      # (dec_len,)
        # 加一维 feature 维度，变成 (seq_len, 1) 方便喂 LSTM
        return x.unsqueeze(-1), y.unsqueeze(-1)


full_dataset = M5Seq2SeqDataset(series_scaled, ENCODER_LEN, DECODER_LEN)
print("Total samples:", len(full_dataset))

# 按时间顺序切分：前 80% 做训练，后 20% 做验证
n_total = len(full_dataset)
n_train = int(n_total * 0.8)

train_indices = list(range(n_train))
val_indices = list(range(n_train, n_total))

train_ds = torch.utils.data.Subset(full_dataset, train_indices)
val_ds = torch.utils.data.Subset(full_dataset, val_indices)

BATCH_SIZE = 64

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# 看一个 batch 形状
x_batch, y_batch = next(iter(train_loader))
print("x_batch:", x_batch.shape)  # (batch, enc_len, 1)
print("y_batch:", y_batch.shape)  # (batch, dec_len, 1)



# Cell 4：定义 Encoder / Decoder / Seq2Seq 模型
class Encoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, src):
        # src: (batch, enc_len, input_size)
        # LSTM 期望: (enc_len, batch, input_size)
        src = src.permute(1, 0, 2)
        outputs, (hidden, cell) = self.lstm(src)
        # 我们只用最终 hidden / cell 作为“压缩后的状态”
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, out_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, input_step, hidden, cell):
        # input_step: (batch, 1, input_size)
        input_step = input_step.permute(1, 0, 2)   # (1, batch, input_size)
        output, (hidden, cell) = self.lstm(input_step, (hidden, cell))
        pred = self.fc(output.squeeze(0))          # (batch, out_size)
        return pred.unsqueeze(1), hidden, cell     # (batch, 1, out_size)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch, enc_len, 1)
        trg: (batch, dec_len, 1)
        """
        batch_size = src.size(0)
        dec_len = trg.size(1)
        out_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, dec_len, out_size, device=self.device)

        hidden, cell = self.encoder(src)

        # 第一个 decoder 输入：使用真实的第一个目标值（经典 teacher forcing 设定）
        input_step = trg[:, 0:1, :]   # (batch, 1, 1)

        for t in range(dec_len):
            output, hidden, cell = self.decoder(input_step, hidden, cell)
            outputs[:, t : t+1, :] = output

            if t + 1 < dec_len:
                use_teacher = (torch.rand(1).item() < teacher_forcing_ratio)
                input_step = trg[:, t+1:t+2, :] if use_teacher else output

        return outputs

    def predict(self, src, dec_len):
        """
        纯推理：只给 encoder 序列，让 decoder 自回归预测 dec_len 步。
        src: (1, enc_len, 1)
        """
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(src)
            # 用最后一个 encoder 值作为起点
            last_value = src[:, -1:, :]          # (1, 1, 1)
            input_step = last_value

            outputs = []
            for _ in range(dec_len):
                output, hidden, cell = self.decoder(input_step, hidden, cell)
                outputs.append(output)           # list of (1, 1, 1)
                input_step = output              # 自回归

            outputs = torch.cat(outputs, dim=1)  # (1, dec_len, 1)
        return outputs
        

encoder = Encoder(input_size=1, hidden_size=64, num_layers=1)
decoder = Decoder(input_size=1, hidden_size=64, num_layers=1, out_size=1)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(model)



# Cell 5：训练循环（兼容新版本 PyTorch）
def train_one_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x, y, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = criterion(outputs.squeeze(-1), y.squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x, y, teacher_forcing_ratio=0.0)  # 验证不 teacher forcing
            loss = criterion(outputs.squeeze(-1), y.squeeze(-1))
            running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


EPOCHS = 15

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                 teacher_forcing_ratio=0.5)
    val_loss = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")



# Cell 6：单次预测 & 可视化
# 用最后一段历史做预测
last_history = series_scaled[-ENCODER_LEN:]         # (enc_len,)
last_history_tensor = torch.tensor(last_history, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
# shape: (1, enc_len, 1)

pred_scaled = model.predict(last_history_tensor, DECODER_LEN)   # (1, dec_len, 1)
pred_scaled = pred_scaled.squeeze().cpu().numpy()

# 反标准化 + 反 log1p
pred_log = pred_scaled * std + mean
pred_values = np.expm1(pred_log)

true_future = series_values[-DECODER_LEN:]   # 原始尺度上的真实销量

# 画图对比
plt.figure(figsize=(10, 4))

plt.plot(range(DECODER_LEN), true_future, label="True future sales")
plt.plot(range(DECODER_LEN), pred_values, label="Predicted sales")

plt.title("Seq2Seq forecast: next 28 days")
plt.xlabel("Forecast day")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()

print("First 5 true values:    ", true_future[:5])
print("First 5 predicted vals: ", pred_values[:5])


