# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# A. LSTM+Attention 模型训练（改进：多轮训练、Dropout、LR Scheduler、EarlyStopping；已确保序列为 list）
import json, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) 读入并映射
with open('/kaggle/input/clean-student-logs/clean_student_logs.json','r') as f:
    raw = json.load(f)
all_qids = sorted({int(e['question_id']) for e in raw})
qid2idx = {qid:i+1 for i,qid in enumerate(all_qids)}
num_questions = len(qid2idx)

# 2) 按 student 构建 list 序列
seqs = {}
for e in raw:
    sid = e['student_id']
    q = qid2idx[int(e['question_id'])]
    c = int(e['is_correct'])
    seqs.setdefault(sid, []).append((q,c))

data = []
for seq in seqs.values():
    if len(seq) < 2: continue
    qs = [q for q,_ in seq]
    cs = [c for _,c in seq]
    data.append((qs[:-1], cs[:-1], cs[1:]))

train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

# 3) Dataset
class SeqDataset(Dataset):
    def __init__(self, data, max_len=100):
        self.data, self.max_len = data, max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        qs, cs, tar = self.data[i]
        L = len(qs)
        pad_q = [0]*(self.max_len-L) + qs[-self.max_len:]
        pad_c = [0]*(self.max_len-L) + cs[-self.max_len:]
        pad_t = [0]*(self.max_len-L) + tar[-self.max_len:]
        return torch.LongTensor(pad_q), torch.FloatTensor(pad_c), torch.FloatTensor(pad_t)

batch_size = 128
train_loader = DataLoader(SeqDataset(train_data), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(SeqDataset(val_data),   batch_size=batch_size)

# 4) 模型 + EarlyStopping
class LSTMAttnModel(nn.Module):
    def __init__(self, num_q, emb_dim=64, hid_dim=128, dp=0.2):
        super().__init__()
        self.emb     = nn.Embedding(num_q+1, emb_dim, padding_idx=0)
        self.lstm    = nn.LSTM(emb_dim+1, hid_dim, batch_first=True, dropout=dp)
        self.attn    = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dp)
        self.fc      = nn.Linear(hid_dim,1)
    def forward(self, q_seq, c_seq):
        emb = self.emb(q_seq)
        x   = torch.cat([emb, c_seq.unsqueeze(-1)], dim=-1)
        out,(h,_) = self.lstm(x)
        h_last    = h[-1]
        scores    = torch.bmm(out, self.attn(h_last).unsqueeze(-1)).squeeze(-1)
        α         = torch.softmax(scores, dim=1)
        ctx       = (out * α.unsqueeze(-1)).sum(dim=1)
        ctx       = self.dropout(ctx)
        return torch.sigmoid(self.fc(ctx).squeeze(-1))

class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4, path='best_lstm.pth'):
        self.patience,self.delta,self.path = patience,delta,path
        self.best_loss,self.counter = None,0
        self.early_stop = False
    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss, self.counter = val_loss, 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

model = LSTMAttnModel(num_questions).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2, verbose=True)
crit  = nn.BCELoss()
stopper = EarlyStopping(patience=5)

# 5) 训练循环
for epoch in range(1, 51):
    model.train()
    tr_loss = 0
    for q,c,t in train_loader:
        q,c,t = q.to(device), c.to(device), t.to(device)
        pred = model(q,c)
        loss = crit(pred, t[:,-1])
        opt.zero_grad(); loss.backward(); opt.step()
        tr_loss += loss.item()
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for q,c,t in val_loader:
            q,c,t = q.to(device), c.to(device), t.to(device)
            val_loss += crit(model(q,c), t[:,-1]).item()
    tr_loss /= len(train_loader)
    val_loss /= len(val_loader)
    sched.step(val_loss)
    stopper(val_loss, model)
    print(f"LSTM Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}")
    if stopper.early_stop:
        print("Early stopping.")
        break

# 加载最佳
model.load_state_dict(torch.load('best_lstm.pth'))


# B. DKT 模型训练（同样改进；已确保序列为 list）
import pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', nrows=200_000)
df = df[df['content_type_id']==0]
qs_unique = df['content_id'].unique()
qid2idx2 = {q:i+1 for i,q in enumerate(qs_unique)}
num_q2 = len(qid2idx2)

grouped = df.groupby('user_id').apply(lambda d: list(zip(d['content_id'], d['answered_correctly'])))
data2 = []
for seq in grouped:
    if len(seq) < 2: continue
    qs = [qid2idx2[q] for q,_ in seq]
    cs = [c for _,c in seq]
    data2.append((qs[:-1], cs[:-1], cs[1:]))

train2, val2 = train_test_split(data2, test_size=0.1, random_state=42)

class SeqDataset2(Dataset):
    def __init__(self, data, max_len=100):
        self.data, self.max_len = data, max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        qs, cs, tar = self.data[i]
        L = len(qs)
        pad_q = [0]*(self.max_len-L) + qs[-self.max_len:]
        pad_c = [0]*(self.max_len-L) + cs[-self.max_len:]
        pad_t = [0]*(self.max_len-L) + tar[-self.max_len:]
        return torch.LongTensor(pad_q), torch.FloatTensor(pad_c), torch.FloatTensor(pad_t)

batch_size = 128
train_loader2 = DataLoader(SeqDataset2(train2), batch_size=batch_size, shuffle=True)
val_loader2   = DataLoader(SeqDataset2(val2),   batch_size=batch_size)

class DKTModel(nn.Module):
    def __init__(self, num_q, emb_dim=64, hid_dim=128, dp=0.2):
        super().__init__()
        self.emb     = nn.Embedding(num_q+1, emb_dim, padding_idx=0)
        self.lstm    = nn.LSTM(emb_dim+1, hid_dim, batch_first=True, dropout=dp)
        self.dropout = nn.Dropout(dp)
        self.fc      = nn.Linear(hid_dim,1)
    def forward(self, q_seq, c_seq):
        emb = self.emb(q_seq)
        x   = torch.cat([emb, c_seq.unsqueeze(-1)], dim=-1)
        out,(h,_) = self.lstm(x)
        h_last    = self.dropout(h[-1])
        return torch.sigmoid(self.fc(h_last).squeeze(-1))

class EarlyStopping2:
    def __init__(self, patience=5, delta=1e-4, path='best_dkt.pth'):
        self.patience,self.delta,self.path = patience,delta,path
        self.best_loss,self.counter = None,0
        self.early_stop = False
    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss, self.counter = val_loss,0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

model2   = DKTModel(num_q2).to(device)
opt2     = torch.optim.Adam(model2.parameters(), lr=1e-3)
sched2   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode='min', factor=0.5, patience=2, verbose=True)
crit2    = nn.BCELoss()
stopper2 = EarlyStopping2(patience=5)

for epoch in range(1, 51):
    model2.train()
    trl=0
    for q,c,t in train_loader2:
        q,c,t = q.to(device),c.to(device),t.to(device)
        loss = crit2(model2(q,c), t[:,-1])
        opt2.zero_grad(); loss.backward(); opt2.step()
        trl += loss.item()
    model2.eval()
    vl=0
    with torch.no_grad():
        for q,c,t in val_loader2:
            q,c,t = q.to(device),c.to(device),t.to(device)
            vl += crit2(model2(q,c), t[:,-1]).item()
    trl /= len(train_loader2); vl /= len(val_loader2)
    sched2.step(vl)
    stopper2(vl, model2)
    print(f"DKT Epoch {epoch}: train_loss={trl:.4f}, val_loss={vl:.4f}")
    if stopper2.early_stop:
        print("Early stopping.")
        break

model2.load_state_dict(torch.load('best_dkt.pth'))


# C. Advanced Ensemble (Hierarchical Stacking + Mixture-of-Experts + Temporal Features + EarlyStopping)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128

# ——— Helpers: train_fn & predict_fn ———
def train_fn(model, loader, optimizer, criterion, epochs=1):
    model.train()
    for _ in range(epochs):
        for q_seq, c_seq, tar in loader:
            q_seq, c_seq, tar = q_seq.to(device), c_seq.to(device), tar.to(device)
            loss = criterion(model(q_seq, c_seq), tar[:, -1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def predict_fn(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for q_seq, c_seq, _ in loader:
            q_seq, c_seq = q_seq.to(device), c_seq.to(device)
            preds.append(model(q_seq, c_seq).cpu().numpy())
    return np.concatenate(preds)

# ——— Assume train_data, val_data, SeqDataset, LSTMAttnModel, DKTModel, num_questions already defined ———

# 1) Collect 10-fold OOF & test preds for Base Models A/B
oof_A, oof_B = np.zeros(len(train_data)), np.zeros(len(train_data))
test_preds_A, test_preds_B = [], []
test_loader = DataLoader(SeqDataset(val_data), batch_size=batch_size)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for fold, (tr_idx, va_idx) in enumerate(kf.split(train_data), 1):
    print(f"Fold {fold}/10")
    tr_sub = [train_data[i] for i in tr_idx]
    va_sub = [train_data[i] for i in va_idx]
    tr_loader = DataLoader(SeqDataset(tr_sub), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(SeqDataset(va_sub), batch_size=batch_size)

    # Model A
    mA  = LSTMAttnModel(num_questions).to(device)
    optA = torch.optim.Adam(mA.parameters(), lr=1e-3)
    train_fn(mA, tr_loader, optA, nn.BCELoss(), epochs=1)
    oof_A[va_idx] = predict_fn(mA, va_loader)
    test_preds_A.append(predict_fn(mA, test_loader))

    # Model B (DKT on same qid mapping)
    mB  = DKTModel(num_questions).to(device)
    optB = torch.optim.Adam(mB.parameters(), lr=1e-3)
    train_fn(mB, tr_loader, optB, nn.BCELoss(), epochs=1)
    oof_B[va_idx] = predict_fn(mB, va_loader)
    test_preds_B.append(predict_fn(mB, test_loader))

# average test preds
test_A = np.mean(np.stack(test_preds_A, axis=1), axis=1)
test_B = np.mean(np.stack(test_preds_B, axis=1), axis=1)

# 2) Temporal features
rates      = np.array([np.mean(cs)      for _,cs,_ in train_data])
lengths    = np.array([len(cs)         for _,cs,_ in train_data])
rates_test = np.array([np.mean(cs)      for _,cs,_ in val_data])
lens_test  = np.array([len(cs)         for _,cs,_ in val_data])

# 3) GatingNet definition
class GatingNet(nn.Module):
    def __init__(self, in_dim, hid=16, dp=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(hid, 2)
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)

# 4) Prepare gating dataset
X_meta = np.vstack([oof_A, oof_B, rates, lengths]).T
y_meta = np.array([tar[-1] for _,_,tar in train_data])

X_tr, X_val, y_tr, y_val = train_test_split(X_meta, y_meta, test_size=0.2, random_state=42)

class MetaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

gd_train = DataLoader(MetaDataset(X_tr, y_tr), batch_size=64, shuffle=True)
gd_val   = DataLoader(MetaDataset(X_val, y_val), batch_size=64)

# 5) Train gating with EarlyStopping
gating = GatingNet(in_dim=4).to(device)
opt_g  = torch.optim.Adam(gating.parameters(), lr=1e-3)
sch_g  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode='min', factor=0.5, patience=2, verbose=True)
crit   = nn.BCELoss()

class EarlyStop:
    def __init__(self, patience=5, delta=1e-4, path='best_gating.pth'):
        self.patience, self.delta, self.path = patience, delta, path
        self.best, self.count = None, 0
        self.stop = False
    def __call__(self, loss, model):
        if self.best is None or loss < self.best - self.delta:
            self.best, self.count = loss, 0
            torch.save(model.state_dict(), self.path)
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True

stopper = EarlyStop(patience=5)

for epoch in range(1, 51):
    gating.train()
    t_loss = 0
    for xb, yb in gd_train:
        xb, yb = xb.to(device), yb.to(device)
        w = gating(xb)
        pred = w[:,0]*xb[:,0] + w[:,1]*xb[:,1]
        loss = crit(pred, yb)
        opt_g.zero_grad(); loss.backward(); opt_g.step()
        t_loss += loss.item()
    gating.eval()
    v_loss = 0
    with torch.no_grad():
        for xb, yb in gd_val:
            xb, yb = xb.to(device), yb.to(device)
            w = gating(xb)
            pred = w[:,0]*xb[:,0] + w[:,1]*xb[:,1]
            v_loss += crit(pred, yb).item()
    t_loss /= len(gd_train); v_loss /= len(gd_val)
    sch_g.step(v_loss)
    stopper(v_loss, gating)
    print(f"Gating Epoch {epoch}: train={t_loss:.4f}, val={v_loss:.4f}")
    if stopper.stop:
        print("Early stopping on gating")
        break

gating.load_state_dict(torch.load('best_gating.pth'))

# 6) Final MoE prediction on val set
X_test_meta = torch.tensor(
    np.vstack([test_A, test_B, rates_test, lens_test]).T,
    dtype=torch.float32, device=device
)
with torch.no_grad():
    w_test = gating(X_test_meta)
    final_pred = (w_test[:,0]*X_test_meta[:,0] +
                  w_test[:,1]*X_test_meta[:,1]).cpu().numpy()

y_true = np.array([tar[-1] for _,_,tar in val_data])
print("Final MoE AUC:", roc_auc_score(y_true, final_pred))


# D. 用 “2012data” 做纯 Hold-out 测试（修复 dtype mismatch）

import os
import pandas as pd
import numpy as np
import torch
import zipfile
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss

# 1) 定位并加载 “2012data” 文件
base   = '/kaggle/input'
folder = [d for d in os.listdir(base) if '2012data' in d][0]
file0  = os.listdir(os.path.join(base, folder))[0]
path0  = os.path.join(base, folder, file0)
# 如果是 ZIP 就解压
if zipfile.is_zipfile(path0):
    with zipfile.ZipFile(path0, 'r') as z:
        z.extractall('/kaggle/working/2012data')
    csvs = [f for f in os.listdir('/kaggle/working/2012data') if f.lower().endswith('.csv')]
    assert csvs, "No CSV inside ZIP"
    path = os.path.join('/kaggle/working/2012data', csvs[0])
else:
    path = path0

print("Loading hold-out data from:", path)
df = pd.read_csv(path)
print("Total rows in hold-out:", len(df))

# 2) 构造序列 (qs, cs, tar)
seqs = {}
for r in df.itertuples():
    seqs.setdefault(r.user_id, []).append((r.problem_id, int(r.correct)))

holdout = []
for logs in seqs.values():
    if len(logs) < 2:
        continue
    qs, cs = zip(*logs)
    holdout.append((list(qs[:-1]), list(cs[:-1]), list(cs[1:])))
print("Hold-out sequences:", len(holdout))

# 3) Dataset：unknown qids → 0
class HoldDataset(Dataset):
    def __init__(self, data, max_len=100):
        self.data, self.max_len = data, max_len
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, i):
        qs, cs, tar = self.data[i]
        L = len(qs)
        pad_q = [0]*self.max_len
        pad_c = [0]*(self.max_len - L) + cs[-self.max_len:]
        pad_t = [0]*(self.max_len - L) + tar[-self.max_len:]
        return (
            torch.LongTensor(pad_q),
            torch.FloatTensor(pad_c),
            torch.FloatTensor(pad_t)
        )

hold_loader = DataLoader(HoldDataset(holdout), batch_size=128, shuffle=False)

# 4) 动态恢复和加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LSTM
lstm_ckpt = torch.load('best_lstm.pth')
vocab_lstm = lstm_ckpt['emb.weight'].shape[0] - 1
model_lstm = LSTMAttnModel(vocab_lstm).to(device)
model_lstm.load_state_dict(lstm_ckpt)
model_lstm.eval()

# DKT
dkt_ckpt = torch.load('best_dkt.pth')
vocab_dkt = dkt_ckpt['emb.weight'].shape[0] - 1
model_dkt = DKTModel(vocab_dkt).to(device)
model_dkt.load_state_dict(dkt_ckpt)
model_dkt.eval()

# Gating
gating_ckpt = torch.load('best_gating.pth')
gating = GatingNet(in_dim=4).to(device)
gating.load_state_dict(gating_ckpt)
gating.eval()

# 5) 时序特征
rates   = np.array([np.mean(cs) for _, cs, _ in holdout], dtype=np.float32)
lengths = np.array([len(cs)    for _, cs, _ in holdout], dtype=np.float32)

# 6) 推断并收集
all_preds = []
idx = 0
with torch.no_grad():
    for q, c, _ in hold_loader:
        q, c = q.to(device), c.to(device)
        pA = model_lstm(q, c)
        pB = model_dkt (q, c)
        b  = pA.size(0)
        r  = torch.tensor(rates[idx:idx+b],   dtype=torch.float32, device=device).unsqueeze(1)
        l  = torch.tensor(lengths[idx:idx+b], dtype=torch.float32, device=device).unsqueeze(1)
        meta_in = torch.cat([pA.unsqueeze(1), pB.unsqueeze(1), r, l], dim=1)
        w       = gating(meta_in)
        pred    = w[:,0]*pA + w[:,1]*pB
        all_preds.append(pred.cpu().numpy())
        idx += b

y_true = [tar[-1] for _,_,tar in holdout]
y_pred = np.concatenate(all_preds)

# 7) 评估
print("Hold-out AUC:     ", roc_auc_score(y_true, y_pred))
print("Hold-out LogLoss:", log_loss(y_true, y_pred))


# E. 在 SkillBuilder 2012–2013 数据集上微调 LSTM+Attention 和 DKT（跳过大小不匹配的 embedding 层导入）

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1) 加载 SkillBuilder 2012–2013 数据
csv_path = '/kaggle/input/2012data/2012-2013-data-with-predictions-4-final.csv'
df = pd.read_csv(csv_path)

# 2) 构造新的 qid2idx & 序列，留下 90% 训练，10% 验证
qids2 = df['problem_id'].unique()
qid2idx2 = {q:i+1 for i,q in enumerate(qids2)}
num_q2   = len(qid2idx2)

seqs2 = {}
for r in df.itertuples():
    seqs2.setdefault(r.user_id, []).append((qid2idx2[r.problem_id], int(r.correct)))

data2 = []
for logs in seqs2.values():
    if len(logs) < 2: continue
    qs, cs = zip(*logs)
    data2.append((list(qs[:-1]), list(cs[:-1]), list(cs[1:])))

train2, val2 = train_test_split(data2, test_size=0.1, random_state=42, shuffle=True)

# 3) Dataset & DataLoader
class SeqDataset2(Dataset):
    def __init__(self, data, max_len=100):
        self.data, self.max_len = data, max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        qs, cs, tar = self.data[i]
        L = len(qs)
        pad_q = [0]*(self.max_len-L) + qs[-self.max_len:]
        pad_c = [0]*(self.max_len-L) + cs[-self.max_len:]
        pad_t = [0]*(self.max_len-L) + tar[-self.max_len:]
        return (
            torch.LongTensor(pad_q),
            torch.FloatTensor(pad_c),
            torch.FloatTensor(pad_t)
        )

batch_size = 128
train_loader2 = DataLoader(SeqDataset2(train2), batch_size=batch_size, shuffle=True)
val_loader2   = DataLoader(SeqDataset2(val2),   batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()

# 4) 构建 EarlyStopping
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4, path='checkpoint.pth'):
        self.patience, self.delta, self.path = patience, delta, path
        self.best, self.count = None, 0
        self.stop = False
    def __call__(self, loss, model):
        if self.best is None or loss < self.best - self.delta:
            self.best, self.count = loss, 0
            torch.save(model.state_dict(), self.path)
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True

# 5) 微调 LSTM+Attention
#    5.1 实例化模型
model_lstm_ft = LSTMAttnModel(num_q2).to(device)
#    5.2 加载预训练权重，忽略大小不匹配的 embedding 层
pre = torch.load('best_lstm.pth')
mdict = model_lstm_ft.state_dict()
# 只读取尺寸匹配的键
pre_filt = {k:v for k,v in pre.items() if k in mdict and v.size()==mdict[k].size()}
mdict.update(pre_filt)
model_lstm_ft.load_state_dict(mdict)

opt_lstm = torch.optim.Adam(model_lstm_ft.parameters(), lr=1e-4)
sched_lstm = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_lstm, mode='min', factor=0.5, patience=2, verbose=True)
stopper_lstm = EarlyStopping(patience=5, path='ft_lstm.pth')

for epoch in range(1, 31):
    # 训练
    model_lstm_ft.train()
    tr_loss = 0
    for q,c,t in train_loader2:
        q,c,t = q.to(device), c.to(device), t.to(device)
        p = model_lstm_ft(q,c)
        loss = criterion(p, t[:, -1])
        opt_lstm.zero_grad(); loss.backward(); opt_lstm.step()
        tr_loss += loss.item()
    # 验证
    model_lstm_ft.eval()
    val_loss = 0
    with torch.no_grad():
        for q,c,t in val_loader2:
            q,c,t = q.to(device), c.to(device), t.to(device)
            val_loss += criterion(model_lstm_ft(q,c), t[:, -1]).item()
    tr_loss /= len(train_loader2)
    val_loss /= len(val_loader2)
    sched_lstm.step(val_loss)
    stopper_lstm(val_loss, model_lstm_ft)
    print(f"[LSTM ft] Epoch {epoch}: train={tr_loss:.4f}, val={val_loss:.4f}")
    if stopper_lstm.stop:
        print("Early stopping LSTM fine-tune")
        break

model_lstm_ft.load_state_dict(torch.load('ft_lstm.pth'))


# 6) 微调 DKT
model_dkt_ft = DKTModel(num_q2).to(device)
pre2 = torch.load('best_dkt.pth')
mdict2 = model_dkt_ft.state_dict()
pre2_filt = {k:v for k,v in pre2.items() if k in mdict2 and v.size()==mdict2[k].size()}
mdict2.update(pre2_filt)
model_dkt_ft.load_state_dict(mdict2)

opt_dkt = torch.optim.Adam(model_dkt_ft.parameters(), lr=1e-4)
sched_dkt = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_dkt, mode='min', factor=0.5, patience=2, verbose=True)
stopper_dkt = EarlyStopping(patience=5, path='ft_dkt.pth')

for epoch in range(1, 31):
    model_dkt_ft.train()
    trl = 0
    for q,c,t in train_loader2:
        q,c,t = q.to(device), c.to(device), t.to(device)
        loss = criterion(model_dkt_ft(q,c), t[:, -1])
        opt_dkt.zero_grad(); loss.backward(); opt_dkt.step()
        trl += loss.item()
    model_dkt_ft.eval()
    vll = 0
    with torch.no_grad():
        for q,c,t in val_loader2:
            q,c,t = q.to(device), c.to(device), t.to(device)
            vll += criterion(model_dkt_ft(q,c), t[:, -1]).item()
    trl /= len(train_loader2); vll /= len(val_loader2)
    sched_dkt.step(vll)
    stopper_dkt(vll, model_dkt_ft)
    print(f"[DKT ft] Epoch {epoch}: train={trl:.4f}, val={vll:.4f}")
    if stopper_dkt.stop:
        print("Early stopping DKT fine-tune")
        break

model_dkt_ft.load_state_dict(torch.load('ft_dkt.pth'))


# 7) 在验证集上评估微调效果
model_lstm_ft.eval(); model_dkt_ft.eval()
all_lstm, all_dkt, ys = [], [], []

with torch.no_grad():
    for q,c,t in val_loader2:
        q,c,t = q.to(device), c.to(device), t.to(device)
        all_lstm.append(model_lstm_ft(q,c).cpu().numpy())
        all_dkt.append (model_dkt_ft(q,c).cpu().numpy())
        ys.append(t[:, -1].cpu().numpy())

pred_lstm = np.concatenate(all_lstm)
pred_dkt  = np.concatenate(all_dkt)
y_true    = np.concatenate(ys)

print("Fine-tuned LSTM AUC:", roc_auc_score(y_true, pred_lstm))
print("Fine-tuned DKT  AUC:", roc_auc_score(y_true, pred_dkt))


# 8) 融合 & 评估—权重平均 与 Logistic Regression Stacking

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

# --- 8.1) 收集 Fine-tuned 模型在验证集上的预测与真实标签 ---
preds_lstm = []
preds_dkt  = []
y_true     = []
with torch.no_grad():
    for q, c, t in val_loader2:
        q, c = q.to(device), c.to(device)
        preds_lstm.append(model_lstm_ft(q, c).cpu().numpy())
        preds_dkt .append(model_dkt_ft (q, c).cpu().numpy())
        y_true    .append(t[:, -1].cpu().numpy())
preds_lstm = np.concatenate(preds_lstm)
preds_dkt  = np.concatenate(preds_dkt)
y_true     = np.concatenate(y_true)

# --- 8.2) 方法 1: AUC 加权平均 (以各自 AUC 为权重) ---
auc_lstm = roc_auc_score(y_true, preds_lstm)
auc_dkt  = roc_auc_score(y_true, preds_dkt)
w1 = auc_lstm / (auc_lstm + auc_dkt)
w2 = auc_dkt  / (auc_lstm + auc_dkt)
weighted_pred = w1 * preds_lstm + w2 * preds_dkt
print(f"Weighted avg AUC:     {roc_auc_score(y_true, weighted_pred):.4f}")
print(f"Weighted avg LogLoss: {log_loss(y_true, weighted_pred):.4f}")

# --- 8.3) 方法 2: Logistic Regression Stacking ---
# 构造特征矩阵 [pred_lstm, pred_dkt]
X = np.vstack([preds_lstm, preds_dkt]).T
meta = LogisticRegression()
meta.fit(X, y_true)
stack_pred = meta.predict_proba(X)[:, 1]
print(f"Stacking (LR) AUC:     {roc_auc_score(y_true, stack_pred):.4f}")
print(f"Stacking (LR) LogLoss: {log_loss(y_true, stack_pred):.4f}")

