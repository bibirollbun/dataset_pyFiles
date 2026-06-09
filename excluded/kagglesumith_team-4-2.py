import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

# === Load the preprocessed datasets ===
train_data = pd.read_pickle("/kaggle/input/processed-data/processed_train.pkl")
test_data = pd.read_pickle("/kaggle/input/processed-data/processed_test.pkl")

# === Optimize numeric column types to reduce memory usage ===
def optimize_numeric_dtypes(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    return df

train_data = optimize_numeric_dtypes(train_data)
test_data = optimize_numeric_dtypes(test_data)

print(f"Training data dimensions: {train_data.shape}")
print(f"Testing data dimensions: {test_data.shape}")



print("Train columns:", train_data.columns.tolist())
print("Test columns:", test_data.columns.tolist())



# ==========================
# Imports & Config
# ==========================
import os, time, math, random, gc
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPTIMIZER_NAME = "AdamW"


# ==========================
# Utilities
# ==========================
def infer_grid_shape(n_channels):
    r = int(round(math.sqrt(n_channels)))
    return (r, r) if r * r == n_channels else (8, 8)

def grid_adjacency(H, W, mode='8n', weight='uniform'):
    N = H * W
    A = np.zeros((N, N), dtype=np.float32)
    def idx(r, c): return r * W + c
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if mode == '8n': nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(H):
        for c in range(W):
            i = idx(r, c)
            for dr, dc in nbrs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    j = idx(rr, cc)
                    if weight == 'uniform':
                        A[i, j] = 1.0
                    else:
                        d = math.sqrt(dr**2 + dc**2)
                        A[i, j] = 1.0 / (d + 1e-6)
    A += np.eye(N)
    D = A.sum(axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(D + 1e-8)
    A_hat = (A * D_inv_sqrt[:, None]) * D_inv_sqrt[None, :]
    return torch.from_numpy(A_hat)

def add_mask_channel(x, sentinel=-1.0):
    val = np.where(x == sentinel, 0, x)
    mask = (~np.isclose(x, sentinel)).astype(np.float32)
    return np.stack([val, mask], axis=-1)



# ==========================
# Dataset
# ==========================
class ToFDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = np.transpose(self.X[i], (0, 3, 1, 2))  # [T, C, H, W]
        return torch.from_numpy(x), int(self.y[i])



class GraphConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, X, A):
        return torch.matmul(A, self.lin(X))

class SimpleGCN(torch.nn.Module):
    def __init__(self, H, W, in_ch, nclass):
        super().__init__()
        self.N = H * W
        self.gc1 = GraphConv(in_ch, 32)
        self.gc2 = GraphConv(32, 64)
        self.fc = torch.nn.Linear(64, nclass)

    def forward(self, x, A):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        x = x.reshape(B, T, H * W, C)
        outs = []
        for t in range(T):
            h = F.relu(self.gc1(x[:, t], A))
            h = F.relu(self.gc2(h, A))
            pooled = h.mean(1)
            outs.append(pooled)
        out = torch.stack(outs, 1).mean(1)
        return self.fc(out)

class SimpleGAT(torch.nn.Module):
    def __init__(self, H, W, in_ch, nclass, heads=4):
        super().__init__()
        self.N = H * W
        self.fc1 = torch.nn.Linear(in_ch, 32)
        self.attn = torch.nn.MultiheadAttention(32, num_heads=heads, batch_first=True)
        self.fc2 = torch.nn.Linear(32, nclass)

    def forward(self, x, A=None):  # A không dùng, nhưng giữ để tương thích
        B, T, H, W, C = x.shape
        x = x.reshape(B, T, H * W, C)
        outs = []
        for t in range(T):
            h = F.relu(self.fc1(x[:, t]))
            attn_out, _ = self.attn(h, h, h)
            pooled = attn_out.mean(1)
            outs.append(pooled)
        out = torch.stack(outs, 1).mean(1)
        return self.fc2(out)

class CNN2D(torch.nn.Module):
    def __init__(self, nclass, in_ch):
        super().__init__()
        self.frame = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.temporal = torch.nn.LSTM(64, 64, batch_first=True)
        self.fc = torch.nn.Linear(64, nclass)

    def forward(self, x):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        feats = []
        for t in range(T):
            xt = x[:, t].permute(0, 3, 1, 2)  # -> (B, C, H, W)
            feat = self.frame(xt).flatten(1)
            feats.append(feat)
        z = torch.stack(feats, 1)
        z, _ = self.temporal(z)
        z = z.mean(1)
        return self.fc(z)

class CNN3D(torch.nn.Module):
    def __init__(self, nclass, in_ch):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d((2,2,2)),
            torch.nn.Conv3d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d(1)
        )
        self.fc = torch.nn.Linear(32, nclass)

    def forward(self, x):
        # x: (B, T, H, W, C)
        x = x.permute(0, 4, 1, 2, 3)  # -> (B, C, T, H, W)
        z = self.net(x).flatten(1)
        return self.fc(z)




import numpy as np

# Giả lập dữ liệu (bạn thay bằng dữ liệu thực nếu có)
N, T, H, W, C = 100, 10, 8, 8, 3   # 100 mẫu, 10 khung, 8x8, 3 kênh
X = np.random.rand(N, T, H, W, C).astype(np.float32)
y = np.random.randint(0, 2, size=(N,))  # ví dụ 2 lớp (nhị phân)

# Chia train/val
X_train, X_val = X[:80], X[80:]
y_train, y_val = y[:80], y[80:]



from sklearn.preprocessing import StandardScaler
import numpy as np

X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])

scaler = StandardScaler().fit(X_train_reshaped)
X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_val = scaler.transform(X_val_reshaped).reshape(X_val.shape)



def run_experiments(X_train, y_train, X_val, y_val, H=8, W=8, mask_channel=None, epochs=80):
    import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time, pandas as pd
    from sklearn.metrics import f1_score, accuracy_score

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = len(np.unique(y_train))
    in_channels = X_train.shape[-1]

    # === Helper: adjacency 8-neighbor ===
    def build_adjacency(H, W):
        A = torch.zeros(H*W, H*W)
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < H and 0 <= nj < W:
                            n_idx = ni * W + nj
                            A[idx, n_idx] = 1
        D_inv = torch.diag(1.0 / (A.sum(1) + 1e-5))
        return (D_inv @ A).to(DEVICE)

    A_hat = build_adjacency(H, W)

    # === DataLoader ===
    bs = 32
    dl_tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                       torch.tensor(y_train, dtype=torch.long)),
        batch_size=bs, shuffle=True)
    dl_va = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                       torch.tensor(y_val, dtype=torch.long)),
        batch_size=bs, shuffle=False)

    # === Models ===
    models = {
        "GCN": lambda: SimpleGCN(H, W, in_ch=in_channels, nclass=n_classes),
        "GAT": lambda: SimpleGAT(H, W, in_ch=in_channels, nclass=n_classes),
        "CNN2D": lambda: CNN2D(nclass=n_classes, in_ch=in_channels),
        "CNN3D": lambda: CNN3D(nclass=n_classes, in_ch=in_channels),
    }

    # === Evaluate ===
    @torch.no_grad()
    def evaluate(model, dl, A=None):
        model.eval()
        y_true, y_pred = [], []
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb, A) if A is not None else model(xb)
            y_pred += logits.argmax(1).cpu().tolist()
            y_true += yb.cpu().tolist()
        macro = f1_score(y_true, y_pred, average='macro')
        binary = f1_score(y_true, y_pred, average='binary') if len(set(y_true)) == 2 else 0
        acc = accuracy_score(y_true, y_pred)
        return binary, macro, acc

    results = []

    # === Train each model ===
    for name, fn in models.items():
        print(f"\n▶ Training {name} ...")
        model = fn().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        best_macro = 0
        patience = 0
        start = time.time()

        for ep in range(1, epochs+1):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits = model(xb, A_hat) if name in ["GCN", "GAT"] else model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
            scheduler.step()

            bin_f1, mac_f1, acc = evaluate(model, dl_va, A_hat if name in ["GCN","GAT"] else None)
            if ep % 5 == 0:
                print(f"Epoch {ep}/{epochs}: loss={loss.item():.3f}, Macro={mac_f1:.3f}, Acc={acc:.3f}")
            if mac_f1 > best_macro:
                best_macro = mac_f1
                patience = 0
            else:
                patience += 1
                if patience > 10:
                    print(f"⏹ Early stop at epoch {ep}")
                    break

        train_time = time.time() - start
        start_inf = time.time()
        evaluate(model, dl_va, A_hat if name in ["GCN","GAT"] else None)
        inf_time = (time.time() - start_inf) / len(dl_va.dataset)

        bin_f1, mac_f1, acc = evaluate(model, dl_va, A_hat if name in ["GCN","GAT"] else None)
        results.append({
            "features_used": "ToF_Grid",
            "window_size": 10,
            "optimizer/solver": "AdamW+Cosine",
            "params (M)": round(params_m, 4),
            "Binary": round(bin_f1, 4),
            "Macro": round(mac_f1, 4),
            "Final Score": round(mac_f1/2, 4),
            "val_acc": round(acc, 4),
            "train_time": round(train_time, 2),
            "inference_time": round(inf_time, 4),
        })

    df = pd.DataFrame(results)
    print("\n=== RESULTS ===")
    print(df)
    return df



# Run experiments
report = run_experiments(X_train, y_train, X_val, y_val, H=8, W=8, epochs=80)

# Define output path
output_path = "/kaggle/working/submission.parquet"

# Save the submission file in the correct format and location
report.to_parquet(output_path, index=False)

print(f"✅ Submission file saved successfully at: {output_path}")





