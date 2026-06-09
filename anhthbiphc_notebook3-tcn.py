import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


# Load preprocessed data (replace with your dataset path if needed)
train_df = pd.read_pickle("//kaggle/input/processed-data/processed_train.pkl")
test_df = pd.read_pickle("/kaggle/input/processed-data/processed_test.pkl")

# Convert all numeric columns to float32 to save memory
numeric_cols_train = train_df.select_dtypes(include=['float64', 'int64']).columns
train_df[numeric_cols_train] = train_df[numeric_cols_train].astype('float32')

numeric_cols_test = test_df.select_dtypes(include=['float64', 'int64']).columns
test_df[numeric_cols_test] = test_df[numeric_cols_test].astype('float32')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


print("Train columns:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())



%%writefile nb3_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------- 1. Vanilla Transformer Encoder ----------
class TransformerEncoderModel(nn.Module):
    def __init__(self, in_channels, num_classes, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ---------- 2. Time Series Transformer (TST) ----------
class TSTModel(nn.Module):
    def __init__(self, in_channels, num_classes, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers=num_layers
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B, T, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, self.embed(x)), dim=1)
        pos = self.pos_embed[:, :x.size(1)]
        x = x + pos
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)

# ---------- 3. InformerLite ----------
class ProbSparseAttention(nn.Module):
    def __init__(self, topk=64):
        super().__init__()
        self.topk = topk
    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / q.size(-1)**0.5
        topk = min(self.topk, scores.size(-1))
        topk_idx = scores.topk(topk, dim=-1).indices
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)
        scores = scores * mask - 1e9 * (1 - mask)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

class InformerLite(nn.Module):
    def __init__(self, in_channels, num_classes, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1, topk=64):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.attn = ProbSparseAttention(topk)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            attn_out = self.attn(x, x, x)
            x = layer(attn_out)
        x = x.mean(1)
        return self.fc(x)

# ---------- 4. Temporal Convolutional Network (TCN) ----------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_channels, num_classes, levels=4, channels=64, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_inputs = in_channels
        for i in range(levels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(num_inputs, channels, kernel_size, stride=1,
                                        dilation=dilation, padding=padding, dropout=dropout))
            num_inputs = channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)     # [B, C, T]
        out = self.network(x)
        out = out.mean(-1)
        return self.fc(out)



# ================== NB3 HARNESS (chuáº©n schema) ==================
import os, time, math, re, json, warnings
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import uuid
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 1) Dataset ----------
class ArrayTSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ---------- 2) Feature selector ----------
# Feature selector
def select_feature_cols(df: pd.DataFrame, features_used: str, target_col: str, seq_col: str) -> List[str]:
    # TÃ¹y vÃ o schema cá»§a báº¡n, chá»‰nh prefix dÆ°á»›i Ä‘Ã¢y cho phÃ¹ há»£p
    imu_prefixes = ("acc_", "gyro_", "rot_", "mag_")
    tof_prefixes = ("tof_",)
    thm_prefixes = ("thm_", "temp_", "thermo_")

    def pick(prefixes):
        return [c for c in df.columns
                if c not in [target_col, seq_col]
                and df[c].dtype in ["float32","float64","int32","int64"]
                and any(c.startswith(p) for p in prefixes)]

    if features_used.lower() == "imu":
        cols = pick(imu_prefixes)
    elif features_used.lower() == "tof":
        cols = pick(tof_prefixes)
    elif features_used.lower() == "thermo":
        cols = pick(thm_prefixes)
    elif features_used.lower() in ["full","all","fusion"]:
        cols = [c for c in df.columns if c not in [target_col, seq_col] and df[c].dtype in ["float32","float64","int32","int64"]]
    else:
        # fallback: dÃ¹ng táº¥t cáº£ numeric
        cols = [c for c in df.columns if c not in [target_col, seq_col] and df[c].dtype in ["float32","float64","int32","int64"]]
    if len(cols) == 0:
        raise ValueError(f"KhÃ´ng tÃ¬m tháº¥y cá»™t feature phÃ¹ há»£p cho '{features_used}'.")
    return cols



# ---------- 3) Group per sequence -> [N, T, C] ----------
# Group per sequence -> [N, T, C]
def build_sequences(df: pd.DataFrame, seq_col: str, target_col: str, feature_cols: List[str],
                    window_size: int) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    grouped, labels = [], []
    for seq, g in df.groupby(seq_col):
        X_seq = g[feature_cols].to_numpy(dtype=np.float32)  # [Ti, C]
        labels.append(g[target_col].iloc[0])  # Target is grouped gesture_grouped_id (0-8)
        # pad/crop to window_size
        if len(X_seq) >= window_size:
            grouped.append(X_seq[:window_size])
        else:
            pad = window_size - len(X_seq)
            grouped.append(np.pad(X_seq, ((0,pad),(0,0)), mode="constant"))
    X = np.stack(grouped)

    # â�— Sá»¬A Lá»–I: DÃ¹ng ID Ä‘Ã£ encode sáºµn (0-8) vÃ  tráº£ vá»� LabelEncoder toÃ n cá»¥c
    y = np.asarray(labels).astype(np.int64)
    global le_gesture_grouped
    
    return X, y, le_gesture_grouped
# =========================
# 1) Group gesture cho TRAIN
# =========================

# Gesture mapping (8 target + 10 non-target)
GESTURE_MAPPER = {
    "Above ear - pull hair": 0,
    "Cheek - pinch skin": 1,
    "Eyebrow - pull hair": 2,
    "Eyelash - pull hair": 3,
    "Forehead - pull hairline": 4,
    "Forehead - scratch": 5,
    "Neck - pinch skin": 6,
    "Neck - scratch": 7,

    "Drink from bottle/cup": 8,
    "Feel around in tray and pull out an object": 9,
    "Glasses on/off": 10,
    "Pinch knee/leg skin": 11,
    "Pull air toward your face": 12,
    "Scratch knee/leg skin": 13,
    "Text on phone": 14,
    "Wave hello": 15,
    "Write name in air": 16,
    "Write name on leg": 17,
}

TARGET_LABELS = [k for k,v in GESTURE_MAPPER.items() if v <= 7]

def to_grouped_label(name):
    return name if name in TARGET_LABELS else "non_target"

# â�— chá»‰ Ã¡p dá»¥ng cho train_df, KHÃ”NG Ã¡p dá»¥ng test_df
train_df["gesture_grouped"] = train_df["gesture"].apply(to_grouped_label)

from sklearn.preprocessing import LabelEncoder

le_gesture_grouped = LabelEncoder()
train_df["gesture_grouped_id"] = le_gesture_grouped.fit_transform(train_df["gesture_grouped"])
# CLASS_NAMES dÃ¹ng cho evaluate()
CLASS_NAMES = list(le_gesture_grouped.classes_)

print(CLASS_NAMES, len(CLASS_NAMES))

print("Grouped labels:", le_gesture_grouped.classes_)


@torch.no_grad()
def evaluate(model, loader, class_names=None, plot_cm=False, save_path=None) -> Dict[str, float]:
    """
    Evaluate model with F1, accuracy, and optional confusion matrix visualization.
    Compatible with both binary and multiclass tasks.
    """
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb).argmax(1)
        y_true.extend(yb.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = len(np.unique(y_true))

    # --- Compute F1 scores ---
    if n_classes == 2:
        bin_f1 = f1_score(y_true, y_pred, pos_label=1)
        macro = f1_score(y_true, y_pred, average="macro")
    else:
        bin_f1 = f1_score(y_true, y_pred, average="micro")  # dÃ¹ng micro-F1 cho multiclass
        macro = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    final = (bin_f1 + macro) / 2

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    labels = class_names if class_names else np.arange(n_classes)

    if plot_cm:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"ğŸ“� Saved confusion matrix to {save_path}")
        plt.show()

    return {
        "Binary F1": float(bin_f1),
        "Macro F1": float(macro),
        "Final Score": float(final),
        "val_acc": float(acc),
        "confusion_matrix": cm
    }

# ---------- 5) Train (mini-protocol ready) ----------
def train_one(model, train_loader, val_loader, optimizer, epochs: int, grad_accum: int = 1, amp: bool = True):
    model.to(DEVICE)
    amp_enabled = (amp and DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss()
    total_train_time = 0.0
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    best = {"Final Score": -1, "epoch": 0}
    epoch_logs = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if torch.__version__.startswith("2.1") or torch.__version__.startswith("2.2"):
                autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled)

            with autocast_ctx:
                logits = model(xb)
                loss = criterion(logits, yb) / grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        total_train_time += (time.time() - t0)

        # --- Evaluate sau má»—i epoch ---
        metrics = evaluate(model, val_loader, class_names=CLASS_NAMES, plot_cm=False)
        epoch_logs.append({
            "epoch": ep + 1,
            **{k: round(v, 4) for k, v in metrics.items() if k in ["Binary F1", "Macro F1", "Final Score", "val_acc"]}
        })


        if metrics["Final Score"] > best["Final Score"]:
            best = metrics
        # --- Cáº­p nháº­t best náº¿u tá»‘t hÆ¡n ---
        if metrics["Final Score"] > best["Final Score"]:
            best = metrics
            best["epoch"] = ep + 1

        # --- In chi tiáº¿t sau má»—i epoch ---
        print(f"Epoch {ep+1}/{epochs} | "
              f"Loss={loss.item():.4f} | "
              f"Binary F1={metrics['Binary F1']:.3f} | "
              f"Macro F1={metrics['Macro F1']:.3f} | "
              f"Final={metrics['Final Score']:.3f} | "
              f"Acc={metrics['val_acc']:.3f}")

    # --- Sau khi train xong: in tÃ³m táº¯t best ---
    best_epoch = best.get("epoch", epochs)
    print(f"\nğŸ�† Best Epoch: {best_epoch} | Final Score={best['Final Score']:.3f} | "
      f"Acc={best['val_acc']:.3f}")

    
    
    evaluate(
        model, val_loader,
        class_names=CLASS_NAMES,
        plot_cm=True,
        save_path=f"/kaggle/working/confmat_{model.__class__.__name__}_final.png"
    )
    # inference time / sample (trÃªn 1 batch val)
    model.eval()
    xb, _ = next(iter(val_loader))
    xb = xb.to(DEVICE)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t1 = time.time(); _ = model(xb)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    infer_ms = (time.time() - t1) / xb.size(0) * 1000

    peak_gb = (torch.cuda.max_memory_allocated(DEVICE) / (1024**3)) if DEVICE.type == "cuda" else 0.0
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return best, total_train_time, infer_ms, params_m, peak_gb, epoch_logs

# ---------- 6) Logging theo Ä‘Ãºng schema ----------
SCHEMA_COLS = [
    "model_name","runtime_id","notebook","features_used","window_size",
    "optimizer/solver","params (M)","Binary F1","Macro F1","Final Score",
    "val_acc","train_time","inference_time","notes"
]

def append_result(row: Dict[str, Any], csv_path: str = "/kaggle/working/tcn_results.csv"):
    df_row = pd.DataFrame([row], columns=SCHEMA_COLS)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row
    df.to_csv(csv_path, index=False)
    print(f"âœ… Logged to {csv_path}")

# ---------- 7) Length sweep {50,100,200} ----------
def run_length_sweep(train_df: pd.DataFrame,
                     model_builder,            # hÃ m: (in_channels, num_classes) -> nn.Module
                     model_name: str,
                     notebook_name: str,
                     features_used: str,       # "IMU" | "ToF" | "Thermo" | "Full"
                     seq_col: str,             # vÃ­ dá»¥ "sequence_counter"
                     target_col: str,          # vÃ­ dá»¥ "behavior"
                     lengths: List[int] = [50,100,200],
                     optimizer_name: str = "AdamW",
                     lr: float = 1e-3,
                     epochs: int = 5,
                     batch_size: int = 64,
                     grad_accum: int = 1,
                     amp: bool = True,
                     runtime_prefix: str = "mp01",
                     results_csv: str = "/kaggle/working/tcn_results.csv",
                     max_train_sequences: Optional[int] = 1000,   # mini-protocol cap
                     max_val_sequences: Optional[int] = 300):     # mini-protocol cap

    feat_cols_full = select_feature_cols(train_df, features_used, target_col, seq_col)

    for L in lengths:
        print(f"\n===== Sequence length = {L} | features={features_used} =====")
        # Build sequences
        X, y, le = build_sequences(train_df[[seq_col, target_col] + feat_cols_full], seq_col, target_col, feat_cols_full, L)
        C = X.shape[2]
        num_classes = len(CLASS_NAMES)
        print(f"[DEBUG CHECK] GiÃ¡ trá»‹ num_classes hiá»‡n táº¡i: {num_classes}")

        # Split & cap for mini-protocol
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        if max_train_sequences: X_tr, y_tr = X_tr[:max_train_sequences], y_tr[:max_train_sequences]
        if max_val_sequences:   X_va, y_va = X_va[:max_val_sequences], y_va[:max_val_sequences]

        train_loader = DataLoader(ArrayTSDataset(X_tr, y_tr), batch_size=batch_size, sampler=RandomSampler(ArrayTSDataset(X_tr, y_tr)))
        val_loader   = DataLoader(ArrayTSDataset(X_va, y_va), batch_size=batch_size, sampler=SequentialSampler(ArrayTSDataset(X_va, y_va)))

        # Model + Optim
        model = model_builder(C, num_classes)
        if optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Train
        best, train_time_s, infer_ms, params_m, peak_gb, epoch_logs = train_one(model, train_loader, val_loader, optimizer,
                                                                    epochs=epochs, grad_accum=grad_accum, amp=amp)

# --- DEBUG: Kiá»ƒm tra sá»‘ lá»›p Ä�áº¦U RA cá»§a mÃ´ hÃ¬nh vá»«a huáº¥n luyá»‡n ---
# Giáº£ sá»­ lá»›p Ä‘áº§u ra cuá»‘i cÃ¹ng cá»§a báº¡n lÃ  self.fc
                # --- DEBUG: Kiá»ƒm tra sá»‘ lá»›p Ä‘áº§u ra ---
        try:
            # Náº¿u model.fc lÃ  Sequential â†’ láº¥y pháº§n tá»­ cuá»‘i
            if isinstance(model.fc, nn.Sequential):
                fc_last = model.fc[-1]
            else:
                # Náº¿u chá»‰ lÃ  1 Linear Layer
                fc_last = model.fc

            output_shape = fc_last.weight.shape   # (num_classes, input_dim)
            num_classes_trained = output_shape[0]

            print(f"\n[DEBUG] Output layer size = {num_classes_trained}")

            if num_classes_trained != num_classes:
                print("ğŸš¨ WARNING: Output layer size KHÃ”NG TRÃ™NG num_classes. Kiá»ƒm tra MODEL_BUILDER.")

        except Exception as e:
            print(f"[DEBUG] KhÃ´ng Ä‘á»�c Ä‘Æ°á»£c lá»›p fc cá»§a model: {e}")

# -------------------------------------------------------------
# -------------------------------------------------------------
        # 1) LÆ°u feature cols
        np.save("/kaggle/working/feature_cols.npy", np.array(feat_cols_full, dtype=object))

# 2) LÆ°u danh sÃ¡ch gesture classes
        np.save("/kaggle/working/gesture_classes.npy", le_gesture_grouped.classes_)


# 3) LÆ°u sequence length (window_size)
        np.save("/kaggle/working/sequence_maxlen.npy", np.array([L]))

# 4) LÆ°u scaler náº¿u cÃ³
        if 'scaler' in globals() and scaler is not None:
            joblib.dump(scaler, "/kaggle/working/scaler.pkl")

# 5) LÆ°u checkpoint model theo window-size
        ckpt_path = f"/kaggle/working/{model_name}_L{L}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"ğŸ’¾ Saved checkpoint to {ckpt_path}")


        # Log 1 dÃ²ng theo schema
        row = {
            "model_name": model_name,
            "runtime_id": f"{runtime_prefix}_{model_name}_L{L}_{uuid.uuid4().hex[:4]}",
            "notebook": notebook_name,
            "features_used": features_used,
            "window_size": L,
            "optimizer/solver": optimizer_name,
            "params (M)": round(params_m, 3),
            "Binary F1": round(best["Binary F1"], 3),
            "Macro F1": round(best["Macro F1"], 3),
            "Final Score": round(best["Final Score"], 3),
            "val_acc": round(best["val_acc"], 3),
            "train_time": f"{train_time_s:.2f}s",
            "inference_time": f"{infer_ms:.2f} ms/sample",
            "notes": f"mini-protocol; peak_mem={peak_gb:.2f}GB; grad_accum={grad_accum}; amp={'on' if (amp and DEVICE.type=='cuda') else 'off'}"
        }
        append_result(row, results_csv)
# ================== /NB3 HARNESS ==================
# ================== 8) Evaluate on IMU-only and Full splits ==================
def run_full_evaluation(
    train_df: pd.DataFrame,
    model_builder,
    model_name: str,
    notebook_name: str,
    seq_col: str = "sequence_counter",
    target_col: str = "gesture_grouped_id",
    lengths: List[int] = [50,100,200],
    optimizer_name: str = "AdamW",
    lr: float = 1e-3,
    epochs: int = 5,
    batch_size: int = 64,
    grad_accum: int = 1,
    amp: bool = True,
    runtime_prefix: str = "mp01",
    results_csv: str = "/kaggle/working/tcn_results.csv",
    max_train_sequences: Optional[int] = 1000,
    max_val_sequences: Optional[int] = 300,
):
    """Run both IMU-only and Full splits (Task 6)."""
    print("\n===============================")
    print(f"ğŸ�� Starting full evaluation for {model_name}")
    print("===============================")

    for feature_mode in ["IMU", "Full"]:
        print(f"\n### Running {feature_mode}-only mode ###")
        run_length_sweep(
            train_df=train_df,
            model_builder=model_builder,
            model_name=model_name,
            notebook_name=notebook_name,
            features_used=feature_mode,
            seq_col=seq_col,
            target_col=target_col,
            lengths=lengths,
            optimizer_name=optimizer_name,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            amp=amp,
            runtime_prefix=f"{runtime_prefix}_{feature_mode}",
            results_csv=results_csv,
            max_train_sequences=max_train_sequences,
            max_val_sequences=max_val_sequences,
        )

    print("\nâœ… Completed both IMU and Full runs. Results appended to CSV.")



from nb3_models import TCN

def MODEL_BUILDER(in_channels, num_classes):
    return TCN(in_channels, num_classes, levels=4, channels=64, kernel_size=3, dropout=0.1)

MODEL_NAME = "TCN"
NOTEBOOK_NAME = "NB3-4 TCN"

run_full_evaluation(train_df, MODEL_BUILDER, MODEL_NAME, NOTEBOOK_NAME,
                    seq_col="sequence_counter", target_col="gesture_grouped_id",
                    lengths=[50,100,200], optimizer_name="SGD", lr=1e-2,
                    epochs=5, batch_size=64, grad_accum=1, amp=True,
                    runtime_prefix="mp01", results_csv="/kaggle/working/tcn_results.csv",
                    max_train_sequences=1000, max_val_sequences=300)



import pandas as pd

# --- Ä�Æ°á»�ng dáº«n Ä‘áº¿n file káº¿t quáº£ ---
csv_path = "/kaggle/working/tcn_results.csv"

# --- Ä�á»�c vÃ  sáº¯p xáº¿p ---
df = pd.read_csv(csv_path)
df = df.sort_values(by=["model_name", "features_used", "window_size"]).reset_index(drop=True)

# --- Chá»�n cá»™t hiá»ƒn thá»‹ ---
cols_to_show = [
    "model_name", "features_used", "window_size",
    "Binary F1", "Macro F1", "Final Score",
    "val_acc", "params (M)", "train_time", "inference_time"
]

# --- In toÃ n bá»™ báº£ng ---
print("\n==============================================")
print("ğŸ“Š NB3 Full Results (All Window Sizes)")
print("==============================================\n")
print(df[cols_to_show].to_string(index=False))





import os
from IPython.display import Image, display

# --- Láº·p qua tá»«ng dÃ²ng káº¿t quáº£ ---
for _, row in df.iterrows():
    model_name = row["model_name"]
    feature = row["features_used"]
    window = int(row["window_size"])

    cm_path = f"/kaggle/working/confmat_{model_name}_{feature}_L{window}.png"

    print(f"\nğŸ”¹ {model_name} | {feature} | window={window}")
    if os.path.exists(cm_path):
        display(Image(cm_path))
    else:
        print(f"âš ï¸� Confusion matrix not found at {cm_path}")



# ======================================================
# ğŸ”¥ CREATE SUBMISSION.PARQUET FROM YOUR TRAINED MODEL
# ======================================================

import numpy as np
import pandas as pd
import polars as pl
import joblib
import torch
import torch.nn.functional as F

# Load artefacts
FEATURE_COLS = np.load("/kaggle/working/feature_cols.npy", allow_pickle=True).tolist()
GESTURE_CLASSES = np.load("/kaggle/working/gesture_classes.npy", allow_pickle=True).tolist()
MAXLEN = int(np.load("/kaggle/working/sequence_maxlen.npy")[0])

scaler = None
if os.path.exists("/kaggle/working/scaler.pkl"):
    scaler = joblib.load("/kaggle/working/scaler.pkl")

# Load best checkpoint (thay model_name + L cho Ä‘Ãºng file báº¡n lÆ°u)
CHECKPOINT = "/kaggle/working/TCN_L100.pth"

model = MODEL_BUILDER(len(FEATURE_COLS), len(GESTURE_CLASSES))
state = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(state, strict=True)
# Inspect fc layer key
fc_keys = [k for k in state.keys() if k.startswith("fc")]
print("Available fc keys:", fc_keys)

# Print the actual weight
print(state[fc_keys[0]].shape)


print("Loaded checkpoint:", CHECKPOINT)

model.eval()

# ----------------- PREPROCESS FUNCTION -----------------

def preprocess_sequence(df):
    # Táº¡o DataFrame má»›i chá»‰ chá»©a cÃ¡c cá»™t FEATURE_COLS
    # DÃ¹ng reindex Ä‘á»ƒ Ä‘áº£m báº£o táº¥t cáº£ FEATURE_COLS Ä‘á»�u cÃ³ máº·t. 
    # Nhá»¯ng cá»™t cÃ³ sáºµn sáº½ giá»¯ nguyÃªn giÃ¡ trá»‹; nhá»¯ng cá»™t thiáº¿u sáº½ Ä‘Æ°á»£c Ä‘iá»�n 0 (fill_value=0).
    seq_df = df.reindex(columns=FEATURE_COLS, fill_value=0).astype("float32")
    
    # LÃºc nÃ y, seq_df Ä‘Ã£ cÃ³ Ä‘áº§y Ä‘á»§ cÃ¡c cá»™t vÃ  khÃ´ng bá»‹ fragmented.
    X = seq_df.values

    if scaler:
        X = scaler.transform(X)
    
    # Pad/truncate
    T = X.shape[0]
    if T >= MAXLEN:
        X = X[:MAXLEN]
    else:
        pad = np.zeros((MAXLEN - T, X.shape[1]), dtype=np.float32)
        X = np.vstack([X, pad])
    return X

# ----------------- LOAD TEST DATA -----------------
test_df = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv").to_pandas()
demo_df = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv").to_pandas()

records = []
for seq, g in test_df.groupby("sequence_id"):
    seq_pd = g  # Ä‘Ã£ lÃ  pandas rá»“i
    X = preprocess_sequence(seq_pd)

    X_tensor = torch.tensor(X).unsqueeze(0)

    with torch.no_grad():
        logits = model(X_tensor)
        pred = logits.argmax(dim=1).item()

    gesture = GESTURE_CLASSES[pred]
    records.append((seq, gesture))

# ----------------- CREATE SUBMISSION -----------------
df_sub = pd.DataFrame(records, columns=["sequence_id", "gesture"])  # Ensure the column is 'gesture'

# ----------------- VERIFY THE STRUCTURE -----------------
print("First few rows of the submission data:")
print(df_sub.head())  # Verify the structure and the values in 'gesture' column

# Check if 'gesture' contains valid gesture classes
if df_sub["gesture"].isin(GESTURE_CLASSES).all():
    print("âœ… All gesture values are valid.")
else:
    print("â�Œ There are invalid gesture values in the 'gesture' column.")

# Check the number of rows and columns
print(f"Number of rows: {df_sub.shape[0]}, Number of columns: {df_sub.shape[1]}")
print(f"Column names: {df_sub.columns.tolist()}")

# ----------------- SAVE TO PARQUET -----------------
df_sub.to_parquet("/kaggle/working/submission.parquet", index=False)

print("âœ… DONE! Saved /kaggle/working/submission.parquet")


