import numpy as np
import pandas as pd 
import os
import ast, random, re
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassF1Score
from tqdm import tqdm



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Nhét LODGO LOSO, nhét khử nhiễu


# ------------------- Cấu hình chung -----------------------
class CFG:
    seed          = 42
    # device        = torch.device("xla" if torch.cuda.is_available() else "cpu")
    device        = torch.device("cuda")

    data_path     = "/kaggle/input/2nd-wear-dataset-challenge"
    test_path     = "/kaggle/input/2nd-wear-dataset-challenge/test.csv"

    window_size   = 50   # 1 giây @ 50 Hz
    stride        = 25

    # ViT-LSTM
    vit_dim       = 256
    
    vit_heads     = 32
    heads         = 32

    mlp_ratio     = 4.0 
    vit_depth     = 16
    vit_mlp_ratio = 4.0
    vit_dropout   = 0.3
    dropout        = 0.3
    
    patch_size    = 5
    num_blocks    = 8
    
    lstm_hidden   = 256
    lstm_layers   = 4
    lstm_dropout  = 0.3

    num_classes   = 19
    batch_size    = 256
    epochs        = 200
    lr            = 1e-3

    
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(CFG.seed)
print(f"Using device: {CFG.device}")


pd.options.mode.use_inf_as_na = True       
ACC_RE = re.compile(r'_(acc_[xyz])$')  

label_map = {
    'null': 0, 'jogging': 1, 'jogging (rotating arms)': 2, 'jogging (skipping)': 3,
    'jogging (sidesteps)': 4, 'jogging (butt-kicks)': 5, 'stretching (triceps)': 6,
    'stretching (lunging)': 7, 'stretching (shoulders)': 8, 'stretching (hamstrings)': 9,
    'stretching (lumbar rotation)': 10, 'push-ups': 11, 'push-ups (complex)': 12,
    'sit-ups': 13, 'sit-ups (complex)': 14, 'burpees': 15, 'lunges': 16,
    'lunges (complex)': 17, 'bench-dips': 18
}

def load_one_train(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={'label': 'string'},
        low_memory=False
    )

    # ---- ép nhãn ----
    if 'label' in df.columns:
        df['label'] = (df['label'].str.strip().str.lower()
                        .map(label_map).astype('Int8', errors='ignore'))

    acc_cols = [c for c in df.columns if ACC_RE.search(c)]
    df[acc_cols] = df[acc_cols].astype(np.float32)


    if acc_cols:
        df[acc_cols] = (
            df.groupby('sbj_id', group_keys=False)[acc_cols]
              .apply(lambda g: g.interpolate(method='linear',
                                              limit_direction='both'))
        )

    return df

# --------- load TRAIN ----------
train_dir   = os.path.join(CFG.data_path, "train")
train_files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]

df_train_raw = pd.concat(
    [load_one_train(os.path.join(train_dir, f)) for f in tqdm(train_files, desc="Loading train files")],
    ignore_index=True
)

# --------- load TEST ----------
df_test_raw = pd.read_csv(
    CFG.test_path,
    converters={'x_axis': ast.literal_eval,
                'y_axis': ast.literal_eval,
                'z_axis': ast.literal_eval}
)

print("Train shape:", df_train_raw.shape, "| Test shape:", df_test_raw.shape)


# Testing!!!


df_test = df_test_raw.copy()
df_test['window_idx'] = df_test.groupby(['sbj_id','sensor_location']).cumcount()

# pivot sang wide format
df_wide = df_test.pivot(
    index=['sbj_id','window_idx'],
    columns='sensor_location',
    values=['x_axis','y_axis','z_axis']
)
df_wide.columns = [f"{axis}_{loc}" for axis,loc in df_wide.columns]
df_wide = df_wide.reset_index()

channels = []
for loc in ['right_arm','left_arm','right_leg','left_leg']:
    for axis in ['x_axis','y_axis','z_axis']:
        channels.append(f"{axis}_{loc}")

X_test = np.stack(
    [ np.stack(df_wide[ch].tolist(), axis=0) for ch in channels ],
    axis=1
)


df_train_raw['label'] = (
    df_train_raw['label']
      .fillna(0)
      .astype('Int8')
)


vals = df_train_raw['label'].dropna().unique()
df_train_raw['label'] = df_train_raw['label'].fillna(0)

df_train_raw['label'] = df_train_raw['label'].astype(int)

print("Tổng cộng", len(vals), "giá trị:")
print(*sorted(vals), sep="\n")

print("Train data shape:", df_train_raw.shape)
print("Test  data shape:", df_test_raw.shape)
display(df_train_raw)


SENSOR_LOCATIONS = ['right_arm', 'left_arm', 'right_leg', 'left_leg']

def create_location_specific_windows(df_raw, win_size, stride):
    all_windows = []
    
    for sbj_id, group in tqdm(df_raw.groupby('sbj_id'), desc="Creating windows"):
        labels = group['label'].values
        
        # Tạo dữ liệu cho từng vị trí
        location_data = {}
        for loc in SENSOR_LOCATIONS:
            cols = [f'{loc}_acc_x', f'{loc}_acc_y', f'{loc}_acc_z']
            location_data[loc] = group[cols].values.T # Shape (3, N)
            
        n_samples = len(group)
        for start in range(0, n_samples - win_size + 1, stride):
            end = start + win_size
            
            # Lấy nhãn cho window (lấy nhãn cuối cùng)
            window_label = labels[end-1]
            
            # Tạo một window cho mỗi vị trí
            for loc in SENSOR_LOCATIONS:
                window_tensor = location_data[loc][:, start:end].astype(np.float32)
                all_windows.append({
                    'tensor': window_tensor,
                    'label': window_label,
                    'sensor_location': loc,
                    'sbj_id': sbj_id
                })
                
    return pd.DataFrame(all_windows)

df_windows = create_location_specific_windows(df_train_raw, CFG.window_size, CFG.stride)

print(f"Total windows created: {len(df_windows)}")


import numpy as np
from scipy.interpolate import CubicSpline

def augment_jitter(x: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    return (x + noise).astype(x.dtype)


def augment_scale(x: np.ndarray, scale_range: tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    factor = np.random.uniform(scale_range[0], scale_range[1])
    return (x * factor).astype(x.dtype)


def augment_rotation(x: np.ndarray) -> np.ndarray:
    if x.shape[0] != 3:
        raise ValueError(f"augment_rotation expects x.shape[0] == 3, got {x.shape[0]}")
    
    # random unit axis
    axis = np.random.randn(3).astype(x.dtype)
    axis /= np.linalg.norm(axis) + 1e-8
    
    # random angle in [-π, π]
    angle = np.random.uniform(-np.pi, np.pi)
    
    # skew-symmetric cross-product matrix K
    K = np.array([
        [0,        -axis[2],  axis[1]],
        [axis[2],   0,       -axis[0]],
        [-axis[1],  axis[0],  0      ]
    ], dtype=x.dtype)
    
    # Rodrigues' rotation formula
    R = (
        np.eye(3, dtype=x.dtype) +
        np.sin(angle) * K +
        (1 - np.cos(angle)) * (K @ K)
    )
    
    return R @ x  # shape (3, T)


def augment_time_warp(x: np.ndarray, sigma: float = 0.2, n_knots: int = 4) -> np.ndarray:
    C, T = x.shape
    # control points evenly spaced in [0, T-1]
    knots = np.linspace(0, T - 1, n_knots)
    # warp factors around 1.0
    factors = np.random.normal(loc=1.0, scale=sigma, size=n_knots)
    spline = CubicSpline(knots, factors)
    
    # evaluate warp curve at each time step
    warp_curve = spline(np.arange(T)).astype(x.dtype)
    
    # broadcast and apply
    return (x * warp_curve).astype(x.dtype)


def augment_timenet(x: np.ndarray, max_warp: float = 0.2) -> np.ndarray:
    C, T = x.shape
    # sample warp factor
    warp = np.random.uniform(1 - max_warp, 1 + max_warp)
    new_T = max(5, int(round(T * warp)))
    
    # original and target time indices
    orig_idx = np.arange(T)
    target_idx = np.linspace(0, T - 1, new_T)
    
    # vectorized per-channel interpolation
    stretched = np.vstack([
        np.interp(target_idx, orig_idx, x_channel)
        for x_channel in x
    ]).astype(x.dtype)  # shape (C, new_T)
    
    # prepare output and fill
    out = np.zeros((C, T), dtype=x.dtype)
    if new_T >= T:
        out[:] = stretched[:, :T]
    else:
        out[:, :new_T] = stretched
        # pad remainder with last value of each channel
        out[:, new_T:] = stretched[:, -1, np.newaxis]
    
    return out



class WearDataset(Dataset):
    def __init__(self, dataframe, mean, std, is_train=True, has_label=True):
        self.df = dataframe.reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.is_train = is_train
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row['tensor'] # Shape (3, 50)
        # x_np = row['tensor'].astype(np.float32)

        if self.is_train:
            if random.random() < 0.4:
                x = augment_jitter(x)
            if random.random() < 0.6:
                x = augment_scale(x)
            if random.random() < 0.6:
                x = augment_rotation(x)
            if random.random() < 0.5:
                x = augment_timenet(x)
            # if random.random() < 0.3:
            #     x = augment_time_warp(x)
        x = torch.from_numpy(x.copy()).to(self.mean.device)
        x = (x - self.mean) / (self.std + 1e-8)

        # x = (x - self.mean) / self.std
        
        # label = row['label'] if self.has_label else -1
        # return x, torch.tensor(label, dtype=torch.long)

        if self.has_label and 'label' in row:
            label = int(row['label'])
        else:
            label = 0
        return x, torch.tensor(label, dtype=torch.long)
def make_loader(df, mean, std, is_train, has_label=True, shuffle=False, batch_size=CFG.batch_size):
    ds = WearDataset(df, mean, std, is_train=is_train, has_label=has_label)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=8 if torch.cuda.is_available() else 0,
                      persistent_workers=True if torch.cuda.is_available() else False,
                      pin_memory=True,
                      drop_last=shuffle)


import torch
import torch.nn as nn

class PatchEmbed1D(nn.Module):
    """
    Splits a 1D signal into non-overlapping patches and projects them to an embedding dimension.
    This acts as the first CNN layer.
    Input:  x of shape (B, C, T)
    Output: tensor of shape (B, L, D) where L = T // patch_size
    """
    def __init__(self, in_ch: int, dim: int, patch: int):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=in_ch,
            out_channels=dim,
            kernel_size=patch,
            stride=patch,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 1)


class LocationSpecificModel(nn.Module):

    def __init__(self, cfg: CFG, in_channels: int = 3):
        super().__init__()
        num_patches = cfg.window_size // cfg.patch_size

        self.patch = PatchEmbed1D(in_ch=in_channels, dim=cfg.vit_dim, patch=cfg.patch_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, cfg.vit_dim))
        nn.init.trunc_normal_(self.pos_emb, std=.02)

        self.lstm = nn.LSTM(
            input_size=cfg.vit_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        self.attn_pool = nn.Sequential(
            nn.Linear(cfg.lstm_hidden * 2, cfg.lstm_hidden * 2),
            nn.Tanh(),
            nn.Linear(cfg.lstm_hidden * 2, 1),
            nn.Softmax(dim=1)
        )

        merge_conv_out_dim = 256
        self.merge_conv = nn.Conv1d(
            in_channels=cfg.lstm_hidden * 2,
            out_channels=merge_conv_out_dim,
            kernel_size=1,  # Different from patch_size
            bias=True
        )

        self.head = nn.Sequential(
            nn.LayerNorm(merge_conv_out_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(merge_conv_out_dim, cfg.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.patch(x)
        x = x + self.pos_emb

        lstm_out, _ = self.lstm(x)

        attn_weights = self.attn_pool(lstm_out)
        pooled = torch.sum(lstm_out * attn_weights, dim=1)

        pooled_reshaped = pooled.unsqueeze(2)
        
        merged_out = self.merge_conv(pooled_reshaped)
        
        merged_flat = merged_out.squeeze(2)
        
        return self.head(merged_flat)




def run_epoch(model, loader, optim, loss_fn, meter, device):
    is_train = optim is not None
    model.train() if is_train else model.eval()
    
    running_loss = 0.0
    
    pbar = tqdm(loader, leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = loss_fn(logits, y)
            
            if is_train:
                optim.zero_grad()
                loss.backward()
                optim.step()
        
        running_loss += loss.item() * x.size(0)
        meter.update(logits.softmax(-1), y)
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = meter.compute().item()
    meter.reset()
    
    return epoch_loss, epoch_f1


# TESTING !!!!
models = {}
means = {}
stds = {}


for loc in SENSOR_LOCATIONS:
    print(f"===== Training model for {loc} =====")
    
    # Lọc dữ liệu cho vị trí hiện tại
    df_loc = df_windows[df_windows['sensor_location'] == loc].copy()
    
    all_tensors = np.stack(df_loc['tensor'].values) # Shape (N, 3, 50)
    mu  = all_tensors.mean(axis=(0, 2))            # (3,) 
    sig = all_tensors.std(axis=(0, 2)) + 1e-8

    mean = torch.from_numpy(mu).float().unsqueeze(1)
    std  = torch.from_numpy(sig).float().unsqueeze(1)

    means[loc] = mean
    stds[loc] = std
    
    # Chia dữ liệu
    all_sbj_ids = df_loc['sbj_id'].unique()
    train_sbj_ids = np.random.choice(all_sbj_ids, size=int(len(all_sbj_ids)*0.7), replace=False)
    val_sbj_ids = np.setdiff1d(all_sbj_ids, train_sbj_ids)
    
    df_train_loc = df_loc[df_loc['sbj_id'].isin(train_sbj_ids)]
    df_val_loc = df_loc[df_loc['sbj_id'].isin(val_sbj_ids)]
    
    train_loader = make_loader(df_train_loc, mean, std, is_train=True, shuffle=True)
    val_loader = make_loader(df_val_loc, mean, std, is_train=False, shuffle=False)
    

    model = LocationSpecificModel(CFG, in_channels=3).to(CFG.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    meter = MulticlassF1Score(num_classes=CFG.num_classes, average='macro').to(CFG.device)


    best_f1   = -1.0
    wait_cnt  = 0 
    patience  = 14
    for epoch in range(CFG.epochs):
        train_loss, train_f1 = run_epoch(model, train_loader, optimizer, loss_fn,
                                         meter, CFG.device)
    
        val_loss,   val_f1   = run_epoch(model, val_loader,   None,      loss_fn,
                                         meter, CFG.device)
    
    
        print(f"Epoch {epoch+1}/{CFG.epochs} │ "
              f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f} │ "
              f"Val Loss: {val_loss:.4f},   F1: {val_f1:.4f}")
    
        if val_f1 > best_f1:
            best_f1  = val_f1
            wait_cnt = 0                         
            torch.save(model.state_dict(), f"best_model_{loc}.pth")
            print(f"  ↳ New best for {loc}: F1 = {best_f1:.4f}  (model saved)")
        else:                   
            wait_cnt += 1
            if wait_cnt >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience reached).")
                break
    # ------------------------------------
    
    # nạp lại trọng số tốt nhất trước khi đánh giá / lưu
    model.load_state_dict(torch.load(f"best_model_{loc}.pth"))
    models[loc] = model.eval()





import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]

all_true_labels = []
all_predictions = []



def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    

print("------------------------------------------------------------------")

seed_everything(CFG.seed) 

for loc in SENSOR_LOCATIONS:
    print(f"\n===== For sensor: {loc.upper()} =====")
    
    model = LocationSpecificModel(CFG, in_channels=3)
    model.load_state_dict(torch.load(f"best_model_{loc}.pth"))
    model.to(CFG.device)
    model.eval()
    
    mean = means[loc]
    std = stds[loc]
    
    df_loc = df_windows[df_windows['sensor_location'] == loc].copy()
    all_sbj_ids = df_loc['sbj_id'].unique()
    train_sbj_ids = np.random.choice(all_sbj_ids, size=int(len(all_sbj_ids)*0.8), replace=False)
    val_sbj_ids = np.setdiff1d(all_sbj_ids, train_sbj_ids)
    df_val_loc = df_loc[df_loc['sbj_id'].isin(val_sbj_ids)]

    val_loader = make_loader(df_val_loc, mean, std, is_train=False, shuffle=False)
    
    loc_true_labels = []
    loc_predictions = []
    
    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_loader, desc=f"Dự đoán trên tập Val ({loc})"):
            x_batch = x_batch.to(CFG.device)
            logits = model(x_batch)
            preds = logits.argmax(1)
            
            loc_true_labels.extend(y_batch.cpu().numpy())
            loc_predictions.extend(preds.cpu().numpy())
            
    all_true_labels.extend(loc_true_labels)
    all_predictions.extend(loc_predictions)
            
    plot_confusion_matrix(loc_true_labels, loc_predictions, class_names, 
                          title=f'{loc.replace("_", " ").title()}')


print("\n------------------------------------------------------------------")
print("===== All sensor =====")
plot_confusion_matrix(all_true_labels, all_predictions, class_names, 
                      title='All sensor')


predictions = []
for _, row in tqdm(df_test_raw.iterrows(),
                   total=len(df_test_raw), desc="Predicting"):

    loc    = row['sensor_location']
    model  = models[loc]
    mean   = means[loc].float()          # (3,1)
    std    = stds[loc].float()

    x_test   = np.stack([row['x_axis'],
                       row['y_axis'],
                       row['z_axis']], dtype=np.float32)

    x      = torch.from_numpy(x_test).unsqueeze(0)        # (1,3,50)
    x      = (x - mean) / (std + 1e-8)
    x      = x.to(CFG.device, dtype=torch.float32)

    with torch.no_grad():
        pred = model(x).argmax(1).item()
    predictions.append(pred)

df_submission = pd.DataFrame({'id': df_test_raw['id'], 'target_value': predictions})
df_submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")
df_submission.head()


print(model)

