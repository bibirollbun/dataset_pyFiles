import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import gc
import random
import warnings

warnings.filterwarnings('ignore')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


class Config:
    data_path = '/kaggle/input/stanford-covid-vaccine/'
    
    # Quan trọng: 130 để cover cả Private Test
    seq_len = 130
    pred_len = 68  # Vị trí được chấm điểm (Scored positions)
    target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
    
    # Model Params
    embed_dim = 128
    hidden_dim = 384  # Tăng hidden size
    num_layers = 3
    dropout = 0.4     # Tăng dropout để chống lại augmentation mạnh
    bidirectional = True
    
    # Training
    batch_size = 16   # Giảm batch size do model lớn hơn
    epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-4
    n_folds = 5
    
    # Augmentation Params
    aug_prob = 0.5
    mask_prob = 0.1   # Tỷ lệ che features
    bpp_dropout = 0.1 # Tỷ lệ drop kết nối trong BPP

config = Config()


# Dictionary Mapping
token2int = {x: i for i, x in enumerate('().AUGC')}
loop2int = {x: i for i, x in enumerate('SMIBHEX')}


def calculate_metrics_dict(y_true, y_pred):
    """
    Tính toán 4 chỉ số yêu cầu.
    Lưu ý: y_true và y_pred phải là numpy array đã được flatten hoặc đúng shape.
    """
    # 1. MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # 2. RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 3. R2 Score
    r2 = r2_score(y_true, y_pred)
    
    # 4. NSE (Nash-Sutcliffe Efficiency)
    # Công thức NSE giống R2 trong ngữ cảnh này: 1 - (SS_res / SS_tot)
    # Tuy nhiên, ta code thủ công để đảm bảo logic
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8
    nse = 1 - (numerator / denominator)
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'NSE': nse}


train_df = pd.read_json(config.data_path + 'train.json', lines=True)
test_df = pd.read_json(config.data_path + 'test.json', lines=True)
sample_sub = pd.read_csv(config.data_path + 'sample_submission.csv')


# Filter Noisy Data (Best practice)
train_df = train_df[train_df['signal_to_noise'] > 1.0].reset_index(drop=True)

# Load BPP Matrix (Quan trọng cho Feature Engineering)
print("Loading BPP Matrices...")
bpp_dict = {}
all_ids = pd.concat([train_df['id'], test_df['id']]).unique()
for seq_id in all_ids:
    path = os.path.join(config.data_path, 'bpps', seq_id + '.npy')
    if os.path.exists(path):
        bpp_dict[seq_id] = np.load(path)


class RNAAugmenter:
    def __init__(self, config):
        self.config = config
        
    def __call__(self, seq_ints, cont_feats, targets=None, sn_ratio=None):
        # Feature Masking
        if np.random.rand() < self.config.aug_prob:
            mask = np.random.rand(self.config.seq_len) < self.config.mask_prob
            seq_ints[mask, :] = 0 
            
        # Continuous Features Perturbation
        if np.random.rand() < self.config.aug_prob:
            # Thêm nhiễu nhẹ vào cả BPP và Distance features
            noise = np.random.normal(0, 0.02, cont_feats.shape)
            cont_feats = cont_feats + noise
            
        # Target Noise (giữ nguyên)
        if targets is not None and sn_ratio is not None and np.random.rand() < self.config.aug_prob:
            safe_sn = max(float(sn_ratio), 1e-5)
            scale = 0.05 * (1.0 / safe_sn)
            noise = np.random.normal(0, scale, targets.shape)
            targets = targets + noise
            
        return seq_ints, cont_feats, targets

augmenter = RNAAugmenter(config)



# =============================================================================
# FEATURE ENGINEERING HELPER FUNCTIONS
# =============================================================================
def get_bpp_features(bpp_matrix, seq_len):
    # 1. Resize (như cũ)
    curr_len = bpp_matrix.shape[0]
    if curr_len < seq_len:
        pad_amt = seq_len - curr_len
        bpp_matrix = np.pad(bpp_matrix, ((0, pad_amt), (0, pad_amt)), 'constant')
    else:
        bpp_matrix = bpp_matrix[:seq_len, :seq_len]
        
    # 2. Extract Basic Features
    bpp_max = np.max(bpp_matrix, axis=1)
    bpp_sum = np.sum(bpp_matrix, axis=1)
    bpp_nb  = np.sum(bpp_matrix > 0, axis=1)
    
    # 3. NEW: Shannon Entropy
    # Tránh log(0) bằng cách cộng epsilon
    entropy = -np.sum(bpp_matrix * np.log2(bpp_matrix + 1e-10), axis=1)
    
    # Feature 5: Difference between max and sum (measure of promiscuity)
    # Nếu Max gần bằng Sum -> Base này rất chung thủy (chỉ bắt cặp với 1 thằng) -> Bền
    promiscuity = bpp_sum - bpp_max
    
    # Trả về [seq_len, 5] (Thêm Entropy và Promiscuity)
    return np.stack([bpp_max, bpp_sum, bpp_nb, entropy, promiscuity], axis=1)

def get_distance_to_pair(structure, seq_len):
    """
    Biến đổi Structure Adjacency thành Distance Feature cho LSTM.
    Thay vì trả về ma trận N*N, ta trả về mảng N*1:
    - Nếu vị trí i bắt cặp với j: giá trị là chiều (j - i) / seq_len
    - Nếu không bắt cặp: giá trị là -1
    """
    # Khởi tạo mảng feature
    pair_dist = np.full(seq_len, -1.0, dtype=np.float32)
    
    stack = []
    for i, char in enumerate(structure):
        if i >= seq_len: break # Safety check
        
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start_idx = stack.pop()
                # Tính khoảng cách chuẩn hóa
                dist = (i - start_idx) / seq_len
                pair_dist[start_idx] = dist # Chiều xuôi
                pair_dist[i] = -dist        # Chiều ngược
                
    return pair_dist.reshape(-1, 1) # [seq_len, 1]


class RNADataset(Dataset):
    def __init__(self, df, mode='train', augment=False):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_seq_len = len(row['sequence'])
        
        # --- Basic Indices (Giữ nguyên) ---
        seq = [token2int.get(x, 0) for x in row['sequence']]
        struct = [token2int.get(x, 0) for x in row['structure']]
        loop = [loop2int.get(x, 0) for x in row['predicted_loop_type']]
        
        pad_len = config.seq_len - raw_seq_len
        seq += [0] * pad_len
        struct += [0] * pad_len
        loop += [0] * pad_len
        inputs_cat = np.stack([seq, struct, loop], axis=1)
        
        # --- Advanced Continuous Features ---
        # 1. BPP Features (Đã có thêm Entropy)
        bpp_mat = bpp_dict.get(row['id'], np.zeros((raw_seq_len, raw_seq_len)))
        bpp_feats = get_bpp_features(bpp_mat, config.seq_len) # [130, 5]
        
        # 2. Distance to Pair (Feature cũ)
        dist_feat = get_distance_to_pair(row['structure'], config.seq_len) # [130, 1]
        
        # 3. NEW: Relative Positional Encoding
        # Khoảng cách đến đầu (normalize) và khoảng cách đến cuối (normalize)
        pos = np.arange(config.seq_len)
        dist_5prime = pos / config.seq_len # Khoảng cách đến đầu 5'
        dist_3prime = (config.seq_len - pos - 1) / config.seq_len # Khoảng cách đến đầu 3'
        pos_feats = np.stack([dist_5prime, dist_3prime], axis=1) # [130, 2]
        
        # Gộp tất cả continuous features
        # Tổng: 5 (BPP) + 1 (PairDist) + 2 (Pos) = 8 features
        inputs_cont = np.concatenate([bpp_feats, dist_feat, pos_feats], axis=1)
        
        targets = np.zeros((config.seq_len, 5))
        loss_mask = np.zeros(config.seq_len)
        sample_weight = 1.0 # Mặc định
        
        if self.mode == 'train':
            t = np.array([row[c] for c in config.target_cols]).T
            targets[:len(t)] = t
            loss_mask[:len(t)] = 1.0
            
            # NEW: Tính Sample Weight dựa trên log(Signal_to_Noise)
            # Log giúp giảm sự chênh lệch quá lớn giữa SN=10 và SN=1
            sn = float(row['signal_to_noise'])
            sample_weight = np.log1p(sn) # log(1 + sn)
            
            if self.augment:
                inputs_cat, inputs_cont, targets = augmenter(
                    inputs_cat, inputs_cont, targets, row['signal_to_noise']
                )
        
        return (torch.LongTensor(inputs_cat), 
                torch.FloatTensor(inputs_cont), 
                torch.FloatTensor(targets), 
                torch.FloatTensor(loss_mask),
                torch.FloatTensor([sample_weight])) # Trả về thêm weight


class EncoderDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.seq_emb = nn.Embedding(7, config.embed_dim)
        self.struct_emb = nn.Embedding(7, config.embed_dim)
        self.loop_emb = nn.Embedding(7, config.embed_dim)
        
        # CẬP NHẬT INPUT DIMENSION
        # 3 Embeddings + 3 BPP Features + 1 Distance Feature = 4 Continuous
        self.in_dim = (config.embed_dim * 3) + 8 # lên 8
        
        # Phần còn lại giữ nguyên
        self.encoder = nn.LSTM(
            self.in_dim, config.hidden_dim, 
            num_layers=config.num_layers, 
            dropout=config.dropout, 
            bidirectional=True, batch_first=True
        )
        
        # Attention Layer (Dot-product attention)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim * 2, 
            num_heads=8, 
            dropout=config.dropout,
            batch_first=True
        )
        
        # Decoder (Bi-LSTM) - Refines the representation
        # Trong bài toán này input length == output length, nên decoder
        # đóng vai trò là lớp xử lý hậu kỳ (post-processing) sâu hơn.
        self.decoder = nn.LSTM(
            config.hidden_dim * 2, config.hidden_dim,
            num_layers=2, # Decoder nhẹ hơn encoder chút
            dropout=config.dropout,
            bidirectional=True, batch_first=True
        )
        
        # Output Head
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 5)
        )
        
    def forward(self, cat_feats, bpp_feats):
        # cat_feats: [B, 130, 3], bpp_feats: [B, 130, 3]
        
        # 1. Embeddings
        e1 = self.seq_emb(cat_feats[:,:,0])
        e2 = self.struct_emb(cat_feats[:,:,1])
        e3 = self.loop_emb(cat_feats[:,:,2])
        
        # Concatenate features
        x = torch.cat([e1, e2, e3, bpp_feats], dim=2)
        
        # 2. Encoder
        enc_out, _ = self.encoder(x) # [B, 130, hidden*2]
        
        # 3. Attention (Self-Attention on Encoder Output)
        attn_out, _ = self.attention(enc_out, enc_out, enc_out)
        
        # Residual Connection
        x = enc_out + attn_out
        
        # 4. Decoder
        dec_out, _ = self.decoder(x)
        
        # 5. Output
        out = self.head(dec_out)
        return out



def mcrmse_loss(pred, target, mask):
    # Expand mask [B, 130] -> [B, 130, 5]
    mask = mask.unsqueeze(-1).expand_as(target)
    
    # Calculate MSE only on masked region
    loss = (pred - target) ** 2
    loss = loss * mask
    
    # Mean per column
    loss = torch.sum(loss, dim=1) / (torch.sum(mask, dim=1) + 1e-8)
    rmse = torch.sqrt(loss + 1e-8)
    
    return torch.mean(rmse)


def weighted_mcrmse_loss(pred, target, mask, weights):
    """
    pred, target: [B, 130, 5]
    mask: [B, 130]
    weights: [B, 1]  <-- Input từ DataLoader có shape này
    """
    # Expand mask: [B, 130] -> [B, 130, 5]
    mask = mask.unsqueeze(-1).expand_as(target)
    
    # --- SỬA LỖI TẠI ĐÂY ---
    # Input weights là [B, 1]. Ta chỉ cần thêm 1 chiều nữa để thành [B, 1, 1]
    # Tensor [B, 1, 1] sẽ broadcast chuẩn với [B, 130, 5]
    weights = weights.unsqueeze(-1) 
    # -----------------------
    
    # Tính lỗi bình phương
    loss = (pred - target) ** 2
    loss = loss * mask
    
    # Nhân với trọng số (Broadcasting: [B, 130, 5] * [B, 1, 1])
    loss = loss * weights 
    
    # Tính Mean theo cột
    # Cộng thêm epsilon vào mẫu số để tránh chia cho 0
    loss_per_col = torch.sum(loss, dim=1) / (torch.sum(mask, dim=1) + 1e-8)
    rmse = torch.sqrt(loss_per_col + 1e-8)
    
    return torch.mean(rmse)


kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)

# Containers for results
oof_preds = np.zeros((len(train_df), config.pred_len, 5))
test_preds_accum = []

for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_df)):
    print(f"\n{'#'*30}")
    print(f"FOLD {fold+1}/{config.n_folds}")
    
    # Datasets
    train_ds = RNADataset(train_df.iloc[train_idx], mode='train', augment=True)
    valid_ds = RNADataset(train_df.iloc[valid_idx], mode='train', augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)
    
    # Model Setup
    model = EncoderDecoderModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, 
        steps_per_epoch=len(train_loader), epochs=config.epochs,
        pct_start=0.1
    )
    scaler = torch.cuda.amp.GradScaler()
    
    best_loss = float('inf')
    
    for epoch in range(config.epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        for cat, bpp, tgt, mask, w in train_loader:
            cat, bpp, tgt, mask, w = cat.to(device), bpp.to(device), tgt.to(device), mask.to(device), w.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(cat, bpp)
                loss = weighted_mcrmse_loss(out, tgt, mask, w)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
        
        # --- VALIDATE ---
        model.eval()
        valid_loss = 0
        val_preds_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for cat, bpp, tgt, mask, w in valid_loader:
                cat, bpp, tgt, mask = cat.to(device), bpp.to(device), tgt.to(device), mask.to(device)
                
                out = model(cat, bpp)
                # Valid dùng hàm loss gốc (không weight) để so sánh công bằng
                loss = mcrmse_loss(out, tgt, mask)
                valid_loss += loss.item()
                
                # Store for metrics (Only take first 68 positions)
                # mask[0] has 1s for first 68, 0s for rest
                # We simply slice manually to be safe
                val_preds_list.append(out[:, :config.pred_len, :].cpu().numpy())
                val_targets_list.append(tgt[:, :config.pred_len, :].cpu().numpy())
                
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        
        # Calculate Advanced Metrics (Epoch Level)
        val_p = np.concatenate(val_preds_list, axis=0).flatten() # Flatten all
        val_t = np.concatenate(val_targets_list, axis=0).flatten()
        metrics = calculate_metrics_dict(val_t, val_p)
        
        if (epoch+1) % 10 == 0:
            print(f"Ep {epoch+1} | Loss: {train_loss:.4f} / {valid_loss:.4f} | "
                  f"MAE: {metrics['MAE']:.4f} | R2: {metrics['R2']:.4f} | NSE: {metrics['NSE']:.4f}")
            
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'model_fold{fold}.pth')

    # --- OOF PREDICTION ---
    model.load_state_dict(torch.load(f'model_fold{fold}.pth'))
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(valid_idx):
            # Trick to get single item batch
            ds_item = RNADataset(train_df.iloc[[idx]], mode='train')
            c, b, _, _, _ = ds_item[0]
            c, b = c.unsqueeze(0).to(device), b.unsqueeze(0).to(device)
            p = model(c, b).cpu().numpy()[0]
            oof_preds[idx] = p[:config.pred_len] # Save only 68
            
    # --- TEST PREDICTION ---
    test_ds = RNADataset(test_df, mode='test')
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    preds_fold = []
    with torch.no_grad():
        for cat, bpp, _, _, _ in test_loader:
            cat, bpp = cat.to(device), bpp.to(device)
            out = model(cat, bpp)
            preds_fold.append(out.cpu().numpy())
    test_preds_accum.append(np.concatenate(preds_fold, axis=0))


print("\nFinal OOF Evaluation:")
y_true_all = []
y_pred_all = []

for idx in range(len(train_df)):
    for i, col in enumerate(config.target_cols):
        y_true_all.extend(train_df.iloc[idx][col][:config.pred_len])
        y_pred_all.extend(oof_preds[idx, :, i])

final_metrics = calculate_metrics_dict(np.array(y_true_all), np.array(y_pred_all))
print(f"Overall MAE:  {final_metrics['MAE']:.4f}")
print(f"Overall RMSE: {final_metrics['RMSE']:.4f}")
print(f"Overall R2:   {final_metrics['R2']:.4f}")
print(f"Overall NSE:  {final_metrics['NSE']:.4f}")


print("\nGenerating Submission...")
final_test_preds = np.mean(test_preds_accum, axis=0) # [N_test, 130, 5]

# Build Dictionary
pred_dict = {}
for i, uid in enumerate(test_df['id']):
    pred_dict[uid] = final_test_preds[i]

# Map to sample submission
sub_df = sample_sub.copy()
id_seqpos = sub_df['id_seqpos'].values
mapped_preds = np.zeros((len(sub_df), 5))

for i, val in enumerate(id_seqpos):
    # Split: id_00073f8be_0 -> id, 00073f8be, 0
    # Safe splitting based on last underscore
    last_us = val.rfind('_')
    seq_id = val[:last_us]
    pos = int(val[last_us+1:])
    
    if seq_id in pred_dict:
        mapped_preds[i] = pred_dict[seq_id][pos]

sub_df[config.target_cols] = mapped_preds
sub_df.to_csv('submission.csv', index=False)
print("Submission saved successfully!")




