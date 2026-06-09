# ğŸ“¦ Install Dependencies (CPU/GPU-safe)
!pip -q install numpy pandas scikit-learn matplotlib torch pytorch-lightning
print("âœ… Dependencies ready")


import os, math, random, gc, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('ğŸ”¥ Using', DEVICE)


DATA_DIR = "/kaggle/input/m5-forecasting-accuracy"
calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv")
prices = pd.read_csv(f"{DATA_DIR}/sell_prices.csv")
sales = pd.read_csv(f"{DATA_DIR}/sales_train_validation.csv")
print(calendar.shape, prices.shape, sales.shape)


# --- Parameters ---
N_ITEMS = 100      # smaller sample for faster demo
PRED_DAYS = 28     # forecast horizon
SEED = 42          # random seed for reproducibility

# --- Column setup ---
id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
d_cols = [c for c in sales.columns if c.startswith('d_')]

# --- Sample a subset of items to speed up experimentation ---
sales_small = sales.sample(N_ITEMS, random_state=SEED).reset_index(drop=True)

print(f"Subset shape: {sales_small.shape}")


# Melt to long
df_long = sales_small.melt(id_vars=id_cols, value_vars=d_cols, var_name='d', value_name='sales')

# Merge with calendar and prices
df = df_long.merge(calendar[['d','date','wm_yr_wk','wday','month','event_name_1']],
                   how='left', on='d')
df = df.merge(prices, on=['store_id','item_id','wm_yr_wk'], how='left')

# Convert types and fill missing values
df['date'] = pd.to_datetime(df['date'])
df['wday'] = df['wday'].astype(int)
df['month'] = df['month'].astype(int)
df['event_name_1'] = df['event_name_1'].fillna('None')

# Sort and create lags/rollings per (id)
df = df.sort_values(['id','date']).reset_index(drop=True)

def add_lags_rollings(g):
    g = g.copy()
    for lag in [1,7,28]:
        g[f'lag_{lag}'] = g['sales'].shift(lag)
    g['rolling_7'] = g['sales'].shift(1).rolling(7).mean()
    g['rolling_28'] = g['sales'].shift(1).rolling(28).mean()
    return g

df = df.groupby('id', group_keys=False).apply(add_lags_rollings)

# Drop initial NaNs from lags
df = df.dropna().reset_index(drop=True)
print('After FE:', df.shape)
df.head(3)



ENC_LEN = 56
CONT_FEATS = ['sell_price','lag_1','lag_7','lag_28','rolling_7','rolling_28']
CAT_FEATS  = ['wday','month','event_name_1','store_id','dept_id','cat_id']

# Encode categories to ints
cat_maps = {}
for c in CAT_FEATS:
    cat_maps[c] = {v:i for i,v in enumerate(df[c].astype(str).unique())}
    df[c+'_idx'] = df[c].astype(str).map(cat_maps[c])
CAT_IDX = [c+'_idx' for c in CAT_FEATS]

# Scale continuous features
scaler = StandardScaler()
df[CONT_FEATS] = scaler.fit_transform(df[CONT_FEATS])

# Build per-id time series tensors
def build_id_windows(g):
    X_enc_cont, X_enc_cat, X_dec_cat, Y = [], [], [], []
    g = g.sort_values('date')
    arr_cont = g[CONT_FEATS].values
    arr_cat  = g[CAT_IDX].values
    y = g['sales'].values
    for t in range(ENC_LEN, len(g)-PRED_DAYS):
        enc_cont = arr_cont[t-ENC_LEN:t]
        enc_cat  = arr_cat[t-ENC_LEN:t]
        dec_cat  = arr_cat[t:t+PRED_DAYS]  # future known cats (calendar-like)
        target   = y[t:t+PRED_DAYS]
        X_enc_cont.append(enc_cont)
        X_enc_cat.append(enc_cat)
        X_dec_cat.append(dec_cat)
        Y.append(target)
    return (
        np.array(X_enc_cont, dtype=np.float32),
        np.array(X_enc_cat, dtype=np.int64),
        np.array(X_dec_cat, dtype=np.int64),
        np.array(Y, dtype=np.float32)
    )

data = [build_id_windows(g) for _,g in df.groupby('id')]
if not data: raise RuntimeError('No sequences produced. Try increasing N_ITEMS or reducing ENC_LEN.')
Xec = np.concatenate([d[0] for d in data])
Xea = np.concatenate([d[1] for d in data])
Xdc = np.concatenate([d[2] for d in data])
Y   = np.concatenate([d[3] for d in data])
print('Shapes enc_cont, enc_cat, dec_cat, target:', Xec.shape, Xea.shape, Xdc.shape, Y.shape)

Xec_tr, Xec_va, Xea_tr, Xea_va, Xdc_tr, Xdc_va, Y_tr, Y_va = train_test_split(
    Xec, Xea, Xdc, Y, test_size=0.2, random_state=SEED
)
print('Train windows:', len(Y_tr), 'Valid windows:', len(Y_va))


class GatedLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.gate = nn.Linear(d_in, d_out)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.fc(x) * self.sig(self.gate(x))

class VariableSelector(nn.Module):
    # Simple variable selection: per-feature gate to weight cont feats
    def __init__(self, n_feats, d_model):
        super().__init__()
        self.proj = nn.ModuleList([GatedLinear(1, d_model) for _ in range(n_feats)])
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):  # (B,T,F)
        B,T,F = x.size()
        outs = []
        for i in range(F):
            outs.append(self.proj[i](x[:,:,i:i+1]))
        H = torch.stack(outs, dim=2)  # (B,T,F,d)
        weights = self.softmax(H.mean(dim=1))  # (B,F,d) -> poor-man selector
        # Apply weights per feature (broadcast)
        Hw = H * weights.unsqueeze(1)
        return Hw.sum(dim=2)  # (B,T,d)

class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_ff=256, p=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=p, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Dropout(p), nn.Linear(dim_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        h,_ = self.attn(x,x,x)
        x = self.ln1(x + self.drop(h))
        h = self.ff(x)
        x = self.ln2(x + self.drop(h))
        return x

class TFTLite(pl.LightningModule):
    def __init__(self, n_cat_maps, n_cont, pred_days=28, d_model=128, p=0.1, lr=2e-3):
        super().__init__()
        self.save_hyperparameters()
        self.pred_days = pred_days
        self.lr = lr
        # Categorical embeddings (sum)
        self.cat_embs = nn.ModuleList([nn.Embedding(num, d_model) for num in n_cat_maps])
        # Variable selection for continuous
        self.varsel = VariableSelector(n_cont, d_model)
        self.enc = nn.Sequential(SimpleTransformerBlock(d_model,4,256,p), SimpleTransformerBlock(d_model,4,256,p))
        self.dec = nn.Sequential(SimpleTransformerBlock(d_model,4,256,p))
        self.head = nn.Linear(d_model, 3)  # quantiles P10,P50,P90
        self.quantiles = torch.tensor([0.1,0.5,0.9], dtype=torch.float32)
    def quantile_loss(self, preds, target):
        # preds: (B,H,3) target: (B,H)
        q = self.quantiles.to(preds.device).view(1,1,3)
        e = target.unsqueeze(-1) - preds
        return torch.mean(torch.max(q*e, (q-1)*e))
    def forward(self, enc_cont, enc_cat, dec_cat):
        # enc_cont: (B,T,Fc), enc_cat: (B,T,Fk), dec_cat: (B,H,Fk)
        B,T,Fc = enc_cont.size()
        H = dec_cat.size(1)
        # cont selector -> (B,T,d)
        h_cont = self.varsel(enc_cont)
        # cat sum emb -> (B,T,d)
        def emb_cat(arr):
            # arr: (B,L,Fk)
            embs = [emb(arr[:,:,i]) for i,emb in enumerate(self.cat_embs)]
            return torch.stack(embs, dim=0).sum(0)
        h_enc = h_cont + emb_cat(enc_cat)
        h_enc = self.enc(h_enc)
        h_dec = emb_cat(dec_cat)
        # cross-attend by concatenating and letting MHAttention mix via dec block
        h = self.dec(torch.cat([h_enc[:,-1:].repeat(1,H,1), h_dec], dim=1))[:, -H:, :]
        out = self.head(h)
        return out  # (B,H,3)
    def training_step(self, batch, batch_idx):
        enc_cont, enc_cat, dec_cat, y = batch
        qhat = self(enc_cont, enc_cat, dec_cat)
        loss = self.quantile_loss(qhat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        enc_cont, enc_cat, dec_cat, y = batch
        qhat = self(enc_cont, enc_cat, dec_cat)
        loss = self.quantile_loss(qhat, y)
        self.log('val_loss', loss, prog_bar=True)
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

class TSDS(Dataset):
    def __init__(self, Xec, Xea, Xdc, Y):
        self.Xec = torch.from_numpy(Xec)
        self.Xea = torch.from_numpy(Xea)
        self.Xdc = torch.from_numpy(Xdc)
        self.Y   = torch.from_numpy(Y)
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, i):
        return self.Xec[i], self.Xea[i], self.Xdc[i], self.Y[i]

train_ds = TSDS(Xec_tr, Xea_tr, Xdc_tr, Y_tr)
valid_ds = TSDS(Xec_va, Xea_va, Xdc_va, Y_va)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=256, shuffle=False, num_workers=2)

n_cat_maps = [len(cat_maps[c]) for c in CAT_FEATS]
model = TFTLite(n_cat_maps, n_cont=len(CONT_FEATS), pred_days=PRED_DAYS)

ckpt = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
trainer = pl.Trainer(max_epochs=10, accelerator='auto', callbacks=[ckpt, es], log_every_n_steps=20)
trainer.fit(model, train_loader, valid_loader)
print('âœ… Training done. Best:', ckpt.best_model_path)


best = TFTLite.load_from_checkpoint(ckpt.best_model_path, n_cat_maps=n_cat_maps, n_cont=len(CONT_FEATS), pred_days=PRED_DAYS)
best.eval().to(DEVICE)
batch = next(iter(valid_loader))
with torch.no_grad():
    preds = best(batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)).cpu().numpy()  # (B,H,3)
    ytrue = batch[3].numpy()

b = 0
p10, p50, p90 = preds[b,:,0], preds[b,:,1], preds[b,:,2]
plt.figure(figsize=(10,4))
plt.plot(ytrue[b], label='True')
plt.plot(p50, label='P50')
plt.fill_between(np.arange(len(p50)), p10, p90, alpha=0.2, label='P10-P90')
plt.title('Fan-chart Forecast')
plt.legend(); plt.show()
print('âœ¨ Tip: Increase N_ITEMS/EPOCHS for better accuracy.')

