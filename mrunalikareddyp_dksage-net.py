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


# ================================================================
# RSNA ICH – DK-SageNet (Dynamic Knowledge GraphSAGE) + KG
# SINGLE CELL – Graph construction integrated – Stable + Better Acc
# ================================================================
import os, random
import numpy as np
import pandas as pd
import cv2, pydicom

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, cohen_kappa_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ---------------- CONFIG ----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

BASE = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection"
CSV_PATH = os.path.join(BASE, "stage_2_train.csv")
IMG_DIR  = os.path.join(BASE, "stage_2_train")

IMG_SIZE = 256
SAMPLES_PER_CLASS = 500     # raise if GPU
BATCH = 16
EMB_BATCH = 16

PCA_DIM = 256
K_NEIGH = 12

EPOCHS_EFF = 2
EPOCHS_SAGE = 50
PATIENCE = 8

HIDDEN = 256
DROP = 0.30
LABEL_SMOOTH = 0.03

# ================================================================
# 1) CT triple-window -> 3ch
# ================================================================
def window(img, wl, ww):
    lo, hi = wl - ww/2, wl + ww/2
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)

def load_dcm_3ch(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    ch1 = cv2.resize(window(img, 40, 80), (IMG_SIZE, IMG_SIZE))
    ch2 = cv2.resize(window(img, 80, 200), (IMG_SIZE, IMG_SIZE))
    ch3 = cv2.resize(window(img, 600, 2800), (IMG_SIZE, IMG_SIZE))
    return np.stack([ch1, ch2, ch3], 0)  # (3,H,W)

# ================================================================
# 2) Load CSV -> pivot -> binary -> balanced
# ================================================================
df = pd.read_csv(CSV_PATH)
df["Image"]   = df["ID"].apply(lambda x: x.split("_")[1])
df["Subtype"] = df["ID"].apply(lambda x: x.split("_")[2])

df_g = df.groupby(["Image","Subtype"], as_index=False)["Label"].max()
df_p = df_g.pivot(index="Image", columns="Subtype", values="Label").fillna(0.0)
df_p["Label_binary"] = df_p.max(axis=1).astype(int)

subtype_cols = list(df_p.columns[:-1])
C = len(subtype_cols)

parts = []
for lbl in [0, 1]:
    sub = df_p[df_p["Label_binary"] == lbl]
    parts.append(sub.sample(min(SAMPLES_PER_CLASS, len(sub)), random_state=SEED))
df_bal = pd.concat(parts).reset_index()           # includes Image column
df_indexed = df_bal.set_index("Image")
B_full = df_bal[subtype_cols].values.astype(np.float32)

print("Balanced counts:", df_bal["Label_binary"].value_counts().to_dict())
print("Concepts:", subtype_cols)

# ================================================================
# 3) Dataset
# ================================================================
class RSNADataset(Dataset):
    def __init__(self, df_subset):
        self.df = df_subset.reset_index(drop=True)
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),   # float32
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img_id = r["Image"]
        y = int(r["Label_binary"])
        path = os.path.join(IMG_DIR, f"ID_{img_id}.dcm")
        img3 = load_dcm_3ch(path)                       # (3,H,W) float32 [0..1]
        img3 = (img3 * 255).astype(np.uint8).transpose(1,2,0)  # (H,W,3)
        x = self.tf(img3)
        return x, y, img_id

# ================================================================
# 4) Splits
# ================================================================
train_df, temp = train_test_split(
    df_bal, test_size=0.3, stratify=df_bal["Label_binary"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp, test_size=0.5, stratify=temp["Label_binary"], random_state=SEED
)

train_loader = DataLoader(RSNADataset(train_df), batch_size=BATCH, shuffle=True, num_workers=2)

# ================================================================
# 5) EfficientNet short warmup
# ================================================================
eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, 2)
eff = eff.to(DEVICE)

opt_eff = torch.optim.AdamW(eff.parameters(), lr=3e-4, weight_decay=1e-5)
crit_eff = nn.CrossEntropyLoss()

print("\n=== Stage 1: EfficientNet warmup ===")
for ep in range(1, EPOCHS_EFF + 1):
    eff.train()
    for x, y, _ in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt_eff.zero_grad()
        loss = crit_eff(eff(x), y)
        loss.backward()
        opt_eff.step()
    print(f"Eff Ep {ep}/{EPOCHS_EFF} done")

class Embed(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.f = m.features
        self.a = m.avgpool
        self.drop = m.classifier[0]
    def forward(self, x):
        x = self.f(x)
        x = self.a(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return x

embedder = Embed(eff).to(DEVICE).eval()

def extract_embeddings(df_subset):
    dl = DataLoader(RSNADataset(df_subset), batch_size=EMB_BATCH, shuffle=False, num_workers=2)
    X, ids = [], []
    with torch.no_grad():
        for x, _, ids_b in dl:
            z = embedder(x.to(DEVICE)).cpu().numpy()
            X.append(z)
            ids += list(ids_b)
    return np.vstack(X).astype(np.float32), ids

print("\n=== Stage 2: Extracting embeddings ===")
X_tr, ids_tr   = extract_embeddings(train_df)
X_val, ids_val = extract_embeddings(val_df)
X_te, ids_te   = extract_embeddings(test_df)

pca = PCA(PCA_DIM, random_state=SEED)
X_tr = pca.fit_transform(X_tr).astype(np.float32)
X_val = pca.transform(X_val).astype(np.float32)
X_te  = pca.transform(X_te).astype(np.float32)

# ================================================================
# 6) Graph construction: image-image kNN + image-concept + PMI KG
#    + self-loops + symmetric norm (float32)
# ================================================================
def build_graph(X, ids):
    N = X.shape[0]

    # image-image kNN cosine graph
    sim = cosine_similarity(X).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    sim = sim ** 3

    A_ii = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        k = min(K_NEIGH, N - 1)
        if k <= 0: continue
        nbrs = np.argsort(sim[i])[-k:]
        A_ii[i, nbrs] = sim[i, nbrs]
    A_ii = np.minimum(A_ii, A_ii.T)

    # image-concept incidence
    B = df_indexed.loc[ids, subtype_cols].values.astype(np.float32)  # (N,C)
    B = B / (B.sum(1, keepdims=True) + 1e-6)

    # concept-concept PMI (KG)
    p = B_full.mean(0) + 1e-9
    co = (B_full.T @ B_full).astype(np.float32)
    pmi = np.log((co + 1e-6) / (p[:, None] * p[None, :])).astype(np.float32)
    pmi[pmi < 0] = 0.0

    # combined adjacency
    A = np.block([[A_ii, B],
                  [B.T, pmi]]).astype(np.float32)

    # self-loops
    A = A + np.eye(A.shape[0], dtype=np.float32)

    # symmetric norm
    deg = A.sum(1).astype(np.float32)
    inv_sqrt = 1.0 / (np.sqrt(deg) + 1e-9)
    A = (inv_sqrt[:, None] * A) * inv_sqrt[None, :]

    # node features
    X_all = np.vstack([X, np.zeros((C, X.shape[1]), dtype=np.float32)]).astype(np.float32)

    y = np.array([df_indexed.loc[i, "Label_binary"] for i in ids], dtype=int)
    return (
        torch.tensor(X_all, dtype=torch.float32, device=DEVICE),
        torch.tensor(A, dtype=torch.float32, device=DEVICE),
        np.arange(N),
        np.arange(N, N + C),
        y
    )

X_all_tr, A_tr, img_tr, con_tr, y_tr = build_graph(X_tr, ids_tr)
X_all_val, A_val, img_val, con_val, y_val = build_graph(X_val, ids_val)
X_all_te, A_te, img_te, con_te, y_te = build_graph(X_te, ids_te)

# ================================================================
# 7) DK-SageNet (GraphSAGE + KG) – Dense mean aggregator
#    - concept embeddings injected
#    - 2 SAGE layers with residual + LN
# ================================================================
class DenseSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_nei  = nn.Linear(in_dim, out_dim, bias=False)
        self.ln = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, A_mask):
        # mean neighbor aggregation (A_mask includes self-loops)
        denom = A_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        neigh = (A_mask @ H) / denom

        out = self.lin_self(H) + self.lin_nei(neigh)
        out = F.gelu(out)
        out = self.drop(out)
        return self.ln(out)

class DKSageNet(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=2, dropout=0.3):
        super().__init__()
        self.ce = nn.Embedding(C, in_dim)
        nn.init.xavier_uniform_(self.ce.weight)

        self.sage1 = DenseSAGELayer(in_dim, hidden, dropout=dropout)
        self.sage2 = DenseSAGELayer(hidden, hidden, dropout=dropout)

        self.res_proj = nn.Linear(in_dim, hidden)
        self.cls = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, X_all, A, img_idx, con_idx):
        X = X_all.clone().float()
        A_mask = (A > 0).float()

        # inject KG concept embeddings
        con_t = torch.tensor(con_idx, device=X.device)
        X[con_t] = self.ce(torch.arange(C, device=X.device))

        # Layer 1
        H1 = self.sage1(X, A_mask)
        H1 = H1 + self.res_proj(X)          # residual from input
        H1 = self.drop(H1)

        # Layer 2
        H2 = self.sage2(H1, A_mask)
        H2 = H2 + H1                        # residual
        H2 = self.drop(H2)

        img_t = torch.tensor(img_idx, device=X.device)
        return self.cls(H2[img_t])

model = DKSageNet(in_dim=PCA_DIM, hidden=HIDDEN, out_dim=2, dropout=DROP).to(DEVICE)

# balanced CE
pos_w = (len(y_tr) - int(np.sum(y_tr))) / (int(np.sum(y_tr)) + 1e-6)
w = torch.tensor([1.0, float(pos_w)], device=DEVICE, dtype=torch.float32)
crit = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)

# ================================================================
# 8) Train (early stop on Val F1)
# ================================================================
def val_f1():
    model.eval()
    with torch.no_grad():
        p = model(X_all_val, A_val, img_val, con_val)
        pr = torch.argmax(p, 1).cpu().numpy()
    return f1_score(y_val, pr, zero_division=0)

best_f1 = -1.0
best_state = None
pat = 0

print("\n=== Stage 3: Training DK-SageNet ===")
for ep in range(1, EPOCHS_SAGE + 1):
    model.train()
    opt.zero_grad()
    logits = model(X_all_tr, A_tr, img_tr, con_tr)
    loss = crit(logits, y_tr_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()

    f1v = val_f1()
    print(f"Ep {ep}/{EPOCHS_SAGE} | Loss:{loss.item():.4f} | ValF1:{f1v:.4f}")

    if f1v > best_f1:
        best_f1 = f1v
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pat = 0
    else:
        pat += 1
        if pat >= PATIENCE:
            print("Early stopping.")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    model.to(DEVICE)
    print(f"Loaded best DK-SageNet (ValF1={best_f1:.4f})")

# ================================================================
# 9) Metrics (SAFE AUC)
# ================================================================
def metrics(X_all, A, img_idx, con_idx, y):
    model.eval()
    with torch.no_grad():
        p = model(X_all, A, img_idx, con_idx)
        pr = torch.argmax(p, 1).cpu().numpy()
        pb = torch.softmax(p, 1)[:, 1].detach().cpu().numpy()
        pb = np.nan_to_num(pb, nan=0.0, posinf=1.0, neginf=0.0)

    acc = accuracy_score(y, pr)
    prec = precision_score(y, pr, zero_division=0)
    rec = recall_score(y, pr, zero_division=0)
    f1 = f1_score(y, pr, zero_division=0)
    kappa = cohen_kappa_score(y, pr)

    if len(np.unique(y)) < 2:
        auc = float("nan")
    else:
        try:
            auc = roc_auc_score(y, pb)
        except Exception:
            auc = float("nan")

    return acc, prec, rec, f1, auc, kappa

print("\n=== FINAL METRICS (DK-SageNet + KG) ===")
print("Train (Acc,Prec,Rec,F1,AUC,Kappa):", metrics(X_all_tr, A_tr, img_tr, con_tr, y_tr))
print("Val   (Acc,Prec,Rec,F1,AUC,Kappa):", metrics(X_all_val, A_val, img_val, con_val, y_val))
print("Test  (Acc,Prec,Rec,F1,AUC,Kappa):", metrics(X_all_te, A_te, img_te, con_te, y_te))


