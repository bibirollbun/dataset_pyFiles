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
# RSNA ICH - EfficientNet Embeddings + GCN + Knowledge Graph
# Single-cell full pipeline (uses your CSV + ID_<Image>.dcm files)
# ================================================================
import os, time, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2, pydicom

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, roc_auc_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ---------------- CONFIG ----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

BASE = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection"
CSV_PATH = os.path.join(BASE, "stage_2_train.csv")
TRAIN_DIR = os.path.join(BASE, "stage_2_train")

IMG_SIZE = 256
SAMPLES_PER_CLASS = 600   # total ~1200 images → faster on CPU
BATCH = 16
EMB_BATCH = 16

PCA_DIM = 256
K_NEIGH = 10

EPOCHS_EFF = 2            # small to stay fast
EPOCHS_GCN = 25
LR_EFF = 3e-4
LR_GCN = 3e-4

LABEL_SMOOTH = 0.05

# ================================================================
#   1. TRIPLE-WINDOW CT → 3-CHANNEL IMAGE
# ================================================================
def window_image(img, wl, ww):
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    out = (img - low) / (high - low + 1e-9)
    return np.clip(out, 0.0, 1.0)

def make_3ch_dcm(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)

    # common RSNA windows
    ch1 = window_image(img, 40, 80)       # brain
    ch2 = window_image(img, 80, 200)      # subdural-ish
    ch3 = window_image(img, 600, 2800)    # bone

    ch1 = cv2.resize(ch1, (IMG_SIZE, IMG_SIZE))
    ch2 = cv2.resize(ch2, (IMG_SIZE, IMG_SIZE))
    ch3 = cv2.resize(ch3, (IMG_SIZE, IMG_SIZE))

    return np.stack([ch1, ch2, ch3], axis=0)   # (3,H,W)

# ================================================================
#   2. LOAD CSV + BUILD BINARY + BALANCED SUBSET
# ================================================================
df = pd.read_csv(CSV_PATH)

# Your CSV example already has Image, Subtype, but we make sure:
if "Image" not in df.columns or "Subtype" not in df.columns:
    df["Image"] = df["ID"].apply(lambda x: x.split("_")[1])
    df["Subtype"] = df["ID"].apply(lambda x: x.split("_")[2])

# Aggregate to per-image multi-label
df_group = df.groupby(["Image","Subtype"], as_index=False)["Label"].max()
df_pivot = df_group.pivot(index="Image", columns="Subtype", values="Label").fillna(0)

# Binary: any hemorrhage
df_pivot["Label_binary"] = df_pivot.max(axis=1).astype(int)

subtype_cols = list(df_pivot.columns[:-1])  # all subtypes
C = len(subtype_cols)
print("Subtypes:", subtype_cols)

# Reset index so "Image" is a column
df_pivot = df_pivot.reset_index()   # columns: Image, <subtypes...>, Label_binary

# Balanced subset
samples = []
for lbl in [0,1]:
    sub = df_pivot[df_pivot["Label_binary"] == lbl]
    n = min(SAMPLES_PER_CLASS, len(sub))
    samples.append(sub.sample(n, random_state=SEED))
df_bal = pd.concat(samples).reset_index(drop=True)
print("Balanced counts:\n", df_bal["Label_binary"].value_counts())

# Keep an index by Image for later graph / labels
df_indexed = df_bal.set_index("Image")   # index: Image, has subtypes + Label_binary

# ================================================================
#   3. DATASET FOR EFFICIENTNET
# ================================================================
class RSNADataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment

        self.tf_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        self.tf_eval = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["Image"]                # <-- IMPORTANT: correct image ID
        label = int(row["Label_binary"])

        dcm_path = os.path.join(self.img_dir, f"ID_{img_id}.dcm")
        img3 = make_3ch_dcm(dcm_path)        # (3,H,W), float [0..1]
        img3 = (img3 * 255).astype(np.uint8) # to uint8
        img3 = np.transpose(img3, (1,2,0))   # (H,W,3) for ToPILImage

        tf = self.tf_aug if self.augment else self.tf_eval
        img_t = tf(img3)
        return img_t, label, img_id

# ================================================================
#   4. SPLIT TO TRAIN / VAL / TEST
# ================================================================
train_df, temp_df = train_test_split(
    df_bal,
    test_size=0.3,
    stratify=df_bal["Label_binary"],
    random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["Label_binary"],
    random_state=SEED
)

print("Split sizes: Train =", len(train_df), "Val =", len(val_df), "Test =", len(test_df))

train_ds = RSNADataset(train_df, TRAIN_DIR, augment=True)
val_ds   = RSNADataset(val_df, TRAIN_DIR, augment=False)
test_ds  = RSNADataset(test_df, TRAIN_DIR, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# ================================================================
#   5. EFFICIENTNET-B0 FINE-TUNE (BINARY)
# ================================================================
def make_efficientnet_model(num_classes=2):
    effnet = models.efficientnet_b0(pretrained=True)
    in_features = effnet.classifier[1].in_features
    effnet.classifier[1] = nn.Linear(in_features, num_classes)
    return effnet

effnet = make_efficientnet_model(2).to(DEVICE)

for p in effnet.parameters():
    p.requires_grad = True

crit_eff = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
opt_eff  = torch.optim.AdamW(effnet.parameters(), lr=LR_EFF)

def eval_eff_acc(model, loader):
    model.eval()
    all_preds, all_labs = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labs  = labels.cpu().numpy()
            all_preds.extend(preds); all_labs.extend(labs)
    return accuracy_score(all_labs, all_preds)

print("\n=== Stage 1: EfficientNet fine-tuning (short) ===")
for ep in range(1, EPOCHS_EFF+1):
    effnet.train()
    running_loss = 0.0
    t0 = time.time()
    for imgs, labels, _ in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        opt_eff.zero_grad()
        logits = effnet(imgs)
        loss = crit_eff(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(effnet.parameters(), max_norm=5.0)
        opt_eff.step()

        running_loss += loss.item() * imgs.size(0)

    train_loss = running_loss / len(train_ds)
    val_acc = eval_eff_acc(effnet, val_loader)
    print(f"Eff Ep {ep}/{EPOCHS_EFF} | Loss:{train_loss:.4f} | ValAcc:{val_acc:.4f} | time:{time.time()-t0:.1f}s")

# ================================================================
#   6. EMBEDDING EXTRACTOR
# ================================================================
class EffnetEmbedder(nn.Module):
    def __init__(self, eff):
        super().__init__()
        self.features = eff.features
        self.avgpool  = eff.avgpool
        self.dropout  = eff.classifier[0]  # dropout
        self.feat_dim = eff.classifier[1].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x   # (N, feat_dim)

embedder = EffnetEmbedder(effnet).to(DEVICE)
embedder.eval()

def extract_embeddings_for_df(df_subset):
    ds = RSNADataset(df_subset, TRAIN_DIR, augment=False)
    dl = DataLoader(ds, batch_size=EMB_BATCH, shuffle=False, num_workers=2, pin_memory=True)

    embs, ids = [], []
    with torch.no_grad():
        for imgs, _, id_batch in tqdm(dl, desc="Extracting embeddings"):
            imgs = imgs.to(DEVICE)
            feat = embedder(imgs)          # (B, feat_dim)
            embs.append(feat.cpu().numpy())
            ids.extend(id_batch)
    X = np.vstack(embs)
    return X, ids

print("\n=== Stage 2: Extracting embeddings ===")
X_tr, ids_tr = extract_embeddings_for_df(train_df)
X_val, ids_val = extract_embeddings_for_df(val_df)
X_te, ids_te = extract_embeddings_for_df(test_df)

print("Embedding dims:", X_tr.shape, X_val.shape, X_te.shape)

# ================================================================
#   7. PCA REDUCTION
# ================================================================
if PCA_DIM is not None and PCA_DIM < X_tr.shape[1]:
    print("Fitting PCA to", PCA_DIM, "dims...")
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_tr = pca.fit_transform(X_tr)
    X_val = pca.transform(X_val)
    X_te = pca.transform(X_te)
    print("New shapes:", X_tr.shape, X_val.shape, X_te.shape)
else:
    PCA_DIM = X_tr.shape[1]

# For PMI later
B_full = df_bal[subtype_cols].values.astype(float)

# ================================================================
#   8. GRAPH CONSTRUCTION (IMAGE + KG)
# ================================================================
def construct_combined_graph(X_images, ids_images, k_neighbors=K_NEIGH):
    """
    Combined graph:
      - image-image: mutual kNN cosine
      - image-concept: subtype incidence
      - concept-concept: PMI
    """
    N = X_images.shape[0]

    # image-image similarity
    sim = cosine_similarity(X_images)
    np.fill_diagonal(sim, 0.0)
    sim = np.power(sim, 3)

    A_ii = np.zeros_like(sim)
    for i in range(N):
        k = min(k_neighbors, N-1)
        if k <= 0:
            continue
        nbrs = np.argsort(sim[i])[-k:]
        A_ii[i, nbrs] = sim[i, nbrs]
    A_ii = np.minimum(A_ii, A_ii.T)

    # image-concept incidence
    B_img = df_indexed.loc[ids_images, subtype_cols].values.astype(float)  # (N, C)
    row_sum = B_img.sum(axis=1, keepdims=True)
    B_norm = B_img / (row_sum + 1e-6)
    B_norm[row_sum.squeeze(-1) == 0] = 0.0
    A_ic = B_norm
    A_ci = A_ic.T.copy()

    # concept-concept PMI (based on full balanced set)
    p = B_full.mean(axis=0) + 1e-9
    co = B_full.T @ B_full
    pmi = np.log((co + 1e-6) / (p[:, None] * p[None, :]))
    pmi[pmi < 0] = 0.0
    A_cc = pmi

    # block concat
    top = np.concatenate([A_ii, A_ic], axis=1)
    bottom = np.concatenate([A_ci, A_cc], axis=1)
    A = np.concatenate([top, bottom], axis=0)

    # symmetric normalization
    deg = A.sum(axis=1)
    inv_sqrt = 1.0 / (np.sqrt(deg) + 1e-9)
    A_norm = (inv_sqrt[:, None] * A) * inv_sqrt[None, :]

    # node features: images have X_images; concepts start as zeros
    X_all = np.vstack([X_images, np.zeros((C, X_images.shape[1]), dtype=float)])

    X_all_t  = torch.tensor(X_all,  dtype=torch.float32, device=DEVICE)
    A_norm_t = torch.tensor(A_norm, dtype=torch.float32, device=DEVICE)

    img_idx     = np.arange(N)
    concept_idx = np.arange(N, N + C)

    labels_img = np.array([df_indexed.loc[iid, "Label_binary"] for iid in ids_images], dtype=int)
    return X_all_t, A_norm_t, img_idx, concept_idx, labels_img

print("\n=== Stage 3: Building graphs (train/val/test) ===")
X_all_tr, A_tr, img_idx_tr, concept_idx_tr, labels_tr = construct_combined_graph(X_tr, ids_tr)
X_all_val, A_val, img_idx_val, concept_idx_val, labels_val = construct_combined_graph(X_val, ids_val)
X_all_te,  A_te,  img_idx_te,  concept_idx_te, labels_te  = construct_combined_graph(X_te, ids_te)

print("Train graph:", X_all_tr.shape, A_tr.shape)

# ================================================================
#   9. GCN + KNOWLEDGE GRAPH MODEL
# ================================================================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, X, A):
        X = self.lin(X)
        X = A @ X
        return F.relu(X)

class GCN_KG(nn.Module):
    """
    GCN over combined Image+KG graph:
      - concept nodes have learnable embeddings
      - 2-layer GCN
      - classifier on image-node embeddings
    """
    def __init__(self, in_dim, hidden=256, out_dim=2,
                 num_concepts=C, concept_emb_dim=256):
        super().__init__()
        self.num_concepts = num_concepts

        self.concept_embed = nn.Embedding(num_concepts, concept_emb_dim)
        nn.init.xavier_uniform_(self.concept_embed.weight)
        self.concept_proj = nn.Linear(concept_emb_dim, in_dim)

        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(hidden, out_dim)

    def inject_concepts(self, X_all, concept_idx):
        if concept_idx is None or len(concept_idx) == 0:
            return X_all
        concept_idx_t = torch.tensor(concept_idx, dtype=torch.long, device=X_all.device)
        concept_ids   = torch.arange(self.num_concepts, dtype=torch.long, device=X_all.device)
        c_emb = self.concept_proj(self.concept_embed(concept_ids))  # (C,in_dim)
        X_new = X_all.clone()
        X_new[concept_idx_t, :] = c_emb
        return X_new

    def forward(self, X_all, A_norm, img_idx, concept_idx):
        X_in = self.inject_concepts(X_all, concept_idx)
        h = self.gcn1(X_in, A_norm)
        h = self.dropout(h)
        h = self.gcn2(h, A_norm)
        h = self.dropout(h)

        img_idx_t = torch.tensor(img_idx, dtype=torch.long, device=h.device)
        h_img = h[img_idx_t]
        logits = self.classifier(h_img)
        return logits, h_img

def eval_gcn_kg(model, X_all, A_norm, img_idx, concept_idx, labels_img):
    model.eval()
    with torch.no_grad():
        logits, h_img = model(X_all, A_norm, img_idx, concept_idx)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()

    labels = labels_img
    acc   = accuracy_score(labels, preds)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    f1    = f1_score(labels, preds, zero_division=0)
    kappa = cohen_kappa_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    return acc, prec, rec, f1, auc, kappa

model = GCN_KG(in_dim=X_all_tr.shape[1], hidden=256, out_dim=2, num_concepts=C, concept_emb_dim=256).to(DEVICE)
opt_gcn = torch.optim.AdamW(model.parameters(), lr=LR_GCN, weight_decay=1e-5)
crit_gcn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

labels_tr_t = torch.tensor(labels_tr, dtype=torch.long, device=DEVICE)

print("\n=== Stage 4: Training GCN + KG ===")
best_val_acc = -1.0
best_state = None
patience = 7
pat = 0

for ep in range(1, EPOCHS_GCN+1):
    t0 = time.time()
    model.train()
    opt_gcn.zero_grad()

    logits, _ = model(X_all_tr, A_tr, img_idx_tr, concept_idx_tr)
    loss = crit_gcn(logits, labels_tr_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    opt_gcn.step()

    # metrics
    tr_acc, tr_prec, tr_rec, tr_f1, tr_auc, tr_kappa = eval_gcn_kg(
        model, X_all_tr, A_tr, img_idx_tr, concept_idx_tr, labels_tr
    )
    val_acc, val_prec, val_rec, val_f1, val_auc, val_kappa = eval_gcn_kg(
        model, X_all_val, A_val, img_idx_val, concept_idx_val, labels_val
    )

    print(f"Ep {ep}/{EPOCHS_GCN} | Loss:{loss.item():.4f} | "
          f"TrAcc:{tr_acc:.4f} ValAcc:{val_acc:.4f} | "
          f"ValF1:{val_f1:.4f} ValAUC:{val_auc:.4f} | time:{time.time()-t0:.1f}s")

    # early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()
        pat = 0
    else:
        pat += 1
        if pat >= patience:
            print("Early stopping GCN+KG.")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Loaded best GCN+KG model (ValAcc={best_val_acc:.4f})")

# ================================================================
#   10. FINAL METRICS
# ================================================================
train_metrics = eval_gcn_kg(model, X_all_tr, A_tr, img_idx_tr, concept_idx_tr, labels_tr)
val_metrics   = eval_gcn_kg(model, X_all_val, A_val, img_idx_val, concept_idx_val, labels_val)
test_metrics  = eval_gcn_kg(model, X_all_te, A_te, img_idx_te, concept_idx_te, labels_te)

print("\n=== FINAL METRICS (GCN + KG) ===")
print("Train Acc, Prec, Rec, F1, AUC, Kappa:", train_metrics)
print("Val   Acc, Prec, Rec, F1, AUC, Kappa:", val_metrics)
print("Test  Acc, Prec, Rec, F1, AUC, Kappa:", test_metrics)


