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


# ==========================================================
# RSNA ICH – DGCN + Knowledge Graph (High-Accuracy Version)
# - Triple-window CT → 3-channel EfficientNet-B0
# - Stage 1: Fine-tune EfficientNet on balanced subset
# - Stage 2: Extract embeddings → PCA → build DGCN+KG graph
# - Gated DGCN with learned concept embeddings
# - Train/Val/Test metrics printed (Acc, Prec, Rec, F1, AUC, Kappa)
# ==========================================================
import os, time, random
import numpy as np, pandas as pd
from tqdm import tqdm
import pydicom, cv2

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

SAMPLES_PER_CLASS = 1200   # you can increase to 2000–3000 if time allows
IMG_SIZE = 256
BATCH = 32                 # EfficientNet training batch size
EMB_BATCH = 32             # embedding extraction
EPOCHS_EFF = 5             # EfficientNet fine-tune epochs
EPOCHS_DGCN = 35           # DGCN training epochs
LR_EFF = 3e-4
LR_DGCN = 3e-4
WEIGHT_DECAY = 1e-5
PATIENCE_EFF = 3
PATIENCE_DGCN = 6
K_NEIGH = 15               # mutual kNN neighbors
PCA_DIM = 256              # embedding dimension after PCA
CONCEPT_EMB_DIM = 256
LABEL_SMOOTH = 0.05

# ---------------- Triple-window CT → 3-ch ----------------
def window_image(img, wl, ww):
    minv = wl - ww/2.0
    maxv = wl + ww/2.0
    out = (img - minv) / (maxv - minv + 1e-9)
    return np.clip(out, 0.0, 1.0)

def make_3ch_from_dcm(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    ch1 = window_image(img, 40, 80)      # brain window
    ch2 = window_image(img, 80, 200)     # subdural-ish
    ch3 = window_image(img, 600, 2800)   # bone window
    ch1 = cv2.resize(ch1, (IMG_SIZE, IMG_SIZE))
    ch2 = cv2.resize(ch2, (IMG_SIZE, IMG_SIZE))
    ch3 = cv2.resize(ch3, (IMG_SIZE, IMG_SIZE))
    return np.stack([ch1, ch2, ch3], axis=0)  # (3, H, W)

# ---------------- Load & Balance Labels ----------------
df = pd.read_csv(CSV_PATH)
df["Image"] = df["ID"].apply(lambda x: x.split("_")[1])
df["Subtype"] = df["ID"].apply(lambda x: x.split("_")[2])

df_group = df.groupby(["Image","Subtype"], as_index=False)["Label"].max()
df_pivot = df_group.pivot(index="Image", columns="Subtype", values="Label").reset_index().fillna(0)

# Binary: any hemorrhage
df_pivot["Label_binary"] = df_pivot.iloc[:,1:].max(axis=1).astype(int)
print("Total unique images:", len(df_pivot))
print("Class counts (full):", df_pivot["Label_binary"].value_counts().to_dict())

# Balanced subset
samples = []
for lbl in [0, 1]:
    sub = df_pivot[df_pivot["Label_binary"]==lbl]
    n = min(len(sub), SAMPLES_PER_CLASS)
    samples.append(sub.sample(n, random_state=SEED))
df_bal = pd.concat(samples).reset_index(drop=True)
print("Balanced counts:", df_bal["Label_binary"].value_counts())

subtype_cols = [c for c in df_bal.columns if c not in ["Image","Label_binary"]]
C = len(subtype_cols)
df_indexed = df_bal.set_index("Image")

# For concept–concept PMI later
B_full = df_bal[subtype_cols].values.astype(float)  # (N_bal, C)

# ---------------- Dataset for EfficientNet ----------------
class RSNAEffDataset(Dataset):
    def __init__(self, df, img_dir, img_size=IMG_SIZE, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment
        
        self.aug_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(img_size, scale=(0.85,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.eval_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_id = self.df.loc[idx,"Image"]
        label = int(self.df.loc[idx,"Label_binary"])
        path = os.path.join(self.img_dir, f"ID_{img_id}.dcm")
        img3 = make_3ch_from_dcm(path)   # (3,H,W) float [0..1]
        img3 = (img3*255).astype(np.uint8)
        tf = self.aug_tf if self.augment else self.eval_tf
        img_t = tf(np.transpose(img3, (1,2,0)))
        return img_t, label, img_id

# Split df_bal into train/val/test
train_df, temp_df = train_test_split(df_bal, test_size=0.3, stratify=df_bal["Label_binary"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label_binary"], random_state=SEED)
print("Splits (images):", len(train_df), len(val_df), len(test_df))

train_ds = RSNAEffDataset(train_df, TRAIN_DIR, augment=True)
val_ds   = RSNAEffDataset(val_df, TRAIN_DIR, augment=False)
test_ds  = RSNAEffDataset(test_df, TRAIN_DIR, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# ---------------- EfficientNet-B0 Fine-tune ----------------
def make_efficientnet_model(num_classes=2):
    eff = models.efficientnet_b0(pretrained=True)
    in_feat = eff.classifier[1].in_features
    eff.classifier[1] = nn.Linear(in_feat, num_classes)
    return eff

effnet = make_efficientnet_model(num_classes=2).to(DEVICE)

# For max accuracy, fine-tune the whole network
for name, p in effnet.named_parameters():
    p.requires_grad = True

crit_eff = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
opt_eff = torch.optim.AdamW(effnet.parameters(), lr=LR_EFF, weight_decay=WEIGHT_DECAY)
sched_eff = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_eff, mode='max', factor=0.5, patience=1, verbose=True)

def eval_eff(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labs = labels.cpu().numpy()
            all_preds.extend(preds); all_labels.extend(labs)
    acc = accuracy_score(all_labels, all_preds)
    return acc

print("\n=== Stage 1: Fine-tuning EfficientNet-B0 ===")
best_val_acc = -1.0
best_eff_state = None
eff_patience = 0

for ep in range(1, EPOCHS_EFF+1):
    effnet.train()
    running_loss=0.0
    preds_tr, labs_tr = [], []
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

        running_loss += loss.item()*imgs.size(0)
        preds_tr.extend(torch.argmax(logits,dim=1).cpu().numpy())
        labs_tr.extend(labels.cpu().numpy())
    train_loss = running_loss / len(train_ds)
    train_acc = accuracy_score(labs_tr, preds_tr)
    val_acc = eval_eff(effnet, val_loader)
    sched_eff.step(val_acc)
    print(f"Eff Ep {ep}/{EPOCHS_EFF} | Loss:{train_loss:.4f} | TrAcc:{train_acc:.4f} | ValAcc:{val_acc:.4f} | time:{time.time()-t0:.1f}s")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_eff_state = effnet.state_dict()
        eff_patience = 0
    else:
        eff_patience += 1
        if eff_patience >= PATIENCE_EFF:
            print("Early stopping EfficientNet.")
            break

if best_eff_state is not None:
    effnet.load_state_dict(best_eff_state)
    print("Loaded best EfficientNet, ValAcc:", best_val_acc)

# ---------------- Stage 2: Extract Embeddings ----------------
class EffnetEmbedder(nn.Module):
    def __init__(self, eff):
        super().__init__()
        self.features = eff.features
        self.avgpool = eff.avgpool
        self.dropout = eff.classifier[0]  # dropout
        self.feat_dim = eff.classifier[1].in_features
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x

embedder = EffnetEmbedder(effnet).to(DEVICE)
embedder.eval()

def extract_embeddings_for_df(df_subset, batch_size=EMB_BATCH):
    ds = RSNAEffDataset(df_subset, TRAIN_DIR, augment=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    embs=[]; labs=[]; ids=[]
    with torch.no_grad():
        for imgs, labels, id_batch in tqdm(dl, desc="Extracting embeddings"):
            imgs = imgs.to(DEVICE)
            feat = embedder(imgs)   # (B, feat_dim)
            embs.append(feat.cpu().numpy())
            labs.extend(labels.numpy().tolist())
            ids.extend(id_batch)
    X = np.vstack(embs); y = np.array(labs)
    return X, y, ids

print("\n=== Extracting embeddings (train/val/test) ===")
X_tr, y_tr, ids_tr = extract_embeddings_for_df(train_df)
X_val, y_val, ids_val = extract_embeddings_for_df(val_df)
X_te, y_te, ids_te   = extract_embeddings_for_df(test_df)
print("Embeddings shapes:", X_tr.shape, X_val.shape, X_te.shape)

# PCA for stability & speed
if PCA_DIM is not None and PCA_DIM < X_tr.shape[1]:
    print("Fitting PCA...")
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_tr = pca.fit_transform(X_tr)
    X_val = pca.transform(X_val)
    X_te = pca.transform(X_te)
    print("PCA dim:", X_tr.shape[1])

# ---------------- Graph Construction (DGCN + KG) ----------------
def construct_combined_graph(X_images, ids_images, k_neighbors=K_NEIGH):
    """
    Build combined graph:
      - Image–image: mutual kNN on sharpened cosine sim
      - Image–concept: direct subtype incidence (binary/multi-hot)
      - Concept–concept: PMI matrix from subtype co-occurrence across df_bal
    Returns: X_all_t, A_norm_t, img_idx, concept_idx, labels_img
    """
    N = X_images.shape[0]

    # ----- image–image (mutual kNN, sharpened cosine) -----
    sim = cosine_similarity(X_images)
    np.fill_diagonal(sim, 0)
    sim = np.power(sim, 3)  # sharpen similarities

    A_ii = np.zeros_like(sim)
    for i in range(N):
        nbrs = np.argsort(sim[i])[-k_neighbors:]
        A_ii[i, nbrs] = sim[i, nbrs]
    A_ii = np.minimum(A_ii, A_ii.T)  # mutual kNN

    # ----- image–concept incidence (subtype labels) -----
    # shape: (N, C)
    B_img = df_indexed.loc[ids_images, subtype_cols].values.astype(float)

    # row-normalize so each image's outgoing mass over concepts sums to 1 (if any subtype)
    row_sum = B_img.sum(axis=1, keepdims=True)
    B_norm = B_img / (row_sum + 1e-6)
    B_norm[row_sum.squeeze(-1) == 0] = 0.0  # if all zero, keep zeros

    A_ic = B_norm          # (N, C)
    A_ci = A_ic.T.copy()   # (C, N)

    # ----- concept–concept PMI from global balanced matrix B_full -----
    p = B_full.mean(axis=0) + 1e-9
    co = B_full.T @ B_full                     # (C, C)
    pmi = np.log((co + 1e-6) / (p[:, None] * p[None, :]))
    pmi[pmi < 0] = 0.0
    A_cc = pmi

    # ----- combine all blocks -----
    top = np.concatenate([A_ii, A_ic], axis=1)       # (N, N+C)
    bottom = np.concatenate([A_ci, A_cc], axis=1)    # (C, N+C)
    A = np.concatenate([top, bottom], axis=0)        # (N+C, N+C)

    # ----- symmetric normalization -----
    A = A + np.eye(A.shape[0]) * 1e-6
    deg = A.sum(axis=1)
    inv = 1.0 / (np.sqrt(deg) + 1e-9)
    D_inv = np.diag(inv)
    A_norm = D_inv @ A @ D_inv

    # ----- node features: image embeddings + concept placeholders -----
    X_all = np.vstack([X_images, np.zeros((C, X_images.shape[1]), dtype=float)])
    X_all_t = torch.tensor(X_all, dtype=torch.float32, device=DEVICE)
    A_norm_t = torch.tensor(A_norm, dtype=torch.float32, device=DEVICE)

    img_idx = np.arange(N)
    concept_idx = np.arange(N, N + C)

    labels_img = np.array([df_indexed.loc[iid, "Label_binary"] for iid in ids_images], dtype=int)

    return X_all_t, A_norm_t, img_idx, concept_idx, labels_img

print("\n=== Building DGCN+KG graphs (train/val/test) ===")
X_all_tr, A_tr, img_idx_tr, concept_idx_tr, labels_tr = construct_combined_graph(X_tr, ids_tr)
X_all_val, A_val, img_idx_val, concept_idx_val, labels_val = construct_combined_graph(X_val, ids_val)
X_all_te,  A_te, img_idx_te, concept_idx_te, labels_te  = construct_combined_graph(X_te, ids_te)
print("Combined train graph:", X_all_tr.shape, A_tr.shape)

# ---------------- DGCN-KG Model (Gated) ----------------
class DGCN_KG(nn.Module):
    def __init__(self, in_feats, hidden=256, out_feats=2, num_concepts=C, concept_emb_dim=CONCEPT_EMB_DIM):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_embed = nn.Embedding(num_concepts, concept_emb_dim)
        nn.init.xavier_uniform_(self.concept_embed.weight)
        self.concept_proj = nn.Linear(concept_emb_dim, in_feats)

        self.proj = nn.Linear(in_feats, hidden)

        self.gate1 = nn.Linear(hidden, hidden)
        self.msg1  = nn.Linear(hidden, hidden)
        self.gate2 = nn.Linear(hidden, hidden)
        self.msg2  = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(hidden, out_feats)

    def forward(self, X_all, A_norm, concept_idx):
        # Replace concept node features with learned embeddings
        if concept_idx is not None and len(concept_idx) > 0:
            c_ids = torch.arange(len(concept_idx), device=DEVICE)
            c_emb = self.concept_proj(self.concept_embed(c_ids))  # (C,in_feats)
            X_all = X_all.clone()
            X_all[concept_idx, :] = c_emb

        h = F.relu(self.proj(X_all))

        # Layer 1 (gated)
        g1 = torch.sigmoid(self.gate1(h))
        m1 = A_norm @ self.msg1(h)
        h  = g1 * m1
        h  = self.dropout(h)

        # Layer 2 (gated)
        g2 = torch.sigmoid(self.gate2(h))
        m2 = A_norm @ self.msg2(h)
        h  = g2 * m2
        h  = self.dropout(h)

        logits = self.classifier(h)
        return logits, h

model = DGCN_KG(in_feats=X_all_tr.shape[1], hidden=256, out_feats=2).to(DEVICE)
opt_dg = torch.optim.AdamW(model.parameters(), lr=LR_DGCN, weight_decay=WEIGHT_DECAY)
crit_dg = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
sched_dg = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_dg, mode='max', factor=0.5, patience=2, verbose=True)

def eval_on_graph(model, X_all, A_norm, img_idx, concept_idx, labels_img):
    model.eval()
    with torch.no_grad():
        logits_all, _ = model(X_all, A_norm, concept_idx)
        logits_img = logits_all[img_idx]
        preds = torch.argmax(logits_img, dim=1).cpu().numpy()
        probs = F.softmax(logits_img, dim=1)[:,1].cpu().numpy()
        labels = labels_img
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = float("nan")
        kappa = cohen_kappa_score(labels, preds)
    return acc, prec, rec, f1, auc, kappa

# ---------------- Train DGCN+KG ----------------
print("\n=== Stage 3: Training DGCN + KG ===")
best_val_acc = -1.0
best_state = None
pat_dg = 0

labels_tr_t = torch.tensor(labels_tr, dtype=torch.long, device=DEVICE)

for ep in range(1, EPOCHS_DGCN+1):
    t0 = time.time()
    model.train()
    opt_dg.zero_grad()
    logits_all, _ = model(X_all_tr, A_tr, concept_idx_tr)
    logits_img = logits_all[img_idx_tr]
    loss = crit_dg(logits_img, labels_tr_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    opt_dg.step()

    # Metrics
    train_acc, train_prec, train_rec, train_f1, train_auc, train_kappa = eval_on_graph(
        model, X_all_tr, A_tr, img_idx_tr, concept_idx_tr, labels_tr
    )
    val_acc, val_prec, val_rec, val_f1, val_auc, val_kappa = eval_on_graph(
        model, X_all_val, A_val, img_idx_val, concept_idx_val, labels_val
    )
    test_acc, test_prec, test_rec, test_f1, test_auc, test_kappa = eval_on_graph(
        model, X_all_te, A_te, img_idx_te, concept_idx_te, labels_te
    )

    sched_dg.step(val_acc)

    print(f"Ep {ep}/{EPOCHS_DGCN} | Loss:{loss.item():.5f} | "
          f"TrAcc:{train_acc:.4f} ValAcc:{val_acc:.4f} TeAcc:{test_acc:.4f} | "
          f"ValF1:{val_f1:.4f} ValAUC:{val_auc:.4f} | time:{time.time()-t0:.1f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {"model": model.state_dict(), "epoch": ep}
        pat_dg = 0
    else:
        pat_dg += 1
        if pat_dg >= PATIENCE_DGCN:
            print("Early stopping DGCN.")
            break

if best_state is not None:
    model.load_state_dict(best_state["model"])
    print(f"Loaded best DGCN model from epoch {best_state['epoch']} (ValAcc={best_val_acc:.4f})")

# ---------------- Final Metrics ----------------
train_acc, train_prec, train_rec, train_f1, train_auc, train_kappa = eval_on_graph(
    model, X_all_tr, A_tr, img_idx_tr, concept_idx_tr, labels_tr
)
val_acc, val_prec, val_rec, val_f1, val_auc, val_kappa = eval_on_graph(
    model, X_all_val, A_val, img_idx_val, concept_idx_val, labels_val
)
test_acc, test_prec, test_rec, test_f1, test_auc, test_kappa = eval_on_graph(
    model, X_all_te, A_te, img_idx_te, concept_idx_te, labels_te
)

print("\n=== FINAL METRICS (DGCN + KG, High-Accuracy) ===")
print(f"Train Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f} | AUC: {train_auc:.4f} | Kappa: {train_kappa:.4f}")
print(f"Val   Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f} | Kappa: {val_kappa:.4f}")
print(f"Test  Acc: {test_acc:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f} | Kappa: {test_kappa:.4f}")





