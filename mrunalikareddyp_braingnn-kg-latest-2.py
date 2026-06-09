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


# ===============================================================
# BrainGNN + Knowledge Graph (RSNA Intracranial Hemorrhage)
# ===============================================================
import os, random, time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pydicom, cv2

# -------------------- CONFIG --------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

BASE_PATH = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection"
CSV_PATH = os.path.join(BASE_PATH, "stage_2_train.csv")
IMG_DIR = os.path.join(BASE_PATH, "stage_2_train")

SAMPLES_PER_CLASS = 2000
IMG_SIZE = 160
BATCH = 64
FT_EPOCHS = 2
EPOCHS = 20
LR_BACKBONE = 2e-4
LR_GNN = 1e-3
K_NEIGH = 16
PCA_DIM = 256
KG_EMB_DIM = 16
TEMP = 0.2
ALPHA_SUP = 1.0
ALPHA_CON = 1.0

SUBTYPE_COLS = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']

# -------------------- HELPERS --------------------
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
set_seed()

def apply_brain_window(img, level=40, width=80):
    low = level - width/2.0
    high = level + width/2.0
    img_w = np.clip(img, low, high)
    img_w = (img_w - low) / (high - low + 1e-6)
    return img_w.astype(np.float32)

def load_dicom(path):
    d = pydicom.dcmread(path)
    arr = d.pixel_array.astype(np.float32)
    arr = apply_brain_window(arr)
    return arr

def make_3ch(path):
    arr = load_dicom(path)
    arr = cv2.resize(arr,(IMG_SIZE,IMG_SIZE))
    return np.stack([arr,arr,arr], axis=-1)

# -------------------- LOAD & BALANCE LABELS --------------------
df = pd.read_csv(CSV_PATH)
df["Image"] = df["ID"].apply(lambda x: x.split("_")[1])
df["Subtype"] = df["ID"].apply(lambda x: x.split("_")[2])
df2 = df.groupby(["Image","Subtype"], as_index=False)["Label"].max()
df_pivot = df2.pivot(index="Image", columns="Subtype", values="Label").reset_index().fillna(0)
df_pivot["Label_binary"] = df_pivot.iloc[:,1:].max(axis=1).astype(int)

# Balanced sampling
pos = df_pivot[df_pivot["Label_binary"]==1].sample(n=SAMPLES_PER_CLASS, random_state=SEED)
neg = df_pivot[df_pivot["Label_binary"]==0].sample(n=SAMPLES_PER_CLASS, random_state=SEED)
df_bal = pd.concat([pos,neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# -------------------- DATASET --------------------
class RSNADataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.aug_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
        self.eval_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"ID_{row.Image}.dcm")
        img = make_3ch(img_path)
        tf = self.aug_tf if self.augment else self.eval_tf
        img_t = tf(img)
        label = int(row.Label_binary)
        return img_t.float(), label, row.Image

# -------------------- SPLIT --------------------
train_df, temp_df = train_test_split(df_bal,test_size=0.3,stratify=df_bal["Label_binary"],random_state=SEED)
val_df, test_df = train_test_split(temp_df,test_size=0.5,stratify=temp_df["Label_binary"],random_state=SEED)

train_ds = RSNADataset(train_df, IMG_DIR, augment=True)
val_ds = RSNADataset(val_df, IMG_DIR, augment=False)
test_ds = RSNADataset(test_df, IMG_DIR, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

# -------------------- BACKBONE --------------------
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
embedding_dim = resnet.fc.in_features
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE)

# -------------------- EXTRACT EMBEDDINGS --------------------
def extract_embeddings(loader):
    resnet.eval()
    embs=[]; labs=[]; ids=[]
    with torch.no_grad():
        for imgs, lab, idlist in tqdm(loader):
            imgs = imgs.to(DEVICE)
            feat = resnet(imgs)
            embs.append(feat.cpu().numpy())
            labs.extend(lab.numpy())
            ids.extend(idlist)
    return np.vstack(embs), np.array(labs), ids

X_tr, y_tr, ids_tr = extract_embeddings(train_loader)
X_val, y_val, ids_val = extract_embeddings(val_loader)
X_te, y_te, ids_te = extract_embeddings(test_loader)

# PCA dim reduction
if PCA_DIM is not None and PCA_DIM < X_tr.shape[1]:
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_tr = pca.fit_transform(X_tr)
    X_val = pca.transform(X_val)
    X_te = pca.transform(X_te)

# -------------------- CONSTRUCT GRAPH --------------------
class ConstructGraph:
    def __init__(self, df_indexed, subtype_cols, k_neigh=K_NEIGH, kg_dim=KG_EMB_DIM):
        self.df_indexed = df_indexed
        self.subtype_cols = subtype_cols
        self.k_neigh = k_neigh
        self.kg_dim = kg_dim
        self.concept_feats = np.random.normal(0,0.01,(len(subtype_cols), kg_dim))

    def build_knn_adj(self,X,k=None):
        if k is None: k=self.k_neigh
        sim = cosine_similarity(X); np.fill_diagonal(sim,0)
        N=sim.shape[0]; A=np.zeros_like(sim)
        for i in range(N):
            idx = np.argsort(sim[i])[-k:]
            A[i, idx] = sim[i, idx]
        A = np.maximum(A, A.T) + np.eye(N)*1e-6
        deg = A.sum(1); deg_inv_sqrt=1.0/np.sqrt(deg)
        return (deg_inv_sqrt[:,None]*A)*deg_inv_sqrt[None,:]

    def build_graph(self,X_img,img_ids):
        N = X_img.shape[0]; C = self.concept_feats.shape[0]
        A_ii = self.build_knn_adj(X_img)
        A_ic = np.zeros((N,C))
        for i,iid in enumerate(img_ids):
            if iid in self.df_indexed.index:
                A_ic[i,:] = self.df_indexed.loc[iid,self.subtype_cols].values
        A_ci = A_ic.T
        simc = cosine_similarity(self.concept_feats); np.fill_diagonal(simc,0)
        A_cc = simc
        top = np.concatenate([A_ii,A_ic],1)
        bottom = np.concatenate([A_ci,A_cc],1)
        A = np.concatenate([top,bottom],0) + np.eye(N+C)*1e-6
        deg = A.sum(1); deg_inv_sqrt=1.0/np.sqrt(deg)
        A_norm = (deg_inv_sqrt[:,None]*A)*deg_inv_sqrt[None,:]
        D_img = X_img.shape[1]; D_con = self.concept_feats.shape[1]
        if D_con<D_img:
            con_padded = np.concatenate([self.concept_feats,np.zeros((C,D_img-D_con))],1)
        else:
            con_padded = self.concept_feats[:,:D_img]
        X_all = np.vstack([X_img,con_padded])
        labels_img = np.array([self.df_indexed.loc[iid,"Label_binary"] if iid in self.df_indexed.index else 0 for iid in img_ids])
        return X_all, A_norm, labels_img

df_indexed = df_pivot.set_index("Image")
graph_builder = ConstructGraph(df_indexed, SUBTYPE_COLS)
X_all_tr, A_tr, labels_tr = graph_builder.build_graph(X_tr, ids_tr)
X_all_val, A_val, labels_val = graph_builder.build_graph(X_val, ids_val)
X_all_te,  A_te,  labels_te  = graph_builder.build_graph(X_te, ids_te)

# Convert to torch
X_tr_t = torch.tensor(X_all_tr, dtype=torch.float32, device=DEVICE)
A_tr_t = torch.tensor(A_tr, dtype=torch.float32, device=DEVICE)
y_tr_img = torch.tensor(labels_tr, dtype=torch.long, device=DEVICE)
X_val_t = torch.tensor(X_all_val, dtype=torch.float32, device=DEVICE)
A_val_t = torch.tensor(A_val, dtype=torch.float32, device=DEVICE)
y_val_img = torch.tensor(labels_val, dtype=torch.long, device=DEVICE)
X_te_t = torch.tensor(X_all_te, dtype=torch.float32, device=DEVICE)
A_te_t = torch.tensor(A_te, dtype=torch.float32, device=DEVICE)
N_img_tr = X_tr.shape[0]; N_img_val = X_val.shape[0]; N_img_te = X_te.shape[0]

# -------------------- BrainGNN (Simple GNN block) --------------------
class SimpleGCNBlock(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim,hid_dim)
        self.lin2 = nn.Linear(hid_dim,hid_dim)
        self.dropout = nn.Dropout(0.4)
        self.bn = nn.LayerNorm(hid_dim)
    def forward(self,X,A):
        h = F.relu(self.bn(self.lin1(X)))
        h = A@h
        h = self.dropout(h)
        h = F.relu(self.lin2(h))
        h = A@h
        return h

class BrainGNN(nn.Module):
    def __init__(self, feat_dim,hid_dim=256,n_classes=2):
        super().__init__()
        self.encoder = nn.Linear(feat_dim,hid_dim)
        self.gcn = SimpleGCNBlock(hid_dim,hid_dim)
        self.classifier = nn.Linear(hid_dim,n_classes)
    def forward(self,X,A):
        H = F.relu(self.encoder(X))
        H_g = self.gcn(H,A)
        logits = self.classifier(H_g)
        return logits

# -------------------- TRAIN --------------------
model = BrainGNN(X_all_tr.shape[1], hid_dim=256).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR_GNN)
crit = nn.CrossEntropyLoss()

for ep in range(1,EPOCHS+1):
    model.train()
    opt.zero_grad()
    logits_all = model(X_tr_t, A_tr_t)
    logits_img = logits_all[:N_img_tr]
    loss = crit(logits_img, y_tr_img)
    loss.backward(); opt.step()
    if ep%5==0:
        print(f"Epoch {ep}/{EPOCHS} Loss:{loss.item():.4f}")

# -------------------- FINAL METRICS --------------------
model.eval()
with torch.no_grad():
    logits_te_all = model(X_te_t, A_te_t)
    preds = torch.argmax(logits_te_all[:N_img_te],dim=1).cpu().numpy()
    probs = F.softmax(logits_te_all[:N_img_te],dim=1)[:,1].cpu().numpy()
    acc = accuracy_score(labels_te, preds)
    prec = precision_score(labels_te, preds, zero_division=0)
    rec = recall_score(labels_te, preds, zero_division=0)
    f1s = f1_score(labels_te, preds, zero_division=0)
    auc = roc_auc_score(labels_te, probs)
    kappa = cohen_kappa_score(labels_te, preds)
print("\n=== FINAL METRICS ===")
print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1s:.4f}")
print(f"Cohen Kappa : {kappa:.4f}")



# ===============================================================
# BrainGNN + Knowledge Graph (RSNA Intracranial Hemorrhage)
# ===============================================================
import os, random, time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pydicom, cv2

# -------------------- CONFIG --------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

BASE_PATH = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection"
CSV_PATH = os.path.join(BASE_PATH, "stage_2_train.csv")
IMG_DIR = os.path.join(BASE_PATH, "stage_2_train")

SAMPLES_PER_CLASS = 2000
IMG_SIZE = 160
BATCH = 64
FT_EPOCHS = 2
EPOCHS = 50
LR_BACKBONE = 2e-4
LR_GNN = 1e-3
K_NEIGH = 16
PCA_DIM = 256
KG_EMB_DIM = 16
TEMP = 0.2
ALPHA_SUP = 1.0
ALPHA_CON = 1.0

SUBTYPE_COLS = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']

# -------------------- HELPERS --------------------
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
set_seed()

def apply_brain_window(img, level=40, width=80):
    low = level - width/2.0
    high = level + width/2.0
    img_w = np.clip(img, low, high)
    img_w = (img_w - low) / (high - low + 1e-6)
    return img_w.astype(np.float32)

def load_dicom(path):
    d = pydicom.dcmread(path)
    arr = d.pixel_array.astype(np.float32)
    arr = apply_brain_window(arr)
    return arr

def make_3ch(path):
    arr = load_dicom(path)
    arr = cv2.resize(arr,(IMG_SIZE,IMG_SIZE))
    return np.stack([arr,arr,arr], axis=-1)

# -------------------- LOAD & BALANCE LABELS --------------------
df = pd.read_csv(CSV_PATH)
df["Image"] = df["ID"].apply(lambda x: x.split("_")[1])
df["Subtype"] = df["ID"].apply(lambda x: x.split("_")[2])
df2 = df.groupby(["Image","Subtype"], as_index=False)["Label"].max()
df_pivot = df2.pivot(index="Image", columns="Subtype", values="Label").reset_index().fillna(0)
df_pivot["Label_binary"] = df_pivot.iloc[:,1:].max(axis=1).astype(int)

# Balanced sampling
pos = df_pivot[df_pivot["Label_binary"]==1].sample(n=SAMPLES_PER_CLASS, random_state=SEED)
neg = df_pivot[df_pivot["Label_binary"]==0].sample(n=SAMPLES_PER_CLASS, random_state=SEED)
df_bal = pd.concat([pos,neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# -------------------- DATASET --------------------
class RSNADataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.aug_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
        self.eval_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"ID_{row.Image}.dcm")
        img = make_3ch(img_path)
        tf = self.aug_tf if self.augment else self.eval_tf
        img_t = tf(img)
        label = int(row.Label_binary)
        return img_t.float(), label, row.Image

# -------------------- SPLIT --------------------
train_df, temp_df = train_test_split(df_bal,test_size=0.3,stratify=df_bal["Label_binary"],random_state=SEED)
val_df, test_df = train_test_split(temp_df,test_size=0.5,stratify=temp_df["Label_binary"],random_state=SEED)

train_ds = RSNADataset(train_df, IMG_DIR, augment=True)
val_ds = RSNADataset(val_df, IMG_DIR, augment=False)
test_ds = RSNADataset(test_df, IMG_DIR, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

# -------------------- BACKBONE --------------------
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
embedding_dim = resnet.fc.in_features
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE)

# -------------------- EXTRACT EMBEDDINGS --------------------
def extract_embeddings(loader):
    resnet.eval()
    embs=[]; labs=[]; ids=[]
    with torch.no_grad():
        for imgs, lab, idlist in tqdm(loader):
            imgs = imgs.to(DEVICE)
            feat = resnet(imgs)
            embs.append(feat.cpu().numpy())
            labs.extend(lab.numpy())
            ids.extend(idlist)
    return np.vstack(embs), np.array(labs), ids

X_tr, y_tr, ids_tr = extract_embeddings(train_loader)
X_val, y_val, ids_val = extract_embeddings(val_loader)
X_te, y_te, ids_te = extract_embeddings(test_loader)

# PCA dim reduction
if PCA_DIM is not None and PCA_DIM < X_tr.shape[1]:
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_tr = pca.fit_transform(X_tr)
    X_val = pca.transform(X_val)
    X_te = pca.transform(X_te)

# -------------------- CONSTRUCT GRAPH --------------------
class ConstructGraph:
    def __init__(self, df_indexed, subtype_cols, k_neigh=K_NEIGH, kg_dim=KG_EMB_DIM):
        self.df_indexed = df_indexed
        self.subtype_cols = subtype_cols
        self.k_neigh = k_neigh
        self.kg_dim = kg_dim
        self.concept_feats = np.random.normal(0,0.01,(len(subtype_cols), kg_dim))

    def build_knn_adj(self,X,k=None):
        if k is None: k=self.k_neigh
        sim = cosine_similarity(X); np.fill_diagonal(sim,0)
        N=sim.shape[0]; A=np.zeros_like(sim)
        for i in range(N):
            idx = np.argsort(sim[i])[-k:]
            A[i, idx] = sim[i, idx]
        A = np.maximum(A, A.T) + np.eye(N)*1e-6
        deg = A.sum(1); deg_inv_sqrt=1.0/np.sqrt(deg)
        return (deg_inv_sqrt[:,None]*A)*deg_inv_sqrt[None,:]

    def build_graph(self,X_img,img_ids):
        N = X_img.shape[0]; C = self.concept_feats.shape[0]
        A_ii = self.build_knn_adj(X_img)
        A_ic = np.zeros((N,C))
        for i,iid in enumerate(img_ids):
            if iid in self.df_indexed.index:
                A_ic[i,:] = self.df_indexed.loc[iid,self.subtype_cols].values
        A_ci = A_ic.T
        simc = cosine_similarity(self.concept_feats); np.fill_diagonal(simc,0)
        A_cc = simc
        top = np.concatenate([A_ii,A_ic],1)
        bottom = np.concatenate([A_ci,A_cc],1)
        A = np.concatenate([top,bottom],0) + np.eye(N+C)*1e-6
        deg = A.sum(1); deg_inv_sqrt=1.0/np.sqrt(deg)
        A_norm = (deg_inv_sqrt[:,None]*A)*deg_inv_sqrt[None,:]
        D_img = X_img.shape[1]; D_con = self.concept_feats.shape[1]
        if D_con<D_img:
            con_padded = np.concatenate([self.concept_feats,np.zeros((C,D_img-D_con))],1)
        else:
            con_padded = self.concept_feats[:,:D_img]
        X_all = np.vstack([X_img,con_padded])
        labels_img = np.array([self.df_indexed.loc[iid,"Label_binary"] if iid in self.df_indexed.index else 0 for iid in img_ids])
        return X_all, A_norm, labels_img

df_indexed = df_pivot.set_index("Image")
graph_builder = ConstructGraph(df_indexed, SUBTYPE_COLS)
X_all_tr, A_tr, labels_tr = graph_builder.build_graph(X_tr, ids_tr)
X_all_val, A_val, labels_val = graph_builder.build_graph(X_val, ids_val)
X_all_te,  A_te,  labels_te  = graph_builder.build_graph(X_te, ids_te)

# Convert to torch
X_tr_t = torch.tensor(X_all_tr, dtype=torch.float32, device=DEVICE)
A_tr_t = torch.tensor(A_tr, dtype=torch.float32, device=DEVICE)
y_tr_img = torch.tensor(labels_tr, dtype=torch.long, device=DEVICE)
X_val_t = torch.tensor(X_all_val, dtype=torch.float32, device=DEVICE)
A_val_t = torch.tensor(A_val, dtype=torch.float32, device=DEVICE)
y_val_img = torch.tensor(labels_val, dtype=torch.long, device=DEVICE)
X_te_t = torch.tensor(X_all_te, dtype=torch.float32, device=DEVICE)
A_te_t = torch.tensor(A_te, dtype=torch.float32, device=DEVICE)
N_img_tr = X_tr.shape[0]; N_img_val = X_val.shape[0]; N_img_te = X_te.shape[0]

# -------------------- BrainGNN (Simple GNN block) --------------------
class SimpleGCNBlock(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim,hid_dim)
        self.lin2 = nn.Linear(hid_dim,hid_dim)
        self.dropout = nn.Dropout(0.4)
        self.bn = nn.LayerNorm(hid_dim)
    def forward(self,X,A):
        h = F.relu(self.bn(self.lin1(X)))
        h = A@h
        h = self.dropout(h)
        h = F.relu(self.lin2(h))
        h = A@h
        return h

class BrainGNN(nn.Module):
    def __init__(self, feat_dim,hid_dim=256,n_classes=2):
        super().__init__()
        self.encoder = nn.Linear(feat_dim,hid_dim)
        self.gcn = SimpleGCNBlock(hid_dim,hid_dim)
        self.classifier = nn.Linear(hid_dim,n_classes)
    def forward(self,X,A):
        H = F.relu(self.encoder(X))
        H_g = self.gcn(H,A)
        logits = self.classifier(H_g)
        return logits

# -------------------- TRAIN --------------------
model = BrainGNN(X_all_tr.shape[1], hid_dim=256).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR_GNN)
crit = nn.CrossEntropyLoss()

for ep in range(1,EPOCHS+1):
    model.train()
    opt.zero_grad()
    logits_all = model(X_tr_t, A_tr_t)
    logits_img = logits_all[:N_img_tr]
    loss = crit(logits_img, y_tr_img)
    loss.backward(); opt.step()
    if ep%5==0:
        print(f"Epoch {ep}/{EPOCHS} Loss:{loss.item():.4f}")

# -------------------- FINAL METRICS --------------------
model.eval()
with torch.no_grad():
    logits_te_all = model(X_te_t, A_te_t)
    preds = torch.argmax(logits_te_all[:N_img_te],dim=1).cpu().numpy()
    probs = F.softmax(logits_te_all[:N_img_te],dim=1)[:,1].cpu().numpy()
    acc = accuracy_score(labels_te, preds)
    prec = precision_score(labels_te, preds, zero_division=0)
    rec = recall_score(labels_te, preds, zero_division=0)
    f1s = f1_score(labels_te, preds, zero_division=0)
    auc = roc_auc_score(labels_te, probs)
    kappa = cohen_kappa_score(labels_te, preds)
print("\n=== FINAL METRICS ===")
print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1s:.4f}")
print(f"Cohen Kappa : {kappa:.4f}")


