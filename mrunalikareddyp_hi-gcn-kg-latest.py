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
# GATE Model + Knowledge Graph (RSNA Intracranial Hemorrhage)
# ===============================================================
import os, time, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom, cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

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
LR_PROBE = 2e-4
LR_AGCL = 1e-3
K_NEIGH = 16
PCA_DIM = 256
KG_EMB_DIM = 16
TEMP = 0.2
ALPHA_CON = 1.0
ALPHA_SUP = 1.0

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
    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))
    return np.stack([arr, arr, arr], axis=-1)

# -------------------- LOAD & BALANCE LABELS --------------------
df = pd.read_csv(CSV_PATH)
df["Image"] = df["ID"].apply(lambda x: x.split("_")[1])
df["Subtype"] = df["ID"].apply(lambda x: x.split("_")[2])
df2 = df.groupby(["Image","Subtype"], as_index=False)["Label"].max()
df_pivot = df2.pivot(index="Image", columns="Subtype", values="Label").reset_index().fillna(0)
df_pivot["Label_binary"] = df_pivot.iloc[:,1:].max(axis=1).astype(int)

pos = df_pivot[df_pivot["Label_binary"]==1].sample(n=SAMPLES_PER_CLASS, random_state=SEED)
neg = df_pivot[df_pivot["Label_binary"]==0].sample(n=SAMPLES_PER_CLASS, random_state=SEED)
df_bal = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)

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
train_df, temp_df = train_test_split(df_bal, test_size=0.3, stratify=df_bal["Label_binary"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label_binary"], random_state=SEED)

train_ds = RSNADataset(train_df, IMG_DIR, augment=True)
val_ds = RSNADataset(val_df, IMG_DIR, augment=False)
test_ds = RSNADataset(test_df, IMG_DIR, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

# -------------------- RESNET18 PROBE --------------------
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
embedding_dim = resnet.fc.in_features
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE)

probe = nn.Linear(embedding_dim, 2).to(DEVICE)

# freeze except layer4
for name,p in resnet.named_parameters():
    p.requires_grad = False
    if "layer4" in name:
        p.requires_grad = True
params_ft = list(filter(lambda p: p.requires_grad, resnet.parameters())) + list(probe.parameters())
opt_ft = torch.optim.AdamW(params_ft, lr=LR_PROBE)
crit_ce = nn.CrossEntropyLoss()

# -------------------- PROBE FINE-TUNE --------------------
best_val=-1; best_state=None
for ep in range(FT_EPOCHS):
    resnet.train(); probe.train()
    running_loss=0; preds=[]; labs=[]
    for imgs, lab, _ in train_loader:
        imgs, lab = imgs.to(DEVICE), lab.to(DEVICE)
        opt_ft.zero_grad()
        feats = resnet(imgs)
        logits = probe(feats)
        loss = crit_ce(logits, lab)
        loss.backward(); opt_ft.step()
        running_loss += loss.item()*imgs.size(0)
        preds.extend(torch.argmax(logits,1).cpu().numpy()); labs.extend(lab.cpu().numpy())
    train_acc = accuracy_score(labs,preds)

    # validation
    resnet.eval(); probe.eval()
    v_preds=[]; v_labs=[]
    with torch.no_grad():
        for imgs, lab, _ in val_loader:
            imgs, lab = imgs.to(DEVICE), lab.to(DEVICE)
            logits = probe(resnet(imgs))
            v_preds.extend(torch.argmax(logits,1).cpu().numpy())
            v_labs.extend(lab.cpu().numpy())
    val_acc = accuracy_score(v_labs,v_preds)
    if val_acc>best_val: best_val=val_acc; best_state=(resnet.state_dict(), probe.state_dict())
    print(f"FT Epoch {ep+1}/{FT_EPOCHS} Loss:{running_loss/len(train_ds):.4f} TrainAcc:{train_acc:.4f} ValAcc:{val_acc:.4f}")

if best_state:
    resnet.load_state_dict(best_state[0]); probe.load_state_dict(best_state[1])

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

# PCA
if PCA_DIM is not None and PCA_DIM<X_tr.shape[1]:
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_tr = pca.fit_transform(X_tr)
    X_val = pca.transform(X_val)
    X_te = pca.transform(X_te)

# -------------------- Knowledge Graph --------------------
df_indexed = df_pivot.set_index("Image")
concept_feats = np.random.normal(0,0.01,(len(SUBTYPE_COLS), KG_EMB_DIM))

class ConstructGraph:
    def __init__(self, X_img, img_ids, concept_feats):
        self.X_img = X_img
        self.img_ids = img_ids
        self.concept_feats = concept_feats
        self.N = X_img.shape[0]; self.C = concept_feats.shape[0]

    def build_img_knn_weighted(self, k=K_NEIGH):
        sim = cosine_similarity(self.X_img); np.fill_diagonal(sim,0)
        A=np.zeros_like(sim)
        for i in range(self.N):
            idx = np.argsort(sim[i])[-k:]
            A[i, idx] = sim[i, idx]
        A = np.maximum(A, A.T) + np.eye(self.N)*1e-6
        deg = A.sum(1); deg_inv_sqrt=1.0/np.sqrt(deg)
        return (deg_inv_sqrt[:,None]*A)*deg_inv_sqrt[None,:]

    def build_combined_graph(self):
        A_ii = self.build_img_knn_weighted()
        A_ic = np.zeros((self.N, self.C))
        for i,iid in enumerate(self.img_ids):
            if iid in df_indexed.index:
                A_ic[i,:] = df_indexed.loc[iid,SUBTYPE_COLS].values
        A_ci = A_ic.T
        simc = cosine_similarity(self.concept_feats); np.fill_diagonal(simc,0)
        A_cc = simc
        top = np.concatenate([A_ii,A_ic],1); bottom = np.concatenate([A_ci,A_cc],1)
        A = np.concatenate([top,bottom],0) + np.eye(self.N+self.C)*1e-6
        deg = A.sum(1); deg_inv_sqrt=1.0/np.sqrt(deg)
        A_norm = (deg_inv_sqrt[:,None]*A)*deg_inv_sqrt[None,:]

        # combine features
        D_img = self.X_img.shape[1]; D_con = self.concept_feats.shape[1]
        if D_con<D_img: con_padded = np.concatenate([self.concept_feats, np.zeros((self.C,D_img-D_con))],1)
        else: con_padded = self.concept_feats[:,:D_img]
        X_all = np.vstack([self.X_img, con_padded])

        labels_img = np.array([df_indexed.loc[iid,"Label_binary"] if iid in df_indexed.index else 0 for iid in self.img_ids])
        return X_all, A_norm, labels_img

# -------------------- Build Graphs --------------------
graph_tr = ConstructGraph(X_tr, ids_tr, concept_feats)
X_all_tr, A_tr, labels_tr = graph_tr.build_combined_graph()

graph_val = ConstructGraph(X_val, ids_val, concept_feats)
X_all_val, A_val, labels_val = graph_val.build_combined_graph()

graph_te = ConstructGraph(X_te, ids_te, concept_feats)
X_all_te, A_te, labels_te = graph_te.build_combined_graph()

# torch tensors
X_tr_t = torch.tensor(X_all_tr, dtype=torch.float32, device=DEVICE)
A_tr_t = torch.tensor(A_tr, dtype=torch.float32, device=DEVICE)
y_tr_img = torch.tensor(labels_tr, dtype=torch.long, device=DEVICE)
X_val_t = torch.tensor(X_all_val, dtype=torch.float32, device=DEVICE)
A_val_t = torch.tensor(A_val, dtype=torch.float32, device=DEVICE)
y_val_img = torch.tensor(labels_val, dtype=torch.long, device=DEVICE)
X_te_t = torch.tensor(X_all_te, dtype=torch.float32, device=DEVICE)
A_te_t = torch.tensor(A_te, dtype=torch.float32, device=DEVICE)
N_img_tr = X_tr.shape[0]; N_img_val = X_val.shape[0]; N_img_te = X_te.shape[0]

# -------------------- GATE MODEL --------------------
class SimpleGCNBlock(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.lin1=nn.Linear(in_dim,hid_dim)
        self.lin2=nn.Linear(hid_dim,hid_dim)
        self.dropout=nn.Dropout(0.4)
        self.bn=nn.LayerNorm(hid_dim)
    def forward(self,X,A):
        h=F.relu(self.bn(self.lin1(X)))
        h=A@h
        h=self.dropout(h)
        h=F.relu(self.lin2(h))
        h=A@h
        return h

class GATE_Model(nn.Module):
    def __init__(self, feat_dim,hid=512,n_classes=2):
        super().__init__()
        self.encoder=nn.Linear(feat_dim,hid)
        self.gate_proj=nn.Linear(feat_dim,1)
        self.gcn=SimpleGCNBlock(hid,hid)
        self.classifier=nn.Linear(hid,n_classes)
    def forward(self,X_all,A_all):
        H = F.relu(self.encoder(X_all))
        gate = torch.sigmoid(self.gate_proj(X_all)).squeeze(-1)
        Hg = self.gcn(H,A_all)
        logits = self.classifier(Hg)
        return logits,H,gate

# -------------------- CONTRASTIVE LOSS --------------------
def nt_xent_loss(Z1,Z2,temperature=TEMP):
    N = Z1.shape[0]
    z = torch.cat([Z1,Z2],0)
    sim = (z@z.T)/temperature
    sim_exp = torch.exp(sim - torch.max(sim,1,keepdim=True)[0])
    mask = (~torch.eye(2*N, dtype=bool, device=Z1.device)).float()
    denom = (sim_exp*mask).sum(1)
    positives = torch.exp(torch.sum(Z1*Z2,1)/temperature)
    positives = torch.cat([positives,positives],0)
    loss = -torch.log(positives/denom)
    return loss.mean()

# -------------------- TRAIN --------------------
model = GATE_Model(X_all_tr.shape[1], hid=256, n_classes=2).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR_AGCL)
crit = nn.CrossEntropyLoss()

for ep in range(1,EPOCHS+1):
    t0 = time.time(); model.train(); opt.zero_grad()
    logits_all,H_all,gate_all = model(X_tr_t,A_tr_t)
    logits_img = logits_all[:N_img_tr]
    loss_sup = crit(logits_img,y_tr_img)

    # contrastive views
    X_view1 = X_tr_t + 0.01*torch.randn_like(X_tr_t)
    mask = (torch.rand_like(X_tr_t)>0.1).float()
    X_view1 = X_view1*mask
    X_view2 = X_tr_t*(1+0.02*torch.randn_like(X_tr_t))
    model.eval()
    with torch.no_grad():
        Z1 = F.normalize(F.relu(model.encoder(X_view1)),dim=1)
        Z2 = F.normalize(F.relu(model.encoder(X_view2)),dim=1)
    model.train()
    Z1_img = Z1[:N_img_tr]; Z2_img = Z2[:N_img_tr]
    loss_con = nt_xent_loss(Z1_img,Z2_img)
    loss = ALPHA_SUP*loss_sup + ALPHA_CON*loss_con
    loss.backward(); opt.step()

    print(f"Epoch {ep}/{EPOCHS} Loss:{loss.item():.4f} Time:{time.time()-t0:.1f}s")

# -------------------- FINAL EVALUATION --------------------
model.eval()
with torch.no_grad():
    t0 = time.time()
    logits_te_all, _, _ = model(X_te_t, A_te_t)
    preds = torch.argmax(logits_te_all[:N_img_te], dim=1).cpu().numpy()
    probs = F.softmax(logits_te_all[:N_img_te], dim=1)[:,1].cpu().numpy()
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


