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
# DGPMIL – Full Pipeline with Embeddings + Graph
# ==========================================================
import os, time, random
import numpy as np, pandas as pd
from tqdm import tqdm
import pydicom, cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

BASE = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection"
CSV_PATH = os.path.join(BASE, "stage_2_train.csv")
TRAIN_DIR = os.path.join(BASE, "stage_2_train")

EMB_PATH = "/kaggle/working/embeddings"
os.makedirs(EMB_PATH, exist_ok=True)

SAMPLES_PER_CLASS = 1200
IMG_SIZE = 256
BATCH = 32
EMB_BATCH = 32
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-5
K_NEIGH = 15
LABEL_SMOOTH = 0.05

# ---------------- Helper Functions ----------------
def window_image(img, wl, ww):
    minv = wl - ww/2.0
    maxv = wl + ww/2.0
    out = (img - minv) / (maxv - minv + 1e-9)
    return np.clip(out, 0.0, 1.0)

def make_3ch_from_dcm(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    ch1 = window_image(img, 40, 80)
    ch2 = window_image(img, 80, 200)
    ch3 = window_image(img, 600, 2800)
    ch1 = cv2.resize(ch1,(IMG_SIZE,IMG_SIZE))
    ch2 = cv2.resize(ch2,(IMG_SIZE,IMG_SIZE))
    ch3 = cv2.resize(ch3,(IMG_SIZE,IMG_SIZE))
    return np.stack([ch1,ch2,ch3],axis=0)

# ---------------- Load & Balance Labels ----------------
df = pd.read_csv(CSV_PATH)
df["Image"] = df["ID"].apply(lambda x: x.split("_")[1])
df["Subtype"] = df["ID"].apply(lambda x: x.split("_")[2])

df_group = df.groupby(["Image","Subtype"], as_index=False)["Label"].max()
df_pivot = df_group.pivot(index="Image", columns="Subtype", values="Label").reset_index().fillna(0)
df_pivot["Label_binary"] = df_pivot.iloc[:,1:].max(axis=1).astype(int)

# Balance classes
samples = []
for lbl in [0,1]:
    sub = df_pivot[df_pivot["Label_binary"]==lbl]
    n = min(len(sub), SAMPLES_PER_CLASS)
    samples.append(sub.sample(n, random_state=SEED))
df_bal = pd.concat(samples).reset_index(drop=True)

subtype_cols = [c for c in df_bal.columns if c not in ["Image","Label_binary"]]
C = len(subtype_cols)
df_indexed = df_bal.set_index("Image")
B_full = df_bal[subtype_cols].values.astype(float)

# ---------------- Dataset ----------------
class RSDataset(Dataset):
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
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_id = self.df.loc[idx,"Image"]
        label = int(self.df.loc[idx,"Label_binary"])
        path = os.path.join(self.img_dir,f"ID_{img_id}.dcm")
        img3 = make_3ch_from_dcm(path)
        img3 = (img3*255).astype(np.uint8)
        tf = self.aug_tf if self.augment else self.eval_tf
        img_t = tf(np.transpose(img3,(1,2,0)))
        return img_t, label, img_id

# ---------------- EfficientNet Embeddings ----------------
def make_effnet(num_classes=2):
    eff = models.efficientnet_b0(pretrained=True)
    in_feat = eff.classifier[1].in_features
    eff.classifier[1] = nn.Linear(in_feat,num_classes)
    return eff

effnet = make_effnet().to(DEVICE)
for _, p in effnet.named_parameters(): p.requires_grad = True

class EffEmbedder(nn.Module):
    def __init__(self, eff):
        super().__init__()
        self.features = eff.features
        self.avgpool = eff.avgpool
        self.dropout = eff.classifier[0]
        self.feat_dim = eff.classifier[1].in_features
    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        return x

embedder = EffEmbedder(effnet).to(DEVICE)
embedder.eval()

def extract_embeddings(df_subset, batch_size=EMB_BATCH):
    ds = RSDataset(df_subset, TRAIN_DIR, augment=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    embs, ids = [], []
    with torch.no_grad():
        for imgs, _, id_batch in tqdm(dl):
            imgs = imgs.to(DEVICE)
            feat = embedder(imgs)
            embs.append(feat.cpu().numpy())
            ids.extend(id_batch)
    X = np.vstack(embs)
    # Save embeddings
    for i, img_id in enumerate(ids):
        np.save(os.path.join(EMB_PATH,f"{img_id}.npy"), X[i])
    return X, ids

# ---------------- Precompute Embeddings ----------------
train_df, temp_df = train_test_split(df_bal, test_size=0.3, stratify=df_bal["Label_binary"], random_state=SEED)
val_df, test_df  = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label_binary"], random_state=SEED)

X_tr, ids_tr = extract_embeddings(train_df)
X_val, ids_val = extract_embeddings(val_df)
X_te, ids_te = extract_embeddings(test_df)

# ---------------- Graph Construction ----------------
def construct_graph(X_images, ids_images, k_neighbors=K_NEIGH):
    N = X_images.shape[0]
    sim = cosine_similarity(X_images)
    np.fill_diagonal(sim,0)
    A_ii = np.zeros_like(sim)
    for i in range(N):
        nbrs = np.argsort(sim[i])[-k_neighbors:]
        A_ii[i,nbrs] = sim[i,nbrs]
    A_ii = np.minimum(A_ii,A_ii.T)
    B_img = df_indexed.loc[ids_images,subtype_cols].values.astype(float)
    row_sum = B_img.sum(axis=1,keepdims=True)
    B_norm = B_img/(row_sum+1e-6); B_norm[row_sum.squeeze()==0]=0.0
    A_ic = B_norm; A_ci = A_ic.T.copy()
    p = B_full.mean(axis=0)+1e-9; co = B_full.T @ B_full
    pmi = np.log((co+1e-6)/(p[:,None]*p[None,:])); pmi[pmi<0]=0
    A_cc = pmi
    top = np.concatenate([A_ii,A_ic],axis=1)
    bottom = np.concatenate([A_ci,A_cc],axis=1)
    A = np.concatenate([top,bottom],axis=0)
    A += np.eye(A.shape[0])*1e-6
    deg = A.sum(axis=1)
    D_inv = np.diag(1/np.sqrt(deg+1e-9))
    A_norm = D_inv @ A @ D_inv
    X_all = np.vstack([X_images, np.zeros((C,X_images.shape[1]))])
    X_all_t = torch.tensor(X_all,dtype=torch.float32,device=DEVICE)
    A_norm_t = torch.tensor(A_norm,dtype=torch.float32,device=DEVICE)
    img_idx = np.arange(N)
    concept_idx = np.arange(N,N+C)
    labels_img = np.array([df_indexed.loc[iid,"Label_binary"] for iid in ids_images],dtype=int)
    return X_all_t,A_norm_t,img_idx,concept_idx,labels_img

X_all_tr,A_tr,img_idx_tr,concept_idx_tr,labels_tr = construct_graph(X_tr,ids_tr)
X_all_val,A_val,img_idx_val,concept_idx_val,labels_val = construct_graph(X_val,ids_val)
X_all_te,A_te,img_idx_te,concept_idx_te,labels_te = construct_graph(X_te,ids_te)

# ---------------- DGPMIL Model ----------------
class DGPMIL(nn.Module):
    def __init__(self, in_feats, hidden=256, out_feats=2, num_concepts=C):
        super().__init__()
        self.num_concepts = num_concepts
        self.fc1 = nn.Linear(in_feats, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(hidden, out_feats)
    def encode(self,X_all,A_norm,concept_idx):
        X = F.relu(self.fc1(X_all))
        X = A_norm @ X + X
        X = F.relu(self.fc2(X))
        X = self.dropout(X)
        return X
    def forward(self,X_all,A_norm,concept_idx,img_idx):
        H_all = self.encode(X_all,A_norm,concept_idx)
        H_img = H_all[img_idx]
        logits = self.classifier(H_img)
        return logits,H_img,H_all

# ---------------- Train DGPMIL ----------------
model = DGPMIL(X_all_tr.shape[1]).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
crit = nn.CrossEntropyLoss()

def eval_model(model,X_all,A_norm,img_idx,concept_idx,labels_img):
    model.eval()
    with torch.no_grad():
        logits,_,_ = model(X_all,A_norm,concept_idx,img_idx)
        preds = torch.argmax(logits,1).cpu().numpy()
        probs = F.softmax(logits,dim=1)[:,1].cpu().numpy()
        labels = labels_img
        acc = accuracy_score(labels,preds)
        prec = precision_score(labels,preds,zero_division=0)
        rec = recall_score(labels,preds,zero_division=0)
        f1 = f1_score(labels,preds,zero_division=0)
        try: auc = roc_auc_score(labels,probs)
        except: auc = float("nan")
    return acc,prec,rec,f1,auc

best_val_acc = -1
for ep in range(1,EPOCHS+1):
    model.train()
    opt.zero_grad()
    labels_tr_t = torch.tensor(labels_tr,dtype=torch.long,device=DEVICE)
    logits,_,_ = model(X_all_tr,A_tr,concept_idx_tr,img_idx_tr)
    loss = crit(logits,labels_tr_t)
    loss.backward()
    opt.step()

    tr_acc,tr_prec,tr_rec,tr_f1,tr_auc = eval_model(model,X_all_tr,A_tr,img_idx_tr,concept_idx_tr,labels_tr)
    val_acc,val_prec,val_rec,val_f1,val_auc = eval_model(model,X_all_val,A_val,img_idx_val,concept_idx_val,labels_val)
    print(f"Ep {ep}/{EPOCHS} | Loss:{loss.item():.4f} | TrAcc:{tr_acc:.4f} ValAcc:{val_acc:.4f} | ValF1:{val_f1:.4f} ValAUC:{val_auc:.4f}")

te_acc,te_prec,te_rec,te_f1,te_auc = eval_model(model,X_all_te,A_te,img_idx_te,concept_idx_te,labels_te)
print(f"\nTest Acc:{te_acc:.4f} | F1:{te_f1:.4f} | AUC:{te_auc:.4f}")


