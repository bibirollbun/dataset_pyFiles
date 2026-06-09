# ============================================================
# Mayo-Clinic-STRIP-AI  |  ConvNeXt Embeddings â�œ XGBoost (5-fold)
# ============================================================
!pip install -q timm tqdm xgboost albumentations --upgrade

import os, random, gc, warnings, time
from pathlib import Path
from collections import defaultdict

import numpy as np, pandas as pd
import cv2, tifffile
from tqdm.auto import tqdm

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A; import albumentations.pytorch
import timm, xgboost as xgb

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")

# -----------------------------
# 0ï¸�âƒ£  CONFIG
# -----------------------------
class CFG:
    TILE_SIZE  = 512
    BATCH      = 16
    NUM_WK     = 0
    SEED       = 42

    EPOCHS_CNN = 0     # no end-to-end training, just feature extractor
    DEBUG      = True
    DEBUG_FRAC = 0.5
    BY_PATIENT = True

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)

# -----------------------------
# 1ï¸�âƒ£  LOAD CSV
# -----------------------------
DATA = Path("/kaggle/input/mayo-clinic-strip-ai")
train_csv = pd.read_csv(DATA/"train.csv")
test_csv  = pd.read_csv(DATA/"test.csv")

if CFG.DEBUG:
    if CFG.BY_PATIENT:
        pats = train_csv.patient_id.unique()
        subset = np.random.RandomState(CFG.SEED).choice(
            pats, size=int(len(pats)*CFG.DEBUG_FRAC), replace=False)
        train_csv = train_csv[train_csv.patient_id.isin(subset)]
    else:
        train_csv = (train_csv.groupby("label", group_keys=False)
                     .apply(lambda x: x.sample(frac=CFG.DEBUG_FRAC,
                                               random_state=CFG.SEED))
                     .reset_index(drop=True))
    print(f"[DEBUG] using {len(train_csv)} tiles ({train_csv.patient_id.nunique()} patients)")

label_map = {l:i for i,l in enumerate(sorted(train_csv.label.unique()))}
train_csv["label_id"] = train_csv.label.map(label_map)

# -----------------------------
# 2ï¸�âƒ£  DATASET + TRANSFORMS
# -----------------------------
tfm = A.Compose([
    A.Resize(CFG.TILE_SIZE, CFG.TILE_SIZE),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    A.pytorch.ToTensorV2(),
])

class TileDS(Dataset):
    def __init__(self, df, split):
        self.df, self.split = df, split
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder = "train" if self.split=="train" else "test_images"
        img = tifffile.imread(DATA/folder/f"{row.image_id}.tif")
        if img.ndim==2: img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = tfm(image=img)["image"]
        y   = row.label_id if "label_id" in row else -1
        return img.float(), y, row.patient_id

# -----------------------------
# 3ï¸�âƒ£  FEATURE EXTRACTOR (ConvNeXt-Tiny)
# -----------------------------
fe_model = timm.create_model(
    "convnext_tiny_in22ft1k", pretrained=True,
    num_classes=0, global_pool="avg"
).to(CFG.DEVICE).eval()

@torch.no_grad()
def get_embeds(dl):
    feats,lbls,pids = [],[],[]
    for x,y,pid in tqdm(dl, leave=False):
        f = fe_model(x.to(CFG.DEVICE,non_blocking=True)).cpu().numpy()
        feats.append(f); lbls.extend(y.numpy()); pids.extend(pid)
    return np.vstack(feats), np.array(lbls), np.array(pids)

def pool_max(feats, labels, pids):
    bag = defaultdict(list)
    for f,y,p in zip(feats,labels,pids): bag[p].append((f,y))
    X,y,ids=[],[],[]
    for p,lst in bag.items():
        vecs,ys = zip(*lst)
        X.append(np.max(vecs,0)); y.append(ys[0]); ids.append(p)
    return np.vstack(X), np.array(y), np.array(ids)

# -----------------------------
# 4ï¸�âƒ£  5-FOLD CV + XGBoost
# -----------------------------
outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
oof_pred, oof_true = [], []
fold_models        = []

params = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    eta=0.03, max_depth=7,
    subsample=0.8, colsample_bytree=0.6,
    lambda_=1.0, alpha=0.3,
    min_child_weight=3,
    seed=CFG.SEED
)
EMBED_BATCH = 8
def build_embed_loader(df, split):
    return DataLoader(TileDS(df, split),
                      batch_size=EMBED_BATCH,
                      shuffle=False,
                      num_workers=0,
                      pin_memory=False)
for fold,(tr_idx,va_idx) in enumerate(
        outer.split(train_csv, train_csv.label_id, train_csv.patient_id)):
    print(f"\nâ˜… Fold {fold}")

    tr_df = train_csv.iloc[tr_idx].reset_index(drop=True)
    va_df = train_csv.iloc[va_idx].reset_index(drop=True)

    tr_dl = DataLoader(TileDS(tr_df,"train"),
                       batch_size=CFG.BATCH, shuffle=False,
                       num_workers=CFG.NUM_WK)
    va_dl = DataLoader(TileDS(va_df,"train"),
                       batch_size=CFG.BATCH, shuffle=False,
                       num_workers=CFG.NUM_WK)

    tr_f,tr_y,_ = pool_max(*get_embeds(tr_dl))
    va_f,va_y,va_pid = pool_max(*get_embeds(va_dl))

    pos_ratio = (tr_y==1).mean()
    params["scale_pos_weight"] = (1-pos_ratio)/pos_ratio

    mdl = xgb.train(params,
                    xgb.DMatrix(tr_f,tr_y),
                    num_boost_round=2000,
                    evals=[(xgb.DMatrix(va_f,va_y),"val")],
                    early_stopping_rounds=200,
                    verbose_eval=100)
    preds = mdl.predict(xgb.DMatrix(va_f))
    auc   = roc_auc_score(va_y, preds)
    print(f"  fold AUC = {auc:.3f}")

    oof_pred.extend(preds); oof_true.extend(va_y)
    fold_models.append(mdl)

print("\nOOF AUC =", roc_auc_score(oof_true, oof_pred))

# -----------------------------
# 5ï¸�âƒ£  PREDICT TEST & SUBMIT
# -----------------------------
test_dl = DataLoader(TileDS(test_csv,"test"),
                     batch_size=CFG.BATCH, shuffle=False,
                     num_workers=CFG.NUM_WK)

f_test, _, pid_test = pool_max(*get_embeds(test_dl))

bag = defaultdict(list)
for mdl in fold_models:
    bag_p = mdl.predict(xgb.DMatrix(f_test))
    for p,pr in zip(pid_test, bag_p): bag[p].append(pr)

sub = pd.DataFrame({
    "patient_id": list(bag.keys()),
    "LAA": [np.mean(v) for v in bag.values()]
})
sub["CE"] = 1 - sub["LAA"]
sub[["CE","LAA"]] = sub[["CE","LAA"]].clip(1e-15,1-1e-15)
sub = sub[["patient_id","CE","LAA"]]
sub.to_csv("submission.csv", index=False)
print("\nâœ… submission.csv saved!", sub.head())

