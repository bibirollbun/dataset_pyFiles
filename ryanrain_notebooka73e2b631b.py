# =========================================
# Mayo-Clinic-STRIP-AI  |  CNN & XGBoost NB
# =========================================
import os, random, time, warnings, gc, joblib
from pathlib import Path
from collections import defaultdict

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A; import albumentations.pytorch
import tifffile, cv2, timm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss
import xgboost as xgb
from tqdm.auto import tqdm 
warnings.filterwarnings("ignore")

# ---------------------
# 0️⃣  配置
# ---------------------
class CFG:
    COMP          = "mayo-clinic-strip-ai"
    TILE_SIZE     = 640          # 小一点防 OOM
    BATCH_SIZE    = 24
    EPOCHS        = 16
    LR            = 3e-4 
    IMG_MEAN      = (0.485,0.456,0.406)
    IMG_STD       = (0.229,0.224,0.225)
    NUM_WORKERS   = 0
    SEED          = 42
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
    DEBUG         = True      # ← True: 只抽样；False: 全量训练
    DEBUG_FRAC    = 0.5      # 抽多少？5 %的 tile
    BY_PATIENT    = True      # True ⇒ 抽患者；False ⇒ 抽 tile
    
cfg = CFG()
random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)

# ----------------------------------------
# 1️⃣  数据加载
# ----------------------------------------
data_dir  = Path("/kaggle/input/mayo-clinic-strip-ai")
train_csv = pd.read_csv(data_dir / "train.csv")

# ---------- DEBUG 抽样 ----------
if CFG.DEBUG:
    if CFG.BY_PATIENT:
        pats = train_csv.patient_id.unique()
        rnd  = np.random.RandomState(CFG.SEED)
        sel  = rnd.choice(
            pats, size=int(len(pats)*CFG.DEBUG_FRAC), replace=False)
        train_csv = train_csv[train_csv.patient_id.isin(sel)]
    else:  # tile-level 分层抽样
        train_csv = (train_csv
            .groupby("label", group_keys=False)
            .apply(lambda x: x.sample(frac=CFG.DEBUG_FRAC,
                                      random_state=CFG.SEED))
            .reset_index(drop=True))
    print(f"[DEBUG] using {len(train_csv)} tiles ({len(train_csv.patient_id.unique())} patients)")
test_csv  = pd.read_csv(data_dir / "test.csv")

label_map = {lbl:i for i,lbl in enumerate(sorted(train_csv["label"].unique()))}
train_csv["label_id"] = train_csv["label"].map(label_map)
num_classes = len(label_map)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
tr_idx, va_idx = next(sgkf.split(train_csv, train_csv.label_id, train_csv.patient_id))
train_df = train_csv.iloc[tr_idx].reset_index(drop=True)
val_df   = train_csv.iloc[va_idx].reset_index(drop=True)

# ----------------------------------------
# 2️⃣  Dataset
# ----------------------------------------
tfm_train = A.Compose([
    A.RandomResizedCrop(size=(cfg.TILE_SIZE, cfg.TILE_SIZE), scale=(0.7,1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(0.1,0.1,0.1,0.05,p=0.3),           # add mild color jitter
    A.Normalize(cfg.IMG_MEAN, cfg.IMG_STD),
    A.pytorch.ToTensorV2(),
])

tfm_val = A.Compose([
    # 下面两种写法均可，二选一
    A.Resize(height=cfg.TILE_SIZE, width=cfg.TILE_SIZE),      # 原写法保留
    # A.Resize(size=(cfg.TILE_SIZE, cfg.TILE_SIZE)),          # 或也用 size=
    A.Normalize(cfg.IMG_MEAN, cfg.IMG_STD),
    A.pytorch.ToTensorV2(),
])

class TileDataset(Dataset):
    def __init__(self, df, transforms=None, split="train"):
        self.df, self.tfm, self.split = df, transforms, split
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder = "train" if self.split=="train" else "test"
        img_path = data_dir / folder / f"{row.image_id}.tif"
        img = tifffile.imread(img_path)
        if img.ndim==2: img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.tfm: img=self.tfm(image=img)["image"]
        label = row.label_id if "label_id" in row else -1
        return img.float(), torch.tensor(label).long(), row.patient_id

train_dl = DataLoader(TileDataset(train_df, tfm_train, "train"),
                      batch_size=cfg.BATCH_SIZE, shuffle=True,
                      num_workers=cfg.NUM_WORKERS, pin_memory=True)

val_dl   = DataLoader(TileDataset(val_df, tfm_val, "train"),
                      batch_size=cfg.BATCH_SIZE, shuffle=False,
                      num_workers=cfg.NUM_WORKERS, pin_memory=True)

test_dl  = DataLoader(TileDataset(test_csv, tfm_val, "test"),
                      batch_size=cfg.BATCH_SIZE, shuffle=False,
                      num_workers=cfg.NUM_WORKERS, pin_memory=True)

# ----------------------------------------
# 3️⃣  端到端 CNN  (可选)
# ----------------------------------------
loss_type   = "focal"      # "ce" | "weighted" | "focal"
# 计算每个类别的出现频率（在 train_df 上）
freq = train_df.label_id.value_counts(normalize=True).sort_index()  # e.g. [0.85, 0.15]

# 反比频率做权重；转换到 GPU
w = torch.tensor(1.0 / freq.values, dtype=torch.float32).to(cfg.DEVICE)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            ce_loss = ce_loss * at
        loss = ((1-pt)**self.gamma * ce_loss).mean()
        return loss
# Focal Loss（或 Weighted CE）就能用到 alpha / weight
criterion = FocalLoss(alpha=w, gamma=2)
train_mode  = "cnn"           # "cnn" | "xgb"  (决定提交用哪个模型)



if train_mode == "cnn":
    model = timm.create_model("resnet50d", pretrained=True, num_classes=num_classes).to(cfg.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
    )
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "weighted":
        freq = train_df.label_id.value_counts(normalize=True).sort_index()
        w = torch.tensor(1.0/freq.values, dtype=torch.float32).to(cfg.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:  # focal
        alpha = None
        criterion = FocalLoss(alpha=alpha, gamma=2)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.EPOCHS)

    def run_epoch(dl, train=True):
        model.train() if train else model.eval()
        total, correct, loss_sum = 0,0,0
        for x, y, _ in tqdm(dl, leave=False, desc="batch"):
            x,y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            with torch.set_grad_enabled(train):
                out = model(x); loss=criterion(out,y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            pred = out.argmax(1); total+=y.size(0); correct+=(pred==y).sum().item()
            loss_sum += loss.item()*y.size(0)
        return correct/total, loss_sum/total

    for ep in range(cfg.EPOCHS):
        tr_acc, tr_loss = run_epoch(train_dl,True)
        va_acc, _ = run_epoch(val_dl,False)
        sch.step()
        print(f"Epoch {ep+1}/{cfg.EPOCHS}  train-acc={tr_acc:.3f}  val-acc={va_acc:.3f}")

# ----------------------------------------
# 4️⃣  CNN → XGBoost
# ----------------------------------------
if train_mode == "xgb":
    fe_model = timm.create_model(
        "resnet50d",  # or "convnext_tiny"
        pretrained=True, num_classes=0, global_pool="avg"
    ).to(cfg.DEVICE).eval()

    def get_embed(dl):
        feats,lbls,pids = [],[],[]
        with torch.no_grad():
            for x, y, pid in tqdm(dl, leave=False, desc="embed"):
                f = fe_model(x.to(cfg.DEVICE)).cpu().numpy()
                feats.append(f); lbls.extend(y.numpy()); pids.extend(pid)
        return np.vstack(feats), np.array(lbls), np.array(pids)

    def agg(feats, labels, pids):
        bag=defaultdict(list)
        for f,y,p in zip(feats,labels,pids): bag[p].append((f,y))
        X,y,ids=[],[],[]
        for p,lst in bag.items():
            vecs,ys=zip(*lst)
            X.append(np.mean(vecs,0)); y.append(ys[0]); ids.append(p)
        return np.vstack(X), np.array(y), np.array(ids)

    print("⏳  Extracting embeddings …")
    tr_f,tr_y,tr_pid = agg(*get_embed(train_dl))
    va_f,va_y,va_pid = agg(*get_embed(val_dl))

    pos_ratio = (tr_y==1).mean()
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        eta=0.05,max_depth=6,
        subsample=0.9,colsample_bytree=0.5,
        seed=cfg.SEED,
        scale_pos_weight=(1-pos_ratio)/pos_ratio  # 处理不平衡
    )
    dtrain,dval = xgb.DMatrix(tr_f,tr_y), xgb.DMatrix(va_f,va_y)
    print("⏳  Training XGBoost …")
    params.update({
    "eta":0.03, "max_depth":7,
    "subsample":0.8,"colsample_bytree":0.6,
    "lambda":1.0,"alpha":0.3,
    "min_child_weight":3,
    "scale_pos_weight":(1-pos_ratio)/pos_ratio,
    })
    model_xgb = xgb.train(params,dtrain,500,
                          evals=[(dtrain,"tr"),(dval,"val")],
                          early_stopping_rounds=200,verbose_eval=50)
    pred_val = model_xgb.predict(dval)
    print(f"Val AUC {roc_auc_score(va_y,pred_val):.4f}  logloss {log_loss(va_y,pred_val):.4f}")

# ----------------------------------------
# 5️⃣  生成 submission.csv
# ----------------------------------------
def make_submission():
    # --- 抽取 test patient-level ---------------
    if train_mode=="cnn":
        model.eval()
        probs_ce,probs_laa,patient_ids=[],[],[]
        with torch.no_grad():
            for x,_,pid in test_dl:
                p=model(x.to(cfg.DEVICE)).softmax(1).cpu().numpy()
                probs_ce.extend(p[:,label_map["CE"]])
                probs_laa.extend(p[:,label_map["LAA"]])
                patient_ids.extend(pid)
    else:
        f_test,_,pid_test = get_embed(test_dl)
        X_test,_,ids = agg(f_test, np.zeros_like(pid_test), pid_test)
        p_laa = model_xgb.predict(xgb.DMatrix(X_test))
        p_ce  = 1 - p_laa
        patient_ids, probs_ce, probs_laa = ids, p_ce, p_laa

    sub = pd.DataFrame({"patient_id":patient_ids,
                        "CE":probs_ce,"LAA":probs_laa})
    sub = sub.groupby("patient_id")[["CE","LAA"]].mean().reset_index()
    sub[["CE","LAA"]] = sub[["CE","LAA"]].clip(1e-15,1-1e-15)
    sub.to_csv("submission.csv",index=False)
    print("✅ submission.csv saved! shape:",sub.shape)
    print(sub.head())
make_submission()

