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


import pandas as pd, numpy as np, os
import cudf

PATH = "/kaggle/input/playground-series-s5e8/"
train = cudf.read_csv(f"{PATH}train.csv").set_index('id')
print("Train shape", train.shape )
train.head()


test = cudf.read_csv(f"{PATH}test.csv").set_index('id')
test['y'] = np.random.randint(0, 2, len(test))
print("Test shape", test.shape )
test.head()


orig = cudf.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv",delimiter=";")
orig['y'] = orig.y.map({'yes':1,'no':0})
orig['id'] = (np.arange(len(orig))+1e6).astype('int')
orig = orig.set_index('id')
print("Original data shape", orig.shape )
orig.head()
combine = cudf.concat([train,test,orig],axis=0)
print("Combined data shape", combine.shape )


CATS = []
NUMS = []
for c in combine.columns[:-1]:
    t = "CAT"
    if combine[c].dtype=='object':
        CATS.append(c)
    else:
        NUMS.append(c)
        t = "NUM"
    n = combine[c].nunique()
    na = combine[c].isna().sum()
    print(f"[{t}] {c} has {n} unique and {na} NA")
print("CATS:", CATS )
print("NUMS:", NUMS )


CATS1 = []
SIZES = {}
for c in NUMS + CATS:
    n = c
    if c in NUMS: 
        n = f"{c}2"
        CATS1.append(n)
    combine[n],_ = combine[c].factorize()
    SIZES[n] = combine[n].max()+1

    combine[c] = combine[c].astype('int32')
    combine[n] = combine[n].astype('int32')

print("New CATS:", CATS1 )
print("Cardinality of all CATS:", SIZES )
from itertools import combinations

pairs = combinations(CATS + CATS1, 2)
new_cols = {}
CATS2 = []

for c1, c2 in pairs:
    name = "_".join(sorted((c1, c2)))
    new_cols[name] = combine[c1] * SIZES[c2] + combine[c2]
    CATS2.append(name)
if new_cols:
    new_df = cudf.DataFrame(new_cols)         
    combine = cudf.concat([combine, new_df], axis=1) 

print(f"Created {len(CATS2)} new CAT columns")


CE = []
CC = CATS+CATS1+CATS2
combine['i'] = np.arange( len(combine) )

print(f"Processing {len(CC)} columns... ",end="")
for i,c in enumerate(CC):
    if i%10==0: print(f"{i}, ",end="")
    tmp = combine.groupby(c).y.count()
    tmp = tmp.astype('int32')
    tmp.name = f"CE_{c}"
    CE.append( f"CE_{c}" )
    combine = combine.merge(tmp, on=c, how='left')
combine = combine.sort_values('i')
print()
train = combine.iloc[:len(train)]
test = combine.iloc[len(train):len(train)+len(test)]
orig = combine.iloc[-len(orig):]
del combine
print("Train shape", train.shape,"Test shape", test.shape,"Original shape", orig.shape )
TE = []
CC = CATS+CATS1+CATS2

mn = orig.y.mean()
print(f"Processing {len(CC)} columns... ",end="")
for i,c in enumerate(CC):
    if i%10==0: print(f"{i}, ",end="")
    tmp = orig.groupby(c).y.mean()
    tmp = tmp.astype('float32')
    NAME = f"TE_ORIG_{c}"
    tmp.name = NAME
    TE.append( NAME )
    train = train.merge(tmp, on=c, how='left')
    train[NAME] = train[NAME].fillna(mn)
    test = test.merge(tmp, on=c, how='left')
    test[NAME] = test[NAME].fillna(mn)
train = train.sort_values('i')
test = test.sort_values('i')
print()








from cuml.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
import xgboost as xgb

print(f"XGBoost version {xgb.__version__}")

FEATURES = NUMS+CATS+CATS1+CATS2+CE
print(f"We have {len(FEATURES)} features.")

FOLDS = 10
SEED = 42

params = {
    "objective": "binary:logistic",  
    "eval_metric": "auc",           
    "learning_rate": 0.02,
    "max_depth": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "seed": SEED,
    "device": "cuda",
    "grow_policy": "lossguide", 
    "max_leaves": 32,          
    "alpha": 2.0,
}

class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256*1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0 
        self.batch_size = batch_size
        self.batches = int( np.ceil( len(df) / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        if self.it == self.batches:
            return 0 # Return 0 when there's no more batch.
        
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, len(self.df) )
        #dt = cudf.DataFrame(self.df.iloc[a:b])
        dt = self.df.iloc[a:b]
        input_data(data=dt[self.features], label=dt[self.target]) 
        self.it += 1
        return 1

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

REPEATS = 1
for kk in range(REPEATS):
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        print("#"*25)
        print(f"### REPEAT {kk+1}, Fold {fold+1} ###")
        print("#"*25)
    
        Xy_train = train.iloc[train_idx][ FEATURES+['y'] ].copy()
        Xy_more = orig[ FEATURES+['y'] ]
        for k in range(1):
            Xy_train = cudf.concat([Xy_train,Xy_more],axis=0,ignore_index=True)
        
        X_valid = train.iloc[val_idx][FEATURES].copy()
        y_valid = train.iloc[val_idx]['y']
        X_test = test[FEATURES].copy()
    
        CC = CATS1+CATS2
        print(f"Target encoding {len(CC)} features... ",end="")
        for i,c in enumerate(CC):
            if i%10==0: print(f"{i}, ",end="")
            TE0 = TargetEncoder(n_folds=10, smooth=0, split_method='random', stat='mean')
            Xy_train[c] = TE0.fit_transform(Xy_train[c],Xy_train['y']).astype('float32')
            X_valid[c] = TE0.transform(X_valid[c]).astype('float32')
            X_test[c] = TE0.transform(X_test[c]).astype('float32')
        print()
    
        Xy_train[CATS] = Xy_train[CATS].astype('category')
        X_valid[CATS] = X_valid[CATS].astype('category')
        X_test[CATS] = X_test[CATS].astype('category')
    
        Xy_train = IterLoadForDMatrix(Xy_train, FEATURES, 'y')
        dtrain = xgb.QuantileDMatrix(Xy_train, enable_categorical=True, max_bin=256)
        dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
        dtest  = xgb.DMatrix(X_test, enable_categorical=True)
    
        params['seed'] = kk*FOLDS + fold
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100_000, 
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=250,
            verbose_eval=500
        )
    
        oof_preds[val_idx] += model.predict(dval, iteration_range=(0, model.best_iteration + 1)) / REPEATS
        test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS / REPEATS


from sklearn.metrics import roc_auc_score

m = roc_auc_score(train.y.to_numpy(), oof_preds)
print(f"XGB (Train More) with Original Data as rows CV = {m}")


FEATURES += TE
print(f"We have {len(FEATURES)} features.")

oof_preds2 = np.zeros(len(train))
test_preds2 = np.zeros(len(test))

REPEATS = 1
for kk in range(REPEATS):
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        print("#"*25)
        print(f"### REPEAT {kk+1}, Fold {fold+1} ###")
        print("#"*25)
    
        Xy_train = train.iloc[train_idx][ FEATURES+['y'] ].copy()  
        X_valid = train.iloc[val_idx][FEATURES].copy()
        y_valid = train.iloc[val_idx]['y']
        X_test = test[FEATURES].copy()
    
        CC = CATS1+CATS2
        print(f"Target encoding {len(CC)} features... ",end="")
        for i,c in enumerate(CC):
            if i%10==0: print(f"{i}, ",end="")
            TE0 = TargetEncoder(n_folds=10, smooth=0, split_method='random', stat='mean')
            Xy_train[c] = TE0.fit_transform(Xy_train[c],Xy_train['y']).astype('float32')
            X_valid[c] = TE0.transform(X_valid[c]).astype('float32')
            X_test[c] = TE0.transform(X_test[c]).astype('float32')
        print()
    
        Xy_train[CATS] = Xy_train[CATS].astype('category')
        X_valid[CATS] = X_valid[CATS].astype('category')
        X_test[CATS] = X_test[CATS].astype('category')
    
        Xy_train = IterLoadForDMatrix(Xy_train, FEATURES, 'y')
        dtrain = xgb.QuantileDMatrix(Xy_train, enable_categorical=True, max_bin=256)
        dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
        dtest  = xgb.DMatrix(X_test, enable_categorical=True)
    
        params['seed'] = kk*FOLDS + fold 
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100_000, 
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=250,
            verbose_eval=500
        )
    
        oof_preds2[val_idx] += model.predict(dval, iteration_range=(0, model.best_iteration + 1)) / REPEATS
        test_preds2 += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS / REPEATS

# FILL NAN FOR NN BELOW (we skipped this above for xgb)
CC = CATS+CATS1+CATS2
mn = orig.y.mean()
for i,c in enumerate(CC):
    NAME = f"TE_ORIG_{c}"
    train[NAME] = train[NAME].fillna(mn)
    test[NAME] = test[NAME].fillna(mn)


from sklearn.metrics import roc_auc_score

m = roc_auc_score(train.y.to_numpy(), oof_preds2)
print(f"XGB (Train More) with Original Data as rows CV = {m}")


LOG = ['balance','duration','campaign','pdays','previous']

for c in LOG+CE:
    if c in LOG: 
        mn = min( (train[c].min(), test[c].min()) )
        train[c] = train[c]-mn
        test[c] = test[c]-mn
    train[c] = np.log1p( train[c] )
    test[c] = np.log1p( test[c] )


FEATURES = CATS+NUMS+CATS1+CE+TE
TARGET_COL = 'y'
print(f"We have {len( FEATURES )} features.")


# =========================
# PyTorch MLP w/ Cat Embeddings + KFold OOF
# =========================
import os, math, random, gc, json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

# -------------------------
# Config
# -------------------------
SEED          = 42
FOLDS         = 10
EPOCHS        = 4
BATCH_SIZE    = 512
LR_MAX        = 3e-3          # peak LR for OneCycle
WD            = 1e-4          # weight decay (AdamW)
EARLY_STOP    = 4             # epochs with no val AUC improvement
GRAD_CLIP     = 1.0
NUM_WORKERS   = 4
MODEL_DIR     = "./mlp_catemb_models"
OOF_PATH      = "./oof_catemb.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

KNOWN_CARDINALITIES = SIZES
df = train[ FEATURES+['y'] ].to_pandas()
df2 = test[ FEATURES+['y'] ].to_pandas()

# -------------------------
# Repro
# -------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Load/define your dataframe `df`
# -------------------------
# df = pd.read_parquet("your_data.parquet")
# assert TARGET_COL in df.columns

categorical_cols = list( KNOWN_CARDINALITIES.keys() )
numeric_cols = [c for c in df.columns if c not in categorical_cols + [TARGET_COL]]

print(f"#categoricals={len(categorical_cols)}, #numerics={len(numeric_cols)}")

# -------------------------
# Label-encode categoricals (reserve 0=UNK)
# -------------------------
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    tmp = pd.concat([ df[[col]],df2[[col]] ],axis=0)
    tmp[col] = tmp[col].astype(str).fillna("NaN")
    le.fit(tmp[col].values)
    df[col] = le.transform(df[col].astype(str).values) + 1
    df2[col] = le.transform(df2[col].astype(str).values) + 1
    encoders[col] = le
    del tmp

cardinalities = {}
for col in categorical_cols:
    if col in KNOWN_CARDINALITIES:
        cardinalities[col] = KNOWN_CARDINALITIES[col] + 1  # +1 for UNK
        df[col] = np.clip(df[col], 0, KNOWN_CARDINALITIES[col])
        df2[col] = np.clip(df2[col], 0, KNOWN_CARDINALITIES[col])
    else:
        cardinalities[col] = int( max(df[col].max(),df2[col].max()) ) + 1

# -------------------------
# Scale numerics
# -------------------------
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols].astype(np.float32))
df2[numeric_cols] = scaler.transform(df2[numeric_cols].astype(np.float32))

# -------------------------
# Embedding dims
# -------------------------
def emb_dim_from_card(n):
    return int(min(50, round(1.6 * (n**0.56))))

emb_info = [(cardinalities[c], emb_dim_from_card(cardinalities[c])) for c in categorical_cols]
total_emb_dim = sum(d for _, d in emb_info)
print("Embedding config:", dict(zip(categorical_cols, emb_info)))
print("Total embedding dim:", total_emb_dim, " + numeric:", len(numeric_cols))

# -------------------------
# Dataset
# -------------------------
class TabDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols, target_col=None, idx=None):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col
        self.df = df if idx is None else df.iloc[idx]
        self.cats = self.df[self.cat_cols].values.astype(np.int64)
        self.nums = self.df[self.num_cols].values.astype(np.float32)
        self.y = None if self.target_col is None else self.df[self.target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        cats = torch.from_numpy(self.cats[i])
        nums = torch.from_numpy(self.nums[i])
        if self.y is None:
            return cats, nums
        return cats, nums, torch.tensor(self.y[i])

# -------------------------
# Model
# -------------------------
class MLPWithCatEmb(nn.Module):
    def __init__(self, emb_info, n_num, hidden=[512, 256, 128], dropout=0.15):
        super().__init__()
        self.emb_layers = nn.ModuleList(
            [nn.Embedding(num_embeddings=card, embedding_dim=dim, padding_idx=0)
             for card, dim in emb_info]
        )
        emb_total = sum(dim for _, dim in emb_info)
        in_dim = emb_total + n_num

        self.bn_nums = nn.BatchNorm1d(n_num) if n_num > 0 else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        layers = []
        last = in_dim
        for h in hidden:
            layers += [
                nn.Linear(last, h),
                nn.BatchNorm1d(h),
                nn.SiLU(),
                nn.Dropout(dropout)
            ]
            last = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(last, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_cat, x_num):
        emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
        x_emb = torch.cat(emb, dim=1) if emb else None
        if x_num is not None and x_num.shape[1] > 0:
            x_num = self.bn_nums(x_num)
            x = torch.cat([x_emb, x_num], dim=1) if x_emb is not None else x_num
        else:
            x = x_emb
        x = self.mlp(x)
        logit = self.head(x).squeeze(1)
        return logit

# -------------------------
# Train / Eval helpers
# -------------------------
def train_one_epoch(model, loader, optimizer, scaler, scheduler=None):
    model.train()
    running = 0.0
    for cats, nums, y in loader:
        cats = cats.to(device, non_blocking=True)
        nums = nums.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(cats, nums)
            loss = F.binary_cross_entropy_with_logits(logits, y)
        scaler.scale(loss).backward()
        if GRAD_CLIP is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running += loss.detach().item() * y.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        if len(batch) == 3:
            cats, nums, y = batch
            all_y.append(y.detach().cpu())
        else:
            cats, nums = batch
        cats = cats.to(device, non_blocking=True)
        nums = nums.to(device, non_blocking=True)
        all_logits.append(model(cats, nums).detach().cpu())

    logits = torch.cat(all_logits).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    probs_c = np.clip(probs, 1e-7, 1 - 1e-7)  # clip instead of log_loss(..., eps=...)

    if all_y:
        y_true = torch.cat(all_y).numpy().astype(np.int64)
        auc = roc_auc_score(y_true, probs)
        ll  = log_loss(y_true, probs_c)
        acc = accuracy_score(y_true, (probs >= 0.5).astype(int))
        return probs, {"auc": auc, "logloss": ll, "acc": acc}
    return probs, {}

oof = np.zeros(len(df), dtype=np.float32)
preds = np.zeros(len(df2), dtype=np.float32)

REPEATS = 5
for kk in range(REPEATS):
    print(f"##### REPEAT {kk+1} of {REPEATS} #####")
    seed_everything(SEED+kk)

    # -------------------------
    # KFold training
    # -------------------------
    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    
    y = df[TARGET_COL].astype(int).values
    oof2 = np.zeros(len(df), dtype=np.float32)
    fold_metrics = []
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        print(f"\n========== Fold {fold}/{FOLDS} ==========")
        trn_ds = TabDataset(df, categorical_cols, numeric_cols, TARGET_COL, trn_idx)
        val_ds = TabDataset(df, categorical_cols, numeric_cols, TARGET_COL, val_idx)
        test_ds = TabDataset(df2, categorical_cols, numeric_cols, TARGET_COL, np.arange(len(df2)) )
    
        trn_loader = DataLoader(
            trn_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
    
        model = MLPWithCatEmb(emb_info=emb_info, n_num=len(numeric_cols)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WD)
    
        total_steps = EPOCHS * len(trn_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR_MAX, total_steps=total_steps,
            pct_start=0.0, div_factor=25.0, final_div_factor=10.5, anneal_strategy='cos'
        )
    
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    
        best_auc = -1.0
        best_epoch = -1
        epochs_no_improve = 0
        best_path = os.path.join(MODEL_DIR, f"fold{fold}.pt")
    
        for epoch in range(1, EPOCHS+1):
            train_loss = train_one_epoch(model, trn_loader, optimizer, scaler, scheduler)
            _, val_stats = validate(model, val_loader)
            auc = val_stats.get("auc", float("nan"))
            print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} | "
                  f"val_auc={auc:.5f} val_logloss={val_stats['logloss']:.5f} val_acc={val_stats['acc']:.4f}")
    
            if 1: #auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "config": {
                        "emb_info": emb_info,
                        "numeric_cols": numeric_cols,
                        "categorical_cols": categorical_cols
                    }
                }, best_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP:
                    print(f"Early stopping at epoch {epoch}. Best AUC {best_auc:.5f} @ epoch {best_epoch}")
                    break
    
        # Load best
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
    
        # OOF for this fold
        val_probs, val_stats = validate(model, val_loader)
        oof[val_idx] += val_probs / REPEATS
        oof2[val_idx] = val_probs
        fold_metrics.append({"fold": fold, **val_stats})
        print(f"[Fold {fold}] AUC={val_stats['auc']:.5f}  LogLoss={val_stats['logloss']:.5f}  Acc={val_stats['acc']:.4f}")
    
        test_probs, _ = validate(model, test_loader)
        preds += test_probs / FOLDS / REPEATS
    
    # -------------------------
    # Overall OOF metrics
    # -------------------------
    oof_c = np.clip(oof2, 1e-7, 1 - 1e-7)
    oof_auc = roc_auc_score(y, oof2)
    oof_ll  = log_loss(y, oof_c)     # no eps kwarg
    oof_acc = accuracy_score(y, (oof2>=0.5).astype(int))
    print("\n========== Overall OOF ==========")
    print(f"OOF AUC={oof_auc:.5f}  LogLoss={oof_ll:.5f}  Acc={oof_acc:.4f}")
    
    # Save OOF and metrics
    #pd.DataFrame({
    #    "oof_pred": oof,
    #    TARGET_COL: y
    #}).to_csv(OOF_PATH, index=False)
    
    #with open(os.path.join(MODEL_DIR, "fold_metrics.json"), "w") as f:
    #    json.dump({"folds": fold_metrics, "oof": {"auc": oof_auc, "logloss": oof_ll, "acc": oof_acc}}, f, indent=2)
    
    #print(f"Saved OOF -> {OOF_PATH}")
    #print(f"Saved models -> {MODEL_DIR}")


m = roc_auc_score(train.y.to_numpy(), oof)
print(f"NN (Train More) CV = {m}")


sub = pd.read_csv(f"{PATH}sample_submission.csv")
sub['y'] = preds
sub.to_csv("submission_nn_train_more.csv",index=False)
np.save("oof_nn_train_more",oof)
print('Submission shape',sub.shape)
#sub.head()


ub = pd.read_csv(f"{PATH}sample_submission.csv")
preds_xgb = (test_preds+test_preds2)/2. 
sub['y'] = preds_xgb
sub.to_csv("submission_xgb_train_more.csv",index=False)
np.save("oof_xgb_rows_train_more",oof_preds)
np.save("oof_xgb_cols_train_more",oof_preds2)
print('Submission shape',sub.shape)


oof_xgb = (oof_preds+oof_preds2)/2.
m = roc_auc_score(train.y.to_numpy(), oof_xgb)
print(f"Both XGB rows and XGB cols (Train More) CV = {m}")


best_m = 0
best_w = 0
for w in np.arange(0,1.01,0.01):
    oof_ensemble = (1-w)*oof_xgb + w*oof
    m = roc_auc_score(train.y.to_numpy(), oof_ensemble)
    if m>best_m:
        best_m = m
        best_w = w
        
oof_ensemble = (1-best_w)*oof_xgb + best_w*oof
m = roc_auc_score(train.y.to_numpy(), oof_ensemble)
print(f"Ensemble XGB and NN (Train More) CV = {m}")
print(f" using best NN weight = {best_w}")


xgb_preds = preds_xgb 
sub = pd.read_csv(f"{PATH}sample_submission.csv")
sub['y'] = (1-best_w)*xgb_preds + best_w*preds
sub.to_csv("submission_ensemble_train_more.csv",index=False)
print('Submission shape',sub.shape)
sub.head()

