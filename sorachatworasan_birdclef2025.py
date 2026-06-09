import os
import random
import gc
import glob

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
import torchaudio
import timm

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm
from warnings import filterwarnings
filterwarnings("ignore")



class Config:
    train_dir       = "/kaggle/input/birdclef-2025/train_audio"
    train_csv       = "/kaggle/input/birdclef-2025/train.csv"
    
    sr              = 32_000
    n_fft           = 1024
    hop_length      = 500
    n_mels          = 128
    fmin            = 50
    fmax            = 16_000
    power           = 2
    
    seed            = 42
    num_classes     = 206
    batch_size      = 64
    num_workers     = 2
    epochs          = 20
    lr_max          = 1e-4
    weight_decay    = 1e-6
    use_amp         = True
    mixup_alpha     = 0.5
    
    # threshold for binarizing sigmoid outputs
    pred_thresh     = 0.5


def set_seed(s=Config.seed):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# load metadata
df = pd.read_csv(Config.train_csv)
df['filename'] = df['filename'].apply(lambda x: os.path.join(Config.train_dir, x))



class BirdClefDataset(Dataset):
    def __init__(self, df, mode='train'):
        self.df   = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):
        y, _ = librosa.load(path, sr=Config.sr)
        # pad / trim to exactly 10 seconds
        target_len = 10 * Config.sr
        if len(y) < target_len:
            y = np.tile(y, int(np.ceil(target_len / len(y))))
        y = y[:target_len]
        return y

    def _to_melspec(self, y):
        S = librosa.feature.melspectrogram(
            y=y, sr=Config.sr,
            n_fft=Config.n_fft,
            hop_length=Config.hop_length,
            n_mels=Config.n_mels,
            fmin=Config.fmin,
            fmax=Config.fmax,
            power=Config.power
        )
        S = librosa.power_to_db(S, ref=np.max)
        # normalize to [0,1]
        S = (S - S.min()) / (S.max() - S.min() + 1e-6)
        # to 3-channel by duplicating
        img = np.stack([S, S, S], axis=0)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        y   = self._load_audio(row.filename)
        x   = self._to_melspec(y)
        if self.mode == 'train':
            # single-label primary_label
            label = row.primary_label
            # map label to int 0..205
            target = label_mapper[label]
            return x, target
        else:
            return x


# build label mapper
labels = sorted(df.primary_label.unique())
label_mapper = {lab:i for i,lab in enumerate(labels)}
rev_mapper   = {i:lab for lab,i in label_mapper.items()}



class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                                                     (1,1)).pow(1./self.p).view(x.size(0), -1)


class BirdCLEFNet(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b0', pretrained=False):
        super().__init__()
        # extract channels from intermediate layers 3 and 4
        self.feat_extractor = timm.create_model(
            backbone, pretrained=pretrained,
            features_only=True, out_indices=(3,4),
            in_chans=3
        )
        chans = self.feat_extractor.feature_info.channels()  # e.g. [80, 320]
        self.gpools = nn.ModuleList([GeM() for _ in chans])
        self.bn     = nn.BatchNorm1d(sum(chans))
        self.fc     = nn.Linear(sum(chans), Config.num_classes)

    def forward(self, x):
        feats = self.feat_extractor(x)              # list of 2 feature maps
        pooled = [g(f) for g,f in zip(self.gpools, feats)]
        h = torch.cat(pooled, dim=1)
        return self.fc(self.bn(h))


class FocalBCE(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce   = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        # focal component
        prob = torch.sigmoid(logits)
        p_t  = prob*targets + (1-prob)*(1-targets)
        focal = ((1-p_t)**self.gamma * 
                 (-self.alpha*targets*torch.log(prob+1e-6) 
                  - (1-self.alpha)*(1-targets)*torch.log(1-prob+1e-6))).mean()
        return bce + focal



def compute_metrics(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    
    # subset accuracy
    acc = accuracy_score(y_true, y_pred)
    
    n_samples = y_true.shape[0]
    valid = np.where((y_true.sum(axis=0) > 0) & (y_true.sum(axis=0) < n_samples))[0]
    if len(valid) > 0:
        roc = roc_auc_score(y_true[:, valid], y_prob[:, valid], average='macro')
    else:
        roc = float('nan')
    
    return {
        'precision_macro': p_m, 'recall_macro': r_m, 'f1_macro': f_m,
        'precision_weighted': p_w, 'recall_weighted': r_w, 'f1_weighted': f_w,
        'accuracy': acc, 'roc_auc': roc
    }

def plot_roc(y_true, y_prob, class_names):
    n_cls = y_true.shape[1]
    # individual curves
    plt.figure(figsize=(8,6))
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    for i in range(n_cls):
        if y_true[:,i].sum()==0: continue
        fpr, tpr, _ = roc_curve(y_true[:,i], y_prob[:,i])
        auc = roc_auc_score(y_true[:,i], y_prob[:,i])
        plt.plot(fpr, tpr, linewidth=1, label=f"{class_names[i]} ({auc:.2f})")
    # micro-average
    fpr_m, tpr_m, _ = roc_curve(y_true.ravel(), y_prob.ravel())
    auc_m = roc_auc_score(y_true, y_prob, average='micro')
    plt.plot(fpr_m, tpr_m, color='m', linestyle='--',
             label=f"Micro (AUC={auc_m:.2f})")
    plt.title("ROC Curves")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(fontsize='small',ncol=2)
    plt.tight_layout(); plt.show()



def mixup(x, y, alpha=Config.mixup_alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    x2, y2 = x[idx], y[idx]
    return lam*x + (1-lam)*x2, lam*y + (1-lam)*y2

def train_one_epoch(model, loader, opt, scaler):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="Train"):
        x, y = x.cuda(), y.cuda()
        # one-hot for BCE
        y_oh = nn.functional.one_hot(y, Config.num_classes).float()
        # to multi-label mixing
        x_mix, y_mix = mixup(x, y_oh)
        with autocast(enabled=Config.use_amp):
            logits = model(x_mix)
            loss   = crit(logits, y_mix)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        scaler.step(opt); scaler.update(); opt.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    all_probs = []
    all_true  = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Valid"):
            x = x.cuda()
            logits = model(x)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            # one-hot
            y_oh = nn.functional.one_hot(y, Config.num_classes).numpy()
            all_true.append(y_oh)
    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_true)
    return y_true, y_prob


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.seed)
df['fold'] = -1
for f,(_,v) in enumerate(skf.split(df, df.primary_label)):
    df.loc[v,'fold'] = f

metrics_list = []
roc_data_list = []
for fold in range(5):
    print(f"\n=== Fold {fold} ===")
    trn = df[df.fold!=fold]
    val = df[df.fold==fold]
    train_loader = DataLoader(BirdClefDataset(trn,'train'),
                              batch_size=Config.batch_size, shuffle=True,
                              num_workers=Config.num_workers, pin_memory=True)
    val_loader   = DataLoader(BirdClefDataset(val,'train'),
                              batch_size=Config.batch_size, shuffle=False,
                              num_workers=Config.num_workers, pin_memory=True)

    model = BirdCLEFNet().cuda()
    crit  = FocalBCE().cuda() 
    opt   = AdamW(model.parameters(), lr=Config.lr_max,
                  weight_decay=Config.weight_decay)
    scaler= GradScaler(enabled=Config.use_amp)
    
    best_roc = 0.0
    for ep in range(Config.epochs):
        loss = train_one_epoch(model, train_loader, opt, scaler)
        y_true, y_prob = validate(model, val_loader)
        m = compute_metrics(y_true, y_prob)
        print(f"Epoch {ep} | Loss {loss:.4f} | "
              f"ROC {m['roc_auc']:.4f} | Acc {m['accuracy']:.4f} | "
              f"F1_macro {m['f1_macro']:.4f}")
        if m['roc_auc'] > best_roc:
            best_roc = m['roc_auc']
            torch.save(model.state_dict(), f"best_fold{fold}.pth")
    # final metrics & plot
    y_true, y_prob = validate(model, val_loader)
    m = compute_metrics(y_true, y_prob)
    print(">> Fold",fold,"final metrics:",m)
    plot_roc(y_true, y_prob, labels)
    metrics_list.append(m)
    roc_data_list.append((y_true, y_prob))
    del model; gc.collect(); torch.cuda.empty_cache()


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

# 1) แปลง metrics_list ➔ DataFrame
dfm = pd.DataFrame(metrics_list, index=[f"Fold {i+1}" for i in range(len(metrics_list))])
# Bar chart เปรียบเทียบ metrics
ax = dfm[['acc','prec_macro','rec_macro','f1_macro','prec_w','rec_w','f1_w']] \
    .plot.bar(figsize=(12,6))
ax.set_title('Per-Fold Classification Metrics')
ax.set_ylabel('Score')
ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 2) Overlay micro-average ROC curves per fold
plt.figure(figsize=(8,6))
for i, (y_true, y_prob) in enumerate(roc_data_list):
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_prob.ravel())
    auc = roc_auc_score(y_true, y_prob, average='micro')
    plt.plot(fpr, tpr, label=f'Fold {i+1} (AUC={auc:.3f})')
plt.plot([0,1],[0,1],'k--',alpha=0.5)
plt.title('Micro-average ROC Curve per Fold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


