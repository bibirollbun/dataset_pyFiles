# =========================================================
# 0) CÃ€I Ä�áº¶T & THAM Sá»� CHUNG
# =========================================================
!pip -q install transformers==4.44.2 accelerate==0.34.2 scikit-learn matplotlib torchmetrics --progress-bar off

import os, json, random, gc, math, time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_cosine_schedule_with_warmup
)

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score
)

import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Ä�Æ¯á»œNG DáºªN KAGGLE
DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
SAVE_DIR = "/kaggle/working/my_trained_model_DistilBERT_toxic_2"
os.makedirs(SAVE_DIR, exist_ok=True)

# Cáº¤U HÃŒNH
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 192            # 128â€“192 Ä‘á»§ cho comment
BATCH_TRAIN= 32             # T4 á»•n (giáº£m náº¿u OOM)
BATCH_VAL  = 64
EPOCHS     = 3
LR         = 2e-5
WARMUP_PCT = 0.1
WEIGHT_DEC = 0.01
GRAD_ACCUM = 1              # tÄƒng náº¿u cáº§n batch áº£o lá»›n
FP16       = True           # AMP cho nhanh/Ä‘á»¡ VRAM

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
N_LABELS = len(LABELS)



# =========================================================
# 1) Táº¢I Dá»® LIá»†U & CHIA Táº¬P
# =========================================================
#import os, pandas as pd


def read_csv_auto(basename: str) -> pd.DataFrame:
    csv_path = f"{DATA_DIR}/{basename}.csv"
    zip_path = f"{DATA_DIR}/{basename}.csv.zip"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    if os.path.exists(zip_path):
        return pd.read_csv(zip_path)  # pandas tá»± nháº­n zip
    raise FileNotFoundError(f"KhÃ´ng tÃ¬m tháº¥y {basename}.csv(.zip) trong {DATA_DIR}")

train_df = read_csv_auto("train")
test_df  = read_csv_auto("test")
sub_df   = read_csv_auto("sample_submission")

# Báº£o Ä‘áº£m khÃ´ng NaN
train_df['comment_text'] = train_df['comment_text'].fillna("")

# Chia train/val (stratify theo tá»•ng nhÃ£n dÆ°Æ¡ng)
from sklearn.model_selection import train_test_split
y = train_df[LABELS].values
strat = (y>0).sum(1)
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=SEED,
    stratify=np.clip(strat, 0, 3) # gá»™p bá»›t Ä‘á»ƒ á»•n Ä‘á»‹nh stratify
)
trn_df = train_df.iloc[train_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

print(trn_df.shape, val_df.shape, test_df.shape)
trn_df.tail(55)



import matplotlib.pyplot as plt

# Ä�áº¿m sá»‘ lÆ°á»£ng bÃ¬nh luáº­n dÃ¡n nhÃ£n 1 cho tá»«ng nhÃ£n
label_counts = train_df[LABELS].sum().sort_values(ascending=False)

plt.figure(figsize=(8,5))
label_counts.plot(kind="bar", color="royalblue")
plt.title("Sá»‘ lÆ°á»£ng bÃ¬nh luáº­n dÃ¡n nhÃ£n (train set)", fontsize=14)
plt.ylabel("Sá»‘ máº«u")
plt.xticks(rotation=45)
plt.show()

# TÃ­nh tá»‰ lá»‡ %
label_ratios = (train_df[LABELS].sum() / len(train_df) * 100).sort_values(ascending=False)

plt.figure(figsize=(8,5))
label_ratios.plot(kind="bar", color="darkorange")
plt.title("Tá»‰ lá»‡ pháº§n trÄƒm nhÃ£n trong táº­p train", fontsize=14)
plt.ylabel("Tá»‰ lá»‡ (%)")
plt.xticks(rotation=45)
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# TÃ­nh ma tráº­n tÆ°Æ¡ng quan giá»¯a cÃ¡c nhÃ£n
corr = train_df[LABELS].corr()

plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", cbar=True)
plt.title("TÆ°Æ¡ng quan giá»¯a cÃ¡c nhÃ£n (train set)", fontsize=14)
plt.show()



import matplotlib.pyplot as plt

# Ä�áº¿m sá»‘ nhÃ£n dÆ°Æ¡ng trÃªn má»—i comment
label_per_comment = train_df[LABELS].sum(axis=1)

# Trung bÃ¬nh
avg_labels = label_per_comment.mean()
print(f"Sá»‘ nhÃ£n dÆ°Æ¡ng trung bÃ¬nh má»—i comment: {avg_labels:.3f}")

# PhÃ¢n bá»‘
plt.figure(figsize=(6,4))
label_per_comment.value_counts().sort_index().plot(kind="bar", color="teal")
plt.title("PhÃ¢n bá»‘ sá»‘ nhÃ£n dÆ°Æ¡ng trÃªn má»—i comment", fontsize=14)
plt.xlabel("Sá»‘ nhÃ£n dÆ°Æ¡ng")
plt.ylabel("Sá»‘ lÆ°á»£ng comment")
plt.show()



# =========================================================
# 2) DATASET & TOKENIZER
# =========================================================
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class ToxicDataset(Dataset):
    def __init__(self, texts, labels=None, max_len=128, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tok = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        text = str(self.texts[i])
        enc = self.tok(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k,v in enc.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[i]).float()
        return item

train_ds = ToxicDataset(trn_df['comment_text'].tolist(), trn_df[LABELS].values, MAX_LEN, tokenizer)
val_ds   = ToxicDataset(val_df['comment_text'].tolist(), val_df[LABELS].values, MAX_LEN, tokenizer)
test_ds  = ToxicDataset(test_df['comment_text'].tolist(), None, MAX_LEN, tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_VAL,   shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_VAL,   shuffle=False, num_workers=2, pin_memory=True)



# =========================================================
# 3) MÃ” HÃŒNH + LOSS (BCE vá»›i pos_weight xá»­ lÃ½ lá»‡ch lá»›p)
# =========================================================
from transformers.utils import logging
logging.set_verbosity_error()  # giáº£m bá»›t cáº£nh bÃ¡o tá»« transformers

# pos_weight = (#negative / #positive) cho má»—i nhÃ£n
pos_w = []
for col in LABELS:
    p = trn_df[col].sum()
    n = len(trn_df) - p
    pos_w.append(n / max(p, 1))
pos_w = torch.tensor(pos_w, dtype=torch.float, device=device)
print("pos_weight:", pos_w.cpu().numpy().round(2))

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=N_LABELS,
    problem_type="multi_label_classification"
).to(device)

# BCEWithLogitsLoss sáº½ dÃ¹ng bÃªn dÆ°á»›i thá»§ cÃ´ng Ä‘á»ƒ truyá»�n pos_weight
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DEC)

# Láº­p lá»‹ch LR cosine
num_update_steps_per_epoch = math.ceil(len(train_loader)/GRAD_ACCUM)
t_total = EPOCHS * num_update_steps_per_epoch
num_warmup = int(WARMUP_PCT * t_total)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, t_total)

# cÅ© (váº«n cháº¡y):
# scaler = torch.cuda.amp.GradScaler(enabled=FP16)

# má»›i (khuyáº¿n nghá»‹):
try:
    scaler = torch.amp.GradScaler(device="cuda", enabled=FP16)  # PyTorch >= 2.0
except TypeError:
    scaler = torch.cuda.amp.GradScaler(enabled=FP16)            # fallback cho báº£n cÅ©



# =========================================================
# 4) HÃ€M TRAIN/EVAL & METRICS  (báº£n cáº£i tiáº¿n)
# =========================================================
# Hai loss: cÃ³/khÃ´ng pos_weight
bce_loss_w     = nn.BCEWithLogitsLoss(pos_weight=pos_w)  # weighted
bce_loss_plain = nn.BCEWithLogitsLoss()                  # unweighted (Ä‘á»ƒ bÃ¡o cÃ¡o)

def compute_metrics(y_true, y_prob):
    metrics = {}
    ap_per_label, roc_per_label, f1_per_label = [], [], []
    y_pred05 = (y_prob >= 0.5).astype(int)
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:, j], y_prob[:, j]
        try: ap = average_precision_score(yt, yp)
        except: ap = np.nan
        try: roc = roc_auc_score(yt, yp)
        except: roc = np.nan
        f1 = f1_score(yt, y_pred05[:, j], zero_division=0)
        ap_per_label.append(ap); roc_per_label.append(roc); f1_per_label.append(f1)
    metrics['PR_AUC_macro']  = float(np.nanmean(ap_per_label))
    metrics['ROC_AUC_macro'] = float(np.nanmean(roc_per_label))
    metrics['F1_macro@0.5']  = float(np.mean(f1_per_label))
    metrics['PR_AUC_per_label']  = ap_per_label
    metrics['ROC_AUC_per_label'] = roc_per_label
    metrics['F1_per_label@0.5']  = f1_per_label
    return metrics

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    val_loss_w, val_loss_plain = 0.0, 0.0
    for batch in loader:
        ids  = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        y    = batch['labels'].to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        val_loss_w     += bce_loss_w(logits, y).item()     * ids.size(0)
        val_loss_plain += bce_loss_plain(logits, y).item() * ids.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    metrics = compute_metrics(all_labels, all_probs)
    val_loss_w     /= len(loader.dataset)
    val_loss_plain /= len(loader.dataset)
    return (val_loss_w, val_loss_plain), metrics, all_probs, all_labels

def optimize_thresholds(y_true, y_prob):
    ths_grid = np.linspace(0.05, 0.95, 37)
    best_th = []
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:, j], y_prob[:, j]
        f1s = [f1_score(yt, (yp >= t).astype(int), zero_division=0) for t in ths_grid]
        best_th.append(ths_grid[int(np.argmax(f1s))])
    return np.array(best_th, dtype=float)



# =========================================================
# 5) HUáº¤N LUYá»†N  (thÃªm Early Stopping + Save Best theo PR-AUC)
# =========================================================
train_losses = []
val_losses_w = []       # weighted loss
val_losses_p = []       # plain loss
history_pr   = []
history_roc  = []
history_f1   = []

BEST_DIR   = SAVE_DIR + "_best"
os.makedirs(BEST_DIR, exist_ok=True)
best_score = -1.0       # theo PR-AUC macro
patience   = 2          # dá»«ng sá»›m náº¿u khÃ´ng cáº£i thiá»‡n
bad_epochs = 0

# autocast/GradScaler API má»›i (fallback náº¿u báº£n torch cÅ©)
try:
    autocast_ctx = torch.amp.autocast('cuda', enabled=FP16)
    scaler = torch.amp.GradScaler('cuda', enabled=FP16)
except Exception:
    autocast_ctx = torch.cuda.amp.autocast(enabled=FP16)
    scaler = torch.cuda.amp.GradScaler(enabled=FP16)

for epoch in range(1, EPOCHS+1):
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for step, batch in pbar:
        ids  = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        y    = batch['labels'].to(device)

        with autocast_ctx:
            logits = model(input_ids=ids, attention_mask=attn).logits
            loss = bce_loss_w(logits, y) / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running += loss.item() * GRAD_ACCUM
        if (step + 1) % 100 == 0:
            pbar.set_postfix(train_loss=running/(step+1))

    train_loss = running / len(train_loader)
    train_losses.append(train_loss)

    # ---- Ä�Ã¡nh giÃ¡ + log hai loáº¡i loss ----
    (val_w, val_p), metrics, val_prob, val_true = evaluate(model, val_loader)
    val_losses_w.append(val_w)
    val_losses_p.append(val_p)
    history_pr.append(metrics['PR_AUC_macro'])
    history_roc.append(metrics['ROC_AUC_macro'])
    history_f1.append(metrics['F1_macro@0.5'])

    print(f"\n[Epoch {epoch}] Train: {train_loss:.4f} | "
          f"Val loss (weighted): {val_w:.4f} | Val loss (plain): {val_p:.4f} | "
          f"PR-AUC: {metrics['PR_AUC_macro']:.4f} | ROC-AUC: {metrics['ROC_AUC_macro']:.4f} | "
          f"F1@0.5: {metrics['F1_macro@0.5']:.4f}")

    # ---- Early stopping theo PR-AUC macro ----
    score = metrics['PR_AUC_macro']
    if score > best_score + 1e-4:
        best_score = score
        bad_epochs = 0
        # LÆ°u best checkpoint + val prob/true Ä‘á»ƒ tá»‘i Æ°u ngÆ°á»¡ng sau cÃ¹ng
        model.save_pretrained(BEST_DIR)
        tokenizer.save_pretrained(BEST_DIR)
        np.save(os.path.join(BEST_DIR, "val_prob.npy"), val_prob)
        np.save(os.path.join(BEST_DIR, "val_true.npy"), val_true)
        print(f"âœ… New best (PR-AUC={best_score:.4f}) â†’ saved to {BEST_DIR}\n")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("â�¹ Early stopping (no improvement).")
            break

# Sau training: náº¡p láº¡i best Ä‘á»ƒ tá»‘i Æ°u ngÆ°á»¡ng & suy luáº­n
VAL_PROB = np.load(os.path.join(BEST_DIR, "val_prob.npy"))
VAL_TRUE = np.load(os.path.join(BEST_DIR, "val_true.npy"))



# =========================================================
# 6) Váº¼ BIá»‚U Ä�á»’: TRAIN/VAL LOSS & METRICS
# =========================================================
plt.figure(figsize=(7,4.5))
plt.plot(train_losses, label='Train loss')
# náº¿u báº¡n cÃ³ hai loss: val_losses_w (weighted) & val_losses_p (plain)
if 'val_losses_w' in globals() and 'val_losses_p' in globals():
    plt.plot(val_losses_w, label='Val loss (weighted)')
    plt.plot(val_losses_p, label='Val loss (plain)')
else:
    plt.plot(val_losses, label='Val loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Train/Val Loss'); plt.legend(); plt.grid(axis='y', ls='--', alpha=.5)
plt.show()

plt.figure(figsize=(7,4.5))
plt.plot(history_pr,  label='Val PR-AUC')
plt.plot(history_roc, label='Val ROC-AUC')
plt.plot(history_f1,  label='Val F1@0.5')
plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Val Metrics'); plt.legend(); plt.grid(axis='y', ls='--', alpha=.5)
plt.show()



# =========================================================
# 7) Tá»�I Æ¯U NGÆ¯á» NG & BÃ�O CÃ�O CHI TIáº¾T (DÃ™NG BEST CHECKPOINT)
# =========================================================
# Load láº¡i VAL_PROB/VAL_TRUE Ä‘Ã£ lÆ°u khi Ä‘áº¡t PR-AUC tá»‘t nháº¥t
VAL_PROB = np.load(os.path.join(BEST_DIR, "val_prob.npy"))
VAL_TRUE = np.load(os.path.join(BEST_DIR, "val_true.npy"))

best_th = optimize_thresholds(VAL_TRUE, VAL_PROB)
print("Best thresholds per-label:", dict(zip(LABELS, np.round(best_th, 3))))

VAL_PRED = (VAL_PROB >= best_th).astype(int)

per_label_rows = []
for j, name in enumerate(LABELS):
    yt = VAL_TRUE[:, j]; yp = VAL_PROB[:, j]; yh = VAL_PRED[:, j]
    ap  = average_precision_score(yt, yp)
    roc = roc_auc_score(yt, yp)
    f1  = f1_score(yt, yh, zero_division=0)
    per_label_rows.append([name, ap, roc, f1, best_th[j]])

rep_df = pd.DataFrame(per_label_rows, columns=["label","PR_AUC","ROC_AUC","F1@best_th","best_th"])
display(rep_df)

print("PR-AUC macro:", rep_df["PR_AUC"].mean().round(4))
print("ROC-AUC macro:", rep_df["ROC_AUC"].mean().round(4))
print("F1 macro @best_th:", rep_df["F1@best_th"].mean().round(4))



# =========================================================
# 8) LÆ¯U THRESHOLDS (Cáº NH BEST MODEL)
# =========================================================
with open(os.path.join(BEST_DIR, "thresholds.json"), "w") as f:
    json.dump({LABELS[i]: float(best_th[i]) for i in range(N_LABELS)}, f, indent=2)

print("Ä�Ã£ lÆ°u thresholds vÃ o:", os.path.join(BEST_DIR, "thresholds.json"))
# Gá»£i Ã½: tá»« Ä‘Ã¢y trá»Ÿ Ä‘i, dÃ¹ng BEST_DIR nhÆ° thÆ° má»¥c model chÃ­nh



# =========================================================
# 9) SUY LUáº¬N TRÃŠN Táº¬P TEST & Táº O SUBMISSION (DÃ™NG BEST MODEL)
# =========================================================
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

best_tok    = DistilBertTokenizerFast.from_pretrained(BEST_DIR)
best_model  = DistilBertForSequenceClassification.from_pretrained(BEST_DIR).to(device)

@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    all_probs = []
    for batch in loader:
        ids  = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)

test_probs = predict_probs(best_model, test_loader)
# load thresholds tá»« BEST_DIR
with open(os.path.join(BEST_DIR, "thresholds.json")) as f:
    th_dict = json.load(f)
best_th = np.array([th_dict[k] for k in LABELS], dtype=float)
test_pred = (test_probs >= best_th).astype(int)

# LÆ°u file trong BEST_DIR
sub_out = pd.DataFrame(test_probs, columns=LABELS)
sub_out.insert(0, 'id', test_df['id'].values)
sub_path = os.path.join(BEST_DIR, "submission_probs.csv")
sub_out.to_csv(sub_path, index=False)

sub_bin = pd.DataFrame(test_pred, columns=LABELS)
sub_bin.insert(0, 'id', test_df['id'].values)
sub_bin_path = os.path.join(BEST_DIR, "submission_binary.csv")
sub_bin.to_csv(sub_bin_path, index=False)

print("Ä�Ã£ lÆ°u:", sub_path, "vÃ ", sub_bin_path)
display(sub_out.head())



# =========================================================
# 10) KIá»‚M THá»¬ NHANH: Náº P Láº I MÃ” HÃŒNH & Dá»° Ä�OÃ�N MáºªU (BEST CHECKPOINT)
# =========================================================
import json, os
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Æ¯u tiÃªn BEST_DIR; náº¿u chÆ°a cÃ³ thÃ¬ dÃ¹ng SAVE_DIR
MODEL_DIR = BEST_DIR if os.path.exists(BEST_DIR) else SAVE_DIR
print("Using model from:", MODEL_DIR)

infer_tok   = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
infer_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

# Ä�á»�c thresholds.json (Ä‘Ã£ tá»‘i Æ°u theo best checkpoint)
with open(os.path.join(MODEL_DIR, "thresholds.json"), "r") as f:
    th_dict = json.load(f)
infer_th = np.array([th_dict[k] for k in LABELS], dtype=float)

def predict_texts(texts, max_len=MAX_LEN):
    enc = infer_tok(texts, truncation=True, padding=True, max_length=max_len, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    infer_model.eval()
    with torch.no_grad():
        logits = infer_model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= infer_th).astype(int)
    return probs, preds

# VÃ­ dá»¥ kiá»ƒm thá»­
samples = [
    "I hate you. You are disgusting!",
    "This is a normal comment, nothing wrong here.",
    "Go back to your country!",
    "I'll kill you if you come again."
]
probs, preds = predict_texts(samples)
pd.DataFrame({
    "text": samples,
    **{f"p_{lab}": probs[:, i] for i, lab in enumerate(LABELS)},
    **{f"y_{lab}": preds[:, i] for i, lab in enumerate(LABELS)},
})



# =============== PHÃ‚N LOáº I CÃ‚U: Ä�á»˜C Háº I / KHÃ”NG Ä�á»˜C Háº I ===============
import numpy as np
import pandas as pd
import torch

VN_LABELS = {
    "toxic": "Ä‘á»™c háº¡i",
    "severe_toxic": "ráº¥t Ä‘á»™c háº¡i",
    "obscene": "tá»¥c tÄ©u",
    "threat": "Ä‘e doáº¡",
    "insult": "xÃºc pháº¡m",
    "identity_hate": "thÃ¹ ghÃ©t Ä‘á»‹nh danh",
}

def classify_texts(texts, max_len=MAX_LEN, return_breakdown=True, thresholds=None, use_vn_labels=True):
    """
    Tráº£ vá»� DataFrame:
      - overall_label: 'Ä�á»™c háº¡i' / 'KhÃ´ng Ä‘á»™c háº¡i'
      - is_toxic: bool
      - reasons: cÃ¡c nhÃ£n con vÆ°á»£t ngÆ°á»¡ng (náº¿u cÃ³)
      - (tuá»³ chá»�n) p_<label>, top_label, p_max
    """
    # Chuáº©n hoÃ¡ input
    if isinstance(texts, str):
        texts = [texts]

    # NgÆ°á»¡ng: dÃ¹ng cÃ¡i Ä‘Ã£ tá»‘i Æ°u hoáº·c override
    th = np.array(thresholds if thresholds is not None else infer_th, dtype=float)
    assert th.shape[0] == len(LABELS), "thresholds pháº£i cÃ³ Ä‘Ãºng 6 pháº§n tá»­"

    # Encode & predict
    enc = infer_tok(texts, truncation=True, padding=True, max_length=max_len, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    infer_model.eval()
    with torch.no_grad():
        logits = infer_model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()  # [N, 6]

    # Ã�p ngÆ°á»¡ng tá»«ng nhÃ£n
    preds = (probs >= th).astype(int)

    # Gá»™p nhÃ£n â†’ Ä‘á»™c háº¡i náº¿u cÃ³ >=1 nhÃ£n báº­t
    toxic_flags = (preds.sum(axis=1) > 0)

    # LÃ½ do: liá»‡t kÃª nhÃ£n vÆ°á»£t ngÆ°á»¡ng (tiáº¿ng Anh/Tiáº¿ng Viá»‡t tuá»³ chá»�n)
    shown_labels = [VN_LABELS.get(l, l) if use_vn_labels else l for l in LABELS]
    reasons = []
    for i in range(len(texts)):
        active = [shown_labels[j] for j in range(len(LABELS)) if preds[i, j] == 1]
        if not active:
            j_top = int(np.argmax(probs[i]))
            active = [f"(gáº§n ngÆ°á»¡ng: {shown_labels[j_top]}={probs[i, j_top]:.2f})"]
        reasons.append(", ".join(active))

    # ThÃ´ng tin tham kháº£o
    top_idx = np.argmax(probs, axis=1)
    top_label = [shown_labels[j] for j in top_idx]
    p_max = probs[np.arange(len(texts)), top_idx]

    df = pd.DataFrame({
        "text": texts,
        "is_toxic": toxic_flags,
        "overall_label": np.where(toxic_flags, "Ä�á»™c háº¡i", "KhÃ´ng Ä‘á»™c háº¡i"),
        "reasons": reasons,
    })

    if return_breakdown:
        for j, lab in enumerate(LABELS):
            df[f"p_{lab}"] = probs[:, j]
        df["top_label"] = top_label
        df["p_max"] = p_max

    return df



samples = [
    "I hate you. You are disgusting!",
    "This is a normal comment, nothing wrong here.",
    "Go back to your country!",
    "I'll kill you if you come again."
]
result_df = classify_texts(samples, return_breakdown=True)
result_df[["text","overall_label","reasons","p_max","top_label"]]



minimal = classify_texts(samples, return_breakdown=False)
minimal[["text","overall_label","reasons"]]



# =========================================================
# 11) (TUá»² CHá»ŒN) Ä�Ã“NG GÃ“I ZIP Ä�á»‚ Táº¢I Vá»€ MÃ�Y
# =========================================================
import shutil
zip_path = "/kaggle/working/my_trained_model_DistilBERT_toxic_2_best"
shutil.make_archive(zip_path.replace(".zip",""), 'zip', '/kaggle/working/my_trained_model_DistilBERT_toxic_2_best')
zip_path



import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_fscore_support
)

def _safe_auc(func, y_true_col, y_prob_col):
    """Tráº£ vá»� NaN náº¿u cá»™t toÃ n 0/1 (khÃ´ng tÃ­nh Ä‘Æ°á»£c AUC)."""
    try:
        return func(y_true_col, y_prob_col)
    except Exception:
        return np.nan

def optimize_thresholds(y_true, y_prob, grid=None):
    """
    Tá»‘i Æ°u ngÆ°á»¡ng cho tá»«ng nhÃ£n theo F1.
    y_true, y_prob: ndarray [N, L]
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 37)
    L = y_true.shape[1]
    best_th = np.zeros(L, dtype=float)
    for j in range(L):
        yt, yp = y_true[:, j], y_prob[:, j]
        f1s = [f1_score(yt, (yp >= t).astype(int), zero_division=0) for t in grid]
        best_th[j] = grid[int(np.argmax(f1s))]
    return best_th

def eval_multilabel(y_true, y_prob, labels, optimize_th=True, grid=None):
    """
    TÃ­nh ROC-AUC macro, PR-AUC macro, F1 macro (@0.5 & @best_th) + báº£ng per-label.
    - y_true: np.array shape [N, L] vá»›i 0/1
    - y_prob: np.array shape [N, L] vá»›i xÃ¡c suáº¥t 0..1
    - labels: list tÃªn nhÃ£n Ä‘á»™ dÃ i L
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    N, L = y_true.shape
    assert y_prob.shape == (N, L), "y_prob pháº£i cÃ¹ng shape vá»›i y_true"
    assert len(labels) == L, "labels pháº£i cÃ³ Ä‘á»™ dÃ i báº±ng sá»‘ cá»™t"

    # --- AUCs (threshold-free) ---
    roc_list = []
    pr_list  = []
    for j in range(L):
        yt, yp = y_true[:, j], y_prob[:, j]
        roc_list.append(_safe_auc(roc_auc_score, yt, yp))
        pr_list.append(_safe_auc(average_precision_score, yt, yp))

    roc_auc_macro = np.nanmean(roc_list)
    pr_auc_macro  = np.nanmean(pr_list)

    # --- F1 @0.5 ---
    y_pred05 = (y_prob >= 0.5).astype(int)
    f1_per_label_05 = [f1_score(y_true[:, j], y_pred05[:, j], zero_division=0) for j in range(L)]
    f1_macro_05 = float(np.mean(f1_per_label_05))

    # --- Tá»‘i Æ°u ngÆ°á»¡ng per-label theo F1 ---
    if optimize_th:
        best_th = optimize_thresholds(y_true, y_prob, grid=grid)
    else:
        best_th = np.array([0.5]*L)

    y_pred_best = (y_prob >= best_th).astype(int)

    # F1/Precision/Recall @best_th
    f1_per_label_best = []
    prec_best = []
    rec_best  = []
    for j in range(L):
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, j], y_pred_best[:, j], average='binary', zero_division=0
        )
        prec_best.append(p); rec_best.append(r); f1_per_label_best.append(f)

    f1_macro_best = float(np.mean(f1_per_label_best))

    # --- Báº£ng per-label ---
    per_label_df = pd.DataFrame({
        "label": labels,
        "ROC_AUC": roc_list,
        "PR_AUC": pr_list,
        "F1@0.5": f1_per_label_05,
        "best_th": best_th,
        "Precision@best": prec_best,
        "Recall@best": rec_best,
        "F1@best": f1_per_label_best,
    })

    # --- Káº¿t quáº£ tá»•ng há»£p ---
    summary = {
        "ROC_AUC_macro": float(roc_auc_macro),
        "PR_AUC_macro":  float(pr_auc_macro),
        "F1_macro@0.5":  float(f1_macro_05),
        "F1_macro@best": float(f1_macro_best),
        "best_thresholds": {labels[i]: float(best_th[i]) for i in range(L)},
        "per_label": per_label_df,  # DataFrame
    }
    return summary



# y_true, y_prob láº¥y tá»« evaluate() cá»§a báº¡n
summary = eval_multilabel(VAL_TRUE, VAL_PROB, LABELS, optimize_th=True)

print("ROC-AUC macro:", round(summary["ROC_AUC_macro"], 4))
print("PR-AUC  macro:", round(summary["PR_AUC_macro"], 4))
print("F1 macro @0.5:", round(summary["F1_macro@0.5"], 4))
print("F1 macro @best:", round(summary["F1_macro@best"], 4))
print("best thresholds:", summary["best_thresholds"])

# Báº£ng per-label
summary["per_label"].sort_values("label")



import matplotlib.pyplot as plt

# Giáº£ sá»­ metrics_plot cÃ³ sáºµn
metrics_plot = {
    "ROC-AUC (macro)": 0.959,
    "PR-AUC (macro)": 0.708,
    "F1 macro @0.5": 0.643,
    "F1 macro @best": 0.693,
}

plt.figure(figsize=(8,6))
bars = plt.bar(metrics_plot.keys(), metrics_plot.values(),
               color="#1f77b4", edgecolor="black", width=0.6)

# TiÃªu Ä‘á»� & trá»¥c
plt.title("Evaluation Metrics on Validation/Test Set", fontsize=16, fontweight="bold", pad=15)
plt.ylabel("Score", fontsize=13)
plt.ylim(0, 1.05)
plt.xticks(rotation=15, fontsize=12)
plt.yticks(fontsize=11)

# Hiá»ƒn thá»‹ giÃ¡ trá»‹ trÃªn Ä‘áº§u cá»™t
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
             f"{height:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Grid nháº¹ Ä‘á»ƒ nhÃ¬n rÃµ
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()



# ==== 0) CÃ€I & IMPORT (náº¿u cáº§n) ====
# !pip -q install transformers torch --progress-bar off

import os, json, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==== 1) CHá»ŒN THÆ¯ Má»¤C MÃ” HÃŒNH ====
MODEL_DIR = "/kaggle/input/distilbert-toxic-best"  # Ä‘á»•i theo Ä‘Æ°á»�ng dáº«n cá»§a báº¡n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 2) Náº P TOKENIZER + MODEL ====
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

# ==== 3) NHÃƒN & NGÆ¯á» NG ====
# Náº¿u báº¡n dÃ¹ng bÃ i toÃ¡n Ä‘a nhÃ£n Jigsaw:
DEFAULT_LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# thresholds.json cÃ³ dáº¡ng {"toxic": 0.xx, ...}
thr_path = os.path.join(MODEL_DIR, "thresholds.json")
if os.path.exists(thr_path):
    with open(thr_path, "r") as f:
        th_dict = json.load(f)
else:
    th_dict = {lab:0.5 for lab in DEFAULT_LABELS}  # fallback

# Láº¥y sá»‘ nhÃ£n tá»« config cá»§a model Ä‘á»ƒ biáº¿t binary hay multilabel
num_labels = int(model.config.num_labels)
if num_labels == 1:
    LABELS = ["toxic"]
else:
    LABELS = DEFAULT_LABELS[:num_labels]

# vector ngÆ°á»¡ng theo Ä‘Ãºng thá»© tá»± LABELS
THRESHOLDS = np.array([th_dict.get(lab, 0.5) for lab in LABELS], dtype=float)

# ==== 4) HÃ€M Dá»° Ä�OÃ�N ====
@torch.no_grad()
def predict_probs(texts, max_len=192):
    if isinstance(texts, str):
        texts = [texts]
    enc = tok(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs  # shape [N, L]

def classify_texts(texts, max_len=192, return_breakdown=True):
    probs = predict_probs(texts, max_len=max_len)
    # náº¿u binary (num_labels==1) thÃ¬ THRESHOLDS cÃ³ 1 pháº§n tá»­
    preds = (probs >= THRESHOLDS).astype(int)

    # "overall toxic" = cÃ³ Ã­t nháº¥t 1 nhÃ£n báº­t; vá»›i binary chÃ­nh lÃ  cá»™t 0
    overall = preds.sum(axis=1) > 0
    if isinstance(texts, str):
        texts = [texts]

    df = pd.DataFrame({"text": texts})
    if num_labels == 1:
        df["p_toxic"] = probs[:, 0]
        df["pred"] = preds[:, 0]
        df["result"] = np.where(df["pred"]==1, "Toxic", "Non-Toxic")
    else:
        # Ä‘a nhÃ£n
        for i, lab in enumerate(LABELS):
            df[f"p_{lab}"] = probs[:, i]
            df[f"y_{lab}"] = preds[:, i]
        df["result"] = np.where(overall, "Toxic", "Non-Toxic")

    return df

# ==== 5) KIá»‚M THá»¬ NHANH ====
samples = [
    "I hate you. You are disgusting!",
    "This is a normal comment, nothing wrong here.",
    "Go back to your country!",
    "I'll kill you if you come again."
]
res = classify_texts(samples, return_breakdown=True)
res


