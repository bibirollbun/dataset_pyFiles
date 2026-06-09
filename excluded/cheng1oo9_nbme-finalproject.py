############################################################
# nbme_deberta_v3_small.py  â€“â€“ Kaggle 1-click
############################################################
import os, ast, random, json
from pathlib import Path
import glob
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold

import torch, torch.nn as nn
import io, logging  # for tqdmâ†’logger bridge

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer, AutoModel,
    get_cosine_schedule_with_warmup,
)

# -------------- 1. å…¨åŸŸè¨­å®š -------------------------------
class CFG:
    data_dir      = Path("/kaggle/input/nbme-score-clinical-patient-notes")  # Local rawâ€‘data directory
    competition   = "nbme-score-clinical-patient-notes"  # kept for reference
    model_name    = "/kaggle/input/deberta-v3-small/transformers/default/1"  # local checkpoint
    max_len       = 512          # ç”±final_EDA å¾—çŸ¥ 443çš„é•·åº¦å…¶å¯¦å·²ç¶“è¶³å¤ ï¼Œé �ç•™é¤˜åº¦
    batch_size    = 16
    gradient_accum = 1
    epochs        = 4
    lr            = 2e-5
    weight_decay  = 0.01
    scheduler     = "cosine"
    warmup_ratio  = 0.1
    n_folds       = 5
    seed          = 42
    output_dir    = "./nbme_ckpt"
    device        = "cuda" if torch.cuda.is_available() else "cpu"

# -------------- Utils: logger & seed --------------------
OUTPUT_DIR = CFG.output_dir  # For log file path

def get_logger(filename=OUTPUT_DIR + '/train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    log_path = Path(filename).parent
    log_path.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    logger = getLogger(__name__)
    if logger.handlers:  # avoid duplicate handlers
        return logger
    logger.setLevel(INFO)

    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))

    handler2 = FileHandler(f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

class _TqdmToLogger(io.StringIO):
    """Fileâ€‘like object that redirects tqdm output to our LOGGER."""
    def __init__(self, logger: logging.Logger, level=logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, buf):
        buf = buf.strip()
        if buf:
            self.logger.log(self.level, buf)

    def flush(self):
        pass  # tqdm expects this method.

TQDM_LOGGER = _TqdmToLogger(logging.getLogger(__name__))

LOGGER = get_logger()

def seed_everything(seed: int = 42):
    """Seed all RNGs for full reproducibility (deterministic cuDNN)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# å�¯é�¸ï¼šå°‡ CFG è¼¸å‡ºæˆ� JSON ä¾¿æ–¼æ—¥å¾Œè¿½è¹¤

from pathlib import Path
(Path(CFG.output_dir).mkdir(exist_ok=True, parents=True))
cfg_dict = {
    k: (str(v) if isinstance(v, Path) else v)
    for k, v in CFG.__dict__.items()
    if not k.startswith("__") and not callable(v)
}
json.dump(cfg_dict, open(Path(CFG.output_dir)/"cfg.json", "w"), indent=2)

seed_everything(CFG.seed)

# -------------- 2. è³‡æ–™è®€å�–èˆ‡é �è™•ç�† -----------------------
BASE = CFG.data_dir
# æ”¯æ�´ä»»æ„�æª”å��å‰�ç¶´ï¼›ä»¥ *train*, *test* ç­‰é—œé�µå­—è¾¨è­˜
def _find_csv(keyword: str) -> Path:
    patt = str(BASE / "*.csv")
    files = [Path(f) for f in glob.glob(patt) if keyword.lower() in Path(f).stem.lower()]
    if not files:
        raise FileNotFoundError(f"[Error] CSV containing '{keyword}' not found under {BASE.resolve()}")
    return files[0]

train  = pd.read_csv(_find_csv("train"))
test   = pd.read_csv(_find_csv("test"))
feats  = pd.read_csv(_find_csv("features"))
pnotes = pd.read_csv(_find_csv("patient_notes"))

# ---------- ä¿®è£œ incorrect annotations -----------------
def _parse_list(x: str):
    """Safe eval to list; returns [] on empty/malformed."""
    try:
        return ast.literal_eval(x) if isinstance(x, str) and x else []
    except Exception:
        return []

def _find_all_spans(note: str, phrase: str) -> List[Tuple[int, int]]:
    """Return all (start,end) spans (inclusive-exclusive) of phrase in note, case-insensitive."""
    spans = []
    for m in re.finditer(re.escape(phrase), note, flags=re.I):
        spans.append((m.start(), m.end()))
    return spans

def apply_annotation_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known bad rows and attempt autoâ€‘repair when location missing.
    Returns a *new* DataFrame (copy).
    """
    df = df.copy()

    # 1) manual whitelist based on public notebooks
    MANUAL_FIX: dict[int, tuple[str, str]] = {
        338:  ("father heart attack", "764 783"),
        621:  ("for the last 2-3 months", "77 100"),
        1262: ("mother thyroid problem", "551 572"),
    }
    for idx, (ann, loc) in MANUAL_FIX.items():
        if idx in df.index:
            df.at[idx, "annotation"] = f'["{ann}"]'
            df.at[idx, "location"]   = f'["{loc}"]'

    # 2) auto repair for rows where len(annotation_list) != len(location_list)
    auto_fixed = 0
    for i, row in df.iterrows():
        anns = _parse_list(row["annotation"])
        locs = _parse_list(row["location"])
        if len(anns) == len(locs) and len(locs) > 0:
            continue  # already aligned
        note = row["pn_history"]
        new_locs = []
        used = set()
        for ann in anns:
            spans = _find_all_spans(note, ann)
            # pick the first span not yet used
            chosen = None
            for s, e in spans:
                if (s, e) not in used:
                    chosen = (s, e); break
            if chosen is None:
                break  # cannot find unique span
            used.add(chosen)
            new_locs.append(f"{chosen[0]} {chosen[1]}")
        if len(new_locs) == len(anns):  # successful repair
            df.at[i, "location"] = str([*new_locs])
            auto_fixed += 1
    LOGGER.info(f"[annotation fix] manual={len(MANUAL_FIX)}, auto={auto_fixed}")
    return df

#
# merge
train = (train.merge(feats, on=["feature_num","case_num"], how="left")
               .merge(pnotes, on=["pn_num","case_num"],    how="left"))
test  = (test .merge(feats, on=["feature_num","case_num"], how="left")
               .merge(pnotes, on=["pn_num","case_num"],    how="left"))

# apply annotation fixes now that pn_history is present
train = apply_annotation_fixes(train)

# è§£æ�� annotation / location æˆ� listâ‡¢list[int]
def str2list(x): 
    return ast.literal_eval(x) if isinstance(x,str) and x!="" else []
train["annotation_list"] = train["annotation"].apply(str2list)
train["location_list"]   = train["location"].apply(str2list)

tok = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)
def create_char_targets(text: str, spans: List[str]) -> np.ndarray:
    """Return a 0/1 vector marking characters belonging to any span."""
    targets = np.zeros(len(text), dtype=np.int8)
    for span in spans:
        if not span:
            continue
        for loc in span.split(";"):
            if loc == "":
                continue
            try:
                start, end = map(int, loc.split())
            except ValueError:
                continue  # skip malformed
            if start >= len(text):
                continue
            end = min(end, len(text))
            targets[start:end] = 1
    return targets

def encode_example(note: str, feature: str, targets: np.ndarray | None):
    """
    Tokenise (feature, note) pair and return a dict of tensors compatible
    with the model.  `labels` tensor is float32 for BCE loss; others are long.
    """
    enc = tok(
        feature,
        note,
        truncation="only_second",
        padding="max_length",
        max_length=CFG.max_len,
        return_offsets_mapping=True
    )

    if targets is not None:
        labels = np.zeros(len(enc["input_ids"]), dtype=np.float32)
        seq_ids = enc.sequence_ids()
        for idx, (s, e) in enumerate(enc["offset_mapping"]):
            if seq_ids[idx] != 1 or s == e:
                continue
            if targets[s:e].max() > 0:
                labels[idx] = 1.0
        enc["labels"] = labels

    enc.pop("offset_mapping")  # no longer needed

    tensor_dict = {}
    for k, v in enc.items():
        if k == "labels":
            tensor_dict[k] = torch.tensor(v, dtype=torch.float)
        else:
            tensor_dict[k] = torch.tensor(v, dtype=torch.long)
    return tensor_dict

# -------------- 4. è‡ªè¨‚ Dataset ---------------------------
class NBMEDataset(Dataset):
    def __init__(self, df:pd.DataFrame, is_train=True):
        self.df = df
        self.is_train = is_train
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.is_train:
            # row.location_list is already List[str] like ["12 20", "32 40"]
            # just pass it directly
            char_targets = create_char_targets(row.pn_history, row.location_list)
        else:
            char_targets = None
        return encode_example(row.pn_history, row.feature_text, char_targets)

# -------------- 5. è©•åˆ†å·¥å…· (rule.md è¦�å‰‡) ----------------
def span_to_char_set(spans: List[str]) -> set[int]:
    """
    Convert list of span strings ["12 20", "30 35;40 45", ...] to a set of char indices.
    Any malformed span tokens are skipped gracefully.
    """
    char_set = set()
    for sp in spans:
        if not sp:
            continue
        for loc in sp.split(";"):
            if not loc.strip():
                continue
            parts = loc.split()
            if len(parts) != 2:
                continue  # skip malformed "start end"
            try:
                a, b = map(int, parts)
            except ValueError:
                continue
            if a >= b:
                continue
            char_set.update(range(a, b))
    return char_set
def compute_micro_f1(pred_df:pd.DataFrame) -> float:
    """pred_df éœ€å�« columns: id, ground (list[str]), pred (list[str])"""
    tp=fp=fn=0
    for g,p in zip(pred_df.ground, pred_df.pred):
        gset, pset = span_to_char_set(g), span_to_char_set(p)
        tp+=len(gset&pset); fp+=len(pset-gset); fn+=len(gset-pset)
    return 2*tp/(2*tp+fp+fn+1e-8)
def span_micro_f1(y_true, y_pred):
    """Alias wrapper so external utils can call the same scorer name."""
    return compute_micro_f1(pd.DataFrame({"ground": y_true, "pred": y_pred}))
def get_score(y_true, y_pred):
    return span_micro_f1(y_true, y_pred)

# -------------- 6. æ¨¡å�‹ -------------------------------
class DebertaForTokenBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(CFG.model_name)
        self.dropout  = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)
    def forward(self, **batch):
        labels = batch.pop("labels", None)
        out = self.backbone(**batch)
        logits = self.classifier(self.dropout(out.last_hidden_state)).squeeze(-1)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        return {"loss":loss, "logits":logits}

# -------------- 7. äº¤å�‰é©—è­‰è¨“ç·´ ------------------------
oof_preds, oof_gts = [], []
gkf = GroupKFold(n_splits=CFG.n_folds)
for fold,(trn_idx,val_idx) in enumerate(gkf.split(train, groups=train.pn_num)):
    LOGGER.info(f"\n========== FOLD {fold} ==========")
    trn_ds = NBMEDataset(train.iloc[trn_idx])
    val_ds = NBMEDataset(train.iloc[val_idx])
    pin_memory = CFG.device == "cuda"
    trn_loader = DataLoader(trn_ds, batch_size=CFG.batch_size,
                            shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size,
                            shuffle=False,num_workers=2, pin_memory=pin_memory)

    model = DebertaForTokenBinary().to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr,
                                  weight_decay=CFG.weight_decay)
    num_training_steps = CFG.epochs * len(trn_loader) // CFG.gradient_accum
    num_warmup = int(CFG.warmup_ratio * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup, num_training_steps)

    best_f1 = -1; best_path = Path(CFG.output_dir)/f"fold{fold}.pt"
    for epoch in range(CFG.epochs):
        # ---- train
        model.train(); running=0
        pbar = tqdm(
            trn_loader,
            total=len(trn_loader),
            desc=f"Train E{epoch}",
            dynamic_ncols=True,
            leave=False,
        )
        for step,batch in enumerate(pbar):
            batch = {k:v.to(CFG.device) for k,v in batch.items()}
            out = model(**batch)
            loss = out["loss"]/CFG.gradient_accum
            loss.backward()
            running += loss.item()
            if (step+1)%CFG.gradient_accum==0 or step+1==len(trn_loader):
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                pbar.set_postfix(loss=running/((step+1)//CFG.gradient_accum+1e-6))
        # ---- valid
        model.eval(); preds=[]; gts=[]; ids=[]
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Valid", dynamic_ncols=True, leave=False):
                ids.extend(batch["input_ids"].shape[0]*[None]) # placeholder
                batch = {k:v.to(CFG.device) for k,v in batch.items()}
                logits = model(**batch)["logits"].sigmoid().cpu().numpy()
                preds.append(logits)
        pred_logits = np.concatenate(preds, axis=0)
        if pred_logits.shape[0] != len(val_idx):
            raise RuntimeError(f"Mismatch between predictions ({pred_logits.shape[0]}) and validation size ({len(val_idx)})")
        # ğŸ‘‰ char-levelé‡�å»º
        th=0.5
        for i,row in enumerate(train.iloc[val_idx].itertuples()):
            prob = pred_logits[i]
            # token offset éœ€é‡�æ–°è¨ˆç®—
            enc = tok(row.feature_text,row.pn_history,
                      truncation="only_second", max_length=CFG.max_len,
                      return_offsets_mapping=True)
            char_prob = np.zeros(len(row.pn_history))
            seq_ids_val = enc.sequence_ids()
            for t,(s,e) in enumerate(enc["offset_mapping"]):
                if seq_ids_val[t]==1 and s<e:
                    if s < len(char_prob):
                        char_prob[s:min(e, len(char_prob))] = prob[t]
            # æ ¹æ“šé–¾å€¼å�–é€£çºŒå�€æ®µ
            spans=[]; start=None
            for idx,pv in enumerate(char_prob):
                if pv>=th and start is None:
                    start=idx
                elif (pv<th or idx==len(char_prob)-1) and start is not None:
                    end=idx if pv<th else idx+1
                    spans.append(f"{start} {end}")
                    start=None
            pred_span=";".join(spans)
            oof_preds.append(pred_span)
            oof_gts.append(";".join(row.location_list))
        f1 = compute_micro_f1(pd.DataFrame({"ground":oof_gts[-len(val_idx):],
                                            "pred":  oof_preds[-len(val_idx):]}))
        LOGGER.info(f"Fold {fold} Epoch {epoch} F1={f1:.4f}")
        if f1>best_f1:
            best_f1=f1
            torch.save(model.state_dict(), best_path)
    LOGGER.info(f"Fold {fold} best F1={best_f1:.4f}")

# -------------- 8. æ•´é«” OOF åˆ†æ•¸ ------------------------
overall_f1 = compute_micro_f1(pd.DataFrame({"ground":oof_gts,"pred":oof_preds}))
LOGGER.info(f"\n========== CV micro-F1: {overall_f1:.4f} ==========")

# -------------- 9. æ¸¬è©¦æ�¨è«– & æ��äº¤ ----------------------
test_ds = NBMEDataset(test, is_train=False)
pin_memory = CFG.device == "cuda"
test_loader = DataLoader(test_ds, batch_size=CFG.batch_size,
                         shuffle=False, num_workers=2, pin_memory=pin_memory)

all_preds=[]
for fold in range(CFG.n_folds):
    model = DebertaForTokenBinary().to(CFG.device)
    model.load_state_dict(torch.load(Path(CFG.output_dir)/f"fold{fold}.pt",
                                     map_location=CFG.device))
    model.eval(); fold_pred=[]
    with torch.no_grad():
        for batch in test_loader:
            batch = {k:v.to(CFG.device) for k,v in batch.items()}
            logits = model(**batch)["logits"].sigmoid().cpu().numpy()
            fold_pred.append(logits)
    all_preds.append(np.concatenate(fold_pred,axis=0))
# K-fold average
pred_logits = np.mean(all_preds, axis=0)

subs=[]
for i,row in enumerate(test.itertuples()):
    enc = tok(row.feature_text,row.pn_history,
              truncation="only_second", max_length=CFG.max_len,
              return_offsets_mapping=True)
    char_prob = np.zeros(len(row.pn_history))
    seq_ids_test = enc.sequence_ids()
    for t,(s,e) in enumerate(enc["offset_mapping"]):
        if seq_ids_test[t]==1 and s<e:
            if s < len(char_prob):
                char_prob[s:min(e, len(char_prob))] = pred_logits[i,t]
    # same span-recovery as val
    spans=[];start=None;th=0.5
    for idx,pv in enumerate(char_prob):
        if pv>=th and start is None:
            start=idx
        elif (pv<th or idx==len(char_prob)-1) and start is not None:
            end=idx if pv<th else idx+1
            spans.append(f"{start} {end}")
            start=None
    subs.append({"id":row.id, "location":";".join(spans)})

sub_df = pd.DataFrame(subs)
sub_df.to_csv("submission.csv", index=False)
LOGGER.info("âœ… submission.csv saved!")

