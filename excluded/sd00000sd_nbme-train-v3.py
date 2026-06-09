import kagglehub
nbme_score_clinical_patient_notes_path = kagglehub.competition_download('nbme-score-clinical-patient-notes')

print('Data source import complete.')


# -*- coding: utf-8 -*-
############################################################
# nbme_deberta_v3_fixed.py  â€“â€“ DeBERTa-v3 ç‰ˆæœ¬
############################################################
import os, ast, random, json, gc
from pathlib import Path
import glob
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold

import torch, torch.nn as nn
import io, logging

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer, AutoModel,
    get_cosine_schedule_with_warmup,
)

# -------------- 1. å…¨åŸŸè¨­å®š -------------------------------
class CFG:
    data_dir      = Path("/kaggle/input/nbme-score-clinical-patient-notes")
    competition   = "nbme-score-clinical-patient-notes"
    
    # ğŸ”¥ æ”¯æ�´æœ¬åœ°æ¨¡å�‹è·¯å¾‘
    use_local_model = True  # è¨­ç‚ºTrueä½¿ç”¨ä¸Šå‚³çš„æ¨¡å�‹
    local_model_path = "/kaggle/input/deberta-v3-base/deberta_base_cache"  # ğŸ”¥ ä¿®æ”¹ç‚ºä½ çš„å¯¦éš›è·¯å¾‘
    model_name    = "microsoft/deberta-v3-base"  # å‚™ç”¨æ¨¡å�‹å��ç¨±
    
    max_len       = 512
    batch_size    = 8
    gradient_accum = 2
    epochs        = 4
    lr            = 1e-5
    weight_decay  = 0.01
    scheduler     = "cosine"
    warmup_ratio  = 0.1
    n_folds       = 5
    seed          = 42
    output_dir    = "./nbme_ckpt"
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory optimization settings
    mixed_precision = False
    gradient_checkpointing = True
    
    # Debug settings
    debug_mode = False
    run_single_fold = False

# è¨­ç½®tokenizerä¸¦è¡Œ
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------- Utils: logger & seed --------------------
OUTPUT_DIR = CFG.output_dir

def get_logger(filename=OUTPUT_DIR + '/train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    log_path = Path(filename).parent
    log_path.mkdir(parents=True, exist_ok=True)

    logger = getLogger(__name__)
    if logger.handlers:
        return logger
    logger.setLevel(INFO)

    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))

    handler2 = FileHandler(f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

# ğŸ”¥ ç�²å�–å¯¦éš›æ¨¡å�‹è·¯å¾‘
def get_model_path():
    if CFG.use_local_model and Path(CFG.local_model_path).exists():
        LOGGER.info(f"ä½¿ç”¨æœ¬åœ°æ¨¡å�‹: {CFG.local_model_path}")
        return CFG.local_model_path
    else:
        LOGGER.info(f"ä½¿ç”¨ç·šä¸Šæ¨¡å�‹: {CFG.model_name}")
        return CFG.model_name

MODEL_PATH = get_model_path()

# -------------- Utils: seed & memory --------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def cleanup_memory():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Configure mixed precision
if CFG.mixed_precision:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
else:
    from contextlib import nullcontext
    autocast = nullcontext

# Save config
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

def _parse_list(x: str):
    try:
        return ast.literal_eval(x) if isinstance(x, str) and x else []
    except Exception:
        return []

def _find_all_spans(note: str, phrase: str) -> List[Tuple[int, int]]:
    spans = []
    for m in re.finditer(re.escape(phrase), note, flags=re.I):
        spans.append((m.start(), m.end()))
    return spans

def apply_annotation_fixes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    MANUAL_FIX: dict[int, tuple[str, str]] = {
        338:  ("father heart attack", "764 783"),
        621:  ("for the last 2-3 months", "77 100"),
        1262: ("mother thyroid problem", "551 572"),
    }
    for idx, (ann, loc) in MANUAL_FIX.items():
        if idx in df.index:
            df.at[idx, "annotation"] = f'["{ann}"]'
            df.at[idx, "location"]   = f'["{loc}"]'

    auto_fixed = 0
    for i, row in df.iterrows():
        anns = _parse_list(row["annotation"])
        locs = _parse_list(row["location"])
        if len(anns) == len(locs) and len(locs) > 0:
            continue
        note = row["pn_history"]
        new_locs = []
        used = set()
        for ann in anns:
            spans = _find_all_spans(note, ann)
            chosen = None
            for s, e in spans:
                if (s, e) not in used:
                    chosen = (s, e); break
            if chosen is None:
                break
            used.add(chosen)
            new_locs.append(f"{chosen[0]} {chosen[1]}")
        if len(new_locs) == len(anns):
            df.at[i, "location"] = str([*new_locs])
            auto_fixed += 1
    LOGGER.info(f"[annotation fix] manual={len(MANUAL_FIX)}, auto={auto_fixed}")
    return df

# Merge data
train = (train.merge(feats, on=["feature_num","case_num"], how="left")
               .merge(pnotes, on=["pn_num","case_num"],    how="left"))
test  = (test .merge(feats, on=["feature_num","case_num"], how="left")
               .merge(pnotes, on=["pn_num","case_num"],    how="left"))

train = apply_annotation_fixes(train)

def str2list(x):
    return ast.literal_eval(x) if isinstance(x,str) and x!="" else []

train["annotation_list"] = train["annotation"].apply(str2list)
train["location_list"]   = train["location"].apply(str2list)

# æ•¸æ“šè³ªé‡�æª¢æŸ¥
def debug_data_quality(train_df):
    if not CFG.debug_mode:
        return
        
    LOGGER.info("=== æ•¸æ“šè³ªé‡�æª¢æŸ¥ ===")
    
    empty_annotations = 0
    empty_locations = 0
    mismatch_count = 0
    valid_samples = 0
    
    for idx, row in train_df.iterrows():
        anns = row["annotation_list"]
        locs = row["location_list"]
        
        if len(anns) == 0:
            empty_annotations += 1
        if len(locs) == 0:
            empty_locations += 1
        if len(anns) != len(locs):
            mismatch_count += 1
        if len(anns) > 0 and len(locs) > 0 and len(anns) == len(locs):
            valid_samples += 1
    
    LOGGER.info(f"Empty annotations: {empty_annotations}")
    LOGGER.info(f"Empty locations: {empty_locations}")
    LOGGER.info(f"Annotation-location mismatch: {mismatch_count}")
    LOGGER.info(f"Valid samples: {valid_samples}")
    LOGGER.info(f"Total rows: {len(train_df)}")

debug_data_quality(train)

# ğŸ”¥ è¼‰å…¥ tokenizer
def load_tokenizer():
    """è¼‰å…¥ tokenizerï¼Œå„ªå…ˆä½¿ç”¨æœ¬åœ°æ¨¡å�‹"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        LOGGER.info(f"âœ… Tokenizer loaded from: {MODEL_PATH}")
        return tokenizer
    except Exception as e:
        LOGGER.error(f"â�Œ Tokenizer loading failed: {e}")
        if CFG.use_local_model:
            LOGGER.info("å˜—è©¦ä½¿ç”¨ç·šä¸Šæ¨¡å�‹...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)
                LOGGER.info(f"âœ… Tokenizer loaded from online: {CFG.model_name}")
                return tokenizer
            except Exception as e2:
                LOGGER.error(f"â�Œ Online tokenizer also failed: {e2}")
                raise
        else:
            raise

tok = load_tokenizer()

def create_char_targets(text: str, spans: List[str]) -> np.ndarray:
    """å‰µå»ºå­—ç¬¦ç´šåˆ¥çš„ç›®æ¨™æ¨™ç±¤"""
    targets = np.zeros(len(text), dtype=np.int8)
    for span in spans:
        if not span:
            continue
        # ğŸ”¥ ä¿®å¾©ï¼šæ­£ç¢ºè™•ç�†åˆ†è™Ÿåˆ†éš”çš„å¤šå€‹ä½�ç½®
        for loc in span.split(";"):
            loc = loc.strip()
            if not loc:
                continue
            try:
                start, end = map(int, loc.split())
                if start >= len(text):
                    continue
                end = min(end, len(text))
                if start < end:  # ç¢ºä¿�æœ‰æ•ˆç¯„åœ�
                    targets[start:end] = 1
            except (ValueError, IndexError):
                continue
    return targets

def encode_example(note: str, feature: str, targets: np.ndarray | None):
    """ç·¨ç¢¼æ¨£æœ¬ä¸¦å‰µå»ºtokenç´šåˆ¥æ¨™ç±¤"""
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
            # å�ªè™•ç�†æ–‡æœ¬éƒ¨åˆ†çš„tokens
            if seq_ids[idx] != 1 or s == e:
                continue
            
            # ğŸ”¥ ä¿®å¾©ï¼šæ”¹é€²token-characterå°�æ‡‰é‚�è¼¯
            if s < len(targets) and e <= len(targets):
                # å¦‚æ�œtokenè¦†è“‹çš„ä»»ä½•å­—ç¬¦è¢«æ¨™è¨˜ç‚ºæ­£ä¾‹ï¼Œå‰‡tokenç‚ºæ­£ä¾‹
                if targets[s:e].sum() > 0:
                    labels[idx] = 1.0
        
        enc["labels"] = labels

    enc.pop("offset_mapping")

    tensor_dict = {}
    for k, v in enc.items():
        if k == "labels":
            tensor_dict[k] = torch.tensor(v, dtype=torch.float)
        else:
            tensor_dict[k] = torch.tensor(v, dtype=torch.long)
    return tensor_dict

# æ¨™ç±¤å»ºç«‹æª¢æŸ¥
def debug_label_creation(train_df, sample_size=3):
    if not CFG.debug_mode:
        return
        
    LOGGER.info("\n=== æ¨™ç±¤å»ºç«‹æª¢æŸ¥ ===")
    
    valid_samples = []
    for idx, row in train_df.iterrows():
        if len(row["location_list"]) > 0:
            valid_samples.append((idx, row))
        if len(valid_samples) >= sample_size:
            break
    
    for i, (idx, row) in enumerate(valid_samples):
        char_targets = create_char_targets(row.pn_history, row.location_list)
        encoded = encode_example(row.pn_history, row.feature_text, char_targets)
        
        LOGGER.info(f"\n--- Sample {i} (Row {idx}) ---")
        LOGGER.info(f"Text length: {len(row.pn_history)}")
        LOGGER.info(f"Location list: {row.location_list}")
        LOGGER.info(f"Char targets sum: {char_targets.sum()}")
        LOGGER.info(f"Token labels sum: {encoded['labels'].sum().item()}")
        LOGGER.info(f"Total tokens: {len(encoded['labels'])}")
        
        # é¡¯ç¤ºä¸€äº›ground truthæ–‡æœ¬
        for loc in row.location_list:
            try:
                start, end = map(int, loc.split())
                gt_text = row.pn_history[start:end]
                LOGGER.info(f"GT span ({start},{end}): '{gt_text}'")
            except:
                pass

debug_label_creation(train)

# -------------- 4. è‡ªè¨‚ Dataset ---------------------------
class NBMEDataset(Dataset):
    def __init__(self, df:pd.DataFrame, is_train=True):
        self.df = df
        self.is_train = is_train
    
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.is_train:
            char_targets = create_char_targets(row.pn_history, row.location_list)
        else:
            char_targets = None
        return encode_example(row.pn_history, row.feature_text, char_targets)

# -------------- 5. è©•åˆ†å·¥å…· ------------------------
def span_to_char_set(spans: List[str]) -> set[int]:
    """å°‡spanåˆ—è¡¨è½‰æ�›ç‚ºå­—ç¬¦ç´¢å¼•é›†å�ˆ"""
    char_set = set()
    for sp in spans:
        if not sp:
            continue
        for loc in sp.split(";"):
            loc = loc.strip()
            if not loc:
                continue
            try:
                a, b = map(int, loc.split())
                if a < b:
                    char_set.update(range(a, b))
            except (ValueError, IndexError):
                continue
    return char_set

def compute_micro_f1(pred_df: pd.DataFrame) -> float:
    """è¨ˆç®—micro F1åˆ†æ•¸"""
    tp = fp = fn = 0
    for g, p in zip(pred_df.ground, pred_df.pred):
        gset = span_to_char_set([g] if isinstance(g, str) else g)
        pset = span_to_char_set([p] if isinstance(p, str) else p)
        tp += len(gset & pset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    return 2 * tp / (2 * tp + fp + fn + 1e-8)

# ğŸ”¥ ä¿®å¾©ï¼šæ”¹é€²spané‡�å»ºé‚�è¼¯
def reconstruct_spans_from_char_probs(char_prob: np.ndarray, threshold: float = 0.5, min_span_len: int = 1) -> List[str]:
    """å¾�å­—ç¬¦æ¦‚ç�‡é‡�å»ºspans"""
    spans = []
    start = None
    
    for idx, prob in enumerate(char_prob):
        if prob >= threshold and start is None:
            start = idx
        elif (prob < threshold or idx == len(char_prob) - 1) and start is not None:
            end = idx if prob < threshold else idx + 1
            # å�ªä¿�ç•™è¶³å¤ é•·çš„spans
            if end - start >= min_span_len:
                spans.append(f"{start} {end}")
            start = None
    
    return spans

def find_optimal_threshold(pred_logits, val_idx, train_df):
    """å°‹æ‰¾æœ€ä½³é–¾å€¼"""
    LOGGER.info("\n=== å°‹æ‰¾æœ€ä½³é–¾å€¼ ===")
    
    thresholds = np.arange(0.01, 0.99, 0.02)  # æ›´ç´°ç²’åº¦çš„é–¾å€¼æ�œç´¢
    best_f1 = 0
    best_th = 0.5
    
    for th in thresholds:
        fold_preds = []
        fold_gts = []
        
        for i, row in enumerate(train_df.iloc[val_idx].itertuples()):
            prob = pred_logits[i]
            
            # ç·¨ç¢¼ä»¥ç�²å�–offset mapping
            enc = tok(row.feature_text, row.pn_history,
                      truncation="only_second", max_length=CFG.max_len,
                      return_offsets_mapping=True)
            
            # é‡�å»ºå­—ç¬¦ç´šæ¦‚ç�‡
            char_prob = np.zeros(len(row.pn_history))
            seq_ids = enc.sequence_ids()
            
            for t, (s, e) in enumerate(enc["offset_mapping"]):
                if seq_ids[t] == 1 and s < e and s < len(char_prob):
                    end_idx = min(e, len(char_prob))
                    char_prob[s:end_idx] = np.maximum(char_prob[s:end_idx], prob[t])
            
            # é‡�å»ºspans
            pred_spans = reconstruct_spans_from_char_probs(char_prob, th)
            pred_span = ";".join(pred_spans)
            
            fold_preds.append(pred_span)
            fold_gts.append(";".join(row.location_list))
        
        f1 = compute_micro_f1(pd.DataFrame({"ground": fold_gts, "pred": fold_preds}))
        
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            if CFG.debug_mode and f1 > 0:
                LOGGER.info(f"Threshold {th:.3f}: F1={f1:.4f} â­�")
    
    LOGGER.info(f"Best threshold: {best_th:.3f} (F1={best_f1:.4f})")
    return best_th

# -------------- 6. æ¨¡å�‹ -------------------------------
class DebertaV3ForTokenBinary(nn.Module):
    def __init__(self):
        super().__init__()
        # ğŸ”¥ ä½¿ç”¨çµ±ä¸€çš„æ¨¡å�‹è·¯å¾‘è¼‰å…¥
        try:
            self.backbone = AutoModel.from_pretrained(MODEL_PATH)
            LOGGER.info(f"âœ… Model loaded from: {MODEL_PATH}")
        except Exception as e:
            LOGGER.error(f"â�Œ Model loading failed: {e}")
            if CFG.use_local_model:
                LOGGER.info("å˜—è©¦ä½¿ç”¨ç·šä¸Šæ¨¡å�‹...")
                try:
                    self.backbone = AutoModel.from_pretrained(CFG.model_name)
                    LOGGER.info(f"âœ… Model loaded from online: {CFG.model_name}")
                except Exception as e2:
                    LOGGER.error(f"â�Œ Online model also failed: {e2}")
                    raise
            else:
                raise
        
        if CFG.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)
        
    def forward(self, **batch):
        labels = batch.pop("labels", None)
        out = self.backbone(**batch)
        logits = self.classifier(self.dropout(out.last_hidden_state)).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {"loss": loss, "logits": logits}

# -------------- 7. äº¤å�‰é©—è­‰è¨“ç·´ ------------------------
oof_preds, oof_gts = [], []
gkf = GroupKFold(n_splits=CFG.n_folds)

fold_splits = list(gkf.split(train, groups=train.pn_num))
if CFG.run_single_fold:
    fold_splits = fold_splits[:1]

for fold, (trn_idx, val_idx) in enumerate(fold_splits):
    LOGGER.info(f"\n========== FOLD {fold} ==========")
    
    cleanup_memory()
    
    trn_ds = NBMEDataset(train.iloc[trn_idx])
    val_ds = NBMEDataset(train.iloc[val_idx])
    
    pin_memory = CFG.device == "cuda"
    trn_loader = DataLoader(trn_ds, batch_size=CFG.batch_size,
                            shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size,
                            shuffle=False, num_workers=0, pin_memory=pin_memory)

    model = DebertaV3ForTokenBinary().to(CFG.device)  # ğŸ”¥ ä½¿ç”¨æ–°çš„é¡�å��
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr,
                                  weight_decay=CFG.weight_decay)
    num_training_steps = CFG.epochs * len(trn_loader) // CFG.gradient_accum
    num_warmup = int(CFG.warmup_ratio * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup, num_training_steps)

    best_f1 = -1
    best_path = Path(CFG.output_dir) / f"fold{fold}.pt"
    
    for epoch in range(CFG.epochs):
        # Training
        model.train()
        running_loss = 0
        pbar = tqdm(trn_loader, desc=f"Train E{epoch}", dynamic_ncols=True, leave=False)
        
        for step, batch in enumerate(pbar):
            batch = {k: v.to(CFG.device) for k, v in batch.items()}
            
            if CFG.mixed_precision:
                with autocast():
                    out = model(**batch)
                    loss = out["loss"] / CFG.gradient_accum
                scaler.scale(loss).backward()
            else:
                out = model(**batch)
                loss = out["loss"] / CFG.gradient_accum
                loss.backward()
            
            running_loss += loss.item()
            
            if (step + 1) % CFG.gradient_accum == 0 or step + 1 == len(trn_loader):
                if CFG.mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                pbar.set_postfix(loss=running_loss * CFG.gradient_accum / (step + 1))
        
        # Validation
        model.eval()
        preds = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Valid", dynamic_ncols=True, leave=False):
                batch = {k: v.to(CFG.device) for k, v in batch.items()}
                
                if CFG.mixed_precision:
                    with autocast():
                        logits = model(**batch)["logits"].sigmoid().cpu().numpy()
                else:
                    logits = model(**batch)["logits"].sigmoid().cpu().numpy()
                preds.append(logits)
        
        pred_logits = np.concatenate(preds, axis=0)
        
        # å°‹æ‰¾æœ€ä½³é–¾å€¼
        best_th = find_optimal_threshold(pred_logits, val_idx, train)
        
        # ä½¿ç”¨æœ€ä½³é–¾å€¼è¨ˆç®—F1
        fold_preds = []
        fold_gts = []
        
        for i, row in enumerate(train.iloc[val_idx].itertuples()):
            prob = pred_logits[i]
            
            enc = tok(row.feature_text, row.pn_history,
                      truncation="only_second", max_length=CFG.max_len,
                      return_offsets_mapping=True)
            
            char_prob = np.zeros(len(row.pn_history))
            seq_ids = enc.sequence_ids()
            
            for t, (s, e) in enumerate(enc["offset_mapping"]):
                if seq_ids[t] == 1 and s < e and s < len(char_prob):
                    end_idx = min(e, len(char_prob))
                    char_prob[s:end_idx] = np.maximum(char_prob[s:end_idx], prob[t])
            
            pred_spans = reconstruct_spans_from_char_probs(char_prob, best_th)
            pred_span = ";".join(pred_spans)
            
            fold_preds.append(pred_span)
            fold_gts.append(";".join(row.location_list))
        
        oof_preds.extend(fold_preds)
        oof_gts.extend(fold_gts)
        
        f1 = compute_micro_f1(pd.DataFrame({"ground": fold_gts, "pred": fold_preds}))
        LOGGER.info(f"Fold {fold} Epoch {epoch} F1={f1:.4f} (threshold={best_th:.3f})")
        
        if f1 > best_f1:
            best_f1 = f1
            # ğŸ”¥ æ›´å®‰å…¨çš„ä¿�å­˜æ–¹å¼�ï¼Œå�ªä¿�å­˜å¿…è¦�æ•¸æ“š
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'best_threshold': float(best_th),  # ç¢ºä¿�æ˜¯Python float
                'best_f1': float(best_f1),  # ç¢ºä¿�æ˜¯Python float
                'epoch': epoch,
                'fold': fold
            }
            torch.save(checkpoint_data, best_path)
    
    LOGGER.info(f"Fold {fold} best F1={best_f1:.4f}")
    
    del model, optimizer, scheduler
    cleanup_memory()

# -------------- 8. æ•´é«” OOF åˆ†æ•¸ ------------------------
overall_f1 = compute_micro_f1(pd.DataFrame({"ground": oof_gts, "pred": oof_preds}))
LOGGER.info(f"\n========== CV micro-F1: {overall_f1:.4f} ==========")

# -------------- 9. æ¸¬è©¦æ�¨è«– & æ��äº¤ ----------------------
if not CFG.run_single_fold:
    test_ds = NBMEDataset(test, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size,
                             shuffle=False, num_workers=0, pin_memory=False)

    all_preds = []
    all_thresholds = []

    for fold in range(CFG.n_folds):
        LOGGER.info(f"Loading fold {fold} for inference...")
        cleanup_memory()
        
        model = DebertaV3ForTokenBinary().to(CFG.device)  # ğŸ”¥ ä½¿ç”¨æ–°çš„é¡�å��
        
        # ğŸ”¥ æ›´å®‰å…¨çš„åŠ è¼‰æ–¹å¼�
        try:
            checkpoint = torch.load(Path(CFG.output_dir) / f"fold{fold}.pt", map_location=CFG.device)
        except Exception as e:
            LOGGER.info(f"Loading with weights_only=False due to: {e}")
            checkpoint = torch.load(Path(CFG.output_dir) / f"fold{fold}.pt", map_location=CFG.device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        all_thresholds.append(checkpoint['best_threshold'])
        model.eval()
        
        fold_pred = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Inference F{fold}", leave=False):
                batch = {k: v.to(CFG.device) for k, v in batch.items()}
                
                if CFG.mixed_precision:
                    with autocast():
                        logits = model(**batch)["logits"].sigmoid().cpu().numpy()
                else:
                    logits = model(**batch)["logits"].sigmoid().cpu().numpy()
                fold_pred.append(logits)
        
        all_preds.append(np.concatenate(fold_pred, axis=0))
        del model
        cleanup_memory()

    # å¹³å�‡é �æ¸¬å’Œé–¾å€¼
    pred_logits = np.mean(all_preds, axis=0)
    avg_threshold = np.mean(all_thresholds)
    
    LOGGER.info(f"Using average threshold: {avg_threshold:.3f}")

    subs = []
    for i, row in enumerate(test.itertuples()):
        enc = tok(row.feature_text, row.pn_history,
                  truncation="only_second", max_length=CFG.max_len,
                  return_offsets_mapping=True)
        
        char_prob = np.zeros(len(row.pn_history))
        seq_ids = enc.sequence_ids()
        
        for t, (s, e) in enumerate(enc["offset_mapping"]):
            if seq_ids[t] == 1 and s < e and s < len(char_prob):
                end_idx = min(e, len(char_prob))
                # ğŸ”¥ ä¿®å¾©ï¼šä½¿ç”¨ np.maximum è™•ç�†æ•¸çµ„
                char_prob[s:end_idx] = np.maximum(char_prob[s:end_idx], pred_logits[i, t])
        
        pred_spans = reconstruct_spans_from_char_probs(char_prob, avg_threshold)
        subs.append({"id": row.id, "location": ";".join(pred_spans)})

    sub_df = pd.DataFrame(subs)
    sub_df.to_csv("submission.csv", index=False)
    LOGGER.info("submission.csv saved!")
else:
    LOGGER.info("Debug mode: Set CFG.run_single_fold=False for full training.")

cleanup_memory()

if CFG.debug_mode:
    LOGGER.info("\n=== èª¿è©¦ç¸½çµ� ===")
    LOGGER.info(f"Training completed with CV F1: {overall_f1:.4f}")
    if overall_f1 > 0:
        LOGGER.info("è¨“ç·´æˆ�åŠŸï¼�")
    else:
        LOGGER.info("ä»�éœ€è¦�é€²ä¸€æ­¥èª¿è©¦")

