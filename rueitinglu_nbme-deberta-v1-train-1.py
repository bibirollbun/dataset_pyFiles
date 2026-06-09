import kagglehub
nbme_score_clinical_patient_notes_path = kagglehub.competition_download('nbme-score-clinical-patient-notes')

print('Data source import complete.')


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
import torch.nn.functional as F
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
    model_name    = "microsoft/deberta-base"  # ğŸ”¥ å�‡ç´šåˆ°v3ç‰ˆæœ¬
    max_len       = 512
    batch_size    = 8
    gradient_accum = 2
    epochs        = 5  # ğŸ”¥ å¢�åŠ è¨“ç·´è¼ªæ•¸
    lr            = 2e-5  # ğŸ”¥ èª¿æ•´å­¸ç¿’ç�‡
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

    # ğŸ”¥ æ–°å¢�å„ªåŒ–å�ƒæ•¸
    use_focal_loss = True
    focal_alpha = 0.25
    focal_gamma = 2.0
    use_dice_loss = True
    dice_weight = 0.3
    dropout_rate = 0.2
    hidden_dropout = 0.1

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

# ğŸ”¥ æ”¹é€²çš„æ–‡æœ¬é �è™•ç�†
def preprocess_text(text: str) -> str:
    """é �è™•ç�†æ–‡æœ¬ä»¥æ”¹å–„tokenization"""
    if not isinstance(text, str):
        return ""
    
    # è¦�ç¯„åŒ–ç©ºæ ¼
    text = re.sub(r'\s+', ' ', text)
    # ç§»é™¤å¤šé¤˜çš„æ¨™é»�
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
    # è¦�ç¯„åŒ–ç ´æŠ˜è™Ÿ
    text = re.sub(r'[-â€“â€”]+', '-', text)
    
    return text.strip()

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

train = apply_annotation_fixes(train)

# ğŸ”¥ æ–‡æœ¬é �è™•ç�†
train['pn_history'] = train['pn_history'].apply(preprocess_text)
train['feature_text'] = train['feature_text'].apply(preprocess_text)

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

tok = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)

# ğŸ”¥ æ”¹é€²çš„æ¨™ç±¤å‰µå»ºå‡½æ•¸
def create_char_targets(text: str, spans: List[str]) -> np.ndarray:
    """å‰µå»ºå­—ç¬¦ç´šåˆ¥çš„ç›®æ¨™æ¨™ç±¤ - æ”¹é€²ç‰ˆæœ¬"""
    targets = np.zeros(len(text), dtype=np.int8)
    for span in spans:
        if not span:
            continue
        for loc in span.split(";"):
            loc = loc.strip()
            if not loc:
                continue
            try:
                start, end = map(int, loc.split())
                if start >= len(text):
                    continue
                end = min(end, len(text))
                if start < end:
                    targets[start:end] = 1
            except (ValueError, IndexError):
                continue
    return targets

# ğŸ”¥ æ”¹é€²çš„ç·¨ç¢¼å‡½æ•¸
def encode_example(note: str, feature: str, targets: np.ndarray | None):
    """ç·¨ç¢¼æ¨£æœ¬ä¸¦å‰µå»ºtokenç´šåˆ¥æ¨™ç±¤ - æ”¹é€²ç‰ˆæœ¬"""
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

            if s < len(targets) and e <= len(targets):
                # ğŸ”¥ æ”¹é€²ï¼šä½¿ç”¨é‡�ç–Šæ¯”ä¾‹ä¾†æ±ºå®šæ¨™ç±¤
                overlap = targets[s:e].sum()
                span_len = e - s
                if overlap > 0:
                    # å¦‚æ�œé‡�ç–Šè¶…é��50%ï¼Œæ¨™è¨˜ç‚ºæ­£ä¾‹
                    overlap_ratio = overlap / span_len
                    labels[idx] = min(1.0, overlap_ratio * 2)  # å¢�å¼·ä¿¡è™Ÿ

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

# ğŸ”¥ æ”¹é€²çš„spané‡�å»ºé‚�è¼¯
def reconstruct_spans_from_char_probs(char_prob: np.ndarray, threshold: float = 0.5, 
                                     min_span_len: int = 2, max_gap: int = 3) -> List[str]:
    """å¾�å­—ç¬¦æ¦‚ç�‡é‡�å»ºspans - æ”¹é€²ç‰ˆæœ¬"""
    spans = []
    start = None
    
    # ğŸ”¥ å…ˆé€²è¡Œå¹³æ»‘è™•ç�†
    smoothed_prob = np.copy(char_prob)
    for i in range(1, len(char_prob) - 1):
        if char_prob[i-1] > threshold and char_prob[i+1] > threshold:
            smoothed_prob[i] = max(smoothed_prob[i], (char_prob[i-1] + char_prob[i+1]) / 2)

    for idx, prob in enumerate(smoothed_prob):
        if prob >= threshold and start is None:
            start = idx
        elif prob < threshold and start is not None:
            # æª¢æŸ¥å°�é–“éš™
            gap_end = idx
            while gap_end < len(smoothed_prob) and gap_end - idx < max_gap:
                if smoothed_prob[gap_end] >= threshold:
                    idx = gap_end - 1  # è·³é��é–“éš™
                    break
                gap_end += 1
            else:
                # çœŸæ­£çš„çµ�æ�Ÿ
                if idx - start >= min_span_len:
                    spans.append(f"{start} {idx}")
                start = None
    
    # è™•ç�†åˆ°æœ«å°¾çš„æƒ…æ³�
    if start is not None and len(char_prob) - start >= min_span_len:
        spans.append(f"{start} {len(char_prob)}")

    return spans

# ğŸ”¥ æ”¹é€²çš„é–¾å€¼å„ªåŒ–
def find_optimal_threshold(pred_logits, val_idx, train_df):
    """å°‹æ‰¾æœ€ä½³é–¾å€¼ - æ”¹é€²ç‰ˆæœ¬"""
    LOGGER.info("\n=== å°‹æ‰¾æœ€ä½³é–¾å€¼ ===")

    # ğŸ”¥ ä½¿ç”¨æ›´æ™ºèƒ½çš„é–¾å€¼æ�œç´¢ç­–ç•¥
    base_thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_th = 0.5
    
    # ç¬¬ä¸€è¼ªç²—æ�œç´¢
    for th in base_thresholds:
        f1 = evaluate_threshold(pred_logits, val_idx, train_df, th)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    # ç¬¬äºŒè¼ªç´°æ�œç´¢
    fine_thresholds = np.arange(max(0.01, best_th - 0.1), min(0.99, best_th + 0.1), 0.01)
    for th in fine_thresholds:
        f1 = evaluate_threshold(pred_logits, val_idx, train_df, th)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    LOGGER.info(f"Best threshold: {best_th:.3f} (F1={best_f1:.4f})")
    return best_th

def evaluate_threshold(pred_logits, val_idx, train_df, threshold):
    """è©•ä¼°ç‰¹å®šé–¾å€¼çš„F1åˆ†æ•¸"""
    fold_preds = []
    fold_gts = []

    for i, row in enumerate(train_df.iloc[val_idx].itertuples()):
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

        pred_spans = reconstruct_spans_from_char_probs(char_prob, threshold)
        pred_span = ";".join(pred_spans)

        fold_preds.append(pred_span)
        fold_gts.append(";".join(row.location_list))

    return compute_micro_f1(pd.DataFrame({"ground": fold_gts, "pred": fold_preds}))


# -------------- 6. è‡ªè¨‚æ¨¡å�‹ -------------------------------
class FocalLoss(nn.Module):
    """ğŸ”¥ Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    """ğŸ”¥ Dice Loss for better overlap optimization"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice

class CustomModel(nn.Module):
    """ğŸ”¥ è‡ªè¨‚æ¨¡å�‹æ�¶æ§‹ - å¤§å¹…æ”¹é€²ç‰ˆæœ¬"""
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(CFG.model_name)
        
        if CFG.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        hidden_size = self.backbone.config.hidden_size
        
        # ğŸ”¥ å¤šå±¤ç‰¹å¾µæ��å�–
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(CFG.hidden_dropout)
            ) for _ in range(3)
        ])
        
        # ğŸ”¥ å¤šé ­æ³¨æ„�åŠ›æ©Ÿåˆ¶
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=CFG.hidden_dropout,
            batch_first=True
        )
        
        # ğŸ”¥ æ®˜å·®é€£æ�¥å’Œå±¤æ­£è¦�åŒ–
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        
        # ğŸ”¥ è¨ˆç®—æ‹¼æ�¥å¾Œçš„ç‰¹å¾µç¶­åº¦
        combined_feature_size = hidden_size + len(self.feature_extractors) * (hidden_size // 2)
        self.layer_norm2 = nn.LayerNorm(combined_feature_size)
        
        # ğŸ”¥ æ”¹é€²çš„åˆ†é¡�é ­
        self.classifier = nn.Sequential(
            nn.Dropout(CFG.dropout_rate),
            nn.Linear(combined_feature_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(CFG.dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # ğŸ”¥ æ��å¤±å‡½æ•¸
        if CFG.use_focal_loss:
            self.focal_loss = FocalLoss(CFG.focal_alpha, CFG.focal_gamma)
        if CFG.use_dice_loss:
            self.dice_loss = DiceLoss()

    def forward(self, **batch):
        labels = batch.pop("labels", None)
        
        # åŸºç¤�ç·¨ç¢¼
        backbone_out = self.backbone(**batch)
        hidden_states = backbone_out.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # ğŸ”¥ å¤šé ­è‡ªæ³¨æ„�åŠ›
        attn_out, _ = self.multihead_attn(hidden_states, hidden_states, hidden_states)
        hidden_states = self.layer_norm1(hidden_states + attn_out)  # æ®˜å·®é€£æ�¥
        
        # ğŸ”¥ å¤šå±¤ç‰¹å¾µæ��å�–
        extracted_features = []
        for extractor in self.feature_extractors:
            extracted_features.append(extractor(hidden_states))
        
        # ğŸ”¥ ç‰¹å¾µè��å�ˆ
        combined_features = torch.cat([hidden_states] + extracted_features, dim=-1)
        combined_features = self.layer_norm2(combined_features)
        
        # åˆ†é¡�
        logits = self.classifier(combined_features).squeeze(-1)

        loss = None
        if labels is not None:
            # ğŸ”¥ æ··å�ˆæ��å¤±å‡½æ•¸
            total_loss = 0
            loss_count = 0
            
            if CFG.use_focal_loss:
                focal_loss_val = self.focal_loss(logits, labels)
                total_loss += focal_loss_val
                loss_count += 1
            
            if CFG.use_dice_loss:
                dice_loss_val = self.dice_loss(logits, labels)
                total_loss += CFG.dice_weight * dice_loss_val
                loss_count += CFG.dice_weight
            
            if loss_count == 0:  # fallback
                total_loss = F.binary_cross_entropy_with_logits(logits, labels)
            else:
                total_loss = total_loss / loss_count

            loss = total_loss

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

    # ğŸ”¥ ä½¿ç”¨CustomModel
    model = CustomModel().to(CFG.device)
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
                    # ğŸ”¥ æ¢¯åº¦è£�å‰ª
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': float(best_th),
                'best_f1': float(best_f1)
            }, best_path)

    LOGGER.info(f"Fold {fold} best F1={best_f1:.4f}")

    del model, optimizer, scheduler
    cleanup_memory()



# -------------- 8. æ•´é«” OOF åˆ†æ•¸ ------------------------
overall_f1 = compute_micro_f1(pd.DataFrame({"ground": oof_gts, "pred": oof_preds}))
LOGGER.info(f"\n========== CV micro-F1: {overall_f1:.4f} ==========")

cleanup_memory()

if CFG.debug_mode:
    LOGGER.info("\n=== èª¿è©¦ç¸½çµ� ===")
    LOGGER.info(f"Training completed with CV F1: {overall_f1:.4f}")
    if overall_f1 > 0:
        LOGGER.info("âœ… è¨“ç·´æˆ�åŠŸï¼�")
    else:
        LOGGER.info("âš ï¸�  ä»�éœ€è¦�é€²ä¸€æ­¥èª¿è©¦")

