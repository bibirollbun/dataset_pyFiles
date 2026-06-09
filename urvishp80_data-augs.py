!pip install -qq pyspellchecker nlpaug sentence-transformers


from sklearn.preprocessing import LabelEncoder

import os
import re
import gc
import json
import math
import random
import warnings
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

# HF imports (translation + paraphrase)
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, pipeline,
    AutoModelForSequenceClassification
)

# Sentence-BERT for similarity
from sentence_transformers import SentenceTransformer, util
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

# Optional: char-level typo augmenter (we'll wrap it carefully to avoid placeholders)
import nlpaug.augmenter.char as nac


# ----------------
# CONFIG
# ----------------
class CONFIG:
    # Data
    TRAIN_CSV = "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
    SAVE_CSV  = "train_augmented.csv"

    # Teacher (your 65-class model + tokenizer)
    TEACHER_BASE = "Qwen/Qwen2.5-Math-1.5B"
    TEACHER_NUM_LABELS = 65
    # Set one of the following:
    # 1) A directory with Peft LoRA weights already merged (preferred), OR
    # 2) peft adapter dir + merge at load time.
    # If you have a pure HF folder with the fine-tuned head, set TEACHER_FINETUNED_DIR.
    TEACHER_FINETUNED_DIR = "/kaggle/working/qwen_teacher_finetuned"  # put your finetuned checkpoint dir here

    # Label encoder classes path (np.save of le.classes_)
    LABEL_CLASSES_NPY = "/kaggle/working/label_classes.npy"  # optional; not required if teacher outputs same order

    # Rare class target
    TARGET_PER_CLASS = 200
    MAX_AUG_PER_SOURCE = 200  # cap per original row to keep diversity

    # Filtering thresholds
    SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SIM_THRESH_TYPO = 0.90
    SIM_THRESH_PARA = 0.85
    SIM_THRESH_BT   = 0.85

    TEACHER_MIN_CONF = 0.00        # absolute min prob of top-1 on augmented
    TEACHER_MAX_DROP = 0.00        # don't accept if prob(top1_aug) < prob(top1_orig) - drop
    REQUIRE_SAME_TOP1 = True        # enforce teacher top-1 unchanged

    # Translation models (MarianMT)
    EN_FR = ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en")
    EN_ES = ("Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-es-en")

    # Monolingual paraphrase model (pick one you have access to)
    # Examples: "eugenesiow/bart-paraphrase", "Vamsi/T5_Paraphrase_Paws"
    PARAPHRASE_MODEL = "eugenesiow/bart-paraphrase"

    # Typos
    TYPO_CHAR_P = 0.15
    TYPO_WORD_P = 0.20
    MAX_GEN_PER_METHOD = 25  # candidates per method per source

    # Repro
    SEED = 124

    # Batching
    BATCH_SIZE_BT = 16
    BATCH_SIZE_TEACHER = 64
    BATCH_SIZE_SBERT = 128

    # TARGETS_WHITELIST = {
    # "True_Misconception:Longer_is_bigger",
    # "True_Misconception:Adding_across",
    # "True_Misconception:Whole_numbers_larger",
    # "False_Misconception:FlipChange",
    # "False_Misconception:Division",
    # "True_Misconception:Additive",
    # "False_Misconception:Longer_is_bigger",
    # "False_Misconception:Ignores_zeroes",
    # "False_Misconception:Base_rate",
    # "False_Misconception:Inverse_operation",
    # "False_Misconception:Certainty",
    # "True_Misconception:Shorter_is_bigger",
    # "True_Misconception:Firstterm",
    # "True_Misconception:SwapDividend",
    # "True_Misconception:Incomplete",
    # "True_Misconception:Wrong_term",
    # "True_Misconception:Mult",
    # "True_Misconception:WNB",
    # "False_Misconception:Incorrect_equivalent_fraction_addition",
    # "False_Misconception:Wrong_Operation",
    # "True_Misconception:Duplication",
    # "True_Misconception:Wrong_fraction",
    # "False_Misconception:Shorter_is_bigger",
    # "True_Misconception:Inversion",
    # "True_Misconception:Division",
    # "True_Misconception:FlipChange",
    # "True_Misconception:Denominator-only_change",
    # "True_Misconception:Definition",
    # "True_Misconception:Multiplying_by_4",
    # "True_Misconception:Subtraction",
    # "True_Misconception:Incorrect_equivalent_fraction_addition",
    # "True_Misconception:Positive",
    # "True_Misconception:Base_rate",
    # "True_Misconception:Not_variable",
    


random.seed(CONFIG.SEED)
np.random.seed(CONFIG.SEED)
torch.manual_seed(CONFIG.SEED)


# le = LabelEncoder()
# train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
# train.Misconception = train.Misconception.fillna('NA')
# train['target'] = train.Category+":"+train.Misconception
# train['label'] = le.fit_transform(train['target'])
# n_classes = len(le.classes_)
# # train["soft_labels"] = list(make_soft_labels(train["label"].values, n_classes, eps=0.1))
# print(f"Train shape: {train.shape} with {n_classes} target classes")
# train.head()


# ----------------
# Utilities
# ----------------
def normalize_unicode_fractions(text: str) -> str:
    mapping = {
        "¼": "1/4", "½": "1/2", "¾": "3/4",
        "⅐": "1/7", "⅑": "1/9", "⅒": "1/10",
        "⅓": "1/3", "⅔": "2/3",
        "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
        "⅙": "1/6", "⅚": "5/6",
        "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


OPS_WORDS = [
    "plus", "add", "sum", "minus", "subtract", "difference",
    "times", "multiply", "multiplied by", "product",
    "divide", "divided by", "over", "quotient",
    "equals", "equal to", "equivalent", "equivalent to"
]
OPS_WORDS_RE = re.compile(r"\b(" + "|".join([re.escape(w) for w in sorted(OPS_WORDS, key=len, reverse=True)]) + r")\b", re.I)

# Symbols to protect
OPS_SYMS = ["×", "x", "÷", "/", "+", "-", "−", "=", "^", "√", "≤", "≥", "≠", "•", "∙", "·", "π", "%"]
OPS_SYMS_RE = re.compile("|".join([re.escape(s) for s in OPS_SYMS]))

# Fractions (ASCII) first, then numbers (to avoid double-protecting numerator/denominator)
FRAC_RE = re.compile(r"\b\d+\s*/\s*\d+\b")
MIXED_FRAC_RE = re.compile(r"\b\d+\s+\d+\s*/\s*\d+\b")  # e.g., 3 1/2
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

PLACEHOLDER_TOKEN = re.compile(r"<(FRAC|MFRAC|NUM|OPW|OPS)_(\d+)>")

def protect_math(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace math spans with placeholders. Return protected_text, mapping.
    Order: unicode→ascii fractions; mixed fractions; a/b fractions; numbers; operator symbols; operator words.
    """
    original = text
    text = normalize_unicode_fractions(text)

    mapping = {}
    counter = {"MFRAC": 0, "FRAC": 0, "NUM": 0, "OPS": 0, "OPW": 0}

    def _sub_and_store(regex, tag, s):
        def repl(m):
            key = f"<{tag}_{counter[tag]}>"
            counter[tag] += 1
            mapping[key] = m.group(0)
            return key
        return regex.sub(repl, s)

    # Mixed fractions (e.g., "3 1/2")
    text = _sub_and_store(MIXED_FRAC_RE, "MFRAC", text)
    # Simple fractions a/b
    text = _sub_and_store(FRAC_RE, "FRAC", text)
    # Numbers (avoid ones inside placeholders already)
    # We'll protect numbers that are not inside placeholders by splitting
    def protect_numbers_outside_placeholders(s):
        parts = PLACEHOLDER_TOKEN.split(s)
        # split keeps separators; reconstruct while protecting only non-placeholder chunks
        out = []
        i = 0
        while i < len(parts):
            if i + 2 < len(parts) and parts[i] == "" and parts[i+1] in {"FRAC","MFRAC","NUM","OPS","OPW"}:
                # this is a placeholder triple split: "", TAG, idx, rest...
                tag = parts[i+1]; idx = parts[i+2]
                ph = f"<{tag}_{idx}>"
                rest = parts[i+3] if (i+3) < len(parts) else ""
                out.append(ph)
                i += 4
                if rest:
                    # rest is plain string to be processed
                    rest = _sub_and_store(NUM_RE, "NUM", rest)
                    out.append(rest)
            else:
                chunk = parts[i]
                chunk = _sub_and_store(NUM_RE, "NUM", chunk)
                out.append(chunk)
                i += 1
        return "".join(out)
    text = protect_numbers_outside_placeholders(text)

    # Operator symbols
    def _ops_sym_repl(m):
        key = f"<OPS_{counter['OPS']}>"
        counter['OPS'] += 1
        mapping[key] = m.group(0)
        return key
    text = OPS_SYMS_RE.sub(_ops_sym_repl, text)

    # Operator words (case-insensitive)
    def _ops_word_repl(m):
        key = f"<OPW_{counter['OPW']}>"
        counter['OPW'] += 1
        mapping[key] = m.group(0)
        return key
    text = OPS_WORDS_RE.sub(_ops_word_repl, text)

    return text, mapping


def restore_math(text: str, mapping: Dict[str, str]) -> str:
    # Ensure every placeholder from mapping appears the same count in text
    for ph in mapping.keys():
        if text.count(ph) != 1:
            # if placeholder missing or duplicated, reject by returning empty marker
            return ""
    for ph, val in mapping.items():
        text = text.replace(ph, val)
    return text

def same_placeholder_signature(a: str, b: str) -> bool:
    # Verify same kinds and counts of placeholders exist in both strings
    def sig(s):
        counts = {}
        for m in PLACEHOLDER_TOKEN.finditer(s):
            tag = m.group(1)
            counts[tag] = counts.get(tag, 0) + 1
        return counts
    return sig(a) == sig(b)


def fast_normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("’","'").replace("`","'").replace("–","-").replace("—","-")
    s = re.sub(r"\s+", " ", s)
    return s

# ----------------
# Typos (safe)
# ----------------
typo_aug = nac.KeyboardAug(aug_char_p=CONFIG.TYPO_CHAR_P, aug_word_p=CONFIG.TYPO_WORD_P, min_char=4)

def add_typos_safe(text: str) -> str:
    # Split around placeholders; only augment non-placeholder spans
    pieces = re.split(r"(<[^>]+>)", text)
    out = []
    for p in pieces:
        if p.startswith("<") and p.endswith(">"):
            out.append(p)
        else:
            # avoid messing with numbers accidentally: keep digits unchanged
            # simple heuristic: skip augmentation if chunk has >40% digits
            if p and (sum(ch.isdigit() for ch in p) / max(1,len(p)) < 0.4):
                try:
                    out.append(typo_aug.augment(p))
                except Exception:
                    out.append(p)
            else:
                out.append(p)
    return "".join(out)


# ----------------
# Translation + Paraphrase loaders
# ----------------
def load_translation_pipelines():
    device = 0 if torch.cuda.is_available() else -1
    # we rely on device_map="auto" inside pipeline to spread across GPUs if available
    en_fr_tok = AutoTokenizer.from_pretrained(CONFIG.EN_FR[0])
    en_fr_mod = AutoModelForSeq2SeqLM.from_pretrained(CONFIG.EN_FR[0])
    fr_en_tok = AutoTokenizer.from_pretrained(CONFIG.EN_FR[1])
    fr_en_mod = AutoModelForSeq2SeqLM.from_pretrained(CONFIG.EN_FR[1])

    en_es_tok = AutoTokenizer.from_pretrained(CONFIG.EN_ES[0])
    en_es_mod = AutoModelForSeq2SeqLM.from_pretrained(CONFIG.EN_ES[0])
    es_en_tok = AutoTokenizer.from_pretrained(CONFIG.EN_ES[1])
    es_en_mod = AutoModelForSeq2SeqLM.from_pretrained(CONFIG.EN_ES[1])

    pipe_en_fr = pipeline("translation", model=en_fr_mod, tokenizer=en_fr_tok, device=0)
    pipe_fr_en = pipeline("translation", model=fr_en_mod, tokenizer=fr_en_tok, device=0)
    pipe_en_es = pipeline("translation", model=en_es_mod, tokenizer=en_es_tok, device=1)
    pipe_es_en = pipeline("translation", model=es_en_mod, tokenizer=es_en_tok, device=1)
    return (pipe_en_fr, pipe_fr_en, pipe_en_es, pipe_es_en)


def load_paraphrase_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(CONFIG.PARAPHRASE_MODEL)
    mod = AutoModelForSeq2SeqLM.from_pretrained(CONFIG.PARAPHRASE_MODEL)
    para = pipeline("text2text-generation", model=mod, tokenizer=tok, device=device)
    return para


# ----------------
# SBERT + Teacher
# ----------------
def load_sbert():
    return SentenceTransformer(CONFIG.SBERT_MODEL)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def load_teacher():
    lora_path = f"/kaggle/input/map-qwen-1-5b-tpu-train-fold-3-5/qwen_15b_math_4.pth"
    CONFIG.TEACHER_BASE = "/kaggle/input/qwenqwen2.5-math-1.5b/transformers/default/1/models/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.TEACHER_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG.TEACHER_BASE,
        num_labels=CONFIG.TEACHER_NUM_LABELS,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    # LoRa configuration
    # better way would be to save adapter.json in training, maybe in next version
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias='none',
        inference_mode=True,
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    
    
    # Get peft
    model = get_peft_model(model, lora_config)
    # Load weights
    model.load_state_dict(torch.load(lora_path), strict=False)
    # model = PeftModel.from_pretrained(model, lora_path)
    model = torch.nn.DataParallel(model)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def teacher_predict_proba(texts: List[str], tknzr, model, batch_size=32) -> np.ndarray:
    model.cuda()
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tknzr(batch, padding=True, truncation=True, max_length=312, return_tensors="pt")
        if torch.cuda.is_available():
            for k in enc:
                enc[k] = enc[k].cuda()
        out = model(**enc).logits.float()
        p = F.softmax(out, dim=-1).cpu().numpy()
        probs.append(p)
    return np.vstack(probs)


# ----------------
# Paraphrase/BT helpers (protected)
# ----------------
def paraphrase_mono_protected(text: str, para_pipe, num_return=2) -> List[str]:
    prot, mapping = protect_math(text)
    outs = para_pipe(prot, num_return_sequences=num_return, num_beams=max(4, num_return), do_sample=False, max_new_tokens=128)
    cands = []
    for o in outs:
        cand = o["generated_text"]
        rest = restore_math(cand, mapping)
        if rest:
            cands.append(rest)
    return cands


def back_translate_protected(text: str, en_x, x_en, num_return=2, max_new_tokens=128) -> List[str]:
    prot, mapping = protect_math(text)
    # EN -> X
    inters = en_x(prot, num_return_sequences=num_return, num_beams=max(4, num_return), do_sample=False, max_new_tokens=max_new_tokens)
    cands = []
    for it in inters:
        mid = it["translation_text"]
        # X -> EN
        finals = x_en(mid, num_return_sequences=1, num_beams=4, do_sample=False, max_new_tokens=max_new_tokens)
        for f in finals:
            cand = f["translation_text"]
            rest = restore_math(cand, mapping)
            if rest:
                cands.append(rest)
    return cands


# ----------------
# Validation filters
# ----------------
def sbert_filter(orig_emb: np.ndarray, cands: List[str], sbert, thresh: float) -> List[Tuple[str, float]]:
    if not cands:
        return []
    emb_cands = sbert.encode(cands, convert_to_numpy=True, show_progress_bar=False)
    outs = []
    for text, e in zip(cands, emb_cands):
        sim = cosine_sim(orig_emb, e)
        if sim >= thresh:
            outs.append((text, sim))
    return outs


def teacher_filter(orig_text_formatted: str, cand_text_formatted: str, tknzr, model,
                   y_true: int, p_orig: float) -> Tuple[bool, float, int]:
    # returns (keep, p_aug, y_pred_aug)
    probs = teacher_predict_proba([cand_text_formatted], tknzr, model, batch_size=1)[0]
    y_pred = int(probs.argmax())
    p_aug = float(probs[y_pred])
    # Enforce top-1 unchanged?
    # if CONFIG.REQUIRE_SAME_TOP1 and y_pred != y_true:
    #     return False, p_aug, y_pred
    # Confidence floor
    # if p_aug < CONFIG.TEACHER_MIN_CONF:
    #     return False, p_aug, y_pred
    # Don't allow large confidence collapse
    # if p_orig - p_aug > CONFIG.TEACHER_MAX_DROP:
    #     return False, p_aug, y_pred
    return True, p_aug, y_pred


# ----------------
# Driver
# ----------------
def format_input(row: pd.Series) -> str:
    x = "This is Correct answer." if row["is_correct"] == 1 else "This is Incorrect answer."
    return (
        f"• Question: {row['QuestionText']}\n"
        f"• Answer: {row['MC_Answer']}\n"
        f"• Correctness: {x}\n"
        f"• Student Explanation: {row['StudentExplanation']}"
    )

def build_teacher_text(row: pd.Series, student_expl: str) -> str:
    x = "This is Correct answer." if row["is_correct"] == 1 else "This is Incorrect answer."
    return (
        f"• Question: {row['QuestionText']}\n"
        f"• Answer: {row['MC_Answer']}\n"
        f"• Correctness: {x}\n"
        f"• Student Explanation: {student_expl}"
    )


def generate_aug_for_row(row: pd.Series,
                         sbert, sbert_orig_emb: np.ndarray,
                         tknzr, teacher_model,
                         teacher_y_true: int, teacher_p_orig: float,
                         pipes: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Try several methods in order; run filters; return list of accepted aug dicts.
    """
    orig_SE = row["StudentExplanation"]

    accepted = []
    seen_norm = set()

    def consider(cand_text: str, aug_type: str, sim_thresh: float):
        cand_text = cand_text.strip()
        if not cand_text or fast_normalize(cand_text) == fast_normalize(orig_SE):
            return
        # Quick placeholder signature check
        prot_o, _map_o = protect_math(orig_SE)
        prot_c, _map_c = protect_math(cand_text)
        if not same_placeholder_signature(prot_o, prot_c):
            return
        # SBERT filter
        sims = sbert_filter(sbert_orig_emb, [cand_text], sbert, sim_thresh)
        if not sims:
            return
        _, sim = sims[0]
        # Teacher filter
        text_formatted = build_teacher_text(row, cand_text)
        # keep, p_aug, y_pred = teacher_filter(
        #     build_teacher_text(row, orig_SE),
        #     text_formatted,
        #     tknzr, teacher_model,
        #     teacher_y_true, teacher_p_orig
        # )
        # if not keep:
        #     return
        norm = fast_normalize(cand_text)
        if norm in seen_norm:
            return
        seen_norm.add(norm)
        accepted.append({
            "StudentExplanation": cand_text,
            "aug_type": aug_type,
            "sbert_sim": sim,
            # "teacher_p_aug": p_aug
        })

    # 1) Typos (at most MAX_GEN_PER_METHOD)
    prot, mapping = protect_math(orig_SE)
    for _ in range(CONFIG.MAX_GEN_PER_METHOD):
        try:
            ty = add_typos_safe(prot)
            restored = restore_math(ty, mapping)
            if restored:
                consider(restored, "typo", CONFIG.SIM_THRESH_TYPO)
        except Exception:
            pass

    # 2) Monolingual paraphrase
    if pipes.get("para"):
        try:
            paras = paraphrase_mono_protected(orig_SE, pipes["para"], num_return=CONFIG.MAX_GEN_PER_METHOD)
            for p in paras:
                consider(p, "mono_para", CONFIG.SIM_THRESH_PARA)
        except Exception:
            pass

    # 3) Back-translation EN↔FR
    if pipes.get("en_fr") and pipes.get("fr_en"):
        try:
            cands = back_translate_protected(orig_SE, pipes["en_fr"], pipes["fr_en"], num_return=CONFIG.MAX_GEN_PER_METHOD)
            for c in cands:
                consider(c, "bt_fr", CONFIG.SIM_THRESH_BT)
        except Exception:
            pass

    # 4) Back-translation EN↔ES
    if pipes.get("en_es") and pipes.get("es_en"):
        try:
            cands = back_translate_protected(orig_SE, pipes["en_es"], pipes["es_en"], num_return=CONFIG.MAX_GEN_PER_METHOD)
            for c in cands:
                consider(c, "bt_es", CONFIG.SIM_THRESH_BT)
        except Exception:
            pass

    return accepted


import joblib


def main():
    le = LabelEncoder()
    warnings.filterwarnings("ignore")
    print("Loading train...")
    df = pd.read_csv(CONFIG.TRAIN_CSV)
    df["Misconception"] = df["Misconception"].fillna("NA")
    df["target"] = df["Category"] + ":" + df["Misconception"]

    # Assuming you already computed is_correct in your training code.
    # If not, add your mapping here. For augmentation we assume it's present.
    df['label'] = le.fit_transform(df['target'])
    n_classes = len(le.classes_)
    print(f"Train shape: {df.shape} with {n_classes} target classes")

    idx = df.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
    correct = df.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c',ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])
    correct = correct[['QuestionId','MC_Answer']]
    correct['is_correct'] = 1
    
    df = df.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    df.is_correct = df.is_correct.fillna(0)
    

    # Build label encoder matching teacher ordering if needed
    if "label" not in df.columns:
        # Fit a temporary label encoder compatible with your training
        tgts = df["target"].astype(str)
        classes = np.unique(tgts)
        classes_sorted = np.sort(classes)
        class2id = {c:i for i,c in enumerate(classes_sorted)}
        df["label"] = df["target"].map(class2id)
        np.save(CONFIG.LABEL_CLASSES_NPY, classes_sorted)
    else:
        # Save classes for reference
        classes_sorted = np.array(sorted(df["target"].unique()))
        np.save(CONFIG.LABEL_CLASSES_NPY, classes_sorted)

    # counts = df["target"].value_counts()
    # rare_labels = counts[counts < CONFIG.TARGET_PER_CLASS].index.tolist()
    # print(f"Rare labels (< {CONFIG.TARGET_PER_CLASS}): {len(rare_labels)}")

    counts = df["target"].value_counts()
    if hasattr(CONFIG, "TARGETS_WHITELIST") and CONFIG.TARGETS_WHITELIST:
        rare_labels = list(CONFIG.TARGETS_WHITELIST)
    else:
        rare_labels = counts[counts < CONFIG.TARGET_PER_CLASS].index.tolist()
    print(f"Augmenting labels: {len(rare_labels)}")

    # Load SBERT and teacher
    print("Loading SBERT...")
    sbert = load_sbert()
    print("Loading teacher...")
    tknzr, teacher = load_teacher()

    # Precompute teacher probs for originals (for confidence-drop filter)
    print("Scoring originals with teacher...")
    df["text"] = df.apply(format_input, axis=1)
    # orig_probs = []
    # for i in tqdm(range(0, len(df), CONFIG.BATCH_SIZE_TEACHER)):
    #     batch = df["text"].iloc[i:i+CONFIG.BATCH_SIZE_TEACHER].tolist()
    #     p = teacher_predict_proba(batch, tknzr, teacher, batch_size=CONFIG.BATCH_SIZE_TEACHER)
    #     orig_probs.append(p)
    # orig_probs = np.vstack(orig_probs)
    # df["teacher_top1"] = orig_probs.argmax(axis=1)
    # df["teacher_p_top1"] = orig_probs.max(axis=1)

    # Translation/paraphrase pipes
    print("Loading translation/paraphrase models...")
    en_fr, fr_en, en_es, es_en = load_translation_pipelines()
    para = load_paraphrase_pipeline()
    pipes = {
        "en_fr": en_fr, "fr_en": fr_en,
        "en_es": en_es, "es_en": es_en,
        "para": para
    }

    # Build quotas per rare class
    target_counts = counts.copy()
    # quotas = {lbl: CONFIG.TARGET_PER_CLASS - int(target_counts[lbl]) for lbl in rare_labels}
    quotas = {lbl: max(0, CONFIG.TARGET_PER_CLASS - int(counts.get(lbl, 0))) for lbl in rare_labels}
    by_label = {lbl: df[df["target"] == lbl].copy() for lbl in rare_labels}

    total_needed = sum([max(0,q) for q in quotas.values()])
    print(f"Total augmented examples target: {total_needed}")

    # Index by target for fast sampling
    # by_label = {lbl: df[df["target"] == lbl].copy() for lbl in rare_labels}

    # Prepare output rows
    aug_rows = []

    # Iterate labels and grow to target
    for lbl in tqdm(rare_labels):
        need = quotas[lbl]
        if need <= 0:
            continue
        sub = by_label[lbl]
        if sub.empty:
            continue

        # Shuffle to diversify sources
        sub = sub.sample(frac=1.0, random_state=CONFIG.SEED).reset_index(drop=True)

        produced = 0
        # Precompute SBERT embeddings of originals (StudentExplanation only)
        sub_SE = sub["StudentExplanation"].astype(str).tolist()
        sub_emb = sbert.encode(sub_SE, convert_to_numpy=True, show_progress_bar=False)

        for idx, row in sub.iterrows():
            if produced >= need:
                break

            # teacher original stats
            y_true = int(row["label"])
            # p_orig = float(row["teacher_p_top1"])
            p_orig = 0.99
            orig_emb = sub_emb[idx]

            cands = generate_aug_for_row(row, sbert, orig_emb, tknzr, teacher, y_true, p_orig, pipes)
            # random.shuffle(cands)

            # Take up to MAX_AUG_PER_SOURCE but not beyond need
            # take = min(CONFIG.MAX_AUG_PER_SOURCE, need - produced, len(cands))
            for k in range(len(cands)):
                cand = cands[k]
                new_row = row.copy()
                new_row["StudentExplanation"] = cand["StudentExplanation"]
                new_row["text"] = build_teacher_text(row, cand["StudentExplanation"])
                new_row["aug_type"] = cand["aug_type"]
                new_row["sbert_sim"] = cand["sbert_sim"]
                # new_row["teacher_p_aug"] = cand["teacher_p_aug"]
                new_row["aug_weight"] = 0.8 if lbl.endswith(":NA") else 1.0  # example: slightly downweight NA
                aug_rows.append(new_row.to_dict())
                produced += 1

            if produced >= need:
                break

        print(f"Label {lbl}: produced {produced} / needed {need}")

    aug_df = pd.DataFrame(aug_rows)
    if aug_df.empty:
        print("No augmentations produced. Check thresholds or teacher/pipe loading.")
        return

    # Ensure label/target intact
    assert (aug_df["target"].value_counts() >= 0).all()

    aug_df.to_csv("only_aug_data.csv", index=False)
    # Concatenate
    out_df = pd.concat([df, aug_df], ignore_index=True)
    out_df.to_csv(CONFIG.SAVE_CSV, index=False)
    print(f"Saved augmented dataset: {CONFIG.SAVE_CSV}")
    print(f"Original size: {len(df):,} | Augmented size: {len(out_df):,} | Added: {len(aug_df):,}")

    # Optional: memory cleanup
    del aug_df, df
    gc.collect()


main()


# Label True_Misconception:Longer_is_bigger: produced 4 / needed 199
# Label True_Misconception:Adding_across: produced 0 / needed 199
# Label True_Misconception:Whole_numbers_larger: produced 3 / needed 199
# Label False_Misconception:FlipChange: produced 75 / needed 126
# Label False_Misconception:Division: produced 60 / needed 142
# Label True_Misconception:Additive: produced 55 / needed 162
# Label False_Misconception:Longer_is_bigger: produced 56 / needed 177
# Label False_Misconception:Ignores_zeroes: produced 59 / needed 177
# Label False_Misconception:Base_rate: produced 31 / needed 178
# Label False_Misconception:Inverse_operation: produced 35 / needed 179
# Label False_Misconception:Certainty: produced 29 / needed 182
# Label True_Misconception:Shorter_is_bigger: produced 54 / needed 183
# Label True_Misconception:Firstterm: produced 19 / needed 189
# Label True_Misconception:SwapDividend: produced 5 / needed 192
# Label True_Misconception:Incomplete: produced 19 / needed 192
# Label True_Misconception:Wrong_term: produced 20 / needed 192
# Label True_Misconception:Mult: produced 10 / needed 192
# Label True_Misconception:WNB: produced 2 / needed 192
# Label False_Misconception:Incorrect_equivalent_fraction_addition: produced 9 / needed 193
# Label False_Misconception:Wrong_Operation: produced 1 / needed 194
# Label True_Misconception:Duplication: produced 9 / needed 194
# Label True_Misconception:Wrong_fraction: produced 4 / needed 194
# Label False_Misconception:Shorter_is_bigger: produced 15 / needed 194
# Label True_Misconception:Inversion: produced 5 / needed 195
# Label True_Misconception:Division: produced 6 / needed 195
# Label True_Misconception:FlipChange: produced 7 / needed 196
# Label True_Misconception:Denominator-only_change: produced 2 / needed 196
# Label True_Misconception:Definition: produced 7 / needed 197
# Label True_Misconception:Multiplying_by_4: produced 8 / needed 197
# Label True_Misconception:Subtraction: produced 4 / needed 198
# Label True_Misconception:Incorrect_equivalent_fraction_addition: produced 3 / needed 198
# Label True_Misconception:Positive: produced 4 / needed 198
# Label True_Misconception:Base_rate: produced 0 / needed 199
# Label True_Misconception:Not_variable: produced 1 / needed 199
# Label True_Misconception:Whole_numbers_larger: produced 3 / needed 199

