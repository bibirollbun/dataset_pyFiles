#!/usr/bin/env python3
from __future__ import annotations
import os
import re
import math
import random
import logging
from typing import List, Tuple, Dict, Any
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb

# Optional heavy libs: import inside functions to avoid startup cost / allow fallback
import warnings
warnings.filterwarnings("ignore")

# -------------------- CONFIG --------------------
SEED = 42
TF_WORD_MAX = 8000
TF_CHAR_MAX = 4000
TF_PUNC_MAX = 50
TRANS_SVD_COMPONENTS = 128
BATCH_SIZE = 16
MAX_TRANSFORMER_LEN = 512

KAGGLE_TRAIN = "/kaggle/input/mercor-ai-detection/train.csv"
KAGGLE_TEST = "/kaggle/input/mercor-ai-detection/test.csv"
KAGGLE_SAMPLE = "/kaggle/input/mercor-ai-detection/sample_submission.csv"

# Minimal lexicons (kept from original, reduced duplication)
TOP_AI_FEATURES = [
    'study', 'known', 'branch', 'called', 'concerned', 'various', 'used',
    'discipline', 'scientific', 'subject', 'process', 'methods', 'applied',
    'work', 'practice', 'typically', 'science', 'particular'
]
TOP_HUMAN_FEATURES = [
    'like', 'explores', 'field', 'examines', 'topic', 'studies', 'text',
    'think', 'feel', 'believe', 'i think', 'i feel', 'in my opinion'
]
FORMAL_WORDS = {'therefore', 'however', 'moreover', 'furthermore', 'consequently'}
CONVERSATIONAL_MARKERS = {'like', 'you know', 'i mean', 'well', 'so', 'actually'}
DETERMINERS = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your'}
CONJUNCTIONS = {'and', 'or', 'but', 'so', 'because', 'although', 'while'}
AUX_VERBS = {'be', 'am', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'may'}
NOUN_LIKE_SUFFIXES = ('tion', 'ment', 'ity', 'ness', 'ship', 'ence', 'ance')
EMOTIVE_WORDS = {'love', 'hate', 'wonderful', 'terrible', 'amazing', 'awful', 'excited', 'disappointed'}
NEGATIVE_WORDS = {'not', "n't", 'no', 'never', 'none'}
POSITIVE_WORDS = {'yes', 'indeed', 'certainly', 'definitely', 'absolutely'}

# -------------------- SETUP --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("mercor_v4")

def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

set_seeds()

# -------------------- UTILITIES --------------------
def clean_text(s: Any) -> str:
    if pd.isnull(s):
        return ""
    t = str(s)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"<.*?>", " ", t)
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def count_syllables(word: str) -> int:
    w = word.lower()
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    if w[0] in vowels:
        count += 1
    for i in range(1, len(w)):
        if w[i] in vowels and w[i-1] not in vowels:
            count += 1
    if w.endswith("e"):
        count -= 1
    if count <= 0:
        count = 1
    return count

def get_readability_scores(texts: List[str]) -> np.ndarray:
    """Return (Flesch Reading Ease, Flesch-Kincaid Grade) per text (float32)."""
    feats = []
    for text in texts:
        text = text.strip()
        if not text:
            feats.append([0.0, 0.0]); continue
        # sentences: count punctuation-terminated fragments (avoid zero)
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        n_sent = max(1, len(sentences))
        words = re.findall(r'\w+', text)
        n_words = len(words)
        n_words = max(1, n_words)
        syllables = sum(count_syllables(w) for w in words)
        asl = n_words / n_sent
        asw = syllables / n_words
        flesch_ease = 206.835 - 1.015 * asl - 84.6 * asw
        grade_level = 0.39 * asl + 11.8 * asw - 15.59
        feats.append([float(flesch_ease), float(grade_level)])
    return np.array(feats, dtype=np.float32)

# -------------------- EMBEDDINGS & PERPLEXITY (SAFE) --------------------
def get_transformer_embeddings(texts: List[str], model_name: str = "microsoft/deberta-v3-base", batch_size: int = BATCH_SIZE, max_length: int = MAX_TRANSFORMER_LEN) -> np.ndarray:
    """
    Compute mean-pooled transformer embeddings for a list of texts.
    Returns numpy array (n_texts, hidden_size). If model loading fails, returns zeros.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state  # (B, L, H)
                mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                summed = (last_hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-9)
                mean_pooled = (summed / denom).cpu().numpy()
                embs.append(mean_pooled)
        embs = np.vstack(embs) if embs else np.zeros((len(texts), TRANS_SVD_COMPONENTS), dtype=np.float32)
        # cleanup
        try:
            model.cpu(); del model; del tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
        return embs.astype(np.float32)
    except Exception as e:
        logger.warning("Transformer embedding failure (%s). Returning zeros. - %s", model_name, e)
        return np.zeros((len(texts), TRANS_SVD_COMPONENTS), dtype=np.float32)

def get_perplexity(texts: List[str], model_name: str = "gpt2", batch_size: int = BATCH_SIZE, max_length: int = MAX_TRANSFORMER_LEN) -> np.ndarray:
    """
    Compute GPT-2 perplexities. If model is unavailable or memory errors occur, return a high constant vector.
    """
    fallback_val = 1e6  # big perplexity signals 'unknown / unpredictable' (or we could use nan)
    try:
        from transformers import GPT2TokenizerFast, GPT2LMHeadModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        # Ensure pad token exists so batching works
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        ppls = []
        loss_fct = None
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if not batch:
                continue
            # truncate tokens to max_length to avoid OOM
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
                logits = outputs.logits  # (B, L, V)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                # compute token-wise loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(input_ids.size(0), -1)
                mask = (shift_labels != tokenizer.pad_token_id).float()
                sum_loss = (loss * mask).sum(dim=1)
                n_tokens = mask.sum(dim=1).clamp(min=1.0)
                per_sample_loss = sum_loss / n_tokens
                batch_ppls = torch.exp(per_sample_loss).cpu().numpy()
                ppls.extend(batch_ppls.tolist())
        # cleanup
        try:
            model.cpu(); del model; del tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
        if not ppls:
            return np.full((len(texts), 1), fallback_val, dtype=np.float32)
        arr = np.array(ppls, dtype=np.float32).reshape(-1, 1)
        # if length mismatch (shouldn't) pad/fill
        if arr.shape[0] < len(texts):
            pad = np.full((len(texts) - arr.shape[0], 1), fallback_val, dtype=np.float32)
            arr = np.vstack([arr, pad])
        return arr
    except Exception as e:
        logger.warning("GPT-2 perplexity computation failed (%s). Using fallback perplexity=%s for all samples.", e, fallback_val)
        return np.full((len(texts), 1), fallback_val, dtype=np.float32)

# -------------------- LINGUISTIC FEATURES --------------------
def calculate_lexical_richness(texts: List[str]) -> np.ndarray:
    feats = []
    for text in texts:
        t_norm = (text or "").lower()
        words = [w for w in re.findall(r"\b[a-z0-9']+\b", t_norm)]
        total = max(1, len(words))
        counts = Counter(words)
        unique = len(counts)
        ttr = unique / total
        hapax = sum(1 for v in counts.values() if v == 1) / total
        simpson = 1.0 - sum((c/total)**2 for c in counts.values()) if counts else 0.0
        entropy = -sum((c/total) * math.log2(c/total) for c in counts.values()) if counts else 0.0
        avg_freq = total / unique if unique else total
        vocab_richness = (unique / total) * 100
        word_lengths = [len(w) for w in words] or [0]
        avg_word_length = float(np.mean(word_lengths))
        word_length_std = float(np.std(word_lengths)) if len(word_lengths) > 1 else 0.0
        feats.append([ttr, hapax, simpson, entropy, avg_freq, vocab_richness, avg_word_length, word_length_std])
    return np.array(feats, dtype=np.float32)

def compute_corpus_word_ranks(all_texts: List[str]) -> Dict[str, int]:
    freq = Counter()
    for txt in all_texts:
        words = [w for w in re.findall(r"\b[a-z0-9']+\b", (txt or "").lower())]
        freq.update(words)
    if not freq:
        return {}
    sorted_words = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return {w: idx for idx, (w, _) in enumerate(sorted_words, start=1)}

def rare_word_rank_features(texts: List[str], rank_map: Dict[str, int]) -> np.ndarray:
    if not rank_map:
        return np.zeros((len(texts), 3), dtype=np.float32)
    max_rank = max(rank_map.values())
    feats = []
    for txt in texts:
        words = [w for w in re.findall(r"\b[a-z0-9']+\b", (txt or "").lower())]
        if not words:
            feats.append([0.0, 0.0, 0.0]); continue
        ranks = [rank_map.get(w, max_rank) for w in words]
        mean_rank = float(np.mean(ranks) / max_rank)
        median_rank = float(np.median(ranks) / max_rank)
        thresh = 0.75 * max_rank
        rare_prop = sum(1 for r in ranks if r > thresh) / len(ranks)
        feats.append([mean_rank, median_rank, rare_prop])
    return np.array(feats, dtype=np.float32)

def repetition_features(texts: List[str], ngram_n: int = 3) -> np.ndarray:
    feats = []
    for txt in texts:
        words = [w for w in re.findall(r"\b[a-z0-9']+\b", (txt or "").lower())]
        total = len(words)
        if total == 0:
            feats.append([0.0, 0.0, 0.0]); continue
        wc = Counter(words)
        top_repeat_ratio = wc.most_common(1)[0][1] / total if wc else 0.0
        unique_word_ratio = len(wc) / total
        ngrams = [' '.join(words[i:i+ngram_n]) for i in range(max(0, total - ngram_n + 1))]
        if not ngrams:
            repeated_ngram_ratio = 0.0
        else:
            ngc = Counter(ngrams)
            repeated = sum(cnt for _, cnt in ngc.items() if cnt > 1)
            repeated_ngram_ratio = repeated / len(ngrams)
        feats.append([top_repeat_ratio, 1.0 - unique_word_ratio, repeated_ngram_ratio])
    return np.array(feats, dtype=np.float32)

def function_word_features(texts: List[str]) -> np.ndarray:
    feats = []
    for txt in texts:
        words = [w for w in re.findall(r"\b[a-zA-Z']+\b", (txt or ""))]
        total = max(1, len(words))
        det = sum(1 for w in words if w.lower() in DETERMINERS) / total
        conj = sum(1 for w in words if w.lower() in CONJUNCTIONS) / total
        aux = sum(1 for w in words if w.lower() in AUX_VERBS) / total
        noun_like = sum(1 for w in words if w.lower().endswith(NOUN_LIKE_SUFFIXES) or (w[0].isupper() and len(w) > 1)) / total
        feats.append([det, conj, aux, noun_like])
    return np.array(feats, dtype=np.float32)

def emotion_and_punctuation_features(texts: List[str]) -> np.ndarray:
    feats = []
    for txt in texts:
        words = re.findall(r"\b[a-z0-9']+\b", (txt or "").lower())
        total_words = len(words) or 1
        exclaims = txt.count('!'); ellipses = len(re.findall(r"\.\.\.+", txt)); questions = txt.count('?')
        emotive = sum(1 for w in words if w in EMOTIVE_WORDS)
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)
        func_count = sum(1 for w in words if (w in DETERMINERS or w in CONJUNCTIONS or w in AUX_VERBS))
        feats.append([exclaims/(1+total_words), ellipses/(1+total_words), questions/(1+total_words),
                      emotive/(1+total_words), neg/(1+total_words), func_count/(1+total_words)])
    return np.array(feats, dtype=np.float32)

def factuality_proxies(texts: List[str]) -> np.ndarray:
    feats = []
    for txt in texts:
        words = re.findall(r"\b[A-Za-z0-9\-/]+\b", (txt or ""))
        total = len(words) or 1
        numbers = sum(1 for w in words if re.fullmatch(r"[0-9]+", w))
        date_like = sum(1 for w in words if re.search(r'\d{4}|\d{1,2}/\d{1,2}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', w.lower()))
        proper_caps = len(re.findall(r'\b[A-Z][a-z]+\b', txt))
        urls = len(re.findall(r"http\S+|www\.\S+", txt))
        feats.append([numbers/total, date_like/total, proper_caps/total, urls/total])
    return np.array(feats, dtype=np.float32)

def calculate_comprehensive_feature_scores(texts: List[str]) -> np.ndarray:
    feats = []
    for text in texts:
        words = [w.lower() for w in re.findall(r"\b[a-z0-9']+\b", (text or ""))]
        total = len(words) or 1
        freq = Counter(words)
        ai_score = sum(freq.get(f, 0) for f in TOP_AI_FEATURES)
        human_score = sum(freq.get(f, 0) for f in TOP_HUMAN_FEATURES)
        formal_score = sum(freq.get(w, 0) for w in FORMAL_WORDS)
        conv_score = sum(freq.get(m, 0) for m in CONVERSATIONAL_MARKERS)
        ai_score_norm = ai_score / total
        human_score_norm = human_score / total
        formal_score_norm = formal_score / total
        conv_score_norm = conv_score / total
        total_feature_count = ai_score + human_score
        ai_human_ratio = ai_score / (total_feature_count + 1e-8)
        total_style_count = formal_score + conv_score
        formal_conv_ratio = formal_score / (total_style_count + 1e-8)
        feats.append([ai_score_norm, human_score_norm, formal_score_norm, conv_score_norm, ai_human_ratio, formal_conv_ratio])
    return np.array(feats, dtype=np.float32)

def analyze_text_complexity(texts: List[str]) -> np.ndarray:
    feats = []
    for text in texts:
        words = re.findall(r"\b[a-zA-Z0-9']+\b", (text or ""))
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', (text or "")) if s.strip()]
        characters = len(text or "")
        total_words = len(words) or 1
        sentence_lengths = [len(re.findall(r"\b[a-zA-Z0-9']+\b", s)) for s in sentences if s.strip()]
        avg_sentence_len = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        sentence_complexity = float(np.std(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0
        unique_word_ratio = len(set(w.lower() for w in words)) / total_words
        avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
        punctuation_count = len(re.findall(r'[.,!?;:]', text or ""))
        punctuation_density = punctuation_count / (characters + 1e-8)
        capital_ratio = len(re.findall(r'\b[A-Z][a-z]+\b', text or "")) / total_words
        contraction_ratio = len(re.findall(r"\b\w+'(\w+)\b", text or "")) / total_words
        question_ratio = (text or "").count('?') / (len(sentences) + 1e-8)
        feats.append([avg_sentence_len, unique_word_ratio, avg_word_len, sentence_complexity, punctuation_density, capital_ratio, contraction_ratio, question_ratio])
    return np.array(feats, dtype=np.float32)

def extract_linguistic_patterns(texts: List[str]) -> np.ndarray:
    feats = []
    for text in texts:
        words = re.findall(r"\b[a-z0-9']+\b", (text or "").lower())
        total = max(1, len(words))
        first_person = len(re.findall(r'\b(i|me|my|mine|we|us|our|ours)\b', (text or "").lower()))
        second_person = len(re.findall(r'\b(you|your|yours)\b', (text or "").lower()))
        passive_indicators = len(re.findall(r'\b(am|is|are|was|were|be|being|been)\s+\w+ed\b', (text or "").lower()))
        hedging = len(re.findall(r'\b(maybe|perhaps|possibly|probably|might|could|seems|appears)\b', (text or "").lower()))
        exclamation_ratio = (text or "").count('!') / total
        emotive_count = sum(1 for w in words if w in EMOTIVE_WORDS)
        emotive_ratio = emotive_count / total
        feats.append([first_person/total, second_person/total, passive_indicators/total, hedging/total, exclamation_ratio, emotive_ratio])
    return np.array(feats, dtype=np.float32)

def add_basic_stats(texts: List[str]) -> np.ndarray:
    chars = np.array([len(t or "") for t in texts], dtype=np.float32)
    words = np.array([len(re.findall(r"\b[a-zA-Z0-9']+\b", (t or ""))) for t in texts], dtype=np.float32)
    sentences = np.array([len([s for s in re.split(r'(?<=[.!?])\s+', (t or "")) if s.strip()]) for t in texts], dtype=np.float32)
    avg_w = chars / np.clip(words, a_min=1, a_max=None)
    avg_s = np.divide(words, np.clip(sentences, a_min=1, a_max=None))
    return np.vstack([chars, words, sentences, avg_w, avg_s]).T.astype(np.float32)

# -------------------- MAIN PIPELINE --------------------
def detect_text_label_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """Heuristic for picking text and label columns."""
    text_candidates = ["answer", "text", "prompt", "content", "body", "essay", "document"]
    label_candidates = ["generated", "is_generated", "generated_by", "is_ai", "generated_label", "label", "target", "is_cheating"]
    for c in text_candidates:
        if c in df.columns:
            text_col = c
            break
    else:
        # fallback: first column with dtype object
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        text_col = obj_cols[0] if obj_cols else df.columns[0]
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break
    else:
        # fallback: last column (common in Kaggle)
        label_col = df.columns[-1]
    return text_col, label_col

def run_enhanced_pipeline(
    kaggle_train_path: str = KAGGLE_TRAIN,
    kaggle_test_path: str = KAGGLE_TEST,
    kaggle_sample_path: str = KAGGLE_SAMPLE,
    output_path: str = "/kaggle/working/submission.csv",
    compute_perplexity: bool = True,
    transformer_model_name: str = "microsoft/deberta-v3-base",
    perplexity_model_name: str = "gpt2"
):
    set_seeds()
    # Check files
    for p in (kaggle_train_path, kaggle_test_path, kaggle_sample_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    train = pd.read_csv(kaggle_train_path)
    test = pd.read_csv(kaggle_test_path)
    sample = pd.read_csv(kaggle_sample_path)
    logger.info("Loaded datasets: train=%s, test=%s, sample=%s", train.shape, test.shape, sample.shape)

    text_col, label_col = detect_text_label_cols(train)
    logger.info("Using text column '%s' and label column '%s'", text_col, label_col)

    # Clean texts
    train[text_col] = train[text_col].map(clean_text)
    test[text_col] = test[text_col].map(clean_text)

    le = LabelEncoder()
    try:
        y = le.fit_transform(train[label_col].astype(str))
    except Exception as e:
        logger.warning("Label encoding failed: %s. Falling back to zero labels.", e)
        y = np.zeros(len(train), dtype=np.int32)

    X_texts = train[text_col].astype(str).tolist()
    test_texts = test[text_col].astype(str).tolist()
    corpus = X_texts + test_texts
    logger.info("Computing corpus word ranks...")
    rank_map = compute_corpus_word_ranks(corpus)

    # -------------------- Sparse TF-IDF --------------------
    logger.info("Building TF-IDF features...")
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,3), max_features=TF_WORD_MAX, min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=TF_CHAR_MAX, min_df=2, sublinear_tf=True)
    char_ngram_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=2000, min_df=2, sublinear_tf=True)
    word_ngram_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), max_features=2000, min_df=2, sublinear_tf=True)
    punc_vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 1), token_pattern=r'[^\w\s]', max_features=TF_PUNC_MAX, min_df=1, sublinear_tf=True)

    Xw_train = word_vec.fit_transform(X_texts); Xw_test = word_vec.transform(test_texts)
    Xc_train = char_vec.fit_transform(X_texts); Xc_test = char_vec.transform(test_texts)
    char_ngram_train = char_ngram_vec.fit_transform(X_texts); char_ngram_test = char_ngram_vec.transform(test_texts)
    word_ngram_train = word_ngram_vec.fit_transform(X_texts); word_ngram_test = word_ngram_vec.transform(test_texts)
    Xp_train = punc_vec.fit_transform(X_texts); Xp_test = punc_vec.transform(test_texts)

    # -------------------- DENSE FEATURES --------------------
    logger.info("Computing dense linguistic features...")
    basic_train = add_basic_stats(X_texts); basic_test = add_basic_stats(test_texts)
    lexical_train = calculate_lexical_richness(X_texts); lexical_test = calculate_lexical_richness(test_texts)
    rare_train = rare_word_rank_features(X_texts, rank_map); rare_test = rare_word_rank_features(test_texts, rank_map)
    feature_scores_train = calculate_comprehensive_feature_scores(X_texts); feature_scores_test = calculate_comprehensive_feature_scores(test_texts)
    complexity_train = analyze_text_complexity(X_texts); complexity_test = analyze_text_complexity(test_texts)
    linguistic_train = extract_linguistic_patterns(X_texts); linguistic_test = extract_linguistic_patterns(test_texts)
    repetition_train = repetition_features(X_texts, ngram_n=3); repetition_test = repetition_features(test_texts, ngram_n=3)
    function_train = function_word_features(X_texts); function_test = function_word_features(test_texts)
    emotion_train = emotion_and_punctuation_features(X_texts); emotion_test = emotion_and_punctuation_features(test_texts)
    factual_train = factuality_proxies(X_texts); factual_test = factuality_proxies(test_texts)

    # readability
    readability_train = get_readability_scores(X_texts); readability_test = get_readability_scores(test_texts)

    # perplexity (optional & safe)
    if compute_perplexity:
        logger.info("Computing GPT-2 perplexity (this may be slow / memory heavy)...")
        try:
            ppl_train = get_perplexity(X_texts, model_name=perplexity_model_name, batch_size=BATCH_SIZE)
            ppl_test = get_perplexity(test_texts, model_name=perplexity_model_name, batch_size=BATCH_SIZE)
        except Exception as e:
            logger.warning("Perplexity stage failed: %s. Filling with large constant.", e)
            ppl_train = np.full((len(X_texts), 1), 1e6, dtype=np.float32)
            ppl_test = np.full((len(test_texts), 1), 1e6, dtype=np.float32)
    else:
        ppl_train = np.full((len(X_texts), 1), 1e6, dtype=np.float32)
        ppl_test = np.full((len(test_texts), 1), 1e6, dtype=np.float32)

    # transformer embeddings + SVD
    logger.info("Computing transformer embeddings (may be slow)...")
    trans_train = get_transformer_embeddings(X_texts, model_name=transformer_model_name, batch_size=BATCH_SIZE)
    trans_test = get_transformer_embeddings(test_texts, model_name=transformer_model_name, batch_size=BATCH_SIZE)
    # reduce to TRANS_SVD_COMPONENTS (guard if embedding dim smaller)
    try:
        if trans_train.shape[1] >= TRANS_SVD_COMPONENTS:
            svd = TruncatedSVD(n_components=TRANS_SVD_COMPONENTS, random_state=SEED)
            trans_svd_train = svd.fit_transform(trans_train)
            trans_svd_test = svd.transform(trans_test)
        else:
            # pad or trim to desired size
            pad_train = np.zeros((trans_train.shape[0], TRANS_SVD_COMPONENTS - trans_train.shape[1]), dtype=np.float32)
            pad_test = np.zeros((trans_test.shape[0], TRANS_SVD_COMPONENTS - trans_test.shape[1]), dtype=np.float32)
            trans_svd_train = np.hstack([trans_train, pad_train])
            trans_svd_test = np.hstack([trans_test, pad_test])
    except Exception as e:
        logger.warning("SVD failed: %s. Using raw embeddings or zeros.", e)
        trans_svd_train = trans_train if trans_train.shape[1] >= TRANS_SVD_COMPONENTS else np.pad(trans_train, ((0,0),(0, TRANS_SVD_COMPONENTS - trans_train.shape[1])), 'constant')
        trans_svd_test = trans_test if trans_test.shape[1] >= TRANS_SVD_COMPONENTS else np.pad(trans_test, ((0,0),(0, TRANS_SVD_COMPONENTS - trans_test.shape[1])), 'constant')

    # combine dense
    dense_train = np.hstack([
        basic_train, lexical_train, rare_train, feature_scores_train, complexity_train,
        linguistic_train, repetition_train, function_train, emotion_train, factual_train,
        readability_train, ppl_train, trans_svd_train
    ]).astype(np.float32)
    dense_test = np.hstack([
        basic_test, lexical_test, rare_test, feature_scores_test, complexity_test,
        linguistic_test, repetition_test, function_test, emotion_test, factual_test,
        readability_test, ppl_test, trans_svd_test
    ]).astype(np.float32)

    X_train = hstack([Xw_train, Xc_train, char_ngram_train, word_ngram_train, Xp_train, csr_matrix(dense_train)])
    X_test = hstack([Xw_test, Xc_test, char_ngram_test, word_ngram_test, Xp_test, csr_matrix(dense_test)])
    logger.info("Feature shapes: X_train=%s, X_test=%s, dense=%s", X_train.shape, X_test.shape, dense_train.shape)

    # -------------------- CROSS-VALIDATION & BASE MODELS --------------------
    class_counts = Counter(y)
    min_class_count = min(class_counts.values()) if class_counts else 0
    n_splits = min(5, min_class_count) if min_class_count >= 2 else 2
    n_splits = max(2, n_splits)
    logger.info("Using StratifiedKFold with n_splits=%s; class_counts=%s", n_splits, dict(class_counts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    num_classes = len(np.unique(y))
    n_samples = X_train.shape[0]
    # For multiclass stacking we will keep all class probabilities from each model
    oof_probs = {
        "lr": np.zeros((n_samples, num_classes), dtype=np.float32),
        "lgb": np.zeros((n_samples, num_classes), dtype=np.float32),
        "xgb": np.zeros((n_samples, num_classes), dtype=np.float32)
    }
    test_probs = {
        "lr": np.zeros((X_test.shape[0], num_classes), dtype=np.float32),
        "lgb": np.zeros((X_test.shape[0], num_classes), dtype=np.float32),
        "xgb": np.zeros((X_test.shape[0], num_classes), dtype=np.float32)
    }

    fold_auc = {"lr": [], "lgb": [], "xgb": []}
    logger.info("Beginning CV training...")

    # convert X_train/X_test to formats expected by models where needed
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_samples), y), start=1):
        logger.info("Fold %d/%d", fold_idx, n_splits)
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # --- Logistic Regression (sparse friendly)
        lr = LogisticRegression(max_iter=2500, solver="liblinear", C=0.1, random_state=SEED, class_weight="balanced")
        lr.fit(X_tr, y_tr)
        va_proba_lr = lr.predict_proba(X_va)
        test_proba_lr = lr.predict_proba(X_test)
        oof_probs["lr"][va_idx] = va_proba_lr
        test_probs["lr"] += test_proba_lr / n_splits
        if num_classes == 2:
            try:
                auc_lr = roc_auc_score(y_va, va_proba_lr[:, 1])
                fold_auc["lr"].append(auc_lr)
            except Exception:
                pass

        # --- LightGBM
        lgb_params = {
            "objective": "binary" if num_classes == 2 else "multiclass",
            "metric": "auc" if num_classes == 2 else "multi_logloss",
            "learning_rate": 0.04, "num_leaves": 31, "max_depth": -1, "verbosity": -1, "seed": SEED,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "min_child_samples": 20, "n_jobs": -1
        }
        if num_classes > 2:
            lgb_params["num_class"] = num_classes
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        bst = lgb.train(lgb_params, dtrain, num_boost_round=500, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(0)])
        va_proba_lgb = bst.predict(X_va)
        test_proba_lgb = bst.predict(X_test)
        # ensure shape (n_samples, num_classes)
        if num_classes == 2:
            va_proba_lgb = np.vstack([1 - va_proba_lgb, va_proba_lgb]).T
            test_proba_lgb = np.vstack([1 - test_proba_lgb, test_proba_lgb]).T
        oof_probs["lgb"][va_idx] = va_proba_lgb
        test_probs["lgb"] += test_proba_lgb / n_splits
        if num_classes == 2:
            try:
                fold_auc["lgb"].append(roc_auc_score(y_va, va_proba_lgb[:, 1]))
            except Exception:
                pass

        # --- XGBoost
        xgb_params = {
            "objective": "binary:logistic" if num_classes == 2 else "multi:softprob",
            "eval_metric": "auc" if num_classes == 2 else "mlogloss",
            "eta": 0.05, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.8, "seed": SEED,
            "tree_method": "hist", "verbosity": 0
        }
        if num_classes > 2:
            xgb_params["num_class"] = num_classes
        dxgb_train = xgb.DMatrix(X_tr, label=y_tr)
        dxgb_val = xgb.DMatrix(X_va, label=y_va)
        dxgb_test = xgb.DMatrix(X_test)
        xgb_bst = xgb.train(xgb_params, dxgb_train, num_boost_round=500, evals=[(dxgb_val, "val")],
                            early_stopping_rounds=30, verbose_eval=False)
        va_proba_xgb = xgb_bst.predict(dxgb_val)
        test_proba_xgb = xgb_bst.predict(dxgb_test)
        # ensure shape
        if num_classes == 2:
            va_proba_xgb = np.vstack([1 - va_proba_xgb, va_proba_xgb]).T
            test_proba_xgb = np.vstack([1 - test_proba_xgb, test_proba_xgb]).T
        oof_probs["xgb"][va_idx] = va_proba_xgb
        test_probs["xgb"] += test_proba_xgb / n_splits
        if num_classes == 2:
            try:
                fold_auc["xgb"].append(roc_auc_score(y_va, va_proba_xgb[:, 1]))
            except Exception:
                pass

    # CV summaries
    def safe_mean(l): return float(np.mean(l)) if l else 0.0
    logger.info("CV AUCs -> LR: %.6f, LGB: %.6f, XGB: %.6f", safe_mean(fold_auc["lr"]), safe_mean(fold_auc["lgb"]), safe_mean(fold_auc["xgb"]))

    # -------------------- STACKING (meta model) --------------------
    logger.info("Preparing stacking meta-features...")
    # create meta features by concatenating probabilities for each model (all classes)
    X_meta = np.hstack([oof_probs["lr"], oof_probs["lgb"], oof_probs["xgb"]])
    X_meta_test = np.hstack([test_probs["lr"], test_probs["lgb"], test_probs["xgb"]])
    logger.info("Meta feature shapes: train=%s, test=%s", X_meta.shape, X_meta_test.shape)

    # meta model
    meta = LogisticRegression(solver="liblinear", C=0.05, random_state=SEED, max_iter=2000)
    try:
        meta.fit(X_meta, y)
        final_proba = meta.predict_proba(X_meta_test)
    except Exception as e:
        logger.warning("Meta model failed (%s). Falling back to average of base probabilities.", e)
        final_proba = (test_probs["lr"] + test_probs["lgb"] + test_probs["xgb"]) / 3.0

    # -------------------- prepare submission --------------------
    logger.info("Preparing submission to %s", output_path)
    sample_out = sample.copy()
    # choose second column (common Kaggle sample format: id, target)
    if sample_out.shape[1] < 2:
        target_col = sample_out.columns[-1]
    else:
        target_col = sample_out.columns[1]
    if final_proba.shape[1] == 2:
        # binary: write probability of positive class
        probs = final_proba[:, 1]
        sample_out[target_col] = probs
    else:
        # multiclass: try to map back to label encoder classes if possible; otherwise write argmax
        preds_idx = np.argmax(final_proba, axis=1)
        try:
            inv = le.inverse_transform(preds_idx)
            sample_out[target_col] = inv
        except Exception:
            sample_out[target_col] = preds_idx
    sample_out.to_csv(output_path, index=False)
    logger.info("Saved submission to %s", output_path)

# -------------------- RUN as SCRIPT --------------------
if __name__ == "__main__":
    logger.info("Starting improved Mercor AI detection pipeline v4...")
    try:
        run_enhanced_pipeline()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        raise





