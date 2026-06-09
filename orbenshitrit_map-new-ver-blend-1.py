%%writefile map_lr_map.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, gc, re, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ================== ENV / PATHS ==================
IS_KAGGLE = os.path.exists("/kaggle/input")
print(">>> ENV:", "KAGGLE" if IS_KAGGLE else "LOCAL")

# --- safe env readers ---
def _env_token(name: str, default) -> str:
    raw = os.getenv(name, str(default))
    raw = str(raw).split('#', 1)[0].strip()
    return raw.split()[0] if raw else str(default)
def _env_int(name: str, default: int) -> int:
    try: return int(float(_env_token(name, default)))
    except: return int(default)
def _env_float(name: str, default: float) -> float:
    try: return float(_env_token(name, default))
    except: return float(default)
def _env_bool(name: str, default="0") -> bool:
    return _env_token(name, default).lower() in ("1","true","yes")

@dataclass
class Config:
    data_dir: str = "/kaggle/input/map-charting-student-math-misunderstandings" if IS_KAGGLE else "data/raw"
    out_dir:  str = "/kaggle/working" if IS_KAGGLE else "outputs"

    # base features (single-view)
    word_max_features: int = _env_int("WORD_MAX", 60000)
    char_max_features: int = _env_int("CHAR_MAX", 30000)
    word_ngram: Tuple[int, int] = (1, 2)
    char_ngram: Tuple[int, int] = (3, 5)
    min_df: int = 2

    # weights
    w_exp: float = 1.0
    w_ctx: float = _env_float("W_CTX", 0.32)

    # LR
    C: float      = _env_float("C", 2.0)
    max_iter: int = _env_int("MAX_ITER", 3500)
    tol: float    = 1e-4
    n_jobs: int   = -1
    random_state: int = 42

    # split-views defaults
    qa_word_max:  int = _env_int("QA_WORD_MAX", 30000)
    exp_word_max: int = _env_int("EXP_WORD_MAX", 15000)
    exp_char_max: int = _env_int("EXP_CHAR_MAX", 25000)

CFG = Config()

# toggles (×‘×¨×™×¨×ª ×”×�×—×“×œ = ×”×‘×™×™×¡×œ×™×™×Ÿ + ×©×“×¨×•×’×™×�)
FIT_ON_ALL  = _env_bool("FIT_ON_ALL", "1")
MAX_DF_EXP  = _env_float("MAX_DF_EXP", 0.90)
NUMFEAT     = _env_bool("NUMFEAT", "1")
NUM_W       = _env_float("NUM_W", 0.8)

ISCORR      = _env_bool("ISCORR", "1")
ISCORR_W    = _env_float("ISCORR_W", 1.0)
KEEP_TOKENS = _env_bool("KEEP_TOKENS", "1")
TEXT_ORDER  = _env_token("TEXT_ORDER", "qa_first")  # ×�×• "exp_first"

PRIORS      = _env_bool("PRIORS", "1")
PRIORS_TAU  = _env_float("PRIORS_TAU", 0.62)
PRIORS_QIA  = _env_bool("PRIORS_QIA", "1")

QMASK       = _env_bool("QMASK", "1")
QMASK_EPS   = _env_float("QMASK_EPS", 0.02)
DYN_TAU     = _env_bool("DYN_TAU", "1")
TAU_LO      = _env_float("TAU_LO", 0.55)
TAU_HI      = _env_float("TAU_HI", 0.90)
CONF_HI     = _env_float("CONF_HI", 0.65)
CONF_LO     = _env_float("CONF_LO", 0.40)

DIVERSIFY   = _env_bool("DIVERSIFY", "0")

SPLIT_VIEWS = _env_bool("SPLIT_VIEWS", "0")
W_QA_W      = _env_float("W_QA_W", 1.0)
W_EXP_W     = _env_float("W_EXP_W", 1.0)
W_EXP_C     = _env_float("W_EXP_C", CFG.w_ctx)

HIER        = _env_bool("HIER", "1")
HIER_BETA   = _env_float("HIER_BETA", 0.6)

# Guard-rails ×œ×�×©×¤×—×•×ª (True_/False_)
FAMILY_GUARD = _env_bool("FAMILY_GUARD", "1")
FAMILY_MODE  = _env_token("FAMILY_MODE", "hard")  # "hard" ×�×• "soft"
FAMILY_EPS   = _env_float("FAMILY_EPS", 0.02)

# external blend (×�×� ×ª×¨×¦×” ×œ×¢×¨×‘×œ ×¢×� ×�×•×“×œ LLM ×©×©×�×¨ probs.npy+labels.json)
BLEND2        = _env_bool("BLEND2", "0")
BLEND2_DIR    = _env_token("BLEND2_DIR", "")
BLEND2_WEIGHT = _env_float("BLEND2_WEIGHT", 0.60)

# ×©×�×™×¨×ª ×�×¨×˜×™×¤×§×˜×™×� ×œ-Ensemble
SAVE_PROBS = _env_bool("SAVE_PROBS", "1")
SAVE_TOPK  = _env_int("SAVE_TOPK", 25)

print(f">>> FIT_ON_ALL={FIT_ON_ALL} | W_CTX={CFG.w_ctx} | C={CFG.C} | MAX_ITER={CFG.max_iter}")
print(f">>> MAX_DF_EXP={MAX_DF_EXP} | NUMFEAT={NUMFEAT} (NUM_W={NUM_W})")
print(f">>> ISCORR={ISCORR} (W={ISCORR_W}, KEEP_TOKENS={KEEP_TOKENS})")
print(f">>> PRIORS={PRIORS} (TAU={PRIORS_TAU}, QIA={PRIORS_QIA}, DYN_TAU={DYN_TAU}) | QMASK={QMASK} (EPS={QMASK_EPS})")
print(f">>> SPLIT_VIEWS={SPLIT_VIEWS} | HIER={HIER} (BETA={HIER_BETA})")
print(f">>> FAMILY_GUARD={FAMILY_GUARD} (MODE={FAMILY_MODE}, EPS={FAMILY_EPS})")
print(f">>> BLEND2={BLEND2} (DIR='{BLEND2_DIR}', W={BLEND2_WEIGHT}) | SAVE_PROBS={SAVE_PROBS} (TOPK={SAVE_TOPK})")

# ================== Utils ==================

# × ×¨×�×•×œ ×˜×§×¡×˜ ×�×ª×�×˜×™ ×§×œ (×�×™×™×©×¨ LaTeX/×¡×™×�× ×™×� ×œ×’×¨×¡×�×•×ª ×˜×§×¡×˜×•×�×œ×™×•×ª ×™×¦×™×‘×•×ª ×œ-TFIDF)
def normalize_math(s: str) -> str:
    s = str(s)
    s = re.sub(r'\\\(|\\\)|\$\$?', '', s)
    s = re.sub(r'\\frac\s*{([^}]*)}\s*{([^}]*)}', lambda m: f"{m.group(1)} over {m.group(2)}", s)
    s = s.replace('\\div', ' divided by ').replace('Ã—', ' times ').replace('Ã·', ' divided by ')
    s = re.sub(r'\s*=\s*', ' equals ', s)
    s = re.sub(r'(?<=\d)\s*\+\s*(?=\d)', ' plus ', s)
    s = re.sub(r'(?<=\d)\s*\*\s*(?=\d)', ' times ', s)
    s = re.sub(r'(?<=\d)\s*-\s*(?=\d)', ' minus ', s)
    s = re.sub(r'(?<!\w)-(?=\d)', ' minus ', s)
    s = s.replace('^', ' to the power of ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _fit_corpus(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.concat([a, b], ignore_index=True) if FIT_ON_ALL else a

def compute_is_correct_map(train_df: pd.DataFrame) -> pd.DataFrame:
    idx = train_df["Category"].astype(str).str.startswith("True")
    correct = train_df.loc[idx, ["QuestionId","MC_Answer"]].copy()
    correct["c"] = correct.groupby(["QuestionId","MC_Answer"])["MC_Answer"].transform("count")
    correct = correct.sort_values("c", ascending=False).drop_duplicates(["QuestionId"])
    correct = correct[["QuestionId","MC_Answer"]].rename(columns={"MC_Answer":"correct_answer"})
    correct["is_correct"] = 1
    return correct

def attach_is_correct(train_df, test_df):
    corr = compute_is_correct_map(train_df)
    def _merge(df):
        out = df.merge(corr, on="QuestionId", how="left")
        is_corr = (out["MC_Answer"].astype(str) == out["correct_answer"].astype(str)).astype(float)
        out["is_correct"] = is_corr.fillna(0.0).astype(np.float32)
        out.drop(columns=["correct_answer"], inplace=True)
        return out
    return _merge(train_df.copy()), _merge(test_df.copy())

def build_views(df: pd.DataFrame) -> pd.Series:
    exp = df["StudentExplanation"].fillna("").astype(str).map(normalize_math)
    q   = df["QuestionText"].fillna("").astype(str).map(normalize_math)
    ans = df["MC_Answer"].fillna("").astype(str).map(normalize_math)
    if ISCORR and "is_correct" in df.columns:
        tag = pd.Series(np.where(df["is_correct"].values > 0.5, "[CORRECT]", "[INCORRECT]"), index=df.index)
    else:
        tag = pd.Series([""] * len(df), index=df.index)
    if TEXT_ORDER == "exp_first":
        return exp + " [SEP] Q: " + q + " [ANS] " + ans + " " + tag
    return "Q: " + q + " [ANS] " + ans + " " + tag + " [SEP] " + exp

def build_views_split(df: pd.DataFrame):
    exp = df["StudentExplanation"].fillna("").astype(str).map(normalize_math)
    q   = df["QuestionText"].fillna("").astype(str).map(normalize_math)
    ans = df["MC_Answer"].fillna("").astype(str).map(normalize_math)
    if ISCORR and "is_correct" in df.columns:
        tag = pd.Series(np.where(df["is_correct"].values > 0.5, "[CORRECT]", "[INCORRECT]"), index=df.index)
    else:
        tag = pd.Series([""] * len(df), index=df.index)
    qa  = "Q: " + q + " [ANS] " + ans + " " + tag
    return qa, exp

def extract_light_num_feats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[csr_matrix, csr_matrix]:
    def _make(df):
        t = df["StudentExplanation"].fillna("").astype(str)
        c = ("Q: " + df["QuestionText"].fillna("").astype(str) + " [ANS] " + df["MC_Answer"].fillna("").astype(str))
        feat = pd.DataFrame({
            "exp_len": t.str.len(),
            "exp_words": t.str.split().map(len),
            "exp_digits": t.str.count(r"\d"),
            "exp_ops": t.str.count(r"[+\-*/=^]"),
            "ctx_len": c.str.len(),
        })
        feat["exp_ctx_ratio"] = feat["exp_len"] / (feat["ctx_len"]+1)
        if ISCORR and "is_correct" in df.columns:
            feat["is_correct"] = df["is_correct"].astype(float) * ISCORR_W
        return feat.fillna(0)
    tr = _make(train_df); te = _make(test_df)
    scaler = StandardScaler()
    tr_scaled = scaler.fit_transform(tr)
    te_scaled = scaler.transform(te)
    return csr_matrix(NUM_W * tr_scaled), csr_matrix(NUM_W * te_scaled)

# --------- Vectorizers ---------
def vectorize_single(text_tr, text_te) -> Tuple[csr_matrix, csr_matrix]:
    token_pat = r"(?u)\b\w+\b|\[SEP\]|\[ANS\]"
    if KEEP_TOKENS:
        token_pat = r"(?u)\b\w+\b|\[SEP\]|\[ANS\]|\[CORRECT\]|\[INCORRECT\]"
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=CFG.word_ngram, min_df=CFG.min_df,
        max_features=CFG.word_max_features, lowercase=True, strip_accents="unicode",
        sublinear_tf=True, dtype=np.float32, max_df=MAX_DF_EXP, token_pattern=token_pat
    )
    char_vec = TfidfVectorizer(
        analyzer="char", ngram_range=CFG.char_ngram, min_df=CFG.min_df,
        max_features=CFG.char_max_features, lowercase=True, sublinear_tf=True, dtype=np.float32
    )
    word_vec.fit(_fit_corpus(text_tr, text_te))
    char_vec.fit(_fit_corpus(text_tr, text_te))
    X_tr_w = word_vec.transform(text_tr);  X_te_w = word_vec.transform(text_te)
    X_tr_c = char_vec.transform(text_tr);  X_te_c = char_vec.transform(text_te)
    X_tr = hstack([1.0*X_tr_w, CFG.w_ctx*X_tr_c], format="csr")
    X_te = hstack([1.0*X_te_w, CFG.w_ctx*X_te_c], format="csr")
    del X_tr_w, X_tr_c, X_te_w, X_te_c; gc.collect()
    return X_tr, X_te

def vectorize_split(qa_tr, qa_te, exp_tr, exp_te) -> Tuple[csr_matrix, csr_matrix]:
    token_pat = r"(?u)\b\w+\b|\[SEP\]|\[ANS\]"
    if KEEP_TOKENS:
        token_pat = r"(?u)\b\w+\b|\[SEP\]|\[ANS\]|\[CORRECT\]|\[INCORRECT\]"
    qa_word = TfidfVectorizer(analyzer="word", ngram_range=CFG.word_ngram, min_df=CFG.min_df,
                              max_features=CFG.qa_word_max, lowercase=True, strip_accents="unicode",
                              sublinear_tf=True, dtype=np.float32, max_df=MAX_DF_EXP, token_pattern=token_pat)
    exp_word = TfidfVectorizer(analyzer="word", ngram_range=CFG.word_ngram, min_df=CFG.min_df,
                               max_features=CFG.exp_word_max, lowercase=True, strip_accents="unicode",
                               sublinear_tf=True, dtype=np.float32, max_df=MAX_DF_EXP)
    exp_char = TfidfVectorizer(analyzer="char", ngram_range=CFG.char_ngram, min_df=CFG.min_df,
                               max_features=CFG.exp_char_max, lowercase=True, sublinear_tf=True, dtype=np.float32)

    qa_word.fit(_fit_corpus(qa_tr, qa_te))
    exp_word.fit(_fit_corpus(exp_tr, exp_te))
    exp_char.fit(_fit_corpus(exp_tr, exp_te))

    Xtr_qw = qa_word.transform(qa_tr); Xte_qw = qa_word.transform(qa_te)
    Xtr_ew = exp_word.transform(exp_tr); Xte_ew = exp_word.transform(exp_te)
    Xtr_ec = exp_char.transform(exp_tr); Xte_ec = exp_char.transform(exp_te)

    X_tr = hstack([W_QA_W*Xtr_qw, W_EXP_W*Xtr_ew, W_EXP_C*Xtr_ec], format="csr")
    X_te = hstack([W_QA_W*Xte_qw, W_EXP_W*Xte_ew, W_EXP_C*Xte_ec], format="csr")
    del Xtr_qw, Xtr_ew, Xtr_ec, Xte_qw, Xte_ew, Xte_ec; gc.collect()
    return X_tr, X_te

# --------- Model ----------
def train_and_predict(X_tr: csr_matrix, y: np.ndarray, X_te: csr_matrix) -> tuple[np.ndarray, LogisticRegression]:
    clf = LogisticRegression(
        solver="saga", penalty="l2", C=CFG.C,
        max_iter=CFG.max_iter, tol=CFG.tol,
        n_jobs=CFG.n_jobs, random_state=CFG.random_state,
        multi_class="multinomial", verbose=0
    )
    clf.fit(X_tr, y)
    return clf.predict_proba(X_te), clf

# --------- Priors ----------
def build_priors_is(train_df: pd.DataFrame, labels: List[str], alpha: float=1.0):
    key_cols = ["QuestionId","is_correct","label"]
    grp = train_df[key_cols].groupby(key_cols).size().reset_index(name="cnt")
    priors = {}
    label2idx = {lbl:i for i,lbl in enumerate(labels)}
    for (qid, ic), g in grp.groupby(["QuestionId","is_correct"]):
        arr = np.full(len(labels), alpha, dtype=np.float64)
        for _, r in g.iterrows():
            arr[label2idx[r["label"]]] += r["cnt"]
        priors[(int(qid), float(ic)>0.5)] = (arr / arr.sum()).astype(np.float32)
    return priors

def apply_priors_is(P: np.ndarray, df: pd.DataFrame, priors: Dict, tau: float) -> np.ndarray:
    P = P.copy()
    qids = df["QuestionId"].values
    ics  = (df["is_correct"].values > 0.5)
    for i,(q,ic) in enumerate(zip(qids,ics)):
        key = (int(q), bool(ic))
        if key in priors:
            pv = priors[key]
            t = tau
            if DYN_TAU:
                conf = float(P[i].max())
                if conf >= CONF_HI: t = TAU_LO
                elif conf <= CONF_LO: t = TAU_HI
                else:
                    u = (CONF_HI - conf) / max(CONF_HI - CONF_LO, 1e-6)
                    t = TAU_LO + u * (TAU_HI - TAU_LO)
            mixed = P[i] * (pv ** t)
            s = mixed.sum()
            if s > 0: P[i] = mixed / s
    return P

def build_priors_qia(train_df: pd.DataFrame, labels: List[str], alpha: float=1.0):
    key = ["QuestionId","is_correct","MC_Answer","label"]
    g = train_df[key].groupby(key).size().reset_index(name="cnt")
    label2idx = {lbl:i for i,lbl in enumerate(labels)}
    pri = {}
    for (qid, ic, ans), sub in g.groupby(["QuestionId","is_correct","MC_Answer"]):
        arr = np.full(len(labels), alpha, dtype=np.float64)
        for _, r in sub.iterrows():
            arr[label2idx[r["label"]]] += r["cnt"]
        pri[(int(qid), float(ic)>0.5, str(ans))] = (arr / arr.sum()).astype(np.float32)
    return pri

def apply_priors_qia(P: np.ndarray, df: pd.DataFrame, pri: Dict, tau: float) -> np.ndarray:
    P = P.copy()
    q = df["QuestionId"].values
    ic = (df["is_correct"].values > 0.5)
    a  = df["MC_Answer"].astype(str).values
    for i,(qid,icc,ans) in enumerate(zip(q,ic,a)):
        key = (int(qid), bool(icc), str(ans))
        if key in pri:
            pv = pri[key]
            mixed = P[i] * (pv ** tau)
            s = mixed.sum()
            if s > 0: P[i] = mixed / s
    return P

# --------- Q-Mask ----------
def build_seen_labels_by_qid(train_df: pd.DataFrame) -> Dict[int, set]:
    seen = {}
    for qid, g in train_df.groupby("QuestionId"):
        seen[int(qid)] = set(g["label"].astype(str).unique().tolist())
    return seen

def apply_qmask(P: np.ndarray, df: pd.DataFrame, seen: Dict[int,set], labels: List[str], eps: float=0.01):
    if not seen: return P
    label_set = np.array(labels, dtype=str)
    P = P.copy()
    qids = df["QuestionId"].values
    for i, q in enumerate(qids):
        s = seen.get(int(q))
        if not s: continue
        mask = ~np.isin(label_set, list(s))
        if mask.any():
            P[i, mask] *= float(eps)
            ssum = P[i].sum()
            if ssum > 0: P[i] /= ssum
    return P

# --------- Family guard & backfill ----------
def build_family_prefix(df: pd.DataFrame) -> np.ndarray:
    if "is_correct" not in df.columns:
        raise ValueError("is_correct missing; call attach_is_correct() first")
    return np.where(df["is_correct"].values > 0.5, "True_", "False_")

def apply_family_guard(P: np.ndarray, labels: List[str], fam_prefix: np.ndarray,
                       mode: str = "hard", eps: float = 0.02) -> np.ndarray:
    P = P.copy()
    lab = np.array(labels, dtype=str)
    for i in range(P.shape[0]):
        pref = fam_prefix[i]
        bad = ~np.char.startswith(lab, pref)
        if mode == "hard":
            P[i, bad] = 0.0
        else:
            P[i, bad] *= float(eps)
        s = P[i].sum()
        if s > 0: P[i] /= s
    return P

def top3_with_safe_backfill(P: np.ndarray, id2label: Dict[int,str], fam_prefix: np.ndarray) -> List[str]:
    order = np.argsort(-P, axis=1)
    out = []
    for i in range(P.shape[0]):
        pref = fam_prefix[i]
        row = [id2label[j] for j in order[i] if id2label[j].startswith(pref)]
        # dedupe
        seen=set(); row=[x for x in row if not (x in seen or seen.add(x))]
        fillers = [f"{pref}Neither:NA"] + ([f"{pref}Correct:NA"] if pref=="True_" else [])
        for f in fillers:
            if len(row)>=3: break
            if f not in row: row.append(f)
        while len(row)<3:
            row.append(fillers[0])
        out.append(" ".join(row[:3]))
    return out

# --------- External blend ----------
def blend_with_external(P_base: np.ndarray, labels: List[str], label2id: Dict[str,int]) -> np.ndarray:
    if not BLEND2 or not BLEND2_DIR:
        return P_base
    try:
        path = Path(BLEND2_DIR)
        L = json.loads((path / "labels.json").read_text(encoding="utf-8"))
        P2 = np.load(path / "probs.npy")  # shape [n_test, len(L)]
        Q = np.zeros_like(P_base, dtype=np.float32)
        for j, lbl in enumerate(L):
            idx = label2id.get(lbl)
            if idx is not None:
                Q[:, idx] = P2[:, j]
        w = float(BLEND2_WEIGHT)
        P = (np.clip(P_base, 1e-9, 1.0) ** (1.0 - w)) * (np.clip(Q, 1e-9, 1.0) ** w)
        P /= np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
        print(f"[BLEND2] blended with {BLEND2_DIR} (w={w})")
        return P
    except Exception as e:
        print("[BLEND2] skip (error):", e)
        return P_base

# --------- Hierarchical gating ----------
def train_category_model(X_tr: csr_matrix, y_label: np.ndarray, X_te: csr_matrix, labels: List[str]):
    cats = [lbl.split(":",1)[0] for lbl in labels]
    cat2id = {c:i for i,c in enumerate(sorted(set(cats)))}
    y_cat = np.array([cat2id[cats[i]] for i in y_label], dtype=int)
    clf = LogisticRegression(
        solver="saga", penalty="l2", C=CFG.C, max_iter=CFG.max_iter, tol=CFG.tol,
        n_jobs=CFG.n_jobs, random_state=CFG.random_state, multi_class="multinomial"
    )
    clf.fit(X_tr, y_cat)
    P_cat = clf.predict_proba(X_te)
    cat_idx = np.array([cat2id[c] for c in cats])
    return P_cat, cat_idx

# ================== Main ==================
def main():
    out_dir = Path(CFG.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(CFG.data_dir)

    train = pd.read_csv(data_dir/"train.csv")
    test  = pd.read_csv(data_dir/"test.csv")

    # labels
    train["Category"]      = train["Category"].astype("string")
    train["Misconception"] = train["Misconception"].fillna("NA").astype("string")
    train["label"]         = train["Category"] + ":" + train["Misconception"]
    labels   = sorted(train["label"].unique())
    label2id = {lbl:i for i,lbl in enumerate(labels)}
    id2label = {i:lbl for lbl,i in label2id.items()}
    y        = train["label"].map(label2id).values

    # is_correct
    if ISCORR:
        train, test = attach_is_correct(train, test)

    # text
    if SPLIT_VIEWS:
        qa_tr, exp_tr = build_views_split(train)
        qa_te, exp_te = build_views_split(test)
    else:
        text_tr = build_views(train)
        text_te = build_views(test)

    # numeric
    if NUMFEAT:
        num_tr, num_te = extract_light_num_feats(train, test)
    else:
        num_tr = num_te = None

    # vectorize
    if SPLIT_VIEWS:
        X_tr, X_te = vectorize_split(qa_tr, qa_te, exp_tr, exp_te)
    else:
        X_tr, X_te = vectorize_single(text_tr, text_te)
    if NUMFEAT:
        X_tr = hstack([X_tr, num_tr], format="csr")
        X_te = hstack([X_te, num_te], format="csr")

    # train & predict
    P, clf = train_and_predict(X_tr, y, X_te)

    # priors
    if PRIORS and ISCORR:
        if PRIORS_QIA:
            pri = build_priors_qia(train, labels, alpha=1.0)
            P   = apply_priors_qia(P, test, pri, tau=PRIORS_TAU)
        else:
            pri = build_priors_is(train, labels, alpha=1.0)
            P   = apply_priors_is(P, test, pri, tau=PRIORS_TAU)

    # q-mask (×¨×š)
    if QMASK:
        seen = build_seen_labels_by_qid(train)
        P = apply_qmask(P, test, seen, labels, eps=QMASK_EPS)

    # hierarchical gating (×§×˜×’×•×¨×™×”)
    if HIER:
        P_cat, cat_idx = train_category_model(X_tr, y, X_te, labels)
        P *= np.power(P_cat[:, cat_idx], HIER_BETA)
        P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)

    # external blend (×�×� ×“×œ×•×§)
    P = blend_with_external(P, labels, label2id)

    # Guard-rails ×œ×�×©×¤×—×” + backfill ×‘×˜×•×—
    if FAMILY_GUARD and ISCORR:
        fam = build_family_prefix(test)
        P = apply_family_guard(P, labels, fam, mode=FAMILY_MODE, eps=FAMILY_EPS)
        pred_strings = top3_with_safe_backfill(P, id2label, fam)
    else:
        top3_idx = np.argsort(-P, axis=1)[:, :3]
        pred_strings = [" ".join(id2label[i] for i in row) for row in top3_idx]

    # ×©×�×™×¨×ª submission
    sub = pd.DataFrame({"row_id": test["row_id"], "Category:Misconception": pred_strings})
    sub["Category:Misconception"] = sub["Category:Misconception"].astype(str).str.replace(":nan", ":NA", regex=False)
    path = Path(CFG.out_dir)/"submission.csv"
    sub.to_csv(path, index=False)
    print(f"âœ… Saved submission â†’ {path}")

    # ×©×�×™×¨×ª ×�×¨×˜×™×¤×§×˜×™×� ×œ-Ensemble/× ×™×ª×•×—
    if SAVE_PROBS:
        topk = min(SAVE_TOPK, P.shape[1])
        order = np.argsort(-P, axis=1)[:, :topk]
        top_lbl = np.array([[id2label[j] for j in row] for row in order])
        top_prb = np.array([P[i, order[i]] for i in range(P.shape[0])])
        rows=[]
        for i in range(P.shape[0]):
            d = {f"prob_{j}": float(top_prb[i,j]) for j in range(topk)}
            d["row_id"] = int(test["row_id"].iloc[i])
            d["top_classes"] = " ".join(top_lbl[i])
            rows.append(d)
        prob_path = Path(CFG.out_dir)/"submission_lr_probabilities.csv"
        pd.DataFrame(rows).to_csv(prob_path, index=False)
        (Path(CFG.out_dir)/"labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
        np.save(Path(CFG.out_dir)/"probs.npy", P.astype(np.float32))
        print(f"ğŸ’¾ Saved probs â†’ {prob_path} + labels.json + probs.npy")

    # meta
    meta = {
        "fit_on_all": FIT_ON_ALL, "max_df_exp": MAX_DF_EXP,
        "numfeat": NUMFEAT, "num_w": NUM_W,
        "iscorr": ISCORR, "iscorr_w": ISCORR_W, "keep_tokens": KEEP_TOKENS, "text_order": TEXT_ORDER,
        "priors": PRIORS, "priors_tau": PRIORS_TAU, "priors_qia": PRIORS_QIA,
        "qmask": QMASK, "qmask_eps": QMASK_EPS, "dyn_tau": DYN_TAU,
        "split_views": SPLIT_VIEWS, "w_qa_w": W_QA_W, "w_exp_w": W_EXP_W, "w_exp_c": W_EXP_C,
        "hier": HIER, "hier_beta": HIER_BETA,
        "family_guard": FAMILY_GUARD, "family_mode": FAMILY_MODE, "family_eps": FAMILY_EPS,
        "blend2": BLEND2, "blend2_dir": BLEND2_DIR, "blend2_weight": BLEND2_WEIGHT,
        "C": CFG.C, "max_iter": CFG.max_iter, "w_ctx": CFG.w_ctx,
        "word_max": CFG.word_max_features, "char_max": CFG.char_max_features,
        "qa_word_max": CFG.qa_word_max, "exp_word_max": CFG.exp_word_max, "exp_char_max": CFG.exp_char_max
    }
    (Path(CFG.out_dir)/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()



%env FIT_ON_ALL=1
%env WORD_MAX=45000
%env CHAR_MAX=25000
%env W_CTX=0.32
%env C=2.0
%env MAX_ITER=3000
%env MAX_DF_EXP=0.90

%env NUMFEAT=1
%env NUM_W=0.8

%env ISCORR=1
%env ISCORR_W=1.0
%env KEEP_TOKENS=1
%env TEXT_ORDER=qa_first

%env PRIORS=1
%env PRIORS_TAU=0.62
%env PRIORS_QIA=1

%env QMASK=1
%env QMASK_EPS=0.02
%env DYN_TAU=1
%env TAU_LO=0.55
%env TAU_HI=0.90
%env CONF_HI=0.65
%env CONF_LO=0.40

%env SPLIT_VIEWS=0
%env QA_WORD_MAX=30000
%env EXP_WORD_MAX=15000
%env EXP_CHAR_MAX=25000
%env W_QA_W=1.0
%env W_EXP_W=1.0
%env W_EXP_C=0.32

%env HIER=1
%env HIER_BETA=0.6

%env FAMILY_GUARD=1
%env FAMILY_MODE=hard
%env FAMILY_EPS=0.02

%env SAVE_PROBS=1
%env SAVE_TOPK=25

# ×�×•×¤×¦×™×•× ×œ×™ ×œ×‘×œ× ×“ ×¢×� ×�×•×“×œ ×—×™×¦×•× ×™ ×©×©×�×¨ labels.json+probs.npy
%env BLEND2=0
%env BLEND2_DIR=
%env BLEND2_WEIGHT=0.60



!python map_lr_map.py


