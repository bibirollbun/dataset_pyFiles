import os, re, math, gc, random, string
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Toggles and small HPs
USE_SWAP_AUG = False      # Set True to train with swap-augmentation (small extra boost sometimes)
FF_MIN_DF_RARE = 1        # Rare-token threshold (1 worked best)
LR_C = 1.0                # Logistic Regression C
SVC_C = 1.0     


_FLAT_RE = re.compile(r"^(.+)_file_([12])\.txt$")

def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read().strip()

def read_texts_from_dir(dir_path: str) -> pd.DataFrame:
    """
    Reads pairs (file_1.txt, file_2.txt) from a directory and returns DataFrame:
      columns: ['id', 'file_1', 'file_2'], index: 'id'
    Supports nested and flat layouts.
    """
    rows = []
    nested_found = False

    # Try nested layout
    for folder_name in sorted(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder_name)
        f1 = os.path.join(folder_path, "file_1.txt")
        f2 = os.path.join(folder_path, "file_2.txt")
        if os.path.isdir(folder_path) and os.path.isfile(f1) and os.path.isfile(f2):
            nested_found = True
            try:
                text1 = _read_text_file(f1)
                text2 = _read_text_file(f2)
                pair_id = folder_name
                rows.append({"id": pair_id, "file_1": text1, "file_2": text2})
            except Exception as e:
                print(f"Warning: Skipping directory {folder_name}: {e}")

    # Flat layout if nothing found
    if not rows:
        buckets = {}
        for name in os.listdir(dir_path):
            m = _FLAT_RE.match(name)
            if not m:
                continue
            pid, which = m.group(1), m.group(2)
            buckets.setdefault(pid, {})
            buckets[pid][which] = os.path.join(dir_path, name)

        for pid, files in buckets.items():
            if "1" in files and "2" in files:
                try:
                    text1 = _read_text_file(files["1"])
                    text2 = _read_text_file(files["2"])
                    rows.append({"id": pid, "file_1": text1, "file_2": text2})
                except Exception as e:
                    print(f"Warning: Skipping id {pid}: {e}")

    if not rows:
        raise RuntimeError(f"No valid (file_1, file_2) pairs found under {dir_path}")

    df = pd.DataFrame(rows)
    try:
        df["_id_num"] = df["id"].astype(int)
        df = df.sort_values("_id_num").drop(columns=["_id_num"])
    except Exception:
        df = df.sort_values("id")
    df = df.set_index("id")
    print(f"Found {len(df)} pairs in: {dir_path} (nested={nested_found})")
    return df



def _normalize_id_int(x) -> int:
    """
    Normalize id-like strings by extracting trailing digits and returning an integer.
    Examples: '0001'->1, 'pair_0012'->12, 34->34
    """
    s = str(x)
    m = re.search(r"(\d+)$", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    try:
        return int(s)
    except Exception:
        raise ValueError(f"Cannot normalize id to int: {x!r}")

def load_train(DATA_DIR: str) -> pd.DataFrame:
    """
    Load training pairs and merge with train.csv on normalized integer id.
    Returns: ['id', 'text1', 'text2', 'real_text_id']
    """
    train_dir = os.path.join(DATA_DIR, "train")
    train_csv = os.path.join(DATA_DIR, "train.csv")

    df_texts = read_texts_from_dir(train_dir).reset_index()
    df_texts = df_texts.rename(columns={"file_1": "text1", "file_2": "text2"})
    df_texts["norm_id_int"] = df_texts["id"].apply(_normalize_id_int)

    df_csv = pd.read_csv(train_csv)
    df_csv["norm_id_int"] = df_csv["id"].apply(_normalize_id_int)
    df_csv["real_text_id"] = df_csv["real_text_id"].astype(int)

    df = df_csv.merge(df_texts, on="norm_id_int", how="inner", suffixes=("_csv", "_dir"))
    df = df.rename(columns={"id_csv": "id"})
    df["id"] = df["id"].astype(str)
    df = df[["id", "text1", "text2", "real_text_id"]].reset_index(drop=True)

    print(f"Loaded train: csv_ids={df_csv['id'].nunique()} dir_ids={df_texts['id'].nunique()} merged={len(df)}")
    return df


def load_test(DATA_DIR: str) -> pd.DataFrame:
    """
    Load test pairs. Normalizes ids to integer then string for submission.
    Returns: ['id', 'text1', 'text2']
    """
    test_dir = os.path.join(DATA_DIR, "test")
    df_texts = read_texts_from_dir(test_dir).reset_index()
    df_texts = df_texts.rename(columns={"file_1": "text1", "file_2": "text2"})
    df_texts["id_int"] = df_texts["id"].apply(_normalize_id_int)
    df_texts = df_texts.sort_values("id_int")
    df_texts["id"] = df_texts["id_int"].astype(str)
    df_texts = df_texts[["id", "text1", "text2"]].reset_index(drop=True)
    print(f"Loaded test: dir_ids={len(df_texts)}")
    return df_texts

def find_competition_dir() -> str:
    base = "/kaggle/input"
    if os.path.isdir(base):
        candidates = []
        for name in os.listdir(base):
            d = os.path.join(base, name)
            if os.path.isdir(d):
                if os.path.isfile(os.path.join(d, "train.csv")) and \
                   os.path.isdir(os.path.join(d, "train")) and \
                   os.path.isdir(os.path.join(d, "test")):
                    candidates.append(d)
        if candidates:
            candidates.sort(key=lambda p: ("impostor" not in p.lower(), p.lower()))
            return candidates[0]
    if os.path.isfile("train.csv") and os.path.isdir("train") and os.path.isdir("test"):
        return "."
    fallback = "/kaggle/input/fake-or-real-the-impostor-hunt/data"
    if os.path.isdir(fallback):
        return fallback
    raise FileNotFoundError("Could not locate competition directory. Please set DATA_DIR manually.")



DATA_DIR = find_competition_dir()
print("Using DATA_DIR:", DATA_DIR)

train_df = load_train(DATA_DIR)
test_df = load_test(DATA_DIR)

print("Train size:", len(train_df), "Test size:", len(test_df))
display(train_df.head(3)[["id", "real_text_id"]])


STOPWORDS = set("""
a an and are as at be by for from has have in is it its of on or that the this to was were will with about into over under between among such not no nor but if then than so very more most less least few many much each other only same own 
he she they we you i him her them us their our your my mine yours ours theirs his hers itself themselves ourselves yourself yourselves
could should would may might can cannot can't don't didn't doesn't isn't aren't wasn't weren't won't
""".split())

WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
SENT_SPLIT_RE = re.compile(r"[.!?]+")

UNITS_RE = re.compile(r"\b(km|m|cm|mm|km/s|m/s|kg|g|mg|K|C|°C|°K|GHz|MHz|kHz|%|AU|mbar|Pa|kPa|MPa|W|kW|MW|GW|nm|µm|um|deg)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

HEDGES = set("approximately roughly about around likely possibly reportedly suggests suggest may might could appears appear generally often typically sometimes somewhat relatively largely arguably presumably potentially".split())
ASSERTIVES = set("clearly undoubtedly definitely certainly conclusively unequivocally demonstrates demonstrate shows show proves prove".split())



def count_syllables(word: str) -> int:
    w = word.lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w: return 0
    groups = re.findall(r"[aeiouy]+", w)
    count = len(groups)
    if w.endswith("e") and not w.endswith("le") and count > 1:
        count -= 1
    return max(1, count)

def readability_stats(text: str) -> Dict[str, float]:
    words = WORD_RE.findall(text)
    sentences = [s for s in SENT_SPLIT_RE.split(text) if s.strip()]
    if not words or not sentences:
        return {"fk_grade": 0.0, "fre": 0.0, "smog": 0.0}
    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = sum(count_syllables(w) for w in words)
    fk_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    fre = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    poly = sum(1 for w in words if count_syllables(w) >= 3)
    smog = 1.0430 * math.sqrt(max(poly, 1) * (30.0 / max(total_sentences, 1))) + 3.1291
    return {"fk_grade": fk_grade, "fre": fre, "smog": smog}

def char_entropy(text: str) -> float:
    if not text: return 0.0
    counts = Counter(text)
    total = sum(counts.values())
    ent = 0.0
    for _, k in counts.items():
        p = k / total
        ent -= p * math.log2(p)
    return ent

def text_stats(text: str) -> Dict[str, float]:
    tokens = WORD_RE.findall(text)
    lower = [t.lower() for t in tokens]
    n_tokens = len(tokens)
    unique = len(set(lower))
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    spaces = sum(ch.isspace() for ch in text)
    punct = sum(ch in string.punctuation for ch in text)
    exclam = text.count("!")
    quest = text.count("?")
    colon = text.count(":")
    semi = text.count(";")
    quotes_straight = text.count('"') + text.count("'")
    quotes_smart = text.count("“") + text.count("”") + text.count("‘") + text.count("’")
    paren = text.count("(") + text.count(")")
    bracket = text.count("[") + text.count("]")
    brace = text.count("{") + text.count("}")
    caps_initial = sum(1 for t in tokens if t[:1].isupper())
    all_caps = sum(1 for t in tokens if len(t) >= 2 and t.isupper())
    stop = sum(1 for t in lower if t in STOPWORDS)
    years = len(YEAR_RE.findall(text))
    units = len(UNITS_RE.findall(text))

    sents = [s for s in SENT_SPLIT_RE.split(text) if s.strip()]
    sent_lens = [len(WORD_RE.findall(s)) for s in sents]
    avg_sent = np.mean(sent_lens) if sent_lens else 0.0
    std_sent = np.std(sent_lens) if len(sent_lens) > 1 else 0.0
    n_sents = len(sents)
    q_sents = sum(1 for s in re.split(r"(?<=[.!?])\s+", text) if s.strip().endswith("?"))
    e_sents = sum(1 for s in re.split(r"(?<=[.!?])\s+", text) if s.strip().endswith("!"))

    def max_repeat_ngram(tokens, n):
        if len(tokens) < n: return 0
        grams = [" ".join(tokens[i:i+n]).lower() for i in range(len(tokens)-n+1)]
        c = Counter(grams)
        most = c.most_common(1)[0][1] if c else 0
        return most
    rep2 = max_repeat_ngram(tokens, 2)
    rep3 = max_repeat_ngram(tokens, 3)

    hedges = sum(1 for t in lower if t in HEDGES)
    assertv = sum(1 for t in lower if t in ASSERTIVES)

    read = readability_stats(text)
    ent = char_entropy(text)

    total_chars = len(text) if len(text) > 0 else 1
    feat = {
        "tok_count": n_tokens,
        "type_token_ratio": (unique / n_tokens) if n_tokens else 0.0,
        "stopword_ratio": (stop / n_tokens) if n_tokens else 0.0,
        "letters_per_char": letters / total_chars,
        "digits_per_char": digits / total_chars,
        "punct_per_char": punct / total_chars,
        "spaces_per_char": spaces / total_chars,
        "exclam_count": exclam,
        "quest_count": quest,
        "colon_count": colon,
        "semi_count": semi,
        "quotes_straight": quotes_straight,
        "quotes_smart": quotes_smart,
        "paren_count": paren,
        "bracket_count": bracket,
        "brace_count": brace,
        "caps_initial_ratio": (caps_initial / n_tokens) if n_tokens else 0.0,
        "all_caps_ratio": (all_caps / n_tokens) if n_tokens else 0.0,
        "year_count": years,
        "unit_count": units,
        "avg_sent_len": avg_sent,
        "std_sent_len": std_sent,
        "sent_count": n_sents,
        "q_sent_count": q_sents,
        "e_sent_count": e_sents,
        "hedge_count": hedges,
        "assert_count": assertv,
        "rep2_max": rep2,
        "rep3_max": rep3,
        "fk_grade": read["fk_grade"],
        "flesch_reading_ease": read["fre"],
        "smog": read["smog"],
        "char_entropy": ent,
        "text_len": len(text)
    }
    return feat



def build_doc_freq(tokens_list: List[List[str]]) -> Counter:
    df = Counter()
    for toks in tokens_list:
        df.update(set(t.lower() for t in toks))
    return df

def rare_token_rate(text: str, df: Counter, min_df: int = 1) -> float:
    toks = [t.lower() for t in WORD_RE.findall(text)]
    if not toks: return 0.0
    rare = sum(1 for t in toks if df.get(t, 0) <= min_df)
    return rare / len(toks)

def normalize_char(c: str) -> str:
    if c.isalpha(): return c.lower()
    if c.isdigit(): return c
    if c in " .,!?:;()[]{}'\"-/%":
        return c
    return "_"

def train_char_ngram_lm(texts: List[str], n: int = 5, alpha: float = 0.5):
    ngram_counts = Counter()
    ctx_counts = Counter()
    vocab = set()
    for txt in texts:
        for ch in txt:
            vocab.add(normalize_char(ch))
    vocab = sorted(vocab)
    if "_" not in vocab: vocab.append("_")
    for txt in texts:
        seq = [normalize_char(ch) for ch in txt]
        seq = ["^"]*(n-1) + seq + ["$"]
        for i in range(n-1, len(seq)):
            ctx = tuple(seq[i-n+1:i])
            gram = tuple(seq[i-n+1:i+1])
            ngram_counts[gram] += 1
            ctx_counts[ctx] += 1
    return ngram_counts, ctx_counts, vocab, alpha

def avg_log_prob(text: str, ngram_counts, ctx_counts, vocab: List[str], n: int = 5, alpha: float = 0.5) -> float:
    V = len(vocab)
    seq = [normalize_char(ch) for ch in text]
    seq = ["^"]*(n-1) + seq + ["$"]
    ll = 0.0
    steps = 0
    for i in range(n-1, len(seq)):
        ctx = tuple(seq[i-n+1:i])
        gram = tuple(seq[i-n+1:i+1])
        num = ngram_counts.get(gram, 0) + alpha
        den = ctx_counts.get(ctx, 0) + alpha * V
        ll += math.log(num / den)
        steps += 1
    return ll / max(steps, 1)

class FoldFeatureBuilder:
    def __init__(self, min_df_rare=FF_MIN_DF_RARE, tfidf_min_df=2, tfidf_max_features=60000):
        self.min_df_rare = min_df_rare
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_features = tfidf_max_features
        self.df_counter = None
        self.tfidf = None
        self.lm_real = None
        self.lm_fake = None

    def fit(self, texts1_train: List[str], texts2_train: List[str], real_ids_train: List[int]):
        toks_all = [WORD_RE.findall(t) for t in texts1_train] + [WORD_RE.findall(t) for t in texts2_train]
        self.df_counter = build_doc_freq(toks_all)

        self.tfidf = TfidfVectorizer(
            analyzer='word', ngram_range=(1,2), min_df=self.tfidf_min_df,
            max_features=self.tfidf_max_features, strip_accents='unicode'
        )
        self.tfidf.fit(texts1_train + texts2_train)

        # build char LM per class
        real_texts, fake_texts = [], []
        for t1, t2, rid in zip(texts1_train, texts2_train, real_ids_train):
            if int(rid) == 1:
                real_texts.append(t1); fake_texts.append(t2)
            else:
                real_texts.append(t2); fake_texts.append(t1)
        self.lm_real = train_char_ngram_lm(real_texts, n=5, alpha=0.5)
        self.lm_fake = train_char_ngram_lm(fake_texts, n=5, alpha=0.5)
        return self

    def transform_pair(self, t1: str, t2: str) -> Dict[str, float]:
        rare1 = rare_token_rate(t1, self.df_counter, self.min_df_rare)
        rare2 = rare_token_rate(t2, self.df_counter, self.min_df_rare)

        v1 = self.tfidf.transform([t1])
        v2 = self.tfidf.transform([t2])
        sim = cosine_similarity(v1, v2)[0, 0]

        (ng_r, ctx_r, vocab_r, a_r) = self.lm_real
        (ng_f, ctx_f, vocab_f, a_f) = self.lm_fake
        s1 = avg_log_prob(t1, ng_r, ctx_r, vocab_r, n=5, alpha=a_r) - avg_log_prob(t1, ng_f, ctx_f, vocab_f, n=5, alpha=a_f)
        s2 = avg_log_prob(t2, ng_r, ctx_r, vocab_r, n=5, alpha=a_r) - avg_log_prob(t2, ng_f, ctx_f, vocab_f, n=5, alpha=a_f)

        return {
            "rare_rate_delta": (rare2 - rare1),
            "tfidf_cosine_sim": sim,
            "char_lm_delta": (s2 - s1),
        }

BASE_FEATURE_KEYS = [
    "tok_count","type_token_ratio","stopword_ratio","letters_per_char","digits_per_char","punct_per_char","spaces_per_char",
    "exclam_count","quest_count","colon_count","semi_count","quotes_straight","quotes_smart","paren_count","bracket_count","brace_count",
    "caps_initial_ratio","all_caps_ratio","year_count","unit_count","avg_sent_len","std_sent_len","sent_count","q_sent_count","e_sent_count",
    "hedge_count","assert_count","rep2_max","rep3_max","fk_grade","flesch_reading_ease","smog","char_entropy","text_len"
]



def build_pair_features_static(t1: str, t2: str) -> Dict[str, float]:
    f1 = text_stats(t1)
    f2 = text_stats(t2)
    feats = {}
    # directional deltas f2 - f1
    for k in BASE_FEATURE_KEYS:
        feats[f"delta_{k}"] = f2[k] - f1[k]
    # symmetric overlaps
    set1 = set(w.lower() for w in WORD_RE.findall(t1))
    set2 = set(w.lower() for w in WORD_RE.findall(t2))
    inter = len(set1 & set2)
    union = len(set1 | set2) if set1 | set2 else 1
    feats["jaccard_tokens"] = inter / union

    caps1 = set(w for w in WORD_RE.findall(t1) if w[:1].isupper())
    caps2 = set(w for w in WORD_RE.findall(t2) if w[:1].isupper())
    inter_c = len(caps1 & caps2)
    union_c = len(caps1 | caps2) if caps1 | caps2 else 1
    feats["caps_overlap"] = inter_c / union_c

    nums1 = set(re.findall(r"\b\d+(?:\.\d+)?\b", t1))
    nums2 = set(re.findall(r"\b\d+(?:\.\d+)?\b", t2))
    inter_n = len(nums1 & nums2)
    union_n = len(nums1 | nums2) if nums1 | nums2 else 1
    feats["numeric_jaccard"] = inter_n / union_n
    feats["delta_numeric_count"] = len(nums2) - len(nums1)

    # robust pair stats
    feats["len_log_ratio"] = math.log1p(len(t2)) - math.log1p(len(t1))
    feats["abs_delta_year_count"] = abs(f2["year_count"] - f1["year_count"])
    feats["abs_delta_unit_count"] = abs(f2["unit_count"] - f1["unit_count"])
    feats["abs_delta_fk_grade"] = abs(f2["fk_grade"] - f1["fk_grade"])
    feats["abs_delta_avg_sent_len"] = abs(f2["avg_sent_len"] - f1["avg_sent_len"])
    return feats

def assemble_feature_matrix_fold(df_fold_train: pd.DataFrame, df_fold_valid: pd.DataFrame, df_test: pd.DataFrame):
    builder = FoldFeatureBuilder(min_df_rare=FF_MIN_DF_RARE, tfidf_min_df=2, tfidf_max_features=60000)
    builder.fit(
        texts1_train=df_fold_train["text1"].tolist(),
        texts2_train=df_fold_train["text2"].tolist(),
        real_ids_train=df_fold_train["real_text_id"].tolist()
    )

    def rows_to_matrices(df_: pd.DataFrame):
        feat_rows = []
        feat_rows_sw = []
        for _, r in df_.iterrows():
            t1, t2 = r["text1"], r["text2"]

            feats = build_pair_features_static(t1, t2)
            feats.update(builder.transform_pair(t1, t2))
            feat_rows.append(feats)

            feats_sw = build_pair_features_static(t2, t1)
            feats_sw.update(builder.transform_pair(t2, t1))
            feat_rows_sw.append(feats_sw)

        X = pd.DataFrame(feat_rows)
        X_sw = pd.DataFrame(feat_rows_sw)
        return X, X_sw

    X_tr, X_tr_sw = rows_to_matrices(df_fold_train)
    X_va, X_va_sw = rows_to_matrices(df_fold_valid)
    X_te, X_te_sw = rows_to_matrices(df_test)

    cols = sorted(list(set(X_tr.columns) | set(X_tr_sw.columns) | set(X_va.columns) | set(X_va_sw.columns) | set(X_te.columns) | set(X_te_sw.columns)))
    return (
        X_tr.reindex(columns=cols).fillna(0.0),
        X_tr_sw.reindex(columns=cols).fillna(0.0),
        X_va.reindex(columns=cols).fillna(0.0),
        X_va_sw.reindex(columns=cols).fillna(0.0),
        X_te.reindex(columns=cols).fillna(0.0),
        X_te_sw.reindex(columns=cols).fillna(0.0),
        cols
    )



def train_and_stack(train_df: pd.DataFrame, test_df: pd.DataFrame, n_splits=5, seed=SEED):
    y = (train_df["real_text_id"].values == 1).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    model_defs = [
        ("logreg", Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(C=LR_C, max_iter=2000, solver="lbfgs"))
        ])),
        ("linsvc", "CalibratedLinearSVC"),
    ]

    model_names = [name for name, _ in model_defs]
    oof_preds = {name: np.zeros(len(train_df), dtype=float) for name in model_names}
    test_preds = {name: np.zeros(len(test_df), dtype=float) for name in model_names}
    fold_acc = defaultdict(list)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y), 1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
        df_va = train_df.iloc[va_idx].reset_index(drop=True)

        X_tr, X_tr_sw, X_va, X_va_sw, X_te, X_te_sw, cols = assemble_feature_matrix_fold(df_tr, df_va, test_df)

        if USE_SWAP_AUG:
            # Train-time swap augmentation
            y_tr = y[tr_idx]
            X_tr_aug = pd.concat([X_tr, X_tr_sw], axis=0).reset_index(drop=True)
            y_tr_aug = np.concatenate([y_tr, 1 - y_tr])
        else:
            X_tr_aug = X_tr
            y_tr_aug = y[tr_idx]

        # 1) Logistic Regression with TTA
        name, pipe = model_defs[0]
        pipe.fit(X_tr_aug, y_tr_aug)

        proba_va = pipe.predict_proba(X_va)[:, 1]
        proba_va_sw = pipe.predict_proba(X_va_sw)[:, 1]
        proba_va_final = 0.5 * (proba_va + (1.0 - proba_va_sw))

        proba_te = pipe.predict_proba(X_te)[:, 1]
        proba_te_sw = pipe.predict_proba(X_te_sw)[:, 1]
        proba_te_final = 0.5 * (proba_te + (1.0 - proba_te_sw))

        oof_preds[name][va_idx] = proba_va_final
        test_preds[name] += proba_te_final / n_splits
        acc = accuracy_score(y[va_idx], (proba_va_final >= 0.5).astype(int))
        fold_acc[name].append(acc)
        print(f"{name} fold acc (TTA{' + swap-aug' if USE_SWAP_AUG else ''}): {acc:.4f}")

        # 2) LinearSVC + calibration with TTA
        name = model_defs[1][0]
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr_s   = scaler.fit_transform(X_tr_aug)
        X_va_s   = scaler.transform(X_va)
        X_va_sw_s= scaler.transform(X_va_sw)
        X_te_s   = scaler.transform(X_te)
        X_te_sw_s= scaler.transform(X_te_sw)

        svc = LinearSVC(C=SVC_C, random_state=seed)
        svc.fit(X_tr_s, y_tr_aug)
        calib = CalibratedClassifierCV(svc, method="sigmoid", cv="prefit")
        calib.fit(X_va_s, y[va_idx])

        proba_va = calib.predict_proba(X_va_s)[:, 1]
        proba_va_sw = calib.predict_proba(X_va_sw_s)[:, 1]
        proba_va_final = 0.5 * (proba_va + (1.0 - proba_va_sw))

        proba_te = calib.predict_proba(X_te_s)[:, 1]
        proba_te_sw = calib.predict_proba(X_te_sw_s)[:, 1]
        proba_te_final = 0.5 * (proba_te + (1.0 - proba_te_sw))

        oof_preds[name][va_idx] = proba_va_final
        test_preds[name] += proba_te_final / n_splits
        acc = accuracy_score(y[va_idx], (proba_va_final >= 0.5).astype(int))
        fold_acc[name].append(acc)
        print(f"{name} fold acc (TTA{' + swap-aug' if USE_SWAP_AUG else ''}): {acc:.4f}")

        del X_tr, X_tr_sw, X_va, X_va_sw, X_te, X_te_sw, X_tr_aug, y_tr_aug
        gc.collect()

    # OOF summary
    for name in model_names:
        acc = accuracy_score(y, (oof_preds[name] >= 0.5).astype(int))
        print(f"\n{name} OOF accuracy (TTA{' + swap-aug' if USE_SWAP_AUG else ''}): {acc:.4f} | per-fold: {', '.join(f'{a:.4f}' for a in fold_acc[name])}")

    # Also compute simple L2 stacker for reference (not used for final submission)
    X_meta = np.vstack([oof_preds[name] for name in model_names]).T
    X_meta_test = np.vstack([test_preds[name] for name in model_names]).T
    stacker = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    stacker.fit(X_meta, y)
    meta_oof = stacker.predict_proba(X_meta)[:, 1]
    meta_acc = accuracy_score(y, (meta_oof >= 0.5).astype(int))
    print(f"\nL2 stacker OOF accuracy (TTA{' + swap-aug' if USE_SWAP_AUG else ''}): {meta_acc:.4f}")

    meta_test_proba = stacker.predict_proba(X_meta_test)[:, 1]
    return model_names, oof_preds, test_preds, meta_test_proba, meta_acc, meta_oof



model_names, oof_preds, test_preds, meta_test_proba, meta_acc, meta_oof = train_and_stack(train_df, test_df, n_splits=5, seed=SEED)

# Use LogReg predictions for threshold tuning
y_true = (train_df["real_text_id"].values == 1).astype(int)
p_oof = oof_preds["logreg"]
p_test = test_preds["logreg"]

grid = np.linspace(0.3, 0.7, 81)
best_t, best_a = 0.5, -1.0
for t in grid:
    a = accuracy_score(y_true, (p_oof >= t).astype(int))
    if a > best_a:
        best_a, best_t = a, t
print(f"Single-run (LogReg-only) best OOF threshold: {best_t:.3f} | OOF acc: {best_a:.4f}")

sub_single = pd.DataFrame({
    "id": test_df["id"].astype(str).values,
    "prob_text1_real": p_test
})
sub_single["real_text_id"] = np.where(sub_single["prob_text1_real"] >= best_t, 1, 2)
sub_single = sub_single[["id", "real_text_id"]].sort_values("id").reset_index(drop=True)
sub_single.to_csv("submission_single.csv", index=False)
print("Saved submission_single.csv")
display(sub_single.head())


# %% [code]
SEEDS = [42, 1337, 2025, 2718, 31415]

all_lr_test = []
all_lr_oof = []

for s in SEEDS:
    print(f"\n=== Seed {s} (LogReg-only) ===")
    mn, oof_p, test_p, _, _, _ = train_and_stack(train_df, test_df, n_splits=5, seed=s)
    assert "logreg" in oof_p and "logreg" in test_p
    all_lr_oof.append(oof_p["logreg"])
    all_lr_test.append(test_p["logreg"])

lr_oof_avg = np.mean(np.vstack(all_lr_oof), axis=0)
lr_test_avg = np.mean(np.vstack(all_lr_test), axis=0)

best_t, best_a = 0.5, -1.0
for t in np.linspace(0.3, 0.7, 81):
    a = accuracy_score(y_true, (lr_oof_avg >= t).astype(int))
    if a > best_a:
        best_a, best_t = a, t

print(f"[LR-only ensemble] Best OOF threshold: {best_t:.3f} | OOF acc: {best_a:.4f}")

sub = pd.DataFrame({
    "id": test_df["id"].astype(str).values,
    "prob_text1_real": lr_test_avg
})
sub["real_text_id"] = np.where(sub["prob_text1_real"] >= best_t, 1, 2)
sub = sub[["id", "real_text_id"]].sort_values("id").reset_index(drop=True)
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv (LogReg-only seed ensemble)")
display(sub.head())




