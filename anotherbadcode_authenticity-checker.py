import gc
gc.collect()

!pip install -q textstat
!pip install -q pyspellchecker
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm
import random
from sklearn.utils import shuffle
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, log_loss, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import string
from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import shap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

!pip install -q langdetect
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))
from spellchecker import SpellChecker
spell_checker = SpellChecker()

import spacy
nlp = spacy.load("en_core_web_sm")
import textstat
from collections import Counter
from math import log2
!pip install -q langdetect
from langdetect import detect, DetectorFactory

from sklearn.feature_extraction.text import TfidfVectorizer
import contextlib
from typing import List
import zlib
from collections import Counter
import unicodedata as ud
import torch
from huggingface_hub import snapshot_download
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_global_seed(42)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

train = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
test = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')

model_path = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformer(model_path, local_files_only=True)

deberta_path = snapshot_download(
    repo_id="microsoft/deberta-v3-base",
    allow_patterns=[
        "spm.model",
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "config.json",
        "pytorch_model.bin"
    ]
)

encoder = AutoModel.from_pretrained(
    deberta_path,
    local_files_only=True,
    trust_remote_code=True
)

cfg = AutoConfig.from_pretrained(
    deberta_path,
    local_files_only=True,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    deberta_path,
    local_files_only=True,
    trust_remote_code=True,
    use_fast=False,
)


train.head(3)


class EmbeddingAdapter(nn.Module):
    def __init__(self, base_model, hidden_dim=384, adapter_dim=128):
        super().__init__()
        self.base = base_model
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, text_emb):
        adapted = self.adapter(text_emb)
        logits = self.classifier(adapted)
        return logits, adapted

class EmbeddingDataset(Dataset):
    def __init__(self, texts, labels, embedder):
        self.texts = texts
        self.labels = labels
        self.embedder = embedder

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        emb = self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def train_adapter(train_df, embedder, epochs=5, lr=1e-4, batch_size=32, device="cuda"):
    dataset = EmbeddingDataset(train_df["answer"].tolist(), train_df["is_cheating"].values, embedder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    adapter_model = EmbeddingAdapter(embedder, hidden_dim=384).to(device)
    opt = torch.optim.AdamW(adapter_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    adapter_model.train()
    for epoch in range(epochs):
        total = 0
        total_loss = 0
        for emb, y in loader:
            emb, y = emb.to(device), y.to(device)
            logits, _ = adapter_model(emb)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += y.size(0)
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
    return adapter_model

def generate_tuned_embeddings(df, embedder, adapter_model, device="cuda"):
    adapter_model.eval()
    embs = []
    with torch.no_grad():
        for text in df["answer"]: # .tolist()
            emb = embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            emb = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            _, adapted = adapter_model(emb)
            embs.append(adapted.squeeze(0).cpu().numpy())
    return np.array(embs)



train_raw = train.copy()
test_raw  = test.copy()

train_raw["combined_text"] = (
    train_raw["topic"].fillna("") + " [SEP] " + train_raw["answer"].fillna("")
).astype(str)

test_raw["combined_text"] = (
    test_raw["topic"].fillna("") + " [SEP] " + test_raw["answer"].fillna("")
).astype(str)

train_for_embed = pd.DataFrame({
    "answer": train_raw["combined_text"],
    "is_cheating": train_raw["is_cheating"]
})

test_for_embed = pd.DataFrame({
    "answer": test_raw["combined_text"]
})

adapter_model = train_adapter(
    train_df=train_for_embed,
    embedder=embedder,
    epochs=10,
    lr=1e-4,
    batch_size=8
)

train_emb = generate_tuned_embeddings(
    train_for_embed,
    embedder,
    adapter_model
)

test_emb = generate_tuned_embeddings(
    test_for_embed,
    embedder,
    adapter_model
)

pca = PCA(n_components=0.99, svd_solver="full")
train_pca = pca.fit_transform(train_emb)
test_pca = pca.transform(test_emb)

pc_names = [f"embed_pca_{i}" for i in range(train_pca.shape[1])]
print(f"PCA reduced embeddings to {len(pc_names)} dims.")

train_emb_df = pd.DataFrame(train_pca, columns=pc_names, index=train_raw.index)
test_emb_df  = pd.DataFrame(test_pca,  columns=pc_names, index=test_raw.index)


train = pd.concat([train.reset_index(drop=True),
                            train_emb_df.reset_index(drop=True)], axis=1)

test = pd.concat(
    [test.reset_index(drop=True),  test_emb_df.reset_index(drop=True)],
    axis=1
)

print("Final train shape:", train.shape)
print("Final test shape:", test.shape)


# Tokenizer that keeps symbols/emojis and doesn't lose weird ones # Splits on whitespace; then further splits into words or single non-space chars.
TOKEN_RE = re.compile(r"\w+|[^\s\w]", flags=re.UNICODE)

# Common ASCII punctuations
ASCII_PUNCT = set(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}

def tokenize_keep_symbols(text: str):
    text = text or ""
    return TOKEN_RE.findall(text)

def has_non_ascii(token: str) -> bool:
    try:
        token.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True

def has_zero_width(token: str) -> bool:
    return any(z in token for z in ZERO_WIDTH)

def unicode_categories(token: str):
    return {ud.category(ch) for ch in token}

def mixed_scripts(token: str) -> bool:
    # different Unicode "script-ish" ranges # Latin letters vs everything else
    latin = any('LATIN' in ud.name(ch, '') for ch in token if ch.isalpha())
    non_latin_alpha = any(ch.isalpha() and 'LATIN' not in ud.name(ch, '') for ch in token)
    return latin and non_latin_alpha

def weird_ratio(token: str) -> float:
    # proportion of chars that are not ascii letters/digits or common ASCII punct
    if not token:
        return 0.0
    weird = sum(
        not (ch.isascii() and (ch.isalnum() or ch in ASCII_PUNCT)) 
        for ch in token
    )
    return weird / len(token)

def looks_weird(token: str) -> bool:
    if token.strip() == "":
        return False
    if has_zero_width(token):
        return True
    if has_non_ascii(token):
        return True
    if mixed_scripts(token):
        return True
    if weird_ratio(token) >= 0.05:  # 5% or more unusual chars
        return True
    if re.search(r"[^\w\s].*[A-Za-z0-9]|[A-Za-z0-9].*[^\w\s]", token) and any(not ch.isascii() for ch in token):
        return True
    return False

def is_plain_english_word(token: str) -> bool:
    return token.isascii() and token.isalpha() and len(token) > 2

def log_odds_with_prior(counts_a, counts_b, prior_alpha=0.01):

    vocab = set(counts_a) | set(counts_b)
    V = len(vocab)
    alpha = prior_alpha

    ya = np.array([counts_a.get(t, 0) for t in vocab], dtype=np.float64)
    yb = np.array([counts_b.get(t, 0) for t in vocab], dtype=np.float64)

    Na = ya.sum()
    Nb = yb.sum()

    pa = (ya + alpha) / (Na + alpha * V)
    pb = (yb + alpha) / (Nb + alpha * V)

    delta = np.log(pa) - np.log(1 - pa) - (np.log(pb) - np.log(1 - pb))
    var = 1/(ya + alpha) + 1/(yb + alpha)
    z = delta / np.sqrt(var)

    df = pd.DataFrame({
        "token": list(vocab),
        "count_real": ya.astype(int),
        "count_fake": yb.astype(int),
        "log_odds_real_vs_fake": delta,
        "z_real_vs_fake": z
    })
    return df.sort_values("z_real_vs_fake", ascending=False)


def find_discriminative_weird_tokens(data: pd.DataFrame,
                                     min_count: int = 3,
                                     top_n: int = 100,
                                     exclude_plain_english: bool = True) -> pd.DataFrame:
    """
    Identifies tokens that disproportionately occur in real vs. AI-generated (cheating) text.
    Returns : 
    pd.DataFrame : Table of discriminative weird tokens with stats and skew direction.
    """
    assert {"id","answer","is_cheating"}.issubset(data.columns), "Data must have id, answer, is_cheating"

    real_counts = Counter()
    fake_counts = Counter()
    weird_counts_real = Counter()
    weird_counts_fake = Counter()

    for _, row in data.iterrows():
        tokens = tokenize_keep_symbols(str(row["answer"]))
        lbl = int(row["is_cheating"])
        for tok in tokens:
            if lbl == 0:
                real_counts[tok] += 1
            else:
                fake_counts[tok] += 1
            if looks_weird(tok):
                if exclude_plain_english and is_plain_english_word(tok):
                    continue
                if lbl == 0:
                    weird_counts_real[tok] += 1
                else:
                    weird_counts_fake[tok] += 1

    # Filter by minimum frequency
    vocab_weird = {t for t, c in (weird_counts_real + weird_counts_fake).items() if c >= min_count}
    if len(vocab_weird) == 0:
        return pd.DataFrame(columns=["token","count_real","count_fake","log_odds_real_vs_fake","z_real_vs_fake","skew_to"])

    counts_real = {t: weird_counts_real[t] for t in vocab_weird}
    counts_fake = {t: weird_counts_fake[t] for t in vocab_weird}

    table = log_odds_with_prior(counts_real, counts_fake, prior_alpha=0.01)
    table["skew_to"] = np.where(table["z_real_vs_fake"] > 0, "real", "fake")
    table["abs_z"] = table["z_real_vs_fake"].abs()

    out = pd.concat([
        table.query("z_real_vs_fake > 0").head(top_n),
        table.query("z_real_vs_fake < 0").tail(top_n)
    ]).sort_values("z_real_vs_fake", ascending=False)

    out["freq_total"] = out["count_real"] + out["count_fake"]
    return out[["token","count_real","count_fake","freq_total","log_odds_real_vs_fake","z_real_vs_fake","skew_to"]]


weird_table = find_discriminative_weird_tokens(train, min_count=2, top_n=100)
weird_tokens = weird_table["token"].tolist()

weird_table.head(7)


def compute_entropy(text: str) -> float:
    if not text: return 0.0
    prob = [freq / len(text) for freq in Counter(text).values()]
    return -sum(p * log2(p) for p in prob)

def ngram_repetition(text, n=3):
    words = word_tokenize(text.lower())
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    count = Counter(ngrams)
    total = len(ngrams)
    repeated = sum(1 for v in count.values() if v > 1)
    return repeated / total if total > 0 else 0

def insert_space_in_compounds(text):
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # e.g., "highResolution" → "high Resolution"

def count_spelling_errors(words):
    misspelled = spell_checker.unknown(words)
    return len(misspelled)

def fit_tfidf_vectorizer(train_df: pd.DataFrame, text_col: str, max_features: int = 1000, ngram_range=(1, 2)) -> TfidfVectorizer:

    vectorizer = TfidfVectorizer(min_df=2, 
                                 max_df=0.9,
                                 sublinear_tf=True,
                                 stop_words='english',
                                 max_features=max_features, 
                                 ngram_range=ngram_range,
                                )
    vectorizer.fit(train_df[text_col].fillna(""))
    return vectorizer

def transform_tfidf_features(df: pd.DataFrame, vectorizer: TfidfVectorizer, text_col: str = "text") -> pd.DataFrame:

    tfidf_array = vectorizer.transform(df[text_col].fillna("")).toarray()
    tfidf_features = pd.DataFrame(tfidf_array, columns=[f"tfidf_{f}" for f in vectorizer.get_feature_names_out()])
    tfidf_features.reset_index(drop=True, inplace=True)
    return tfidf_features

def fit_multi_tfidf_vectorizers(train_df, vectorizer_specs):
    vec_dict = {}
    for vec, col, key in vectorizer_specs:
        vec.fit(train_df[col].fillna("").astype(str))
        vec_dict[key] = vec
        print(f"Fitted TF-IDF for {key} on '{col}' -> {len(vec.get_feature_names_out())} features")
    return vec_dict

def transform_multi_tfidf(df, vec_dict, vectorizer_specs):
    out = []
    for vec, col, key in vectorizer_specs:
        arr = vec_dict[key].transform(df[col].fillna("").astype(str)).toarray()
        cols = [f"{key}_{f}" for f in vec_dict[key].get_feature_names_out()]
        out.append(pd.DataFrame(arr, columns=cols, index=df.index))
    return pd.concat(out, axis=1)

def build_final_features(train_df, test_df):
    train_style = extract_text_features(train_df, text_col="answer")
    test_style  = extract_text_features(test_df,  text_col="answer")
    vec_dict = fit_multi_tfidf_vectorizers(train_df, word2vec_models)

    tfidf_train = transform_multi_tfidf(train_df, vec_dict, word2vec_models)
    tfidf_test  = transform_multi_tfidf(test_df,  vec_dict, word2vec_models)

    train_features = pd.concat([train_style.reset_index(drop=True), tfidf_train], axis=1)
    test_features  = pd.concat([test_style.reset_index(drop=True),  tfidf_test], axis=1)

    # print("Final Train shape:", train_features.shape)
    # print("Final Test shape:",  test_features.shape)

    return train_features, test_features


def extract_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:

    def detect_script_ratios(text: str) -> dict:
        total = len(text)
        if total == 0:
            return {"english_ratio": 0.0, "non_ascii_ratio": 0.0}
        english_count = len(re.findall(r'[a-zA-Z]', text))
        non_ascii_count = len(re.findall(r'[^\x00-\x7F]', text))
        return {
            "english_ratio": english_count / total,
            "non_ascii_ratio": non_ascii_count / total
        }

    def feature_row(text: str) -> dict:
        text = str(text)
        words = word_tokenize(text)
        word_count = len(words)
        num_chars = len(text)
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)
        avg_sentence_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        punct_count = sum(1 for c in text if c in string.punctuation)
        emdash_count = text.count("—")
        long_words = sum(1 for w in words if len(w) > 6)
        short_words = sum(1 for w in words if len(w) <= 3)
        stopword_count = sum(1 for w in words if w.lower() in stop_words)
        unique_words = len(set(words))
        upper_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())
        avg_word_len = np.mean([len(w) for w in words]) if word_count > 0 else 0
        ent = compute_entropy(text)
        spelling_errors = count_spelling_errors([w for w in words if w.isalpha()])
        type_token_ratio_sqrt = unique_words / (word_count ** 0.5) if word_count else 0
        script_ratios = detect_script_ratios(text)

        doc = nlp(text)
        ner_count = len(doc.ents)
        compression_ratio = len(zlib.compress(text.encode())) / len(text.encode()) if text else 1.0
        pos_counts = Counter([token.pos_ for token in doc])
        total_pos = sum(pos_counts.values()) or 1
        noun_ratio = pos_counts.get("NOUN", 0) / total_pos
        verb_ratio = pos_counts.get("VERB", 0) / total_pos
        adj_ratio = pos_counts.get("ADJ", 0) / total_pos
        adv_ratio = pos_counts.get("ADV", 0) / total_pos
        dep_depths = [len(list(token.ancestors)) for token in doc if token.head != token]
        avg_dep_depth = np.mean(dep_depths) if dep_depths else 0.0

        toks = [w.lower() for w in words]
        weird_token_count = sum(1 for w in toks if w in weird_tokens)
        weird_token_ratio = (weird_token_count / word_count) if word_count else 0.0

        return {
            'char_count': num_chars,
            'word_count': word_count,
            'sentence_count': num_sentences,
            'avg_word_length': avg_word_len,
            'avg_sentence_length': avg_sentence_len,
            'unique_word_count': unique_words,
            'ttr': unique_words / word_count if word_count else 0,
            'stopword_count': stopword_count,
            'stopword_ratio': stopword_count / word_count if word_count else 0,
            'punctuation_count': punct_count,
            'english_ratio': script_ratios["english_ratio"],
            'non_ascii_ratio': script_ratios["non_ascii_ratio"],
            'digit_count': digit_count,
            'uppercase_ratio': upper_count / num_chars if num_chars else 0,
            'long_word_count': long_words,
            'short_word_count': short_words,
            'type_token_ratio_sqrt': type_token_ratio_sqrt,
            'entropy': ent,
            'emdash_count': emdash_count,
            'ngram_repetition': ngram_repetition(text),
            'ner_count': ner_count,
            'spelling_errors': spelling_errors,
            'compression_ratio': compression_ratio,
            'noun_ratio': noun_ratio,
            'verb_ratio': verb_ratio,
            'adj_ratio': adj_ratio,
            'adv_ratio': adv_ratio,
            'weird_token_count': weird_token_count,
            'weird_token_ratio': weird_token_ratio,
        }

    features = df[text_col].apply(feature_row)
    return pd.concat([df.drop(columns=[text_col]), features.apply(pd.Series)], axis=1)



word2vec_models = [
    (TfidfVectorizer(analyzer='char', ngram_range=(1,3), max_features=256),
     'topic', 'tfidf_topic_char'),
    (TfidfVectorizer(analyzer='word', ngram_range=(1,3), stop_words='english', max_features=500),
     'topic', 'tfidf_topic_word'),
    (TfidfVectorizer(analyzer='char', ngram_range=(3,6), max_features=1000, min_df=2),
     'answer', 'tfidf_answer_char'),
    (TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words='english', max_features=3000, sublinear_tf=True),
     'answer', 'tfidf_answer_word'),
]

# vectorizer = fit_tfidf_vectorizer(train, text_col="answer", max_features=3000, ngram_range=(1,3))
# tfidf_train = transform_tfidf_features(train, vectorizer, text_col="answer")
# tfidf_test  = transform_tfidf_features(test, vectorizer, text_col="answer")

tfidf_train, tfidf_test = build_final_features(train, test)

train_style = extract_text_features(train, text_col="answer")
test_style  = extract_text_features(test, text_col="answer")

train_features = pd.concat([train_style.reset_index(drop=True), tfidf_train], axis=1)
test_features  = pd.concat([test_style.reset_index(drop=True), tfidf_test], axis=1)

print("Final Train shape:", train_features.shape)
print("Final Test shape:", test_features.shape)


def compute_shap_importance(model, X, model_name="Model", sample_size=200, plot_top_n=20):

    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X.copy()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"[{model_name}] SHAP computation failed: {e}")
        return None, None
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

    return mean_abs_shap, shap_values


def fit_kfold_and_predict_ensemble(X, y, X_test, seed=42, META = 'LRCV', verbose=True):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    X_pd, X_test_pd = X.copy(), X_test.copy()
    y_np = y.values if isinstance(y, pd.Series) else y
    scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

    # cat_features = [c for c in X_pd.columns if X_pd[c].dtype == "object"]
    cat_features = list(X_pd.select_dtypes(include=["object", "category"]).columns)

    X_xgb = X_pd.copy()
    X_test_xgb = X_test_pd.copy()
    for col in cat_features:
        X_xgb[col] = X_xgb[col].astype("category")
        X_test_xgb[col] = X_test_xgb[col].astype("category")

    oof_cat, oof_xgb = np.zeros(len(y_np)), np.zeros(len(y_np))
    test_cat, test_xgb = np.zeros(len(X_test_pd)), np.zeros(len(X_test_pd))
    oof_lgb, test_lgb = np.zeros(len(y_np)), np.zeros(len(X_test_pd))
    
    all_cat_shap = []
    all_xgb_shap = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_pd, y_np)):
        X_tr, X_va = X_pd.iloc[tr_idx], X_pd.iloc[va_idx]
        X_tr_xgb, X_va_xgb = X_xgb.iloc[tr_idx], X_xgb.iloc[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]

        if verbose:
            print(f"\n[Fold {fold+1}] : ")

        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_valid = lgb.Dataset(X_va, label=y_va, reference=lgb_train)
        lgb_params = {
                    "objective": "binary",
                    "metric": "auc",
                    "learning_rate": 0.05,
                    "num_leaves": 15,
                    "min_data_in_leaf": 10,
                    "feature_fraction": 0.7,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 3,
                    "lambda_l2": 2.0,
                    "seed": seed + fold,
                    "verbose": -1,
                    }

        callbacks = [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(0),
                    ]

        lgb_model = lgb.train(
                                lgb_params,
                                lgb_train,
                                num_boost_round=iterations,
                                valid_sets=[lgb_train, lgb_valid],
                                valid_names=["train", "valid"],
                                callbacks=callbacks,
                             )

        oof_lgb[va_idx] = lgb_model.predict(X_va)
        test_lgb += lgb_model.predict(X_test_pd) / kf.n_splits
        auc_lgb = roc_auc_score(y_va, oof_lgb[va_idx])

        cat_model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=20.0,
            eval_metric="Logloss",
            loss_function="Logloss",
            random_seed=seed + fold,
            verbose=False,
            task_type="GPU"
        )
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds = 50,
            cat_features=cat_features
        )

        oof_cat[va_idx] = cat_model.predict_proba(X_va)[:, 1]
        test_cat += cat_model.predict_proba(X_test_pd)[:, 1] / kf.n_splits

        X_tr_d = xgb.DMatrix(X_tr_xgb, label=y_tr, enable_categorical=True)
        X_va_d = xgb.DMatrix(X_va_xgb, label=y_va, enable_categorical=True)
        X_test_d = xgb.DMatrix(X_test_xgb, enable_categorical=True)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": "cuda",
            "min_child_weight": 10,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "lambda": 2,
            "alpha": 0.8,
            "random_state": seed + fold,
        }

        evals = [(X_tr_d, "train"), (X_va_d, "valid")]
        xgb_model = xgb.train(
            params,
            X_tr_d,
            num_boost_round=iterations,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )

        oof_xgb[va_idx] = xgb_model.predict(X_va_d)
        test_xgb += xgb_model.predict(X_test_d) / kf.n_splits

        auc_cat = roc_auc_score(y_va, oof_cat[va_idx])
        auc_xgb = roc_auc_score(y_va, oof_xgb[va_idx])

        if verbose:
            print(f"CatBoost AUC: {auc_cat:.5f} | XGBoost AUC: {auc_xgb:.5f} | LGBM AUC: {auc_lgb:.5f}")

        mean_abs_shap_cat, _ = compute_shap_importance(cat_model, 
                                                       X_va, 
                                                       model_name=f"CatBoost Fold {fold+1}", 
                                                       plot_top_n=0)
        mean_abs_shap_xgb, _ = compute_shap_importance(xgb_model, 
                                                       X_va_xgb, 
                                                       model_name=f"XGBoost Fold {fold+1}", 
                                                       plot_top_n=0)

        if mean_abs_shap_cat is not None: all_cat_shap.append(mean_abs_shap_cat)
        if mean_abs_shap_xgb is not None: all_xgb_shap.append(mean_abs_shap_xgb)

    print()
    if all_cat_shap:
        global_cat_shap = pd.concat(all_cat_shap, axis=1).mean(axis=1).sort_values(ascending=False)
        plt.figure(figsize=(8,6))
        global_cat_shap.head(20).plot(kind="barh", color="teal")
        plt.title("Global CatBoost SHAP Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    print()
    if all_xgb_shap:
        global_xgb_shap = pd.concat(all_xgb_shap, axis=1).mean(axis=1).sort_values(ascending=False)
        plt.figure(figsize=(8,6))
        global_xgb_shap.head(20).plot(kind="barh", color="orange")
        plt.title("Global XGBoost SHAP Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    # oof_meta = pd.DataFrame({"cat": oof_cat, "xgb": oof_xgb})
    # test_meta = pd.DataFrame({"cat": test_cat, "xgb": test_xgb})

    oof_meta = pd.DataFrame({
                            "cat": oof_cat,
                            "xgb": oof_xgb,
                            "lgb": oof_lgb,
                            })
    
    test_meta = pd.DataFrame({
                            "cat": test_cat,
                            "xgb": test_xgb,
                            "lgb": test_lgb,
                            })


    if META != 'LRCV':
        meta_model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=seed + fold
        )
        meta_model.fit(oof_meta, y_np)
    
        final_oof = meta_model.predict_proba(oof_meta)[:, 1]
        final_test = meta_model.predict_proba(test_meta)[:, 1]

    else:
        meta_model = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=np.logspace(-3, 3, 10),
                cv=5,
                penalty="l2",
                solver="lbfgs",
                scoring="roc_auc", # "neg_log_loss",#"roc_auc",
                max_iter=1000,
                random_state=seed,
            )
        )
    
        meta_model.fit(oof_meta, y_np)
        
        # final_oof = meta_model.predict_proba(oof_meta)[:, 1]
        # final_test = meta_model.predict_proba(test_meta)[:, 1]
        meta_model_calibrated = CalibratedClassifierCV(meta_model, method="sigmoid", cv="prefit")
        meta_model_calibrated.fit(oof_meta, y_np)
        
        final_oof = meta_model_calibrated.predict_proba(oof_meta)[:, 1]
        final_test = meta_model_calibrated.predict_proba(test_meta)[:, 1]

    # Computing Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_np, final_oof)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thr = thresholds[best_idx]
    best_thr = float(np.clip(best_thr, 0.05, 0.95))

    print(f"\nBest ROC-based threshold = {best_thr:.4f} (TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})")

    metrics = {
        "auc": float(roc_auc_score(y_np, final_oof)),
        "logloss": float(log_loss(y_np, np.clip(final_oof, 1e-6, 1 - 1e-6)))
    }

    if verbose:
        try:
            print("\nMeta-Blender Feature Importances:")
            print(pd.DataFrame({
                "Feature": oof_meta.columns,
                "Coefficient": meta_model.coef_.flatten()
            }).sort_values("Coefficient", ascending=False))
            print("[Meta-Blender] CV metrics:", metrics)
        except:
            pass

    final_oof_label = (final_oof >= best_thr).astype(int)
    acc = accuracy_score(y_np, final_oof_label)
    print(f"Meta-Model Accuracy: {acc:.4f}\n")
    print("\nClassification Report (Meta-Model):")
    print(classification_report(
        y_np,
        final_oof_label,
        target_names=["Not Cheating (Human)", "Cheating (AI/Copy)"],
        digits=3
    ))

    return final_test, metrics, oof_meta, test_meta, best_thr


def apply_variance_threshold(X_train, X_test, threshold=1e-3):

    print(f"\nApplying Variance Threshold (threshold={threshold})")

    selector = VarianceThreshold(threshold=threshold)
    X_train_reduced = selector.fit_transform(X_train)
    kept_features = X_train.columns[selector.get_support()]

    X_train_reduced = pd.DataFrame(X_train_reduced, columns=kept_features, index=X_train.index)
    X_test_reduced = X_test[kept_features].copy()

    print(f"Removed {X_train.shape[1] - len(kept_features)} low-variance features")
    print(f"Remaining features: {len(kept_features)}")

    return X_train_reduced, X_test_reduced, kept_features

def ensure_unique_columns(df):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def clean_feature_names(df):
    df = df.copy()
    df.columns = [
        re.sub(r'[^A-Za-z0-9_]', '_', col.strip().replace(" ", "_"))
        for col in df.columns
    ]
    return df

def align_train_test(train_df, test_df):
    common_cols = train_df.columns.intersection(test_df.columns)
    train_df = train_df[common_cols].copy()
    test_df = test_df[common_cols].copy()
    return train_df, test_df


train_features = clean_feature_names(train_features)
test_features = clean_feature_names(test_features)

train_features = ensure_unique_columns(train_features)
test_features = ensure_unique_columns(test_features)

iterations = 1000
train_ids = train_features["id"].copy()
test_ids = test_features["id"].copy()


X = train_features.drop(columns=["id", "topic", "answer", "combined_text", "is_cheating"], errors="ignore")
X_test = test_features.drop(columns=["id", "topic", "answer", "combined_text"], errors="ignore")
y = train["is_cheating"]

X, X_test = align_train_test(X, X_test)

print("Initial Train features shape:", X.shape)
print("Initial Test features shape:", X_test.shape)
X, X_test, kept_features = apply_variance_threshold(X, X_test, threshold=1e-3)
print("\nTrain features shape after filtering:", X.shape)
print("Test features shape after filtering:", X_test.shape)


final_test, metrics, oof_meta, test_meta, best_thr = fit_kfold_and_predict_ensemble(X, y, X_test)


class ContrastiveHardDataset(Dataset):
    def __init__(self, df, num_ref=128):
        self.df = df.reset_index(drop=True)
        self.humans = df[df.is_cheating == 0]["answer"].tolist()
        self.cheats = df[df.is_cheating == 1]["answer"].tolist()
        self.human_emb = embedder.encode(self.humans, 
                                         convert_to_numpy=True, 
                                         normalize_embeddings=True,
                                         show_progress_bar=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.df.loc[idx, "answer"]
        is_cheat = self.df.loc[idx, "is_cheating"]

        anc_emb = embedder.encode(anchor, 
                                  convert_to_numpy=True, 
                                  normalize_embeddings=True,
                                  show_progress_bar=False)

        if is_cheat == 0:
            sims = np.dot(self.human_emb, anc_emb)
            j = int(np.argmax(sims)) # hardest positive
            pos = self.humans[j]
            return anchor, pos, torch.tensor(0., dtype=torch.float32)

        sims = np.dot(self.human_emb, anc_emb)
        j = int(np.argmax(sims))
        neg = self.humans[j]
        return anchor, neg, torch.tensor(1., dtype=torch.float32)

class ContrastiveHardDataset_(Dataset):
    def __init__(self, df, num_ref=128, top_k=5):
        self.df = df.reset_index(drop=True)
        self.top_k = top_k
        humans = df[df.is_cheating == 0]["answer"].tolist()
        if len(humans) > num_ref:
            humans = random.sample(humans, num_ref)

        self.humans = humans
        self.human_emb = embedder.encode(
            humans,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        anchor = row["answer"]
        label = float(row["is_cheating"])

        anc_emb = embedder.encode(
            anchor, 
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        sims = np.dot(self.human_emb, anc_emb)
        top_indices = np.argsort(sims)[-self.top_k:]
        j = np.random.choice(top_indices)
        ref = self.humans[j]

        return anchor, ref, torch.tensor(label, dtype=torch.float32)


def contrastive_margin_loss(logits, labels, margin=0.5):

    probs = torch.softmax(logits, dim=1)[:, 1] # probability of cheating
    pos_loss = labels * torch.clamp(margin - probs, min=0)
    neg_loss = (1 - labels) * torch.clamp(probs - margin, min=0)
    return (pos_loss + neg_loss).mean()

def masked_mean_pool(last_hidden_state, attention_mask):

    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom

class AttentionRealFakeClassifier(nn.Module):
    def __init__(self, encoder, cfg, dropout=0.35, num_heads=8):
        super().__init__()
        self.encoder = encoder
        cfg = cfg
        self.h = cfg.hidden_size

        # bi-directional cross-attention
        self.cross12 = nn.MultiheadAttention(self.h, num_heads, batch_first=True)
        self.cross21 = nn.MultiheadAttention(self.h, num_heads, batch_first=True)

        # residual + layernorm around each cross-attn
        self.ln12 = nn.LayerNorm(self.h)
        self.ln21 = nn.LayerNorm(self.h)
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Sequential(
            nn.Linear(self.h, self.h),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        fusion_in = 4 * self.h + 1
        self.gate = nn.Sequential(
            nn.Linear(fusion_in, 2 * self.h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.h, self.h),
            nn.GELU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.h, 2)
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # encode
        out1 = self.encoder(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state
        out2 = self.encoder(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state

        c12, _ = self.cross12(query=out1, key=out2, value=out2)
        c21, _ = self.cross21(query=out2, key=out1, value=out1)
        c12 = self.ln12(out1 + self.dropout(c12))
        c21 = self.ln21(out2 + self.dropout(c21))

        # masked mean pooling
        p1 = masked_mean_pool(c12, attention_mask1)
        p2 = masked_mean_pool(c21, attention_mask2)

        # light projection
        p1 = self.proj(p1)
        p2 = self.proj(p2)

        # pairwise features
        diff = torch.abs(p1 - p2)
        prod = p1 * p2
        cos = F.cosine_similarity(p1, p2).unsqueeze(-1)  # [B,1]

        fused = torch.cat([p1, p2, diff, prod, cos], dim=-1)
        fused = self.gate(fused)
        logits = self.classifier(fused)
        return logits

class AttentionTrainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, lr=2e-5, epochs=5, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = contrastive_margin_loss

        # Track losses
        self.train_losses = []
        self.val_losses = []

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            # for text1, text2, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
            for text1, text2, labels in self.train_loader:
                inputs1 = self.tokenizer(text1, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                inputs2 = self.tokenizer(text2, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                labels = labels.to(self.device).float()

                logits = self.model(inputs1.input_ids, 
                                    inputs1.attention_mask, 
                                    inputs2.input_ids, 
                                    inputs2.attention_mask)
                loss = self.loss_fn(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                del inputs1, inputs2, labels, logits
                torch.cuda.empty_cache()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f}")

            val_loss, val_acc = self.evaluate()
            self.val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

            torch.cuda.empty_cache()
            gc.collect()

        # self.plot_loss()

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for text1, text2, labels in self.val_loader:
                inputs1 = self.tokenizer(text1, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                inputs2 = self.tokenizer(text2, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                labels = labels.to(self.device).float()

                logits = self.model(inputs1.input_ids, inputs1.attention_mask, inputs2.input_ids, inputs2.attention_mask)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                # preds = torch.argmax(logits, dim=1)
                preds = (torch.softmax(logits, dim=1)[:,1] > 0.5).long()

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(self.val_loader), correct / total

    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.epochs + 1), self.train_losses, label="Train Loss", marker='o')
        plt.plot(range(1, self.epochs + 1), self.val_losses, label="Val Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

train_df, val_df = train_test_split(train, test_size=0.2, stratify=train["is_cheating"], random_state=42)

train_dataset = ContrastiveHardDataset(train_df)
val_dataset = ContrastiveHardDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print("Shape of train data :", len(train_dataset))
print("Shape of validation data :", len(val_dataset))

model = AttentionRealFakeClassifier(encoder, cfg, dropout=0.25, num_heads=8)

trainer = AttentionTrainer(model, tokenizer, train_loader, val_loader, lr=1.5e-5, epochs=7)
trainer.loss_fn = contrastive_margin_loss
trainer.train()


human_pool = train[train.is_cheating == 0]["answer"].tolist()

human_pool_emb = embedder.encode(
    human_pool,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False
)

def get_top_k_refs(text, k=5):
    anc_emb = embedder.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    sims = np.dot(human_pool_emb, anc_emb)
    top_k_idx = np.argsort(sims)[-k:][::-1]
    return [human_pool[i] for i in top_k_idx]

def predict_cheating_probability(text):
    anc_emb = embedder.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    sims = np.dot(human_pool_emb, anc_emb)
    j = int(np.argmax(sims))
    ref_text = human_pool[j]

    inp1 = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(trainer.device)
    inp2 = tokenizer(ref_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(trainer.device)

    with torch.no_grad():
        logits = trainer.model(
            inp1.input_ids, inp1.attention_mask,
            inp2.input_ids, inp2.attention_mask
        )

        cheat_logit = logits[:, 1] - logits[:, 0]
        prob_cheating = torch.sigmoid(cheat_logit).cpu().item()

    return float(prob_cheating)

def predict_cheating_probability_boosted(text, k=3, agg="mean"):
    refs = get_top_k_refs(text, k)

    probs = []
    for ref in refs:
        inp1 = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(trainer.device)
        inp2 = tokenizer(ref,  padding=True, truncation=True, max_length=512, return_tensors="pt").to(trainer.device)

        with torch.no_grad():
            logits = trainer.model(inp1.input_ids, inp1.attention_mask, inp2.input_ids, inp2.attention_mask)
            p = torch.softmax(logits, dim=1).cpu().numpy()[0, 1]
            probs.append(float(p))

    if agg == "mean":
        return np.mean(probs)
    elif agg == "max":
        return np.max(probs)
    elif agg == "weighted":
        weights = np.linspace(1, 0.5, len(probs))
        weights /= weights.sum()
        return np.sum(np.array(probs) * weights)

    return np.mean(probs)


from torch.utils.data import TensorDataset

mpnet_model_path = snapshot_download("sentence-transformers/all-mpnet-base-v2")
mpnet_embedder = SentenceTransformer(mpnet_model_path, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingRegressor(nn.Module):
    def __init__(self, input_dim=768, hidden=256, dropout=0.25):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_embedding_regressor(train_df, embedder, epochs=15, batch_size=32, lr=1e-3):
    X_emb = embedder.encode(train_df["answer"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    y = train_df["is_cheating"].values.reshape(-1, 1)

    dataset = TensorDataset(
        torch.tensor(X_emb, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EmbeddingRegressor(input_dim=X_emb.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    return model

mpnet_model = train_embedding_regressor(train, mpnet_embedder)

oof_mpnet = mpnet_model(
    torch.tensor(mpnet_embedder.encode(train["answer"].tolist(), normalize_embeddings=True, show_progress_bar=False), 
                 dtype=torch.float32).to(device)
).cpu().detach().numpy().flatten()

test_mpnet = mpnet_model(
    torch.tensor(mpnet_embedder.encode(test["answer"].tolist(), normalize_embeddings=True, show_progress_bar=False),
                 dtype=torch.float32).to(device)
).cpu().detach().numpy().flatten()


BOOSTING = True

if BOOSTING:
    oof_contrastive = np.array([predict_cheating_probability_boosted(text, k=5, agg="mean") 
                                for text in train["answer"].tolist()])

    test_contrastive = np.array([predict_cheating_probability_boosted(text, k=5, agg="mean") 
                                 for text in test["answer"].tolist()])

else:
    oof_contrastive = np.array([predict_cheating_probability(text) 
                                for text in train["answer"].tolist()])
    
    test_contrastive = np.array([predict_cheating_probability(text) 
                                 for text in test["answer"].tolist()])

oof_stack = pd.DataFrame({
    "cat": oof_meta["cat"],
    "xgb": oof_meta["xgb"],
    "lgb": oof_meta["lgb"],
    "contrastive": oof_contrastive,
    "mpnet_reg": oof_mpnet
})

test_stack = pd.DataFrame({
    "cat": test_meta["cat"],
    "xgb": test_meta["xgb"],
    "lgb": test_meta["lgb"],
    "contrastive": test_contrastive,
    "mpnet_reg": test_mpnet
})


meta = make_pipeline(
    StandardScaler(),
    LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 10),
        cv=10,
        penalty="l2",
        solver="lbfgs",
        scoring="roc_auc",
        max_iter=1000,
        random_state=42,
    )
)

meta.fit(oof_stack, y)
meta_cal = CalibratedClassifierCV(meta, method="sigmoid", cv="prefit")
meta_cal.fit(oof_stack, y)

final_oof = meta_cal.predict_proba(oof_stack)[:, 1]
final_test = meta_cal.predict_proba(test_stack)[:, 1]

print("FINAL Blended AUC:", roc_auc_score(y, final_oof))
fpr, tpr, thr = roc_curve(y, final_oof)
j = tpr - fpr
best_thr = thr[np.argmax(j)]
best_thr = float(np.clip(best_thr, 0.05, 0.95))

print(f"Best threshold = {best_thr:.4f}")

final_oof_label = (final_oof >= best_thr).astype(int)
acc = accuracy_score(y, final_oof_label)
print(f"\nFinal Stacked Meta-Model Accuracy: {acc:.4f}")

print("\nFinal Stacked Meta-Model Classification Report:")
print(classification_report(
    y,
    final_oof_label,
    target_names=["Not Cheating (Human)", "Cheating (AI/Copy)"],
    digits=3
))

submission = pd.DataFrame({
    "id": test_ids,
    "is_cheating": (final_test >= best_thr).astype(int)
})

submission.to_csv("submission.csv", index=False)
submission.head(7)

