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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

train = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
test = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')


train.head(3)


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
train_features.head(3)


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

        from sklearn.calibration import CalibratedClassifierCV
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

    return final_test, metrics, oof_meta, best_thr


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

iterations = 500
train_ids = train_features["id"].copy()
test_ids = test_features["id"].copy()

X = train_features.drop(columns=["id", "topic", "answer", "is_cheating"], errors="ignore")
X_test = test_features.drop(columns=["id", "topic", "answer"], errors="ignore")
y = train["is_cheating"]

X, X_test = align_train_test(X, X_test)

print("Initial Train features shape:", X.shape)
print("Initial Test features shape:", X_test.shape)
X, X_test, kept_features = apply_variance_threshold(X, X_test, threshold=1e-3)
print("\nTrain features shape after filtering:", X.shape)
print("Test features shape after filtering:", X_test.shape)


final_pred, metrics, oof_meta, best_thr = fit_kfold_and_predict_ensemble(X, y, X_test)


submission = pd.DataFrame({
    "id": test_ids,
    "is_cheating": (final_pred >= best_thr).astype(int)
})
submission.to_csv("submission.csv", index=False)
submission.head(7)

