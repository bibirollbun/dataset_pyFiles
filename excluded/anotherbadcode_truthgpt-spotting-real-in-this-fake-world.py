import gc
gc.collect()

import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import string
!pip install -q textstat
import textstat
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from math import log2
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
!pip install -q langdetect
from langdetect import detect, DetectorFactory
!pip install -q pyspellchecker
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Union, Optional, Tuple
import zlib
from scipy.spatial.distance import euclidean
import shap
import umap

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

data_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
train_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
test_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"


def read_texts_from_dir(dir_path: str) -> pd.DataFrame:
    """
    Reads `file_1.txt` and `file_2.txt` from each `article_XXXX` subfolder
    and returns a DataFrame containing paired text samples.

    Each subfolder is expected to follow the structure:
        article_0001/
            â”œâ”€ file_1.txt
            â””â”€ file_2.txt

    Args:
        dir_path (str): Root path to the data directory (train or test),
                        containing subfolders named as `article_XXXX`.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'id': Integer article ID extracted from folder name
            - 'file_1': Content of `file_1.txt`
            - 'file_2': Content of `file_2.txt`

    Notes:
        - Skips folders with missing or malformed files.
        - Uses tqdm for progress tracking.
        - Prints error summary and last error encountered.
        - Set `DEBUG = True` globally to raise exceptions directly.
    """
    data = []
    error_count = 0
    last_error = None

    for folder_name in tqdm(sorted(os.listdir(dir_path)), desc="Reading folders"):
        folder_path = os.path.join(dir_path, folder_name)
        pos_path = os.path.join(folder_path, 'file_1.txt')
        neg_path = os.path.join(folder_path, 'file_2.txt')

        try:
            with open(pos_path, 'r', encoding='utf-8', errors='replace') as f1:
                text1 = f1.read().strip()
            with open(neg_path, 'r', encoding='utf-8', errors='replace') as f2:
                text2 = f2.read().strip()

            index = int(folder_name.split('_')[-1])
            data.append((index, text1, text2))

        except (FileNotFoundError, ValueError, OSError) as e:
            error_count += 1
            last_error = e
            if globals().get('DEBUG', False):
                raise e

    def clrd(msg, kind='info'):
        """Optional color print wrapper (can remove if unused)."""
        return f"[{kind.upper()}] {msg}"

    print(f"Read {clrd(len(data), 'ok')} records with {clrd(error_count, 'error')} errors")
    if error_count > 0:
        print(clrd('Last Error:', 'warn'), last_error)

    return pd.DataFrame(data, columns=['id', 'file_1', 'file_2'])


train = read_texts_from_dir(train_path)
test = read_texts_from_dir(test_path)
# test = test.sample(frac=0.01)
train = train.merge(pd.read_csv(data_dir), how='inner', on='id')


train.head(3)


test.head(1)


def flatten_real_fake_format(df):
    """
    Flatten a DataFrame containing columns: ['id', 'file_1', 'file_2', 'real_text_id']
    into one row per file with a binary 'label' column:
        - label = 1 if the file is the real one
        - label = 0 if the file is fake

    Returns a new DataFrame with columns: ['id', 'text', 'label']
    """
    # Rename for easier melt
    df_renamed = df.rename(columns={'file_1': 'text_1', 'file_2': 'text_2'})

    # Melt the two file columns into one
    df_melted = df_renamed.melt(
        id_vars=['id', 'real_text_id'],
        value_vars=['text_1', 'text_2'],
        var_name='file_source',
        value_name='text'
    )

    # Extract the number from 'text_1' or 'text_2'
    df_melted['file_id'] = df_melted['file_source'].str.extract(r'_(\d)').astype(int)

    # Assign label: 1 if file_id == real_text_id, else 0
    df_melted['label'] = (df_melted['file_id'] == df_melted['real_text_id']).astype(int)

    return df_melted[['id', 'text', 'label']]

def flatten_test_format(df):
    """
    Flatten a DataFrame containing columns: ['id', 'file_1', 'file_2']
    into one row per file with an added 'file_id' column (1 or 2).

    Useful for prediction where real/fake is unknown.

    Returns:
        pd.DataFrame with columns: ['id', 'file_id', 'text']
    """
    df_renamed = df.rename(columns={'file_1': 'text_1', 'file_2': 'text_2'})

    # Melt the two file columns
    df_melted = df_renamed.melt(
        id_vars=['id'],
        value_vars=['text_1', 'text_2'],
        var_name='file_source',
        value_name='text'
    )

    df_melted['file_id'] = df_melted['file_source'].str.extract(r'_(\d)').astype(int)

    return df_melted[['id', 'file_id', 'text']]


train = flatten_real_fake_format(train)
train.head(3)


DetectorFactory.seed = 0
stop_words = set(stopwords.words('english'))
spell_checker = SpellChecker()
nlp = spacy.load("en_core_web_sm")

embedder = SentenceTransformer("paraphrase-MiniLM-L12-v2") # "BAAI/bge-small-en-v1.5"

def bge_embedding(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Efficiently computes BGE embeddings for a list of texts using batching.
    """
    return embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        device="cpu"  # change to "cuda" if using GPU
    )

fake_flags = ['â˜‰', 'à¦¿', 'àµ‡', 'àµ�', 'áŸ’', 'á�¶', 'áŸ†', 'ï¿½', 'à¦¾', 'à§ˆ', 'à¤¾', 'à±‡', 'à±�', 'Û”', 'Ö·', 'à¥�', 'à°¿', 'ï¼�', 'ã€�', 'àª¾', 'à«€',
              'à«‡', 'á€»', 'á€¹', 'á€¬', 'à´¿', '\u200b', 'à¯�', 'à·Š', 'à®¾', 'à³�', 'à³�', 'ğŸ˜°', 'àµ€', 'Ù‘', 'à¥‹', 'à¥�', 'à¤�', 'à¥ˆ', 'à¤¿', 'à²¾',
              'à¥‡', 'à¸´', 'à§‡', 'à¨¾', 'á€¯', 'à´¾', 'àª¿', 'à¥°', 'à²¿', 'à§�', 'à«�', 'âˆ€', 'ï¼š', 'á�¾', 'à¸·', 'à¹ˆ', 'à¯�', 'à·“', 'à¸¶', 'à±‚',
              '\u0ab1', 'à°¾', '\u200c', 'à¦‚', 'à³ˆ', 'à©‹', 'à¥Œ', 'â‚¬', 'ã€‚', 'á€±', 'â—¸', 'à¥‚', 'à¤‚', 'à¯‡', '\u05f6', 'à·’',
              '\u0ba5', 'à¸µ', 'à±�', 'à¹Œ', 'Ã·', 'à¯†', 'à¥€', 'Ò†', 'Ëš', 'Ìˆ', 'à¥¤', 'ğŸ¥¶', '\u05cd', 'à±Š', 'ã€�', 'ã€‹', 'à¹‰', 'â†�', 'à¸¹',
              'à§‹', 'à«�', 'à®¿', 'à±‹', 'à¨¼', 'à«‹', 'â‰¥', 'ØŒ', 'àµ�', 'à´‚', 'ï¼Œ', 'à»‰', 'â €', 'ã€‘', 'à¼‹', 'á�»', 'Ì®', 'ğŸ�¦', 'à±€', 'à«‚',
              'àª‚', 'ï¼‰', 'à¤¼', 'à©‡', 'Â¡', 'ã€�', 'á�¹', 'à·�', 'à³†', 'à¸¸', 'à§�', 'ï¼Ÿ', 'à¸±', 'à±†', 'à­�', 'àµ‹', 'à³€', 'ï¼�', 'à¯‹',
              'á�¼', 'à·�', 'Â´', 'â�”', 'à¹‡', 'àµƒ', 'Õ›', 'ğŸ•Š', 'â‰¤', 'àµ‚', 'Ö¸', 'à¦¼', 'à¤€', 'à²‚', 'à¨¿', 'Ù”', 'ï¼›', 'á€·', 'á€º', 'áŸ„',
              'áŸ‡', 'Ù�', 'à¯€', 'â‰•', 'ï¼ˆ', 'ã�ˆ', 'à¹‹', 'àµˆ', 'á€²', '\xad', 'à¥’', 'áŸƒ', 'à§ƒ', 'Ù�', '\x04', 'ï¼»', 'à³‡']

# Idea borrowed from : https://www.kaggle.com/code/atharva0577/imposter-distilbert-baseline
def count_special_texts(text: str) -> int:
    """Count how many times fake/foreign characters appear in a single text."""
    return sum(text.count(char) for char in fake_flags)
    
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

def embedding_coherence(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2: return 1.0
    embeddings = np.vstack([nlp(sent).vector for sent in sentences if nlp(sent).has_vector])
    if embeddings.shape[0] < 2: return 1.0
    sims = cosine_similarity(embeddings)
    tril = sims[np.tril_indices_from(sims, k=-1)]
    return np.mean(tril)

spell_checker = SpellChecker()

def count_spelling_errors(words):
    misspelled = spell_checker.unknown(words)
    return len(misspelled)

def detect_script_ratios(text: str) -> dict:
    total = len(text)
    if total == 0:
        return {"english_ratio": 0.0, "latin_ratio": 0.0}
    english_count = len(re.findall(r'[a-zA-Z]', text))
    latin_count = len(re.findall(r'[^\x00-\x7F]', text))
    return {
        "english_ratio": english_count / total,
        "latin_ratio": latin_count / total
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
    emdash_count = text.count("â€”")
    long_words = sum(1 for w in words if len(w) > 6)
    short_words = sum(1 for w in words if len(w) <= 3)
    stopword_count = sum(1 for w in words if w.lower() in stop_words)
    unique_words = len(set(words))
    upper_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    avg_word_len = np.mean([len(w) for w in words]) if word_count > 0 else 0
    ent = compute_entropy(text)
    spelling_errors = count_spelling_errors([w for w in words if w.isalpha()])
    syllables_per_word = textstat.syllable_count(text) / word_count if word_count else 0
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

    readability_scores = [
                        textstat.flesch_reading_ease(text),
                        textstat.flesch_kincaid_grade(text),
                        textstat.gunning_fog(text),
                        textstat.smog_index(text),
                        textstat.coleman_liau_index(text),
                        textstat.automated_readability_index(text),
                        textstat.dale_chall_readability_score(text),
                        textstat.linsear_write_formula(text)
                        ]
    readability_avg = np.mean(readability_scores)

    emb = bge_embedding(text)
    emb_dict = {f"bge_emb_{i}": emb[i] for i in range(len(emb))}
    avg_grad_delta = float(np.mean(np.abs(np.diff(emb)))) if len(emb) > 1 else 0.0

    mean_emb_val = float(np.mean(emb))
    std_emb_val = float(np.std(emb))
    special_char_count = count_special_texts(text)

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
    'flesch_reading_ease': readability_scores[0],
    'flesch_kincaid_grade': readability_scores[1],
    'gunning_fog': readability_scores[2],
    'smog_index': readability_scores[3],
    'coleman_liau_index': readability_scores[4],
    'automated_readability_index': readability_scores[5],
    'dale_chall_readability_score': readability_scores[6],
    'linsear_write_formula': readability_scores[7],
    'english_ratio': script_ratios["english_ratio"],
    'latin_ratio': script_ratios["latin_ratio"],
    'digit_count': digit_count,
    'uppercase_ratio': upper_count / num_chars if num_chars else 0,
    'long_word_count': long_words,
    'short_word_count': short_words,
    'avg_syllables_per_word': syllables_per_word,
    'type_token_ratio_sqrt': type_token_ratio_sqrt,
    'readability_avg': readability_avg,
    'entropy': ent,
    'emdash_count': emdash_count,
    'ngram_repetition': ngram_repetition(text),
    'ner_count': ner_count,
    'spelling_errors': spelling_errors,
    'embedding_coherence': embedding_coherence(text),
    'compression_ratio': compression_ratio,
    'noun_ratio': noun_ratio,
    'verb_ratio': verb_ratio,
    'adj_ratio': adj_ratio,
    'adv_ratio': adv_ratio,
    'avg_dependency_depth': avg_dep_depth,
    'embedding_gradient_delta': avg_grad_delta,
    'mean_emb_val': mean_emb_val,
    'std_emb_val':std_emb_val,
    'special_char_count': special_char_count,
     **emb_dict
    }


def extract_text_features(df: pd.DataFrame, 
                          text_col: str,
                          add_umap: bool = False,
                          umap_n_components: int = 64,
                          umap_neighbors: int = 32,
                          is_train: bool = True,
                          pretrained_umap: Optional[umap.UMAP] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, umap.UMAP]]:

    # features = df[text_col].apply(feature_row)

    features = df[text_col].apply(lambda x: feature_row(str(x)) if isinstance(x, str) else {})
    df_feat = pd.DataFrame(features.tolist(), index=df.index)

    bge_cols = [col for col in df_feat.columns if col.startswith("bge_emb_")]
    if not bge_cols:
        raise ValueError("No BGE embedding columns found in extracted features.")
    bge_embs = df_feat[bge_cols].values

    if add_umap:
        
        reducer = umap.UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_neighbors,
            metric='cosine'
        )
        umap_features = reducer.fit_transform(bge_embs)

        umap_df = pd.DataFrame(
                                umap_features,
                                columns=[f'umap_emb_{i}' for i in range(umap_n_components)],
                                index=df.index
                              )
        # df_feat = pd.concat([df_feat, umap_df], axis=1)
        df_feat = pd.concat([df_feat.drop(columns=bge_cols), umap_df], axis=1)

    if is_train:
        if add_umap:
            return df_feat, reducer
        else:
            return df_feat
            
    else: # means test data
        if pretrained_umap is None:
            raise ValueError("Please provide pretrained_umap when is_train=False.")

        bge_cols = [col for col in df_feat.columns if col.startswith("bge_emb_")]
        umap_features = pretrained_umap.transform(bge_embs)
        umap_df = pd.DataFrame(
                                umap_features,
                                columns=[f'umap_emb_{i}' for i in range(umap_n_components)],
                                index=df.index
                              )
        # df_feat = pd.concat([df_feat, umap_df], axis=1)
        df_feat = pd.concat([df_feat.drop(columns=bge_cols), umap_df], axis=1)
        return df_feat

def fit_tfidf_vectorizer(train_df, text_col, max_features=1000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(max_df = 0.8, max_features=max_features, ngram_range=ngram_range)
    vectorizer.fit(train_df[text_col])
    return vectorizer

def transform_tfidf_features(df, vectorizer, text_col, prefix="tfidf_"):
    tfidf_matrix = vectorizer.transform(df[text_col])
    feature_names = [f"{prefix}{feat}" for feat in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df.index)
    return tfidf_df


def generate_all_text_features(df, 
                               text_col, 
                               tfidf_vectorizer=None, 
                               fit_vectorizer=True, 
                               max_tfidf_features=500,
                               add_umap=True, 
                               is_train=True,
                               pretrained_umap=None):
    """
    Extracts a full set of text-based features including:
        - Handcrafted statistical/linguistic features
        - TF-IDF features (trained or reused vectorizer)

    Args:
        df (pd.DataFrame): DataFrame with a `text_col` column.
        text_col (str): Name of the column containing text.
        tfidf_vectorizer (TfidfVectorizer): Optional prefit TF-IDF vectorizer.
        fit_vectorizer (bool): If True, fits TF-IDF on this dataset.
        max_tfidf_features (int): Number of TF-IDF features to extract.

    Returns:
        features_df (pd.DataFrame): All features combined.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    """
    if is_train:
        text_stat_features, umap_model = extract_text_features(df.copy(), 
                                                               "text", 
                                                               add_umap=add_umap, 
                                                               is_train=is_train,
                                                               pretrained_umap=pretrained_umap)
    else:
        text_stat_features = extract_text_features(df.copy(), 
                                                   "text", 
                                                   add_umap=add_umap, 
                                                   is_train=is_train,
                                                   pretrained_umap=pretrained_umap)
        

    if text_col in text_stat_features.columns:
        text_stat_features = text_stat_features.drop(columns=[text_col])
    
    if tfidf_vectorizer is None and fit_vectorizer:
        tfidf_vectorizer = fit_tfidf_vectorizer(df, text_col, max_features=max_tfidf_features)
        tfidf_features = transform_tfidf_features(df, tfidf_vectorizer, text_col)
    else:
        tfidf_features = transform_tfidf_features(df, tfidf_vectorizer, text_col)

    combined_df = pd.concat([text_stat_features, tfidf_features], axis=1)

    if add_umap and is_train:
        return combined_df, tfidf_vectorizer, umap_model
    else:
        return combined_df, tfidf_vectorizer


train_features, tfidf_vec, umap_model = generate_all_text_features(train, text_col="text")
y = train["label"].values


features_to_plot = [
    'spelling_errors',         # Number of spelling errors
    'stopword_ratio',          # Ratio of stopwords
    'ngram_repetition',        # Repetition ratio of n-grams
    'flesch_kincaid_grade',    # Flesch readability score
    'char_count',              # Total character count
    'avg_dependency_depth'     # Average syntactic depth from spaCy
]

plot_df = train_features[features_to_plot].copy()
plot_df['label'] = y

# Set up 3x2 plot grid
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    sns.violinplot(data=plot_df, x='label', y=feature, ax=axes[i])
    axes[i].set_title(f'Violin Plot of {feature}')
    axes[i].set_xlabel('Label (0 = Fake, 1 = Real)')
    axes[i].set_ylabel(feature.replace("_", " ").title())

plt.tight_layout()
plt.show()


def drop_high_corr_numeric_features(df: pd.DataFrame, threshold: float = 0.95, return_dropped: bool = False):
    """
    Drop numeric columns that are highly correlated (above threshold), keeping one from each correlated pair.

    Args:
        df (pd.DataFrame): DataFrame with numeric and categorical features.
        threshold (float): Correlation threshold for dropping.
        return_dropped (bool): If True, also return list of dropped columns.

    Returns:
        pd.DataFrame: Reduced DataFrame.
        list (optional): List of dropped feature names.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    df_reduced = df.drop(columns=to_drop)

    print(f"Dropped {len(to_drop)} numeric features due to correlation > {threshold}")
    if return_dropped:
        return df_reduced, to_drop
    return df_reduced

train_features, dropped_cols = drop_high_corr_numeric_features(train_features, threshold=0.85, return_dropped=True)


import contextlib
import sys
from sklearn.metrics import classification_report

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def prepare_categorical_features(X: pd.DataFrame, model_name: str):
    """
    Detect categorical columns, cast to category dtype, and return the correct format
    (column names or indices) for different models.
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Cast to 'category' dtype
    for col in cat_cols:
        X[col] = X[col].astype("category")

    # Return correct format based on model
    if model_name == "CatBoost":
        return X, cat_cols  # CatBoost needs column names
    elif model_name == "LightGBM":
        return X, cat_cols  # LightGBM accepts names
    elif model_name == "XGBoost":
        return X, [X.columns.get_loc(col) for col in cat_cols]  # XGB wants indices
    else:
        return X, []


models = {
            "CatBoost": CatBoostClassifier(iterations = 500, 
                                           task_type='GPU', 
                                           loss_function="Logloss", 
                                           early_stopping_rounds=50,
                                           verbose=0),
            "LightGBM": LGBMClassifier(n_estimators = 500, device="gpu"),
            "XGBoost": XGBClassifier(n_estimators = 500, 
                                     tree_method='hist',
                                     eval_metric='logloss',
                                     use_label_encoder=False, 
                                     enable_categorical=True)
        }

kf = StratifiedKFold(n_splits=5, shuffle=True)

final_model_predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    fold_preds = []
    oof_preds = np.zeros((len(train_features), 2))  # [prob_class_0, prob_class_1]

    for fold, (train_index, val_index) in enumerate(kf.split(train_features, y)):
        X_train = train_features.iloc[train_index].copy()
        X_val = train_features.iloc[val_index].copy()
        y_train = y[train_index]
        y_val = y[val_index]

        # Categorical handling
        X_train, cat_feats = prepare_categorical_features(X_train, name)
        X_val, _ = prepare_categorical_features(X_val, name)

        with suppress_output():
            if name == "CatBoost":
                model.fit(X_train, y_train, cat_features=cat_feats)
            elif name == "LightGBM":
                model.fit(X_train, y_train, categorical_feature=cat_feats)
            else:  # XGBoost
                model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)
        oof_preds[val_index] = y_proba

    final_model_predictions[name] = model

    y_pred_labels = np.argmax(oof_preds, axis=1)
    acc = accuracy_score(y, y_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred_labels, average="macro")

    print(f"{name} | Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"Classification Report for {name}:\n")
    print(classification_report(y, y_pred_labels, target_names=["Fake", "Real"]))



def shap_summary_for_catboost(model, train_data):
    """
    Generate SHAP summary plot for CatBoost model.
    Uses TreeExplainer for compatibility without GPU dependencies.

    Args:
        model: Trained CatBoost model
        train_data: Training features DataFrame (used during model training)
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_data)
    shap.summary_plot(shap_values, train_data)


shap_summary_for_catboost(final_model_predictions["CatBoost"], train_features)


def ensemble_predict_real_text_id_fast(test_df, 
                                       models_dict, 
                                       model_names, 
                                       tfidf_vectorizer,
                                       umap_model=None) -> pd.DataFrame:
    """
    Faster version: batch prediction for all rows.
    """
    submission = []
    ids = test_df["id"].tolist()
    text1_all = test_df["file_1"].tolist()
    text2_all = test_df["file_2"].tolist()

    df1 = pd.DataFrame({"text": text1_all})
    df2 = pd.DataFrame({"text": text2_all})

    feats1, _ = generate_all_text_features(df1, text_col="text", 
                                           tfidf_vectorizer=tfidf_vectorizer, 
                                           fit_vectorizer=False,
                                           add_umap=True, is_train=False, 
                                           pretrained_umap=umap_model)

    feats2, _ = generate_all_text_features(df2, text_col="text", 
                                           tfidf_vectorizer=tfidf_vectorizer, 
                                           fit_vectorizer=False,
                                           add_umap=True, is_train=False, 
                                           pretrained_umap=umap_model)

    feats1.drop(columns=dropped_cols, inplace=True, errors="ignore")
    feats2.drop(columns=dropped_cols, inplace=True, errors="ignore")

    feats1 = feats1.loc[:, ~feats1.columns.duplicated()]
    feats2 = feats2.loc[:, ~feats2.columns.duplicated()]

    feats1 = feats1.reindex(columns=train_feature_cols, fill_value=0)
    feats2 = feats2.reindex(columns=train_feature_cols, fill_value=0)

    probs1_total = np.zeros(len(test_df))
    probs2_total = np.zeros(len(test_df))

    for name in model_names:
        model = models_dict[name]

        f1_model, _ = prepare_categorical_features(feats1.copy(), name)
        f2_model, _ = prepare_categorical_features(feats2.copy(), name)

        f1_model = f1_model.loc[:, ~f1_model.columns.duplicated()].reindex(columns=train_feature_cols, fill_value=0)
        f2_model = f2_model.loc[:, ~f2_model.columns.duplicated()].reindex(columns=train_feature_cols, fill_value=0)

        p1 = model.predict_proba(f1_model)[:, 1]
        p2 = model.predict_proba(f2_model)[:, 1]

        probs1_total += p1
        probs2_total += p2

    avg_prob1 = probs1_total / len(model_names)
    avg_prob2 = probs2_total / len(model_names)

    real_text_ids = np.where(avg_prob1 > avg_prob2, 1, 2)

    return pd.DataFrame({"id": ids, "real_text_id": real_text_ids})



%%time

def ensemble_predict_real_text_id(test_df, 
                                  models_dict, 
                                  model_names, 
                                  tfidf_vectorizer,
                                  umap_model=None) -> pd.DataFrame:
    """
    Predict real_text_id using ensemble voting (average probability across models).

    Args:
        test_df (pd.DataFrame): Original test df with ['id', 'file_1', 'file_2']
        models_dict (dict): Trained models {name: model}
        model_names (list): Names in order used for training
        tfidf_vectorizer (TfidfVectorizer): Fitted vectorizer for reuse

    Returns:
        pd.DataFrame: With ['id', 'real_text_id']
    """
    submission = []

    for _, row in test_df.iterrows():
        text1 = row["file_1"]
        text2 = row["file_2"]
        id_ = row["id"]

        probs_file1, probs_file2 = [], []

        for name in model_names:
            model = models_dict[name]

            df1 = pd.DataFrame({"text": [text1]})
            df2 = pd.DataFrame({"text": [text2]})


            feats1, _ = generate_all_text_features(df1, 
                                                   text_col="text", 
                                                   tfidf_vectorizer=tfidf_vectorizer, 
                                                   fit_vectorizer=False, 
                                                   add_umap=True, 
                                                   is_train=False,
                                                   pretrained_umap=umap_model)

            feats2, _ = generate_all_text_features(df2, 
                                                   text_col="text",
                                                   tfidf_vectorizer=tfidf_vectorizer, 
                                                   fit_vectorizer=False,
                                                   add_umap=True, 
                                                   is_train=False,
                                                   pretrained_umap=umap_model)

            feats1.drop(columns=dropped_cols, inplace=True, errors="ignore")
            feats2.drop(columns=dropped_cols, inplace=True, errors="ignore")

            feats1, _ = prepare_categorical_features(feats1.copy(), name)
            feats2, _ = prepare_categorical_features(feats2.copy(), name)

            # Ensure no duplicates in train_feature_cols
            feats1 = feats1.loc[:, ~feats1.columns.duplicated()]
            feats2 = feats2.loc[:, ~feats2.columns.duplicated()]

            feats1 = feats1.reindex(columns=train_feature_cols, fill_value=0)
            feats2 = feats2.reindex(columns=train_feature_cols, fill_value=0)

            prob1 = model.predict_proba(feats1)[0][1]
            prob2 = model.predict_proba(feats2)[0][1]

            probs_file1.append(prob1)
            probs_file2.append(prob2)

        avg_prob1 = np.mean(probs_file1)
        avg_prob2 = np.mean(probs_file2)

        real_text_id = 1 if avg_prob1 > avg_prob2 else 2
        submission.append({"id": id_, "real_text_id": real_text_id})

    return pd.DataFrame(submission)

train_feature_cols = train_features.columns.tolist()

model_names = ["CatBoost", "LightGBM", "XGBoost"]
# submission_df = ensemble_predict_real_text_id(test, final_model_predictions, model_names, tfidf_vec, umap_model)

submission_df = ensemble_predict_real_text_id_fast(test, final_model_predictions, model_names, tfidf_vec, umap_model)
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head(7))

