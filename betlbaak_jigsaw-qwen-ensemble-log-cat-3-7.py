!pip install sentence-transformers --no-deps --find-links=/kaggle/input/sentence-transformers-3-0-1/sentence_transformers-3.0.1-py3-none-any.whl
!pip install transformers --no-deps --find-links=/kaggle/input/transformers-4-44-2/transformers-4.44.2-py3-none-any.wh
!pip install vaderSentiment --no-deps --find-links=/kaggle/input/vadersentiment/vaderSentiment-3.3.2-py2.py3-none-any.whl
!pip install better_profanity --no-index --find-links=/kaggle/input/better-profanity/better_profanity-0.7.0-py3-none-any.whl
!pip install cudf-cuml --no-index --find-links=/kaggle/input/cudf-pip-wheel/cudf-wheel


import numpy as np
import pandas as pd 
import os
import re
#import json
#import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import torch, gc
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sentence_transformers import SentenceTransformer, util
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.max_colwidth", None)


train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

print(f"Train shape: ", train.shape)
print(f"Test shape: ", test.shape)


def augment_with_examples(df, source="train"):
    """
    Augments a dataset by generating additional positive and negative examples 
    based on predefined example columns.

    This function extracts example texts from the input DataFrame that illustrate 
    rule violations (positive examples) and non-violations (negative examples). 
    It reshapes these columns into a unified "text" column format, assigns labels, 
    and returns an augmented DataFrame suitable for model training or evaluation.

    Args:
        df (pd.DataFrame): 
            Input DataFrame containing at least the columns:
            - "rule"
            - "positive_example_1"
            - "negative_example_1"
        source (str, optional): 
            Label indicating the data source, e.g., "train" or "test". 
            Defaults to "train".

    Returns:
        pd.DataFrame: 
            Augmented DataFrame with the following columns:
            - "rule": Identifier for the rule associated with the example.
            - "text": Example text (positive or negative).
            - "rule_violation": Binary label indicating whether the text violates the rule (1) or not (0).
            - "source": The dataset source label provided as input.

    Example:
        >>> augmented = augment_with_examples(train_df, source="train")
        >>> augmented.head()
             rule                 text       rule_violation    source
        0    R1   "Text that violates..."          1            train
        1    R1   "Compliant example..."           0            train
    """
    
    # Positive examples
    pos_cols = ["positive_example_1"]
    pos_df = df.melt(id_vars=["rule"], 
                     value_vars=pos_cols, 
                     value_name="text")[["rule", "text"]]
    pos_df = pos_df.dropna(subset=["text"]).reset_index(drop=True)
    pos_df["rule_violation"] = 1
    pos_df["source"] = source

    # Negative examples
    neg_cols = ["negative_example_1"]
    neg_df = df.melt(id_vars=["rule"], 
                     value_vars=neg_cols, 
                     value_name="text")[["rule", "text"]]
    neg_df = neg_df.dropna(subset=["text"]).reset_index(drop=True)
    neg_df["rule_violation"] = 0
    neg_df["source"] = source

    return pd.concat([pos_df, neg_df], axis=0).reset_index(drop=True)

# 1. Main train
main_train = train[["body", "rule", "rule_violation"]].rename(columns={"body": "text"})
main_train["source"] = "train"

# 2. Examples derived from the train set
aug_train = augment_with_examples(train, source="train")

# 3. Examples derived from the test set
aug_test = augment_with_examples(test, source="test")

# 4. Concat them all
train_df = pd.concat([main_train, aug_train, aug_test], axis=0).reset_index(drop=True)


test_df = test[["row_id", "body", "rule", "positive_example_1", "negative_example_1"]].rename(columns={"body": "text"})

examples_df = train[["rule", "positive_example_2", "negative_example_2"]].rename(columns={"positive_example_2": "positive_example",
                                                                                "negative_example_2": "negative_example"})

test_examples = test[["rule", "positive_example_2", "negative_example_2"]].copy()

test_examples.columns = ["rule", "positive_example", "negative_example"]

examples_df = pd.concat([examples_df, test_examples], ignore_index=True)

print(f"Train_df shape: ", train_df.shape)
print(f"Test_df shape: ", test_df.shape)
print(f"examples_df shape: ", examples_df.shape)


# -----------------------------
# GPU cleaning ve model
# -----------------------------
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
gc.collect()
torch.cuda.empty_cache()
tqdm.pandas() 

embed_model_name = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(embed_model_name, device=device)

# -----------------------------
# Memory-safe batch embedding function
# -----------------------------
def compute_embeddings(texts, batch_size=64):
    """
    Computes sentence embeddings for a list of input texts in batches.

    This function encodes text data into dense vector representations using a 
    preloaded sentence embedding model (e.g., SentenceTransformer). 
    It processes the texts in batches to optimize memory usage and performance, 
    returning all embeddings as a single NumPy array.

    Args:
        texts (list of str): 
            List of input texts to be converted into embeddings.
        batch_size (int, optional): 
            Number of texts to process per batch. 
            Defaults to 64.

    Returns:
        np.ndarray: 
            A 2D NumPy array of shape (n_texts, embedding_dim) containing the 
            normalized embeddings for each input text.

    Example:
        >>> texts = ["The sky is blue.", "Machine learning is fascinating."]
        >>> embeddings = compute_embeddings(texts, batch_size=32)
        >>> embeddings.shape
        (2, 768)
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Texts"):
        batch = texts[i:i+batch_size]
        emb = embedder.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(emb)
        
    return np.vstack(embeddings)

# -----------------------------
# Combine and embed all necessary columns
# -----------------------------

all_texts = (
    train_df["text"].tolist()
    + examples_df["positive_example"].dropna().tolist()
    + examples_df["negative_example"].dropna().tolist()
)
all_texts = list(dict.fromkeys(all_texts))  # unique

print("ğŸ”¹ All text columns are embedded...")
all_embeddings = compute_embeddings(all_texts, batch_size=128)
text_to_emb = {t: e for t, e in zip(all_texts, all_embeddings)}

# -----------------------------
# Diversity examples
# -----------------------------

def select_diverse_examples(examples_df, text_to_emb, k=200):
    """
    Selects a diverse subset of positive and negative examples for each rule 
    based on embedding similarity.

    This function ensures variety among training examples by clustering embeddings 
    of positive and negative examples separately for each rule using K-Means, 
    then selecting the sample closest to each cluster center. 
    This helps reduce redundancy and improve model generalization by maintaining 
    semantic diversity within the dataset.

    Args:
        examples_df (pd.DataFrame):
            DataFrame containing at least the following columns:
            - "rule": The rule identifier.
            - "positive_example": Example texts that violate the rule.
            - "negative_example": Example texts that do not violate the rule.
        text_to_emb (dict):
            A dictionary mapping text strings to their corresponding embedding vectors 
            (e.g., computed via a sentence transformer).
        k (int, optional):
            Number of diverse examples to select per rule and per class (positive/negative).
            If fewer examples exist, all available examples are returned. 
            Defaults to 200.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing:
            - `diverse_pos_df`: DataFrame of selected diverse positive examples.
            - `diverse_neg_df`: DataFrame of selected diverse negative examples.
            Both DataFrames have the columns:
            - "rule": The rule identifier.
            - "example": The selected example text.

    Example:
        >>> diverse_pos_df, diverse_neg_df = select_diverse_examples(examples_df, text_to_emb, k=100)
        âœ… Ã‡eÅŸitli 1200 pozitif ve 1200 negatif Ã¶rnek seÃ§ildi.
        >>> diverse_pos_df.head()
             rule               example
        0    R1   "This text violates rule 1"
        1    R1   "Another violating example"
        2    R2   "Rule 2 example..."

    Notes:
        - K-Means clustering (`sklearn.cluster.KMeans`) is used to select examples closest 
          to cluster centers, ensuring maximum diversity.
        - Progress is displayed with a `tqdm` progress bar.
    """
    
    diverse_pos = []
    diverse_neg = []

    for rule in tqdm(examples_df["rule"].unique(), desc="Selecting diverse examples"):
        rule_df = examples_df[examples_df["rule"] == rule]

        pos_texts = rule_df["positive_example"].dropna().tolist()
        neg_texts = rule_df["negative_example"].dropna().tolist()

        pos_embs = np.array([text_to_emb[t] for t in pos_texts if t in text_to_emb])
        neg_embs = np.array([text_to_emb[t] for t in neg_texts if t in text_to_emb])

        def pick_diverse(embs, texts):
            if len(embs) <= k:
                return texts
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            km.fit(embs)
            centers = km.cluster_centers_
            chosen = []
            for c in range(k):
                cluster_idx = np.where(km.labels_ == c)[0]
                if len(cluster_idx) == 0:
                    continue
                cluster_embs = embs[cluster_idx]
                dists = np.linalg.norm(cluster_embs - centers[c], axis=1)
                chosen.append(texts[cluster_idx[np.argmin(dists)]])
            return chosen

        diverse_pos.extend([(rule, t) for t in pick_diverse(pos_embs, pos_texts)])
        diverse_neg.extend([(rule, t) for t in pick_diverse(neg_embs, neg_texts)])

    diverse_pos_df = pd.DataFrame(diverse_pos, columns=["rule", "example"])
    diverse_neg_df = pd.DataFrame(diverse_neg, columns=["rule", "example"])
    print(f"âœ… {len(diverse_pos_df)} positive and {len(diverse_neg_df)} negative examples selected.")
    return diverse_pos_df, diverse_neg_df

diverse_pos_df, diverse_neg_df = select_diverse_examples(examples_df, text_to_emb)

# -----------------------------
# Calculate similarity scores
# -----------------------------
def compute_body_scores_vectorized(df, diverse_pos_df, diverse_neg_df, text_to_emb, metric="cosine"):
    """
    Computes rule-based similarity scores between input texts and their corresponding
    diverse positive and negative examples using vectorized embedding similarity.

    This function measures how similar each text body is to the rule's representative
    positive and negative examples (derived from `diverse_pos_df` and `diverse_neg_df`).
    It computes average similarities based on the selected metric and derives a 
    probability-like score representing the likelihood that the text violates the rule.

    Args:
        df (pd.DataFrame):
            DataFrame containing at least:
            - "text": The text to evaluate.
            - "rule": The rule identifier corresponding to each text.
        diverse_pos_df (pd.DataFrame):
            DataFrame containing positive examples for each rule with columns:
            - "rule": Rule identifier.
            - "example": Example text that violates the rule.
        diverse_neg_df (pd.DataFrame):
            DataFrame containing negative examples for each rule with columns:
            - "rule": Rule identifier.
            - "example": Example text that does not violate the rule.
        text_to_emb (dict):
            A mapping from text strings to their precomputed embedding vectors.
        metric (str, optional):
            Similarity metric to use. Supported options:
            - "cosine" (default)
            - "dot"
            - "euclidean"
            - "manhattan"
            - "pearson"

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - `prob_pos_list`: Array of rule-violation probabilities for each text.
            - `sim_pos_list`: Mean similarity to positive examples per text.
            - `sim_neg_list`: Mean similarity to negative examples per text.
            Values are aligned with the original order of `df`; entries without
            embeddings are filled with NaN.

    Example:
        >>> prob_pos, sim_pos, sim_neg = compute_body_scores_vectorized(
        ...     df=train_df,
        ...     diverse_pos_df=diverse_pos_df,
        ...     diverse_neg_df=diverse_neg_df,
        ...     text_to_emb=text_to_emb,
        ...     metric="cosine"
        ... )
        >>> np.nanmean(prob_pos)
        0.6421

    Notes:
        - The function uses vectorized operations and computes similarities in batches
          grouped by rule for efficiency.
        - Distance-based metrics (e.g., Euclidean, Manhattan) are negated to convert
          distance into similarity.
        - Pearson similarity is computed manually via normalized covariance.
    """

    def compute_similarity(A, B, metric):
        if metric == "cosine":
            return cosine_similarity(A, B)
        elif metric == "dot":
            return np.dot(A, B.T)
        elif metric == "euclidean":
            return -euclidean_distances(A, B)  # To convert distance to similarity, it is taken negative
        elif metric == "manhattan":
            return -manhattan_distances(A, B)
        elif metric == "pearson":
            A_mean = A - A.mean(axis=1, keepdims=True)
            B_mean = B - B.mean(axis=1, keepdims=True)
            corr = np.dot(A_mean, B_mean.T) / (
                np.linalg.norm(A_mean, axis=1, keepdims=True) * np.linalg.norm(B_mean, axis=1)
            )
            return corr
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    valid_mask = df["text"].isin(text_to_emb.keys())
    embeddings = np.stack([text_to_emb[t] for t in df.loc[valid_mask, "text"]])
    rules = df.loc[valid_mask, "rule"].values

    prob_pos_list = np.full(len(df), np.nan, dtype=float)
    sim_pos_list = np.full(len(df), np.nan, dtype=float)
    sim_neg_list = np.full(len(df), np.nan, dtype=float)

    for rule in tqdm(np.unique(rules), desc=f"Computing with {metric} similarity"):
        idxs = np.where(rules == rule)[0]
        body_embs = embeddings[idxs]

        pos_texts = diverse_pos_df.loc[diverse_pos_df["rule"] == rule, "example"]
        neg_texts = diverse_neg_df.loc[diverse_neg_df["rule"] == rule, "example"]

        pos_embs = np.stack([text_to_emb[t] for t in pos_texts if t in text_to_emb]) if len(pos_texts) else None
        neg_embs = np.stack([text_to_emb[t] for t in neg_texts if t in text_to_emb]) if len(neg_texts) else None

        if pos_embs is None or neg_embs is None:
            continue

        sim_pos = compute_similarity(body_embs, pos_embs, metric).mean(axis=1)
        sim_neg = compute_similarity(body_embs, neg_embs, metric).mean(axis=1)
        prob_pos = sim_pos / (sim_pos + sim_neg + 1e-8)

        prob_pos_list[idxs] = prob_pos
        sim_pos_list[idxs] = sim_pos
        sim_neg_list[idxs] = sim_neg

    return prob_pos_list, sim_pos_list, sim_neg_list


# Cosine similarity (default)
prob_cos, sim_pos_cos, sim_neg_cos = compute_body_scores_vectorized(
    train_df, diverse_pos_df, diverse_neg_df, text_to_emb, metric="cosine"
)

train_df["prob_cosine"] = prob_cos
train_df["sim_pos_cos"] = sim_pos_cos
train_df["sim_neg_cos"] = sim_neg_cos

# Euclidean similarity (negated distance)
prob_euclidean, sim_pos_euc, sim_neg_euc = compute_body_scores_vectorized(
    train_df, diverse_pos_df, diverse_neg_df, text_to_emb, metric="euclidean"
)

train_df["prob_euclidean"] = prob_euclidean
train_df["sim_pos_euc"] = sim_pos_euc
train_df["sim_neg_euc"] = sim_neg_euc


# Dot product similarity
prob_dot, sim_pos_dot, sim_neg_dot = compute_body_scores_vectorized(
    train_df, diverse_pos_df, diverse_neg_df, text_to_emb, metric="dot"
)

train_df["prob_dot"] = prob_dot
train_df["sim_pos_dot"] = sim_pos_dot
train_df["sim_neg_dot"] = sim_neg_dot


all_test_texts = test_df["text"].tolist()
all_test_texts = list(dict.fromkeys(all_test_texts))  # clean duplicate

# Calculate embedding
test_embeddings = compute_embeddings(all_test_texts, batch_size=128)

# create dictionary
for text, emb in zip(all_test_texts, test_embeddings):
    text_to_emb[text] = emb  #update existing dict

prob_cos_t, sim_pos_cos_t, sim_neg_cos_t = compute_body_scores_vectorized(
    test_df, diverse_pos_df, diverse_neg_df, text_to_emb, metric="cosine"
)

test_df["prob_cosine"] = prob_cos_t
test_df["sim_pos_cos"] = sim_pos_cos_t
test_df["sim_neg_cos"] = sim_neg_cos_t


prob_euclidean_t, sim_pos_euc_t, sim_neg_euc_t = compute_body_scores_vectorized(
    test_df, diverse_pos_df, diverse_neg_df, text_to_emb, metric="euclidean"
)

test_df["prob_euclidean"] = prob_euclidean_t
test_df["sim_pos_euc"] = sim_pos_euc_t
test_df["sim_neg_euc"] = sim_neg_euc_t


prob_dot_t, sim_pos_dot_t, sim_pos_dot_t = compute_body_scores_vectorized(
    test_df, diverse_pos_df, diverse_neg_df, text_to_emb, metric="dot"
)

test_df["prob_dot"] = prob_dot_t
test_df["sim_pos_dot"] = sim_pos_dot_t
test_df["sim_neg_dot"] = sim_pos_dot_t


qwen_train = train_df[[
    "prob_cosine", 
    "sim_pos_cos", 
    "sim_neg_cos", 
    "prob_euclidean", 
    "sim_pos_euc", 
    "sim_neg_euc", 
    "prob_dot", 
    "sim_pos_dot", 
    "sim_neg_dot"
]]
qwen_test = test_df[[
    "prob_cosine",
    "sim_pos_cos",
    "sim_neg_cos",
    "prob_euclidean",
    "sim_pos_euc",
    "sim_neg_euc", 
    "prob_dot",
    "sim_pos_dot",
    "sim_neg_dot"
]]

X_train_qwen = csr_matrix(qwen_train)
X_test_qwen = csr_matrix(qwen_test)


tqdm.pandas()
analyzer = SentimentIntensityAnalyzer()

def get_vader_scores(text):
    if not isinstance(text, str):
        return pd.Series([0, 0, 0, 0])
    scores = analyzer.polarity_scores(text)
    return pd.Series([scores['neg'], scores['neu'], scores['pos'], scores['compound']])

train_df[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = \
    train_df['text'].progress_apply(get_vader_scores)

test_df[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = \
    test_df['text'].progress_apply(get_vader_scores)


train_sent_scores = train_df[[
    'vader_neg',
    'vader_neu',
    'vader_pos',
    'vader_compound'
]]

test_sent_scores = test_df[[
    'vader_neg',
    'vader_neu',
    'vader_pos',
    'vader_compound'
]]

X_train_sent = csr_matrix(train_sent_scores)
X_test_sent = csr_matrix(test_sent_scores)


vectorizer = TfidfVectorizer(analyzer="char_wb",
                             sublinear_tf=False,
                             norm="l2",
                             max_features=100000, 
                             ngram_range=(1,3),
                             min_df=2,
                             max_df=0.95)


x_train_tfidf = vectorizer.fit_transform(train_df["text"])
x_test_tfidf = vectorizer.transform(test_df["text"])

print(x_train_tfidf.shape)
print(x_test_tfidf.shape)


svd = TruncatedSVD(n_components=1500, random_state=42)
svd.fit(x_train_tfidf)

explained = np.cumsum(svd.explained_variance_ratio_)

plt.plot(range(1, len(explained)+1), explained)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.grid(True)
plt.show()


lsa_svd = TruncatedSVD(n_components=1400, n_iter=10, random_state=42)
X_train_lsa = lsa_svd.fit_transform(x_train_tfidf)
X_test_lsa = lsa_svd.transform(x_test_tfidf)
print(lsa_svd.explained_variance_ratio_.sum())


X_train_lsa = X_train_lsa.toarray() if hasattr(X_train_lsa, "toarray") else X_train_lsa
X_test_lsa = X_test_lsa.toarray() if hasattr(X_test_lsa, "toarray") else X_test_lsa

X_train_lsa = csr_matrix(X_train_lsa)
X_test_lsa = csr_matrix(X_test_lsa)

print(X_train_lsa.shape)
print(X_test_lsa.shape)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=True, handle_unknown="ignore")

train_ohe = ohe.fit_transform(train_df[["rule"]])
train_ohe = pd.get_dummies(train_df["rule"], drop_first = True)

test_ohe = ohe.fit_transform(test[["rule"]])
test_ohe = pd.get_dummies(test["rule"], drop_first = True)

train_ohe = train_ohe.replace({True: 1, False: 0})
test_ohe = test_ohe.replace({True: 1, False: 0})

train_ohe = csr_matrix(train_ohe)
test_ohe = csr_matrix(test_ohe)


emoji_pattern = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U00002600-\U000026FF"  # miscellaneous symbols
    "]+",
    flags=re.UNICODE
)

from better_profanity import profanity

profanity.load_censor_words_from_file("/kaggle/input/profanity-wordlist/profanity_wordlist.txt")

def fast_regex_feature(df):
    texts = df["text"].astype(str).fillna("")
    
    # Word count
    word_counts = texts.str.split().apply(len).astype(int)

    # Has link?
    has_link = texts.str.contains(r"(http[s]?://|www\.)", case=False, regex=True).astype(int)

    # Has emoji?
    has_emoji = texts.str.contains(emoji_pattern).astype(int)

    # Emoji count
    emoji_counts = texts.str.count(emoji_pattern)

    # Emoji ratio
    emoji_ratio = np.where(word_counts > 0, emoji_counts / word_counts, 0)
    
    # Badword
    badwords = {str(w).strip() for w in profanity.CENSOR_WORDSET if str(w).strip()}
    if badwords: 
        badword_pattern = r"\b(" + "|".join(map(re.escape, badwords)) + r")\b"
        badword_counts = texts.str.count(badword_pattern, flags=re.IGNORECASE)
    else:
        badword_counts = pd.Series(0, index=df.index)

    # Badword ratio
    badword_ratio = np.where(word_counts > 0, badword_counts / word_counts, 0)

    # Add to dataframe
    df["word_count"] = word_counts
    df["has_link"] = has_link
    df["has_emoji"] = has_emoji
    df["emoji_count"] = emoji_counts
    df["emoji_ratio"] = emoji_ratio
    df["badword_count"] = badword_counts
    df["badword_ratio"] = badword_ratio
    return df

train_df = fast_regex_feature(train_df)
test_df = fast_regex_feature(test_df)


train_regex = train_df[[
    "word_count", 
    "has_link",
    "has_emoji",
    "emoji_count",
    "emoji_ratio",
    "badword_count", 
    "badword_ratio"
]]
test_regex = test_df[[
    "word_count", 
   "has_link",
    "has_emoji",
    "emoji_count",
    "emoji_ratio",
    "badword_count", 
    "badword_ratio"
]]

X_train_regex = csr_matrix(train_regex)
X_test_regex = csr_matrix(test_regex)


num_cols = ["rule_violation",
            "prob_cosine",
            "sim_pos_cos",
            "sim_neg_cos",
            "prob_euclidean",
            "sim_pos_euc",
            "sim_neg_euc",
            "prob_dot",
            "sim_pos_dot",
            "sim_neg_dot",
            "vader_neg",
            "vader_neu",
            "vader_pos",
            "vader_compound",
            "word_count",
            "has_link",
            "has_emoji",
            "emoji_ratio",
            "badword_count", 
            "badword_ratio"
           ]


# Draw histogram for each numeric column
plt.figure(figsize=(12, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(len(num_cols)//3 + 1, 3, i)
    plt.hist(train_df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title(col)
    plt.xlabel('')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


corr = train_df[num_cols].corr()["rule_violation"].drop("rule_violation")

plt.figure(figsize=(6,4))
sns.barplot(x=corr.values, y=corr.index, palette="coolwarm")
plt.title("Correlation with rule_violation")
plt.show()


from sklearn.linear_model import LogisticRegression
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.metrics import roc_auc_score as cu_roc_auc_score
import cupy as cp
from sklearn.preprocessing import StandardScaler


X_log = hstack([
    train_ohe,
    X_train_lsa,
    X_train_sent,
    X_train_qwen
], 
               format="csr")

X_test_log = hstack([
    test_ohe,
    X_test_lsa,
    X_test_sent,
    X_test_qwen
], 
           format="csr")

y = train_df['rule_violation']


scaler = StandardScaler(with_mean=False)

X_train_scaled = scaler.fit_transform(X_log)
X_test_scaled = scaler.transform(X_test_log)


print("X_log shape: ", X_train_scaled.shape)
print("X_log_test shape: ", X_test_scaled.shape)
print("y shape: ", y.shape)


groups = train_df["rule"]
num_group = train_df["rule"].nunique()
sgkf = StratifiedGroupKFold(n_splits=num_group, shuffle=True, random_state=42)

test_preds_log = cp.zeros((X_test_scaled.shape[0], sgkf.get_n_splits()))
oof_logreg = cp.zeros(len(y))

aucs = []
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train_scaled, y, groups=groups)):
    print(f"Fold {fold+1}")
    X_train, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

     # cuML model
    model = cuLogisticRegression(
        max_iter=3500,
        C=0.0016564833561936504,
        tol=3.588351617543053e-05,
        alpha= 1.2511817495445187e-06,
        penalty="l1",
        solver="qn",  # cuML uses quasi-Newton solver (faster)
        fit_intercept=True
    )
    
    model.fit(X_train, y_train)   
    y_prob = cp.asarray(model.predict_proba(X_val)[:, 1])
    oof_logreg[val_idx] = y_prob
    auc = cu_roc_auc_score(cp.asarray(y_val), y_prob)
    aucs.append(float(auc))
    print(f"Fold {fold+1} AUC: {auc:.4f}")
    
    test_preds_log[:, fold] = cp.asarray(model.predict_proba(X_test_scaled)[:, 1])
   
print(f"\nMean AUC: {np.mean(aucs):.4f}")
print(f"OOF AUC (overall): {roc_auc_score(cp.asnumpy(y), cp.asnumpy(oof_logreg))}")

test_log_mean = cp.asnumpy(test_preds_log.mean(axis=1))
train_df["log_pred"] = cp.asnumpy(oof_logreg)
test_df["log_pred"] = test_log_mean


#import optuna
#import cupy as cp
#import numpy as np
#from cuml.linear_model import LogisticRegression as cuLogisticRegression
#from cuml.metrics import roc_auc_score as cu_roc_auc_score
#from sklearn.metrics import roc_auc_score

#def objective(trial):
    # ğŸ”§ Hiperparametre aralÄ±klarÄ±
#    solver = trial.suggest_categorical("solver", ["qn"])  # cuML ÅŸimdilik sadece 'qn' destekliyor
#    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
#    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
#    C = trial.suggest_float("C", 1e-3, 100, log=True)
#    tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True)
#    max_iter = trial.suggest_int("max_iter", 500, 4000, step=500)
#    alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)  # cuML Ã¶zel reg parametresi
    
    # cuML model parametreleri
#    model_params = {
#        "solver": solver,
#        "penalty": penalty,
#        "C": C,
#        "fit_intercept": fit_intercept,
#        "tol": tol,
#        "max_iter": max_iter,
#        "alpha": alpha,
#    }

#    aucs = []
 #   oof_preds = cp.zeros(len(y))
    
#    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train_scaled, y, groups=groups)):
#        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
#        y_tr, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

#        model = cuLogisticRegression(**model_params)
#        model.fit(X_tr, y_tr)

 #       y_prob = cp.asarray(model.predict_proba(X_val)[:, 1])
#        auc = cu_roc_auc_score(cp.asarray(y_val), y_prob)
#        aucs.append(float(auc))
#        oof_preds[val_idx] = y_prob

#    mean_auc = np.mean(aucs)
#    return mean_auc


# ğŸ�¯ Optuna Ã§alÄ±ÅŸtÄ±rma
#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=50, show_progress_bar=True)

#print("âœ… Best parameters:", study.best_params)
#print(f"ğŸ�� Best Mean AUC: {study.best_value:.4f}")


from catboost import CatBoostClassifier


X_cat = hstack([train_ohe,
            X_train_lsa,
            X_train_sent,
            X_train_regex,
            X_train_qwen
           ], 
           format="csr")

X_test_cat = hstack([test_ohe,
                 X_test_lsa,
                 X_test_sent,
                 X_test_regex,
                 X_test_qwen
                ], 
           format="csr")

y = train_df['rule_violation']


print("X shape: ", X_cat.shape)
print("X_test shape: ", X_test_cat.shape)
print("y shape: ", y.shape)


sgkf = StratifiedGroupKFold(n_splits=num_group, shuffle=True, random_state=42)

auc_scores = []
test_preds_cat = np.zeros((X_test_cat.shape[0], sgkf.get_n_splits()))
oof_catboost = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_cat, y, groups=groups)):
    print(f"Fold {fold+1}")

    X_train_cat, X_val_cat = X_cat[train_idx], X_cat[val_idx]
    y_train_cat, y_val_cat = y[train_idx], y[val_idx]
    
    cat = CatBoostClassifier(
        iterations = 900,
        learning_rate = 0.043767126303409544,
        depth = 6,
        l2_leaf_reg = 4.749239763680407,
        bagging_temperature = 1.0934205586865593,
        border_count = 73,
        random_strength = 4.439102767051396,
        boosting_type = "Plain",
        grow_policy = "Lossguide",
        od_type = "Iter",
        od_wait = 50,
        eval_metric = "AUC",
        random_seed = 42,
        task_type = "CPU",
        verbose = False
)
    
  
    cat.fit(X_train_cat, y_train_cat)
    y_prob_cat = cat.predict_proba(X_val_cat)[:, 1]
    oof_catboost[val_idx] = y_prob_cat
    auc = roc_auc_score(y_val_cat, y_prob_cat)
    auc_scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.4f}")
    test_preds_cat[:, fold] = cat.predict_proba(X_test_cat)[:, 1]


print(f"\nMean AUC: {np.mean(auc_scores):.4f}")
print(f"OOF AUC (overall): {roc_auc_score(y, oof_catboost)}")

cat_test_preds = np.mean(test_preds_cat, axis=1)
train_df["cat_pred"] = oof_catboost
test_df["cat_pred"] = cat_test_preds


#import optuna
#from catboost import CatBoostClassifier, Pool
#from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
#from sklearn.metrics import roc_auc_score
#import numpy as np
#import warnings
#warnings.filterwarnings("ignore")

#RANDOM_SEED = 42
#NFOLDS = num_group
#N_TRIALS = 60  # zamanÄ±na gÃ¶re arttÄ±rabilirsin

# --- CV ---
#try:
#    cv = StratifiedGroupKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_SEED)
#except Exception:
#    cv = GroupKFold(n_splits=NFOLDS)

# --- Objective function ---
#def objective(trial):
#    boosting_type = trial.suggest_categorical("boosting_type", ["Plain", "Ordered"])
#    if boosting_type == "Ordered":
#        grow_policy = "SymmetricTree"
#    else:
#        grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])

#    params = {
#        "iterations": trial.suggest_int("iterations", 300, 3000, step=300),
#        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
#        "depth": trial.suggest_int("depth", 4, 10),
#        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
#        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
#        "border_count": trial.suggest_int("border_count", 32, 255),
#        "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
#        "boosting_type": trial.suggest_categorical("boosting_type", ["Plain", "Ordered"]),
#        "grow_policy": grow_policy,
#        "od_type": "Iter",
#        "od_wait": 50,
#        "eval_metric": "AUC",
#        "random_seed": RANDOM_SEED,
#        "task_type": "GPU" if trial.suggest_categorical("use_gpu", [True, False]) else "CPU",
#        "verbose": False
#    }

#    aucs = []
#    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
#        X_train, X_val = X[train_idx], X[val_idx]
#        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#        train_pool = Pool(X_train, y_train)
#        val_pool = Pool(X_val, y_val)

#        model = CatBoostClassifier(**params)
#        model.fit(train_pool, eval_set=val_pool, use_best_model=True, early_stopping_rounds=100)
        
#        y_prob = model.predict_proba(X_val)[:, 1]
#        auc = roc_auc_score(y_val, y_prob)
#        aucs.append(auc)

#    return np.mean(aucs)

# --- Run Optuna ---
#from optuna.samplers import TPESampler
#sampler = TPESampler(seed=RANDOM_SEED)
#study = optuna.create_study(direction="maximize", sampler=sampler)
#study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

# --- Best trial ---
#print("âœ… En iyi deneme:")
#print(f"Mean AUC: {study.best_value:.5f}")
#print("Best params:")
#for k, v in study.best_params.items():
#    print(f"  {k}: {v}")


#Get dimensions
n_lsa = X_train_lsa.shape[1]
n_ohe = train_ohe.shape[1]
n_sent = X_train_sent.shape[1]
n_regex = X_train_regex.shape[1]
n_qwen = qwen_train.shape[1]

# Feature names
feature_names = (
    [f"LSA_{i}" for i in range(n_lsa)] +
    [f"OHE_{i}" for i in range(n_ohe)] +
    [f"SENT_{i}" for i in range(n_sent)] +
    [f"REG_{i}" for i in range(n_regex)] +
    [f"QWEN_{i}" for i in range(n_qwen)] 
)

# Feature importance values
importances = cat.get_feature_importance()
feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feat_imp.head(50), x="importance", y="feature", palette="viridis")
plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.show()


print("OOF AUC - LogReg:", roc_auc_score(cp.asnumpy(y), cp.asnumpy(oof_logreg)))
print("OOF AUC - CatBoost:", roc_auc_score(y, oof_catboost))


oof_df = pd.DataFrame({
    "logreg": cp.asnumpy(oof_logreg),
    "catboost": oof_catboost
})
print(oof_df.corr())


blended = (0.3 * cp.asnumpy(oof_logreg)) + (0.7 * oof_catboost)
auc = roc_auc_score(y, blended)
auc


# --- Test prediction ---
final_test_pred = (0.7 * cat_test_preds) + (0.3 * test_log_mean)

# --- Create Submission File---
submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "rule_violation": final_test_pred
})

submission.to_csv("submission.csv", index=False)
print(f"âœ… Ensemble submission saved: submission.csv ({len(submission)} total row)")

