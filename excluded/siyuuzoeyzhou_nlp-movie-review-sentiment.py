# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from pathlib import Path
OUT_DIR = Path("/kaggle/working")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Adjust the paths in following rows
train = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/train.csv')
test = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/test.csv')
sample_submission = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the sturcture of train dataset and test dataset
display(train.head(), test.head())

# Distribution of sentiment labels
sns.countplot(x='sentiment', data=train)
plt.title('Sentiment Label Distribution')
plt.show()


train.duplicated().sum()


train.isnull().any().any()


train['_len'] = train["review"].astype(str).str.split().apply(len)
print("\nTrain text length (words):")
print(train['_len'].describe())

plt.figure()
train['_len'].hist(bins=50)
plt.title("Train Text Length (word count)")
plt.xlabel("words")
plt.ylabel("frequency")
plt.show()


test['_len'] = test["review"].astype(str).str.split().apply(len)
print("\nTest text length (words):")
print(test['_len'].describe())

plt.figure()
test['_len'].hist(bins=50)
plt.title("Test Text Length (word count)")
plt.xlabel("words")
plt.ylabel("frequency")
plt.show()


# Top tokens(without filter on stop words)
from collections import Counter
def top_tokens(series, k=20):
    cnt = Counter()
    for t in series.astype(str):
        cnt.update(t.lower().split())
    return cnt.most_common(k)


print("\nTop 20 tokens in train:")
print(top_tokens(train["review"], k=20))

print("\nTop 20 tokens in test:")
print(top_tokens(test["review"], k=20))


# NLTK libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt_tab')
import string
import torch

punc = string.punctuation
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # stop_words

nltk.download('wordnet') # lemmatizing


# convert all the headlines to lowercase
train['clean_review'] = train['review'].apply(lambda x: x.lower())
test['clean_review'] = test['review'].apply(lambda x: x.lower())


# tokenization: Split the text into individual words or tokens
train['clean_review'] = train['clean_review'].apply(word_tokenize)
test['clean_review'] = test['clean_review'].apply(word_tokenize)


# stop word removal
train['clean_review'] = train['clean_review'].apply(lambda x: [item for item in x if item not in stop_words])
test['clean_review'] = test['clean_review'].apply(lambda x: [item for item in x if item not in stop_words])


# lemmatizing
wnl = nltk.WordNetLemmatizer()
train['clean_review'] = train['clean_review'].apply(lambda x: [wnl.lemmatize(item) for item in x])

test['clean_review'] = test['clean_review'].apply(lambda x: [wnl.lemmatize(item) for item in x])


# reconstruction
train['clean_review'] = train['clean_review'].apply(lambda x: ' '.join(x))
test['clean_review'] = test['clean_review'].apply(lambda x: ' '.join(x))


import re
def remove_punct(s: str) -> str:
    # Only keep: letters, numbers, spaces, apostrophes
    # the rest (punctuation/symbols) are replaced with spaces.
    s = re.sub(r"[^A-Za-z0-9\s']", " ", s)
    # delete extra spaces
    return re.sub(r"\s{2,}", " ", s).strip()

train['clean_review'] = train['clean_review'].apply(lambda x: remove_punct(x))
test['clean_review'] = test['clean_review'].apply(lambda x: remove_punct(x))
display(train.head(), test.head())



print("\nTop 20 tokens in train (after cleaning):")
print(top_tokens(train["clean_review"], k=20))

print("\nTop 20 tokens in test (after cleaning):")
print(top_tokens(test["clean_review"], k=20))


train_reviews = train['clean_review']
train_sentiment = train['sentiment']
test_reviews = test['clean_review']


# # construct models use features from TFIDF

# from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
# tfidf.fit(X_train)

# X_train_tfidf = tfidf.transform(X_train)
# X_valid_tfidf = tfidf.transform(X_valid)
# X_test_tfidf = tfidf.transform(test['clean_review'])


# 3 vectorizer: BoW / TF-IDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# BoW (CounterVector)
bow_vec = CountVectorizer(ngram_range=(1,2), stop_words = "english")

# TF-IDF 
tfidf_vec = TfidfVectorizer(ngram_range=(1,2), stop_words = "english")

# word2vec 
# train a vectorizer for a word, and do the vectorizer average for each sentence
def train_word2vec(corpus, embed_dim=300, window=5, min_count=2):
    # corpus: list[list[str]]
    model = Word2Vec(
        sentences=corpus, vector_size=embed_dim, window=window,
        min_count=min_count, workers=4, sg=1, seed=42
    )
    return model

def sent2vec_avg(tokens, w2v):
    vecs = []
    for t in tokens:
        if t in w2v.wv:
            vecs.append(w2v.wv[t])
    if len(vecs)==0:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)

train_tokens = [s.split() for s in train_reviews.tolist()]
test_tokens  = [s.split() for s in test_reviews.tolist()]

w2v_model = train_word2vec(train_tokens + test_tokens, embed_dim=300, window=5, min_count=2)


# Convert text into three feature vectors (training & testing)
X_bow_train = bow_vec.fit_transform(train_reviews)
X_bow_test = bow_vec.transform(test_reviews)

X_tfidf_train = tfidf_vec.fit_transform(train_reviews)
X_tfidf_test = tfidf_vec.transform(test_reviews)

X_w2v_train = np.vstack([sent2vec_avg(toks, w2v_model) for toks in train_tokens])
X_w2v_test = np.vstack([sent2vec_avg(toks, w2v_model) for toks in test_tokens])


def to_f32(X):
    return X.astype(np.float32) if hasattr(X, "astype") else X
    
X_dict = {
    "bow":    (to_f32(X_bow_train), to_f32(X_bow_test)),
    "tfidf":  (to_f32(X_tfidf_train), to_f32(X_tfidf_test)),
    "w2v":    (to_f32(X_w2v_train), to_f32(X_w2v_test)),
}


# PCA + t-SNE visualization
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from scipy.sparse import issparse

y_train = train_sentiment


def pca_tsne(X, y, title_prefix, random_state=42):
    """
    X: train_features; y is labels (np.array)
    PCA: if sparse, TruncatedSVD is replaced
    t-SNE: if sparse, TruncatedSVD is replaced
    """
    #  prepare PCA 2D coordinates
    if issparse(X):
        Z_pca2 = TruncatedSVD(n_components=2, random_state=random_state).fit_transform(X)
        X_50   = TruncatedSVD(n_components=50, random_state=random_state).fit_transform(X)
    else:
        Z_pca2 = PCA(n_components=2, random_state=random_state).fit_transform(X)
        if X.shape[1] > 50:
            X_50 = PCA(n_components=50, random_state=random_state).fit_transform(X)
        else:
            X_50 = X

    # t-SNE
    Z_tsne2 = TSNE(n_components=2, random_state=random_state,
                   perplexity=30, init="pca", learning_rate="auto").fit_transform(X_50)

    # left: PCA; Right: t-SNE
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, Z, ttl in [
        (axes[0], Z_pca2,  f"{title_prefix} — PCA(2D)"),
        (axes[1], Z_tsne2, f"{title_prefix} — t-SNE(2D)")
    ]:
        for lab in np.unique(y):
            idx = (y == lab)
            ax.scatter(Z[idx, 0], Z[idx, 1], s=8, alpha=0.7, label=str(lab))
        ax.set_title(ttl)
        ax.set_xlabel("Dim 1" if "t-SNE" in ttl else "PC1")
        ax.set_ylabel("Dim 2" if "t-SNE" in ttl else "PC2")
        ax.grid(False)
        ax.legend(markerscale=2, fontsize=8, frameon=True)
    plt.tight_layout()
    plt.show()

#  visualize for 3 vectorizer from X_dict
for name, (Xtr, _) in X_dict.items():
    pca_tsne(Xtr, y_train, title_prefix=name.upper(), random_state=42)



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

# get all traditional model
def get_traditional_estimators():
    est = {}

    # 1) Logistic Regression
    est["logreg"] = LogisticRegression(C=4.0, max_iter=2000)

    # 2) Decision Tree
    est["dtree"] = DecisionTreeClassifier(random_state=42)

    # 3) Random Forest
    est["rf"] = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42)

    # 4) LinearSVC
    est["linearsvc"] = LinearSVC(C=1.0, random_state=42)

    # 5) KNN
    est["knn"] = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)

    # 6) Gradient Boosting
    est["gbdt"] = GradientBoostingClassifier(random_state=42)

    # 7) AdaBoost
    est["ada"] = AdaBoostClassifier(n_estimators=400, learning_rate=0.5, random_state=42)

    # 8) XGBoost（若本机未安装则不会加入）
    est["xgb"] = XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="binary:logistic", tree_method="hist",
        n_jobs=-1, eval_metric="logloss", random_state=42
    )

    # 9) LightGBM（若未安装则不会加入）
    est["lgbm"] = LGBMClassifier(
        # 基本容量
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,          # 增加叶子，提升可分性
        max_depth=-1,           # 不限深度
    
        # 关键：别在预筛选阶段就把列都干掉
        feature_pre_filter=False,
    
        # 让特征有更多分桶，而不是每列只有 2 个 bin
        max_bin=1023,
    
        # 适当放松叶子约束，避免过早停止
        min_data_in_leaf=5,
        min_sum_hessian_in_leaf=1e-3,
    
        # 采样与正则（可按需再调）
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.0,
    
        # 稀疏矩阵建议 row-wise；兼容 CPU
        force_row_wise=True,
    
        # 其余
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    # 10) DNN（sklearn 的 MLP）
    from sklearn.neural_network import MLPClassifier
    est["mlp"] = MLPClassifier(hidden_layer_sizes=(256,128), activation="relu",
                               solver="adam", learning_rate_init=1e-3,
                               batch_size=128, max_iter=20, random_state=42)
    return est

estimators = get_traditional_estimators()




import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.rcParams["axes.grid"] = False  # 关闭网格，图更干净

def train_and_save(classifier, X_all, y_all, X_test, model_name, vec_name):
    """
    极简版：
    1) 切分 train/valid
    2) 训练→验证：打印 Accuracy、classification_report，并画混淆矩阵
    3) 用全量训练集重训→在测试集上预测并保存为  {model_name}_{vec_name}.csv
    """
    # 1) 切分
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    # 2) 训练与验证
    classifier.fit(X_tr, y_tr)
    pred = classifier.predict(X_va)

    print(f"[{model_name} × {vec_name}]  Valid Accuracy: {accuracy_score(y_va, pred):.4f}")
    print(classification_report(y_va, pred, digits=4))

    cm = confusion_matrix(y_va, pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix — {model_name} × {vec_name}')
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout(); plt.show()

    # 3) 全量训练 + 测试预测 + 保存 CSV（无前缀）
    classifier.fit(X_all, y_all)
    test_pred = classifier.predict(X_test)
    out_csv = OUT_DIR / f"{model_name}_{vec_name}.csv"
    pd.DataFrame({"id": test["id"], "Sentiment": test_pred.astype(int)}).to_csv(out_csv, index=False)
    print("Saved:", out_csv)



# 逐向量器 × 逐模型：直接训练、验证、保存测试预测（文件名：{model}_{vec}.csv）
from sklearn.base import clone

for vec_name, (Xtr, Xte) in X_dict.items():
    for mdl_name, mdl in estimators.items():
        clf = clone(mdl)  # ← 每次都克隆一个全新模型实例
        train_and_save(
            classifier=clf,
            X_all=Xtr, y_all=y_train,
            X_test=Xte,
            model_name=mdl_name, vec_name=vec_name
        )




! pip install transformers datasets accelerate


# 纯 PyTorch 版本：不依赖 `datasets` 和 pyarrow；显式 AdamW + 线性 warmup
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt, seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from pathlib import Path

plt.rcParams["axes.grid"] = False

class TextClsDataset(Dataset):
    """只存文本与标签（可为 None）；真正的tokenize在collate_fn里动态完成，便于动态padding。"""
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        if self.labels is None:
            return self.texts[i]
        return self.texts[i], self.labels[i]

def make_collate_fn(tokenizer, max_len, has_label: bool):
    def collate(batch):
        if has_label:
            texts, labels = zip(*batch)
        else:
            texts, = (batch,)
        enc = tokenizer(list(texts), truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")
        if has_label:
            enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate

def train_and_save_hf(
    model_name: str,
    short_name: str,
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    text_col: str,
    label_col: str,
    id_col:    str,
    out_dir,
    *,
    val_ratio: float = 0.2,
    max_len:   int   = 160,
    batch:     int   = 16,
    lr:        float = 2e-5,
    epochs:    int   = 2,
    seed:      int   = 42,
    weight_decay: float = 0.01,
    grad_accum: int = 1,
):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)

    # 1) 切分
    tr_df, va_df = train_test_split(
        train_df[[text_col, label_col]],
        test_size=val_ratio, stratify=train_df[label_col], random_state=seed
    )

    # 2) 分词器（GPT-2 处理 pad_token）
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if "gpt2" in model_name and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) 数据集与 DataLoader（动态 padding）
    ds_tr = TextClsDataset(tr_df[text_col].tolist(), tr_df[label_col].to_numpy())
    ds_va = TextClsDataset(va_df[text_col].tolist(), va_df[label_col].to_numpy())
    ds_te = TextClsDataset(test_df[text_col].tolist(), labels=None)

    collate_tr = make_collate_fn(tokenizer, max_len, has_label=True)
    collate_ev = make_collate_fn(tokenizer, max_len, has_label=True)
    collate_te = make_collate_fn(tokenizer, max_len, has_label=False)

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True,  collate_fn=collate_tr)
    dl_va = DataLoader(ds_va, batch_size=max(1, batch*2), shuffle=False, collate_fn=collate_ev)
    dl_te = DataLoader(ds_te, batch_size=max(1, batch*2), shuffle=False, collate_fn=collate_te)

    # 4) 模型 + AdamW + 线性 warmup 调度
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if "gpt2" in model_name:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # ← AdamW
    total_steps = max(1, (len(dl_tr) // grad_accum) * epochs)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 5) 训练循环（支持梯度累积）
    model.train()
    global_step = 0
    for ep in range(1, epochs+1):
        optimizer.zero_grad()
        running = 0.0
        for step, batch_tr in enumerate(dl_tr, start=1):
            batch_tr = {k: v.to(device) for k, v in batch_tr.items()}
            out = model(**batch_tr)
            loss = out.loss / grad_accum
            loss.backward()
            running += loss.item()
            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                optimizer.zero_grad()
                global_step += 1
        # 每个 epoch 做一次验证
        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for batch_va in dl_va:
                batch_va = {k: v.to(device) for k, v in batch_va.items()}
                out = model(**batch_va)
                preds.append(out.logits.argmax(dim=-1).cpu().numpy())
                golds.append(batch_va["labels"].cpu().numpy())
        preds = np.concatenate(preds); golds = np.concatenate(golds)
        acc = accuracy_score(golds, preds)
        print(f"[{short_name}] epoch {ep:02d}  valid acc={acc:.4f}")
        model.train()

    # 6) 最终验证报告 + 混淆矩阵
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch_va in dl_va:
            batch_va = {k: v.to(device) for k, v in batch_va.items()}
            out = model(**batch_va)
            preds.append(out.logits.argmax(dim=-1).cpu().numpy())
            golds.append(batch_va["labels"].cpu().numpy())
    preds = np.concatenate(preds); golds = np.concatenate(golds)

    print(classification_report(golds, preds, digits=4))
    cm = confusion_matrix(golds, preds)
    plt.figure(figsize=(6,4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {short_name}")
    plt.xlabel("Pred"); plt.ylabel("True"); plt.tight_layout(); plt.show()

    # 7) 测试集预测 + 保存（文件名：short_name.csv）
    test_preds = []
    with torch.no_grad():
        for batch_te in dl_te:
            batch_te = {k: v.to(device) for k, v in batch_te.items()}
            out = model(**batch_te)
            test_preds.append(out.logits.argmax(dim=-1).cpu().numpy())
    test_preds = np.concatenate(test_preds)

    out_csv = Path(out_dir) / f"{short_name}.csv"
    pd.DataFrame({id_col: test_df[id_col], "Sentiment": test_preds.astype(int)}).to_csv(out_csv, index=False)
    print("Saved:", out_csv)



# from transformers import BertTokenizer, BertModel, BartTokenizer, BartModel, BartForConditionalGeneration, BartConfig, BartForSequenceClassification
# from transformers import GPT2TokenizerFast,GPT2Model, AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt


modern_setups = [
    ("bert-base-uncased",         "bert",       160, 16, 2, 2e-5),
    ("roberta-base",              "roberta",    160, 16, 2, 2e-5),
    ("facebook/bart-base",        "bart",       160,  8, 2, 2e-5),
    ("gpt2",                      "gpt2",       128,  8, 2, 2e-5),
    ("microsoft/deberta-v3-base", "deberta_v3", 160, 16, 2, 2e-5),
]

for mdl, short, max_len, batch, epochs, lr in modern_setups:
    train_and_save_hf(
        model_name = mdl,
        short_name = short,
        train_df   = train,
        test_df    = test,
        text_col   = "clean_review",
        label_col  = "sentiment",
        id_col     = "id",
        out_dir    = OUT_DIR,
        val_ratio  = 0.2,
        max_len    = max_len,
        batch      = batch,
        lr         = lr,
        epochs     = epochs,
        seed       = 42,
        weight_decay = 0.01,     # ← AdamW 的 WD
        grad_accum   = 1         # 无GPU就调大，如2/4
    )


