# Bag of Words Meets Bags of Popcorn

!pip install py7zr >/dev/null

import os, re, random
import numpy as np
import pandas as pd
import py7zr, zipfile
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack

RANDOM_SEED = 1993
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

#  7-Zip extraction

def extract_7zip(zip_path, out_dir="."):
    ext = zip_path.lower()
    if ext.endswith(".7z"):
        with py7zr.SevenZipFile(zip_path, mode="r") as z:
            z.extractall(path=out_dir)
    elif ext.endswith(".zip"):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
    else:
        raise ValueError("Unsupported archive format: " + zip_path)

INPUT_DIR = "/kaggle/input/word2vec-nlp-tutorial"
train_zip = os.path.join(INPUT_DIR, "labeledTrainData.tsv.zip")
test_zip  = os.path.join(INPUT_DIR, "testData.tsv.zip")

tmp_dir = "./unzipped"
os.makedirs(tmp_dir, exist_ok=True)

print("Extracting archives...")
extract_7zip(train_zip, tmp_dir)
extract_7zip(test_zip, tmp_dir)

train_tsv = [f for f in os.listdir(tmp_dir) if "labeledTrainData" in f][0]
test_tsv  = [f for f in os.listdir(tmp_dir) if "testData" in f][0]

train_df = pd.read_csv(os.path.join(tmp_dir, train_tsv), sep="\t")
test_df  = pd.read_csv(os.path.join(tmp_dir, test_tsv), sep="\t")

print("Train/Test shapes:", train_df.shape, test_df.shape)

#  Text cleaning

RE_NON_LETTERS = re.compile(r"[^a-zA-Z]")

def clean_review(text: str) -> str:
    text = BeautifulSoup(text, "lxml").get_text()
    text = RE_NON_LETTERS.sub(" ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning reviews...")
train_text = train_df["review"].apply(clean_review).tolist()
test_text  = test_df["review"].apply(clean_review).tolist()
y = train_df["sentiment"].values

all_text = train_text + test_text


# Word + Character TF-IDF features

print("Building TF-IDF features (word + char)...")

word_vec = TfidfVectorizer(
    min_df=3,
    max_df=0.9,
    max_features=40000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
)

char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_features=40000,
    sublinear_tf=True,
)

X_word_all = word_vec.fit_transform(all_text)
X_char_all = char_vec.fit_transform(all_text)

X_all = hstack([X_word_all, X_char_all]).tocsr()

X = X_all[: len(train_text)]
X_test = X_all[len(train_text) :]

print("X shape:", X.shape, "X_test shape:", X_test.shape)

#  NB-SVM log-count ratio

def nbsvm_ratio(X, y, alpha=1.0):
    y = np.asarray(y)
    pos = X[y == 1].sum(axis=0) + alpha
    neg = X[y == 0].sum(axis=0) + alpha
    pos = pos / pos.sum()
    neg = neg / neg.sum()
    r = np.log(pos / neg)
    return np.asarray(r).ravel()  # 1D


#  7-fold CV 

N_FOLDS = 7
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

oof = np.zeros(len(y))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n========== FOLD {fold}/{N_FOLDS} ==========")
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # ----- Model 1: plain LR -----
    lr1 = LogisticRegression(
        C=4.0,
        max_iter=800,   # "more epochs"
        solver="liblinear",
        n_jobs=-1,
    )
    lr1.fit(X_tr, y_tr)
    p1 = lr1.predict_proba(X_va)[:, 1]

    # ----- Model 2: NB-SVM style LR -----
    r = nbsvm_ratio(X_tr, y_tr, alpha=1.0)
    X_tr_nb = X_tr.multiply(r)
    X_va_nb = X_va.multiply(r)

    lr2 = LogisticRegression(
        C=4.0,
        max_iter=800,   # "more epochs"
        solver="liblinear",
        n_jobs=-1,
    )
    lr2.fit(X_tr_nb, y_tr)
    p2 = lr2.predict_proba(X_va_nb)[:, 1]

    # ----- Blend -----
    p = 0.5 * p1 + 0.5 * p2
    oof[va_idx] = p

    fold_auc = roc_auc_score(y_va, p)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} AUC: {fold_auc:.5f}")

cv_auc = roc_auc_score(y, oof)
print("\n==============================")
print("OOF CV AUC:", cv_auc)
print("Fold AUCs:", [round(s, 5) for s in fold_scores])
print("==============================")

# Train final models on all data

print("\nTraining final models on FULL data...")
final_lr1 = LogisticRegression(
    C=4.0,
    max_iter=1000,   # even more iterations for final model
    solver="liblinear",
    n_jobs=-1,
)
final_lr1.fit(X, y)

r_full = nbsvm_ratio(X, y, alpha=1.0)
X_nb_full = X.multiply(r_full)
X_test_nb = X_test.multiply(r_full)

final_lr2 = LogisticRegression(
    C=4.0,
    max_iter=1000,
    solver="liblinear",
    n_jobs=-1,
)
final_lr2.fit(X_nb_full, y)


# Predict test & save submission

print("Predicting test set...")
test_p1 = final_lr1.predict_proba(X_test)[:, 1]
test_p2 = final_lr2.predict_proba(X_test_nb)[:, 1]
test_pred = 0.5 * test_p1 + 0.5 * test_p2

submission = pd.DataFrame({
    "id": test_df["id"],
    "sentiment": test_pred,   # keep as probabilities for ROC AUC
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
submission.head()


