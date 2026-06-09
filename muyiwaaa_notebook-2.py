!pip install iterative-stratification


import os
import re
import html
import gc
import zipfile
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# ------------------------
# Safe, memory-oriented config
# ------------------------
RANDOM_STATE = 42
N_SPLITS = 5
WORD_MAX_FEATURES = 5000    # safe for Kaggle/Colab
CHAR_MAX_FEATURES = 2000
C_VALUES = [0.5, 1.0, 2.0]
MAX_ITER = 50
THRESHOLDS = np.linspace(0.3, 0.7, 9)  # sweep range
BATCH_SIZE_TEST = 4096  # test transform batch size

# ------------------------
# Utilities
# ------------------------
def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'<code>.*?</code>', ' ', text, flags=re.DOTALL)  # drop very long code blocks
    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def iter_batches(indices, batch_size):
    for i in range(0, len(indices), batch_size):
        yield indices[i:i+batch_size]


# ------------------------
# Load and preprocess raw data
# ------------------------
with zipfile.ZipFile('/kaggle/input/facebook-recruiting-iii-keyword-extraction/Train.zip', 'r') as z:
    z.extractall('.')
train_df = pd.read_csv("Train.csv")
train_df.dropna(subset=["Title", "Body", "Tags"])
train_df.reset_index(drop=True, inplace = True)


# get duplicates
train_df_dups = train_df[train_df.duplicated(['Title', 'Body', 'Tags'])]
print('Total Duplicates: ', len(train_df_dups))
print('ratio: ', len(train_df_dups)/len(train_df))


# remove duplicates
train_df = train_df.drop_duplicates(['Title', 'Body', 'Tags'])
print('After removing dups: ', len(train_df))
print('ratio: ', len(train_df)/6034194)


train_df.info()


train_df.reset_index(drop = True, inplace=True)


# combine and clean
train_df['CleanText'] = (train_df['Title'].fillna('') + ' ' + train_df['Body'].fillna('')).map(clean_text)
train_df['TagList'] = train_df['Tags'].str.split()
n_samples = len(train_df)


train_df.isnull().sum()


train_df.dropna(inplace = True)
train_df.reset_index(drop=True, inplace=True)


# label binarizer
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(train_df['TagList'])  # keep in memory; typically smaller than feature matrices
n_labels = Y.shape[1]

# free raw text columns if you want later; keep CleanText for transforms
del train_df['Title'], train_df['Body'], train_df['Tags']

gc.collect()

# ------------------------
# Fit TF-IDF vectorizers once on the corpus
# Only fit, do not transform entire corpus to sparse matrix at once.
# ------------------------
word_vec = TfidfVectorizer(max_features=WORD_MAX_FEATURES, ngram_range=(1,3), dtype=np.float32)
char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,6), max_features=CHAR_MAX_FEATURES, dtype=np.float32)

# fit using generator to avoid extra lists (vectorizer accepts any iterable)
word_vec.fit(train_df['CleanText'])
char_vec.fit(train_df['CleanText'])
gc.collect()

# ------------------------
# Prepare memmap file for out-of-fold probabilities
# store as float32 on disk to avoid large in-RAM array
# ------------------------
tmpdir = tempfile.mkdtemp()
oof_path = os.path.join(tmpdir, 'oof_probs.dat')
oof_shape = (n_samples, n_labels)
oof_probs = np.memmap(oof_path, dtype='float32', mode='w+', shape=oof_shape)


# ------------------------
# Cross validation: compute OOF probabilities for each C sequentially,
# but keep only the best C and threshold. For memory efficiency we compute
# fold transforms and fit per fold.
# ------------------------
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

best_score = -1.0
best_C = None
best_thr = None

# To avoid keeping all fold predictions for each C in memory, create a temporary memmap per C
for C in C_VALUES:
    print(f"Evaluating C={C}")
    # temp memmap for this C
    c_oof_path = os.path.join(tmpdir, f'oof_C{str(C).replace(".","p")}.dat')
    c_oof = np.memmap(c_oof_path, dtype='float32', mode='w+', shape=oof_shape)

    # iterate folds sequentially
    for fold, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(n_samples), Y)):
        print(" fold", fold)
        # transform train split in memory-efficient way
        X_train_word = word_vec.transform(train_df['CleanText'].iloc[train_idx])
        X_train_char = char_vec.transform(train_df['CleanText'].iloc[train_idx])
        X_train = hstack([X_train_word, X_train_char]).tocsr()
        del X_train_word, X_train_char
        gc.collect()

        X_val_word = word_vec.transform(train_df['CleanText'].iloc[val_idx])
        X_val_char = char_vec.transform(train_df['CleanText'].iloc[val_idx])
        X_val = hstack([X_val_word, X_val_char]).tocsr()
        del X_val_word, X_val_char
        gc.collect()

        y_train = Y[train_idx]
        # fit One-vs-Rest logistic regression for this fold
        clf = OneVsRestClassifier(LogisticRegression(
            C=C, solver='saga', penalty='l2', max_iter=MAX_ITER, n_jobs=1
        ), n_jobs=1)
        clf.fit(X_train, y_train)

        # predict probabilities for validation fold in batches if needed
        # predict_proba on the sparse X_val is ok; but keep memory low
        probs_val = clf.predict_proba(X_val)  # shape (n_val, n_labels)
        # write into memmap
        c_oof[val_idx, :] = probs_val.astype('float32')

        # free fold memory
        del X_train, X_val, y_train, probs_val, clf
        gc.collect()

    # after all folds for this C are done, sweep thresholds using c_oof
    # load into working memory fold-by-fold only when evaluating thresholds to save peak memory.
    # For simplicity we will read the whole memmap row by row in batches to compute scores.
    # Build predictions for each threshold incrementally to compute f1.
    best_local_score = -1.0
    best_local_thr = None

    for thr in THRESHOLDS:
        # compute f1 score in streaming way to avoid building full binary matrix
        # accumulate true positives, predicted positives, etc is complex for multilabel F1.
        # here we will accumulate predictions in batches and compute f1 using sklearn on each batch,
        # but we will accumulate row-wise predictions to a temp memmap to allow f1_score on full dataset.
        pred_path = os.path.join(tmpdir, f'pred_C{str(C).replace(".","p")}_t{str(thr).replace(".","p")}.dat')
        preds_mem = np.memmap(pred_path, dtype='uint8', mode='w+', shape=oof_shape)
        # fill preds_mem in batches
        for i in range(0, n_samples, BATCH_SIZE_TEST):
            block = c_oof[i:i+BATCH_SIZE_TEST]
            block_bin = (block >= thr).astype('uint8')
            preds_mem[i:i+BATCH_SIZE_TEST, :] = block_bin
        # compute f1 on full dataset using memmap slices to reduce peak RAM
        # read back in chunks and accumulate for sklearn
        # sklearn expects full arrays; to avoid huge memory we will load preds_mem fully as uint8 (smaller)
        preds_full = np.asarray(preds_mem)  # should be smaller; still watch memory
        score = f1_score(Y, preds_full, average='samples', zero_division=0)
        # clean up
        del preds_mem, preds_full
        os.remove(pred_path)
        gc.collect()
        if score > best_local_score:
            best_local_score = score
            best_local_thr = thr

    print(f"  best thr for C={C}: {best_local_thr} score={best_local_score:.6f}")

    # update global best if improved
    if best_local_score > best_score:
        best_score = best_local_score
        best_C = C
        best_thr = best_local_thr

    # copy c_oof into main oof memmap only if best so far to avoid extra writes
    # we will store final OOF probabilities for best_C later by recomputing if needed.
    # remove c_oof to free disk memory
    del c_oof
    os.remove(c_oof_path)
    gc.collect()


print("Best C, threshold:", best_C, best_thr, "F1:", best_score)

# ------------------------
# Final: retrain on full training set with best_C and produce test predictions
# ------------------------
# Fit final classifier on full training data transformed in batches if necessary
# Transform full training data once if it fits; otherwise train using liblinear is not incremental.
# Here we transform full training data once; for very tight RAM you can use partial training strategies.
X_word_full = word_vec.transform(train_df['CleanText'])
X_char_full = char_vec.transform(train_df['CleanText'])
X_full = hstack([X_word_full, X_char_full]).tocsr()
del X_word_full, X_char_full
gc.collect()

final_clf = OneVsRestClassifier(LogisticRegression(
    C=best_C, solver='saga', penalty='l2', max_iter=MAX_ITER, n_jobs=1
), n_jobs=1)
final_clf.fit(X_full, Y)
gc.collect()

# ------------------------
# Test processing: transform in batches, predict probabilities in batches, apply threshold
# ------------------------
with zipfile.ZipFile('/kaggle/input/facebook-recruiting-iii-keyword-extraction/Test.zip', 'r') as z:
    z.extractall('.')
test_df = pd.read_csv("Test.csv").reset_index(drop=True)
test_df['CleanText'] = (test_df['Title'].fillna('') + ' ' + test_df['Body'].fillna('')).map(clean_text)

n_test = len(test_df)
preds_rows = []

for batch_idx in range(0, n_test, BATCH_SIZE_TEST):
    ids = list(range(batch_idx, min(batch_idx + BATCH_SIZE_TEST, n_test)))
    Xw = word_vec.transform(test_df['CleanText'].iloc[ids])
    Xc = char_vec.transform(test_df['CleanText'].iloc[ids])
    Xb = hstack([Xw, Xc]).tocsr()
    probs = final_clf.predict_proba(Xb).astype('float32')
    bins = (probs >= best_thr).astype('int8')
    # convert each row to tag list
    for row in bins:
        labels = [mlb.classes_[i] for i, v in enumerate(row) if v]
        preds_rows.append(" ".join(labels))
    del Xw, Xc, Xb, probs, bins
    gc.collect()

test_df['Tags'] = preds_rows
submission = test_df[['Id', 'Tags']]
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv")

# ------------------------
# Clean up
# ------------------------
del X_full, final_clf, train_df, test_df, oof_probs
gc.collect()

